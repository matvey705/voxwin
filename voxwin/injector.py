"""Text injection into the focused window of any application.

Two strategies:
  * "clipboard" (default, fast): snapshot clipboard -> set text -> send
    Ctrl+V via SendInput -> restore clipboard. Works in virtually every app.
  * "type": per-character KEYEVENTF_UNICODE SendInput events. Slower but
    works where paste is blocked; layout-independent.

Both run in a background thread; SendInput targets whichever window holds
keyboard focus, so the user's focus is never touched.

Known Windows limitation (UIPI): a non-elevated process cannot inject input
into an elevated (Run-as-administrator) window. Run VoxWin elevated if you
need to dictate into elevated apps.
"""

from __future__ import annotations

import ctypes
import logging
import struct
import time
from ctypes import wintypes
from typing import List, Optional, Tuple

import win32api
import win32clipboard
import win32con
import win32process
import win32security

from .config import Config

log = logging.getLogger(__name__)

user32 = ctypes.WinDLL("user32", use_last_error=True)

# --- SendInput plumbing --------------------------------------------------------

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12
VK_LSHIFT, VK_RSHIFT = 0xA0, 0xA1
VK_LCONTROL, VK_RCONTROL = 0xA2, 0xA3
VK_LMENU, VK_RMENU = 0xA4, 0xA5
VK_LWIN, VK_RWIN = 0x5B, 0x5C
VK_RETURN, VK_TAB = 0x0D, 0x09
VK_V = 0x56

ULONG_PTR = ctypes.c_size_t


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUT_UNION(ctypes.Union):
    _fields_ = (("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUT_UNION))


class InjectionError(Exception):
    pass


def _key_input(vk: int = 0, scan: int = 0, flags: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki = KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
    return inp


def _send_inputs(inputs: List[INPUT]) -> None:
    if not inputs:
        return
    array = (INPUT * len(inputs))(*inputs)
    sent = user32.SendInput(len(inputs), array, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        # Note: UIPI blocking is NOT detectable here — Windows documents that
        # SendInput fails silently (full count, no error) when blocked by an
        # elevated window. That case is caught up-front by the elevation check.
        raise InjectionError(
            f"SendInput доставил {sent}/{len(inputs)} событий "
            f"(WinError {ctypes.get_last_error()})."
        )


def _release_modifiers() -> None:
    """Send key-up for every modifier so a held hotkey can't corrupt Ctrl+V."""
    ups = [
        _key_input(vk=vk, flags=KEYEVENTF_KEYUP)
        for vk in (
            VK_LSHIFT, VK_RSHIFT, VK_SHIFT,
            VK_LCONTROL, VK_RCONTROL, VK_CONTROL,
            VK_LMENU, VK_RMENU, VK_MENU,
            VK_LWIN, VK_RWIN,
        )
    ]
    _send_inputs(ups)


def _send_ctrl_v() -> None:
    _send_inputs(
        [
            _key_input(vk=VK_CONTROL),
            _key_input(vk=VK_V),
            _key_input(vk=VK_V, flags=KEYEVENTF_KEYUP),
            _key_input(vk=VK_CONTROL, flags=KEYEVENTF_KEYUP),
        ]
    )


def _type_unicode(text: str, char_delay_ms: int) -> None:
    """Send text as KEYEVENTF_UNICODE events in small batches."""
    batch: List[INPUT] = []

    def flush() -> None:
        nonlocal batch
        if batch:
            _send_inputs(batch)
            batch = []
            time.sleep(max(char_delay_ms, 1) / 1000.0 if char_delay_ms else 0.003)

    for ch in text.replace("\r\n", "\n"):
        if ch == "\n":
            flush()
            _send_inputs(
                [
                    _key_input(vk=VK_RETURN),
                    _key_input(vk=VK_RETURN, flags=KEYEVENTF_KEYUP),
                ]
            )
            continue
        if ch == "\t":
            flush()
            _send_inputs(
                [_key_input(vk=VK_TAB), _key_input(vk=VK_TAB, flags=KEYEVENTF_KEYUP)]
            )
            continue
        # UTF-16 code units (handles emoji/surrogate pairs correctly)
        units = [int.from_bytes(pair, "little") for pair in _utf16_units(ch)]
        for unit in units:
            batch.append(_key_input(scan=unit, flags=KEYEVENTF_UNICODE))
            batch.append(
                _key_input(scan=unit, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)
            )
        if len(batch) >= 64:
            flush()
    flush()


def _utf16_units(ch: str) -> List[bytes]:
    raw = ch.encode("utf-16-le")
    return [raw[i : i + 2] for i in range(0, len(raw), 2)]


# --- UIPI (elevated windows) -----------------------------------------------------

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _process_elevated(hproc) -> bool:
    token = win32security.OpenProcessToken(hproc, win32con.TOKEN_QUERY)
    try:
        return bool(
            win32security.GetTokenInformation(token, win32security.TokenElevation)
        )
    finally:
        token.Close()


def foreground_window_blocked_by_uipi() -> bool:
    """True when the focused window belongs to an elevated process while we
    are not elevated: Windows then swallows SendInput *silently* (documented
    UIPI behavior — no error code, full success count), so it must be
    detected up-front rather than from SendInput's result."""
    try:
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        _tid, pid = win32process.GetWindowThreadProcessId(hwnd)
        if not pid or pid == win32api.GetCurrentProcessId():
            return False
        if _process_elevated(win32api.GetCurrentProcess()):
            return False  # we are elevated ourselves — UIPI cannot block us
        hproc = win32api.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        try:
            return _process_elevated(hproc)
        finally:
            hproc.Close()
    except Exception:
        # Cannot determine (protected process etc.) — do not block injection.
        return False


# --- Clipboard handling ------------------------------------------------------------

CF_DIBV5 = getattr(win32con, "CF_DIBV5", 17)

_TEXT_FORMATS = (win32con.CF_UNICODETEXT,)
# HGLOBAL formats that round-trip as bytes through pywin32 — enough to
# preserve copied images (Windows re-synthesizes CF_BITMAP from CF_DIB).
_BYTES_FORMATS = (win32con.CF_DIB, CF_DIBV5)


def _pack_dropfiles(paths) -> bytes:
    """Build a DROPFILES structure (CF_HDROP payload) from file paths."""
    header = struct.pack("<IllII", 20, 0, 0, 0, 1)  # pFiles=20, pt, fNC, fWide
    body = "".join(f"{p}\0" for p in paths) + "\0"
    return header + body.encode("utf-16-le")


def _open_clipboard(retries: int = 15, delay: float = 0.02) -> None:
    for attempt in range(retries):
        try:
            win32clipboard.OpenClipboard(0)
            return
        except Exception:
            time.sleep(delay)
    raise InjectionError("Буфер обмена занят другим приложением")


def _snapshot_clipboard() -> Optional[List[Tuple[int, object]]]:
    """Best-effort snapshot of text-like clipboard contents.

    Returns None when the clipboard holds only formats we can't safely
    restore (images, files) — in that case we skip restoration entirely
    rather than destroy the user's data.
    """
    _open_clipboard()
    try:
        formats = []
        fmt = 0
        while True:
            fmt = win32clipboard.EnumClipboardFormats(fmt)
            if fmt == 0:
                break
            formats.append(fmt)

        if not formats:
            return []  # clipboard was empty; restore == clear

        snapshot: List[Tuple[int, object]] = []
        for fmt in formats:
            if fmt in _TEXT_FORMATS:
                try:
                    snapshot.append((fmt, win32clipboard.GetClipboardData(fmt)))
                except Exception:
                    pass
            elif fmt in _BYTES_FORMATS:  # images: CF_DIB / CF_DIBV5
                try:
                    data = win32clipboard.GetClipboardData(fmt)
                    if isinstance(data, bytes):
                        snapshot.append((fmt, data))
                except Exception:
                    pass
            elif fmt == win32con.CF_HDROP:  # copied files in Explorer
                try:
                    paths = win32clipboard.GetClipboardData(fmt)
                    if isinstance(paths, tuple) and paths:
                        snapshot.append((fmt, tuple(str(p) for p in paths)))
                except Exception:
                    pass
            elif fmt >= 0xC000:  # registered formats: HTML Format, RTF, ...
                try:
                    data = win32clipboard.GetClipboardData(fmt)
                    if isinstance(data, (bytes, str)):
                        snapshot.append((fmt, data))
                except Exception:
                    pass

        if not snapshot:
            log.info("Clipboard holds only unsupported formats; will not restore")
            return None
        return snapshot
    finally:
        win32clipboard.CloseClipboard()


def _set_clipboard_text(text: str) -> None:
    _open_clipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


def _restore_clipboard(snapshot: Optional[List[Tuple[int, object]]]) -> None:
    if snapshot is None:
        return
    _open_clipboard()
    try:
        win32clipboard.EmptyClipboard()
        for fmt, data in snapshot:
            try:
                if fmt == win32con.CF_HDROP and isinstance(data, tuple):
                    win32clipboard.SetClipboardData(fmt, _pack_dropfiles(data))
                else:
                    win32clipboard.SetClipboardData(fmt, data)
            except Exception:
                log.debug("Could not restore clipboard format %s", fmt)
    finally:
        win32clipboard.CloseClipboard()


# --- Public API ------------------------------------------------------------------------


def inject_text(text: str, cfg: Config) -> None:
    """Insert `text` into the currently focused input field."""
    if not text:
        return
    if foreground_window_blocked_by_uipi():
        raise InjectionError(
            "Активное окно запущено от имени администратора — Windows (UIPI) "
            "молча блокирует вставку. Запустите VoxWin от администратора, "
            "чтобы диктовать в такие окна."
        )
    if cfg.trailing_space and not text.endswith((" ", "\n", "\t")):
        text += " "

    _release_modifiers()
    time.sleep(0.02)

    if cfg.injection_method == "type":
        _type_unicode(text, cfg.type_char_delay_ms)
        return

    snapshot = _snapshot_clipboard() if cfg.restore_clipboard else None
    _set_clipboard_text(text)
    seq_after_set = user32.GetClipboardSequenceNumber()
    time.sleep(0.05)  # let clipboard viewers settle before we paste
    _send_ctrl_v()
    if cfg.restore_clipboard:
        # The target app reads the clipboard asynchronously after WM_PASTE.
        time.sleep(cfg.clipboard_restore_delay_ms / 1000.0)
        if user32.GetClipboardSequenceNumber() != seq_after_set:
            # Someone else wrote to the clipboard meanwhile — restoring now
            # would clobber THEIR data, not ours.
            log.info("Clipboard changed by another app; skipping restore")
            return
        try:
            _restore_clipboard(snapshot)
        except InjectionError:
            log.warning("Could not restore clipboard (busy)")


def copy_to_clipboard(text: str) -> None:
    _set_clipboard_text(text)
