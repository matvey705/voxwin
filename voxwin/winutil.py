"""Windows-specific utilities: single instance, autostart, CUDA DLLs,
logging setup and sound feedback."""

from __future__ import annotations

import ctypes
import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path

from . import APP_NAME, APP_ID
from .config import LOG_DIR

log = logging.getLogger(__name__)

_mutex_handle = None  # keep a reference so the mutex lives for the process


def inject_system_certificates() -> None:
    """Make all in-process HTTPS (model downloads via httpx/requests/urllib)
    trust the Windows certificate store — vital behind TLS-intercepting
    proxies/antiviruses whose root CA is not in certifi."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        log.debug("truststore not available; using default certifi bundle")


def strip_proxies() -> None:
    """Neutralize system/env proxies for this process.

    Windows boxes often carry a dead or SOCKS-scheme system proxy in the
    registry (v2ray & co); httpx/urllib pick it up via getproxies() and
    model downloads die with 'Unknown scheme for proxy URL'. NO_PROXY='*'
    makes httpx/requests/urllib bypass proxies entirely.
    """
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    log.info("Proxies disabled for this process (NO_PROXY=*)")


def looks_like_proxy_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__} {exc}".lower()
    return "proxy" in message or "socks" in message


def acquire_single_instance() -> bool:
    """Return True if we are the only VoxWin instance."""
    global _mutex_handle
    try:
        import win32event
        import winerror
        import win32api

        # Session-local name (no "Global\\"): VoxWin is per-user — another
        # user's RDP/fast-switch session must be able to run its own copy.
        _mutex_handle = win32event.CreateMutex(None, False, "VoxWinSingleton")
        return win32api.GetLastError() != winerror.ERROR_ALREADY_EXISTS
    except Exception:
        log.exception("Single-instance check failed; continuing anyway")
        return True


def set_app_user_model_id() -> None:
    """Give the process a stable identity for taskbar grouping/notifications."""
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def message_box(text: str, title: str = APP_NAME) -> None:
    ctypes.windll.user32.MessageBoxW(None, text, title, 0x40)  # MB_ICONINFORMATION


# --- Autostart ---------------------------------------------------------------

_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def autostart_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    exe = pythonw if pythonw.exists() else Path(sys.executable)
    # NOT "-m voxwin": processes started from the Run key get an arbitrary
    # cwd, where the package is not importable. The launcher script's own
    # directory lands on sys.path, so an absolute script path always works.
    launcher = Path(__file__).resolve().parents[1] / "voxwin_launcher.py"
    return f'"{exe}" "{launcher}"'


def is_autostart_enabled() -> bool:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            winreg.QueryValueEx(key, APP_NAME)
        return True
    except OSError:
        return False


def set_autostart(enabled: bool) -> None:
    import winreg

    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE
    ) as key:
        if enabled:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, autostart_command())
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
            except OSError:
                pass


# --- CUDA DLLs from pip-installed NVIDIA wheels -------------------------------


def add_cuda_dll_dirs() -> None:
    """Make cuBLAS/cuDNN installed via `pip install nvidia-*-cu12` loadable.

    Pip wheels put the DLLs under site-packages/nvidia/*/bin. CTranslate2
    loads them with a plain LoadLibrary, which searches PATH and IGNORES
    os.add_dll_directory — so the bin dirs must be prepended to PATH too
    (add_dll_directory is kept for libraries that do use the new search).
    """
    try:
        import nvidia  # type: ignore

        prepend: list[str] = []
        for base in nvidia.__path__:
            base_path = Path(base)
            for sub in base_path.iterdir():
                bin_dir = sub / "bin"
                if bin_dir.is_dir():
                    prepend.append(str(bin_dir))
                    try:
                        os.add_dll_directory(str(bin_dir))
                    except OSError:
                        pass
        if prepend:
            current = os.environ.get("PATH", "")
            missing = [p for p in prepend if p not in current]
            if missing:
                os.environ["PATH"] = os.pathsep.join(missing + [current])
    except ImportError:
        pass
    except Exception:
        log.exception("Failed to register CUDA DLL directories")


# --- Sounds -------------------------------------------------------------------
#
# winsound.Beep is a harsh full-volume square wave. Instead we synthesize
# soft sine chimes (gentle attack, exponential decay, quiet 2nd harmonic)
# and play them from memory. Volume is user-configurable.

_SAMPLE_RATE = 44100

# kind -> [(frequency_hz, duration_s, gap_after_s), ...]
_CHIME_NOTES = {
    "start": [(523.25, 0.09, 0.012), (783.99, 0.16, 0.0)],   # C5 -> G5, rising
    "stop": [(783.99, 0.07, 0.012), (523.25, 0.14, 0.0)],    # G5 -> C5, falling
    "done": [(1046.50, 0.16, 0.0)],                           # soft C6 ding
    "error": [(196.00, 0.11, 0.05), (174.61, 0.15, 0.0)],    # low double knock
}

_chime_cache: dict = {}


def _synth_chime(kind: str, volume: int) -> bytes | None:
    """Build an in-memory WAV for the given cue at the given volume (5..100)."""
    notes = _CHIME_NOTES.get(kind)
    if not notes:
        return None
    key = (kind, int(volume))
    cached = _chime_cache.get(key)
    if cached is not None:
        return cached

    import io
    import wave

    import numpy as np

    amplitude = 0.45 * max(5, min(100, int(volume))) / 100.0
    parts = []
    for freq, duration, gap in notes:
        n = int(_SAMPLE_RATE * duration)
        t = np.arange(n, dtype=np.float64) / _SAMPLE_RATE
        tone = np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        attack = np.minimum(1.0, t / 0.006)          # 6 ms fade-in, no click
        decay = np.exp(-t * (5.0 / duration))        # smooth ring-out
        parts.append(tone * attack * decay)
        if gap:
            parts.append(np.zeros(int(_SAMPLE_RATE * gap)))
    signal = np.concatenate(parts) * amplitude
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
        wav.writeframes(pcm)
    data = buf.getvalue()
    _chime_cache[key] = data  # must outlive async playback (SND_MEMORY)
    return data


def play_sound(kind: str, volume: int = 35) -> None:
    """Short non-blocking audio cues: record start/stop, completion, error."""
    try:
        import winsound

        data = _synth_chime(kind, volume)
        if data is None:
            return
        winsound.PlaySound(
            data,
            winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
    except Exception:
        pass


# --- Logging ------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
    )
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "voxwin.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    if sys.stderr is not None:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

    def _excepthook(exc_type, exc, tb):
        root.critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        root.critical(
            "Unhandled exception in thread %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook
