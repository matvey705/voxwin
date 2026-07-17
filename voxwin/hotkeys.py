"""Global hotkeys via low-level keyboard hooks (`keyboard` library).

Two bindings are active simultaneously:
  * PTT hotkey  — hold to record, release to transcribe;
  * toggle hotkey — press once to start, once again to stop.

IMPORTANT: callbacks fire inside the keyboard hook thread. They must return
immediately — the app bridges them onto the Qt main thread via signals.

Implementation notes (verified against keyboard 0.13.5 sources):
  * The PTT key uses a single keyboard.hook_key() per key. Two separate
    on_press_key/on_release_key registrations would corrupt the library's
    internal `_hooks[key]` bookkeeping (the second unhook raises KeyError
    before removing the callback, leaving a stale hook installed).
  * For a suppressed (blocking) hook the callback's return value decides
    the event's fate: truthy — pass the key through to applications,
    falsy — swallow it. So a PTT combo like "ctrl+space" must return True
    when the modifiers are not held, otherwise plain Space would be dead
    system-wide.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

import keyboard

from .config import Config

log = logging.getLogger(__name__)

_MOD_ALIASES = {
    "win": "windows",
    "cmd": "windows",
    "control": "ctrl",
    "option": "alt",
}


def _normalize_part(part: str) -> str:
    part = part.strip().lower()
    return _MOD_ALIASES.get(part, part)


# --- Conflict validation against Windows / application shortcuts -------------
#
# Levels:
#   "block" — сочетание нельзя сохранить: либо его physically нельзя
#             перехватить (Ctrl+Alt+Del), либо оно сломает базовый ввод
#             (одиночная буква, Shift+буква) или критичные функции ОС;
#   "warn"  — сохранить можно, но сочетание почти наверняка занято
#             (Ctrl+C, Win+..., Alt+буква) — пользователь предупреждён;
#   "ok"    — свободно.

_MODIFIERS = {"ctrl", "alt", "shift", "windows"}

_SAFE_BARE_KEYS = {f"f{i}" for i in range(2, 25)} | {"scroll lock", "pause"}

_TYPING_KEYS = {
    "space", "enter", "return", "tab", "backspace", "delete", "del",
    "esc", "escape", "up", "down", "left", "right",
    "home", "end", "page up", "page down", "insert",
}

_SYSTEM_BLOCKED = {
    (frozenset({"alt"}), "tab"),
    (frozenset({"alt", "shift"}), "tab"),
    (frozenset({"alt"}), "f4"),
    (frozenset({"alt"}), "esc"),
    (frozenset({"alt"}), "space"),
    (frozenset({"ctrl"}), "esc"),
    (frozenset({"ctrl", "shift"}), "esc"),
    (frozenset({"ctrl", "alt"}), "delete"),
    (frozenset({"ctrl", "alt"}), "del"),
    (frozenset({"windows"}), "l"),
}


def validate_hotkey(combo: str) -> tuple:
    """Классифицирует сочетание: ('ok'|'warn'|'block', пояснение).

    Ловит конфликты с базовыми сочетаниями Windows и обычным набором
    текста ДО того, как пользователь сохранит настройку.
    """
    combo = (combo or "").strip().lower()
    if not combo:
        return "ok", "Отключено."

    parts = [_normalize_part(p) for p in combo.split("+") if p.strip()]
    if not parts:
        return "block", "Пустое сочетание."
    *mod_parts, key = parts
    mods = frozenset(mod_parts)

    if not mods.issubset(_MODIFIERS):
        return "block", "Модификаторами могут быть только Ctrl, Alt, Shift и Win."
    if key in _MODIFIERS:
        return "block", "Нужна основная клавиша (например, Ctrl+Alt+Пробел)."
    if (mods, key) in _SYSTEM_BLOCKED:
        return "block", "Системное сочетание Windows — перехватывать его нельзя."

    if not mods:
        if key in _SAFE_BARE_KEYS:
            return "ok", "Отличный выбор: клавиша почти нигде не занята."
        if key == "f1":
            return "warn", "F1 — «Справка» во многих программах."
        return (
            "block",
            "Одиночная клавиша сломает обычный набор текста — добавьте "
            "модификатор или возьмите F-клавишу (F2–F12).",
        )

    if mods == frozenset({"shift"}) and (len(key) == 1 or key in _TYPING_KEYS):
        return "block", "Shift+символ — это ввод заглавных букв."
    if "windows" in mods:
        return (
            "warn",
            "Windows резервирует многие Win-сочетания — вместо диктовки может "
            "сработать системное действие.",
        )
    if mods == frozenset({"ctrl"}) and len(key) == 1:
        return (
            "warn",
            "Ctrl+буква/цифра занято почти во всех программах "
            "(копирование, вставка, сохранение…).",
        )
    if mods == frozenset({"ctrl", "shift"}) and len(key) == 1:
        return "warn", "Ctrl+Shift+буква часто занято в приложениях."
    if mods == frozenset({"alt"}) and len(key) == 1:
        return "warn", "Alt+буква открывает пункты меню программ."
    if mods == frozenset({"ctrl", "alt"}) and len(key) == 1:
        return (
            "warn",
            "Ctrl+Alt+символ = AltGr на некоторых раскладках — может "
            "вводить спецсимволы.",
        )
    return "ok", "Сочетание выглядит свободным."


class HotkeyManager:
    def __init__(
        self,
        on_ptt_down: Callable[[], None],
        on_ptt_up: Callable[[], None],
        on_toggle: Callable[[], None],
    ):
        self._on_ptt_down = on_ptt_down
        self._on_ptt_up = on_ptt_up
        self._on_toggle = on_toggle
        self._hook_removers: List[Callable[[], None]] = []
        self._hotkey_handles: List[object] = []
        self._ptt_active = False
        self._lock = threading.Lock()

    # ---------------------------------------------------------------------

    def apply(self, cfg: Config) -> None:
        """(Re)bind hotkeys according to config. Safe to call repeatedly."""
        self.unbind_all()
        suppress = cfg.suppress_hotkeys

        ptt = (cfg.ptt_hotkey or "").strip().lower()
        if ptt:
            self._bind_ptt(ptt, suppress)

        toggle = (cfg.toggle_hotkey or "").strip().lower()
        if toggle:
            try:
                handle = keyboard.add_hotkey(
                    toggle, self._on_toggle, suppress=suppress, trigger_on_release=False
                )
                self._hotkey_handles.append(handle)
            except (ValueError, KeyError) as exc:
                log.error("Invalid toggle hotkey %r: %s", toggle, exc)

    def _bind_ptt(self, combo: str, suppress: bool) -> None:
        parts = [_normalize_part(p) for p in combo.split("+") if p.strip()]
        if not parts:
            return
        modifiers, main_key = parts[:-1], parts[-1]

        def handler(event) -> bool:
            # Return value contract (suppress=True): True = pass the event
            # through to applications, False = swallow it.
            if event.event_type == keyboard.KEY_DOWN:
                with self._lock:
                    if self._ptt_active:
                        return False  # key auto-repeat while dictating
                    for mod in modifiers:
                        try:
                            if not keyboard.is_pressed(mod):
                                return True  # combo not engaged: normal key
                        except (ValueError, KeyError):
                            return True
                    self._ptt_active = True
                self._on_ptt_down()
                return False
            # KEY_UP
            with self._lock:
                if not self._ptt_active:
                    return True  # release of a non-PTT press
                self._ptt_active = False
            self._on_ptt_up()
            return False

        try:
            remover = keyboard.hook_key(main_key, handler, suppress=suppress)
            self._hook_removers.append(remover)
        except (ValueError, KeyError) as exc:
            log.error("Invalid PTT hotkey %r: %s", combo, exc)

    def unbind_all(self) -> None:
        for remover in self._hook_removers:
            try:
                remover()
            except (KeyError, ValueError):
                pass
        self._hook_removers = []
        for handle in self._hotkey_handles:
            try:
                keyboard.remove_hotkey(handle)
            except (KeyError, ValueError):
                pass
        self._hotkey_handles = []
        # If the user was mid-dictation (PTT held) when we rebind, the key
        # release will never reach the (removed) hook — deliver a synthetic
        # release so the recording isn't orphaned.
        with self._lock:
            was_active, self._ptt_active = self._ptt_active, False
        if was_active:
            self._on_ptt_up()

    def pause(self) -> None:
        """Temporarily remove all bindings (e.g. while capturing a new hotkey)."""
        self.unbind_all()

    @staticmethod
    def shutdown() -> None:
        try:
            keyboard.unhook_all()
        except Exception:
            pass


def read_hotkey_blocking() -> Optional[str]:
    """Block until the user presses a combination; returns e.g. 'ctrl+alt+d'.

    Runs on a worker thread from the settings dialog.
    """
    try:
        return keyboard.read_hotkey(suppress=False)
    except Exception:
        log.exception("Hotkey capture failed")
        return None
