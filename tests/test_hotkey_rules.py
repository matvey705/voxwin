"""Tests for the hotkey conflict validator (Windows/system shortcuts)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxwin.hotkeys import validate_hotkey  # noqa: E402


class TestValidateHotkey(unittest.TestCase):
    def level(self, combo):
        return validate_hotkey(combo)[0]

    # --- ok ---------------------------------------------------------------
    def test_f_keys_are_ok(self):
        for combo in ("f2", "f9", "F10", "f12", "f24"):
            self.assertEqual(self.level(combo), "ok", combo)

    def test_multi_modifier_combos_are_ok(self):
        for combo in ("ctrl+alt+space", "ctrl+shift+f9", "ctrl+alt+shift+p"):
            self.assertNotEqual(self.level(combo), "block", combo)

    def test_empty_means_disabled(self):
        self.assertEqual(self.level(""), "ok")
        self.assertEqual(self.level("   "), "ok")

    # --- block: breaks Windows or typing ----------------------------------
    def test_system_shortcuts_blocked(self):
        for combo in ("alt+tab", "alt+f4", "ctrl+esc", "ctrl+shift+esc",
                      "ctrl+alt+del", "ctrl+alt+delete", "win+l", "alt+space"):
            self.assertEqual(self.level(combo), "block", combo)

    def test_bare_typing_keys_blocked(self):
        for combo in ("a", "z", "5", "space", "enter", "backspace",
                      "esc", "left", "ф"):
            self.assertEqual(self.level(combo), "block", combo)

    def test_shift_plus_letter_blocked(self):
        for combo in ("shift+a", "shift+7", "shift+space"):
            self.assertEqual(self.level(combo), "block", combo)

    def test_modifier_only_blocked(self):
        for combo in ("ctrl", "ctrl+alt", "shift"):
            self.assertEqual(self.level(combo), "block", combo)

    # --- warn: taken by apps / OS-reserved family -------------------------
    def test_common_app_shortcuts_warn(self):
        for combo in ("ctrl+c", "ctrl+v", "ctrl+s", "ctrl+1"):
            self.assertEqual(self.level(combo), "warn", combo)

    def test_win_combos_warn(self):
        for combo in ("win+f9", "windows+d", "win+shift+s"):
            self.assertEqual(self.level(combo), "warn", combo)

    def test_alt_letter_and_altgr_warn(self):
        self.assertEqual(self.level("alt+f"), "warn")
        self.assertEqual(self.level("ctrl+alt+e"), "warn")
        self.assertEqual(self.level("f1"), "warn")

    # --- aliases ----------------------------------------------------------
    def test_aliases_normalized(self):
        self.assertEqual(self.level("cmd+l"), "block")   # cmd -> windows, win+l
        self.assertEqual(self.level("control+esc"), "block")

    def test_messages_are_human_readable(self):
        for combo in ("alt+tab", "ctrl+c", "f9", "a"):
            level, message = validate_hotkey(combo)
            self.assertTrue(len(message) > 5, combo)


if __name__ == "__main__":
    unittest.main()
