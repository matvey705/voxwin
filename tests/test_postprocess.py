"""Unit tests for the offline text post-processing (no heavy deps needed)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxwin.config import Config  # noqa: E402
from voxwin.postprocess import PostProcessor  # noqa: E402


def make_cfg(**overrides) -> Config:
    cfg = Config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class TestPostProcessor(unittest.TestCase):
    def test_basic_cleanup(self):
        pp = PostProcessor(make_cfg())
        self.assertEqual(pp.process("  привет ,  мир .  "), "Привет, мир.")

    def test_capitalize_first(self):
        pp = PostProcessor(make_cfg())
        self.assertEqual(pp.process("привет"), "Привет")

    def test_capitalize_disabled(self):
        pp = PostProcessor(make_cfg(capitalize_first=False))
        self.assertEqual(pp.process("привет"), "привет")

    def test_disabled_post(self):
        pp = PostProcessor(make_cfg(post_enabled=False))
        self.assertEqual(pp.process("  эм ну текст  "), "эм ну текст")

    def test_vocab_replacement_case_insensitive(self):
        pp = PostProcessor(
            make_cfg(vocab_replacements={"вокс вин": "VoxWin", "джира": "Jira"})
        )
        self.assertEqual(
            pp.process("открой Джира и Вокс Вин"), "Открой Jira и VoxWin"
        )

    def test_vocab_longest_first(self):
        pp = PostProcessor(
            make_cfg(vocab_replacements={"пай": "py", "пай чарм": "PyCharm"})
        )
        self.assertEqual(pp.process("запусти пай чарм"), "Запусти PyCharm")

    def test_no_partial_word_replacement(self):
        pp = PostProcessor(make_cfg(vocab_replacements={"кот": "cat"}))
        self.assertEqual(pp.process("который час"), "Который час")

    def test_fillers_removed(self):
        pp = PostProcessor(make_cfg(remove_fillers=True))
        self.assertEqual(pp.process("эм, привет, ммм, мир"), "Привет, мир")

    def test_fillers_english(self):
        pp = PostProcessor(make_cfg(remove_fillers=True))
        self.assertEqual(pp.process("um, hello uh world"), "Hello world")

    def test_fillers_do_not_eat_real_words(self):
        pp = PostProcessor(make_cfg(remove_fillers=True))
        # "умный" contains "ум" but must survive
        self.assertEqual(pp.process("умный дом"), "Умный дом")

    def test_fillers_off_by_default(self):
        pp = PostProcessor(make_cfg())
        self.assertEqual(pp.process("эм, привет"), "Эм, привет")

    def test_custom_fillers(self):
        pp = PostProcessor(make_cfg(remove_fillers=True, custom_fillers=["короче"]))
        self.assertEqual(pp.process("короче, идём дальше"), "Идём дальше")

    def test_empty_input(self):
        pp = PostProcessor(make_cfg())
        self.assertEqual(pp.process(""), "")
        self.assertEqual(pp.process("   "), "")

    def test_punctuation_spacing(self):
        pp = PostProcessor(make_cfg())
        self.assertEqual(pp.process("да , конечно !"), "Да, конечно!")


if __name__ == "__main__":
    unittest.main()
