"""Regression tests for defects found in the multi-agent code review."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxwin.config import Config  # noqa: E402
from voxwin.postprocess import PostProcessor  # noqa: E402


def make_cfg(**kwargs) -> Config:
    cfg = Config()
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


class TestVocabReplacementLiteral(unittest.TestCase):
    def test_backslashes_in_replacement_are_literal(self):
        # 'C:\Users\me' used to raise re.error: bad escape \U
        cfg = make_cfg(vocab_replacements={"домашняя папка": r"C:\Users\me"})
        out = PostProcessor(cfg).process("открой домашняя папка")
        self.assertIn(r"C:\Users\me", out)

    def test_group_reference_in_replacement_is_literal(self):
        cfg = make_cfg(vocab_replacements={"тег": r"\1x"})
        out = PostProcessor(cfg).process("вставь тег сюда")
        self.assertIn(r"\1x", out)


class TestFillerPunctuation(unittest.TestCase):
    def test_no_comma_period_after_trailing_filler(self):
        cfg = make_cfg(remove_fillers=True)
        out = PostProcessor(cfg).process("Я думаю, эм.")
        self.assertEqual(out, "Я думаю.")

    def test_comma_before_question_mark(self):
        cfg = make_cfg(remove_fillers=True)
        out = PostProcessor(cfg).process("Так же, ммм?")
        self.assertEqual(out, "Так же?")


class TestConfigClampRobustness(unittest.TestCase):
    def _load_from(self, data: dict) -> Config:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return Config.load(path)

    def test_null_float_does_not_crash(self):
        cfg = self._load_from({"vad_threshold": None})
        self.assertEqual(cfg.vad_threshold, Config().vad_threshold)

    def test_word_beam_size_does_not_crash(self):
        cfg = self._load_from({"beam_size": "high"})
        self.assertEqual(cfg.beam_size, Config().beam_size)

    def test_russian_decimal_comma_does_not_crash(self):
        cfg = self._load_from({"vad_threshold": "0,5"})
        self.assertEqual(cfg.vad_threshold, Config().vad_threshold)

    def test_non_string_filler_elements_coerced(self):
        cfg = self._load_from({"custom_fillers": ["em", 42, None]})
        self.assertEqual(cfg.custom_fillers, ["em", "42"])
        # This join is exactly what the settings dialog does on open:
        self.assertEqual(", ".join(cfg.custom_fillers), "em, 42")

    def test_non_string_vocab_coerced(self):
        cfg = self._load_from({"vocab_replacements": {"a": 1, "b": None}})
        self.assertEqual(cfg.vocab_replacements, {"a": "1"})

    def test_numeric_string_fields_survive(self):
        cfg = self._load_from({"model_size": 5, "language": None})
        self.assertIsInstance(cfg.model_size, str)
        self.assertIsInstance(cfg.language, str)


class TestHistoryShrinkSemantics(unittest.TestCase):
    def test_islice_keeps_newest(self):
        # Documents the app.py fix: newest-first deque + islice keeps head.
        from collections import deque
        from itertools import islice

        history = deque(maxlen=20)
        for i in range(1, 7):
            history.appendleft(f"utt{i}")  # utt6 is newest, index 0
        shrunk = deque(islice(history, 0, 3), maxlen=3)
        self.assertEqual(list(shrunk), ["utt6", "utt5", "utt4"])


if __name__ == "__main__":
    unittest.main()
