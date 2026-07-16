"""Config persistence tests: roundtrip, unknown keys, clamping."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxwin.config import Config  # noqa: E402


class TestConfig(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "config.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip(self):
        cfg = Config()
        cfg.model_size = "medium"
        cfg.language = "ru"
        cfg.vocab_replacements = {"джава": "Java"}
        cfg.vocab_terms = ["Kubernetes", "PySide6"]
        cfg.save(self.path)

        loaded = Config.load(self.path)
        self.assertEqual(loaded.model_size, "medium")
        self.assertEqual(loaded.language, "ru")
        self.assertEqual(loaded.vocab_replacements, {"джава": "Java"})
        self.assertEqual(loaded.vocab_terms, ["Kubernetes", "PySide6"])

    def test_missing_file_gives_defaults(self):
        loaded = Config.load(self.path)
        self.assertEqual(loaded.model_size, Config().model_size)

    def test_unknown_keys_ignored(self):
        self.path.write_text(
            json.dumps({"model_size": "tiny", "some_future_key": 42}),
            encoding="utf-8",
        )
        loaded = Config.load(self.path)
        self.assertEqual(loaded.model_size, "tiny")

    def test_corrupt_file_gives_defaults(self):
        self.path.write_text("{not json", encoding="utf-8")
        loaded = Config.load(self.path)
        self.assertEqual(loaded.model_size, Config().model_size)

    def test_clamping(self):
        self.path.write_text(
            json.dumps(
                {
                    "beam_size": 999,
                    "device": "tpu",
                    "injection_method": "telepathy",
                    "vad_threshold": 5.0,
                }
            ),
            encoding="utf-8",
        )
        loaded = Config.load(self.path)
        self.assertEqual(loaded.beam_size, 10)
        self.assertEqual(loaded.device, "auto")
        self.assertEqual(loaded.injection_method, "clipboard")
        self.assertLessEqual(loaded.vad_threshold, 0.95)

    def test_effective_language(self):
        cfg = Config()
        self.assertIsNone(cfg.effective_language())
        cfg.language = "ru"
        self.assertEqual(cfg.effective_language(), "ru")


if __name__ == "__main__":
    unittest.main()
