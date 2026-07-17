"""Tests for the soft chimes and live-preview plumbing."""

import io
import sys
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voxwin import winutil  # noqa: E402
from voxwin.config import Config  # noqa: E402
from voxwin.transcriber import _is_repetition_hallucination  # noqa: E402


class TestRepetitionHallucination(unittest.TestCase):
    def test_flags_repetitive_filler(self):
        for text in (
            "blah blah blah blah blah",
            "you you you you",
            "так так так так так",
            "the the the the the the",
        ):
            self.assertTrue(_is_repetition_hallucination(text), text)

    def test_keeps_real_speech(self):
        for text in (
            "то есть объясни про обсидиан",
            "привет как дела",
            "buy milk and eggs today",
            "раз два три четыре",
        ):
            self.assertFalse(_is_repetition_hallucination(text), text)

    def test_short_repeats_pass(self):
        # 2-3 words are too short to judge; leave them alone.
        self.assertFalse(_is_repetition_hallucination("да да да"))
        self.assertFalse(_is_repetition_hallucination(""))


class TestChimes(unittest.TestCase):
    def test_all_kinds_produce_valid_wav(self):
        for kind in ("start", "stop", "done", "error"):
            data = winutil._synth_chime(kind, 35)
            self.assertIsNotNone(data, kind)
            with wave.open(io.BytesIO(data)) as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), winutil._SAMPLE_RATE)
                self.assertGreater(wav.getnframes(), 1000)

    def test_volume_bounds_amplitude(self):
        import numpy as np

        def peak(volume):
            data = winutil._synth_chime("start", volume)
            with wave.open(io.BytesIO(data)) as wav:
                frames = wav.readframes(wav.getnframes())
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32767
            return float(np.abs(samples).max())

        quiet, loud = peak(10), peak(100)
        self.assertLess(quiet, 0.09)      # 10% volume is genuinely quiet
        self.assertLess(loud, 0.60)       # even 100% stays far below clipping
        self.assertGreater(loud, quiet * 5)

    def test_unknown_kind_is_none(self):
        self.assertIsNone(winutil._synth_chime("nope", 50))

    def test_cache_reuse(self):
        first = winutil._synth_chime("done", 42)
        second = winutil._synth_chime("done", 42)
        self.assertIs(first, second)


class TestNewConfigFields(unittest.TestCase):
    def test_defaults(self):
        cfg = Config()
        self.assertTrue(cfg.live_preview)
        self.assertEqual(cfg.sound_volume, 35)

    def test_volume_clamped(self):
        cfg = Config()
        cfg.sound_volume = 500
        cfg.clamp()
        self.assertEqual(cfg.sound_volume, 100)
        cfg.sound_volume = "тихо"
        cfg.clamp()
        self.assertEqual(cfg.sound_volume, 35)


if __name__ == "__main__":
    unittest.main()
