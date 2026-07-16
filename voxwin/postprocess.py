"""Local text post-processing: vocabulary corrections, filler removal,
capitalization and whitespace/punctuation cleanup.

Whisper already produces punctuation; this stage fixes domain terms and
tidies the result before injection. Everything is pure-Python and offline.
"""

from __future__ import annotations

import re
from typing import Iterable

from .config import Config

# Conservative filler lists: only sounds that are almost never real words.
# ("ну", "как бы", "like" are legitimate words too often — users can add them
# to custom_fillers explicitly.)
DEFAULT_FILLERS_RU = ["эм", "эмм", "эммм", "ээ", "эээ", "мм", "ммм", "ааа", "а-а"]
DEFAULT_FILLERS_EN = ["um", "umm", "uh", "uhh", "uhm", "erm", "hm", "hmm", "mhm"]

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:%)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{«])\s+")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_COMMA = re.compile(r"([,;])\s*(?=[,;])")
_DANGLING_COMMA = re.compile(r"^\s*[,;.]+\s*")
# ",." appears when a filler between a comma and the final period is removed
# ("Я думаю, эм." -> "Я думаю,."). Never legitimate typography.
_COMMA_BEFORE_TERMINAL = re.compile(r"[,;]\s*(?=[.!?…])")


def _build_filler_pattern(fillers: Iterable[str]) -> re.Pattern | None:
    words = [w.strip() for w in fillers if w and w.strip()]
    if not words:
        return None
    alternatives = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    # A filler is a standalone word, optionally followed by a comma.
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)\s*,?\s*", re.IGNORECASE)


class PostProcessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._filler_re = _build_filler_pattern(
            DEFAULT_FILLERS_RU + DEFAULT_FILLERS_EN + list(cfg.custom_fillers)
        )
        # Longest keys first so "джава скрипт" wins over "джава".
        self._replacements = sorted(
            ((k, v) for k, v in cfg.vocab_replacements.items() if k.strip()),
            key=lambda kv: len(kv[0]),
            reverse=True,
        )
        self._replacement_res = [
            (re.compile(rf"(?<!\w){re.escape(k.strip())}(?!\w)", re.IGNORECASE), v)
            for k, v in self._replacements
        ]

    def process(self, text: str) -> str:
        s = text.strip()
        s = _MULTI_SPACE.sub(" ", s)
        if not self.cfg.post_enabled:
            return s

        for pattern, replacement in self._replacement_res:
            # Substitute via a function so the user's replacement text is
            # literal — re.sub templates would choke on backslashes
            # ("C:\Users\..." -> re.error: bad escape) or expand "\1".
            s = pattern.sub(lambda _m, _v=replacement: _v, s)

        if self.cfg.remove_fillers and self._filler_re is not None:
            s = self._filler_re.sub("", s)

        s = _MULTI_SPACE.sub(" ", s)
        s = _COMMA_BEFORE_TERMINAL.sub("", s)
        s = _SPACE_BEFORE_PUNCT.sub(r"\1", s)
        s = _SPACE_AFTER_OPEN.sub(r"\1", s)
        s = _MULTI_COMMA.sub("", s)
        s = _DANGLING_COMMA.sub("", s)
        s = s.strip()

        if self.cfg.capitalize_first and s:
            for i, ch in enumerate(s):
                if ch.isalpha():
                    s = s[:i] + ch.upper() + s[i + 1 :]
                    break

        return s
