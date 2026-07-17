"""Application configuration: a flat dataclass persisted as JSON.

Config lives in %APPDATA%\\VoxWin\\config.json, models are cached in
%LOCALAPPDATA%\\VoxWin\\models (overridable). Unknown keys in the JSON are
ignored so configs survive upgrades in both directions.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

APP_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / "VoxWin"
CONFIG_PATH = APP_DIR / "config.json"
LOG_DIR = APP_DIR / "logs"
DEFAULT_MODELS_DIR = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "VoxWin" / "models"
)

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
DEVICE_CHOICES = ["auto", "cpu", "cuda"]
COMPUTE_CHOICES = ["auto", "int8", "int8_float16", "float16", "float32"]
INJECTION_CHOICES = ["clipboard", "type"]

LANGUAGES = [
    ("auto", "Автоопределение"),
    ("ru", "Русский"),
    ("en", "English"),
    ("uk", "Українська"),
    ("de", "Deutsch"),
    ("fr", "Français"),
    ("es", "Español"),
    ("it", "Italiano"),
    ("pt", "Português"),
    ("pl", "Polski"),
    ("tr", "Türkçe"),
    ("zh", "中文"),
    ("ja", "日本語"),
    ("ko", "한국어"),
    ("ar", "العربية"),
]

DEFAULT_OLLAMA_PROMPT = (
    "Ты корректор надиктованного текста. Исправь пунктуацию и очевидные "
    "ошибки распознавания речи, не меняя смысл, стиль и язык текста. "
    "Верни ТОЛЬКО исправленный текст, без кавычек и пояснений.\n\n"
    "Текст: {text}"
)


@dataclass
class Config:
    # --- Speech engine -----------------------------------------------------
    model_size: str = "small"          # any of MODEL_CHOICES or a HF repo path
    device: str = "auto"               # auto | cpu | cuda
    compute_type: str = "auto"         # auto | int8 | int8_float16 | float16 | float32
    language: str = "auto"             # "auto" or ISO code ("ru", "en", ...)
    beam_size: int = 3
    cpu_threads: int = 0               # 0 = library default
    models_dir: str = ""               # empty -> DEFAULT_MODELS_DIR
    preload_model: bool = True

    # --- VAD (Silero, built into faster-whisper) ---------------------------
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 400
    vad_speech_pad_ms: int = 300

    # --- Hotkeys ------------------------------------------------------------
    ptt_hotkey: str = "f9"             # hold to talk; "" disables
    toggle_hotkey: str = "f10"         # press to start/stop; "" disables
    suppress_hotkeys: bool = True      # swallow the hotkey so apps don't see it

    # --- Audio --------------------------------------------------------------
    input_device: str = ""             # device name substring; "" = system default
    max_record_seconds: int = 120      # safety auto-stop

    # --- Text insertion -----------------------------------------------------
    injection_method: str = "clipboard"   # clipboard | type
    restore_clipboard: bool = True
    clipboard_restore_delay_ms: int = 300
    trailing_space: bool = True
    type_char_delay_ms: int = 0        # extra delay per chunk in "type" mode

    # --- Post-processing ----------------------------------------------------
    post_enabled: bool = True
    capitalize_first: bool = True
    remove_fillers: bool = False
    custom_fillers: list = field(default_factory=list)       # extra filler words
    vocab_replacements: dict = field(default_factory=dict)   # "heard" -> "correct"
    vocab_terms: list = field(default_factory=list)          # bias terms for Whisper
    initial_prompt: str = ""

    # --- Optional local LLM refinement (Ollama) ------------------------------
    ollama_enabled: bool = False
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_prompt: str = DEFAULT_OLLAMA_PROMPT
    ollama_timeout_s: int = 20

    # --- UX -----------------------------------------------------------------
    sound_feedback: bool = True
    sound_volume: int = 35             # 5..100, % of full scale for chimes
    notify_on_success: bool = False    # errors are always notified
    overlay_enabled: bool = True
    live_preview: bool = True          # show partial transcript while recording
    history_size: int = 20

    # ------------------------------------------------------------------------
    def resolved_models_dir(self) -> Path:
        return Path(self.models_dir) if self.models_dir else DEFAULT_MODELS_DIR

    def effective_language(self) -> Optional[str]:
        return None if self.language in ("", "auto") else self.language

    def model_key(self) -> tuple:
        """Identity of the loaded model; if it changes, the model reloads."""
        return (
            self.model_size,
            self.device,
            self.compute_type,
            str(self.resolved_models_dir()),
            self.cpu_threads,
        )

    # ------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        p = path or CONFIG_PATH
        cfg = cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            names = {f.name for f in dataclasses.fields(cls)}
            for key, value in raw.items():
                if key in names:
                    setattr(cfg, key, value)
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("Failed to read config %s, using defaults", p)
        try:
            cfg.clamp()
        except Exception:
            log.exception("Config sanitation failed, using pristine defaults")
            cfg = cls()
        return cfg

    def save(self, path: Optional[Path] = None) -> None:
        p = path or CONFIG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clamp(self) -> None:
        """Coerce out-of-range / malformed values back to safe ones.

        Must never raise: a hand-edited config.json ("beam_size": "high",
        "vad_threshold": null, numbers as strings, ints in string lists)
        degrades to defaults instead of crashing the app at startup.
        """
        default_map = {
            f.name: (f.default if f.default is not dataclasses.MISSING else None)
            for f in dataclasses.fields(Config)
        }

        def as_int(name: str, lo: int, hi: int) -> int:
            try:
                value = int(getattr(self, name))
            except (TypeError, ValueError):
                value = default_map[name]
            return min(max(value, lo), hi)

        def as_float(name: str, lo: float, hi: float) -> float:
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError):
                value = default_map[name]
            return min(max(value, lo), hi)

        def as_str(name: str) -> str:
            value = getattr(self, name)
            return value if isinstance(value, str) else str(default_map[name] or "")

        for field_name in (
            "model_size", "language", "ptt_hotkey", "toggle_hotkey",
            "input_device", "models_dir", "initial_prompt",
            "ollama_url", "ollama_model", "ollama_prompt",
        ):
            setattr(self, field_name, as_str(field_name))

        if self.device not in DEVICE_CHOICES:
            self.device = "auto"
        if self.compute_type not in COMPUTE_CHOICES:
            self.compute_type = "auto"
        if self.injection_method not in INJECTION_CHOICES:
            self.injection_method = "clipboard"

        self.beam_size = as_int("beam_size", 1, 10)
        self.cpu_threads = as_int("cpu_threads", 0, 64)
        self.vad_threshold = as_float("vad_threshold", 0.05, 0.95)
        self.vad_min_silence_ms = as_int("vad_min_silence_ms", 50, 5000)
        self.vad_speech_pad_ms = as_int("vad_speech_pad_ms", 0, 2000)
        self.max_record_seconds = as_int("max_record_seconds", 5, 3600)
        self.clipboard_restore_delay_ms = as_int("clipboard_restore_delay_ms", 50, 5000)
        self.type_char_delay_ms = as_int("type_char_delay_ms", 0, 100)
        self.history_size = as_int("history_size", 1, 100)
        self.sound_volume = as_int("sound_volume", 5, 100)
        self.ollama_timeout_s = as_int("ollama_timeout_s", 1, 300)

        # Element types matter too: a stray int inside custom_fillers would
        # crash ", ".join(...) in the settings dialog on every open.
        if isinstance(self.vocab_replacements, dict):
            self.vocab_replacements = {
                str(k): str(v)
                for k, v in self.vocab_replacements.items()
                if k is not None and v is not None
            }
        else:
            self.vocab_replacements = {}
        if isinstance(self.vocab_terms, list):
            self.vocab_terms = [str(x) for x in self.vocab_terms if x is not None]
        else:
            self.vocab_terms = []
        if isinstance(self.custom_fillers, list):
            self.custom_fillers = [str(x) for x in self.custom_fillers if x is not None]
        else:
            self.custom_fillers = []
