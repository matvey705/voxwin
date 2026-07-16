"""Whisper inference worker.

Runs faster-whisper (CTranslate2) in a dedicated background thread with a
job queue. Handles model download/caching, device auto-selection with a
CUDA -> CPU fallback, and Silero-VAD filtering of silence.
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import winutil
from .config import Config

log = logging.getLogger(__name__)

MIN_AUDIO_SECONDS = 0.25

ResultCallback = Callable[[Optional["TranscriptionResult"], Optional[str]], None]


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    audio_seconds: float
    elapsed_seconds: float
    model: str


def cuda_device_count() -> int:
    try:
        winutil.add_cuda_dll_dirs()
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


class Transcriber:
    """Owns the model and a worker thread; submit() is thread-safe."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._cfg_lock = threading.Lock()
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._model = None
        self._model_key: Optional[tuple] = None
        self._forced_cpu = False
        self.busy = False

    # -- public API (any thread) ------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._worker, name="TranscriberWorker", daemon=True
        )
        self._thread.start()

    def request_preload(self) -> None:
        self._queue.put(("load", None, None))

    def submit(self, audio: np.ndarray, callback: ResultCallback) -> None:
        self._queue.put(("job", audio, callback))

    def apply_config(self, cfg: Config) -> None:
        with self._cfg_lock:
            old_key = self._cfg.model_key()
            self._cfg = cfg
            if cfg.model_key() != old_key:
                self._forced_cpu = False  # user may have fixed CUDA setup

    def shutdown(self, timeout: float = 5.0) -> None:
        self._queue.put(("quit", None, None))
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # -- worker thread ------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            kind, audio, callback = self._queue.get()
            if kind == "quit":
                return
            with self._cfg_lock:
                cfg = self._cfg
            if kind == "load":
                try:
                    self._ensure_model(cfg)
                except Exception:
                    log.exception("Model preload failed")
                continue
            # kind == "job"
            self.busy = True
            try:
                result = self._transcribe(cfg, audio)
                callback(result, None)
            except Exception as exc:
                log.exception("Transcription failed")
                callback(None, f"{type(exc).__name__}: {exc}")
            finally:
                self.busy = False

    # -- model management -----------------------------------------------------------

    def _resolve_device(self, cfg: Config) -> tuple:
        device = cfg.device
        if self._forced_cpu and device in ("auto", "cuda"):
            device = "cpu"
        if device == "auto":
            device = "cuda" if cuda_device_count() > 0 else "cpu"
        compute = cfg.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        elif device == "cpu" and compute in ("float16", "int8_float16"):
            # GPU-only compute types crash on CPU — matters when the user
            # pinned cuda+float16 and we are falling back to CPU.
            compute = "int8"
        return device, compute

    def _ensure_model(self, cfg: Config):
        key = cfg.model_key()
        if self._model is not None and self._model_key == key:
            return self._model

        # Release the previous model BEFORE loading the new one, otherwise
        # both live in RAM/VRAM simultaneously and a model switch can OOM on
        # the GPU (which the fallback would then misread as a broken CUDA
        # setup and permanently degrade to CPU).
        if self._model is not None:
            self._model = None
            self._model_key = None
            gc.collect()

        winutil.add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        device, compute = self._resolve_device(cfg)
        models_dir = str(cfg.resolved_models_dir())
        # Download (or find in cache) first, separately from engine init, so
        # network problems are never mistaken for CUDA problems.
        model_path = self._prepare_model_path(cfg, models_dir)
        started = time.monotonic()
        log.info(
            "Loading model %s (device=%s, compute=%s, path=%s)",
            cfg.model_size, device, compute, model_path,
        )
        try:
            model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute,
                cpu_threads=cfg.cpu_threads,
            )
        except Exception as exc:
            if device == "cuda":
                log.warning("CUDA init failed (%s); falling back to CPU int8", exc)
                self._forced_cpu = True
                model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=cfg.cpu_threads,
                )
            else:
                raise
        log.info("Model loaded in %.1fs", time.monotonic() - started)
        self._model = model
        self._model_key = key
        return model

    @staticmethod
    def _prepare_model_path(cfg: Config, models_dir: str) -> str:
        """Resolve the model to a local directory, downloading if needed.

        A broken system proxy (dead SOCKS from the registry) is a routine
        Windows condition — on a proxy-shaped failure we retry once with
        proxies disabled for the process.
        """
        import os

        size = cfg.model_size
        if os.path.isdir(size):
            return size  # user pointed model_size at a local CTranslate2 dir

        from faster_whisper import download_model

        # Flat per-model folder (no HF cache symlinks: those need admin or
        # Developer Mode on Windows and fail with WinError 1314 otherwise).
        target = os.path.join(models_dir, size.replace("/", "--").replace(":", "_"))
        if os.path.isfile(os.path.join(target, "model.bin")):
            return target  # already downloaded — skip network entirely

        try:
            return download_model(size, output_dir=target)
        except Exception as exc:
            # hf-hub often wraps the real network error (dead proxy -> bare
            # ConnectError inside LocalEntryNotFoundError), so we retry not
            # only on proxy-shaped messages but on ANY failure while system
            # proxies are configured — a proxy-less retry is harmless when
            # the direct connection works.
            proxies_configured = {}
            try:
                import urllib.request

                proxies_configured = urllib.request.getproxies()
            except Exception:
                pass
            if not (winutil.looks_like_proxy_error(exc) or proxies_configured):
                raise
            log.warning(
                "Model download failed (%s); retrying with proxies disabled", exc
            )
            winutil.strip_proxies()
            # hf-hub caches one global httpx client whose proxy mounts were
            # resolved at construction time — drop it so the retry builds a
            # fresh client that sees the cleaned environment.
            try:
                import huggingface_hub

                huggingface_hub.close_session()
            except Exception:
                pass
            return download_model(size, output_dir=target)

    # -- inference ----------------------------------------------------------------------

    def _build_initial_prompt(self, cfg: Config) -> Optional[str]:
        parts = []
        if cfg.initial_prompt.strip():
            parts.append(cfg.initial_prompt.strip())
        terms = [t.strip() for t in cfg.vocab_terms if t.strip()]
        if terms:
            parts.append(", ".join(terms) + ".")
        return " ".join(parts) if parts else None

    def _transcribe(self, cfg: Config, audio: np.ndarray) -> TranscriptionResult:
        audio_seconds = len(audio) / 16000.0
        if audio_seconds < MIN_AUDIO_SECONDS:
            return TranscriptionResult(
                text="", language="", language_probability=0.0,
                audio_seconds=audio_seconds, elapsed_seconds=0.0,
                model=cfg.model_size,
            )

        model = self._ensure_model(cfg)
        started = time.monotonic()

        kwargs = dict(
            language=cfg.effective_language(),
            beam_size=cfg.beam_size,
            initial_prompt=self._build_initial_prompt(cfg),
            condition_on_previous_text=False,
            vad_filter=cfg.vad_enabled,
        )
        if cfg.vad_enabled:
            kwargs["vad_parameters"] = dict(
                threshold=cfg.vad_threshold,
                min_silence_duration_ms=cfg.vad_min_silence_ms,
                speech_pad_ms=cfg.vad_speech_pad_ms,
            )

        try:
            segments, info = model.transcribe(audio, **kwargs)
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            # A model that loaded on CUDA can still fail at inference time
            # (missing cuDNN, out of VRAM). Retry once on CPU.
            message = f"{type(exc).__name__} {exc}".lower()
            device, _ = self._resolve_device(cfg)
            if device == "cuda" and any(
                token in message for token in ("cuda", "cublas", "cudnn", "memory")
            ):
                log.warning("CUDA inference failed (%s); retrying on CPU", exc)
                self._forced_cpu = True
                self._model = None
                self._model_key = None
                model = self._ensure_model(cfg)
                segments, info = model.transcribe(audio, **kwargs)
                text = " ".join(seg.text.strip() for seg in segments).strip()
            else:
                raise

        elapsed = time.monotonic() - started
        log.info(
            "Transcribed %.1fs of audio in %.2fs (lang=%s p=%.2f): %r",
            audio_seconds, elapsed, info.language, info.language_probability,
            text[:120],
        )
        return TranscriptionResult(
            text=text,
            language=info.language or "",
            language_probability=float(info.language_probability or 0.0),
            audio_seconds=audio_seconds,
            elapsed_seconds=elapsed,
            model=cfg.model_size,
        )
