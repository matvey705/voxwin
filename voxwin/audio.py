"""Low-latency microphone capture via sounddevice (PortAudio).

Records mono float32 at 16 kHz (Whisper's native rate). If the device
refuses 16 kHz, records at the device default and resamples on stop.
Handles device selection by name substring and survives hot-unplug:
the stream just goes inactive, which the app-level watchdog detects.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

TARGET_RATE = 16000


class AudioError(Exception):
    pass


# PortAudio snapshots the device list at Pa_Initialize and never rescans, so
# hot-plugged microphones are invisible until we re-initialize. Re-init is
# only safe while no stream is open — track open streams here.
_active_streams = 0


def refresh_device_list() -> bool:
    """Force PortAudio to rescan devices. No-op (False) while recording."""
    if _active_streams > 0:
        return False
    try:
        sd._terminate()
        sd._initialize()
        return True
    except Exception:
        log.exception("PortAudio re-initialization failed")
        return False


def list_input_devices(refresh: bool = False) -> List[dict]:
    """All input-capable devices with their host API, default first."""
    if refresh:
        refresh_device_list()
    devices = []
    try:
        default_index = sd.default.device[0]
    except Exception:
        default_index = -1
    hostapis = sd.query_hostapis()
    for index, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": dev["name"],
                "hostapi": hostapis[dev["hostapi"]]["name"],
                "default_samplerate": dev.get("default_samplerate", 44100.0),
                "is_default": index == default_index,
            }
        )
    devices.sort(key=lambda d: (not d["is_default"], d["index"]))
    return devices


def resolve_device(name_substring: str) -> Optional[int]:
    """Map a saved device name substring to a current device index.

    Returns None (system default) when the substring is empty or the device
    is gone — a vanished mic must not brick dictation. If the name is not in
    the (possibly stale) device list, rescan once: the mic may have been
    plugged in after the app started.
    """
    needle = (name_substring or "").strip().lower()
    if not needle:
        return None
    for refresh in (False, True):
        for dev in list_input_devices(refresh=refresh):
            if needle in dev["name"].lower():
                return dev["index"]
    log.warning("Input device %r not found, falling back to default", name_substring)
    return None


class Recorder:
    """Push-to-talk style recorder: start() ... stop() -> np.float32 mono 16k."""

    def __init__(self) -> None:
        self._stream: Optional[sd.InputStream] = None
        self._chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._level = 0.0
        self._rate = TARGET_RATE
        self._aborted = False

    # -- properties -----------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    @property
    def is_active(self) -> bool:
        """False when the stream died underneath us (device unplugged)."""
        stream = self._stream
        if stream is None:
            return False
        try:
            return bool(stream.active) and not self._aborted
        except Exception:
            return False

    @property
    def level(self) -> float:
        """Smoothed input level 0..1 for the overlay meter."""
        return self._level

    @property
    def recorded_seconds(self) -> float:
        with self._lock:
            frames = sum(len(c) for c in self._chunks)
        return frames / float(self._rate)

    # -- lifecycle --------------------------------------------------------------

    def start(self, device_name: str = "") -> None:
        if self._stream is not None:
            return
        device = resolve_device(device_name)
        with self._lock:
            self._chunks = []
        self._level = 0.0
        self._aborted = False

        def callback(indata, frames, time_info, status) -> None:
            if status and (status.input_overflow or status.input_underflow):
                log.debug("Audio callback status: %s", status)
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            with self._lock:
                self._chunks.append(mono)
            rms = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
            # Perceptual-ish scaling; typical speech RMS is ~0.02..0.2.
            self._level = min(1.0, rms * 12.0)

        def finished() -> None:
            self._aborted = True

        global _active_streams
        last_error: Optional[Exception] = None
        for rate in self._candidate_rates(device):
            try:
                stream = sd.InputStream(
                    device=device,
                    channels=1,
                    samplerate=rate,
                    dtype="float32",
                    blocksize=0,
                    callback=callback,
                    finished_callback=finished,
                )
            except Exception as exc:  # PortAudioError and friends
                last_error = exc
                log.debug("Failed to open stream at %s Hz: %s", rate, exc)
                continue
            try:
                stream.start()
            except Exception as exc:
                # The stream is already open (Pa_OpenStream succeeded) — it
                # MUST be closed here, otherwise the device stays claimed and
                # the handle leaks for the rest of the process lifetime.
                last_error = exc
                log.debug("Failed to start stream at %s Hz: %s", rate, exc)
                try:
                    stream.close()
                except Exception:
                    pass
                continue
            self._stream = stream
            self._rate = int(rate)
            self._aborted = False
            _active_streams += 1
            log.info("Recording started (device=%s, rate=%d)", device, self._rate)
            return

        raise AudioError(
            f"Не удалось открыть микрофон (устройство: "
            f"{device_name or 'по умолчанию'}): {last_error}"
        )

    @staticmethod
    def _candidate_rates(device: Optional[int]) -> List[float]:
        rates: List[float] = [float(TARGET_RATE)]
        try:
            info = sd.query_devices(device if device is not None else sd.default.device[0])
            default_rate = float(info.get("default_samplerate", 44100.0))
            if default_rate and default_rate != TARGET_RATE:
                rates.append(default_rate)
        except Exception:
            rates.append(44100.0)
        if 48000.0 not in rates:
            rates.append(48000.0)
        return rates

    def snapshot(self) -> np.ndarray:
        """Copy of everything recorded so far (float32 mono 16 kHz) WITHOUT
        stopping the stream — feeds the live transcript preview."""
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks)
        if self._rate != TARGET_RATE:
            audio = _resample(audio, self._rate, TARGET_RATE)
        return audio

    def stop(self) -> np.ndarray:
        """Stop recording and return audio as float32 mono at 16 kHz."""
        global _active_streams
        stream = self._stream
        self._stream = None
        if stream is not None:
            _active_streams = max(0, _active_streams - 1)
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.exception("Error closing audio stream")
        with self._lock:
            chunks, self._chunks = self._chunks, []
        self._level = 0.0
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        if self._rate != TARGET_RATE:
            audio = _resample(audio, self._rate, TARGET_RATE)
        return audio

    def cancel(self) -> None:
        global _active_streams
        stream = self._stream
        self._stream = None
        if stream is not None:
            _active_streams = max(0, _active_streams - 1)
            try:
                stream.abort()
                stream.close()
            except Exception:
                pass
        with self._lock:
            self._chunks = []
        self._level = 0.0


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resampling — adequate for 44.1/48k -> 16k speech."""
    if src_rate == dst_rate or len(audio) == 0:
        return audio
    duration = len(audio) / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    dst_times = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_times, src_times, audio).astype(np.float32)
