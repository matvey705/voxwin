"""Application orchestrator: wires hotkeys, recorder, transcriber,
post-processing, injection, tray and overlay together.

Threading model:
  * Qt main thread — all UI + state machine (this class's slots);
  * keyboard hook thread — only emits signals (bridged here);
  * transcriber worker thread — inference, post-processing, refinement and
    text injection (so the GUI never blocks), then signals back;
  * PortAudio callback thread — audio buffering inside Recorder.
"""

from __future__ import annotations

import logging
from collections import deque
from itertools import islice
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

from . import APP_NAME, winutil
from .audio import AudioError, Recorder
from .config import Config, CONFIG_PATH
from .hotkeys import HotkeyManager
from .injector import InjectionError, copy_to_clipboard, inject_text
from .overlay import Overlay
from .postprocess import PostProcessor
from .refine import RefinementError, refine_text
from .transcriber import Transcriber, TranscriptionResult
from .tray import TrayController

log = logging.getLogger(__name__)

MIN_UTTERANCE_SECONDS = 0.3


class VoxApp(QObject):
    # Bridged from the keyboard hook thread:
    sigPttDown = Signal()
    sigPttUp = Signal()
    sigToggle = Signal()
    # Bridged from the transcriber worker thread:
    sigDone = Signal(str, object)      # final text, TranscriptionResult
    sigJobError = Signal(str)
    sigWarn = Signal(str)

    def __init__(self, qt_app: QApplication, open_settings: bool = False):
        super().__init__()
        self._qt_app = qt_app
        self._first_run = not CONFIG_PATH.exists()
        self.cfg = Config.load()

        self.recorder = Recorder()
        self.transcriber = Transcriber(self.cfg)
        self.transcriber.start()

        self.tray = TrayController()
        self.overlay = Overlay()
        self._history: deque[str] = deque(maxlen=self.cfg.history_size)
        self._recording_source: Optional[str] = None
        self._processing_jobs = 0
        self._settings_dialog = None

        # --- timers -------------------------------------------------------
        self._level_timer = QTimer(self)
        self._level_timer.setInterval(50)
        self._level_timer.timeout.connect(self._on_level_tick)

        self._max_timer = QTimer(self)
        self._max_timer.setSingleShot(True)
        self._max_timer.timeout.connect(self._on_max_duration)

        # --- signal wiring ---------------------------------------------------
        self.sigPttDown.connect(self._on_ptt_down)
        self.sigPttUp.connect(self._on_ptt_up)
        self.sigToggle.connect(self._on_toggle)
        self.sigDone.connect(self._on_done)
        self.sigJobError.connect(self._on_job_error)
        self.sigWarn.connect(lambda msg: self.tray.notify(APP_NAME, msg))

        self.tray.toggleRequested.connect(self._on_toggle)
        self.tray.cancelRequested.connect(self._cancel_recording)
        self.tray.settingsRequested.connect(self.open_settings)
        self.tray.quitRequested.connect(self.quit)
        self.tray.copyRequested.connect(self._copy_history_entry)

        # --- hotkeys ---------------------------------------------------------
        self.hotkeys = HotkeyManager(
            on_ptt_down=self.sigPttDown.emit,
            on_ptt_up=self.sigPttUp.emit,
            on_toggle=self.sigToggle.emit,
        )
        self.hotkeys.apply(self.cfg)

        if self.cfg.preload_model:
            self.transcriber.request_preload()

        self._update_ui_state()

        if self._first_run:
            self.cfg.save()
            self.tray.notify(
                APP_NAME,
                f"Готов к работе. Удерживайте {self.cfg.ptt_hotkey.upper()} и говорите, "
                f"или {self.cfg.toggle_hotkey.upper()} — старт/стоп.",
            )
        if open_settings:
            QTimer.singleShot(200, self.open_settings)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _update_ui_state(self) -> None:
        if self._recording_source:
            self.tray.set_state("recording")
        elif self._processing_jobs > 0:
            self.tray.set_state("processing")
        else:
            self.tray.set_state("idle", f"модель {self.cfg.model_size}")

    def _start_recording(self, source: str) -> None:
        if self._recording_source:
            return
        try:
            self.recorder.start(self.cfg.input_device)
        except AudioError as exc:
            log.error("Recorder start failed: %s", exc)
            if self.cfg.sound_feedback:
                winutil.play_sound("error")
            self.tray.notify(APP_NAME, str(exc), error=True)
            return
        self._recording_source = source
        if self.cfg.sound_feedback:
            winutil.play_sound("start")
        if self.cfg.overlay_enabled:
            self.overlay.show_state("recording")
        self._level_timer.start()
        self._max_timer.start(self.cfg.max_record_seconds * 1000)
        self._update_ui_state()

    def _stop_and_transcribe(self) -> None:
        if not self._recording_source:
            return
        self._recording_source = None
        self._level_timer.stop()
        self._max_timer.stop()
        audio = self.recorder.stop()
        if self.cfg.sound_feedback:
            winutil.play_sound("stop")

        if len(audio) < MIN_UTTERANCE_SECONDS * 16000:
            if self._processing_jobs > 0 and self.cfg.overlay_enabled:
                self.overlay.show_state("processing")
            else:
                self.overlay.hide_overlay()
            self._update_ui_state()
            return

        self._processing_jobs += 1
        if self.cfg.overlay_enabled:
            self.overlay.show_state("processing")
        self._update_ui_state()
        self.transcriber.submit(audio, self._on_transcribed)

    def _cancel_recording(self) -> None:
        if not self._recording_source:
            return
        self._recording_source = None
        self._level_timer.stop()
        self._max_timer.stop()
        self.recorder.cancel()
        self.overlay.hide_overlay()
        self._update_ui_state()

    # ------------------------------------------------------------------
    # Hotkey handlers (already on the Qt thread via signals)
    # ------------------------------------------------------------------

    def _on_ptt_down(self) -> None:
        if self._recording_source == "toggle":
            return
        self._start_recording("ptt")

    def _on_ptt_up(self) -> None:
        if self._recording_source == "ptt":
            self._stop_and_transcribe()

    def _on_toggle(self) -> None:
        if self._recording_source:
            self._stop_and_transcribe()
        else:
            self._start_recording("toggle")

    def _on_max_duration(self) -> None:
        if self._recording_source:
            self.tray.notify(
                APP_NAME,
                f"Достигнут лимит записи ({self.cfg.max_record_seconds} с) — "
                f"распознаю то, что записано.",
            )
            self._stop_and_transcribe()

    def _on_level_tick(self) -> None:
        if not self._recording_source:
            return
        self.overlay.set_level(self.recorder.level)
        # Watchdog: microphone unplugged / stream died.
        if not self.recorder.is_active:
            log.warning("Audio stream died mid-recording (device unplugged?)")
            self._cancel_recording()
            if self.cfg.sound_feedback:
                winutil.play_sound("error")
            self.tray.notify(
                APP_NAME, "Микрофон отключился — запись остановлена.", error=True
            )

    # ------------------------------------------------------------------
    # Transcription pipeline (worker thread!) → signals back to Qt thread
    # ------------------------------------------------------------------

    def _on_transcribed(
        self, result: Optional[TranscriptionResult], error: Optional[str]
    ) -> None:
        """Runs in the transcriber worker thread."""
        if error is not None or result is None:
            self.sigJobError.emit(error or "Неизвестная ошибка распознавания")
            return

        text = PostProcessor(self.cfg).process(result.text)

        if text and self.cfg.ollama_enabled:
            try:
                text = PostProcessor(self.cfg).process(refine_text(text, self.cfg))
            except RefinementError as exc:
                log.warning("Refinement skipped: %s", exc)
                self.sigWarn.emit(f"LLM-обработка пропущена: {exc}")

        if text:
            try:
                inject_text(text, self.cfg)
            except InjectionError as exc:
                # Fall back: leave the text in the clipboard so nothing is lost.
                try:
                    copy_to_clipboard(text)
                    self.sigJobError.emit(
                        f"{exc}\nТекст скопирован в буфер обмена — вставьте Ctrl+V."
                    )
                except Exception:
                    self.sigJobError.emit(str(exc))
                return
            except Exception as exc:
                log.exception("Injection failed")
                self.sigJobError.emit(f"Ошибка вставки текста: {exc}")
                return

        self.sigDone.emit(text, result)

    # ------------------------------------------------------------------

    def _overlay_after_job(self, state: str) -> None:
        """Show the job outcome without clobbering an already-active session:
        if the user is recording again (or more jobs are pending), the
        overlay must keep showing THAT, not a stale 'done' that auto-hides
        mid-recording."""
        if not self.cfg.overlay_enabled:
            return
        if self._recording_source:
            return  # recording overlay (with level meter) stays as-is
        if self._processing_jobs > 0:
            self.overlay.show_state("processing")
            return
        self.overlay.show_state(state)

    def _on_done(self, text: str, result: TranscriptionResult) -> None:
        self._processing_jobs = max(0, self._processing_jobs - 1)
        if text:
            self._history.appendleft(text)
            self.tray.set_history(list(self._history))
            self._overlay_after_job("done")
            if self.cfg.sound_feedback:
                winutil.play_sound("done")
            if self.cfg.notify_on_success:
                preview = text if len(text) <= 100 else text[:97] + "…"
                self.tray.notify(
                    APP_NAME,
                    f"{preview}\n({result.audio_seconds:.1f} с аудио за "
                    f"{result.elapsed_seconds:.1f} с)",
                )
        else:
            self._overlay_after_job("error")
            self.tray.notify(APP_NAME, "Речь не распознана (тишина?)")
        self._update_ui_state()

    def _on_job_error(self, message: str) -> None:
        self._processing_jobs = max(0, self._processing_jobs - 1)
        self._overlay_after_job("error")
        if self.cfg.sound_feedback:
            winutil.play_sound("error")
        self.tray.notify(APP_NAME, message, error=True)
        self._update_ui_state()

    def _copy_history_entry(self, text: str) -> None:
        try:
            copy_to_clipboard(text)
            self.tray.notify(APP_NAME, "Скопировано в буфер обмена.")
        except Exception as exc:
            self.tray.notify(APP_NAME, f"Не удалось скопировать: {exc}", error=True)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def open_settings(self) -> None:
        if self._settings_dialog is not None:
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return
        from .settings_ui import SettingsDialog

        dialog = SettingsDialog(
            self.cfg,
            pause_hotkeys=self.hotkeys.pause,
            resume_hotkeys=lambda: self.hotkeys.apply(self.cfg),
        )
        self._settings_dialog = dialog
        try:
            if dialog.exec():
                new_cfg, autostart = dialog.collect()
                self.apply_config(new_cfg, autostart)
        finally:
            self._settings_dialog = None

    def apply_config(self, new_cfg: Config, autostart: Optional[bool] = None) -> None:
        model_changed = new_cfg.model_key() != self.cfg.model_key()
        self.cfg = new_cfg
        self.cfg.save()

        self.hotkeys.apply(self.cfg)
        self.transcriber.apply_config(self.cfg)
        if model_changed and self.cfg.preload_model:
            self.transcriber.request_preload()
        if not self.cfg.overlay_enabled:
            self.overlay.hide_overlay()

        # Keep the NEWEST entries when shrinking: entries are stored
        # newest-first (appendleft), and deque(iter, maxlen) would keep the
        # tail of iteration — i.e. the oldest ones.
        self._history = deque(
            islice(self._history, 0, self.cfg.history_size),
            maxlen=self.cfg.history_size,
        )
        self.tray.set_history(list(self._history))

        if autostart is not None and autostart != winutil.is_autostart_enabled():
            try:
                winutil.set_autostart(autostart)
            except Exception as exc:
                self.tray.notify(
                    APP_NAME, f"Не удалось изменить автозапуск: {exc}", error=True
                )
        self._update_ui_state()
        log.info("Config applied (model_changed=%s)", model_changed)

    # ------------------------------------------------------------------

    def quit(self) -> None:
        log.info("Shutting down")
        try:
            self.hotkeys.unbind_all()
            self.hotkeys.shutdown()
        except Exception:
            pass
        try:
            self.recorder.cancel()
        except Exception:
            pass
        try:
            self.transcriber.shutdown(timeout=2.0)
        except Exception:
            pass
        self.overlay.hide_overlay()
        self.tray.hide()
        self._qt_app.quit()
