"""Settings dialog (PySide6): model, hotkeys, microphone, insertion,
text post-processing, optional Ollama refinement and system options."""

from __future__ import annotations

import threading
from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import APP_NAME, __copyright__, __version__, winutil
from .audio import list_input_devices
from .config import (
    COMPUTE_CHOICES,
    Config,
    DEVICE_CHOICES,
    LANGUAGES,
    MODEL_CHOICES,
)
from .hotkeys import read_hotkey_blocking, validate_hotkey


class HotkeyCaptureButton(QPushButton):
    """Button that captures the next global key combination when clicked.

    While capturing, the app's live hotkeys are paused (otherwise pressing
    the current PTT key to re-record it would start a real dictation), and
    the capture is cancellable: the worker thread stays blocked in
    keyboard.read_hotkey() until the next key-up anywhere, which may happen
    after the dialog is destroyed — emitting into a dead widget then must
    be a no-op, not a RuntimeError.
    """

    captured = Signal(str)

    def __init__(self, target: QLineEdit, pause=None, resume=None):
        super().__init__("Записать…")
        self._target = target
        self._pause = pause
        self._resume = resume
        self._capture_state: Optional[dict] = None
        self.clicked.connect(self._start_capture)
        self.captured.connect(self._on_captured)

    def _start_capture(self) -> None:
        if self._capture_state is not None:
            return
        state = {"cancelled": False}
        self._capture_state = state
        self.setText("Нажмите клавиши…")
        self.setEnabled(False)
        if self._pause:
            try:
                self._pause()
            except Exception:
                pass

        def worker() -> None:
            combo = read_hotkey_blocking()
            if state["cancelled"]:
                return
            try:
                self.captured.emit(combo or "")
            except RuntimeError:
                pass  # C++ side of the button is already destroyed

        threading.Thread(target=worker, daemon=True, name="HotkeyCapture").start()

    def _on_captured(self, combo: str) -> None:
        self._capture_state = None
        self.setText("Записать…")
        self.setEnabled(True)
        if self._resume:
            try:
                self._resume()
            except Exception:
                pass
        if combo:
            self._target.setText(combo)

    def cancel_capture(self) -> None:
        """Called when the dialog closes with a capture still pending."""
        if self._capture_state is None:
            return
        self._capture_state["cancelled"] = True
        self._capture_state = None
        if self._resume:
            try:
                self._resume()
            except Exception:
                pass


class SettingsDialog(QDialog):
    def __init__(
        self,
        cfg: Config,
        parent: Optional[QWidget] = None,
        pause_hotkeys=None,
        resume_hotkeys=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} {__version__} — настройки")
        self.setMinimumWidth(560)
        self._cfg = cfg
        self._pause_hotkeys = pause_hotkeys
        self._resume_hotkeys = resume_hotkeys
        self._capture_buttons: list[HotkeyCaptureButton] = []

        tabs = QTabWidget()
        tabs.addTab(self._build_engine_tab(cfg), "Распознавание")
        tabs.addTab(self._build_hotkeys_tab(cfg), "Клавиши")
        tabs.addTab(self._build_audio_tab(cfg), "Микрофон")
        tabs.addTab(self._build_insert_tab(cfg), "Вставка")
        tabs.addTab(self._build_text_tab(cfg), "Текст")
        tabs.addTab(self._build_ollama_tab(cfg), "LLM (Ollama)")
        tabs.addTab(self._build_system_tab(cfg), "Система")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(tabs)
        root.addWidget(buttons)

    # ------------------------------------------------------------------

    def _hotkey_row(self, edit: QLineEdit) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        capture = HotkeyCaptureButton(
            edit, pause=self._pause_hotkeys, resume=self._resume_hotkeys
        )
        self._capture_buttons.append(capture)
        layout.addWidget(capture)
        clear = QPushButton("×")
        clear.setFixedWidth(28)
        clear.setToolTip("Отключить эту горячую клавишу")
        clear.clicked.connect(lambda: edit.setText(""))
        layout.addWidget(clear)
        return row

    def done(self, result: int) -> None:  # noqa: N802 (Qt naming)
        for button in self._capture_buttons:
            button.cancel_capture()
        super().done(result)

    def _build_engine_tab(self, cfg: Config) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(MODEL_CHOICES)
        self.model_combo.setCurrentText(cfg.model_size)
        form.addRow("Модель Whisper:", self.model_combo)
        form.addRow(
            "", QLabel(
                "tiny/base — мгновенно, small — баланс (рекомендуется),\n"
                "medium/large-v3 — максимум точности. Скачивается при первом запуске."
            )
        )

        self.device_combo = QComboBox()
        for dev in DEVICE_CHOICES:
            self.device_combo.addItem(
                {"auto": "Авто (CUDA, если доступна)", "cpu": "CPU", "cuda": "GPU (CUDA)"}[dev],
                dev,
            )
        self.device_combo.setCurrentIndex(DEVICE_CHOICES.index(cfg.device))
        form.addRow("Устройство:", self.device_combo)

        self.compute_combo = QComboBox()
        for comp in COMPUTE_CHOICES:
            label = {
                "auto": "Авто (float16 на GPU / int8 на CPU)",
                "int8": "int8 (быстро, мало памяти)",
                "int8_float16": "int8_float16 (GPU, экономно)",
                "float16": "float16 (GPU)",
                "float32": "float32 (медленно, точно)",
            }[comp]
            self.compute_combo.addItem(label, comp)
        self.compute_combo.setCurrentIndex(COMPUTE_CHOICES.index(cfg.compute_type))
        form.addRow("Квантизация:", self.compute_combo)

        self.language_combo = QComboBox()
        for code, label in LANGUAGES:
            self.language_combo.addItem(label, code)
        codes = [code for code, _ in LANGUAGES]
        if cfg.language in codes:
            self.language_combo.setCurrentIndex(codes.index(cfg.language))
        else:
            # A Whisper-valid code set by hand (e.g. "nl") must survive a
            # visit to the settings dialog, not silently become "auto".
            self.language_combo.addItem(cfg.language, cfg.language)
            self.language_combo.setCurrentIndex(self.language_combo.count() - 1)
        form.addRow("Язык речи:", self.language_combo)

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(cfg.beam_size)
        self.beam_spin.setToolTip("Меньше — быстрее, больше — точнее. 1–5 разумно.")
        form.addRow("Beam size:", self.beam_spin)

        self.vad_check = QCheckBox("Отсекать тишину (Silero VAD) — ускоряет распознавание")
        self.vad_check.setChecked(cfg.vad_enabled)
        form.addRow(self.vad_check)

        self.vad_threshold = QDoubleSpinBox()
        self.vad_threshold.setRange(0.05, 0.95)
        self.vad_threshold.setSingleStep(0.05)
        self.vad_threshold.setValue(cfg.vad_threshold)
        form.addRow("Чувствительность VAD:", self.vad_threshold)

        self.vad_silence = QSpinBox()
        self.vad_silence.setRange(50, 5000)
        self.vad_silence.setSuffix(" мс")
        self.vad_silence.setValue(cfg.vad_min_silence_ms)
        form.addRow("Мин. пауза тишины:", self.vad_silence)

        return page

    def _build_hotkeys_tab(self, cfg: Config) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.ptt_edit = QLineEdit(cfg.ptt_hotkey)
        form.addRow("Удерживать (PTT):", self._hotkey_row(self.ptt_edit))
        self.ptt_status = QLabel()
        self.ptt_status.setWordWrap(True)
        form.addRow("", self.ptt_status)

        self.toggle_edit = QLineEdit(cfg.toggle_hotkey)
        form.addRow("Старт/стоп:", self._hotkey_row(self.toggle_edit))
        self.toggle_status = QLabel()
        self.toggle_status.setWordWrap(True)
        form.addRow("", self.toggle_status)

        # Живая проверка конфликтов с сочетаниями Windows/приложений.
        self.ptt_edit.textChanged.connect(self._refresh_hotkey_status)
        self.toggle_edit.textChanged.connect(self._refresh_hotkey_status)
        self._refresh_hotkey_status()

        self.suppress_check = QCheckBox(
            "Не передавать горячую клавишу в активное приложение"
        )
        self.suppress_check.setChecked(cfg.suppress_hotkeys)
        form.addRow(self.suppress_check)

        form.addRow(
            "", QLabel(
                "Примеры: f9, ctrl+alt+space, ctrl+shift+f9.\n"
                "Оба режима активны одновременно: PTT — удерживайте и говорите,\n"
                "старт/стоп — нажали, надиктовали, нажали ещё раз.\n"
                "Комбинация перехватывается только целиком: PTT-клавиша без\n"
                "своих модификаторов работает в программах как обычно."
            )
        )
        return page

    def _refresh_hotkey_status(self) -> None:
        colors = {"ok": "#2e8b57", "warn": "#b8860b", "block": "#d9363e"}
        marks = {"ok": "✓", "warn": "⚠", "block": "✕"}
        for edit, label in (
            (self.ptt_edit, self.ptt_status),
            (self.toggle_edit, self.toggle_status),
        ):
            level, message = validate_hotkey(edit.text())
            label.setText(f"{marks[level]} {message}")
            label.setStyleSheet(f"color: {colors[level]};")

    def accept(self) -> None:  # noqa: N802 (Qt naming)
        """Не даём сохранить сочетания, ломающие Windows или ввод текста."""
        problems = []
        ptt = self.ptt_edit.text().strip().lower()
        toggle = self.toggle_edit.text().strip().lower()
        for name, combo in (("Удерживать (PTT)", ptt), ("Старт/стоп", toggle)):
            level, message = validate_hotkey(combo)
            if level == "block":
                problems.append(f"«{name}»: {message}")
        if ptt and ptt == toggle:
            problems.append("PTT и «Старт/стоп» не могут быть одним сочетанием.")
        if problems:
            QMessageBox.warning(
                self,
                "Горячие клавиши",
                "Исправьте горячие клавиши:\n\n" + "\n\n".join(problems),
            )
            return
        super().accept()

    def _build_audio_tab(self, cfg: Config) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.device_input_combo = QComboBox()
        refresh = QPushButton("Обновить")
        refresh.clicked.connect(lambda: self._fill_devices(cfg))
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.device_input_combo, 1)
        layout.addWidget(refresh)
        form.addRow("Микрофон:", row)
        self._fill_devices(cfg)

        self.max_seconds_spin = QSpinBox()
        self.max_seconds_spin.setRange(5, 3600)
        self.max_seconds_spin.setSuffix(" с")
        self.max_seconds_spin.setValue(cfg.max_record_seconds)
        form.addRow("Лимит записи:", self.max_seconds_spin)

        return page

    def _fill_devices(self, cfg: Config) -> None:
        current = (
            self.device_input_combo.currentData()
            if self.device_input_combo.count()
            else cfg.input_device
        )
        self.device_input_combo.clear()
        self.device_input_combo.addItem("Системный по умолчанию", "")
        try:
            devices = list_input_devices()
        except Exception:
            devices = []
        seen = set()
        for dev in devices:
            key = dev["name"]
            if key in seen:
                continue
            seen.add(key)
            label = f"{dev['name']}  [{dev['hostapi']}]"
            if dev["is_default"]:
                label += "  (по умолчанию)"
            self.device_input_combo.addItem(label, dev["name"])
        if current:
            index = self.device_input_combo.findData(current)
            if index < 0:
                self.device_input_combo.addItem(f"{current}  (не подключён)", current)
                index = self.device_input_combo.count() - 1
            self.device_input_combo.setCurrentIndex(index)

    def _build_insert_tab(self, cfg: Config) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.method_combo = QComboBox()
        self.method_combo.addItem("Через буфер обмена (Ctrl+V) — быстро", "clipboard")
        self.method_combo.addItem("Посимвольный ввод — медленнее, универсальнее", "type")
        self.method_combo.setCurrentIndex(0 if cfg.injection_method == "clipboard" else 1)
        form.addRow("Способ вставки:", self.method_combo)

        self.restore_clip_check = QCheckBox("Восстанавливать буфер обмена после вставки")
        self.restore_clip_check.setChecked(cfg.restore_clipboard)
        form.addRow(self.restore_clip_check)

        self.clip_delay_spin = QSpinBox()
        self.clip_delay_spin.setRange(50, 5000)
        self.clip_delay_spin.setSuffix(" мс")
        self.clip_delay_spin.setValue(cfg.clipboard_restore_delay_ms)
        self.clip_delay_spin.setToolTip(
            "Пауза перед восстановлением буфера. Увеличьте, если приложение "
            "вставляет старый текст."
        )
        form.addRow("Задержка восстановления:", self.clip_delay_spin)

        self.trailing_space_check = QCheckBox("Добавлять пробел после вставленного текста")
        self.trailing_space_check.setChecked(cfg.trailing_space)
        form.addRow(self.trailing_space_check)

        self.type_delay_spin = QSpinBox()
        self.type_delay_spin.setRange(0, 100)
        self.type_delay_spin.setSuffix(" мс")
        self.type_delay_spin.setValue(cfg.type_char_delay_ms)
        form.addRow("Пауза между блоками (посимвольный):", self.type_delay_spin)

        return page

    def _build_text_tab(self, cfg: Config) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.post_check = QCheckBox("Включить пост-обработку текста")
        self.post_check.setChecked(cfg.post_enabled)
        layout.addWidget(self.post_check)

        self.capitalize_check = QCheckBox("Заглавная буква в начале фразы")
        self.capitalize_check.setChecked(cfg.capitalize_first)
        layout.addWidget(self.capitalize_check)

        self.fillers_check = QCheckBox("Убирать слова-паразиты (эм, ммм, um, uh…)")
        self.fillers_check.setChecked(cfg.remove_fillers)
        layout.addWidget(self.fillers_check)

        fillers_row = QHBoxLayout()
        fillers_row.addWidget(QLabel("Свои паразиты (через запятую):"))
        self.fillers_edit = QLineEdit(", ".join(cfg.custom_fillers))
        fillers_row.addWidget(self.fillers_edit, 1)
        layout.addLayout(fillers_row)

        layout.addWidget(QLabel("Словарь замен (что услышано → как писать):"))
        self.vocab_table = QTableWidget(0, 2)
        self.vocab_table.setHorizontalHeaderLabels(["Распознано", "Заменять на"])
        self.vocab_table.horizontalHeader().setStretchLastSection(True)
        self.vocab_table.setColumnWidth(0, 220)
        for heard, correct in cfg.vocab_replacements.items():
            self._add_vocab_row(heard, correct)
        layout.addWidget(self.vocab_table)

        buttons_row = QHBoxLayout()
        add_button = QPushButton("+ строка")
        add_button.clicked.connect(lambda: self._add_vocab_row("", ""))
        remove_button = QPushButton("− строка")
        remove_button.clicked.connect(self._remove_vocab_row)
        buttons_row.addWidget(add_button)
        buttons_row.addWidget(remove_button)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        layout.addWidget(QLabel(
            "Термины-подсказки для Whisper (по одному в строке — имена, бренды, жаргон):"
        ))
        self.terms_edit = QPlainTextEdit("\n".join(cfg.vocab_terms))
        self.terms_edit.setMaximumHeight(90)
        layout.addWidget(self.terms_edit)

        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Начальный промпт Whisper:"))
        self.prompt_edit = QLineEdit(cfg.initial_prompt)
        prompt_row.addWidget(self.prompt_edit, 1)
        layout.addLayout(prompt_row)

        return page

    def _add_vocab_row(self, heard: str, correct: str) -> None:
        row = self.vocab_table.rowCount()
        self.vocab_table.insertRow(row)
        self.vocab_table.setItem(row, 0, QTableWidgetItem(heard))
        self.vocab_table.setItem(row, 1, QTableWidgetItem(correct))

    def _remove_vocab_row(self) -> None:
        row = self.vocab_table.currentRow()
        if row >= 0:
            self.vocab_table.removeRow(row)
        elif self.vocab_table.rowCount():
            self.vocab_table.removeRow(self.vocab_table.rowCount() - 1)

    def _build_ollama_tab(self, cfg: Config) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.ollama_check = QCheckBox(
            "Дорабатывать текст локальной LLM (нужен запущенный Ollama)"
        )
        self.ollama_check.setChecked(cfg.ollama_enabled)
        form.addRow(self.ollama_check)

        self.ollama_url_edit = QLineEdit(cfg.ollama_url)
        form.addRow("Адрес Ollama:", self.ollama_url_edit)

        self.ollama_model_edit = QLineEdit(cfg.ollama_model)
        form.addRow("Модель:", self.ollama_model_edit)

        self.ollama_timeout_spin = QSpinBox()
        self.ollama_timeout_spin.setRange(1, 300)
        self.ollama_timeout_spin.setSuffix(" с")
        self.ollama_timeout_spin.setValue(cfg.ollama_timeout_s)
        form.addRow("Таймаут:", self.ollama_timeout_spin)

        form.addRow(QLabel("Промпт ({text} заменяется на распознанный текст):"))
        self.ollama_prompt_edit = QPlainTextEdit(cfg.ollama_prompt)
        self.ollama_prompt_edit.setMaximumHeight(120)
        form.addRow(self.ollama_prompt_edit)

        return page

    def _build_system_tab(self, cfg: Config) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        self.autostart_check = QCheckBox("Запускать VoxWin при входе в Windows")
        try:
            self.autostart_check.setChecked(winutil.is_autostart_enabled())
        except Exception:
            self.autostart_check.setChecked(False)
        form.addRow(self.autostart_check)

        self.sound_check = QCheckBox("Звуковые сигналы записи")
        self.sound_check.setChecked(cfg.sound_feedback)
        form.addRow(self.sound_check)

        volume_row = QWidget()
        volume_layout = QHBoxLayout(volume_row)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        self.volume_spin = QSpinBox()
        self.volume_spin.setRange(5, 100)
        self.volume_spin.setSuffix(" %")
        self.volume_spin.setValue(cfg.sound_volume)
        volume_layout.addWidget(self.volume_spin)
        try_button = QPushButton("Прослушать")
        try_button.clicked.connect(
            lambda: winutil.play_sound("start", self.volume_spin.value())
        )
        volume_layout.addWidget(try_button)
        volume_layout.addStretch(1)
        form.addRow("Громкость сигналов:", volume_row)

        self.overlay_check = QCheckBox("Плавающий индикатор записи (оверлей)")
        self.overlay_check.setChecked(cfg.overlay_enabled)
        form.addRow(self.overlay_check)

        self.live_preview_check = QCheckBox(
            "Показывать текст по мере речи (в оверлее, нужен включённый оверлей)"
        )
        self.live_preview_check.setChecked(cfg.live_preview)
        form.addRow(self.live_preview_check)

        self.notify_check = QCheckBox("Уведомление после каждой успешной вставки")
        self.notify_check.setChecked(cfg.notify_on_success)
        form.addRow(self.notify_check)

        self.preload_check = QCheckBox("Загружать модель при старте (быстрее первая фраза)")
        self.preload_check.setChecked(cfg.preload_model)
        form.addRow(self.preload_check)

        form.addRow(QLabel(
            "Диагностика: запустите в консоли из папки проекта\n"
            "    .venv\\Scripts\\python -m voxwin --diagnose\n"
            "Логи: %APPDATA%\\VoxWin\\logs\\voxwin.log"
        ))

        # GPL-3.0 "Appropriate Legal Notices": программа с интерфейсом должна
        # показывать копирайт, отсутствие гарантий и условия лицензии.
        about = QLabel(
            f"{APP_NAME} {__version__} — {__copyright__}\n"
            "Свободная программа под GNU GPL v3.0: распространяется БЕЗ "
            "КАКИХ-ЛИБО ГАРАНТИЙ.\n"
            "Вы вправе распространять её на условиях GPL — "
            '<a href="https://www.gnu.org/licenses/gpl-3.0.html">текст лицензии</a>. '
            'Исходный код: <a href="https://github.com/matvey705/voxwin">github.com/matvey705/voxwin</a>'
        )
        about.setOpenExternalLinks(True)
        about.setWordWrap(True)
        form.addRow(about)
        return page

    # ------------------------------------------------------------------

    def collect(self) -> Tuple[Config, bool]:
        """Build a new Config from the widgets. Returns (config, autostart)."""
        cfg = Config(**{**self._cfg.__dict__})

        cfg.model_size = self.model_combo.currentText().strip() or "small"
        cfg.device = self.device_combo.currentData()
        cfg.compute_type = self.compute_combo.currentData()
        cfg.language = self.language_combo.currentData()
        cfg.beam_size = self.beam_spin.value()
        cfg.vad_enabled = self.vad_check.isChecked()
        cfg.vad_threshold = self.vad_threshold.value()
        cfg.vad_min_silence_ms = self.vad_silence.value()

        cfg.ptt_hotkey = self.ptt_edit.text().strip()
        cfg.toggle_hotkey = self.toggle_edit.text().strip()
        cfg.suppress_hotkeys = self.suppress_check.isChecked()

        cfg.input_device = self.device_input_combo.currentData() or ""
        cfg.max_record_seconds = self.max_seconds_spin.value()

        cfg.injection_method = self.method_combo.currentData()
        cfg.restore_clipboard = self.restore_clip_check.isChecked()
        cfg.clipboard_restore_delay_ms = self.clip_delay_spin.value()
        cfg.trailing_space = self.trailing_space_check.isChecked()
        cfg.type_char_delay_ms = self.type_delay_spin.value()

        cfg.post_enabled = self.post_check.isChecked()
        cfg.capitalize_first = self.capitalize_check.isChecked()
        cfg.remove_fillers = self.fillers_check.isChecked()
        cfg.custom_fillers = [
            w.strip() for w in self.fillers_edit.text().split(",") if w.strip()
        ]
        replacements = {}
        for row in range(self.vocab_table.rowCount()):
            heard_item = self.vocab_table.item(row, 0)
            correct_item = self.vocab_table.item(row, 1)
            heard = heard_item.text().strip() if heard_item else ""
            correct = correct_item.text().strip() if correct_item else ""
            if heard and correct:
                replacements[heard] = correct
        cfg.vocab_replacements = replacements
        cfg.vocab_terms = [
            line.strip()
            for line in self.terms_edit.toPlainText().splitlines()
            if line.strip()
        ]
        cfg.initial_prompt = self.prompt_edit.text().strip()

        cfg.ollama_enabled = self.ollama_check.isChecked()
        cfg.ollama_url = self.ollama_url_edit.text().strip() or "http://127.0.0.1:11434"
        cfg.ollama_model = self.ollama_model_edit.text().strip() or "llama3.2:3b"
        cfg.ollama_timeout_s = self.ollama_timeout_spin.value()
        cfg.ollama_prompt = self.ollama_prompt_edit.toPlainText().strip()

        cfg.sound_feedback = self.sound_check.isChecked()
        cfg.sound_volume = self.volume_spin.value()
        cfg.overlay_enabled = self.overlay_check.isChecked()
        cfg.live_preview = self.live_preview_check.isChecked()
        cfg.notify_on_success = self.notify_check.isChecked()
        cfg.preload_model = self.preload_check.isChecked()

        cfg.clamp()
        return cfg, self.autostart_check.isChecked()
