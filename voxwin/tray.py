"""System tray icon, menu and notifications (PySide6).

Icons are drawn programmatically (colored circle + mic glyph), so the app
ships with zero image assets.
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from . import APP_NAME, __version__

STATE_COLORS = {
    "idle": "#6f7480",
    "recording": "#e5484d",
    "processing": "#f5a524",
    "error": "#9541e8",
}

STATE_LABELS = {
    "idle": "Готов к диктовке",
    "recording": "Идёт запись…",
    "processing": "Распознаю…",
    "error": "Ошибка",
}


def make_state_icon(state: str) -> QIcon:
    color = QColor(STATE_COLORS.get(state, STATE_COLORS["idle"]))
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setBrush(QBrush(color))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(2, 2, 60, 60)

    # Mic glyph: capsule + stand, white.
    white = QColor("#ffffff")
    painter.setBrush(QBrush(white))
    painter.drawRoundedRect(QRectF(25, 12, 14, 26), 7, 7)
    pen = QPen(white, 5)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawArc(QRectF(18, 22, 28, 24), 180 * 16, 180 * 16)
    painter.drawLine(32, 46, 32, 52)
    painter.end()
    return QIcon(pixmap)


class TrayController(QObject):
    toggleRequested = Signal()
    cancelRequested = Signal()
    settingsRequested = Signal()
    quitRequested = Signal()
    copyRequested = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._icons = {state: make_state_icon(state) for state in STATE_COLORS}
        self._tray = QSystemTrayIcon(self._icons["idle"])
        self._tray.setToolTip(f"{APP_NAME} — локальная диктовка")

        self._menu = QMenu()
        self._status_action = QAction(STATE_LABELS["idle"])
        self._status_action.setEnabled(False)
        self._menu.addAction(self._status_action)
        self._menu.addSeparator()

        self._toggle_action = QAction("Начать запись")
        self._toggle_action.triggered.connect(self.toggleRequested)
        self._menu.addAction(self._toggle_action)

        self._cancel_action = QAction("Отменить запись")
        self._cancel_action.triggered.connect(self.cancelRequested)
        self._cancel_action.setVisible(False)
        self._menu.addAction(self._cancel_action)

        self._menu.addSeparator()
        self._history_menu = QMenu("История (копировать)")
        self._history_menu.setEnabled(False)
        self._menu.addMenu(self._history_menu)

        settings_action = QAction("Настройки…")
        settings_action.triggered.connect(self.settingsRequested)
        self._menu.addAction(settings_action)

        self._menu.addSeparator()
        quit_action = QAction(f"Выход ({APP_NAME} {__version__})")
        quit_action.triggered.connect(self.quitRequested)
        self._menu.addAction(quit_action)

        # Keep Python references alive: QMenu does not own QActions here.
        self._actions = [
            self._status_action, self._toggle_action, self._cancel_action,
            settings_action, quit_action,
        ]
        self._history_actions: List[QAction] = []

        self._tray.setContextMenu(self._menu)
        self._tray.activated.connect(self._on_activated)
        self._tray.show()

    # ------------------------------------------------------------------

    def _on_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:  # left click
            self.toggleRequested.emit()

    def set_state(self, state: str, detail: str = "") -> None:
        icon = self._icons.get(state, self._icons["idle"])
        self._tray.setIcon(icon)
        label = STATE_LABELS.get(state, state)
        if detail:
            label = f"{label} — {detail}"
        self._status_action.setText(label)
        self._tray.setToolTip(f"{APP_NAME}: {label}")
        self._toggle_action.setText(
            "Остановить и распознать" if state == "recording" else "Начать запись"
        )
        self._toggle_action.setEnabled(state in ("idle", "recording"))
        self._cancel_action.setVisible(state == "recording")

    def set_history(self, entries: List[str]) -> None:
        self._history_menu.clear()
        self._history_actions = []
        for entry in entries:
            preview = entry if len(entry) <= 60 else entry[:57] + "…"
            # "&" is a mnemonic marker in Qt menus — escape it or
            # "R&D" renders as "R_D" with the ampersand swallowed.
            action = QAction(preview.replace("&", "&&"))
            action.triggered.connect(
                lambda checked=False, text=entry: self.copyRequested.emit(text)
            )
            self._history_menu.addAction(action)
            self._history_actions.append(action)
        self._history_menu.setEnabled(bool(entries))

    def notify(self, title: str, message: str, error: bool = False) -> None:
        icon = (
            QSystemTrayIcon.MessageIcon.Critical
            if error
            else QSystemTrayIcon.MessageIcon.Information
        )
        self._tray.showMessage(title, message, icon, 4000)

    def hide(self) -> None:
        self._tray.hide()
