"""Floating status overlay: a small always-on-top pill near the bottom of
the screen showing recording state and live microphone level.

Crucially it never steals focus (WA_ShowWithoutActivating +
WindowDoesNotAcceptFocus + transparent for mouse), so dictation lands in
the app the user was typing in.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QWidget

_STATE_TEXT = {
    "recording": "Говорите…",
    "processing": "Распознаю",
    "done": "Готово ✓",
    "error": "Ошибка",
}

_STATE_COLOR = {
    "recording": QColor("#e5484d"),
    "processing": QColor("#f5a524"),
    "done": QColor("#30a46c"),
    "error": QColor("#e5484d"),
}


class Overlay(QWidget):
    WIDTH, HEIGHT = 240, 52

    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.resize(self.WIDTH, self.HEIGHT)

        self._state = "recording"
        self._level = 0.0
        self._dots = 0

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(350)
        self._anim_timer.timeout.connect(self._tick)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide_overlay)

    # ------------------------------------------------------------------

    def _reposition(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + geo.height() - self.height() - 70
        self.move(x, y)

    def show_state(self, state: str) -> None:
        self._state = state
        self._hide_timer.stop()
        self._reposition()
        if state == "processing":
            self._dots = 0
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        if state in ("done", "error"):
            self._hide_timer.start(1100)
        self.show()
        self.update()

    def set_level(self, level: float) -> None:
        self._level = max(0.0, min(1.0, level))
        if self._state == "recording" and self.isVisible():
            self.update()

    def hide_overlay(self) -> None:
        self._anim_timer.stop()
        self._hide_timer.stop()
        self.hide()

    def _tick(self) -> None:
        self._dots = (self._dots + 1) % 4
        self.update()

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QColor(24, 24, 28, 235))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 14, 14)

        color = _STATE_COLOR.get(self._state, QColor("#8a8a8a"))
        painter.setBrush(color)
        painter.drawEllipse(16, self.height() // 2 - 6, 12, 12)

        text = _STATE_TEXT.get(self._state, self._state)
        if self._state == "processing":
            text += "." * self._dots
        painter.setPen(QColor("#f4f4f5"))
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(38, 0, 120, self.height(), Qt.AlignVCenter, text)

        if self._state == "recording":
            # 8-bar level meter on the right side of the pill.
            bars = 8
            base_x = self.width() - 16 - bars * 9
            active = round(self._level * bars)
            for i in range(bars):
                bar_height = 6 + i * 2.6
                y = self.height() / 2 + bar_height / 2
                painter.setBrush(
                    QColor("#30a46c") if i < active else QColor(255, 255, 255, 45)
                )
                painter.drawRoundedRect(
                    int(base_x + i * 9), int(y - bar_height), 6, int(bar_height), 2, 2
                )
        painter.end()
