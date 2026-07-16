"""Entry point: `py -m voxwin` (or the frozen VoxWin.exe)."""

from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="voxwin", description="VoxWin — локальная голосовая диктовка для Windows"
    )
    parser.add_argument("--diagnose", action="store_true",
                        help="проверить окружение и выйти")
    parser.add_argument("--diagnose-full", action="store_true",
                        help="диагностика + тест загрузки модели")
    parser.add_argument("--settings", action="store_true",
                        help="открыть настройки при запуске")
    parser.add_argument("--reset-config", action="store_true",
                        help="сбросить настройки на значения по умолчанию")
    parser.add_argument("--verbose", action="store_true", help="подробные логи")
    args = parser.parse_args(argv)

    from . import APP_NAME, winutil
    from .config import CONFIG_PATH

    if args.reset_config:
        try:
            CONFIG_PATH.unlink(missing_ok=True)
            print(f"Настройки сброшены ({CONFIG_PATH}).")
        except OSError as exc:
            print(f"Не удалось удалить {CONFIG_PATH}: {exc}")
            return 1
        return 0

    winutil.setup_logging(verbose=args.verbose)
    winutil.inject_system_certificates()

    if args.diagnose or args.diagnose_full:
        from .diagnostics import run_diagnostics

        return run_diagnostics(full=args.diagnose_full)

    if not winutil.acquire_single_instance():
        winutil.message_box(
            f"{APP_NAME} уже запущен — ищите иконку в системном трее."
        )
        return 0

    winutil.set_app_user_model_id()

    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName(APP_NAME)
    qt_app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        winutil.message_box("Системный трей недоступен — запуск невозможен.")
        return 1

    from .app import VoxApp

    vox = VoxApp(qt_app, open_settings=args.settings)
    log.info("VoxWin started")
    exit_code = qt_app.exec()
    del vox
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
