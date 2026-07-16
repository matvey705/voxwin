"""Console diagnostics: `py -m voxwin --diagnose [--full]`.

Prints environment info, audio devices, CUDA availability and (in full
mode) loads the configured model and times a test transcription.
"""

from __future__ import annotations

import platform
import sys
import time


def run_diagnostics(full: bool = False) -> int:
    from . import __version__
    from .config import CONFIG_PATH, Config

    cfg = Config.load()

    print(f"=== VoxWin {__version__} diagnostics ===")
    print(f"Python:  {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"OS:      {platform.platform()}")
    print(f"Config:  {CONFIG_PATH} "
          f"({'существует' if CONFIG_PATH.exists() else 'ещё не создан'})")
    print(f"Models:  {cfg.resolved_models_dir()}")
    print(f"Model:   {cfg.model_size} | device={cfg.device} | "
          f"compute={cfg.compute_type} | language={cfg.language}")
    print()

    # --- packages -----------------------------------------------------------
    for module_name in ("faster_whisper", "ctranslate2", "sounddevice", "numpy",
                        "PySide6", "keyboard", "win32clipboard"):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "ok")
            print(f"[OK]   {module_name} {version}")
        except Exception as exc:
            print(f"[FAIL] {module_name}: {exc}")
    print()

    # --- audio devices ---------------------------------------------------------
    try:
        from .audio import list_input_devices

        devices = list_input_devices()
        if not devices:
            print("[WARN] Микрофоны не найдены!")
        for dev in devices:
            marker = " (по умолчанию)" if dev["is_default"] else ""
            print(f"  mic #{dev['index']}: {dev['name']} "
                  f"[{dev['hostapi']}]{marker}")
    except Exception as exc:
        print(f"[FAIL] Опрос аудиоустройств: {exc}")
    print()

    # --- CUDA -------------------------------------------------------------------
    try:
        from .transcriber import cuda_device_count

        count = cuda_device_count()
        if count > 0:
            print(f"[OK]   CUDA-устройств: {count} — будет использован GPU")
        else:
            print("[INFO] CUDA недоступна — будет использован CPU (int8). "
                  "Для GPU: pip install -r requirements-gpu.txt")
    except Exception as exc:
        print(f"[FAIL] Проверка CUDA: {exc}")
    print()

    if not full:
        print("Запустите с --diagnose-full для теста загрузки модели "
              "(скачает модель, если её нет).")
        return 0

    # --- model load + test inference ----------------------------------------------
    try:
        import numpy as np

        from .transcriber import Transcriber

        print(f"Загружаю модель {cfg.model_size}…")
        transcriber = Transcriber(cfg)
        started = time.monotonic()
        transcriber._ensure_model(cfg)  # noqa: SLF001 (diagnostic tool)
        print(f"[OK]   Модель загружена за {time.monotonic() - started:.1f} с")

        silence = np.zeros(16000, dtype=np.float32)
        started = time.monotonic()
        result = transcriber._transcribe(cfg, silence)  # noqa: SLF001
        print(f"[OK]   Тестовое распознавание 1 с тишины: "
              f"{time.monotonic() - started:.2f} с, текст: {result.text!r}")
    except Exception as exc:
        print(f"[FAIL] Тест модели: {exc}")
        return 1

    print("\nВсё готово к работе.")
    return 0
