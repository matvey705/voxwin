# -*- mode: python ; coding: utf-8 -*-
# Сборка: из корня проекта, интерпретатором venv (в нём живут зависимости):
#   .venv\Scripts\python.exe -m pip install pyinstaller
#   .venv\Scripts\python.exe -m PyInstaller packaging\voxwin.spec --noconfirm
# Результат: dist\VoxWin\VoxWin.exe (onedir — быстрее стартует, проще с DLL).

import os

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

project_root = os.path.abspath(os.path.join(SPECPATH, ".."))

# Silero-VAD onnx и прочие ресурсы faster-whisper обязаны попасть в сборку.
datas = collect_data_files("faster_whisper")
binaries = collect_dynamic_libs("ctranslate2")

a = Analysis(
    [os.path.join(project_root, "voxwin_launcher.py")],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "voxwin.settings_ui",
        "win32clipboard",
        "win32event",
        "win32api",
        "winerror",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "matplotlib", "IPython", "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoxWin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # фоновое трей-приложение, без консоли
    disable_windowed_traceback=False,
    icon=None,              # добавьте .ico при желании: icon="voxwin.ico"
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VoxWin",
)
