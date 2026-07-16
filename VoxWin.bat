@echo off
rem Запуск VoxWin без консольного окна.
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m voxwin %*
) else (
    start "" pythonw -m voxwin %*
)
