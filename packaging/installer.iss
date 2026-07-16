; Inno Setup script для VoxWin.
; 1) Соберите приложение (интерпретатором окружения проекта, НЕ глобальным py):
;      .venv\Scripts\python.exe -m PyInstaller packaging\voxwin.spec --noconfirm
; 2) Откройте этот файл в Inno Setup Compiler (https://jrsoftware.org/isinfo.php)
;    и нажмите Compile. Результат: packaging\Output\VoxWin-Setup.exe

#define MyAppName "VoxWin"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "VoxWin"
#define MyAppExeName "VoxWin.exe"

[Setup]
AppId={{6B8F52B7-1B0C-4E1A-9C64-VOXWIN100000}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=VoxWin-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Пользовательская установка без прав администратора:
PrivilegesRequired=lowest
OutputDir=Output
; Приложение держит этот mutex (voxwin/winutil.py) — установщик попросит
; закрыть работающий VoxWin вместо ошибок "файл занят" при обновлении.
AppMutex=VoxWinSingleton

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "autostart"; Description: "Запускать {#MyAppName} при входе в Windows"; \
    GroupDescription: "Дополнительно:"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\VoxWin\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "{#MyAppName}"; \
    ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; \
    Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; \
    Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\VoxWin"
; Скачанные Whisper-модели (0.5–4.5 ГБ) — не оставляем сиротой на диске:
Type: filesandordirs; Name: "{localappdata}\VoxWin"
