; Inno Setup 6 : construit AutoClip-Setup-<version>.exe à partir de dist\AutoClip (PyInstaller) et ffmpeg\.
; Lancer : iscc installateur.iss   (build.bat le fait pour toi)
#ifndef Version
  #define Version "1.1.0"
#endif

[Setup]
AppId={{6E1B0F3A-9C2D-4C61-A8E5-0A11C1A0C1A0}
AppName=AutoClip
AppVersion={#Version}
AppPublisher=AutoClip
DefaultDirName={localappdata}\Programs\AutoClip
DefaultGroupName=AutoClip
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=sortie
OutputBaseFilename=AutoClip-Setup-{#Version}
SetupIconFile=autoclip.ico
UninstallDisplayIcon={app}\AutoClip.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
LicenseFile=LICENCE.txt

[Languages]
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "bureau"; Description: "Créer un raccourci sur le bureau"; GroupDescription: "Raccourcis :"

[Files]
Source: "dist\AutoClip\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "ffmpeg\ffmpeg.exe"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion
Source: "ffmpeg\ffprobe.exe"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion
Source: "ffmpeg\LICENSE.txt"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion skipifsourcedoesntexist
Source: "LICENCE.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AutoClip"; Filename: "{app}\AutoClip.exe"
Name: "{group}\Désinstaller AutoClip"; Filename: "{uninstallexe}"
Name: "{userdesktop}\AutoClip"; Filename: "{app}\AutoClip.exe"; Tasks: bureau

[Run]
Filename: "{app}\AutoClip.exe"; Description: "Lancer AutoClip"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
