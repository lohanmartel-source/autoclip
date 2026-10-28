@echo off
rem Construit l'installateur AutoClip-Setup-<version>.exe (dans build\sortie).
rem A lancer sur Windows, depuis le dossier build\. Il faut Python 3.11 ou 3.12.
chcp 65001 >nul
cd /d "%~dp0"
setlocal

where python >nul 2>nul || (echo Python introuvable. Installe Python 3.11+ depuis python.org. & pause & exit /b 1)

echo === 1/4  Environnement Python ===
if not exist .venv-build python -m venv .venv-build
call .venv-build\Scripts\activate.bat
python -m pip install --upgrade pip >nul
python -m pip install -r ..\app\requirements.txt pyinstaller || goto :erreur

echo === 2/4  Application (PyInstaller) ===
pyinstaller --noconfirm --clean autoclip.spec || goto :erreur

echo === 3/4  FFmpeg ===
if not exist ffmpeg\ffmpeg.exe (
  powershell -NoProfile -ExecutionPolicy Bypass -File telecharger_ffmpeg.ps1 || goto :erreur
)

echo === 4/4  Installateur (Inno Setup) ===
set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo Inno Setup introuvable, installation via winget...
  winget install --id JRSoftware.InnoSetup -e --accept-source-agreements --accept-package-agreements
  set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
)
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
"%ISCC%" installateur.iss || goto :erreur

echo.
echo Termine : l'installateur est dans build\sortie
explorer sortie
exit /b 0

:erreur
echo.
echo ECHEC de la construction (voir les messages au-dessus).
pause
exit /b 1
