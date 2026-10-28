# Télécharge FFmpeg (build GPL Windows 64 bits de BtbN) et garde ffmpeg.exe, ffprobe.exe et la licence.
$ErrorActionPreference = "Stop"
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$ici = Split-Path -Parent $MyInvocation.MyCommand.Path
$zip = Join-Path $env:TEMP "autoclip-ffmpeg.zip"
$tmp = Join-Path $env:TEMP "autoclip-ffmpeg"
$dest = Join-Path $ici "ffmpeg"

Write-Host "Téléchargement de FFmpeg..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath $tmp -Force
$racine = Get-ChildItem $tmp -Directory | Select-Object -First 1
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item (Join-Path $racine.FullName "bin\ffmpeg.exe") $dest -Force
Copy-Item (Join-Path $racine.FullName "bin\ffprobe.exe") $dest -Force
Copy-Item (Join-Path $racine.FullName "LICENSE.txt") $dest -Force -ErrorAction SilentlyContinue
Remove-Item $zip, $tmp -Recurse -Force
Write-Host "FFmpeg prêt dans $dest"
