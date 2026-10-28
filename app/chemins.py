"""Emplacements des fichiers d'AutoClip.

- RESSOURCES : fichiers livrés avec l'app (page web, modèles Valorant). En lecture seule une fois installée.
- DONNEES : ce que l'app écrit (réglages, index des kills, vignettes, styles…).
  App installée (.exe) : %APPDATA%\\AutoClip. Depuis le code source : le dossier du code (mode portable).
- FFmpeg livré avec l'app (dossier ffmpeg/ à côté de l'exe) : ajouté au PATH au démarrage.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

VERSION = "1.1.0"
NOM = "AutoClip"

INSTALLEE = bool(getattr(sys, "frozen", False))
ICI = Path(__file__).resolve().parent
RESSOURCES = Path(getattr(sys, "_MEIPASS", ICI))
DOSSIER_EXE = Path(sys.executable).resolve().parent if INSTALLEE else ICI


def _donnees() -> Path:
    force = os.environ.get("AUTOCLIP_DONNEES")
    if force:
        return Path(force).expanduser()
    if not INSTALLEE:
        return ICI
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / NOM
    return Path.home() / ".autoclip"


DONNEES = _donnees()
DONNEES.mkdir(parents=True, exist_ok=True)


def preparer_ffmpeg() -> None:
    """Met le FFmpeg livré avec l'app en tête du PATH (s'il existe)."""
    for d in (DOSSIER_EXE / "ffmpeg", DOSSIER_EXE / "ffmpeg" / "bin", RESSOURCES / "ffmpeg"):
        exe = d / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
        if exe.is_file():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            return


preparer_ffmpeg()
