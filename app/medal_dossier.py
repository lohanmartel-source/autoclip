"""Trouve tout seul le dossier où Medal enregistre les clips."""
from __future__ import annotations

import json
import os
import re
import string
import sys
from pathlib import Path

EXT = {".mp4", ".mov", ".mkv", ".webm"}


def _a_des_clips(d: Path, profondeur: int = 2) -> bool:
    try:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in EXT:
                return True
            if profondeur and f.is_dir() and f.name.lower() not in ("databases", "thumbnails", "temp"):
                if _a_des_clips(f, profondeur - 1):
                    return True
    except OSError:
        pass
    return False


def _depuis_reglages_medal() -> list[Path]:
    """Chemins mentionnés dans les fichiers de réglages de Medal (%APPDATA%\\Medal)."""
    res: list[Path] = []
    racines = [Path(os.environ[v]) / "Medal" for v in ("APPDATA", "LOCALAPPDATA") if os.environ.get(v)]
    motif = re.compile(r'"([A-Za-z]:(?:\\\\|/)[^"]{1,200})"')
    for racine in racines:
        if not racine.is_dir():
            continue
        for f in list(racine.glob("*.json")) + list(racine.glob("store/*.json")):
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                texte = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in motif.finditer(texte):
                try:
                    chemin = Path(json.loads(f'"{m.group(1)}"'))
                except Exception:
                    continue
                if "medal" in str(chemin).lower() and chemin.is_dir():
                    res.append(chemin)
    return res


def candidats() -> list[Path]:
    liste = _depuis_reglages_medal()
    liste += [Path.home() / "Videos" / "Medal", Path.home() / "Medal"]
    if sys.platform == "win32":
        for lettre in string.ascii_uppercase[2:]:
            racine = Path(f"{lettre}:\\")
            if racine.exists():
                liste += [racine / "Medal", racine / "Medal" / "Clips", racine / "Videos" / "Medal"]
    vus, res = set(), []
    for c in liste:
        cle = str(c).lower()
        if cle not in vus:
            vus.add(cle)
            res.append(c)
    return res


def trouver() -> str:
    """Le dossier Medal le plus probable, ou "" si rien n'est trouvé."""
    for c in candidats():
        try:
            if c.is_dir() and _a_des_clips(c, 3):
                # Un sous-dossier « Clips » de Medal : on remonte au dossier Medal (il contient Databases).
                if c.name.lower() == "clips" and (c.parent / "Databases").is_dir():
                    return str(c.parent)
                return str(c)
        except OSError:
            continue
    return ""
