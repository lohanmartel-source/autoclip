"""Lecture de la base de données locale de Medal pour trouver les vrais moments forts.

Medal enregistre, pour chaque clip, des « bookmarks » d'événements du jeu (ex. Valorant :
« Kill », « 2K », « 3K », « 4K », « ACE ») avec leur position dans le clip. C'est beaucoup
plus fiable que l'analyse du son : on sait exactement quand chaque kill arrive.

La base est dans <dossier Medal>/Databases/medal-<id>.db (SQLite). Medal la garde ouverte,
donc on travaille sur une copie temporaire. Le champ « metadata » est dans un format binaire
maison de Medal : on n'en lit que les quelques champs utiles.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Poids des événements : un multi-kill vaut plus qu'un kill seul.
POIDS = {"kill": 1.0, "1k": 1.0, "2k": 1.6, "3k": 2.4, "4k": 3.3, "5k": 4.5, "ace": 4.5,
         "clutch": 3.0, "one tap": 2.2, "headshot": 1.2, "knife": 2.0, "defuse": 1.3, "plant": 0.8}


@dataclass
class InfoClip:
    titre: str = ""
    duree: float = 0.0
    debut: datetime | None = None                       # début réel du clip (UTC)
    evenements: list[tuple[str, float]] = field(default_factory=list)  # (nom, fraction 0–1)
    match: str = ""                                      # identifiant de la partie (Valorant)
    favori: bool = False                                 # clip mis en favori dans Medal


def poids(nom: str) -> float:
    n = nom.lower().strip()
    if n in POIDS:
        return POIDS[n]
    m = re.match(r"(\d)\s*k", n)
    if m:
        return POIDS.get(f"{m.group(1)}k", 1.0 + 0.8 * (int(m.group(1)) - 1))
    trouves = [val for cle, val in POIDS.items() if cle in n]
    return max(trouves) if trouves else 1.0


def _lire_valeur(md: bytes, pos: int) -> str | None:
    """Lit une chaîne du format Medal : octet d'en-tête (taille << 4 | type), ou 0xC? + taille."""
    if pos >= len(md):
        return None
    b = md[pos]
    taille, type_ = b >> 4, b & 0x0F
    if type_ not in (3, 5, 7):          # 3 = entier, 5 = nombre, 7 = texte
        return None
    debut = pos + 1
    if taille == 0x0C:                  # taille sur l'octet suivant
        if pos + 1 >= len(md):
            return None
        taille, debut = md[pos + 1], pos + 2
    elif taille > 0x0C:
        return None
    brut = md[debut:debut + taille]
    if len(brut) != taille:
        return None
    try:
        return brut.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _champ(md: bytes, cle: bytes, depuis: int = 0, jusqu_a: int | None = None):
    i = md.find(cle, depuis, jusqu_a if jusqu_a is not None else len(md))
    if i < 0:
        return None, -1
    return _lire_valeur(md, i + len(cle)), i + len(cle)


def lire_metadata(md: bytes) -> InfoClip:
    info = InfoClip()
    titre, _ = _champ(md, b"title")
    info.titre = titre or ""
    duree, _ = _champ(md, b"clipDuration")
    try:
        info.duree = float(duree) if duree else 0.0
    except ValueError:
        pass
    debut, _ = _champ(md, b"clipStartTime")
    if debut:
        try:
            info.debut = datetime.fromisoformat(debut.replace("Z", "+00:00"))
        except ValueError:
            pass

    m = re.search(rb"(?s)match_id.{1,2}?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", md)
    if m:
        info.match = m.group(1).decode()
    i = md.find(b"isFavorited")
    info.favori = i >= 0 and i + 11 < len(md) and md[i + 11] == 1   # 1 = vrai, 2 = faux

    vus = set()
    pos = 0
    while True:
        i = md.find(b"bookmarkName", pos)
        if i < 0:
            break
        nom = _lire_valeur(md, i + len(b"bookmarkName"))
        fin = md.find(b"bookmarkName", i + 12)
        pct, _ = _champ(md, b"startTimePercentage", i, fin if fin > 0 else None)
        pos = i + 12
        if not nom or pct is None:
            continue
        try:
            f = float(pct)
        except ValueError:
            continue
        if 0 <= f <= 1 and (nom, round(f, 4)) not in vus:
            vus.add((nom, round(f, 4)))
            info.evenements.append((nom, f))
    info.evenements.sort(key=lambda e: e[1])
    return info


def _normaliser(p) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def trouver_base(dossiers: list[Path]) -> Path | None:
    """Cherche <dossier Medal>/Databases/medal-*.db à partir des dossiers de clips."""
    candidats = []
    for d in dossiers:
        d = Path(d).expanduser()
        for parent in [d, *list(d.parents)[:3]]:
            candidats.append(parent / "Databases")
    candidats.append(Path.home() / "Videos" / "Medal" / "Databases")
    for c in candidats:
        if c.is_dir():
            bases = sorted(c.glob("medal-*.db"), key=lambda f: f.stat().st_mtime, reverse=True)
            if bases:
                return bases[0]
    return None


def charger(dossiers: list[Path], log=print) -> dict[str, InfoClip]:
    """Retourne {chemin normalisé du clip: InfoClip}. Dictionnaire vide si pas de base Medal."""
    base = trouver_base(dossiers)
    if base is None:
        return {}
    temp = Path(tempfile.mkdtemp(prefix="autoclip_medal_"))
    try:
        copie = temp / "medal.db"
        for suffixe in ("", "-wal", "-shm"):
            src = Path(str(base) + suffixe)
            if src.exists():
                shutil.copy2(src, str(copie) + suffixe)
        con = sqlite3.connect(str(copie))
        try:
            lignes = con.execute("select video_path, metadata from contents where video_path is not null").fetchall()
        finally:
            con.close()
    except Exception as e:  # base verrouillée, format changé… on retombe sur l'analyse du son
        log(f"  base Medal illisible ({e}) : analyse du son seulement")
        return {}
    finally:
        shutil.rmtree(temp, ignore_errors=True)

    resultat = {}
    for chemin, md in lignes:
        if not isinstance(md, (bytes, bytearray)):
            md = (md or "").encode("utf-8", "replace")
        try:
            resultat[_normaliser(chemin)] = lire_metadata(bytes(md))
        except Exception:
            continue
    return resultat


def info_pour(infos: dict[str, InfoClip], clip: Path) -> InfoClip | None:
    return infos.get(_normaliser(Path(clip).resolve())) or infos.get(_normaliser(clip))
