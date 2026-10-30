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
# Noms d'événements Medal qui ne sont pas des kills (CS2 : changement de phase de manche).
# « Bomb » (CS2 : bombe posée) et les succès Steam ne sont pas des actions à montrer.
IGNORES = {"roundphasechange", "roundstart", "roundend", "matchstart", "matchend", "bomb", "steam achievement"}
POIDS = {"kill": 1.0, "1k": 1.0, "2k": 1.6, "3k": 2.4, "4k": 3.3, "5k": 4.5, "ace": 4.5,
         "clutch": 3.0, "one tap": 2.2, "headshot": 1.2, "knife": 2.0, "defuse": 1.3, "plant": 0.8,
         # Fortnite, Apex et autres battle royale (noms d'événements Medal possibles).
         "elimination": 1.0, "knock": 0.6, "assist": 0.4, "victory": 4.5, "champion": 4.5,
         "squad wipe": 3.3, "killstreak": 1.6}


@dataclass
class InfoClip:
    titre: str = ""
    duree: float = 0.0
    debut: datetime | None = None                       # début réel du clip (UTC)
    evenements: list[tuple[str, float]] = field(default_factory=list)  # (nom, fraction 0–1)
    match: str = ""                                      # identifiant de la partie (Valorant)
    favori: bool = False                                 # clip mis en favori dans Medal
    armes: dict[float, str] = field(default_factory=dict)  # fraction de l'événement -> arme (CS2 : Medal la donne)


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
    for cle in (b"clipDuration", b"clipDurationSec"):
        duree, _ = _champ(md, cle)
        try:
            info.duree = float(duree) if duree else 0.0
        except ValueError:
            info.duree = 0.0
        if info.duree > 0:
            break
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
    pos = precedent = 0
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
        arme, _ = _champ(md, b"Weapon", max(precedent, i - 160), i)
        precedent = i
        if arme:
            info.armes[round(f, 4)] = arme
    # Liste « bookmarks » (temps en secondes depuis le début du clip). Dans CS2, Medal n'y met souvent
    # QUE là les événements (clips d'ace, de 4K…) : sans elle, ces clips seraient vus « sans kill ».
    if info.duree > 0:
        for m in re.finditer(rb"\x47time", md):
            t = _lire_valeur(md, m.end())
            nom, j = _champ(md, b"\x47name", m.end(), m.end() + 24)
            if t is None or not nom or nom.lower() in IGNORES:
                continue
            try:
                f = float(t) / info.duree
            except ValueError:
                continue
            if not 0 <= f <= 1 or any(n == nom and abs(g - f) * info.duree < 0.08 for n, g in info.evenements):
                continue
            info.evenements.append((nom, f))
            suite = md.find(b"\x47time", j)
            arme, _ = _champ(md, b"Weapon", j, suite if suite > 0 else j + 200)
            if arme:
                info.armes[round(f, 4)] = arme
    # Plusieurs événements au même instant = un seul kill vu de plusieurs façons
    # (CS2 : « Kill », « Headshot », « 3K » et « 4K » sur la même balle) : on les fusionne.
    fusion: list[tuple[str, float]] = []
    for nom, f in sorted(info.evenements, key=lambda e: e[1]):
        if fusion and info.duree > 0 and abs(f - fusion[-1][1]) * info.duree < 0.05:
            if nom.lower() not in fusion[-1][0].lower():
                fusion[-1] = (fusion[-1][0] + " " + nom, fusion[-1][1])
            continue
        fusion.append((nom, f))
    info.evenements = fusion
    info.evenements.sort(key=lambda e: e[1])
    return info


def arme_evenement(info: InfoClip | None, f: float) -> str:
    """Arme enregistrée par Medal pour l'événement à la fraction f (CS2), "" si inconnue."""
    if not info or not info.armes:
        return ""
    k = min(info.armes, key=lambda x: abs(x - f))
    return info.armes[k] if abs(k - f) < 0.01 else ""


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


_PAR_NOM: dict = {}


def info_pour(infos: dict[str, InfoClip], clip: Path) -> InfoClip | None:
    trouve = infos.get(_normaliser(Path(clip).resolve())) or infos.get(_normaliser(clip))
    if trouve is None and infos:
        # Dossier Medal déplacé ou copié : on retrouve le clip par son nom de fichier (unique chez Medal).
        cle = id(infos)
        if _PAR_NOM.get(cle, (None, None))[0] is not infos:
            par_nom = {}
            for k, v in infos.items():
                par_nom.setdefault(os.path.basename(k.replace("\\", "/")).lower(), v)
            _PAR_NOM[cle] = (infos, par_nom)
        par_nom = _PAR_NOM[cle][1]
        trouve = par_nom.get(Path(clip).name.lower())
    return trouve
