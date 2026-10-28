"""Coach : tes morts dans tes clips Valorant.

Quand tu meurs, Valorant ajoute une ligne au fil des kills : le tueur à gauche (fond rouge) et TOI à
droite, avec ton liseré jaune. On lit le fil des kills 2 fois par seconde pour trouver ces lignes, puis
l'arme et l'agent du tueur. Analyse gardée en cache (morts.json), faite en arrière-plan par le serveur.
"""
from __future__ import annotations

import json
import subprocess
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from outils import _FLAGS, infos_video

from chemins import DONNEES as ICI
CACHE = ICI / "morts.json"
_verrou = threading.Lock()
X0, Y0, LARG, HAUT = 1250, 90, 655, 250     # zone du fil des kills (image 1920×1080)


def charger() -> dict:
    with _verrou:
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}


def sauver(d: dict) -> None:
    with _verrou:
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d), encoding="utf-8")
        tmp.replace(CACHE)


def _ligne_de_mort(img: np.ndarray, y0: int):
    """Tueur ennemi (fond rouge à gauche de l'arme) et toi à droite (liseré jaune-vert).
    Retourne (masque de l'arme, x de l'arme) ou None."""
    import vision_valorant as v
    trouve = v._icone_arme(img, y0)
    if trouve is None:
        return None
    masque, x = trouve
    x_fin = x + masque.shape[1]
    allie, ennemi = v._couleurs(img[y0 + 3:y0 + 31, max(1250, x - 200):x].astype(np.int16))
    if ennemi.mean() < 0.04 or ennemi.mean() < 2 * allie.mean():
        return None
    droite = img[y0 - 2:y0 + 36, x_fin:1905].astype(np.int16)
    if droite.size == 0:
        return None
    jaune = (droite[..., 0] > 190) & (droite[..., 1] > 190) & (droite[..., 2] < 140)
    if jaune.mean() < 0.015:
        return None
    return masque, x


def analyser_clip(clip: Path) -> list[dict]:
    import vision_valorant as v
    cmd = ["ffmpeg", "-v", "error", "-i", str(clip), "-vf",
           f"fps=2,scale=1920:1080,crop={LARG}:{HAUT}:{X0}:{Y0}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        brut = subprocess.run(cmd, capture_output=True, creationflags=_FLAGS, timeout=180).stdout
    except Exception:
        return []
    n = len(brut) // (LARG * HAUT * 3)
    if n == 0:
        return []
    images = np.frombuffer(brut[:n * LARG * HAUT * 3], np.uint8).reshape(n, HAUT, LARG, 3)
    img = np.zeros((1080, 1920, 3), np.uint8)
    morts: list[dict] = []
    for i in range(n):
        img[Y0:Y0 + HAUT, X0:X0 + LARG] = images[i]
        t = i / 2
        if morts and t - morts[-1]["t"] < 8:
            continue                      # la même ligne reste affichée quelques secondes
        for y0 in v._lignes(img):
            trouve = _ligne_de_mort(img, y0)
            if trouve is None:
                continue
            masque, x = trouve
            arme, _ = v._reconnaitre_arme(masque)
            agent, _ = v._reconnaitre_agent(img, y0, x)
            if arme is None and agent is None:
                continue                  # ni arme ni portrait reconnus : sûrement un faux positif
            morts.append({"t": max(0.0, t - 0.25), "arme": arme, "agent": agent})
            break
    return morts


def indexer(racine: str, log=print, arret: threading.Event | None = None, max_clips: int | None = None) -> int:
    from autoclip import lister_clips, nom_jeu
    cache = charger()
    faits = 0
    for clip in lister_clips([racine]):
        if arret is not None and arret.is_set():
            break
        if max_clips is not None and faits >= max_clips:
            break
        if "valorant" not in nom_jeu(clip).lower():
            continue
        try:
            cle = f"{clip.name}:{int(clip.stat().st_mtime)}"
        except OSError:
            continue
        if cle in cache:
            continue
        cache[cle] = {"chemin": str(clip), "morts": analyser_clip(clip)}
        faits += 1
        if faits % 5 == 0:
            sauver(cache)
    if faits:
        sauver(cache)
    return faits


def toutes(cache: dict | None = None) -> list[dict]:
    from index_kills import date_clip
    cache = cache if cache is not None else charger()
    res = []
    for cle, c in cache.items():
        p = Path(c["chemin"])
        if not p.exists():
            continue
        for m in c["morts"]:
            res.append({**m, "clip": p, "date": date_clip(p)})
    res.sort(key=lambda m: (m["date"], m["t"]), reverse=True)
    return res


def resume(depuis: float | None = None) -> dict:
    ms = [m for m in toutes() if depuis is None or m["date"] >= depuis]
    armes = Counter(m["arme"] for m in ms if m["arme"])
    agents = Counter(m["agent"] for m in ms if m["agent"])
    cache = charger()
    conseils = []
    if armes:
        a, n = armes.most_common(1)[0]
        if a in ("Operator", "Outlaw", "Marshal") and n >= 3:
            conseils.append(f"Les snipers t'ont eu {n} fois : évite de repeek le même angle et utilise tes utilitaires pour passer.")
        elif n >= 3:
            conseils.append(f"Tu meurs surtout contre le {a} ({n} fois) : soigne ton placement de viseur à hauteur de tête.")
    if len(ms) >= 5:
        conseils.append("Regarde le montage de tes morts au ralenti : cherche d'où vient le tir avant de voir l'ennemi.")
    return {"morts": len(ms), "clips_analyses": len(cache), "armes": armes.most_common(6),
            "agents": agents.most_common(6), "conseils": conseils}


def moments_morts(log=print, maximum: int = 12, depuis: float | None = None) -> list:
    """Moments « mort » pour le montage de tes erreurs (les plus récentes d'abord)."""
    from analyse_clips import Moment
    res = []
    for m in toutes():
        if depuis is not None and m["date"] < depuis:
            continue
        try:
            info = infos_video(m["clip"])
        except Exception:
            continue
        if not 2.0 < m["t"] < info["duree"] - 0.8:
            continue
        etiquette = "Mort" + (f" - {m['arme']}" if m["arme"] else "")
        res.append(Moment(clip=m["clip"], t=m["t"], score=12 - 0.01 * len(res), duree_clip=info["duree"],
                          a_audio=info["a_audio"], largeur=info["largeur"], hauteur=info["hauteur"],
                          etiquette=etiquette, arme=m["arme"] or "", agent=m["agent"] or ""))
        if len(res) >= maximum:
            break
    log(f"  {len(res)} mort(s) retenue(s) pour le montage")
    return res
