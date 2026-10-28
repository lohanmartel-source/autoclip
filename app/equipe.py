"""Montage d'équipe avec tes amis du défi.

Si tu l'actives (onglet Stats → Défi), ton AutoClip envoie chaque heure tes 2 meilleurs kills de la
semaine pas encore partagés : 6 s, 720p, son du jeu seulement (jamais ton micro), via ntfy, sur un
sujet que seuls ceux qui ont le code du défi connaissent. Les AutoClip de tes amis font pareil, et chacun
récupère les clips des autres dans clips_amis/ (gardés 14 jours). ntfy garde les fichiers environ 3 h :
il faut que le PC de chacun soit allumé de temps en temps.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from outils import _FLAGS, flux_audio, infos_video

from chemins import DONNEES as ICI
DOSSIER = ICI / "clips_amis"
NTFY = "https://ntfy.sh"
LIMITE = 14 * 1024 * 1024
RANG = {"ace": 6, "5k": 6, "4k": 5, "clutch": 4, "3k": 3, "2k": 1, "kill": 0}


def _sujet(code: str) -> str:
    return f"autoclip-{code}-clips"


def _valeur(k: dict) -> float:
    n = k["nom"].lower()
    return max((v for c, v in RANG.items() if c in n), default=0) + (2.5 if k.get("one_tap") else 0)


def _extrait(clip: Path, t: float, sortie: Path) -> bool:
    piste = flux_audio(clip).get("jeu")
    carte = ["-map", "0:v:0"] + (["-map", f"0:{piste}"] if piste is not None else ["-map", "0:a:0?"])
    for crf in (26, 30, 34):
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, t - 4.0):.2f}", "-t", "6", "-i", str(clip)]
                           + carte + ["-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                                      "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(sortie)],
                           capture_output=True, creationflags=_FLAGS)
        if r.returncode != 0:
            return False
        if sortie.stat().st_size <= LIMITE:
            return True
    return False


def partager(config: dict, index: dict, log=print) -> int:
    """Envoie tes 2 meilleurs kills de la semaine pas encore partagés. Retourne le nombre envoyé."""
    from defi import semaine_actuelle
    from index_kills import est_kill, kills_plats
    d = config.get("defi") or {}
    if not d.get("code") or not d.get("partager_clips") or not config.get("pseudo"):
        return 0
    deja = set(d.get("clips_envoyes") or [])
    _, lundi = semaine_actuelle()
    kills = [k for k in kills_plats(index, (config.get("reglages") or {}).get("jeux"), lundi)
             if est_kill(k["nom"]) and k["id"] not in deja]
    kills.sort(key=_valeur, reverse=True)
    envoyes = 0
    DOSSIER.mkdir(exist_ok=True)
    for k in kills[:2]:
        if _valeur(k) < 3:                 # seulement les vrais gros moments (3K+, one taps…)
            break
        f = DOSSIER / f"_envoi_{int(time.time())}.mp4"
        try:
            if not _extrait(Path(k["chemin"]), k["t"], f):
                continue
            meta = {"app": "autoclip", "pseudo": config["pseudo"], "nom": k["nom"], "one_tap": bool(k.get("one_tap")),
                    "arme": k.get("arme"), "agent": k.get("agent"), "t": 4.0}
            params = {"filename": f"{config['pseudo']}_{int(time.time())}.mp4",
                      "message": json.dumps(meta, ensure_ascii=False)}
            req = urllib.request.Request(f"{NTFY}/{_sujet(d['code'])}?{urllib.parse.urlencode(params)}",
                                         data=f.read_bytes(), method="PUT")
            urllib.request.urlopen(req, timeout=120).read()
            deja.add(k["id"])
            envoyes += 1
        except Exception as e:
            log(f"  partage du clip impossible : {e}")
        finally:
            f.unlink(missing_ok=True)
    d["clips_envoyes"] = sorted(deja)[-200:]
    config["defi"] = d
    return envoyes


def recevoir(config: dict, log=print) -> int:
    """Télécharge les clips de tes amis (pas les tiens). Retourne le nombre de nouveaux clips."""
    d = config.get("defi") or {}
    if not d.get("code"):
        return 0
    DOSSIER.mkdir(exist_ok=True)
    try:
        brut = urllib.request.urlopen(f"{NTFY}/{_sujet(d['code'])}/json?poll=1&since=12h", timeout=30).read()
    except Exception:
        return 0
    nouveaux = 0
    for ligne in brut.decode("utf-8", "replace").splitlines():
        try:
            m = json.loads(ligne)
            meta = json.loads(m.get("message") or "{}")
        except (ValueError, TypeError):
            continue
        piece = m.get("attachment") or {}
        if meta.get("app") != "autoclip" or not piece.get("url") or meta.get("pseudo") == config.get("pseudo"):
            continue
        cible = DOSSIER / f"{m.get('id', 'x')}.mp4"
        if cible.exists():
            continue
        try:
            with urllib.request.urlopen(piece["url"], timeout=120) as r:
                donnees = r.read(LIMITE + 1)
            if len(donnees) > LIMITE:
                continue
            cible.write_bytes(donnees)
            cible.with_suffix(".json").write_text(json.dumps({**meta, "recu": time.time()}, ensure_ascii=False),
                                                  encoding="utf-8")
            nouveaux += 1
        except Exception:
            continue
    # Ménage : clips d'amis de plus de 14 jours.
    for f in DOSSIER.glob("*.json"):
        try:
            if time.time() - json.loads(f.read_text(encoding="utf-8")).get("recu", 0) > 14 * 86400:
                f.with_suffix(".mp4").unlink(missing_ok=True)
                f.unlink(missing_ok=True)
        except Exception:
            continue
    if nouveaux:
        log(f"  {nouveaux} clip(s) reçu(s) de tes amis")
    return nouveaux


def clips_amis() -> list[dict]:
    res = []
    if not DOSSIER.is_dir():
        return res
    for f in sorted(DOSSIER.glob("*.json")):
        v = f.with_suffix(".mp4")
        if not v.exists():
            continue
        try:
            res.append({**json.loads(f.read_text(encoding="utf-8")), "chemin": str(v)})
        except Exception:
            continue
    return res


def moments_amis(log=print) -> list:
    from analyse_clips import Moment
    from medal_db import poids
    res = []
    for c in clips_amis():
        try:
            info = infos_video(Path(c["chemin"]))
        except Exception:
            continue
        nom = c.get("nom", "Kill") + ("+One Tap" if c.get("one_tap") else "")
        res.append(Moment(clip=Path(c["chemin"]), t=float(c.get("t", 4.0)), score=10 + poids(nom) + 0.5,
                          duree_clip=info["duree"], a_audio=info["a_audio"], largeur=info["largeur"],
                          hauteur=info["hauteur"], etiquette=f"@{c['pseudo']} {nom}",
                          arme=c.get("arme") or "", agent=c.get("agent") or ""))
    log(f"  {len(res)} clip(s) de tes amis")
    return res
