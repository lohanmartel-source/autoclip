"""Index de tous tes kills : une seule analyse par clip, gardée sur le disque.

Pour chaque kill enregistré par Medal, on garde :
  - une petite image (pour choisir tes kills dans l'app) ;
  - one tap ou pas (Valorant : compteur de munitions à l'écran) ;
  - l'arme et l'agent (Valorant : fil des kills, voir vision_valorant.py).
Le serveur remplit l'index en arrière-plan ; les stats, le choix des kills et les filtres
arme/agent le lisent ensuite instantanément.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from outils import _FLAGS, infos_video

from chemins import DONNEES as ICI
FICHIER = ICI / "index_kills.json"
VIGNETTES = ICI / "vignettes"
VERSION = 2
NON_KILLS = ("clutch", "knock", "assist")   # événements Medal qui ne sont pas un kill de plus (knock : Apex, Fortnite)

_verrou = threading.RLock()


# ---------------------------------------------------------------- lecture / écriture
def charger_index() -> dict:
    with _verrou:
        try:
            d = json.loads(FICHIER.read_text(encoding="utf-8"))
            if d.get("version") == VERSION:
                return d
        except Exception:
            pass
        return {"version": VERSION, "clips": {}}


def sauver_index(d: dict) -> None:
    with _verrou:
        temp = FICHIER.with_suffix(".tmp")
        temp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        temp.replace(FICHIER)


def est_kill(nom: str) -> bool:
    return not any(x in nom.lower() for x in NON_KILLS)


def date_clip(clip: Path) -> float:
    """Medal met la date dans le nom : MedalTVValorant20261021212910605.mp4 → 21 oct. 2026 21:29."""
    m = re.search(r"(20\d{12})", clip.stem)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").timestamp()
        except ValueError:
            pass
    return clip.stat().st_mtime


def _vignette(clip: Path, t: float, cible: Path) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}", "-i", str(clip), "-frames:v", "1",
           "-vf", "scale=320:-2", "-q:v", "5", str(cible)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_FLAGS)
    except Exception:
        return False
    return cible.exists()


# ---------------------------------------------------------------- construction
def indexer(racine: str, log=print, arret: threading.Event | None = None, max_clips: int | None = None) -> int:
    """Analyse les clips pas encore indexés (du plus récent au plus ancien). Retourne le nombre traité."""
    from autoclip import lister_clips, nom_jeu
    from analyse_clips import tirs_avant_kill
    from medal_db import charger, info_pour
    import vision_valorant

    VIGNETTES.mkdir(exist_ok=True)
    infos = charger([Path(racine)], log)
    if not infos:
        return 0
    clips = lister_clips([racine])
    index = charger_index()
    faits = 0
    for clip in clips:
        if arret is not None and arret.is_set():
            break
        if max_clips is not None and faits >= max_clips:
            break
        ic = info_pour(infos, clip)
        if ic is None or not ic.evenements:
            continue
        cle = clip.name
        try:
            mtime = clip.stat().st_mtime
        except OSError:
            continue
        deja = index["clips"].get(cle)
        # 1.4.0 lit aussi la liste « bookmarks » de Medal (aces CS2…) : un clip indexé avec moins
        # d'événements qu'il n'en a maintenant est refait.
        if (deja and abs(deja.get("mtime", 0) - mtime) < 1 and deja.get("complet")
                and deja.get("n_evts", len(deja.get("kills", []))) >= len(ic.evenements)):
            continue
        try:
            info = infos_video(clip)
        except Exception:
            continue
        duree = info["duree"]
        valorant = "valorant" in nom_jeu(clip).lower()
        kills, agents = [], Counter()
        for nom, f in ic.evenements:
            t = f * duree
            if not 0.3 < t < duree - 0.2:
                continue
            k = {"t": round(t, 3), "nom": nom, "one_tap": None, "arme": None}
            vig = VIGNETTES / f"{clip.stem}_{int(t * 100)}.jpg"
            if vig.exists() or _vignette(clip, t + 0.03, vig):
                k["vignette"] = vig.name
            if valorant and est_kill(nom):
                if "knife" not in nom.lower():
                    n_tirs = tirs_avant_kill(clip, t, info["largeur"], info["hauteur"])
                    k["one_tap"] = (n_tirs == 1) if n_tirs is not None else None
                try:
                    v = vision_valorant.analyser_kill(clip, t)
                except Exception:
                    v = {}
                k["arme"] = "Couteau" if "knife" in nom.lower() else v.get("arme")
                if v.get("agent"):
                    agents[v["agent"]] += 1
            elif est_kill(nom):
                # CS2 : Medal donne l'arme et marque lui-même les one taps.
                from medal_db import arme_evenement
                k["arme"] = arme_evenement(ic, f) or None
                k["one_tap"] = True if "one tap" in nom.lower() else None
            kills.append(k)
        agent = None
        if agents:
            agent, n = agents.most_common(1)[0]
            if n < 2 and len(agents) > 1:      # pas sûr : on ne devine pas
                agent = None
        index["clips"][cle] = {
            "chemin": str(clip), "mtime": mtime, "date": date_clip(clip), "jeu": nom_jeu(clip),
            "match": ic.match, "favori": ic.favori, "duree": round(duree, 2), "agent": agent,
            "debut": ic.debut.timestamp() if ic.debut else None,
            "taille": clip.stat().st_size, "kills": kills, "complet": True, "n_evts": len(ic.evenements),
        }
        faits += 1
        if faits % 5 == 0:
            sauver_index(index)
            log(f"  index : {faits} clip(s) analysé(s)")
    # On oublie les clips supprimés.
    existants = {c.name for c in clips}
    for cle in [c for c in index["clips"] if c not in existants]:
        del index["clips"][cle]
    sauver_index(index)
    return faits


# ---------------------------------------------------------------- utilisation
def kills_plats(index: dict, jeux: list[str] | None = None, depuis: float | None = None) -> list[dict]:
    """Tous les kills, du plus récent au plus ancien, avec les infos du clip."""
    voulus = {j.lower() for j in jeux} if jeux else None
    res = []
    for nom, c in index["clips"].items():
        if voulus and c["jeu"].lower() not in voulus:
            continue
        if depuis and c["date"] < depuis:
            continue
        for k in c["kills"]:
            quand = (c["debut"] + k["t"]) if c.get("debut") else None
            res.append({**k, "clip": nom, "chemin": c["chemin"], "date": c["date"], "jeu": c["jeu"],
                        "agent": c.get("agent"), "id": f"{nom}@{k['t']:.3f}", "_cle": (c.get("match"), quand)})
    res.sort(key=lambda k: (k["date"], k["t"]), reverse=True)
    # Clips qui se chevauchent : chaque kill une seule fois.
    propres, vus = [], defaultdict(list)
    for k in res:
        match, quand = k.pop("_cle")
        if match and quand is not None:
            if any(abs(quand - x) < 2.5 for x in vus[match]):
                continue
            vus[match].append(quand)
        propres.append(k)
    return propres


def infos_kill(index: dict, clip_nom: str, t: float) -> dict | None:
    c = index["clips"].get(clip_nom)
    if not c:
        return None
    k = min(c["kills"], key=lambda k: abs(k["t"] - t), default=None)
    if k is None or abs(k["t"] - t) > 2.5:
        return None
    return {**k, "agent": c.get("agent")}


def _jour(ts: float) -> str:
    # Une session de soirée qui finit après minuit compte pour la veille (coupure à 5 h).
    return (datetime.fromtimestamp(ts) - timedelta(hours=5)).strftime("%Y-%m-%d")


def statistiques(index: dict, jeux: list[str] | None = None, jours: int = 30, depuis: float | None = None) -> dict:
    voulus = {j.lower() for j in jeux} if jeux else None
    clips = [c for c in index["clips"].values() if (not voulus or c["jeu"].lower() in voulus)
             and (depuis is None or c["date"] >= depuis)]
    par_jour = defaultdict(lambda: {"kills": 0, "aces": 0, "multi": 0, "one_taps": 0, "clips": 0, "matchs": set()})
    armes, agents = Counter(), Counter()
    total = Counter()
    vus: dict[str, list[float]] = defaultdict(list)   # clips qui se chevauchent : un kill compté une fois
    for c in sorted(clips, key=lambda c: c["date"]):
        j = _jour(c["date"])
        d = par_jour[j]
        d["clips"] += 1
        if c.get("match"):
            d["matchs"].add(c["match"])
        if c.get("agent"):
            agents[c["agent"]] += sum(1 for k in c["kills"] if est_kill(k["nom"]))
        for k in c["kills"]:
            n = k["nom"].lower()
            if c.get("match") and c.get("debut"):
                quand = c["debut"] + k["t"]
                if any(abs(quand - x) < 2.5 for x in vus[c["match"]]):
                    continue
                vus[c["match"]].append(quand)
            if not est_kill(n):
                if "clutch" in n:
                    total["clutchs"] += 1
                continue
            d["kills"] += 1
            total["kills"] += 1
            if "ace" in n or "5k" in n:
                d["aces"] += 1
                total["aces"] += 1
            if n in ("3k", "4k"):
                d["multi"] += 1
            if n == "4k":
                total["4k"] += 1
            if n == "3k":
                total["3k"] += 1
            if k.get("one_tap"):
                d["one_taps"] += 1
                total["one_taps"] += 1
            if k.get("arme"):
                armes[k["arme"]] += 1
    aujourd_hui = datetime.now() - timedelta(hours=5)
    serie = []
    for i in range(jours - 1, -1, -1):
        j = (aujourd_hui - timedelta(days=i)).strftime("%Y-%m-%d")
        d = par_jour.get(j)
        serie.append({"jour": j, "kills": d["kills"] if d else 0, "aces": d["aces"] if d else 0,
                      "multi": d["multi"] if d else 0, "one_taps": d["one_taps"] if d else 0,
                      "matchs": len(d["matchs"]) if d else 0})
    semaines = defaultdict(int)
    for j, d in par_jour.items():
        an, sem, _ = datetime.strptime(j, "%Y-%m-%d").isocalendar()
        semaines[f"{an}-S{sem:02d}"] += d["kills"]
    meilleur = max(par_jour.items(), key=lambda kv: kv[1]["kills"], default=None)
    matchs = {c["match"] for c in clips if c.get("match")}
    return {
        "total": {"kills": total["kills"], "aces": total["aces"], "4k": total["4k"], "3k": total["3k"],
                  "one_taps": total["one_taps"], "clutchs": total["clutchs"], "clips": len(clips),
                  "matchs": len(matchs)},
        "par_jour": serie,
        "par_semaine": [{"semaine": s, "kills": n} for s, n in sorted(semaines.items())[-12:]],
        "armes": armes.most_common(8),
        "agents": agents.most_common(8),
        "meilleur_jour": {"jour": meilleur[0], "kills": meilleur[1]["kills"]} if meilleur else None,
        "clips_analyses": len(index["clips"]),
    }


# ---------------------------------------------------------------- ménage
def clips_sans_kill(racine: str, jours_min: int = 2) -> list[dict]:
    """Clips où Medal n'a enregistré AUCUN événement (souvent des clips où tu marches).
    Jamais : les favoris, les clips récents (moins de 2 jours) ni ceux absents de la base Medal."""
    from autoclip import lister_clips, nom_jeu
    from medal_db import charger, info_pour
    infos = charger([Path(racine)])
    if not infos:
        return []
    limite = time.time() - jours_min * 86400
    res = []
    for clip in lister_clips([racine]):
        ic = info_pour(infos, clip)
        if ic is None or ic.evenements or ic.favori:
            continue
        st = clip.stat()
        if st.st_mtime > limite:
            continue
        vig = VIGNETTES / f"{clip.stem}_menage.jpg"
        res.append({"nom": clip.name, "chemin": str(clip), "jeu": nom_jeu(clip), "taille": st.st_size,
                    "date": date_clip(clip), "vignette": vig.name})
    res.sort(key=lambda c: c["date"])
    return res


def vignette_menage(chemin: Path) -> Path | None:
    VIGNETTES.mkdir(exist_ok=True)
    vig = VIGNETTES / f"{chemin.stem}_menage.jpg"
    if vig.exists():
        return vig
    try:
        d = infos_video(chemin)["duree"]
    except Exception:
        d = 4.0
    return vig if _vignette(chemin, d / 2, vig) else None


def mettre_a_la_corbeille(chemins: list[Path]) -> tuple[int, list[str]]:
    """Envoie les fichiers dans la Corbeille de Windows (récupérables), jamais de suppression définitive."""
    import sys
    if sys.platform != "win32":
        return 0, ["Corbeille disponible seulement sous Windows"]
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                    ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]
    FO_DELETE, FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FOF_SILENT, FOF_NOERRORUI = 3, 0x40, 0x10, 0x4, 0x400
    ok, erreurs = 0, []
    for p in chemins:
        op = SHFILEOPSTRUCTW(None, FO_DELETE, str(p) + "\0", None,
                             FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI, False, None, None)
        r = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        if r == 0 and not Path(p).exists():
            ok += 1
        else:
            erreurs.append(Path(p).name)
    return ok, erreurs
