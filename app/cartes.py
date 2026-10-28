"""Carte de match : une image à partager après une partie (kills clippés, meilleur round, agent, armes),
sur fond de ton meilleur kill de la partie."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from index_kills import est_kill
from outils import chemin_filtre, executer

from chemins import DONNEES as ICI
DOSSIER = ICI / "cartes"
ORDRE = {"ace": 5, "5k": 5, "4k": 4, "3k": 3, "2k": 2, "kill": 1}


def _niveau(nom: str) -> int:
    n = nom.lower()
    return max((v for k, v in ORDRE.items() if k in n), default=1)


def matchs_recents(index: dict, nombre: int = 15) -> list[dict]:
    """Les dernières parties (regroupées par identifiant de partie Medal), de la plus récente à la plus vieille."""
    parties: dict[str, dict] = {}
    for nom, c in index.get("clips", {}).items():
        m = c.get("match")
        if not m:
            continue
        p = parties.setdefault(m, {"match": m, "date": c["date"], "clips": [], "jeu": c["jeu"]})
        p["date"] = max(p["date"], c["date"])
        p["clips"].append((nom, c))
    res = []
    for p in parties.values():
        kills, vus, armes, agents = [], [], Counter(), Counter()
        for nom, c in p["clips"]:
            if c.get("agent"):
                agents[c["agent"]] += 1
            for k in c["kills"]:
                if not est_kill(k["nom"]):
                    continue
                quand = (c.get("debut") or 0) + k["t"]
                if c.get("debut") and any(abs(quand - x) < 2.5 for x in vus):
                    continue                    # clips qui se chevauchent : un kill une fois
                vus.append(quand)
                kills.append((nom, c, k))
                if k.get("arme"):
                    armes[k["arme"]] += 1
        if not kills:
            continue
        meilleur = max(kills, key=lambda x: (_niveau(x[2]["nom"]), x[2].get("one_tap") or False))
        # Medal nomme chaque kill d'un round par son rang (Kill, 2K, 3K, 4K, Ace) : un round à 4 kills
        # contient aussi un « 3K ». On compte donc les rounds selon leur meilleur kill.
        n5 = sum(1 for _, _, k in kills if _niveau(k["nom"]) == 5)
        n4 = sum(1 for _, _, k in kills if _niveau(k["nom"]) == 4)
        n3 = sum(1 for _, _, k in kills if _niveau(k["nom"]) == 3)
        res.append({
            "match": p["match"], "date": p["date"], "jeu": p["jeu"], "kills": len(kills),
            "aces": n5, "quatre": max(0, n4 - n5), "trois": max(0, n3 - n4),
            "one_taps": sum(1 for _, _, k in kills if k.get("one_tap")),
            "meilleur": meilleur[2]["nom"], "agent": agents.most_common(1)[0][0] if agents else "",
            "armes": [a for a, _ in armes.most_common(3)],
            "fond": {"chemin": meilleur[1]["chemin"], "t": meilleur[2]["t"]},
        })
    res.sort(key=lambda p: p["date"], reverse=True)
    return res[:nombre]


def _texte_meilleur(nom: str) -> str:
    n = _niveau(nom)
    return {5: "ACE", 4: "4K", 3: "3K", 2: "2K"}.get(n, "1 KILL")


def faire_carte(partie: dict, pseudo: str, police: Path | None) -> Path | None:
    """Image 1080×1350 (format post Instagram / TikTok photo)."""
    if police is None:
        return None
    DOSSIER.mkdir(exist_ok=True)
    sortie = DOSSIER / f"carte_{partie['match'][:8]}.jpg"
    W, H = 1080, 1350
    jour = (datetime.fromtimestamp(partie["date"]) - timedelta(hours=5))
    mois = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
    lignes = [
        ("titre", f"PARTIE DU {jour.day} {mois[jour.month - 1].upper()}"),
        ("gros", str(partie["kills"])),
        ("sous", "KILLS DANS MES CLIPS"),
        ("best", f"MEILLEUR ROUND : {_texte_meilleur(partie['meilleur'])}"),
    ]
    details = []
    if partie["aces"]:
        details.append(f"{partie['aces']} ACE")
    if partie["quatre"]:
        details.append(f"{partie['quatre']} x 4K")
    if partie["trois"]:
        details.append(f"{partie['trois']} x 3K")
    if partie["one_taps"]:
        details.append(f"{partie['one_taps']} ONE TAP" + ("S" if partie["one_taps"] > 1 else ""))
    if details:
        lignes.append(("info", "   ".join(details)))
    if partie["agent"] or partie["armes"]:
        lignes.append(("info", "  ·  ".join(x for x in [partie["agent"].upper()] + [a.upper() for a in partie["armes"]] if x)))
    if pseudo:
        lignes.append(("pseudo", "@" + pseudo))
    styles = {  # taille, couleur, y
        "titre": (64, "white", 150), "gros": (330, "#FFE14D", 260), "sous": (54, "white", 610),
        "best": (70, "#25F4EE", 760), "info": (50, "white", 880), "pseudo": (52, "white@0.8", 1210),
    }
    filtres = [f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
               f"gblur=sigma=5,eq=brightness=-0.22:saturation=1.3,"
               f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.3:t=fill"]
    y_info = styles["info"][2]
    for i, (genre, texte) in enumerate(lignes):
        f = DOSSIER / f"ligne_{i}.txt"
        f.write_text(texte, encoding="utf-8")
        taille, couleur, y = styles[genre]
        if genre == "info":
            y, y_info = y_info, y_info + 80
        filtres.append(f"drawtext=fontfile='{chemin_filtre(police)}':textfile='{chemin_filtre(f)}'"
                       f":fontsize={taille}:fontcolor={couleur}:borderw={max(2, taille // 16)}:bordercolor=black@0.7"
                       f":x=(w-text_w)/2:y={y}")
    filtres.append("format=yuvj420p")
    fond = partie["fond"]
    try:
        executer(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0, fond['t'] - 0.1):.2f}",
                  "-i", fond["chemin"], "-frames:v", "1", "-vf", ",".join(filtres), "-q:v", "2", str(sortie)])
    except Exception:
        return None
    return sortie
