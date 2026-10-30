"""Modèles d'edits (AutoClip Pro) : des recettes prêtes qui reprennent les edits TikTok à succès.

Un modèle fixe :
  - « reglages » : les réglages du montage (champs d'Options : format, durée, style, look, rythme,
    paroles, intro 3D…). L'app les pré-remplit quand on choisit le modèle ; en ligne de commande,
    --modele les met par défaut (une option donnée à la main reste prioritaire) ;
  - « rendu » : ce qui n'est pas un réglage : transitions, effets de kill et leur dosage, durée des plans,
    halo lumineux, grain, force du flou de mouvement, et des comportements propres (intro « showcase »).

Les packs Patreon restent au format {"pack", "styles": […]} (rythme de coupes, voir copie_style.py) ;
un modèle est la recette complète d'un edit. Ajouter un modèle = ajouter une entrée à MODELES.
"""
from __future__ import annotations

import random
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

MODELES: dict[str, dict] = {
    # Façon @sowvlr : 16:9, chaque mot chanté énorme dans le décor, couleurs du skin, ~40 s.
    "paroles_16_9": {
        "nom": "Paroles 16:9",
        "description": "Façon @sowvlr : 16:9, les paroles mot à mot dans le décor, couleurs du skin, ~40 s.",
        "reglages": {"format": "youtube", "duree": 40, "style": "edit", "look": "auto", "rythme": "normal",
                     "paroles": True, "hook": False, "intro_3d": False, "fps": 60, "ralenti": 0.5,
                     "textes_kills": False, "compteur": False},
        "rendu": {
            "court": True,                      # un seul edit 16:9 (pas des chapitres YouTube de 30 s)
            "rythme": 1.15,                     # plans un peu plus longs : le temps de lire les mots
            "transitions_douces": ["hblur", "fade", "hblur", "dissolve"],
            "transitions_coupees": ["whip", "zoom", "coupe", "whip", "zoom", "flash"],
            "effets_kill": ["punch", "pulse", "rgb"],
            "dosage": {"punch": 0.16, "flash": 0.14, "rgb": 6},
            "bloom": 0.45,
            "obturateur": 1.0,
        },
    },
    # Façon @abexval : intro qui montre le skin de près (inspection / plan calme, effet chaud et lumineux),
    # puis les kills avec beaucoup de flou de mouvement.
    "showcase_skin": {
        "nom": "Showcase skin",
        "description": "Façon @abexval : intro sur ton skin (inspection, halo chaud), puis les kills avec "
                       "beaucoup de flou de mouvement.",
        "reglages": {"format": "youtube", "duree": 36, "style": "edit", "look": "auto", "rythme": "normal",
                     "paroles": False, "hook": False, "intro_3d": False, "fps": 60, "ralenti": 0.45,
                     "textes_kills": False, "compteur": False},
        "rendu": {
            "court": True,
            "showcase": 2.4,                    # durée visée de l'intro showcase (s)
            "rythme": 1.0,
            "transitions_douces": ["hblur", "fade", "hblur"],
            "transitions_coupees": ["whip", "zoom", "whip", "spin", "zoom", "whip"],
            "effets_kill": ["punch", "velocity", "rgb", "pulse"],
            "velocity": 0.7,                    # part des kills en speed ramp (accéléré puis ralenti sec)
            "dosage": {"punch": 0.22, "rgb": 8, "flash": 0.16},
            "bloom": 0.55,
            "obturateur": 1.0,                  # flou de mouvement à fond sur tous les mouvements
            "flicks": 0.03,                     # flou des flicks du joueur dès les petits mouvements
        },
    },
    # Phonk : coupes très serrées sur chaque temps, secousses, glitchs, look néon.
    "phonk_rapide": {
        "nom": "Phonk rapide",
        "description": "Coupes très serrées sur chaque temps, secousses, glitchs et look néon. Vertical, ~15 s.",
        "reglages": {"format": "vertical", "duree": 15, "style": "phonk", "look": "neon", "rythme": "nerveux",
                     "paroles": False, "hook": True, "intro_3d": True, "fps": 30, "ralenti": 0.6,
                     "textes_kills": True, "compteur": False},
        "rendu": {
            "rythme": 0.8,
            "transitions_douces": ["fadewhite", "hblur", "fade"],
            "transitions_coupees": ["secousse", "whip", "glitch", "flash", "secousse", "zoom"],
            "effets_kill": ["shake", "rgb", "punch", "flash"],
            "effets_par_kill": 2,
            "dosage": {"shake_xy": 0.95, "shake_z": 0.12, "shake_d": 0.32, "rgb": 18, "punch": 0.32},
            "bloom": 0.5,
            "obturateur": 0.8,
            "flicks": 0.045,
        },
    },
    # Ciné : plans longs, ralentis, couleurs froides, grain, bandes noires 2.39:1.
    "cine": {
        "nom": "Ciné",
        "description": "Plans longs, ralentis sur les kills, look froid, grain et bandes noires. 16:9, ~30 s.",
        "reglages": {"format": "youtube", "duree": 30, "style": "cinematique", "look": "glace", "rythme": "calme",
                     "paroles": False, "hook": False, "intro_3d": False, "fps": 30, "ralenti": 0.4,
                     "textes_kills": False, "compteur": False, "arret_image": False},
        "rendu": {
            "court": True,
            "rythme": 1.1,
            "transitions_douces": ["fade", "dissolve", "fadeblack", "fade"],
            "transitions_coupees": ["coupe", "zoom", "coupe"],
            "rythme_apres": 1.5,                # plans longs après le drop (~2 s) : on voit chaque kill en entier
            "effets_kill": ["punch"],
            "ralentis": True,                   # ralenti sur chaque kill après le drop
            "dosage": {"punch": 0.1, "flash": 0.12, "shake_xy": 0.2},
            "bloom": 0.5,
            "grain": True,
            "obturateur": 0.9,
        },
    },
}

SORTIES = ("whip", "zoom", "spin", "flash")


def get(nom: str | None) -> dict | None:
    return MODELES.get(nom or "")


def rendu_de(nom: str | None) -> dict:
    m = get(nom)
    return m["rendu"] if m else {}


def pour_ui() -> list[dict]:
    """Liste envoyée à l'app (sélecteur « Modèle »)."""
    return [{"id": k, "nom": m["nom"], "description": m["description"], "reglages": m["reglages"],
             "court": bool(m["rendu"].get("court"))} for k, m in MODELES.items()]


def preremplir(o, nom: str | None = None):
    """Options avec les réglages du modèle (pour les scripts ; l'app pré-remplit elle-même)."""
    m = get(nom if nom is not None else getattr(o, "modele", ""))
    if not m:
        return o
    champs = {k: v for k, v in m["reglages"].items() if k in o.__dataclass_fields__}
    return replace(o, **champs)


def est_court_16_9(o) -> bool:
    """Format 16:9 en un seul edit (pas les chapitres YouTube) : modèles « court »."""
    return o.format == "youtube" and bool(rendu_de(getattr(o, "modele", "")).get("court"))


def regler_rendu(r, nom: str | None) -> None:
    """Halo, grain, flou de mouvement du modèle sur le Rendu."""
    rd = rendu_de(nom)
    if not rd:
        return
    r.modele = nom
    if "bloom" in rd:
        r.bloom = rd["bloom"]
    if rd.get("grain"):
        r.grain = True
    r.obturateur = rd.get("obturateur", 0)
    if rd.get("flicks"):
        r.seuil_flicks = rd["flicks"]


def dosage(r, base: dict) -> dict:
    d = rendu_de(getattr(r, "modele", "")).get("dosage")
    return {**base, **d} if d else base


def facteur_rythme(nom: str | None) -> float:
    return float(rendu_de(nom).get("rythme", 1.0))


def facteur_apres(nom: str | None) -> float:
    """Durée des plans après le drop, en plus de facteur_rythme."""
    return float(rendu_de(nom).get("rythme_apres", 1.0))


# ---------------------------------------------------------------- planification

def ajuster_segment(seg, prec, nom: str | None, rng: random.Random) -> None:
    """Transitions et effets du modèle sur un plan (après _choisir_effets et le style)."""
    rd = rendu_de(nom)
    if not rd or seg.role in ("hook", "rewind", "intro3d"):
        return
    douces = rd.get("transitions_douces")
    if douces and seg.transition:
        seg.transition = rng.choice([t for t in douces if t != (prec.transition if prec else None)] or douces)
    if seg.role == "apres" and prec is not None:
        coupees = rd.get("transitions_coupees")
        if coupees:
            t = rng.choice([t for t in coupees if t != prec.entree] or coupees)
            seg.entree = None if t == "coupe" else t
            prec.sortie = t if t in SORTIES else None
        kills = rd.get("effets_kill")
        if rd.get("ralentis"):
            seg.impact = True                     # ralenti sur chaque kill
        if kills and not seg.impact:
            n = int(rd.get("effets_par_kill", 1))
            dispo = [e for e in kills if e != "velocity" and e not in prec.effets] or [e for e in kills if e != "velocity"]
            seg.effets = rng.sample(dispo, min(n, len(dispo)))
            if "velocity" in kills and rng.random() < rd.get("velocity", 0.45) and "velocity" not in prec.effets:
                seg.effets.append("velocity")
        elif kills and seg.impact:
            garder = {"flash", "impact", "punch", "shake"} & (set(kills) | {"punch"})
            if rd.get("ralentis"):
                garder = {"punch"}
            seg.effets = [e for e in seg.effets if e in garder] or ["punch"]
    if seg.role == "drop" and rd.get("ralentis"):
        seg.effets = [e for e in seg.effets if e in ("punch", "flash")]


def plans_showcase(beats, bornes: list[int], drop: int, nom: str | None) -> list[int]:
    """Intro showcase : le 1er plan dure ~rendu.showcase secondes (on fusionne les premières coupes)."""
    duree = rendu_de(nom).get("showcase")
    if not duree:
        return bornes
    bornes = list(bornes)
    while (len(bornes) > 3 and bornes[1] != drop and bornes.index(drop) >= 3
           and float(beats[bornes[1]]) - float(beats[bornes[0]]) < duree - 0.25):
        bornes.pop(1)
    return bornes


def _images(clip: Path, debut: float, duree: float, w: int = 192, h: int = 108, fps: int = 15) -> np.ndarray | None:
    from outils import _FLAGS
    try:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, debut):.3f}", "-t", f"{duree:.3f}",
                            "-i", str(clip), "-vf", f"fps={fps},scale={w}:{h}", "-pix_fmt", "rgb24", "-f",
                            "rawvideo", "-"], capture_output=True, creationflags=_FLAGS, timeout=60)
    except Exception:
        return None
    n = len(p.stdout) // (w * h * 3)
    if n < 4:
        return None
    return np.frombuffer(p.stdout[: n * w * h * 3], np.uint8).reshape(n, h, w, 3).astype(np.float32) / 255


def _decalages(g: np.ndarray) -> np.ndarray:
    """Mouvement global de la caméra (fraction d'image par image) par corrélation de phase."""
    n, h, w = g.shape
    fen = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.rfft2((g - g.mean(axis=(1, 2), keepdims=True)) * fen)
    res = np.zeros(n)
    for i in range(1, n):
        c = F[i] * np.conj(F[i - 1])
        c /= np.abs(c) + 1e-6
        r = np.fft.irfft2(c, s=(h, w))
        y, x = np.unravel_index(int(np.argmax(r)), r.shape)
        if r[y, x] < 0.08:
            res[i] = 0.05                     # changement brutal : compté comme mouvement fort
            continue
        dx = x if x <= w // 2 else x - w
        dy = y if y <= h // 2 else y - h
        res[i] = float(np.hypot(dx / w, dy / h))
    return res


def noter_fenetres(img: np.ndarray, fps: int, duree: float) -> list[tuple[float, float]]:
    """[(score, début en s dans img)] des fenêtres de `duree` s : caméra calme, image claire, arme
    (viewmodel, en bas à droite) bien visible et colorée, bonus si l'arme bouge alors que la caméra
    est immobile (inspection)."""
    n, h, w, _ = img.shape
    g = img.mean(axis=3)
    mv = _decalages(g)
    mx, mn = img.max(axis=3), img.min(axis=3)
    sat = (mx - mn) / (mx + 1e-6) * np.clip((mx - 0.15) / 0.4, 0, 1)
    y0, y1, x0, x1 = int(h * 0.5), int(h * 0.97), int(w * 0.5), int(w * 0.97)
    cy0, cy1, cx0, cx1 = int(h * 0.2), int(h * 0.5), int(w * 0.3), int(w * 0.62)
    vm_sat = sat[:, y0:y1, x0:x1].mean(axis=(1, 2))
    gr = np.abs(np.diff(g[:, y0:y1, x0:x1], axis=2)).mean(axis=(1, 2))      # détails de l'arme
    lum = g.mean(axis=(1, 2))
    d_vm = np.r_[0, np.abs(np.diff(g[:, y0:y1, x0:x1], axis=0)).mean(axis=(1, 2))]
    d_c = np.r_[0, np.abs(np.diff(g[:, cy0:cy1, cx0:cx1], axis=0)).mean(axis=(1, 2))]
    k = max(4, int(round(duree * fps)))
    res = []
    for a in range(0, max(1, n - k + 1), max(1, fps // 3)):
        b = min(n, a + k)
        m = float(np.mean(mv[a + 1:b])) if b - a > 1 else 0.0
        l = float(lum[a:b].mean())
        if l < 0.18 or m > 0.03 or float(np.max(mv[a + 1:b], initial=0)) > 0.025:
            continue                          # trop sombre, ou la caméra tourne (le plan doit être calme)
        inspection = float(np.mean(d_vm[a + 1:b]) / (np.mean(d_c[a + 1:b]) + 0.004))
        score = (2.0 * float(vm_sat[a:b].mean()) + 6.0 * float(gr[a:b].mean())
                 + 0.5 * min(3.0, max(0.0, inspection - 1.2)) - 40.0 * m + 0.4 * min(l, 0.45))
        res.append((score, a / fps))
    return sorted(res, reverse=True)


def trouver_showcase(candidats: list, moments: list, duree: float, log=None):
    """Plan d'intro « showcase » : un moment calme AVANT le premier kill d'un clip (début de manche,
    déplacement, inspection), où l'arme est bien visible. Les temps des kills viennent de Medal.
    Retourne (moment, début dans le clip) ou None."""
    from mouvement import image_propre
    vus, essais = set(), []
    for m in candidats:
        if m.clip in vus or len(essais) >= 6:
            continue
        vus.add(m.clip)
        temps = [x.t for x in moments if x.clip == m.clip] + [t for x in moments if x.clip == m.clip
                                                              for _, t in (x.kills or [])]
        premier = min(temps + [m.t])
        fin = premier - 0.7
        debut = max(0.3, fin - 9.0)
        if fin - debut < duree + 0.3:
            continue
        fps = 15
        img = _images(m.clip, debut, fin - debut, fps=fps)
        if img is None:
            continue
        for score, t in noter_fenetres(img, fps, duree)[:3]:
            essais.append((score, m, debut + t))
    for score, m, t in sorted(essais, key=lambda e: e[0], reverse=True)[:5]:
        if image_propre(m.clip, t, duree):
            if log:
                log(f"  intro showcase : {m.clip.name} @ {t:.1f}s (score {score:.2f})")
            return m, t
    return None
