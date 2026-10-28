"""Reconnaissance de l'arme et de l'agent d'un kill Valorant, 100 % en local (aucune donnée envoyée).

Juste après un kill, Valorant ajoute une ligne au fil des kills (en haut à droite) :
    [portrait de l'agent du tueur] [pseudo] [icône de l'arme] | [victime]
On lit cette ligne :
  - l'arme : la silhouette blanche est comparée aux icônes officielles du jeu ;
  - l'agent : le portrait est comparé aux portraits officiels du fil des kills.
Les modèles (icônes et portraits) sont dans modeles_valorant.npz, à côté de ce fichier.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from outils import _FLAGS

from chemins import RESSOURCES as ICI
MODELES = ICI / "modeles_valorant.npz"
W_REF, H_REF = 1920, 1080
SEUIL_ARME = 0.72
SEUIL_AGENT = 0.55

_modeles = None


def _charger():
    global _modeles
    if _modeles is None:
        d = np.load(MODELES)
        agents = []
        for nom, img, msk, w in zip(d["agents_noms"], d["agents_images"], d["agents_masques"], d["agents_largeurs"]):
            t = img[:, :w].astype(np.float32)
            m = msk[:, :w]
            v = t[m]
            agents.append((str(nom), m, (v - v.mean()) / (v.std() + 1e-6)))
        _modeles = {
            "armes": list(zip([str(n) for n in d["armes_noms"]], d["armes_masques"], d["armes_ratios"])),
            "agents": agents,
        }
    return _modeles


def lire_image(clip: Path, t: float) -> np.ndarray | None:
    """Une image du clip à l'instant t, en 1920×1080 RGB."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(clip), "-frames:v", "1",
           "-vf", f"scale={W_REF}:{H_REF}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        b = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=_FLAGS).stdout
    except Exception:
        return None
    if len(b) != W_REF * H_REF * 3:
        return None
    return np.frombuffer(b, np.uint8).reshape(H_REF, W_REF, 3)


def _couleurs(a: np.ndarray):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    allie = (g > 150) & (g - r > 40) & (b > 100)          # fond vert-turquoise (ton équipe)
    ennemi = (r > 170) & (r - g > 60) & (r - b > 40)      # fond rouge (équipe adverse)
    return allie, ennemi


def _lignes(img: np.ndarray) -> list[int]:
    """Haut de chaque ligne du fil des kills (de la plus ancienne à la plus récente)."""
    a = img[90:340, 1250:1905].astype(np.int16)
    allie, ennemi = _couleurs(a)
    fond = allie | ennemi
    res = []
    for k in range(6):
        y0 = 96 + 39 * k
        if fond[y0 - 90 + 3:y0 - 90 + 31, 50:650].mean() < 0.15:
            break
        res.append(y0)
    return res


def _icone_arme(img: np.ndarray, y0: int):
    """Silhouette blanche de l'arme dans la ligne : (masque, x de début) ou None."""
    x0 = 1300
    bande = img[y0 + 3:y0 + 31, x0:1905].astype(np.int16)
    blanc = bande.min(axis=2) > 205
    compte = blanc.sum(0)
    morceaux, debut, trou, fin = [], None, 0, 0
    for x, c in enumerate(compte):
        if c >= 1:
            if debut is None:
                debut = x
            fin, trou = x, 0
        elif debut is not None:
            trou += 1
            if trou > 5:
                morceaux.append((debut, fin))
                debut, trou = None, 0
    if debut is not None:
        morceaux.append((debut, fin))
    meilleur = None
    for s, e in morceaux:
        w = blanc[:, s:e + 1]
        ys = np.flatnonzero(w.any(1))
        h = ys[-1] - ys[0] + 1
        remplissage = w.sum() / (w.shape[1] * h)
        # Une arme : 16-26 px de haut, large, bien pleine (le texte est plus petit et plus creux).
        epaisse = 16 <= h <= 26 and remplissage >= 0.3
        couteau = 8 <= h < 16 and e - s + 1 >= 55 and remplissage >= 0.45   # icône du couteau, plus fine
        if (epaisse or couteau) and e - s + 1 >= 30:
            if meilleur is None or e - s > meilleur[1] - meilleur[0]:
                meilleur = (s, e, ys[0], ys[-1])
    if meilleur is None:
        return None
    s, e, ya, yb = meilleur
    return blanc[ya:yb + 1, s:e + 1], x0 + s


def _normaliser(m: np.ndarray, W: int = 96, H: int = 24) -> np.ndarray:
    """Redimensionne le masque en W×H (moyenne par zone, sans dépendance à Pillow)."""
    h, w = m.shape
    ys = np.linspace(0, h, H + 1)
    xs = np.linspace(0, w, W + 1)
    f = m.astype(np.float32)
    # Intégrale 2D : moyenne exacte de chaque case.
    ii = np.zeros((h + 1, w + 1), np.float32)
    ii[1:, 1:] = f.cumsum(0).cumsum(1)

    def interp(y, x):
        # valeur de l'intégrale en (y, x) réels, par interpolation bilinéaire
        y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
        dy, dx = y - y0, x - x0
        a = ii[y0][:, x0]
        b = ii[y0 + 1][:, x0]
        c = ii[y0][:, x0 + 1]
        d = ii[y0 + 1][:, x0 + 1]
        return (a * (1 - dy)[:, None] * (1 - dx)[None, :] + b * dy[:, None] * (1 - dx)[None, :]
                + c * (1 - dy)[:, None] * dx[None, :] + d * dy[:, None] * dx[None, :])

    I = interp(ys, xs)
    somme = I[1:, 1:] - I[:-1, 1:] - I[1:, :-1] + I[:-1, :-1]
    aire = np.outer(np.diff(ys), np.diff(xs))
    return somme / aire > 0.5


def _reconnaitre_arme(masque: np.ndarray) -> tuple[str | None, float]:
    n = _normaliser(masque)
    ratio = masque.shape[1] / max(1, masque.shape[0])
    meilleur = (None, 0.0)
    for nom, ref, r_ref in _charger()["armes"]:
        if not 0.7 <= ratio / r_ref <= 1.35:
            continue
        s = float((n & ref).sum() / ((n | ref).sum() + 1e-9))
        if s > meilleur[1]:
            meilleur = (nom, s)
    return meilleur if meilleur[1] >= SEUIL_ARME else (None, meilleur[1])


def _reconnaitre_agent(img: np.ndarray, y0: int, x_arme: int) -> tuple[str | None, float]:
    """Le portrait du tueur est à gauche du pseudo, lui-même à gauche de l'arme."""
    from numpy.lib.stride_tricks import sliding_window_view
    zone = img[y0:y0 + 34, max(0, x_arme - 330):x_arme].astype(np.float32)
    resultats = []
    for nom, m, tv in _charger()["agents"]:
        h, w = m.shape
        if zone.shape[1] <= w + 2:
            continue
        # Toutes les positions d'un coup (pas de 2 px) : corrélation normalisée avec le portrait.
        fen = sliding_window_view(zone, (h, w, 3))[0, ::2, 0]          # (positions, h, w, 3)
        p = fen[:, m].reshape(fen.shape[0], -1)
        p = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-6)
        s = p @ tv.reshape(-1) / p.shape[1]
        i = int(np.argmax(s))
        resultats.append((float(s[i]), nom, 2 * i + w))     # (score, agent, bord droit du portrait)
    if not resultats:
        return None, 0.0
    # Un petit portrait d'assistance (coéquipier) peut apparaître plus à gauche : le tueur est le
    # portrait le plus à droite parmi les bons scores.
    haut = max(r[0] for r in resultats)
    bons = [r for r in resultats if r[0] >= haut - 0.06]
    score, nom, _ = max(bons, key=lambda r: r[2])
    meilleur = (nom, score)
    return meilleur if meilleur[1] >= SEUIL_AGENT else (None, meilleur[1])


def analyser_image(img: np.ndarray, avec_agent: bool = True) -> dict:
    """{"arme": nom ou None, "agent": nom ou None} pour la ligne de kill la plus récente de TON équipe."""
    res = {"arme": None, "agent": None}
    if img is None or not MODELES.exists():
        return res
    lignes = _lignes(img)
    for y0 in reversed(lignes):          # la plus récente est en bas
        trouve = _icone_arme(img, y0)
        if trouve is None:
            continue
        masque, x = trouve
        a = img[y0 + 3:y0 + 31, max(1250, x - 200):x].astype(np.int16)
        allie, ennemi = _couleurs(a)
        if ennemi.mean() > allie.mean():   # kill de l'équipe adverse : pas le tien
            continue
        # TES kills ont un liseré jaune-vert à gauche (ceux de tes coéquipiers non).
        z = img[y0 - 2:y0 + 36, max(0, x - 360):x].astype(np.int16)
        jaune = ((z[..., 0] > 190) & (z[..., 1] > 190) & (z[..., 2] < 140)).mean()
        if jaune < 0.008:
            continue
        res["arme"], _ = _reconnaitre_arme(masque)
        if avec_agent:
            res["agent"], _ = _reconnaitre_agent(img, y0, x)
        break
    return res


def analyser_kill(clip: Path, t: float, avec_agent: bool = True) -> dict:
    """Lit le fil des kills 0,4 s après le kill (le temps que la ligne apparaisse)."""
    return analyser_image(lire_image(clip, t + 0.4), avec_agent)
