"""Arrêt sur image au moment du kill : l'image se fige, le décor passe en noir et blanc assombri,
et l'ennemi (surligné en jaune ou en rouge par Valorant) ressort avec un contour lumineux.
L'ennemi est cherché près du viseur (centre de l'image)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage

from outils import _FLAGS


def _lire(clip: Path, t: float, l: int, h: int) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(clip), "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    b = subprocess.run(cmd, capture_output=True, creationflags=_FLAGS, timeout=30).stdout
    if len(b) != l * h * 3:
        return None
    return np.frombuffer(b, np.uint8).reshape(h, l, 3).astype(np.float32)


def _ecrire_png(img: np.ndarray, sortie: Path) -> bool:
    h, l, _ = img.shape
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{l}x{h}",
                        "-i", "-", str(sortie)], input=np.clip(img, 0, 255).astype(np.uint8).tobytes(),
                       capture_output=True, creationflags=_FLAGS)
    return r.returncode == 0 and sortie.exists()


def masque_ennemi(img: np.ndarray) -> np.ndarray | None:
    """Pixels du contour de l'ennemi (surbrillance jaune ou rouge de Valorant) près du viseur."""
    h, l, _ = img.shape
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    # Surbrillance jaune des ennemis (réglage par défaut de Valorant). Le rouge est ignoré : trop de
    # skins d'armes en ont.
    m = (r > 175) & (g > 165) & (b < 110) & (np.abs(r - g) < 70)
    # Zone autour du viseur (on évite l'arme en bas à droite et le HUD).
    zone = np.zeros_like(m)
    zone[int(h * 0.2):int(h * 0.64), int(l * 0.32):int(l * 0.62)] = True
    m &= zone
    if m.sum() < 25:
        return None
    etiq, n = ndimage.label(ndimage.binary_dilation(m, iterations=4))
    if n == 0:
        return None
    cy, cx = h / 2, l / 2
    meilleur, score_max = None, 0.0
    for i, (sl_y, sl_x) in enumerate(ndimage.find_objects(etiq), 1):
        bloc = m[sl_y, sl_x] & (etiq[sl_y, sl_x] == i)
        taille = float(bloc.sum())
        hb, lb = sl_y.stop - sl_y.start, sl_x.stop - sl_x.start
        # Un joueur : plus haut que large, et un contour (pas une tache pleine comme un flash de tir).
        if hb < 1.0 * lb or taille / (hb * lb) > 0.5:
            continue
        y = (sl_y.start + sl_y.stop) / 2
        x = (sl_x.start + sl_x.stop) / 2
        dist = np.hypot((y - cy) / h, (x - cx) / l)
        score = taille / (1 + 12 * dist)
        if score > score_max:
            meilleur, score_max = i, score
    if meilleur is None or score_max < 30:
        return None
    return etiq == meilleur


def image_gelee(clip: Path, t: float, largeur: int, hauteur: int, sortie: Path) -> bool:
    img = _lire(clip, t, largeur, hauteur)
    if img is None:
        return False
    try:
        from jeux import profil
        contour = profil(clip).contour_ennemi
    except Exception:
        contour = True
    # Le contour jaune n'existe que dans Valorant : ailleurs (CS2, Minecraft…) le sable ou un mur jaune
    # serait pris pour un ennemi. Arrêt sur image simple.
    m = masque_ennemi(img) if contour else None
    gris = img.mean(axis=2, keepdims=True)
    fond = (0.35 * img + 0.65 * gris) * 0.55          # décor délavé et assombri
    if m is None:
        # Pas d'ennemi trouvé : arrêt sur image simple, contraste renforcé.
        return _ecrire_png(np.clip((img - 128) * 1.15 + 128, 0, 255), sortie)
    silhouette = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=6))
    garde = ndimage.gaussian_filter(silhouette.astype(np.float32), 3)[..., None]
    res = fond * (1 - garde) + img * garde * 1.15
    # Contour lumineux blanc-doré autour de l'ennemi.
    bord = ndimage.binary_dilation(silhouette, iterations=5) & ~ndimage.binary_erosion(silhouette, iterations=1)
    lueur = ndimage.gaussian_filter(bord.astype(np.float32), 6)
    lueur = lueur / (lueur.max() + 1e-6)
    couleur = np.array([255, 236, 150], np.float32)
    res = res + lueur[..., None] * couleur * 0.9
    res[bord] = res[bord] * 0.3 + np.array([255, 250, 225]) * 0.7
    return _ecrire_png(res, sortie)
