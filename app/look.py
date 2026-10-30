"""« Looks » d'étalonnage appliqués à tout le montage (même rendu sur tous les plans).

- neon / glace / chaud : looks fixes ;
- auto : la couleur dominante des plans choisis (souvent le skin de l'arme) est détectée,
  puis la saturation et la balance la font ressortir ;
- aucun : l'étalonnage du style (ETALONNAGES dans rendu.py), comme avant.

Chaque look donne aussi une couleur d'accent (hex « #RRGGBB ») : couleur_accent(r).
Le bloom (halo sur les hautes lumières), la vignette et le grain sont ajoutés par chaine_finition().
"""
from __future__ import annotations

import colorsys
import subprocess
from pathlib import Path

import numpy as np

from outils import _FLAGS

LOOKS = ("auto", "neon", "glace", "chaud", "aucun")

# Chaque look : contraste, saturation, gamma, teinte des ombres et des hautes lumières (décalage R, G, B),
# accent. Appliqué avec eq + curves (rapides) : les filtres colorbalance/huesaturation sont 5-10x plus lents.
_PRESETS = {
    "neon": {"contraste": 1.14, "saturation": 1.38, "gamma": 0.96, "ombres": (0.04, -0.03, 0.12),
             "moyens": (0.05, -0.045, 0.06), "hautes": (0.06, -0.04, 0.05), "accent": "#B85CFF"},
    "glace": {"contraste": 1.1, "saturation": 1.1, "gamma": 0.98, "ombres": (-0.05, 0.02, 0.1),
              "moyens": (-0.065, 0.0, 0.06), "hautes": (-0.03, 0.02, 0.05), "accent": "#8ADFFF"},
    "chaud": {"contraste": 1.12, "saturation": 1.3, "gamma": 0.98, "ombres": (0.04, 0.0, -0.06),
              "moyens": (0.06, 0.02, -0.06), "hautes": (0.05, 0.02, -0.08), "accent": "#FFAE45"},
}
_NOMS_COULEURS = [(0, "r"), (60, "y"), (120, "g"), (180, "c"), (240, "b"), (300, "m"), (360, "r")]


def _hex(r: float, g: float, b: float) -> str:
    return "#" + "".join(f"{int(round(max(0, min(1, c)) * 255)):02X}" for c in (r, g, b))


def _rgb(hexa: str) -> tuple[float, float, float]:
    h = hexa.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def couleur_accent(r) -> str:
    """Couleur d'accent du montage (« #RRGGBB »), à utiliser pour teindre les textes/paroles."""
    acc = getattr(r, "accent", "") or ""
    if acc:
        return acc
    look = getattr(r, "look", "aucun")
    return _PRESETS.get(look, {}).get("accent", "#FFFFFF")


def preset_depuis_teinte(teinte: float) -> dict:
    """Look « auto » : la teinte dominante (degrés 0-360) tire les hautes lumières et les tons moyens,
    les ombres partent légèrement vers la couleur opposée (contraste de couleurs, profondeur)."""
    r, g, b = colorsys.hsv_to_rgb(teinte / 360, 0.7, 1.0)
    m = (r + g + b) / 3
    chaud = teinte < 60 or teinte > 330
    # Accent chaud (skin rouge/or) : on ne teinte PAS l'image en orange (les murs le sont déjà) ;
    # base froide et neutre, l'accent ressort par contraste. Accent froid : il colore les tons.
    k_h, k_m = (0.02, 0.0) if chaud else (0.07, 0.07)
    hautes = tuple((c - m) * k_h for c in (r, g, b))
    moyens = tuple((c - m) * k_m + d for c, d in zip((r, g, b), (-0.03, 0.0, 0.03) if chaud else (0, 0, 0)))
    ombres = (-0.03, 0.005, 0.05)
    return {"contraste": 1.16, "saturation": 1.3, "gamma": 0.95, "ombres": ombres, "moyens": moyens,
            "hautes": hautes,
            "accent": _hex(*colorsys.hsv_to_rgb(teinte / 360, 0.62, 1.0)), "teinte": teinte}


def _image(clip: Path, t: float, w: int = 160, h: int = 90) -> np.ndarray | None:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(clip), "-frames:v", "1",
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        p = subprocess.run(cmd, capture_output=True, creationflags=_FLAGS, timeout=30)
    except Exception:
        return None
    if len(p.stdout) != w * h * 3:
        return None
    return np.frombuffer(p.stdout, np.uint8).reshape(h, w, 3).astype(np.float32) / 255


def teinte_dominante(images: list[np.ndarray], vue_fps: bool = True, murs_chauds: bool = True) -> float | None:
    """Teinte (degrés) des pixels colorés les plus présents, surtout vers le bas à droite (l'arme).
    None si les images sont presque grises.
    vue_fps=False (Minecraft, courses, jeux à la 3e personne) : pas d'arme à viser, on regarde le centre.
    murs_chauds=False : les teintes chaudes ternes ne sont pas des murs de map à ignorer."""
    if not images:
        return None
    hist = np.zeros(36)
    total = 0.0
    for img in images:
        h, w, _ = img.shape
        mx, mn = img.max(axis=2), img.min(axis=2)
        v = mx
        s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        d = np.maximum(mx - mn, 1e-6)
        hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
        yy, xx = np.mgrid[0:h, 0:w]
        # Zone de l'arme (bas droite) et centre de l'écran ; on ignore les bords (HUD, minimap).
        if vue_fps:
            poids_zone = (np.exp(-(((xx / w - 0.66) / 0.25) ** 2 + ((yy / h - 0.72) / 0.25) ** 2))
                          + 0.5 * np.exp(-(((xx / w - 0.5) / 0.3) ** 2 + ((yy / h - 0.5) / 0.3) ** 2)))
        else:
            poids_zone = np.exp(-(((xx / w - 0.5) / 0.32) ** 2 + ((yy / h - 0.55) / 0.35) ** 2))
        poids_zone[: int(h * 0.12)] = 0
        # Murs des maps (beiges, oranges, rouges ternes) : ignorés sauf couleur très franche.
        # Les skins et les abilities (bleu, violet, cyan, rouge vif) comptent davantage.
        chaud = (hue < 60) | (hue > 340)
        terne = chaud & ((s < 0.72) | (v < 0.45))
        froid = (hue >= 160) & (hue <= 320)
        if murs_chauds:
            poids_teinte = np.where(terne, 0.0, np.where(chaud, 0.6, np.where(froid, 1.6, 1.0)))
        else:
            # Pas de skins : la couleur du monde du jeu (herbe, ciel, carrosserie) compte telle quelle,
            # les tons ternes un peu moins.
            poids_teinte = np.where(terne, 0.5, 1.0)
        p = np.clip((s - 0.38) / 0.4, 0, 1) * np.clip((v - 0.12) / 0.3, 0, 1) * poids_zone * poids_teinte
        hist += np.bincount(((hue.astype(int) // 10) % 36).ravel(), weights=p.ravel(), minlength=36)[:36]
        total += float(poids_zone.sum())
    lisse = hist + 0.6 * (np.roll(hist, 1) + np.roll(hist, -1))
    k = int(np.argmax(lisse))
    if total <= 0 or lisse[k] / total < 0.006:
        return None
    # Moyenne pondérée autour du pic (sur le cercle).
    idx = np.array([k - 1, k, k + 1]) % 36
    ang = (idx * 10 + 5) * np.pi / 180
    c, s_ = (hist[idx] * np.cos(ang)).sum(), (hist[idx] * np.sin(ang)).sum()
    return float(np.degrees(np.arctan2(s_, c)) % 360)


def preparer(segments, r, log=print) -> None:
    """Choisit le look du montage (une seule fois : les chapitres YouTube gardent le même)."""
    look = getattr(r, "look", "aucun")
    if look == "aucun" or getattr(r, "_look_pret", None):
        return
    if look in _PRESETS:
        p = dict(_PRESETS[look])
    else:
        vus, images = set(), []
        for s in segments:
            if s.role in ("rewind", "intro3d"):
                continue
            cle = (s.moment.clip, round(s.moment.t, 1))
            if cle in vus or len(images) >= 8:
                continue
            vus.add(cle)
            img = _image(s.moment.clip, s.moment.t - 0.15)
            if img is not None:
                images.append(img)
        try:
            from jeux import profil as profil_jeu
            pj = profil_jeu(segments[0].moment.clip) if segments else None
        except Exception:
            pj = None
        teinte = teinte_dominante(images, vue_fps=pj.vue_fps if pj else True,
                                  murs_chauds=pj.murs_chauds if pj else True)
        if teinte is None:
            p = dict(_PRESETS["glace"])
            log("  look auto : pas de couleur dominante nette, look « glace »")
        else:
            p = preset_depuis_teinte(teinte)
            if pj is not None and not pj.vue_fps:
                # Jeux sans arme à l'écran (Minecraft, courses…) : la teinte dominante est celle du monde
                # (herbe, bois, ciel) ; la pousser encore donne une image jaune ou verte délavée.
                # Étalonnage doux : contraste et couleurs à peine relevés, teintes divisées par 3.
                p.update(contraste=1.08, saturation=1.08, gamma=1.0,
                         hautes=tuple(c / 3 for c in p["hautes"]), moyens=tuple(c / 3 for c in p["moyens"]),
                         ombres=(-0.015, 0.0, 0.02))
            log(f"  look auto : couleur dominante {p['accent']} (teinte {teinte:.0f}°)")
    r._look_pret = p
    try:
        if pj is not None and not pj.vue_fps and getattr(r, "bloom", 0.4) > 0.2:
            r.bloom = 0.2      # images souvent très claires (ciel, neige, night vision) : halo léger
    except NameError:
        pass
    if not getattr(r, "accent", ""):
        r.accent = p["accent"]


def _courbe(ombre: float, moyen: float, haute: float) -> str:
    pts = [(0.0, ombre * 0.15), (0.2, ombre), (0.5, moyen), (0.8, haute), (1.0, haute * 0.5)]
    return " ".join(f"{x:.2f}/{min(1.0, max(0.0, x + d)):.3f}" for x, d in pts)


def couleurs(r) -> tuple[str, str] | None:
    """(filtre eq, filtre curves) du look ; None = garder l'étalonnage du style."""
    if getattr(r, "look", "aucun") == "aucun":
        return None
    p = getattr(r, "_look_pret", None) or _PRESETS.get(r.look) or _PRESETS["glace"]
    eq = f"eq=contrast={p['contraste']:.3f}:saturation={p['saturation']:.3f}:gamma={p['gamma']:.3f}"
    o, m, h = p["ombres"], p.get("moyens", p["hautes"]), p["hautes"]
    curves = "curves=" + ":".join(f"{c}='{_courbe(o[i], m[i], h[i])}'" for i, c in enumerate("rgb"))
    # Noirs profonds : courbe générale en S (bas écrasé, hautes lumières gardées).
    curves += ":m='0/0 0.12/0.05 0.35/0.3 0.65/0.69 0.9/0.94 1/1'"
    return eq, curves


def chaine_finition(r, entree: str, sortie: str, force: float | None = None) -> str:
    """Bloom sur les hautes lumières + vignette légère + grain fin (optionnel).
    Le halo est calculé en petit (1/4), puis ajouté en YUV : écran sur la luminance, et sa couleur
    ajoutée à la chrominance (grainmerge). Retourne un morceau de graphe « [entree]…[sortie] »."""
    W, H = r.largeur, r.hauteur
    bw, bh = max(2, (W // 4) // 2 * 2), max(2, (H // 4) // 2 * 2)
    sigma = max(4, min(W, H) / 4 / 22)
    force = getattr(r, "bloom", 0.4) if force is None else force
    g = (f"[{entree}]format=yuv420p,split=2[lk_a][lk_b];"
         f"[lk_b]scale={bw}:{bh}:flags=bilinear,eq=saturation=1.7,format=gbrp,colorlevels=rimin=0.74:gimin=0.74:bimin=0.74,"
         f"gblur=sigma={sigma:.1f},format=yuv420p,scale={W}:{H}:flags=bilinear[lk_h];"
         f"[lk_a][lk_h]blend=c0_mode=screen:c1_mode=grainmerge:c2_mode=grainmerge:all_opacity={force:.2f},"
         f"vignette=PI/5.2")
    if getattr(r, "grain", False):
        g += ",noise=c0s=5:c0f=t"
    return g + f"[{sortie}]"
