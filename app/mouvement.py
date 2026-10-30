"""Mouvement : speed ramps lissés, flou de mouvement réel et images fluides (60 i/s).

Speed ramps : un Morceau peut avoir une « rampe » = liste de points (u, v) où u va de 0 à 1 sur la
durée du morceau à l'écran et v est la vitesse (1 = normal). Entre deux points la vitesse suit une
courbe douce (smoothstep). Morceau.vitesse vaut alors la vitesse MOYENNE, donc la durée source
(images / fps * vitesse) reste juste pour tout le reste du code.
  - image : setpts avec une courbe temps-source -> temps-écran (affine par petits bouts) ;
  - son   : rééchantillonné en Python le long de la même courbe (comme une bande qui ralentit),
            donc toujours calé sur l'image.

Flou de mouvement : pendant les mouvements de caméra rapides (whip, zoom, punch, secousse), le
zoompan est calculé N fois par image (sous-images), puis moyenné (tmix) : c'est un vrai flou
d'obturateur, directionnel pour un whip, radial pour un zoom.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from outils import _FLAGS

SR = 48000


def _lisse(x: np.ndarray) -> np.ndarray:
    return x * x * (3 - 2 * x)


def profil(rampe, images: int) -> np.ndarray:
    """Vitesse de chaque image de sortie (au milieu de l'image)."""
    pts = sorted((float(u), float(v)) for u, v in rampe)
    u = (np.arange(images) + 0.5) / max(1, images)
    us = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    v = np.empty(images)
    for k, x in enumerate(u):
        if x <= us[0]:
            v[k] = vs[0]
        elif x >= us[-1]:
            v[k] = vs[-1]
        else:
            i = int(np.searchsorted(us, x)) - 1
            a = (x - us[i]) / max(1e-9, us[i + 1] - us[i])
            v[k] = vs[i] + (vs[i + 1] - vs[i]) * _lisse(a)
    return np.maximum(v, 0.05)


def temps_source(rampe, images: int, fps: int) -> np.ndarray:
    """s[k] = temps source (depuis le début du morceau) à l'image de sortie k (k = 0..images)."""
    v = profil(rampe, images)
    return np.concatenate([[0.0], np.cumsum(v) / fps])


def vitesse_moyenne(rampe, images: int, fps: int) -> float:
    s = temps_source(rampe, images, fps)
    return float(s[-1] / max(1e-9, images / fps))


def vers_sortie(rampe, images: int, fps: int, s: float) -> float:
    """Temps à l'écran (depuis le début du morceau) où l'on voit l'instant source s."""
    src = temps_source(rampe, images, fps)
    return float(np.interp(s, src, np.arange(images + 1) / fps))


def filtre_setpts(rampe, images: int, fps: int) -> str:
    """setpts : temps source T (à partir de 0) -> temps écran, affine par morceaux (sans if imbriqués)."""
    src = temps_source(rampe, images, fps)
    tau = np.arange(images + 1) / fps
    pas = max(1, images // 40)
    idx = list(range(0, images + 1, pas))
    if idx[-1] != images:
        idx.append(images)
    termes = []
    for a, b in zip(idx[:-1], idx[1:]):
        ds = src[b] - src[a]
        if ds <= 1e-9:
            continue
        pente = (tau[b] - tau[a]) / ds
        termes.append(f"{pente:.5f}*clip(T-{src[a]:.5f},0,{ds:.5f})")
    # Après la fin prévue (marge du décodage) : même pente que le dernier bout.
    pente_fin = (tau[idx[-1]] - tau[idx[-2]]) / max(1e-9, src[idx[-1]] - src[idx[-2]]) if len(idx) > 1 else 1
    termes.append(f"{pente_fin:.5f}*max(0,T-{src[-1]:.5f})")
    return "setpts='(" + "+".join(termes) + ")/TB'"


def filtre_flou_vitesse(rampe, images: int, fps: int) -> str:
    """Pendant les passages accélérés (> 1,3x), on mélange les images sautées : flou de mouvement
    proportionnel à la vitesse. Se place après setpts (t = temps à l'écran)."""
    v = profil(rampe, images)
    rapide = v > 1.3
    if not rapide.any():
        return ""
    k = np.flatnonzero(rapide)
    a, b = k[0] / fps, (k[-1] + 1) / fps
    n = int(min(4, max(2, round(float(v[rapide].mean())))))
    return f"tmix=frames={n}:enable='between(t,{a:.3f},{b:.3f})'"


_cache_fps: dict[str, float] = {}


def fps_source(clip: Path) -> float:
    cle = str(clip)
    if cle not in _cache_fps:
        val = 30.0
        try:
            p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=avg_frame_rate,r_frame_rate", "-of", "json", cle],
                               capture_output=True, creationflags=_FLAGS, timeout=20)
            st = json.loads(p.stdout or b"{}").get("streams", [{}])[0]
            for k in ("avg_frame_rate", "r_frame_rate"):
                n, _, d = str(st.get(k, "0/1")).partition("/")
                if float(d or 1) > 0 and float(n) > 0:
                    val = float(n) / float(d or 1)
                    break
        except Exception:
            pass
        _cache_fps[cle] = val
    return _cache_fps[cle]


def conversion_fps(clip: Path, fps: int, vitesse_min: float = 1.0) -> str:
    """fps=… (simple) ou framerate=… (images intermédiaires fondues) quand la source n'a pas assez
    d'images : 30 i/s affiché en 60, ou ralenti. Évite les images répétées (saccades)."""
    effectif = fps_source(clip) / max(0.05, vitesse_min)
    if effectif < fps * 0.95:
        return f"framerate=fps={fps}:interp_start=0:interp_end=255:scene=100"
    return f"fps={fps}"


def audio_rampe(clip: Path, piste, debut: float, rampe, images: int, fps: int, sortie: Path) -> Path | None:
    """Son du jeu le long de la courbe de vitesse (wav stéréo 48 kHz), None si pas de son."""
    import soundfile as sf
    src = temps_source(rampe, images, fps)
    duree_src = float(src[-1]) + 0.1
    carte = f"0:{piste}" if piste is not None else "0:a:0"
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, debut):.3f}", "-t", f"{duree_src:.3f}",
                        "-i", str(clip), "-map", carte, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, creationflags=_FLAGS)
    y = np.frombuffer(p.stdout, np.float32)
    if not len(y):
        return None
    y = y[: len(y) // 2 * 2].reshape(-1, 2)
    n = int(round(images / fps * SR))
    tau = np.arange(n) / SR
    s = np.interp(tau, np.arange(images + 1) / fps, src) * SR
    s = np.clip(s, 0, len(y) - 1)
    out = np.stack([np.interp(s, np.arange(len(y)), y[:, c]) for c in range(2)], axis=1)
    # Un ralenti fort assourdit un peu le son (comme au cinéma) : filtre passe-bas doux.
    v = np.interp(tau, (np.arange(images) + 0.5) / fps, profil(rampe, images))
    if (v < 0.8).any():
        from scipy.signal import butter, lfilter
        b, a = butter(2, 1800 / (SR / 2))
        sourd = lfilter(b, a, out, axis=0)
        m = np.clip((0.9 - v) / 0.5, 0, 0.85)[:, None]
        out = out * (1 - m) + sourd * m
    sf.write(str(sortie), out.astype(np.float32), SR, subtype="PCM_16")
    return sortie


# ---------------------------------------------------------------- flou de mouvement (caméra virtuelle)

def fusionner(fenetres, total: int) -> list[tuple[int, int, float]]:
    """Fenêtres (début, fin, obturateur 0-1) triées et fusionnées (obturateur le plus grand)."""
    res = []
    for f in sorted((max(0, f[0]), min(total, f[1]), f[2] if len(f) > 2 else 1.0) for f in fenetres):
        a, b, o = f
        if b - a < 1:
            continue
        if res and a <= res[-1][1] + 2:
            res[-1] = (res[-1][0], max(res[-1][1], b), max(res[-1][2], o))
        else:
            res.append((a, b, o))
    return res


def graphe_camera(entree: str, sortie: str, camera: list[str], zoompan: str, fenetres, total: int,
                  fps: int, sous_images: int) -> str:
    """[entree] -> rotation + zoompan -> [sortie]. Dans les fenêtres (en images), zoompan est calculé
    sous_images fois par image puis moyenné : vrai flou de mouvement. Ailleurs : rendu normal.
    zoompan contient « {FPS} » à la place de sa cadence de sortie."""
    cam = ",".join(camera + [""]) if camera else ""
    fen = fusionner(fenetres, total) if sous_images > 1 else []
    if not fen:
        return f"[{entree}]{cam}{zoompan.replace('{FPS}', str(fps))}[{sortie}]"
    # Découpe : [0, a1) normal, [a1, b1) flou, [b1, a2) normal, …
    parts, pos = [], 0
    for a, b, o in fen:
        if a > pos:
            parts.append((pos, a, 0))
        parts.append((a, b, o))
        pos = b
    if pos < total:
        parts.append((pos, total, 0))
    n = len(parts)
    g = [f"[{entree}]split={n}" + "".join(f"[mv{i}]" for i in range(n))]
    N = sous_images
    for i, (a, b, obtu) in enumerate(parts):
        nb = b - a
        tr = f"[mv{i}]trim=start_frame={a}:end_frame={b},"
        if obtu:
            k = max(2, int(round(N * obtu)))   # obturateur : part des sous-images moyennées
            g.append(tr + f"fps={fps * N},{cam}{zoompan.replace('{FPS}', str(fps * N))},"
                     f"tmix=frames={k},select='eq(mod(n\\,{N})\\,{N - 1})',setpts=N/{fps}/TB,"
                     f"tpad=stop_mode=clone:stop=2,trim=end_frame={nb}[mo{i}]")
        else:
            g.append(tr + f"{cam}{zoompan.replace('{FPS}', str(fps))},setpts=PTS-STARTPTS,"
                     f"tpad=stop_mode=clone:stop=2,trim=end_frame={nb},setpts=PTS-STARTPTS[mo{i}]")
    g.append("".join(f"[mo{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0,setpts=N/{fps}/TB[{sortie}]")
    return ";".join(g)


# ---------------------------------------------------------------- flou des mouvements de caméra du joueur

def _decalages(clip: Path, debut: float, duree: float, w: int = 160, h: int = 90) -> tuple[np.ndarray, float]:
    """Déplacement global (dx, dy en fraction de l'image) entre images successives, par corrélation de
    phase sur des images minuscules. Retourne (tableau (n, 2), fps de la source)."""
    fps = fps_source(clip)
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, debut):.3f}", "-t", f"{duree:.3f}", "-i", str(clip),
                        "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                       capture_output=True, creationflags=_FLAGS)
    n = len(p.stdout) // (w * h)
    if n < 3:
        return np.zeros((0, 2)), fps
    im = np.frombuffer(p.stdout[: n * w * h], np.uint8).reshape(n, h, w).astype(np.float32)
    _decalages.luminance = float(im.mean() / 255)
    fen = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.fft.rfft2((im - im.mean(axis=(1, 2), keepdims=True)) * fen)
    res = np.zeros((n, 2))
    for i in range(1, n):
        c = F[i] * np.conj(F[i - 1])
        c /= np.abs(c) + 1e-6
        r = np.fft.irfft2(c, s=(h, w))
        y, x = np.unravel_index(int(np.argmax(r)), r.shape)
        if r[y, x] < 0.08:          # corrélation trop faible (changement de scène, flash) : on ignore
            continue
        dx = x if x <= w // 2 else x - w
        dy = y if y <= h // 2 else y - h
        res[i] = (dx / w, dy / h)
    return res, fps


def filtre_flou_camera(clip: Path, debut: float, duree_src: float, largeur_src: int, seuil: float = 0.045) -> str:
    """Flou directionnel (dblur) pendant les mouvements de caméra rapides du joueur (flicks), dans la
    direction du mouvement et proportionnel à sa vitesse. Temps = temps source depuis le début du
    morceau (à placer avant le changement de vitesse). "" si rien de rapide."""
    try:
        d, fps = _decalages(clip, debut, duree_src + 0.05)
    except Exception:
        return ""
    if not len(d):
        return ""
    vit = np.hypot(d[:, 0], d[:, 1])
    rapide = vit > seuil
    filtres, i = [], 0
    while i < len(vit):
        if not rapide[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(vit) and rapide[j + 1]:
            j += 1
        dx, dy = d[i:j + 1].mean(axis=0)
        v = float(vit[i:j + 1].mean())
        angle = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        rayon = int(min(40, max(3, v * largeur_src * 0.15)))   # dblur étale plus que son rayon : ~ obturateur 120°
        a, b = (i - 0.5) / fps, (j + 0.5) / fps
        filtres.append(f"dblur=angle={angle:.0f}:radius={rayon}:enable='between(t,{a:.3f},{b:.3f})'")
        i = j + 1
    return ",".join(filtres[:6])


def activite(clip: Path, debut: float, duree: float) -> tuple[float, float]:
    """(mouvement moyen de la caméra en fraction d'image par image, luminosité moyenne 0-1) d'une
    fenêtre du clip. Sert à refuser les plans morts (caméra immobile) ou trop sombres."""
    try:
        d, _ = _decalages(clip, debut, max(0.3, duree), w=96, h=54)
    except Exception:
        return 1.0, 0.5
    if not len(d):
        return 1.0, 0.5
    v = np.hypot(d[:, 0], d[:, 1])
    # Moyenne des 70 % d'images les plus mobiles (un plan qui bouge un peu tout le temps compte).
    return float(np.mean(np.sort(v)[int(len(v) * 0.3):])), getattr(_decalages, "luminance", 0.5)


def _lignes_de_tableau(f: np.ndarray) -> int:
    """Nombre de rangées qui contiennent un trait horizontal continu sur au moins 28 % de la largeur."""
    h, w = f.shape
    e = np.abs(np.diff(f[int(h * .1):int(h * .85)], axis=0)) > 6
    d = np.diff(np.pad(e.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    n = 0
    for debuts, fins in ((np.flatnonzero(r == 1), np.flatnonzero(r == -1)) for r in d):
        if len(debuts) and (fins - debuts).max() >= 0.28 * w:
            n += 1
    return n


def _menu(im: np.ndarray, f: np.ndarray, jeu: str) -> bool:
    """Image de menu selon le jeu (im : RGB 0-1, f : luminance 0-255)."""
    if jeu == "valorant":
        # Menus de Valorant (TAB, fin de manche, achat) : beaucoup de longues lignes droites
        # horizontales, où qu'elles soient dans l'image (rangées du tableau).
        return _lignes_de_tableau(f) >= 5
    h, w = f.shape
    mx, mn = im.max(axis=2), im.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    lum = f / 255
    # Écran voilé de gris sombre (menu d'achat de CS2, caméra de mort, écran de pause) : presque toute
    # l'image grise et sombre. Une scène sombre du jeu (tunnel, cave) garde ses couleurs.
    voile = float(((lum > 0.03) & (lum < 0.3) & (sat < 0.3)).mean())
    if jeu == "cs2":
        if voile > 0.5 and float(sat.mean()) < 0.16:
            return True
        # Tableau des scores (TAB) et panneau de fin de manche : rangées de lignes droites dans un
        # panneau translucide plus sombre que le reste de l'image (les planches et les escaliers des
        # maps ont des lignes aussi, mais ne sont pas plus sombres au centre).
        centre = float(lum[int(h * .3):int(h * .7), int(w * .3):int(w * .7)].mean())
        return _lignes_de_tableau(f) >= 5 and centre < 0.82 * float(lum.mean())
    # Autres jeux : on ne connaît pas leurs menus. Un voile gris très net (pause, écran assombri), ou
    # un écran couvert de panneaux et de rangées (inventaire, réglages : 11 à 29 longues lignes dans
    # Minecraft, 0 à 3 en jeu ; seuil haut car les murs de planches en ont quelques-unes).
    return (voile > 0.75 and float(sat.mean()) < 0.1) or _lignes_de_tableau(f) >= 10


def image_propre(clip: Path, debut: float, duree: float) -> bool:
    """Faux si la fenêtre montre un menu (tableau des scores TAB, achat d'armes), une image unie,
    brûlée ou sans détails, ou un écran bouché par une seule couleur (fumée, ultime, flash collés à
    la caméra). Ces plans gâchent un edit."""
    w, h = 320, 180
    try:
        p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, debut):.3f}", "-t", f"{max(0.3, duree):.3f}",
                            "-i", str(clip), "-vf", f"fps=8,scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                           capture_output=True, creationflags=_FLAGS, timeout=60)
    except Exception:
        return True
    n = len(p.stdout) // (w * h * 3)
    if n < 1:
        return True
    rgb = np.frombuffer(p.stdout[: n * w * h * 3], np.uint8).reshape(n, h, w, 3).astype(np.float32) / 255
    try:
        from jeux import profil
        jeu = profil(clip).menus
    except Exception:
        jeu = "valorant"
    menus = sales = 0
    y0, y1, x0, x1 = int(h * .25), int(h * .85), int(w * .22), int(w * .78)
    for im in rgb:
        f = im.mean(axis=2) * 255
        c = f[y0:y1, x0:x1]
        if _menu(im, f, jeu):
            menus += 1
            continue
        g = np.hypot(np.diff(c, axis=0)[:, :-1], np.diff(c, axis=1)[:-1])
        if f.std() < 16 or f.mean() > 0.85 * 255 or (g > 10).mean() < 0.06:
            sales += 1
            continue
        # Écran bouché : la moitié du centre a la même teinte vive.
        z = im[y0:y1, x0:x1]
        mx, mn = z.max(axis=2), z.min(axis=2)
        sat = (mx - mn) / (mx + 1e-6)
        vif = (sat > 0.5) & (mx > 0.45)
        # Hors FPS (Minecraft, courses…), un grand pré vert ou un ciel bleu sont normaux : seuil plus haut.
        if vif.mean() > (0.45 if jeu in ("valorant", "cs2") else 0.7):
            r, gg, b = z[..., 0], z[..., 1], z[..., 2]
            teinte = (np.degrees(np.arctan2(np.sqrt(3) * (gg - b), 2 * r - gg - b)) + 360) % 360
            hist, _ = np.histogram(teinte[vif], bins=12, range=(0, 360))
            if (hist.max() + np.sort(hist)[-2]) / max(1, vif.sum()) * vif.mean() > 0.42:
                sales += 1
    return menus == 0 and sales / n < 0.25
