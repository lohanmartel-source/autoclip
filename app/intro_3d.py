"""Intro 3D : l'image du kill d'accroche se fige, la caméra recule et on découvre que ce clip n'est
qu'une carte parmi d'autres cartes de clips qui flottent en profondeur. Puis la caméra fonce vers
la carte du plan suivant, qui finit par remplir l'écran : on coupe sur la suite du montage.

Mini moteur de rendu logiciel en numpy (pas de navigateur, pas de WebGL, pas de PIL) :
  - chaque carte est un quad texturé, projeté en perspective ;
  - pour chaque pixel de la boîte englobante du quad, l'homographie inverse donne (u, v) dans la
    texture, échantillonnée en bilinéaire dans une pyramide de mipmaps (pas de scintillement) ;
  - coins arrondis, bordure lime et anticrénelage par une fonction de distance signée ;
  - cartes triées de la plus lointaine à la plus proche (algorithme du peintre) ;
  - flou de mouvement bon marché : moyenne de 2 ou 3 sous-images quand la caméra va vite.
    Ces images-là sont rendues en demi-résolution puis agrandies : le flou cache la différence,
    et c'est ce qui tient le rendu sous quelques secondes.
Les images sont envoyées brutes (rgb24) à FFmpeg, qui encode le segment comme les autres plans.
"""
from __future__ import annotations

import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from outils import _FLAGS, args_encodage, est_gpu, signaler_echec_gpu

LIME = np.array([198, 255, 61], np.float32)        # #c6ff3d
FOND = np.array([12, 14, 11], np.float32)          # #0c0e0b
FOV = math.radians(40.0)                           # champ vertical de la caméra
TAILLE_CARTE = (480, 270)                          # images des cartes (16:9), lues petites : rapide
N_MAX = 10                                         # sous-images max (flou de mouvement)
_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


# --------------------------------------------------------------------------------------------
# Lecture des images (même modèle que arret_image._lire : FFmpeg → rawvideo rgb24)

def _lire(video: Path, t: float | None, l: int, h: int, fin: bool = False,
          cle: bool = False, tout: bool = False) -> np.ndarray | None:
    """Une image de la vidéo, mise à l'échelle l×h (recadrée au centre si le format diffère).
    fin=True : la toute dernière image du fichier. cle=True : la première image clé à partir de t
    (pas de décodage image par image : ~4× plus rapide, assez précis pour une petite carte du nuage)."""
    cmd = ["ffmpeg", "-v", "error"]
    if cle:
        cmd += ["-skip_frame", "nokey"]
    if fin and not tout:
        cmd += ["-sseof", "-0.5"]
    elif t:
        cmd += ["-ss", f"{max(0.0, t):.3f}"]
    cmd += ["-i", str(video)]
    if not fin:
        cmd += ["-frames:v", "1"]
    cmd += ["-vf", f"scale={l}:{h}:force_original_aspect_ratio=increase:flags=bicubic,crop={l}:{h}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    try:
        b = subprocess.run(cmd, capture_output=True, creationflags=_FLAGS, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    n = l * h * 3
    if len(b) < n:
        return None
    b = b[len(b) - n:] if fin else b[:n]           # fin : on garde la dernière image décodée
    return np.frombuffer(b, np.uint8).reshape(h, l, 3)


def derniere_image(video: Path, l: int, h: int) -> np.ndarray | None:
    img = _lire(video, None, l, h, fin=True)
    if img is None:   # image plus courte que le son (clip qui finit tôt) : on décode tout le plan
        img = _lire(video, None, l, h, fin=True, tout=True)
    return img


def premiere_image(video: Path, l: int, h: int) -> np.ndarray | None:
    return _lire(video, 0.0, l, h)


def images_cartes(sources: list[tuple[Path, float]], travailleurs: int = 4) -> list[np.ndarray]:
    """Une image 480×270 par (clip, instant), lues en parallèle. Les échecs sont ignorés."""
    l, h = TAILLE_CARTE
    with ThreadPoolExecutor(max_workers=max(1, travailleurs)) as ex:
        # Image clé d'abord (rapide) ; s'il n'y en a pas après t, lecture exacte.
        res = list(ex.map(lambda s: _lire(Path(s[0]), s[1], l, h, cle=True), sources))
        res = list(ex.map(lambda rs: rs[0] if rs[0] is not None else _lire(Path(rs[1][0]), rs[1][1], l, h),
                          zip(res, sources)))
    return [r for r in res if r is not None]


# --------------------------------------------------------------------------------------------
# Textures et échantillonnage

class _Texture:
    """Pyramide de mipmaps (float32, bord dupliqué d'un pixel pour la lecture bilinéaire).
    sauter=1 : pas de niveau pleine résolution (images plein écran : seules leurs versions réduites
    servent, le gel utilise l'image d'origine directement)."""

    def __init__(self, img: np.ndarray, sauter: int = 0):
        self.largeur, self.hauteur = img.shape[1], img.shape[0]
        self.niveaux = [None] * sauter
        x = img.astype(np.float32)
        for _ in range(sauter):
            h2, l2 = x.shape[0] // 2, x.shape[1] // 2
            x = x[:h2 * 2, :l2 * 2].reshape(h2, 2, l2, 2, 3).mean(axis=(1, 3))
        while True:
            h, l = x.shape[:2]
            p = np.empty((h + 1, l + 1, 3), np.float32)
            p[:h, :l] = x
            p[h, :l] = x[-1]
            p[:, l] = p[:, l - 1]
            self.niveaux.append((p.reshape(-1, 3), l, h))
            if min(h, l) < 24:
                break
            h2, l2 = h // 2, l // 2
            x = x[:h2 * 2, :l2 * 2].reshape(h2, 2, l2, 2, 3).mean(axis=(1, 3))

    def echantillonner(self, u: np.ndarray, v: np.ndarray, niveau: int, proche: bool = False) -> np.ndarray:
        """Couleurs aux coordonnées (u, v) ∈ [0, 1]. proche=True : plus proche voisin (sous-images de
        flou de mouvement, où la différence ne se voit pas), sinon bilinéaire."""
        niveau = min(niveau, len(self.niveaux) - 1)
        while self.niveaux[niveau] is None:
            niveau += 1
        plat, l, h = self.niveaux[niveau]
        if proche:
            x = (u * np.float32(l)).astype(np.int32)
            y = (v * np.float32(h)).astype(np.int32)
            np.clip(x, 0, l - 1, out=x)
            np.clip(y, 0, h - 1, out=y)
            y *= l + 1
            y += x
            return np.take(plat, y, axis=0)
        x = u * np.float32(l) - np.float32(0.5)
        y = v * np.float32(h) - np.float32(0.5)
        np.clip(x, 0, l - 1.001, out=x)
        np.clip(y, 0, h - 1.001, out=y)
        x0 = x.astype(np.int32)
        y0 = y.astype(np.int32)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        i = y0 * (l + 1) + x0
        c00 = np.take(plat, i, axis=0)
        c01 = np.take(plat, i + 1, axis=0)
        i += l + 1
        c10 = np.take(plat, i, axis=0)
        c11 = np.take(plat, i + 1, axis=0)
        c01 -= c00
        c01 *= fx
        c00 += c01
        c11 -= c10
        c11 *= fx
        c10 += c11
        c10 -= c00
        c10 *= fy
        c00 += c10
        return c00


def _agrandir(a: np.ndarray) -> np.ndarray:
    """×2 en bilinéaire simple, sur des octets (images de flou de mouvement rendues en basse
    résolution). Calcul en entiers 16 bits : deux fois moins de mémoire à brasser qu'en float32."""
    h, w, _ = a.shape
    a = a.astype(np.uint16)
    b = np.empty((h, 2 * w, 3), np.uint16)
    b[:, 0::2] = a
    m = b[:, 1:-1:2]
    np.add(a[:, :-1], a[:, 1:], out=m)
    m >>= 1
    b[:, -1] = a[:, -1]
    c = np.empty((2 * h, 2 * w, 3), np.uint8)
    c[0::2] = b
    m = b[:-1] + b[1:]
    m >>= 1
    c[1:-1:2] = m
    c[-1] = b[-1]
    return c


# --------------------------------------------------------------------------------------------
# Scène

@dataclass
class _Carte:
    tex: _Texture
    centre: np.ndarray        # position de base (monde)
    largeur: float
    hauteur: float
    rot: tuple = (0.0, 0.0, 0.0)   # tangage, lacet, roulis (rad)
    phase: float = 0.0
    vitesse: float = 0.5
    retard: float = 0.0       # apparition (s)
    lum: float = 0.8
    sat: float = 0.85


def _rotation(rx: float, ry: float, rz: float) -> list:
    """Ry · Rx · Rz (listes Python : appelée des milliers de fois par rendu)."""
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    # Rx · Rz
    a = [[cz, -sz, 0.0], [cx * sz, cx * cz, -sx], [sx * sz, sx * cz, cx]]
    return [[cy * a[0][j] + sy * a[2][j] for j in range(3)], a[1],
            [-sy * a[0][j] + cy * a[2][j] for j in range(3)]]


def _inverse3(m: list) -> np.ndarray | None:
    """Inverse d'une matrice 3×3 (comatrice), en float32 ; None si elle est singulière."""
    (a, b, c), (d, e, f), (g, h, i) = m
    A, B, C = e * i - f * h, f * g - d * i, d * h - e * g
    det = a * A + b * B + c * C
    if abs(det) < 1e-12:
        return None
    return np.array([[A, c * h - b * i, b * f - c * e],
                     [B, a * i - c * g, c * d - a * f],
                     [C, b * g - a * h, a * e - b * d]], np.float32) / np.float32(det)


def _regarder(pos: np.ndarray, cible: np.ndarray, roulis: float) -> np.ndarray:
    """Matrice monde → caméra (lignes = axes droite, haut, arrière de la caméra)."""
    avant = cible - pos
    avant = avant / np.linalg.norm(avant)
    droite = np.cross(avant, [0.0, 1.0, 0.0])
    droite /= np.linalg.norm(droite)
    haut = np.cross(droite, avant)
    c, s = math.cos(roulis), math.sin(roulis)
    droite, haut = c * droite + s * haut, -s * droite + c * haut
    return np.stack([droite, haut, -avant])


def _lisse(x: float) -> float:          # ease-in-out (smootherstep)
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (6 * x - 15) + 10)


def _sortie(x: float, p: float = 3.0) -> float:   # ease-out
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** p


def _lin(t: float, a: float, b: float) -> float:
    return min(1.0, max(0.0, (t - a) / (b - a))) if b > a else float(t >= b)


def phases(duree: float) -> dict:
    """Moments clés de l'intro (s, depuis son début) : utilisés aussi pour placer les whooshs."""
    poussee = min(0.5, max(0.34, 0.3 * duree))
    pause = 0.13 * duree             # la caméra flotte : le temps de lire le nuage de clips
    return {"gel": 0.0, "recul": 0.1, "fin_recul": duree - poussee - pause, "poussee": duree - poussee,
            "fin": duree}


class Scene:
    def __init__(self, depart: np.ndarray, cartes: list[np.ndarray], cible: np.ndarray,
                 largeur: int, hauteur: int, duree: float, graine: int = 7):
        self.W, self.H, self.D = largeur, hauteur, duree
        self.ph = phases(duree)
        self.f = (hauteur / 2) / math.tan(FOV / 2)           # focale en pixels (pleine résolution)
        aspect = largeur / hauteur
        self.aspect = aspect
        # Carte du hook : 2 unités de haut, au format de l'écran ; la caméra la voit plein écran.
        self.hc, self.wc = 2.0, 2.0 * aspect
        self.D0 = 1.0 / math.tan(FOV / 2)
        base = min(self.wc, self.hc)
        rng = np.random.default_rng(graine)
        self.depart = depart.astype(np.float32)
        self.hook = _Carte(_Texture(depart, sauter=1), np.zeros(3), self.wc, self.hc, lum=1.0, sat=1.0)

        # Pose de la caméra à la fin du recul : le hook occupe ~40 % de la hauteur de l'écran.
        recul = self.D0 / (0.40 if aspect < 1 else 0.46)
        self.c1 = np.array([-0.30 * base, 0.22 * base, recul])
        self.v1 = np.array([0.06 * base, -0.05 * base, -0.6])
        self.roulis1 = math.radians(-3.5)
        M1 = _regarder(self.c1, self.v1, self.roulis1)

        def vers_monde(nx, ny, dist):
            """Point vu en (nx, ny) (coordonnées normalisées -1..1) à la distance dist de la caméra 1."""
            d = np.array([nx * math.tan(FOV / 2) * aspect, ny * math.tan(FOV / 2), -1.0])
            return self.c1 + M1.T @ (d * dist)

        # Carte cible (le plan suivant) : au format de l'écran, un peu derrière le hook, en bas à droite.
        pos_cible = (0.62, -0.58) if aspect < 1 else (0.66, -0.52)
        dist_cible = np.linalg.norm(self.c1 - self.v1) * 1.12
        ht = self.hc * 0.52
        self.cible = _Carte(_Texture(cible, sauter=1), vers_monde(*pos_cible, dist_cible), ht * aspect, ht,
                            rot=(0.0, -0.32, 0.05), lum=1.0, sat=1.0)
        self.Dt = (ht / 2) / math.tan(FOV / 2)

        # Nuage de cartes, placé dans le champ de la caméra 1 (positions tirées sans trop se chevaucher).
        self.cartes: list[_Carte] = []
        textures = [_Texture(c) for c in cartes] or [self.hook.tex]
        n = 30
        cw = 0.6 * base
        occupe = [(0.0, 0.0, 0.44 if aspect < 1 else 0.40, 0.42), (pos_cible[0], pos_cible[1], 0.17, 0.2)]
        essais = 0
        while len(self.cartes) < n and essais < 4000:
            essais += 1
            nx, ny = rng.uniform(-1.15, 1.15), rng.uniform(-1.12, 1.12)
            proche = rng.random() < 0.28                      # devant le hook : forte parallaxe
            dist = recul * rng.uniform(0.5, 0.85) if proche else recul * rng.uniform(1.0, 2.4)
            # Taille apparente approximative (en coordonnées normalisées) pour éviter les chevauchements.
            demi_l = cw / 2 / (dist * math.tan(FOV / 2) * aspect)
            demi_h = cw * 9 / 16 / 2 / (dist * math.tan(FOV / 2))
            marge = 0.85 if essais < 2500 else 0.55
            if any(abs(nx - ox) < (demi_l + ol) * marge and abs(ny - oy) < (demi_h + oh) * marge
                   for ox, oy, ol, oh in occupe):
                continue
            occupe.append((nx, ny, demi_l, demi_h))
            tex = textures[len(self.cartes) % len(textures)]
            self.cartes.append(_Carte(
                tex, vers_monde(nx, ny, dist), cw, cw * 9 / 16,
                rot=(rng.uniform(-0.25, 0.25), rng.uniform(-0.45, 0.45), rng.uniform(-0.08, 0.08)),
                phase=rng.uniform(0, 6.28), vitesse=rng.uniform(1.5, 3.0),
                retard=0.05 + 0.25 * rng.random(), lum=rng.uniform(0.62, 0.85), sat=rng.uniform(0.7, 0.9)))
        self._fonds = {}

    def fond(self, W: int, H: int) -> np.ndarray:
        """#0c0e0b avec un léger halo au centre (calculé une fois par résolution)."""
        if (W, H) not in self._fonds:
            m = max(W, H)
            x = ((np.arange(W, dtype=np.float32) - W / 2) / m) ** 2
            y = ((np.arange(H, dtype=np.float32)[:, None] - H * 0.45) / m) ** 2
            halo = np.clip(1 - np.sqrt(x + y) / 0.6, 0, 1) ** 2
            self._fonds[(W, H)] = FOND + halo[..., None] * np.array([10, 13, 6], np.float32)
        return self._fonds[(W, H)]

    # ---- caméra
    def camera(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, matrice monde→caméra) à l'instant t."""
        ph = self.ph
        c0 = np.array([0.0, 0.0, self.D0])
        k = _lisse(_lin(t, ph["recul"], ph["fin_recul"]))
        pos = c0 + (self.c1 - c0) * k
        vise = self.v1 * k
        roulis = self.roulis1 * math.sin(k * math.pi / 2)
        # Anticipation : à la fin du recul, le regard commence déjà à glisser vers la carte cible.
        vise = vise + (self.cible.centre - vise) * 0.22 * _lisse(_lin(t, ph["fin_recul"] - 0.3, ph["poussee"]))
        x = _lin(t, ph["poussee"], ph["fin"])
        if x > 0:
            but = self.cible.centre + np.array([0.0, 0.0, self.Dt])
            # La tête se tourne d'abord vers la cible, puis la caméra fonce (accélère, se pose).
            kv = _sortie(x, 2.2)
            kp = 1 - (1 - x * x) ** 3
            pos = pos + (but - pos) * kp
            vise = vise + (self.cible.centre - vise) * kv
            roulis = roulis * (1 - kv)
        return pos, _regarder(pos, vise, roulis)

    def etat_cible(self, t: float) -> tuple:
        """La carte cible se tourne face à la caméra pendant la poussée ; sa bordure s'efface."""
        x = _lin(t, self.ph["poussee"], self.ph["fin"])
        k = _lisse(x)
        rot = tuple(r * (1 - k) for r in self.cible.rot)
        apparition = _lin(t, 0.15, 0.45)
        bord = 0.035 * apparition * (1 - _lin(x, 0.55, 0.95))
        rayon = 0.12 * (1 - _lin(x, 0.6, 1.0))
        return rot, bord, rayon, apparition

    def points_reperes(self, t: float) -> np.ndarray:
        """Coins du hook et de la cible à l'écran (pixels) : sert à mesurer la vitesse de l'image."""
        pos, M = self.camera(t)
        res = []
        for c in (self.hook, self.cible):
            for dx in (-0.5, 0.5):
                for dy in (-0.5, 0.5):
                    p = M @ (c.centre + np.array([dx * c.largeur, dy * c.hauteur, 0.0]) - pos)
                    z = -p[2]
                    if z > 0.3 and abs(p[0] / z * self.f) < self.W and abs(p[1] / z * self.f) < self.H:
                        res.append((p[0] / z * self.f, p[1] / z * self.f))
                    else:
                        res.append((np.nan, np.nan))       # hors de l'écran : ne compte pas
        return np.array(res)

    # ---- rendu d'une sous-image
    def rendre(self, t: float, echelle: float = 1.0, proche: bool = False) -> np.ndarray:
        W, H = math.ceil(self.W * echelle), math.ceil(self.H * echelle)
        centre_ecran = (self.W * echelle / 2, self.H * echelle / 2)
        ph = self.ph
        g = _lin(t, 0.0, 0.1)
        sat_hook, lum_hook = 1.0 - 0.38 * g, 1.0 - 0.1 * g
        flash = 0.55 * max(0.0, 1 - t / 0.16)
        if t <= ph["recul"] and echelle == 1.0:
            # Gel : la caméra n'a pas encore bougé, l'image du hook est l'écran lui-même.
            img = self.depart.copy()
            gris = img @ _LUMA
            img -= gris[..., None]
            img *= sat_hook * lum_hook
            img += gris[..., None] * lum_hook
        else:
            img = self.fond(W, H).copy()
            pos, M = self.camera(t)
            f = self.f * echelle
            tt = max(0.0, t - ph["recul"])
            quads = []
            kr = _lisse(_lin(t, ph["recul"], ph["fin_recul"]))
            sortie_hook = _lin(t, ph["poussee"], ph["fin"])
            # Hook : désaturé, rétrécit (la caméra recule), coins arrondis, bordure lime, léger lacet.
            quads.append((self.hook, self.hook.centre, (0.0, 0.16 * kr, 0.0), sat_hook, lum_hook,
                          0.045 * _lin(t, ph["recul"], ph["recul"] + 0.25),
                          0.16 * _lin(t, ph["recul"], ph["recul"] + 0.3), 1.0 - 0.35 * sortie_hook))
            for c in self.cartes:
                a = _sortie(_lin(t, c.retard, c.retard + 0.35), 2.0)
                if a <= 0:
                    continue
                # Les cartes arrivent de plus loin et flottent doucement.
                p = c.centre + np.array([0.0, 0.035 * math.sin(tt * c.vitesse + c.phase), -1.6 * (1 - a)])
                rot = (c.rot[0] + 0.06 * math.sin(tt * 1.3 + c.phase), c.rot[1] + 0.10 * tt, c.rot[2])
                quads.append((c, p, rot, c.sat, c.lum, 0.0, 0.08, a))
            rot, bord, rayon, apparition = self.etat_cible(t)
            if apparition > 0:
                x = _lin(t, ph["poussee"], ph["fin"])
                quads.append((self.cible, self.cible.centre, rot, 1.0, 1.0 + 0.06 * apparition * (1 - x),
                              bord, rayon, apparition))
            # Du plus loin au plus proche (profondeur du centre dans le repère caméra).
            quads.sort(key=lambda q: (M @ (q[1] - pos))[2])
            Ml, posl = M.tolist(), pos.tolist()
            for q in quads:
                self._quad(img, posl, Ml, f, centre_ecran, proche, *q)
        if flash > 0:                                  # flash blanc au moment du gel
            img *= 1 - flash
            img += 255 * flash
        return img

    def _quad(self, img, pos, M, f, centre_ecran, proche, carte: _Carte, centre, rot, sat, lum, bord, rayon, alpha):
        H, W = img.shape[:2]
        # Géométrie en flottants Python (beaucoup plus rapide que numpy sur des vecteurs de 3).
        R = _rotation(*rot)
        lc, hc = carte.largeur, carte.hauteur
        ex = (R[0][0] * lc, R[1][0] * lc, R[2][0] * lc)          # bord haut, vers la droite
        ey = (-R[0][1] * hc, -R[1][1] * hc, -R[2][1] * hc)       # bord gauche, vers le bas
        o = (centre[0] - pos[0] - ex[0] / 2 - ey[0] / 2, centre[1] - pos[1] - ex[1] / 2 - ey[1] / 2,
             centre[2] - pos[2] - ex[2] / 2 - ey[2] / 2)
        Oc = [M[i][0] * o[0] + M[i][1] * o[1] + M[i][2] * o[2] for i in range(3)]
        Ex = [M[i][0] * ex[0] + M[i][1] * ex[1] + M[i][2] * ex[2] for i in range(3)]
        Ey = [M[i][0] * ey[0] + M[i][1] * ey[1] + M[i][2] * ey[2] for i in range(3)]
        prof = (-Oc[2], -Oc[2] - Ex[2], -Oc[2] - Ey[2], -Oc[2] - Ex[2] - Ey[2])
        if min(prof) < 0.05:                           # carte qui passe derrière la caméra
            return
        coins = [Oc, [Oc[0] + Ex[0], Oc[1] + Ex[1], Oc[2] + Ex[2]],
                 [Oc[0] + Ey[0], Oc[1] + Ey[1], Oc[2] + Ey[2]],
                 [Oc[0] + Ex[0] + Ey[0], Oc[1] + Ex[1] + Ey[1], Oc[2] + Ex[2] + Ey[2]]]
        alpha *= min(1.0, (min(prof) - 0.05) / 0.5)
        cx, cy = centre_ecran
        sx = [cx + f * c[0] / -c[2] for c in coins]
        sy = [cy - f * c[1] / -c[2] for c in coins]
        x0, x1 = max(0, int(math.floor(min(sx)))), min(W, int(math.ceil(max(sx))) + 1)
        y0, y1 = max(0, int(math.floor(min(sy)))), min(H, int(math.ceil(max(sy))) + 1)
        if x1 <= x0 or y1 <= y0 or alpha <= 0.002:
            return
        # Homographie (u, v, 1) → (sx·w, sy·w, w) avec w = profondeur ; on l'inverse.
        Hi = _inverse3([
            [cx * -Ex[2] + f * Ex[0], cx * -Ey[2] + f * Ey[0], cx * -Oc[2] + f * Oc[0]],
            [cy * -Ex[2] - f * Ex[1], cy * -Ey[2] - f * Ey[1], cy * -Oc[2] - f * Oc[1]],
            [-Ex[2], -Ey[2], -Oc[2]]])
        if Hi is None:
            return
        xs = np.arange(x0, x1, dtype=np.float32) + np.float32(0.5)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None] + np.float32(0.5)
        w = 1 / (Hi[2, 0] * xs + Hi[2, 1] * ys + Hi[2, 2])     # profondeur de chaque pixel
        u = (Hi[0, 0] * xs + Hi[0, 1] * ys + Hi[0, 2]) * w
        v = (Hi[1, 0] * xs + Hi[1, 1] * ys + Hi[1, 2]) * w
        # Distance signée au rectangle arrondi, en unités monde ; px = taille d'un pixel sur la carte.
        demi_l, demi_h = carte.largeur / 2, carte.hauteur / 2
        r = min(rayon, demi_l, demi_h)
        qx = np.abs(u * np.float32(carte.largeur) - np.float32(demi_l)) - np.float32(demi_l - r)
        qy = np.abs(v * np.float32(carte.hauteur) - np.float32(demi_h)) - np.float32(demi_h - r)
        inv_px = np.float32(f) / w
        d = np.maximum(qx, qy)
        if r > 0:
            coin = (qx > 0) & (qy > 0)
            if coin.any():
                d[coin] = np.hypot(qx[coin], qy[coin])
            d -= np.float32(r)
        d *= inv_px                                   # distance en pixels
        a = np.clip(np.float32(0.5) - d, 0, 1)
        if alpha < 1:
            a *= np.float32(alpha)
        lignes = np.nonzero(a.any(axis=1))[0]
        if not len(lignes):
            return
        l0, l1 = lignes[0], lignes[-1] + 1
        colonnes = np.nonzero(a[l0:l1].any(axis=0))[0]
        c0, c1 = colonnes[0], colonnes[-1] + 1
        u, v, d, a = u[l0:l1, c0:c1], v[l0:l1, c0:c1], d[l0:l1, c0:c1], a[l0:l1, c0:c1]
        # Mipmap : texels par pixel sur la partie la plus proche de la carte.
        texels = carte.tex.largeur / carte.largeur * float(w.min()) / f
        niveau = max(0, int(math.floor(math.log2(max(texels, 1e-6)) + 0.3)))
        c = carte.tex.echantillonner(u.ravel(), v.ravel(), niveau, proche).reshape(u.shape + (3,))
        if sat < 0.999:
            gris = (c @ _LUMA)[..., None]
            c -= gris
            c *= np.float32(sat * lum)
            c += gris * np.float32(lum)
        elif abs(lum - 1) > 1e-3:
            c *= np.float32(lum)
        if bord > 0:
            b = np.clip(np.float32(0.5) + d + np.float32(bord) * inv_px[l0:l1, c0:c1], 0, 1)[..., None]
            c += (LIME - c) * b
        # Brouillard léger avec la distance (les cartes lointaines se fondent dans le fond).
        d_c = (centre[0] - pos[0], centre[1] - pos[1], centre[2] - pos[2])
        prof_c = -(M[2][0] * d_c[0] + M[2][1] * d_c[1] + M[2][2] * d_c[2])
        brume = min(0.7, max(0.0, (prof_c - 6.0) / 14.0))
        if brume > 0:
            c += (FOND - c) * np.float32(brume)
        zone = img[y0 + l0:y0 + l1, x0 + c0:x0 + c1]
        c -= zone
        c *= a[..., None]
        zone += c


# --------------------------------------------------------------------------------------------

def rendre_intro(depart: np.ndarray, cartes: list[np.ndarray], cible: np.ndarray, sortie: Path,
                 fps: int, images: int, graine: int = 7, preset: str = "veryfast", crf: str = "16",
                 travailleurs: int | None = None) -> Path:
    """Rend l'intro (images à fps) dans un fichier vidéo au même format que les autres plans :
    H.264 (libx264, ou la carte graphique en Rendu GPU), -r fps, son pcm_s16le 48 kHz stéréo (silence)."""
    if (preset, crf) == ("veryfast", "16"):
        encodage = args_encodage("plan")
    else:
        encodage = ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p"]
    try:
        return _rendre_intro(depart, cartes, cible, sortie, fps, images, graine, encodage, travailleurs)
    except RuntimeError as e:
        if not est_gpu(encodage):
            raise
        signaler_echec_gpu(str(e))        # la carte graphique a refusé : on refait avec libx264
        return _rendre_intro(depart, cartes, cible, sortie, fps, images, graine, args_encodage("plan", False),
                             travailleurs)


def _rendre_intro(depart: np.ndarray, cartes: list[np.ndarray], cible: np.ndarray, sortie: Path,
                  fps: int, images: int, graine: int, encodage: list[str], travailleurs: int | None) -> Path:
    H, W = depart.shape[:2]
    if cible.shape[:2] != (H, W):
        raise ValueError("la carte cible doit avoir la taille de l'écran")
    if travailleurs is None:            # le reste des cœurs sert à l'encodeur
        import os
        travailleurs = max(2, min(4, (os.cpu_count() or 2) - 1))
    duree = images / fps
    scene = Scene(depart, cartes, cible, W, H, duree, graine)
    dt = 1.0 / fps

    def image(i):
        t = i * dt
        # Vitesse de l'image : déplacement à l'écran des coins du hook et de la cible en une demi-image.
        ecart = np.abs(scene.points_reperes(t) - scene.points_reperes(max(0.0, t - dt / 2)))
        v = float(np.nanmax(ecart)) if np.isfinite(ecart).any() else 0.0
        v *= 1920 / max(W, H)
        if t <= scene.ph["recul"] or (v < 3 and t > scene.ph["recul"] + 0.3):
            res = scene.rendre(t)
        elif v < 8:
            res = scene.rendre(t, 0.5)
        else:
            # Flou de mouvement : n sous-images sur [t - dt/2, t] (obturateur 180°), assez pour que
            # deux copies successives ne soient jamais à plus de ~10 pixels l'une de l'autre.
            # Plus ça va vite, plus elles sont rendues petites : le flou cache la différence.
            n = min(N_MAX, 2 + int(v // 8))
            echelle = 0.5 if n <= 3 else 0.25
            res = None
            for k in range(n):
                s = scene.rendre(t - dt / 2 * k / (n - 1), echelle, proche=n > 2)
                res = s if res is None else res + s
            res *= np.float32(1 / n)
        res += np.float32(0.5)
        np.clip(res, 0, 255, out=res)
        res = res.astype(np.uint8)
        while res.shape[0] < H:
            res = _agrandir(res)
        return np.ascontiguousarray(res[:H, :W]).tobytes()

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
           "-map", "0:v", "-map", "1:a", "-t", f"{duree:.4f}",
           *encodage, "-r", str(fps),
           "-c:a", "pcm_s16le", "-ar", "48000", str(sortie)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            creationflags=_FLAGS)
    try:
        with ThreadPoolExecutor(max_workers=max(1, travailleurs)) as ex:
            # Fenêtre glissante : quelques images d'avance seulement (mémoire bornée).
            en_cours = [ex.submit(image, i) for i in range(min(images, 2 * travailleurs))]
            prochaine = len(en_cours)
            for _ in range(images):
                donnees = en_cours.pop(0).result()
                if prochaine < images:
                    en_cours.append(ex.submit(image, prochaine))
                    prochaine += 1
                proc.stdin.write(donnees)
        proc.stdin.close()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    err = proc.stderr.read()
    if proc.wait() != 0 or not sortie.exists():
        raise RuntimeError("Échec de FFmpeg (intro 3D) : " + err.decode("utf-8", "replace")[-500:])
    return sortie
