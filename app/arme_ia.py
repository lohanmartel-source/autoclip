"""Paroles derrière l'arme (Pro) : découpage de l'arme et des mains par IA, sur le CPU.

Modèle : MobileSAM (Segment Anything en version légère, poids Apache-2.0, export ONNX « Acly/MobileSAM »
sous licence MIT), lancé avec onnxruntime (déjà livré avec l'app pour Whisper). ~45 Mo, téléchargé à
la 1re utilisation dans DONNEES/modeles (même mécanisme et même progression que le modèle Whisper).

Choisi en mesurant sur de vrais montages Valorant (CPU i7-12700KF) : MobileSAM ~0,45 s par image clé
et des masques nets ; EfficientSAM-Ti ~1,2 s et des masques qui débordent ; la version int8 de
MobileSAM est plus lente sur ce CPU.

Principe :
1. Images clés : seulement pendant l'affichage des mots, une toutes les PAS images (+ début de chaque
   plan). On découpe le bas de l'image (là où sont l'arme et les mains) : deux images clés sont posées
   côte à côte (ou l'une sous l'autre) dans la même image 1024 × 1024 de l'encodeur, ce qui divise le
   temps par deux sans perdre de détail.
2. Invite : une boîte en bas à droite (zone du viewmodel), deux tailles essayées. On garde le masque le
   plus sûr : confiance du modèle, stabilité (le masque bouge peu quand on change le seuil), taille et
   position plausibles d'un viewmodel (il touche le bas ou le bord droit, il est à droite, il ne
   couvre pas le réticule).
3. Dans le temps : une image clé en désaccord avec ses deux voisines est rejetée ; les logits (sortie
   brute du modèle, 256 × 256) sont lissés entre images clés voisines du même plan, puis interpolés
   pour chaque image : le contour reste net (seuil sur des logits interpolés, pas un masque flou).
4. Un groupe de mots ne passe derrière l'arme que si TOUTES ses images ont un masque sûr ; sinon il
   garde le placement habituel, loin de l'arme (un mot découpé au hasard est pire que pas de masque).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from outils import _FLAGS

NOM = "mobilesam"
REVISION = "0d3b403339b4674a82493d5e97964dd78089ddc8"     # commit figé du dépôt HuggingFace
ENCODEUR = "mobile_sam_image_encoder.onnx"
DECODEUR = "sam_mask_decoder_multi.onnx"
TAILLE = 1024        # entrée de l'encodeur (côté)
BAS = 256            # sortie basse définition du décodeur (logits)
PAS = 6              # une image clé toutes les PAS images (interpolation entre les deux)
MOYENNE = np.array([123.675, 116.28, 103.53], np.float32)

_sessions = None
_echec = 0.0


def _sans_bridage() -> None:
    """Windows 11 : un processus sans fenêtre au premier plan est « bridé » (EcoQoS, cœurs lents) après
    quelques secondes de calcul. Mesuré sur i7-12700KF : l'encodeur passe de 0,35 s à 1 s par passage.
    On demande à Windows de ne pas brider ce processus (sans effet ailleurs, sans droits admin)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class Etat(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k.GetCurrentProcess.restype = wintypes.HANDLE
        e = Etat(1, 1, 0)       # PROCESS_POWER_THROTTLING_EXECUTION_SPEED : contrôlé, désactivé
        k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(e), ctypes.sizeof(e))
    except Exception:
        pass


def charger(log=print):
    """(encodeur, décodeur) onnxruntime, gardés en mémoire. None si onnxruntime ou le modèle manque."""
    global _sessions, _echec
    if _sessions is not None:
        return _sessions
    if time.time() - _echec < 600:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        log("  paroles : onnxruntime manquant, découpage de l'arme par IA indisponible")
        _echec = time.time()
        return None
    import synchro_paroles as sp
    _sans_bridage()
    chemin = sp.chemin_modele(NOM, log, cache_hf=False, fichiers=[ENCODEUR, DECODEUR], revision=REVISION,
                              quoi="découpage de l'arme")
    if not chemin:
        _echec = time.time()
        return None
    try:
        t0 = time.time()
        def options(fils: int):
            so = ort.SessionOptions()
            so.intra_op_num_threads = fils
            so.log_severity_level = 3
            # Indispensable : sans ça, les fils d'une session « tournent à vide » après chaque appel et
            # volent le processeur à l'autre (encodeur / décodeur en alternance) : mesuré 1,9 s au lieu
            # de 0,35 s par passage de l'encodeur.
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.set_denormal_as_zero", "1")
            return so
        # Mesuré sur i7-12700KF : 8 fils = le plus rapide (au-delà, les cœurs E ralentissent tout).
        fils = int(os.environ.get("AUTOCLIP_FILS_IA") or max(1, min(8, (os.cpu_count() or 4) - 2)))
        enc = ort.InferenceSession(str(Path(chemin) / ENCODEUR), options(fils), providers=["CPUExecutionProvider"])
        dec = ort.InferenceSession(str(Path(chemin) / DECODEUR), options(min(fils, 4)), providers=["CPUExecutionProvider"])
        _sessions = (enc, dec)
        log(f"  paroles : modèle de découpage de l'arme chargé ({time.time() - t0:.1f} s)")
        return _sessions
    except Exception as e:
        log(f"  paroles : modèle de découpage de l'arme illisible ({type(e).__name__}: {str(e)[:120]})")
        _echec = time.time()
        return None


# ─────────────────────────────── Géométrie ───────────────────────────────

@dataclass
class Disposition:
    """Région découpée (le bas de la zone de jeu) et sa place dans l'image de l'encodeur."""
    rx: int              # région, en pixels de la vidéo
    ry: int
    rl: int
    rh: int
    l: int               # taille de la région dans le canevas (multiple de 4)
    h: int
    places: list         # [(ox, oy)] : coin de chaque case du canevas (multiple de 4)

    @property
    def ex(self) -> float:          # échelle horizontale vidéo → canevas
        return self.l / self.rl

    @property
    def ey(self) -> float:
        return self.h / self.rh


def disposer(zone, paysage: bool) -> Disposition:
    """Région = bas à droite de la zone de jeu (l'arme n'est jamais ailleurs, hors visée). Plusieurs
    images clés par passage de l'encodeur, tant que chacune garde assez de détail (mesuré : ~0,42
    pixel de canevas par pixel d'une vidéo 1080p ; masques aussi bons qu'en plein cadre)."""
    zx, zy, zl, zh = zone
    gauche, haut = (0.40, 0.30) if paysage else (0.30, 0.36)
    rx, ry = zx + int(zl * gauche), zy + int(zh * haut)
    rl, rh = zx + zl - rx, zy + zh - ry
    s_min = 0.42 * 1080 / min(zl, zh)
    meilleur = None
    for cols in range(1, 5):
        for rangs in range(1, 5):
            s = min((TAILLE - 4 * (cols - 1)) / (cols * rl), (TAILLE - 4 * (rangs - 1)) / (rangs * rh))
            n = cols * rangs
            cle = (s >= s_min, n if s >= s_min else 0, s)
            if meilleur is None or cle > meilleur[0]:
                meilleur = (cle, s, cols, rangs)
    _, s, cols, rangs = meilleur
    s = min(s, 1.0)
    l, h = int(rl * s) // 4 * 4, int(rh * s) // 4 * 4
    places = [(c * (l + 4), r * (h + 4)) for r in range(rangs) for c in range(cols)]
    return Disposition(rx, ry, rl, rh, l, h, places)


# ─────────────────────────────── Résultat ───────────────────────────────

@dataclass
class MasquesIA:
    zone: tuple
    disp: Disposition
    cles: np.ndarray                      # (K,) numéro d'image de chaque image clé
    plans: np.ndarray                     # (K,) plan de chaque image clé
    logits: list                          # K × (h/4, l/4) float32 ou None (image clé rejetée)
    plan_image: np.ndarray                # (T,) plan de chaque image
    stats: dict = field(default_factory=dict)
    debug: dict | None = None

    def _voisines(self, t: int):
        """(ka, kb, f) : images clés sûres qui encadrent t dans le même plan, poids de kb."""
        K = len(self.cles)
        p = self.plan_image[min(t, len(self.plan_image) - 1)]
        j = int(np.searchsorted(self.cles, t, side="right")) - 1        # dernière image clé <= t
        a = j
        while a >= 0 and self.plans[a] == p and self.logits[a] is None:
            a -= 1
        b = j + 1
        while b < K and self.plans[b] == p and self.logits[b] is None:
            b += 1
        ok_a = a >= 0 and self.plans[a] == p and self.logits[a] is not None and t - self.cles[a] <= 2 * PAS
        ok_b = b < K and self.plans[b] == p and self.logits[b] is not None and self.cles[b] - t <= 2 * PAS
        if ok_a and self.cles[a] == t:
            return a, a, 0.0
        if ok_a and ok_b:
            ta, tb = self.cles[a], self.cles[b]
            return a, b, (0.0 if tb == ta else (t - ta) / (tb - ta))
        if ok_a and t - self.cles[a] <= PAS:
            return a, a, 0.0
        if ok_b and self.cles[b] - t <= PAS:
            return b, b, 0.0
        return None

    def couvert(self, t: int) -> bool:
        return self._voisines(t) is not None

    def logits_image(self, t: int):
        v = self._voisines(t)
        if v is None:
            return None
        a, b, f = v
        return self.logits[a] if a == b or f <= 0 else (1 - f) * self.logits[a] + f * self.logits[b]

    def masque(self, t: int, ax: int, ay: int, bx: int, by: int) -> np.ndarray | None:
        """Couverture de l'arme (0-1, bords nets anticrénelés) sur le rectangle vidéo [ax:bx, ay:by]."""
        L = self.logits_image(t)
        if L is None:
            return None
        d = self.disp
        out = np.zeros((by - ay, bx - ax), np.float32)
        # Partie du rectangle dans la région découpée (au-dessus : jamais d'arme).
        y0, x0 = max(ay, d.ry), max(ax, d.rx)
        y1, x1 = min(by, d.ry + d.rh), min(bx, d.rx + d.rl)
        if y1 <= y0 or x1 <= x0:
            return out
        qx, qy = d.ex / 4, d.ey / 4               # pixels vidéo → pixels des logits
        hb, lb = L.shape
        r0 = max(0, int((y0 - d.ry) * qy) - 2)
        r1 = min(hb, int(np.ceil((y1 - d.ry) * qy)) + 3)
        c0 = max(0, int((x0 - d.rx) * qx) - 2)
        c1 = min(lb, int(np.ceil((x1 - d.rx) * qx)) + 3)
        Lc = L[r0:r1, c0:c1]
        if Lc.max() < -4:
            return out
        Ay = _interp(y1 - y0, (y0 - d.ry + 0.5) * qy - 0.5 - r0, qy, r1 - r0)
        Ax = _interp(x1 - x0, (x0 - d.rx + 0.5) * qx - 0.5 - c0, qx, c1 - c0)
        z = Ay @ Lc @ Ax.T
        # Seuil doux sur les logits interpolés : transition de 2 à 4 pixels, contour net.
        pente = 2.0 * qx
        out[y0 - ay:y1 - ay, x0 - ax:x1 - ax] = np.clip(0.5 + z * pente, 0, 1)
        return out

    def masque_bas(self, t: int, lw: int, lh: int) -> np.ndarray | None:
        """Masque binaire de la zone de jeu en basse définition (lh, lw), pour le placement."""
        L = self.logits_image(t)
        if L is None:
            return None
        return _vers_zone(L > 0, self.disp, self.zone, lw, lh)


def _interp(n_out: int, debut: float, pas: float, n_in: int) -> np.ndarray:
    u = np.clip(debut + pas * np.arange(n_out), 0, n_in - 1)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    f = (u - i0).astype(np.float32)
    M = np.zeros((n_out, n_in), np.float32)
    M[np.arange(n_out), i0] += 1 - f
    M[np.arange(n_out), i1] += f
    return M


def _vers_zone(B: np.ndarray, d: Disposition, zone, lw: int, lh: int) -> np.ndarray:
    """Masque binaire en pixels des logits → masque (lh, lw) de toute la zone de jeu."""
    zx, zy, zl, zh = zone
    out = np.zeros((lh, lw), bool)
    ys = (np.arange(lh) + 0.5) * zh / lh + zy
    xs = (np.arange(lw) + 0.5) * zl / lw + zx
    iy = ((ys - d.ry) * d.ey / 4).astype(int)
    ix = ((xs - d.rx) * d.ex / 4).astype(int)
    oky = (ys >= d.ry) & (iy < B.shape[0])
    okx = (ix >= 0) & (ix < B.shape[1])
    out[np.ix_(oky, okx)] = B[np.ix_(iy[oky], ix[okx])]
    return out


# ─────────────────────────────── Découpage ───────────────────────────────

def _lire_images(video: Path, cles: list[int], d: Disposition) -> np.ndarray:
    """Images clés (K, h, l, 3) uint8 : bas de la zone de jeu, à la taille du canevas.
    La vidéo est lue en continu (un filtre « select » avec des centaines d'images ne passe pas)."""
    voulues = set(int(k) for k in cles)
    dernier = max(voulues)
    taille = d.l * d.h * 3
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-an", "-vf",
                          f"crop={d.rl}:{d.rh}:{d.rx}:{d.ry},scale={d.l}:{d.h}:flags=bicubic",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_FLAGS, bufsize=taille)
    res = []
    try:
        for n in range(dernier + 1):
            brut = p.stdout.read(taille)
            if len(brut) < taille:
                break
            if n in voulues:
                res.append(np.frombuffer(brut, np.uint8).reshape(d.h, d.l, 3))
    finally:
        p.stdout.close()
        p.kill()
        p.wait()
    if not res:
        raise RuntimeError("vidéo illisible")
    return np.stack(res)


# Invites, en fraction de la zone de jeu : points sur le viewmodel (1), points sur le décor (0),
# boîte (2 = coin haut-gauche, 3 = coin bas-droit). Mesuré sur de vrais montages : une grande boîte
# seule attrape souvent le sol ou le HUD ; un point sur la main / l'arme (qui sort toujours par le
# bord droit) avec des points « décor » au sol et sur la barre de compétences est bien plus sûr.
INVITES_PORTRAIT = [
    [((0.93, 0.80), 1), ((0.45, 0.94), 0), ((0.38, 0.66), 0)],
    [((0.93, 0.92), 1), ((0.95, 0.66), 1), ((0.45, 0.94), 0), ((0.38, 0.66), 0)],
    [((0.88, 0.86), 1), ((0.55, 0.48), 2), ((1.0, 1.0), 3)],
]
INVITES_PAYSAGE = [
    [((0.80, 0.90), 1), ((0.40, 0.95), 0), ((0.45, 0.70), 0)],
    [((0.72, 0.85), 1), ((0.90, 0.95), 1), ((0.40, 0.95), 0), ((0.45, 0.70), 0)],
    [((0.80, 0.88), 1), ((0.58, 0.50), 2), ((1.0, 1.0), 3)],
]


def _invites(d: Disposition, paysage: bool, zone) -> list[tuple[np.ndarray, np.ndarray]]:
    """(points en pixels de la case, étiquettes) pour chaque invite."""
    zx, zy, zl, zh = zone
    res = []
    for inv in (INVITES_PAYSAGE if paysage else INVITES_PORTRAIT):
        pts, lab = [], []
        for (x, y), e in inv:
            px = np.clip((zx + x * zl - d.rx) * d.ex, 0, d.l - 1)
            py = np.clip((zy + y * zh - d.ry) * d.ey, 0, d.h - 1)
            pts.append((px, py))
            lab.append(e)
        if 2 not in lab:                          # convention de l'export ONNX de SAM : point vide
            pts.append((0.0, 0.0))
            lab.append(-1)
        res.append((np.array(pts, np.float32), np.array(lab, np.float32)))
    return res


def _note(L: np.ndarray, iou: float, d: Disposition, zone, paysage: bool,
          positifs: np.ndarray) -> tuple[float, np.ndarray | None, str]:
    """Note 0-1 d'un masque candidat (logits basse déf. de la case) ; masque binaire nettoyé ; raison."""
    zx, zy, zl, zh = zone
    B = L > 0
    hb, lb = B.shape
    if B.mean() < 0.005:
        return 0.0, None, "vide"
    lab, n = ndimage.label(B)
    # On garde les morceaux posés sur un point « viewmodel » de l'invite (main / arme au bord droit).
    ancres = {int(lab[min(hb - 1, int(y / 4)), min(lb - 1, int(x / 4))]) for x, y in positifs}
    ancres.discard(0)
    if not ancres:
        return 0.0, None, "hors invite"
    garder = np.zeros(n + 1, bool)
    garder[list(ancres)] = True
    B = garder[lab]
    if not (B[-2:, :].any() or B[:, -2:].any()):
        return 0.0, None, "flottant"            # le viewmodel sort toujours par le bas ou la droite
    qx, qy = d.ex / 4, d.ey / 4
    aire = B.sum() / ((zl * qx) * (zh * qy))
    if not 0.012 < aire < (0.30 if paysage else 0.32):
        return 0.0, None, f"aire {aire:.2f}"
    ys, xs = np.nonzero(B)
    xz = (xs / qx + d.rx - zx) / zl
    if xz.mean() < (0.58 if paysage else 0.62):
        return 0.0, None, f"centre {xz.mean():.2f}"
    # Sol, mur ou barre de compétences : ça s'étale vers la gauche ; l'arme, non.
    gauche = float((xz < (0.48 if paysage else 0.45)).mean())
    if gauche > 0.08:
        return 0.0, None, f"gauche {gauche:.2f}"
    # Le réticule (centre de l'écran) ne doit jamais être couvert.
    ry = int((zy + 0.5 * zh - d.ry) * qy)
    rxa, rxb = int((zx + 0.47 * zl - d.rx) * qx), int((zx + 0.53 * zl - d.rx) * qx)
    if 0 <= ry < hb and B[max(0, ry - 1):ry + 2, max(0, rxa):rxb].any():
        return 0.0, None, "réticule"
    # Haut du masque : un viewmodel ne monte pas au-dessus du milieu de l'écran (sauf visée).
    haut = (ys.min() / qy + d.ry - zy) / zh
    if haut < 0.40:
        return 0.0, None, f"haut {haut:.2f}"
    # Stabilité : le masque change peu quand on déplace le seuil des logits (autour du morceau gardé).
    autour = ndimage.binary_dilation(B, iterations=3)
    fort, faible = ((L > 2) & autour).sum(), ((L > -2) & autour).sum()
    stab = float(fort / max(1, faible))
    if stab < 0.70 or iou < 0.70:
        return 0.0, None, f"stab {stab:.2f} iou {iou:.2f}"
    return float(iou * stab), B, "ok"


def segmenter(video: Path, zone, paysage: bool, T: int, cles: list[int], plan_image: np.ndarray,
              log=print, debug: bool = False) -> MasquesIA | None:
    """Découpe l'arme sur les images clés `cles`. None si le modèle n'est pas disponible."""
    sess = charger(log)
    if sess is None:
        return None
    enc, dec = sess
    t0 = time.time()
    d = disposer(zone, paysage)
    cles = sorted(set(int(k) for k in cles if 0 <= k < T))
    if not cles:
        return None
    images = _lire_images(video, cles, d)
    cles = cles[:len(images)]
    t_lecture = time.time() - t0
    invites = _invites(d, paysage, zone)
    hb, lb = d.h // 4, d.l // 4
    logits: list = [None] * len(cles)
    binaires: list = [None] * len(cles)
    notes = np.zeros(len(cles), np.float32)
    n_case = len(d.places)
    t_enc = t_dec = 0.0
    journal_cles: list[str] = []
    candidats: list[list] = [[] for _ in cles]
    for i0 in range(0, len(cles), n_case):
        canevas = np.zeros((TAILLE, TAILLE, 3), np.float32)
        lot = list(range(i0, min(len(cles), i0 + n_case)))
        for j, i in enumerate(lot):
            ox, oy = d.places[j]
            canevas[oy:oy + d.h, ox:ox + d.l] = images[i]
        # L'encodeur normalise et complète lui-même ; le noir (0) devient la couleur moyenne → neutre.
        vide = np.ones((TAILLE, TAILLE), bool)
        for j in range(len(lot)):
            ox, oy = d.places[j]
            vide[oy:oy + d.h, ox:ox + d.l] = False
        canevas[vide] = MOYENNE
        ta = time.time()
        emb = enc.run(None, {"input_image": canevas})[0]
        t_enc += time.time() - ta
        for j, i in enumerate(lot):
            ox, oy = d.places[j]
            meilleur = (0.0, None, None)
            raisons = []
            for pts, lab in invites:
                if meilleur[0] >= 0.80 or (meilleur[0] > 0 and 2 in lab):
                    break      # déjà sûr : pas d'autre invite (la boîte seulement en dernier recours)
                pts_c = pts + np.array([ox, oy], np.float32) * (lab >= 0)[:, None]
                td = time.time()
                _m, iou, low = dec.run(None, {
                    "image_embeddings": emb, "point_coords": pts_c[None],
                    "point_labels": lab[None],
                    "mask_input": np.zeros((1, 1, BAS, BAS), np.float32),
                    "has_mask_input": np.zeros(1, np.float32),
                    "orig_im_size": np.array([BAS, BAS], np.float32)})
                t_dec += time.time() - td
                for c in range(low.shape[1]):
                    L = low[0, c, oy // 4:oy // 4 + hb, ox // 4:ox // 4 + lb]
                    note, B, r = _note(L, float(iou[0, c]), d, zone, paysage, pts[lab == 1])
                    raisons.append(r)
                    if B is not None:
                        candidats[i].append((note, L.copy(), B))
                    if note > meilleur[0]:
                        meilleur = (note, L, B)
            journal_cles.append(", ".join(sorted(set(raisons))))
    # Où est le viewmodel dans CE montage : carte de fréquence des meilleurs masques. Un masque qui
    # sort de cette carte (sol, mur, flou de mouvement pris pour l'arme) est écarté au profit d'un
    # autre candidat de la même image clé, ou l'image clé est rejetée.
    meilleurs = [max(c, key=lambda x: x[0]) for c in candidats if c]
    carte = np.mean([m[2] for m in meilleurs], axis=0) if len(meilleurs) >= 8 else None
    for i in range(len(cles)):
        choix, note_choix = None, 0.0
        for note, L, B in candidats[i]:
            accord = 1.0 if carte is None else float((carte[B] >= 0.3).mean())
            if accord >= 0.75 and note * accord > note_choix:
                choix, note_choix = (L, B), note * accord
        if choix is None:
            if candidats[i]:
                journal_cles[i] += " ; hors de la carte"
            continue
        # Morceaux hors du viewmodel retirés aussi des logits (contour net partout ailleurs).
        L, B = choix
        garde = ndimage.binary_dilation(B, iterations=2)
        L[~garde] = np.minimum(L[~garde], -6.0)
        logits[i], binaires[i], notes[i] = L.astype(np.float32), B, note_choix
    plans = np.array([plan_image[min(k, T - 1)] for k in cles])
    # Cohérence dans le temps : une image clé en désaccord avec ses deux voisines (qui, elles, sont
    # d'accord entre elles) est rejetée ; un plan où moins de la moitié des images clés sont sûres
    # est entièrement rejeté (le modèle hésite : arme cachée, visée, effet plein écran…).
    def iou(a, b):
        return np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())
    rejets = 0
    for i in range(len(cles)):
        if binaires[i] is None:
            continue
        vs = [j for j in (i - 1, i + 1) if 0 <= j < len(cles) and plans[j] == plans[i] and binaires[j] is not None]
        if len(vs) == 2 and iou(binaires[vs[0]], binaires[vs[1]]) > 0.6 \
                and max(iou(binaires[i], binaires[j]) for j in vs) < 0.45:
            logits[i] = None
            rejets += 1
        elif len(vs) == 1 and iou(binaires[i], binaires[vs[0]]) < 0.3:
            # Deux images clés seules en désaccord : on ne sait pas laquelle croire.
            logits[i] = None
            rejets += 1
    for p in np.unique(plans):
        idx = np.flatnonzero(plans == p)
        if sum(logits[i] is not None for i in idx) < 0.5 * len(idx):
            for i in idx:
                logits[i] = None
    # Lissage dans le temps (sur les logits) entre images clés sûres voisines du même plan.
    lisses = list(logits)
    for i in range(len(cles)):
        if logits[i] is None:
            continue
        vs = [j for j in (i - 1, i + 1) if 0 <= j < len(cles) and plans[j] == plans[i] and logits[j] is not None]
        if vs:
            lisses[i] = (0.6 * logits[i] + 0.4 * sum(logits[j] for j in vs) / len(vs)).astype(np.float32)
    n_ok = sum(x is not None for x in lisses)
    res = MasquesIA(zone, d, np.array(cles), plans, lisses, plan_image)
    res.stats = {"cles": len(cles), "sures": n_ok, "rejets_temps": rejets, "lecture": t_lecture,
                 "encodeur": t_enc, "decodeur": t_dec, "secondes": time.time() - t0, "par_passage": n_case}
    if debug:      # pour les planches de test (outils) : images, raisons des rejets, masques bruts
        res.debug = {"images": images, "raisons": journal_cles, "notes": notes, "binaires": binaires}
    log(f"  paroles : arme découpée par IA sur {n_ok}/{len(cles)} images clés "
        f"({time.time() - t0:.1f} s)")
    return res
