"""Effets sonores de montage fabriqués par le code (aucun fichier à télécharger, aucun droit d'auteur) :
whoosh sur les transitions, boom sur le drop et les gros kills, montée (riser) avant le drop,
son de cassette qui rembobine, petit glitch numérique."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

SR = 48000
_rng = np.random.default_rng(7)


def _t(d: float) -> np.ndarray:
    return np.arange(int(d * SR)) / SR


def _bande_variable(bruit: np.ndarray, f_debut: float, f_fin: float, largeur: float = 1.0, bloc: int = 512) -> np.ndarray:
    """Bruit filtré par une bande qui glisse de f_debut à f_fin (par petits blocs)."""
    n = len(bruit)
    sortie = np.zeros(n, np.float32)
    zi = None
    nb = max(1, n // bloc)
    for i in range(nb + 1):
        a, b = i * bloc, min(n, (i + 1) * bloc)
        if a >= b:
            break
        x = i / max(1, nb)
        fc = f_debut * (f_fin / f_debut) ** x
        lo, hi = fc / (1 + largeur), min(fc * (1 + largeur), SR / 2 * 0.95)
        sos = butter(2, [lo, hi], btype="band", fs=SR, output="sos")
        if zi is None or zi.shape[0] != sos.shape[0]:
            zi = np.zeros((sos.shape[0], 2))
        y, zi = sosfilt(sos, bruit[a:b], zi=zi)
        sortie[a:b] = y
    return sortie


def _stereo(m: np.ndarray, pan_debut: float = 0.0, pan_fin: float = 0.0) -> np.ndarray:
    p = np.linspace(pan_debut, pan_fin, len(m))
    g = np.sqrt(0.5 * (1 - p)), np.sqrt(0.5 * (1 + p))
    return np.stack([m * g[0], m * g[1]], axis=1).astype(np.float32)


def _normer(x: np.ndarray, crete: float) -> np.ndarray:
    return x * (crete / (np.max(np.abs(x)) + 1e-9))


def whoosh(d: float = 0.42) -> tuple[np.ndarray, float]:
    """Retourne (son, décalage) : le décalage place le sommet du whoosh pile sur la coupe."""
    t = _t(d)
    x = t / d
    env = np.where(x < 0.7, (x / 0.7) ** 2.2, np.clip((1 - x) / 0.3, 0, 1) ** 1.5)
    s = _bande_variable(_rng.standard_normal(len(t)).astype(np.float32), 350, 2600, 0.9) * env
    return _stereo(_normer(s, 0.30), -0.6, 0.6), 0.7 * d


def boom(d: float = 1.3) -> tuple[np.ndarray, float]:
    t = _t(d)
    f = 38 + 85 * np.exp(-t / 0.09)                       # chute de hauteur 120 Hz -> 40 Hz
    phase = 2 * np.pi * np.cumsum(f) / SR
    grave = np.sin(phase) * np.exp(-t / 0.42)
    bruit = sosfilt(butter(2, 900, fs=SR, output="sos"), _rng.standard_normal(len(t))) * np.exp(-t / 0.05)
    clic = np.zeros_like(t)
    clic[:int(0.004 * SR)] = np.hanning(int(0.008 * SR))[:int(0.004 * SR)]
    s = grave + 0.5 * bruit + 0.3 * clic
    return _stereo(_normer(np.tanh(1.6 * s), 0.55)), 0.0


def impact(d: float = 0.7) -> tuple[np.ndarray, float]:
    s, _ = boom(d)
    return s * 0.65, 0.0


def riser(d: float) -> tuple[np.ndarray, float]:
    """Montée de tension qui finit pile au drop."""
    d = max(0.6, min(d, 4.0))
    t = _t(d)
    x = t / d
    bruit = _bande_variable(_rng.standard_normal(len(t)).astype(np.float32), 400, 7000, 1.2)
    f = 180 * (6.0 ** x)
    ton = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    env = x ** 2.5
    s = (bruit + ton) * env
    return _stereo(_normer(s, 0.22)), d


def rembobinage(d: float) -> tuple[np.ndarray, float]:
    """Cassette qui rembobine : sifflement aigu qui ondule + souffle."""
    d = max(0.3, d)
    t = _t(d)
    f = 900 + 1700 * (t / d) + 260 * np.sin(2 * np.pi * 11 * t)
    ton = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.15
    ton = sosfilt(butter(2, 3500, fs=SR, output="sos"), ton)
    souffle = sosfilt(butter(2, 2500, btype="high", fs=SR, output="sos"), _rng.standard_normal(len(t))) * 0.25
    env = np.clip(t / 0.05, 0, 1) * np.clip((d - t) / 0.08, 0, 1)
    return _stereo(_normer((ton + souffle) * env, 0.20)), 0.0


def glitch(d: float = 0.16) -> tuple[np.ndarray, float]:
    t = _t(d)
    s = np.zeros_like(t)
    pos = 0
    while pos < len(t):
        n = int(_rng.uniform(0.008, 0.03) * SR)
        f = _rng.uniform(300, 3000)
        s[pos:pos + n] = np.sign(np.sin(2 * np.pi * f * t[:len(s[pos:pos + n])])) * _rng.uniform(0.3, 1)
        pos += n + int(_rng.uniform(0, 0.01) * SR)
    return _stereo(_normer(s, 0.14)), 0.0


SONS = {"whoosh": whoosh, "boom": boom, "impact": impact, "riser": riser, "rewind": rembobinage, "glitch": glitch}


def piste_effets(evenements: list[tuple[str, float, float]], n: int) -> np.ndarray:
    """evenements : [(type, temps en s, durée)] → piste stéréo de n échantillons."""
    piste = np.zeros((n, 2), np.float32)
    for type_, t, d in evenements:
        if type_ == "fichier":            # son déjà prêt (ex. la voix du caster), 48 kHz stéréo
            try:
                import soundfile as sf
                son, sr = sf.read(str(d), dtype="float32", always_2d=True)
            except Exception:
                continue
            if sr != SR or son.shape[1] != 2:
                continue
            decalage = 0.0
        else:
            f = SONS.get(type_)
            if f is None:
                continue
            son, decalage = f(d) if type_ in ("riser", "rewind") else f()
        i0 = int(round((t - decalage) * SR))
        a = max(0, i0)
        b = min(n, i0 + len(son))
        if b > a:
            piste[a:b] += son[a - i0:b - i0]
    return piste
