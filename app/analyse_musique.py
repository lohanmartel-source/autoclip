"""Analyse de la musique : tempo, position des beats et du drop, choix de l'extrait."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from outils import ErreurAutoclip, decoder_audio

SR = 22050
HOP = 512


@dataclass
class Musique:
    chemin: Path
    debut: float          # début de l'extrait dans la chanson (s)
    beats: np.ndarray     # temps des beats relatifs au début de l'extrait (beats[0] = 0)
    drop: int             # index (dans beats) du drop
    tempo: float
    periode: float        # durée d'un beat (s)
    accents: np.ndarray | None = None   # force de chaque beat de l'extrait (0–1) : coups forts, drops

    @property
    def duree(self) -> float:
        return float(self.beats[-1])


def _grille_reguliere(beats: np.ndarray, periode: float) -> np.ndarray:
    """Comble les beats manquants (trous) détectés par l'analyse."""
    sortie = [beats[0]]
    for b in beats[1:]:
        ecart = b - sortie[-1]
        manquants = max(0, int(round(ecart / periode)) - 1)
        pas = ecart / (manquants + 1)
        for _ in range(manquants):
            sortie.append(sortie[-1] + pas)
        sortie.append(b)
    return np.array(sortie)


def analyser_musique(chemin: Path, duree_cible: float, debut: float | None = None, log=print,
                     beats_intro: int = 0) -> Musique:
    y = decoder_audio(chemin, SR)
    duree_totale = len(y) / SR
    if duree_totale < 5:
        raise ErreurAutoclip(f"Musique illisible ou trop courte : {Path(chemin).name}")

    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="time")
    tempo = float(np.atleast_1d(tempo)[0]) or 120.0
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8:
        beats = np.arange(0.0, duree_totale, 60.0 / tempo)
    periode = float(np.median(np.diff(beats)))
    beats = _grille_reguliere(beats, periode)

    # Correction de phase : le suivi de beats s'accroche parfois aux charlestons (contretemps).
    # On recale la grille sur la grosse caisse (basses fréquences), qui marque les vrais temps.
    env_basses = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmax=200, n_mels=32)

    def force(decalage):
        idx = librosa.time_to_frames(beats + decalage, sr=SR, hop_length=HOP)
        idx = idx[(idx >= 0) & (idx < len(env_basses))]
        return float(env_basses[idx].mean()) if len(idx) else 0.0

    decalages = np.arange(-periode / 2, periode / 2, HOP / SR)
    meilleur = float(decalages[int(np.argmax([force(d) for d in decalages]))])
    beats = beats + meilleur
    beats = beats[beats >= 0]

    # Ajustement fin : chaque beat est recalé sur le coup de grosse caisse le plus proche (± 45 ms)
    # pour que les coupes tombent pile sur le son, pas juste sur la grille moyenne.
    rayon = max(1, int(0.045 * SR / HOP))
    seuil = np.percentile(env_basses, 75)
    cales = []
    for b in beats:
        i = int(round(b * SR / HOP))
        a, z = max(0, i - rayon), min(len(env_basses), i + rayon + 1)
        if z > a:
            j = a + int(np.argmax(env_basses[a:z]))
            if env_basses[j] > seuil:
                b = j * HOP / SR
        cales.append(b)
    beats = np.maximum.accumulate(np.array(cales))

    # Énergie de la chanson (volume) pour trouver le drop : la plus grosse montée d'énergie.
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=SR, hop_length=HOP)

    def energie(a, b):
        m = (t_rms >= a) & (t_rms < b)
        return float(rms[m].mean()) if m.any() else 0.0

    w = max(2.0, 4 * periode)
    score_drop = np.array([energie(b, b + w) - energie(b - w, b) + 0.5 * energie(b, b + w) for b in beats])

    n_cible = max(8, int(round(duree_cible / periode)))

    if debut is None:
        valides = (beats >= 3.0) & (beats <= duree_totale - min(duree_cible * 0.4, 12))
        drop_abs = int(np.argmax(np.where(valides, score_drop, -np.inf))) if valides.any() \
            else int(np.argmax(score_drop))
        # On commence quelques mesures avant le drop (multiple de 4 beats = début de phrase musicale)
        avance = int(round(min(max(duree_cible * 0.35, 3.0), 10.0) / periode))
        avance = max(4, avance - avance % 4) + beats_intro   # place pour le hook d'intro
        i0 = max(0, drop_abs - avance)
    else:
        i0 = int(np.searchsorted(beats, debut - 0.05))
        if i0 >= len(beats) - 8:
            raise ErreurAutoclip("Le début de musique demandé est trop proche de la fin de la chanson.")
        zone = slice(i0 + 2, min(len(beats) - 1, i0 + n_cible - 3))
        drop_abs = i0 + 2 + int(np.argmax(score_drop[zone])) if zone.stop > zone.start else i0 + 2

    i1 = min(len(beats) - 1, i0 + n_cible)
    drop_abs = int(np.clip(drop_abs, i0, i1 - 1))
    # Fin sur une phrase musicale (multiple de 4 temps après le drop), comme un monteur.
    apres = i1 - drop_abs
    if apres % 4:
        haut = drop_abs + (apres // 4 + 1) * 4
        bas = drop_abs + (apres // 4) * 4
        i1 = haut if haut <= len(beats) - 1 and (haut - i1) <= 2 else max(bas, drop_abs + 4)
        i1 = min(i1, len(beats) - 1)
    if beats[i1] - beats[i0] < duree_cible - 2 * periode:
        log(f"  musique trop courte : le montage fera {beats[i1] - beats[i0]:.1f} s")

    extrait = beats[i0:i1 + 1] - beats[i0]

    # Force de chaque beat : grosse caisse + attaque générale + montée d'énergie (début de section).
    def pic(envl, t):
        i = int(round(t * SR / HOP))
        return float(envl[max(0, i - 2):i + 3].max()) if 0 <= i < len(envl) else 0.0

    brut = np.array([pic(env_basses, b) + 0.5 * pic(env, b) for b in beats[i0:i1 + 1]])
    section = np.clip(score_drop[i0:i1 + 1], 0, None)
    norm = lambda x: x / (np.percentile(x, 95) + 1e-9) if len(x) else x
    accents = np.clip(0.7 * norm(brut) + 0.3 * norm(section), 0, 1.5)
    accents[drop_abs - i0] = max(accents.max(), 1.0)   # le drop est toujours le plus fort
    log(f"  tempo ≈ {tempo:.0f} BPM, extrait {beats[i0]:.1f}s → {beats[i1]:.1f}s, "
        f"drop à {beats[drop_abs]:.1f}s")
    return Musique(chemin=Path(chemin), debut=float(beats[i0]), beats=extrait,
                   drop=drop_abs - i0, tempo=tempo, periode=periode, accents=accents)
