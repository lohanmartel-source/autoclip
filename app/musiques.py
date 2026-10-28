"""Bibliothèque de musiques : analyse chaque chanson d'un dossier (tempo, force du drop, énergie)
et choisit automatiquement celle qui fera le meilleur montage.

Les analyses sont gardées en cache (fichier .autoclip_musiques.json dans le dossier) : une chanson
n'est analysée qu'une fois.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import librosa
import numpy as np

from outils import decoder_audio

EXT_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".mp4"}
SR = 22050
HOP = 512
CACHE = ".autoclip_musiques.json"


def _analyser(chemin: Path) -> dict | None:
    y = decoder_audio(chemin, SR)
    if len(y) < SR * 15:
        return None
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    # Rythme bien marqué ? (force des attaques sur les beats comparée à la moyenne)
    rythme = float(env[beats].mean() / (env.mean() + 1e-9)) if len(beats) > 8 else 0.0
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    # Force du drop : plus grande hausse d'énergie entre 4 s avant et 4 s après (en ignorant les bords).
    par_s = SR / HOP
    w = int(4 * par_s)
    lisse = np.convolve(rms, np.ones(w) / w, mode="same")
    if len(lisse) <= 2 * w + 1:
        return None
    hausse = lisse[2 * w:] - lisse[:-2 * w]
    marge = int(6 * par_s)
    zone = hausse[marge:-marge] if len(hausse) > 2 * marge else hausse
    drop = float(zone.max() / (np.percentile(rms, 95) + 1e-9))
    basses = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, fmax=200, n_mels=32)
    punch = float(np.percentile(basses, 95) / (np.median(basses) + 1e-9))
    return {"tempo": tempo, "drop": drop, "punch": punch, "rythme": rythme, "duree": len(y) / SR,
            "energie": float(np.percentile(rms, 90))}


def bibliotheque(dossier: Path, log=print) -> dict[str, dict]:
    """{nom de fichier: analyse}. Analyse seulement les chansons nouvelles ou modifiées."""
    dossier = Path(dossier)
    cache_f = dossier / CACHE
    try:
        cache = json.loads(cache_f.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    resultat, change = {}, False
    for f in sorted(dossier.iterdir()):
        if f.suffix.lower() not in EXT_AUDIO or f.name.startswith("."):
            continue
        cle = f.name
        mtime = f.stat().st_mtime
        if cle in cache and cache[cle].get("mtime") == mtime:
            resultat[cle] = cache[cle]
            continue
        log(f"  analyse de la musique : {f.name}")
        try:
            a = _analyser(f)
        except Exception:
            a = None
        if a:
            a["mtime"] = mtime
            resultat[cle] = a
            change = True
    if change or set(cache) - set(resultat):
        historique = cache.get("_historique", [])
        resultat_cache = dict(resultat)
        resultat_cache["_historique"] = historique
        try:
            cache_f.write_text(json.dumps(resultat_cache, indent=1), encoding="utf-8")
        except OSError:
            pass
    return resultat


def _historique(dossier: Path) -> list[str]:
    try:
        return json.loads((Path(dossier) / CACHE).read_text(encoding="utf-8")).get("_historique", [])
    except Exception:
        return []


def noter_utilisee(dossier: Path, nom: str) -> None:
    f = Path(dossier) / CACHE
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    h = [x for x in data.get("_historique", []) if x != nom]
    data["_historique"] = ([nom] + h)[:5]
    try:
        f.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def _score(a: dict, duree_voulue: float, rang_historique: int | None) -> float:
    # Un gros drop qui frappe, un tempo qui se coupe bien (80–110 ou 140–175 BPM), assez long.
    t = a["tempo"]
    ecart_tempo = min(abs(t - 95), abs(t - 160), abs(t / 2 - 80), abs(t * 2 - 160))
    score = (2.0 * min(a["drop"], 3.0) + 0.3 * min(a["punch"], 6.0) + 2.5 * min(a.get("rythme", 1.0), 3.0)
             - 0.04 * ecart_tempo)
    if a.get("rythme", 1.0) < 1.3:       # pas de rythme net : mauvais pour un edit
        score -= 4
    if a["duree"] < duree_voulue + 15:
        score -= 3
    if rang_historique is not None:      # varier : la dernière chanson utilisée passe derrière
        score -= 4.0 / (rang_historique + 1)
    return score


def choisir(dossier: Path, duree_voulue: float, combien: int = 1, log=print) -> list[Path]:
    """Les meilleures chansons du dossier pour un montage de cette durée (la meilleure en premier)."""
    biblio = bibliotheque(dossier, log)
    if not biblio:
        return []
    hist = _historique(dossier)
    notes = sorted(biblio.items(), key=lambda kv: _score(kv[1], duree_voulue,
                                                          hist.index(kv[0]) if kv[0] in hist else None),
                   reverse=True)
    choix = [Path(dossier) / nom for nom, _ in notes[:combien]]
    for nom, a in notes[:combien]:
        log(f"  musique choisie : {nom} ({a['tempo']:.0f} BPM, drop {a['drop']:.2f})")
    return choix
