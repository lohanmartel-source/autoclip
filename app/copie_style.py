"""Copier le style d'une vidéo (un TikTok que tu aimes) : AutoClip mesure son rythme de coupes
(en beats de sa musique), la part de coupes pile sur le beat et ses flashs, puis refait tes montages
avec le même rythme. Les profils sont gardés dans styles_copies/ (la vidéo n'est pas gardée)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from outils import _FLAGS, infos_video

from chemins import DONNEES as ICI
DOSSIER = ICI / "styles_copies"


def _images(video: Path, fps: int = 30) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps},scale=64:36,format=gray",
           "-f", "rawvideo", "-"]
    b = subprocess.run(cmd, capture_output=True, creationflags=_FLAGS, timeout=300).stdout
    n = len(b) // (64 * 36)
    return np.frombuffer(b[:n * 64 * 36], np.uint8).reshape(n, 36, 64).astype(np.float32)


def analyser(video: Path, nom: str) -> dict:
    fps = 30
    img = _images(video, fps)
    if len(img) < fps * 3:
        raise ValueError("Vidéo trop courte (3 s minimum).")
    lum = img.mean(axis=(1, 2))
    diff = np.abs(np.diff(img, axis=0)).mean(axis=(1, 2))
    seuil = max(18.0, float(np.median(diff) * 4))
    coupes = [i + 1 for i in range(len(diff)) if diff[i] > seuil and (i == 0 or diff[i] > diff[i - 1])]
    propres = []
    for c in coupes:                       # au moins 3 images entre deux coupes
        if not propres or c - propres[-1] >= 3:
            propres.append(c)
    # Flash : la luminosité saute très haut pendant moins de 5 images.
    flashs = 0
    fond = np.median(lum)
    for c in propres:
        fen = lum[c:c + 5]
        if len(fen) and fen.max() > min(250, fond + 70):
            flashs += 1
    duree = len(img) / fps
    t_coupes = np.array(propres) / fps
    # Beats de la musique de la vidéo.
    periode, sur_beat = 0.5, 0.0
    try:
        import librosa
        from outils import decoder_audio
        y = decoder_audio(video, 22050)
        if len(y) > 22050 * 3:
            tempo, beats = librosa.beat.beat_track(y=y, sr=22050, units="time")
            tempo = float(np.atleast_1d(tempo)[0])
            if tempo > 0:
                periode = 60.0 / tempo
            if len(beats) and len(t_coupes):
                d = np.min(np.abs(t_coupes[:, None] - np.asarray(beats)[None, :]), axis=1)
                sur_beat = float((d < 0.07).mean())
    except Exception:
        pass
    plans = np.diff(np.concatenate([[0.0], t_coupes, [duree]]))
    plans = plans[plans > 0.08]
    duree_plan = float(np.median(plans)) if len(plans) else duree
    profil = {
        "nom": nom[:30], "duree_video": round(duree, 1), "coupes": len(propres),
        "duree_plan": round(duree_plan, 2), "beats_par_plan": round(duree_plan / periode, 2),
        "tempo": round(60 / periode), "sur_beat": round(sur_beat, 2),
        "flash": round(flashs / max(1, len(propres)), 2),
    }
    DOSSIER.mkdir(exist_ok=True)
    (DOSSIER / f"{_slug(nom)}.json").write_text(json.dumps(profil, ensure_ascii=False, indent=1), encoding="utf-8")
    return profil


def _slug(nom: str) -> str:
    return re.sub(r"[^\w\-]", "_", nom.strip().lower())[:30] or "style"


def profils() -> list[dict]:
    if not DOSSIER.is_dir():
        return []
    res = []
    for f in sorted(DOSSIER.glob("*.json")):
        try:
            res.append({**json.loads(f.read_text(encoding="utf-8")), "id": f.stem})
        except Exception:
            continue
    return res


def charger(id_: str) -> dict | None:
    f = DOSSIER / f"{_slug(id_)}.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def facteur_rythme(profil: dict) -> float:
    """Facteur de rythme pour le planificateur : un plan après le drop dure ~0,7 s × facteur.
    On vise la durée de plan médiane de la vidéo copiée."""
    return float(np.clip(profil.get("duree_plan", 0.7) / 0.7, 0.35, 4.0))
