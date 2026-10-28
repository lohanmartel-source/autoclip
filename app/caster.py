"""Commentateur : une voix de caster qui crie « ACE ! », « ONE TAP ! »… sur tes gros kills.
Voix gratuite de Microsoft Edge (edge-tts), mise en écho façon stade.
Chaque phrase est générée une seule fois puis gardée dans le dossier caster/."""
from __future__ import annotations

import asyncio
import hashlib
import random
import subprocess
from pathlib import Path

from outils import _FLAGS

from chemins import DONNEES as ICI
DOSSIER = ICI / "caster"
VOIX = {"en": "en-US-GuyNeural", "fr": "fr-CA-ThierryNeural"}
PHRASES = {
    "en": {"ACE": ["ACE!", "That's an ACE!", "ACE! Unbelievable!"], "4K": ["Four kills!", "QUADRA!"],
           "3K": ["Triple!", "Three down!"], "ONE TAP": ["One tap!", "ONE TAP!", "Headshot!"],
           "CLUTCH": ["CLUTCH!", "What a clutch!"], "KNIFE": ["With the knife!"], "5K": ["ACE!"],
           "": ["What a shot!", "Clean!", "Oh my god!"]},
    "fr": {"ACE": ["ACE !", "C'est un ACE !"], "4K": ["Quatre kills !", "Quadra !"],
           "3K": ["Triplé !", "Et de trois !"], "ONE TAP": ["One tap !", "Une balle !"],
           "CLUTCH": ["Le clutch !", "Quel clutch !"], "KNIFE": ["Au couteau !"], "5K": ["ACE !"],
           "": ["Quel tir !", "Propre !", "Incroyable !"]},
}


def _generer(texte: str, langue: str) -> Path | None:
    DOSSIER.mkdir(exist_ok=True)
    cle = hashlib.md5(f"{langue}:{texte}".encode()).hexdigest()[:12]
    wav = DOSSIER / f"{cle}.wav"
    if wav.exists():
        return wav
    try:
        import edge_tts
    except ImportError:
        return None
    mp3 = DOSSIER / f"{cle}.mp3"
    try:
        async def dire():
            await edge_tts.Communicate(texte, VOIX.get(langue, VOIX["en"]), rate="+12%", pitch="+6Hz",
                                       volume="+20%").save(str(mp3))
        asyncio.run(dire())
    except Exception:
        return None
    # Écho de stade + un peu de saturation : ça sonne « caster » plutôt que GPS.
    filtres = ("highpass=f=120,acompressor=threshold=-18dB:ratio=4:attack=5:release=80,"
               "aecho=0.8:0.7:60|120:0.35|0.2,volume=1.6,alimiter=limit=0.9")
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-af", filtres, "-ar", "48000", "-ac", "2",
                        str(wav)], capture_output=True, creationflags=_FLAGS)
    mp3.unlink(missing_ok=True)
    return wav if r.returncode == 0 and wav.exists() else None


def phrase(texte_kill: str, langue: str, rng: random.Random) -> Path | None:
    choix = PHRASES.get(langue, PHRASES["en"])
    return _generer(rng.choice(choix.get(texte_kill, choix[""])), langue)


def evenements(segments, debuts: list[float], langue: str, graine: int = 0) -> list[tuple]:
    """Le caster parle sur les plus gros moments (au plus 5 fois, jamais deux fois de suite trop près)."""
    rng = random.Random(graine)
    res, dernier = [], -9.0
    for s, d in zip(segments, debuts):
        if s.role in ("hook", "rewind", "montee", "approche") or s.t_kill is None:
            continue
        gros = s.role in ("drop", "sequence") or s.texte in ("ACE", "4K", "ONE TAP", "CLUTCH")
        if not gros:
            continue
        t = d + (s.kills_sortie[-1][0] if s.kills_sortie else s.t_kill) + 0.12
        if t - dernier < 2.5:
            continue
        texte = s.texte or ("4K" if len(s.kills_sortie) >= 4 else "3K" if len(s.kills_sortie) == 3 else "")
        f = phrase(texte, langue, rng)
        if f is not None:
            res.append(("fichier", t, str(f)))
            dernier = t
        if len(res) >= 5:
            break
    return res
