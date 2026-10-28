"""Outils communs : appels FFmpeg / FFprobe et décodage audio."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import chemins  # noqa: F401  (met le FFmpeg livré avec l'app dans le PATH)

# Sous Windows, évite qu'une fenêtre console s'ouvre à chaque appel FFmpeg (mode interface).
_FLAGS = 0x08000000 if sys.platform == "win32" else 0


class ErreurAutoclip(Exception):
    """Erreur affichée telle quelle à l'utilisateur."""


def verifier_ffmpeg() -> None:
    for prog in ("ffmpeg", "ffprobe"):
        if shutil.which(prog) is None:
            raise ErreurAutoclip(
                f"« {prog} » est introuvable. Réinstalle AutoClip : FFmpeg est inclus dans l'installateur."
            )


def executer(cmd: list[str]) -> subprocess.CompletedProcess:
    """Lance une commande et lève une erreur lisible si elle échoue."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_FLAGS)
    if proc.returncode != 0:
        lignes = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ErreurAutoclip("Échec de FFmpeg :\n" + "\n".join(lignes[-15:]))
    return proc


def infos_video(chemin: Path) -> dict:
    """Durée, taille et présence d'audio d'un fichier vidéo."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(chemin)]
    data = json.loads(executer(cmd).stdout.decode("utf-8", "replace") or "{}")
    flux = data.get("streams", [])
    video = next((s for s in flux if s.get("codec_type") == "video"), None)
    if video is None:
        raise ErreurAutoclip(f"Aucune piste vidéo dans {chemin.name}")
    duree = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    return {
        "duree": duree,
        "largeur": int(video.get("width", 0)),
        "hauteur": int(video.get("height", 0)),
        "a_audio": any(s.get("codec_type") == "audio" for s in flux),
    }


def decoder_audio(chemin: Path, sr: int) -> np.ndarray:
    """Décode la piste audio en mono float32 (tableau vide si aucune piste audio)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(chemin), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_FLAGS)
    if proc.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def pair(x: float) -> int:
    """Arrondit à l'entier pair le plus proche (exigé par l'encodeur H.264)."""
    n = int(round(x))
    return n - (n % 2)


def chemin_filtre(p: Path) -> str:
    """Échappe un chemin de fichier pour l'utiliser dans un filtre FFmpeg (Windows compris)."""
    s = str(Path(p).resolve()).replace("\\", "/")
    return s.replace(":", "\\:").replace("'", "\\'")


_cache_flux: dict[str, dict] = {}


def flux_audio(chemin: Path) -> dict:
    """Pistes audio d'un clip par nom (Medal : « All Audio », « All PC Audio », « Microphone »).
    Retourne {"jeu": index du son du jeu SANS micro (ou piste par défaut), "micro": index ou None}."""
    cle = str(chemin)
    if cle in _cache_flux:
        return _cache_flux[cle]
    res = {"jeu": None, "micro": None}
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
               "stream=index:stream_tags=name,title,handler_name", "-print_format", "json", str(chemin)]
        flux = json.loads(executer(cmd).stdout.decode("utf-8", "replace") or "{}").get("streams", [])
        for s in flux:
            tags = {k.lower(): str(v).lower() for k, v in (s.get("tags") or {}).items()}
            nom = tags.get("name") or tags.get("title") or tags.get("handler_name") or ""
            if "micro" in nom or nom == "mic":
                res["micro"] = s["index"]
            elif "pc audio" in nom or "game" in nom:
                res["jeu"] = s["index"]
        if res["jeu"] is None and flux:
            # Pas de piste « jeu seul » : on prend la première piste qui n'est pas le micro.
            res["jeu"] = next((s["index"] for s in flux if s["index"] != res["micro"]), None)
    except Exception:
        pass
    _cache_flux[cle] = res
    return res
