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
    """Lance une commande et lève une erreur lisible si elle échoue.
    Si elle encodait avec la carte graphique et échoue, on la relance une fois avec libx264."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_FLAGS)
    if proc.returncode != 0:
        repli = _commande_repli(cmd)
        if repli is not None:
            signaler_echec_gpu(proc.stderr.decode("utf-8", "replace"))
            return executer(repli)
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


# ---------------------------------------------------------------- encodage vidéo (processeur ou carte graphique)
#
# Tous les encodages H.264 passent par args_encodage(qualite). En Pro, « Rendu GPU » (Options.encodeur_gpu)
# fait encoder par la carte graphique : NVIDIA NVENC, sinon AMD AMF, sinon Intel Quick Sync. L'encodeur est
# détecté une fois par un vrai test d'encodage d'une image (résultat gardé en cache). En cas d'échec d'un
# encodage GPU, la commande est relancée avec libx264 et la suite du montage reste sur le processeur.

# qualite -> (preset libx264, crf, preset NVENC, qualité constante GPU)
# Mesuré sur une RTX 4060 (montage 16:9 60 i/s) : NVENC -cq 22 + AQ ≈ libx264 medium -crf 18 en PSNR,
# pour des fichiers ~1,4x plus gros (le H.264 des cartes graphiques compresse moins bien).
QUALITES_ENCODAGE = {
    "plan": ("veryfast", 16, "p5", 18),      # plans intermédiaires (réencodés ensuite)
    "finale": ("medium", 18, "p5", 22),      # vidéo finale
    "apercu": ("veryfast", 23, "p4", 26),    # aperçu rapide
    "rapide": ("superfast", 18, "p5", 22),   # réencodage par-dessus la vidéo finale (paroles)
    "rapide_apercu": ("ultrafast", 22, "p3", 26),
}
ENCODEURS_GPU = ("h264_nvenc", "h264_amf", "h264_qsv")
NOMS_ENCODEURS = {"h264_nvenc": "NVIDIA NVENC", "h264_amf": "AMD AMF", "h264_qsv": "Intel Quick Sync",
                  "libx264": "processeur (libx264)"}
_gpu = {"actif": False, "panne": False, "message": ""}
_detecte: dict[str, str | None] = {}
_repli: dict[tuple, list[str]] = {}


def _args_gpu(enc: str, qualite: str) -> list[str]:
    _, _, preset, q = QUALITES_ENCODAGE.get(qualite, QUALITES_ENCODAGE["finale"])
    if enc == "h264_nvenc":
        return ["-c:v", enc, "-preset", preset, "-tune", "hq", "-rc", "vbr", "-cq", str(q), "-b:v", "0",
                "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "20", "-pix_fmt", "yuv420p"]
    if enc == "h264_amf":
        return ["-c:v", enc, "-quality", "quality", "-rc", "cqp", "-qp_i", str(q), "-qp_p", str(q + 2),
                "-qp_b", str(q + 4), "-pix_fmt", "yuv420p"]
    return ["-c:v", enc, "-preset", "slower", "-global_quality", str(q), "-pix_fmt", "nv12"]


def _args_cpu(qualite: str) -> list[str]:
    preset, crf, _, _ = QUALITES_ENCODAGE.get(qualite, QUALITES_ENCODAGE["finale"])
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]


def _essayer_encodeur(enc: str) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=0.2", "-frames:v", "1"] \
        + _args_gpu(enc, "finale") + ["-f", "null", "-"]
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
                              creationflags=_FLAGS).returncode == 0
    except Exception:
        return False


def encodeur_gpu() -> str | None:
    """Encodeur H.264 de la carte graphique qui marche vraiment sur ce PC (NVENC, puis AMF, puis QSV), ou
    None. Test d'encodage d'une image, gardé en cache (mémoire + fichier, refait si FFmpeg change)."""
    exe = shutil.which("ffmpeg") or ""
    if exe in _detecte:
        return _detecte[exe]
    cle = f"2|{exe}"                 # 2 : version des réglages testés (_args_gpu)
    try:
        st = Path(exe).stat()
        cle = f"2|{exe}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        pass
    fichier = None
    try:
        from chemins import DONNEES
        fichier = DONNEES / "encodeur_gpu.json"
        cache = json.loads(fichier.read_text(encoding="utf-8"))
        if cache.get("cle") == cle:
            _detecte[exe] = cache.get("encodeur") or None
            return _detecte[exe]
    except Exception:
        pass
    trouve = next((e for e in ENCODEURS_GPU if _essayer_encodeur(e)), None)
    _detecte[exe] = trouve
    if fichier is not None:
        try:
            fichier.write_text(json.dumps({"cle": cle, "encodeur": trouve}), encoding="utf-8")
        except OSError:
            pass
    return trouve


def regler_gpu(actif: bool) -> str:
    """Active (ou non) l'encodage par la carte graphique pour le montage qui commence.
    Retourne l'encodeur réellement utilisé (« h264_nvenc »… ou « libx264 »)."""
    _gpu["actif"] = bool(actif)
    _gpu["panne"] = False
    _gpu["message"] = ""
    return encodeur_actuel()


def encodeur_actuel() -> str:
    if _gpu["actif"] and not _gpu["panne"]:
        return encodeur_gpu() or "libx264"
    return "libx264"


def signaler_echec_gpu(message: str = "") -> None:
    """Un encodage GPU a échoué : le reste du montage repasse sur le processeur."""
    _gpu["panne"] = True
    _gpu["message"] = message.strip().splitlines()[-1][:200] if message.strip() else ""


def args_encodage(qualite: str = "finale", gpu: bool | None = None) -> list[str]:
    """Arguments FFmpeg d'encodage vidéo H.264 (codec, réglages de qualité, -pix_fmt).
    qualite : plan / finale / apercu / rapide / rapide_apercu. gpu=None : réglage du montage en cours."""
    if gpu is None:
        gpu = _gpu["actif"]
    enc = encodeur_gpu() if gpu and not _gpu["panne"] else None
    cpu = _args_cpu(qualite)
    if not enc:
        return cpu
    args = _args_gpu(enc, qualite)
    _repli[tuple(args)] = cpu
    return args


def est_gpu(args: list[str]) -> bool:
    return tuple(args) in _repli


def _commande_repli(cmd: list[str]) -> list[str] | None:
    """La même commande avec libx264 à la place de l'encodeur GPU (None si elle n'encodait pas en GPU)."""
    for gpu, cpu in _repli.items():
        n = len(gpu)
        for i in range(len(cmd) - n + 1):
            if cmd[i] == gpu[0] and tuple(cmd[i:i + n]) == gpu:
                return cmd[:i] + cpu + cmd[i + n:]
    return None
