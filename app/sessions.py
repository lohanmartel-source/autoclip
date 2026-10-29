"""Montages automatiques : ta dernière session de jeu et le best-of de la semaine."""
from __future__ import annotations

import json
import time
from pathlib import Path

from chemins import DONNEES

CONFIG = DONNEES / "serveur.json"
JOURNAL = DONNEES / "sessions.log"

_log_externe = None


def lire_config() -> dict:
    c = {}
    try:
        c = json.loads(CONFIG.read_text(encoding="utf-8"))
    except Exception:
        pass
    c.setdefault("clips", "")
    c.setdefault("dossier_sortie", str(Path.home() / "Videos" / "AutoClip"))
    c.setdefault("dossier_musiques", str(Path.home() / "Music" / "AutoClip"))
    c.setdefault("reglages", {})
    return c


def _log(msg: str) -> None:
    if _log_externe is not None:
        try:
            _log_externe(msg)
        except Exception:
            pass
    try:
        with open(JOURNAL, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except OSError:
        pass


def options_reglages(c: dict, **forces):
    """Options d'un montage automatique, avec les réglages mémorisés de l'app."""
    from autoclip import Options
    reg = c.get("reglages", {})
    base = dict(clips=[c["clips"]], musique=str(Path(c["dossier_musiques"])),
                sortie=str(Path(c["dossier_sortie"]) / "auto.mp4"), duree=float(reg.get("duree") or 20),
                cadrage=reg.get("cadrage") or "flou", rythme=reg.get("rythme") or "normal",
                son_jeu=(reg.get("son_jeu") if reg.get("son_jeu") is not None else 30) / 100,
                filtre="tous", jeux=reg.get("jeux") or None,
                textes_kills=reg.get("textes", True) is not False,
                sequences=reg.get("sequences", True) is not False, derniers=60,
                fps=60 if str(reg.get("fps")) == "60" else 30,
                hook=reg.get("hook", True) is not False, reactions=reg.get("voix", True) is not False,
                sous_titres=reg.get("sous_titres", True) is not False,
                style=reg.get("style") or "auto", look=reg.get("look") or "auto", compteur=reg.get("compteur", True) is not False,
                outro=str(c.get("pseudo") or "") if reg.get("outro", True) is not False else "",
                effets_sonores=reg.get("effets_sonores", True) is not False)
    base.update(forces)
    return Options(**base)


def _musiques_ok(c: dict) -> bool:
    d = Path(c["dossier_musiques"])
    return d.is_dir() and any(f.is_file() for f in d.iterdir())


def best_of(jours: int = 7, log=None) -> list[Path]:
    """Best-of des 7 derniers jours (ou 14 s'il n'y a pas assez de kills)."""
    global _log_externe
    _log_externe = log
    from autoclip import generer
    from outils import ErreurAutoclip
    c = lire_config()
    if not c.get("clips"):
        _log("Choisis d'abord le dossier de tes clips Medal.")
        return []
    if not _musiques_ok(c):
        _log("Ajoute au moins une musique.")
        return []
    sortie = Path(c["dossier_sortie"]) / f"best-of_{time.strftime('%Y-%m-%d')}.mp4"
    for essai in (jours, jours * 2):
        o = options_reglages(c, sortie=str(sortie), duree=30, depuis_jours=essai, derniers=60,
                             texte="BEST-OF DE LA SEMAINE", versions=1)
        try:
            return generer(o, log=_log)
        except ErreurAutoclip as e:
            _log(f"Pas assez de clips sur {essai} jours ({e})")
    return []


def derniere_session(index: dict, jeux: list[str] | None = None,
                     pause_max: float = 75 * 60) -> tuple[float, float, int] | None:
    """La dernière session de jeu : clips qui se suivent avec moins de 75 min de pause.
    Retourne (début, fin, nombre de kills) ou None."""
    from index_kills import est_kill
    voulus = {j.lower() for j in jeux} if jeux else None
    clips = sorted((c for c in index.get("clips", {}).values() if not voulus or c["jeu"].lower() in voulus),
                   key=lambda c: c["date"], reverse=True)
    if not clips:
        return None
    fin = debut = clips[0]["date"]
    kills = 0
    for c in clips:
        if debut - c["date"] > pause_max:
            break
        debut = c["date"]
        kills += sum(1 for k in c["kills"] if est_kill(k["nom"]))
    return debut, fin + 60, kills


def montage_session(log=None, debut: float | None = None, fin: float | None = None, **forces) -> list[Path]:
    """Montage de ta dernière session (ou de la période donnée)."""
    global _log_externe
    _log_externe = log
    from autoclip import generer
    from index_kills import charger_index
    from outils import ErreurAutoclip
    c = lire_config()
    if not c.get("clips"):
        _log("Choisis d'abord le dossier de tes clips Medal.")
        return []
    if not _musiques_ok(c):
        _log("Ajoute au moins une musique.")
        return []
    if debut is None:
        s = derniere_session(charger_index(), c.get("reglages", {}).get("jeux") or None)
        if s is None:
            _log("Aucune session trouvée.")
            return []
        debut, fin, _ = s
    sortie = Path(c["dossier_sortie"]) / f"session_{time.strftime('%Y-%m-%d_%H-%M')}.mp4"
    o = options_reglages(c, sortie=str(sortie), depuis_ts=debut, jusqu_a_ts=fin, derniers=0, **forces)
    try:
        return generer(o, log=_log)
    except ErreurAutoclip as e:
        _log(f"Montage de session impossible : {e}")
        return []
