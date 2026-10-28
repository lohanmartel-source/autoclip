"""Liens publics d'AutoClip (Patreon, site, Discord) et vérification des mises à jour.

À remplir avant de publier une version. Un lien vide cache le bouton correspondant dans l'app.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

from chemins import VERSION

LIENS = {
    "patreon": "https://www.patreon.com/c/TheAutoclip",
    "site": "",      # ex. https://autoclip.app
    "discord": "",   # ex. https://discord.gg/xxxx
}

# Dépôt GitHub où sont publiées les versions (ex. "ton-compte/autoclip"). Vide = pas de vérification.
DEPOT_GITHUB = ""

_maj: dict = {"verifie": 0.0, "version": "", "url": ""}


def _comparer(a: str, b: str) -> int:
    def n(v):
        return tuple(int(x) for x in v.lstrip("v").split(".") if x.isdigit())
    return (n(a) > n(b)) - (n(a) < n(b))


def _verifier() -> None:
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{DEPOT_GITHUB}/releases/latest",
                                     headers={"User-Agent": f"AutoClip/{VERSION}"})
        d = json.loads(urllib.request.urlopen(req, timeout=10).read())
        tag = str(d.get("tag_name", ""))
        if tag and _comparer(tag, VERSION) > 0:
            _maj.update(version=tag.lstrip("v"), url=d.get("html_url", ""))
    except Exception:
        pass


def mise_a_jour_dispo() -> dict | None:
    """{"version", "url"} si une version plus récente existe (vérifié au plus une fois par 6 h)."""
    if not DEPOT_GITHUB:
        return None
    if time.time() - _maj["verifie"] > 6 * 3600:
        _maj["verifie"] = time.time()
        threading.Thread(target=_verifier, daemon=True).start()
    if _maj["version"]:
        LIENS["mise_a_jour"] = _maj["url"]
        return {"version": _maj["version"]}
    return None
