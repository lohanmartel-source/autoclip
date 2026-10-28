"""Défi entre amis : classement de la semaine (kills, aces, 4K, one taps) avec tes potes qui ont AutoClip.

Pas de compte ni de serveur à installer : chaque AutoClip publie ses stats de la semaine sur un « sujet »
ntfy commun (le code du défi), et lit celles des autres. Seules ces stats sont envoyées (pseudo, nombres),
jamais tes vidéos. Toute personne qui connaît le code peut voir le classement : partage-le seulement
avec tes amis.
"""
from __future__ import annotations

import json
import secrets
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

NTFY = "https://ntfy.sh"


def nouveau_code() -> str:
    return "defi-" + secrets.token_hex(4)


def code_valide(code: str) -> bool:
    return code.startswith("defi-") and 9 <= len(code) <= 40 and all(c.isalnum() or c == "-" for c in code)


def semaine_actuelle() -> tuple[str, float]:
    """(« 2026-S43 », timestamp du lundi 5 h) : une soirée qui finit après minuit compte encore pour la veille."""
    maintenant = datetime.now() - timedelta(hours=5)
    lundi = (maintenant - timedelta(days=maintenant.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    an, sem, _ = maintenant.isocalendar()
    return f"{an}-S{sem:02d}", (lundi + timedelta(hours=5)).timestamp()


def mes_scores(index: dict, pseudo: str, jeux: list[str] | None = None) -> dict:
    from index_kills import statistiques
    semaine, debut = semaine_actuelle()
    t = statistiques(index, jeux, jours=1, depuis=debut)["total"]
    return {"pseudo": pseudo, "semaine": semaine, "kills": t["kills"], "aces": t["aces"], "4k": t["4k"],
            "one_taps": t["one_taps"], "matchs": t["matchs"], "maj": int(time.time()), "app": "autoclip"}


def publier(code: str, scores: dict) -> bool:
    if not code_valide(code) or not scores.get("pseudo"):
        return False
    try:
        req = urllib.request.Request(f"{NTFY}/autoclip-{code}?{urllib.parse.urlencode({'tags': 'trophy'})}",
                                     data=json.dumps(scores).encode("utf-8"), method="POST")
        urllib.request.urlopen(req, timeout=20).read()
        return True
    except Exception:
        return False


def lire(code: str, connus: dict | None = None) -> dict:
    """Derniers scores de chaque joueur pour la semaine en cours. connus : scores déjà reçus (gardés même
    si le PC d'un ami est éteint depuis plus de 12 h, la durée de conservation de ntfy)."""
    semaine, _ = semaine_actuelle()
    scores = {p: s for p, s in (connus or {}).items() if s.get("semaine") == semaine}
    if not code_valide(code):
        return scores
    try:
        brut = urllib.request.urlopen(f"{NTFY}/autoclip-{code}/json?poll=1&since=12h", timeout=20).read()
    except Exception:
        return scores
    for ligne in brut.decode("utf-8", "replace").splitlines():
        try:
            m = json.loads(ligne)
            s = json.loads(m.get("message") or "{}")
        except (ValueError, TypeError):
            continue
        if s.get("app") != "autoclip" or s.get("semaine") != semaine or not s.get("pseudo"):
            continue
        p = str(s["pseudo"])[:30]
        propre = {k: (int(s.get(k) or 0) if k not in ("pseudo", "semaine", "app") else s.get(k))
                  for k in ("pseudo", "semaine", "kills", "aces", "4k", "one_taps", "matchs", "maj")}
        propre["pseudo"] = p
        if p not in scores or propre["maj"] >= scores[p].get("maj", 0):
            scores[p] = propre
    return scores


def classement(scores: dict) -> list[dict]:
    """Points : 1 par kill, 3 par one tap, 5 par 4K, 10 par ace."""
    res = []
    for s in scores.values():
        pts = s["kills"] + 3 * s["one_taps"] + 5 * s["4k"] + 10 * s["aces"]
        res.append({**s, "points": pts})
    res.sort(key=lambda s: s["points"], reverse=True)
    return res
