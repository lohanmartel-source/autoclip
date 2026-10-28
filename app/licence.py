"""AutoClip Pro : codes d'activation signés, donnés aux membres Patreon.

Un code = AUTOCLIP-PRO-<données>.<signature>, signé avec la clé privée de l'auteur (qui n'est
jamais dans l'app). L'app vérifie la signature avec la clé publique ci-dessous et la date
d'expiration. Rien n'est envoyé sur internet.

Pro débloque seulement ce qui n'utilise rien de Riot Games : les montages Valorant restent
complets dans la version gratuite.
"""
from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

import ed25519
from chemins import DONNEES

CLE_PUBLIQUE = bytes.fromhex("1ab902d4c43a60bca9dad9c88599d02d0daf83bd20f2f799e796ebb3c3cd3ed9")
PREFIXE = "AUTOCLIP-PRO-"
FICHIER = DONNEES / "licence.txt"
JEU_GRATUIT = "valorant"
MARQUE = "Fait avec AutoClip"


def _b64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def lire_code(code: str) -> dict | None:
    """Données du code s'il est authentique, sinon None (l'expiration est vérifiée à part)."""
    code = "".join(str(code).split())
    if not code.upper().startswith(PREFIXE):
        return None
    try:
        donnees, sig = code[len(PREFIXE):].split(".", 1)
        brut = _b64(donnees)
        if not ed25519.verifier(CLE_PUBLIQUE, brut, _b64(sig)):
            return None
        info = json.loads(brut)
        date.fromisoformat(info["expire"])
        return info
    except Exception:
        return None


def etat() -> dict:
    """{"actif", "expire", "nom"} selon le code enregistré."""
    try:
        code = FICHIER.read_text(encoding="utf-8").strip()
    except OSError:
        return {"actif": False, "expire": "", "nom": ""}
    info = lire_code(code)
    if not info:
        return {"actif": False, "expire": "", "nom": ""}
    debut = date.fromisoformat(info.get("debut") or "2000-01-01")
    actif = debut <= date.today() <= date.fromisoformat(info["expire"])
    return {"actif": actif, "expire": info["expire"], "nom": str(info.get("nom", ""))[:40],
            "expire_bientot": actif and (date.fromisoformat(info["expire"]) - date.today()).days <= 5}


def est_pro() -> bool:
    return etat()["actif"]


def activer(code: str) -> dict:
    info = lire_code(code)
    if not info:
        raise ValueError("Code invalide. Copie-le en entier depuis le post Patreon réservé aux membres.")
    if date.fromisoformat(info["expire"]) < date.today():
        raise ValueError(f"Ce code a expiré le {info['expire']}. Le nouveau code est sur Patreon.")
    if info.get("debut") and date.fromisoformat(info["debut"]) > date.today():
        raise ValueError(f"Ce code ne marche qu'à partir du {info['debut']}.")
    FICHIER.write_text("".join(str(code).split()) + "\n", encoding="utf-8")
    return etat()


def desactiver() -> None:
    try:
        FICHIER.unlink()
    except OSError:
        pass


def est_jeu_gratuit(nom_jeu: str) -> bool:
    return JEU_GRATUIT in str(nom_jeu).lower()


def fonctions_pro_demandees(o) -> list[str]:
    """Ce qui, dans ces options de montage, demande Pro."""
    res = []
    if o.format == "youtube":
        res.append("le format YouTube 16:9")
    if (o.versions or 1) > 1:
        res.append("les 3 versions au choix")
    if o.fps == 60:
        res.append("le 60 images/s")
    if o.jeux and any(not est_jeu_gratuit(j) for j in o.jeux):
        res.append("les autres jeux que Valorant")
    return res


def appliquer(o):
    """Ramène des options de montage à la version gratuite si Pro n'est pas actif."""
    from dataclasses import replace
    if est_pro():
        return o
    jeux = [j for j in (o.jeux or []) if est_jeu_gratuit(j)] or ["Valorant"]
    return replace(o, format="vertical" if o.format == "youtube" else o.format, versions=1,
                   fps=30, jeux=jeux, marque=True)


def creer_code(graine: bytes, expire: date, nom: str = "", debut: date | None = None) -> str:
    """Pour l'auteur seulement (outils/creer_code_pro.py) : fabrique un code signé."""
    d = {"expire": expire.isoformat(), "nom": nom}
    if debut:
        d["debut"] = debut.isoformat()
    brut = json.dumps(d, separators=(",", ":")).encode()
    sig = ed25519.signer(graine, brut)
    enc = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")
    return PREFIXE + enc(brut) + "." + enc(sig)
