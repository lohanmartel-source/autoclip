"""Fabrique le code AutoClip Pro du mois, à mettre dans le post Patreon réservé aux membres.

Utilisation (depuis le dossier du projet) :
    python outils/creer_code_pro.py                      → code valide jusqu'au 7 du mois prochain
    python outils/creer_code_pro.py --expire 2026-12-31  → date choisie
    python outils/creer_code_pro.py --cle C:\\chemin\\cle_privee_autoclip.txt

La clé privée (cle_privee_autoclip.txt) ne doit JAMAIS aller sur GitHub ni être partagée :
avec elle, n'importe qui pourrait fabriquer des codes Pro.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import licence  # noqa: E402

DEFAUT_CLE = Path.home() / "Documents" / "AutoClip-prive" / "cle_privee_autoclip.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description="Code AutoClip Pro du mois")
    ap.add_argument("--expire", help="Date d'expiration AAAA-MM-JJ (défaut : le 7 du mois prochain)")
    ap.add_argument("--nom", default="", help="Étiquette dans le code, ex. « Patreon novembre »")
    ap.add_argument("--cle", default=str(DEFAUT_CLE), help="Fichier de la clé privée")
    a = ap.parse_args()
    f = Path(a.cle)
    if not f.is_file():
        print(f"Clé privée introuvable : {f}")
        return 1
    graine = bytes.fromhex(f.read_text().strip())
    if a.expire:
        expire = date.fromisoformat(a.expire)
    else:
        t = date.today()
        expire = date(t.year + (t.month == 12), t.month % 12 + 1, 7)
    code = licence.creer_code(graine, expire, a.nom or f"Patreon {date.today():%Y-%m}")
    if not licence.lire_code(code):
        print("Erreur : ce code ne passe pas la vérification (mauvaise clé ?).")
        return 1
    print(f"Code valide jusqu'au {expire} :\n\n{code}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
