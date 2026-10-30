"""Profils de jeu : ce qui change d'un jeu à l'autre (AutoClip Pro pour les jeux autres que Valorant).

Medal range les clips par jeu (Clips/<Jeu>/…). À partir du nom du dossier on sait :
  - si Medal enregistre des kills pour ce jeu (Valorant, CS2, Fortnite, Apex…) ou s'il faut un
    montage « moments forts » (pics du son, action, mouvement) : Minecraft, Forza, Ghost Recon… ;
  - comment reconnaître les menus du jeu (tableau des scores, achat, inventaire) : image_propre ;
  - où chercher la couleur dominante pour le look auto (l'arme en bas à droite dans un FPS, le
    centre de l'image ailleurs) et s'il faut ignorer les murs beiges des maps ;
  - les hashtags de la légende.
La porte Pro reste licence.est_jeu_gratuit : ce module ne décide rien de ce qui est payant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Profil:
    cle: str                      # valorant / cs2 / fortnite / apex / fps / autre
    kills: bool = True            # Medal enregistre (ou peut enregistrer) des kills pour ce jeu
    vue_fps: bool = True          # vue à la 1re personne, arme en bas à droite
    murs_chauds: bool = True      # maps aux murs beiges/oranges : ignorés par le look auto
    contour_ennemi: bool = False  # surbrillance jaune des ennemis (arrêt sur image)
    menus: str = "generique"      # heuristique de détection des menus (mouvement.image_propre)
    zoom_crop: float = 1.0        # cadrage « crop » : zoom en plus pour sortir le HUD du bord de l'image
    ancre_crop: float = 0.5       # … et où couper en hauteur (0.5 = centre, 0.85 = surtout en haut)
    hashtags: tuple = field(default_factory=tuple)


PROFILS = {
    "valorant": Profil("valorant", contour_ennemi=True, menus="valorant",
                       hashtags=("#valorant", "#valorantclips", "#valorantmontage", "#valorantedit")),
    "cs2": Profil("cs2", menus="cs2", zoom_crop=1.15, ancre_crop=0.85, hashtags=("#cs2", "#counterstrike2", "#cs2clips", "#csgo")),
    "fortnite": Profil("fortnite", vue_fps=False, murs_chauds=False, menus="generique",
                       hashtags=("#fortnite", "#fortniteclips", "#fortnitemontage", "#fortniteedit")),
    "apex": Profil("apex", menus="generique", hashtags=("#apexlegends", "#apex", "#apexclips", "#apexmontage")),
    "fps": Profil("fps", menus="generique"),
    "autre": Profil("autre", kills=False, vue_fps=False, murs_chauds=False, menus="generique"),
}

# Nom de dossier Medal (en minuscules, lettres et chiffres seulement) -> profil.
_MOTS = [
    ("valorant", "valorant"), ("counterstrike", "cs2"), ("cs2", "cs2"), ("csgo", "cs2"),
    ("fortnite", "fortnite"), ("apex", "apex"),
    # Autres tirs : Medal y enregistre souvent des kills ; même traitement générique que Fortnite/Apex.
    ("callofduty", "fps"), ("warzone", "fps"), ("overwatch", "fps"), ("rainbowsix", "fps"), ("siege", "fps"),
    ("battlefield", "fps"), ("thefinals", "fps"), ("deltaforce", "fps"), ("pubg", "fps"), ("halo", "fps"),
    ("destiny", "fps"), ("escapefromtarkov", "fps"), ("rust", "fps"), ("marvelrivals", "fps"),
]


def cle_nom(nom: str) -> str:
    return "".join(c for c in str(nom).lower() if c.isalnum())


def profil_nom(nom_jeu: str) -> Profil:
    n = cle_nom(nom_jeu)
    for mot, cle in _MOTS:
        if mot in n:
            return PROFILS[cle]
    return PROFILS["autre"]


def profil(clip: Path | str) -> Profil:
    """Profil du jeu d'un clip (dossier parent rangé par Medal)."""
    return profil_nom(Path(clip).parent.name)


def hashtags(nom_jeu: str) -> list[str]:
    p = profil_nom(nom_jeu)
    if p.hashtags:
        return list(p.hashtags)
    c = cle_nom(nom_jeu)
    return [f"#{c}", f"#{c}clips", f"#{c}edit"] if c else []
