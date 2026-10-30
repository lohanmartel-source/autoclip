# AutoClip

Transforme tes clips Medal de Valorant en montages TikTok / Shorts calés sur la musique, en un clic.
Gratuit, tout se passe sur ton PC.

## Pour les joueurs

1. Télécharge `AutoClip-Setup-x.y.z.exe` dans les [Releases](../../releases).
2. Lance-le. Pas besoin d'être administrateur, FFmpeg est inclus.
3. Ouvre AutoClip : il trouve tout seul ton dossier Medal. Ajoute une musique, puis « Générer le montage ».

Configuration : Windows 10 ou 11 (64 bits), Medal installé. Les kills Valorant sont lus dans la base
de données de Medal, donc chaque plan est un vrai kill.

Tes données (réglages, index des kills, styles) sont dans `%APPDATA%\AutoClip`.
Tes montages vont dans `Vidéos\AutoClip`, tes musiques dans `Musique\AutoClip`.

### Nouveautés 1.4.0 (Pro)
- **Modèles d'edits** : Paroles 16:9, Showcase skin, Phonk rapide, Ciné — tous les réglages en un clic.
- **Paroles derrière l'arme** : une IA locale (MobileSAM, ~45 Mo téléchargés la 1re fois) détoure l'arme et la main.
- **Autres jeux** : Counter-Strike 2 au niveau de Valorant (kills, headshots, aces lus dans Medal), Fortnite et Apex,
  et un mode « moments forts » pour les jeux sans kills (Minecraft…).
- **Rendu GPU** : encodage NVENC / AMF / Quick Sync, repli automatique sur le processeur.
- Paroles synchronisées par IA (Whisper) au mot près.

### Nouveautés 1.3.0
- **Style « Edit »** (par défaut) : flou de mouvement réel, speed ramps lissées, effets de beat dosés,
  plans « vibe », fondu d'ouverture, 60 fps fluide (Pro).
- **Looks d'étalonnage** : Auto (couleur de tes skins), Néon, Glace, Chaud, avec glow et vignette.
- **Intro 3D** : l'image du kill se fige, la caméra recule au milieu de tes clips, puis plonge dans le montage.
- **Paroles dans le décor** : chaque mot de la chanson en énorme, au moment où il est chanté, derrière ton arme.
  Paroles trouvées sur LRCLIB (renseigne « Artiste - Titre ») ou dans un fichier `.lrc` à côté de la musique.

## Structure du projet

| Dossier | Contenu |
|---|---|
| `app/` | Le code de l'app (Python). `app.py` ouvre la fenêtre, `serveur.py` est l'API locale, `web/index.html` l'interface. |
| `app/autoclip.py` | Le moteur de montage (aussi utilisable en ligne de commande). |
| `build/` | Tout pour fabriquer l'installateur Windows. |
| `docs/` | Le site (GitHub Pages : Settings > Pages > branche main, dossier /docs). |
| `outils/` | `creer_code_pro.py` : le code Pro du mois. |
| `.github/workflows/build.yml` | Construit l'installateur automatiquement à chaque tag `v*`. |

## Lancer depuis le code

```
cd app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Il faut FFmpeg dans le PATH (`winget install Gyan.FFmpeg`). Depuis le code, les données restent dans le
dossier `app/` (mode portable).

## Fabriquer l'installateur

**Sur ton PC :** double-clique `build\build.bat`. Il installe ce qu'il faut (PyInstaller, Inno Setup),
télécharge FFmpeg, et sort `build\sortie\AutoClip-Setup-<version>.exe`.

**Avec GitHub :** pousse un tag, l'installateur apparaît dans les Releases.
```
git tag v1.0.0
git push origin v1.0.0
```

## Avant de publier une version

1. `app/chemins.py` : `VERSION`.
2. `app/liens.py` : lien Patreon, site, Discord, et `DEPOT_GITHUB` (pour le bouton « Mettre à jour »).
3. `build/installateur.iss` et `build/version_info.txt` : même numéro de version.

## AutoClip Pro (Patreon)

La version gratuite fait des montages Valorant complets, avec un petit « Fait avec AutoClip » sur la vidéo.
Pro enlève cette marque et débloque les autres jeux, le format YouTube 16:9, les 3 versions et le 60 i/s.
Rien de ce qui touche à Valorant n'est payant (règles de Riot Games).

Les membres activent Pro avec un **code du mois** (onglet Plus de l'app). Pour le fabriquer :
```
python outils/creer_code_pro.py
```
Le code est valide jusqu'au 7 du mois suivant. Mets-le dans un post Patreon réservé aux paliers Pro.
Le script lit la clé privée dans `Documents\AutoClip-prive\cle_privee_autoclip.txt`.
**Cette clé ne doit jamais aller sur GitHub** : avec elle, n'importe qui pourrait fabriquer des codes.

## Packs de styles (bonus Patreon)

Un pack est un fichier `.json` que les membres importent depuis l'onglet Plus :
```json
{
  "pack": "Patreon Saison 1",
  "styles": [
    {"nom": "Phonk rapide", "duree_plan": 0.35, "beats_par_plan": 0.5, "sur_beat": 0.95, "flash": 0.3},
    {"nom": "Ciné lent", "duree_plan": 1.4, "beats_par_plan": 2, "sur_beat": 0.7, "flash": 0.02}
  ]
}
```
- `duree_plan` : durée moyenne d'un plan après le drop (0,15 à 4 s).
- `beats_par_plan` : nombre de beats par plan.
- `sur_beat` : part des coupes pile sur le beat (0 à 1).
- `flash` : part des coupes avec un flash (0 à 1).

Le plus simple pour en créer : dans AutoClip, « Copier le style d'une vidéo » sur un TikTok que tu aimes,
puis prends le fichier créé dans `%APPDATA%\AutoClip\styles_copies` et regroupe-les dans un pack.

## Mentions

AutoClip n'est pas affilié à Riot Games ni à Medal. Projet de fan réalisé dans le cadre de la politique
« Legal Jibber Jabber » de Riot Games. Valorant est une marque de Riot Games, Inc.
FFmpeg est distribué sous licence GPL v3.
