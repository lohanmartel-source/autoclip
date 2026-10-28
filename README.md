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

## Structure du projet

| Dossier | Contenu |
|---|---|
| `app/` | Le code de l'app (Python). `app.py` ouvre la fenêtre, `serveur.py` est l'API locale, `web/index.html` l'interface. |
| `app/autoclip.py` | Le moteur de montage (aussi utilisable en ligne de commande). |
| `build/` | Tout pour fabriquer l'installateur Windows. |
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
