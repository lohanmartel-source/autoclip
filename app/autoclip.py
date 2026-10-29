"""AutoClip : transforme tes clips Medal en montage TikTok calé sur la musique.

Utilisation rapide :
    python autoclip.py --clips "D:\\Medal\\Clips" --musique son.mp3
Toutes les options :
    python autoclip.py --help
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path

from analyse_clips import FILTRES, moments_medal, sequences_medal, trouver_moments
from medal_db import charger as charger_medal, info_pour
from analyse_musique import analyser_musique
from outils import ErreurAutoclip, pair, verifier_ffmpeg
from planification import RYTHMES, planifier
from musiques import choisir as choisir_musiques, noter_utilisee
from reactions import bonus_voix, sous_titrer, trouver_reactions
from rendu import Rendu, assembler, debuts_plans, miniature, miniatures, rendre_segments

EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
# Sous-dossiers de Medal qui ne contiennent pas de clips de jeu.
DOSSIERS_IGNORES = {"temp", "thumbnails", "screenshots", "reels", "databases", "autoclip"}
POLICES = [
    "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf", "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


@dataclass
class Options:
    clips: list[str]
    musique: str
    sortie: str = "montage_autoclip.mp4"
    duree: float = 20.0
    cadrage: str = "flou"
    rythme: str = "normal"
    son_jeu: float = 0.3
    ralenti: float = 0.5
    debut_musique: float | None = None
    derniers: int = 25
    max_par_clip: int = 3
    texte: str = ""
    fps: int = 30
    apercu: bool = False
    graine: int | None = None
    garder_temp: bool = False
    filtre: str = "tous"          # tous / onetap / multi / ace
    jeux: list[str] | None = None  # ex. ["Valorant"] ; None ou vide = tous les jeux
    textes_kills: bool = True     # « ONE TAP », « ACE »… affichés sur les kills
    sequences: bool = True        # multi-kills rapides montrés en entier
    versions: int = 1             # nombre de montages différents à produire
    hook: bool = True             # teaser du meilleur kill + rewind au début
    intro_3d: bool = True         # après le teaser : le clip devient une carte en 3D (au lieu du rewind)
    reactions: bool = True        # ta voix (micro) après les kills
    sous_titres: bool = True      # sous-titres de ta voix (Whisper)
    miniature: bool = True        # image de couverture .jpg à côté de la vidéo
    depuis_jours: int | None = None  # seulement les clips des N derniers jours
    depuis_ts: float | None = None   # seulement les clips à partir de cette date (timestamp)
    jusqu_a_ts: float | None = None  # … et jusqu'à cette date (ex. une session de jeu)
    format: str = "vertical"         # vertical (TikTok/Shorts) ou youtube (16:9 long, duree = 120-180 s)
    style: str = "auto"              # auto / edit / phonk / cinematique / clean
    look: str = "auto"               # étalonnage du montage : auto / neon / glace / chaud / aucun (look.py)
    compteur: bool = True           # « KILLS 14 » à l'écran
    outro: str = ""                  # ton pseudo affiché à la fin (vide = pas d'outro)
    effets_sonores: bool = True      # whoosh, boom, riser, rewind
    legende: bool = True             # légende + hashtags (.txt) à côté de la vidéo
    armes: list[str] | None = None   # Valorant : seulement ces armes (ex. ["Operator"])
    agents: list[str] | None = None  # Valorant : seulement ces agents (ex. ["Jett"])
    selection: list | None = None    # kills choisis à la main : [{"clip": nom, "t": secondes}, …]
    drop: dict | None = None         # le kill choisi pour le drop : {"clip": nom, "t": secondes}
    chansons: list[str] | None = None  # musiques imposées (refaire un aperçu en qualité finale)
    version_seule: int | None = None   # ne rendre que cette version (0, 1, 2…)
    mode: str = "kills"              # kills ou fun (fails, rires, cris au micro)
    langue_sous_titres: str = "fr"   # fr = ce que tu dis ; en = traduit en anglais
    filigrane: str = ""              # ton pseudo en petit en haut à droite
    caster: str = ""                 # commentateur : "" (aucun), "en" ou "fr"
    instrumental: bool = True        # musique sans paroles pendant que tu parles (si Demucs est installé)
    arret_image: bool = True         # arrêt sur image + contour de l'ennemi au drop
    profil: str = ""                 # style copié d'une vidéo (nom du profil, voir styles_copies/)
    marque: bool = False             # « Fait avec AutoClip » sur la vidéo (version gratuite)
    paroles: bool = False            # paroles mot à mot dans le décor, derrière l'arme (.lrc, LRCLIB ou Whisper)
    titre_chanson: str = ""          # « Artiste - Titre » pour trouver les paroles (sinon tags / nom du fichier)
    lrc: str = ""                    # fichier .lrc ou texte LRC collé (prioritaire)
    couleur_paroles: str = ""        # couleur des paroles (#rrggbb) ; vide = couleur dominante du montage
    decalage_paroles: float = 0.0    # réglage fin manuel des paroles, en secondes (-0,5 à +0,5 ; + = plus tard)


def lister_clips(sources: list[str], derniers: int = 0) -> list[Path]:
    """Accepte des fichiers et/ou des dossiers (sous-dossiers inclus, comme ceux de Medal)."""
    fichiers: list[Path] = []
    for s in sources:
        p = Path(s).expanduser()
        if p.is_dir():
            fichiers += [f for f in p.rglob("*") if f.suffix.lower() in EXTENSIONS and f.is_file()
                         and not DOSSIERS_IGNORES.intersection(x.lower() for x in f.relative_to(p).parts[:-1])]
        elif p.is_file():
            fichiers.append(p)
        else:
            raise ErreurAutoclip(f"Introuvable : {p}")
    fichiers = sorted(set(fichiers), key=lambda f: f.stat().st_mtime, reverse=True)
    if derniers > 0:
        fichiers = fichiers[:derniers]
    if not fichiers:
        raise ErreurAutoclip("Aucun clip vidéo trouvé dans les sources données.")
    return fichiers


def nom_jeu(clip: Path) -> str:
    """Medal range les clips par jeu : Clips/<Jeu>/fichier.mp4."""
    return clip.parent.name


def _cle_jeu(nom: str) -> str:
    return "".join(c for c in nom.lower() if c.isalnum())


def jeux_disponibles(sources: list[str]) -> dict[str, int]:
    """{nom du jeu: nombre de clips}, du plus fourni au moins fourni."""
    compte: dict[str, int] = {}
    for c in lister_clips(sources):
        compte[nom_jeu(c)] = compte.get(nom_jeu(c), 0) + 1
    return dict(sorted(compte.items(), key=lambda kv: kv[1], reverse=True))


def filtrer_jeux(clips: list[Path], jeux: list[str] | None) -> list[Path]:
    if not jeux:
        return clips
    voulus = {_cle_jeu(j) for j in jeux}
    return [c for c in clips if _cle_jeu(nom_jeu(c)) in voulus]


def trouver_police() -> Path | None:
    for p in POLICES:
        if Path(p).exists():
            return Path(p)
    return None


def _texte_sans_emoji(texte: str) -> str:
    # Les polices standard n'ont pas les emojis (ils s'afficheraient en carrés).
    return "".join(c for c in texte if ord(c) <= 0xFFFF and not 0x2600 <= ord(c) <= 0x27BF).strip()


def _nom_version(sortie: Path, v: int, total: int) -> Path:
    return sortie if total == 1 else sortie.with_name(f"{sortie.stem}_v{v + 1}{sortie.suffix}")


def _preparer(o: Options, log) -> dict:
    """Étapes 1 à 3 communes : clips, moments forts, musiques, séquences."""
    log("1/4  Recherche des clips…")
    tous = filtrer_jeux(lister_clips(o.clips), o.jeux)
    if o.jeux:
        log(f"  jeux : {', '.join(o.jeux)}")
        if not tous:
            raise ErreurAutoclip(f"Aucun clip pour : {', '.join(o.jeux)}. "
                                 f"Jeux trouvés : {', '.join(jeux_disponibles(o.clips))}")
    if o.depuis_jours:
        limite = time.time() - o.depuis_jours * 86400
        tous = [c for c in tous if c.stat().st_mtime >= limite]
        log(f"  clips des {o.depuis_jours} derniers jours : {len(tous)}")
        if not tous:
            raise ErreurAutoclip(f"Aucun clip dans les {o.depuis_jours} derniers jours.")
    if o.depuis_ts or o.jusqu_a_ts:
        from index_kills import date_clip
        tous = [c for c in tous if (not o.depuis_ts or date_clip(c) >= o.depuis_ts)
                and (not o.jusqu_a_ts or date_clip(c) <= o.jusqu_a_ts)]
        log(f"  clips de la période choisie : {len(tous)}")
        if not tous:
            raise ErreurAutoclip("Aucun clip pour cette période.")
    selection = {}
    if o.selection:
        for k in o.selection:
            selection.setdefault(str(k.get("clip")), []).append(float(k.get("t", 0)))
        tous = [c for c in tous if c.name in selection]
        log(f"  {sum(len(v) for v in selection.values())} kill(s) choisi(s) à la main dans {len(tous)} clip(s)")
        if not tous:
            raise ErreurAutoclip("Les kills choisis sont introuvables (clips supprimés ?).")
    infos = charger_medal([Path(c) for c in o.clips], log)
    avec_evts = [c for c in tous if (i := info_pour(infos, c)) is not None and i.evenements]
    if infos:
        log(f"  base Medal trouvée : {len(avec_evts)} clip(s) avec kills enregistrés sur {len(tous)}")
    # Priorité aux clips où Medal a enregistré des kills, puis les autres, du plus récent au plus ancien.
    n = o.derniers if o.derniers > 0 else len(tous)
    if o.format == "youtube":
        n = max(n, 150)
    if o.filtre != "tous" or o.armes or o.agents:
        # Un filtre garde peu de kills par clip : on fouille plus loin dans l'historique.
        n = max(n * 3, 75)
        if o.filtre != "tous":
            log(f"  filtre : {FILTRES.get(o.filtre, o.filtre)}")
        if o.armes:
            log(f"  armes : {', '.join(o.armes)}")
        if o.agents:
            log(f"  agents : {', '.join(o.agents)}")
    if selection:
        n = len(avec_evts)
    clips = avec_evts[:n]
    if len(clips) < n and o.filtre == "tous" and not (o.armes or o.agents or selection):
        reste = [c for c in tous if c not in set(avec_evts)]
        clips += reste[:n - len(clips)]
    log(f"  {len(clips)} clip(s) retenu(s)")

    from index_kills import charger_index
    index = charger_index()
    if o.mode == "morts":
        log("2/4  Recherche de tes morts (coach)…")
        from morts import moments_morts
        moments = moments_morts(log, depuis=o.depuis_ts)
        if len(moments) < 2:
            raise ErreurAutoclip("Pas encore assez de morts analysées. Le serveur analyse tes clips en "
                                 "arrière-plan : réessaie dans quelques minutes.")
        return {"moments": moments, "chansons": _chansons(o, log, 1), "seqs": [],
                "source_musique": Path(o.musique).expanduser()}
    if o.mode == "fun":
        log("2/4  Recherche de tes rires et de tes cris au micro…")
        from reactions import moments_droles
        recents = tous[:max(o.derniers, 60) if o.derniers else len(tous)]
        moments = moments_droles(recents, log)
        if len(moments) < 3:
            raise ErreurAutoclip("Pas assez de moments où tu ris ou cries au micro dans ces clips. "
                                 "Analyse plus de clips (Clips récents analysés) ou une autre période.")
        return {"moments": moments, "chansons": _chansons(o, log, 1), "seqs": [],
                "source_musique": Path(o.musique).expanduser()}
    log("2/4  Analyse des clips (détection des meilleurs moments)…")
    moments, couverts = moments_medal([c for c in clips if c in set(avec_evts)], infos, log, o.filtre,
                                      index=index, armes=o.armes, agents=o.agents)
    if selection:
        def choisi(m):
            return any(m.t - 4.2 <= t <= m.t + 0.15 for t in selection.get(m.clip.name, []))
        moments = [m for m in moments if choisi(m)]
        if o.drop:
            cible = (str(o.drop.get("clip")), float(o.drop.get("t", 0)))
            haut = max((m.score for m in moments), default=0) + 5
            moments = [dc_replace(m, score=haut) if m.clip.name == cible[0] and m.t - 4.2 <= cible[1] <= m.t + 0.15
                       else m for m in moments]
    if moments:
        log(f"  {len(moments)} kill(s)/action(s) trouvés grâce à Medal")
    sans_evts = [c for c in clips if c not in couverts]
    if (o.filtre != "tous" or o.armes or o.agents or selection) and len(moments) < 3:
        raise ErreurAutoclip(f"Seulement {len(moments)} moment(s) avec ces choix. "
                             "Choisis plus de kills, un autre filtre, ou analyse plus de clips.")
    if len(moments) < 8 and sans_evts and o.filtre == "tous" and not (o.armes or o.agents or selection):
        par_son = trouver_moments(sans_evts, o.max_par_clip, log)
        if par_son:
            forts = [m for m in par_son if m.score >= 0.3 * par_son[0].score]
            par_son = forts if len(forts) >= 4 else par_son[:4]
            moments += par_son
    if not moments:
        raise ErreurAutoclip("Aucun clip exploitable.")
    if o.reactions:
        moments = bonus_voix(moments, log)
    log(f"  {len(moments)} moment(s) d'action retenu(s)")

    combien = _nb_chapitres(o) if o.format == "youtube" else max(1, o.versions)
    chansons = _chansons(o, log, combien)
    seqs = []
    if o.sequences and o.filtre in ("tous", "multi", "ace") and avec_evts and not o.armes:
        seqs = sequences_medal(avec_evts if not selection else [c for c in avec_evts if c in set(clips)],
                               infos, log=log)
        if o.agents:
            voulus = {a.lower() for a in o.agents}
            seqs = [m for m in seqs if (index["clips"].get(m.clip.name) or {}).get("agent", "").lower() in voulus]
    if o.mode == "equipe":
        from equipe import moments_amis
        amis = moments_amis(log)
        if not amis:
            raise ErreurAutoclip("Aucun clip de tes amis reçu pour l'instant. Il faut que vous soyez dans le même "
                                 "défi avec « Partager mes meilleurs kills » activé.")
        moments = sorted(moments + amis, key=lambda m: m.score, reverse=True)
    return {"moments": moments, "chansons": chansons, "seqs": seqs, "source_musique": Path(o.musique).expanduser()}


def _chansons(o: Options, log, combien: int) -> list[Path]:
    log("3/4  Analyse de la musique et planification…")
    source_musique = Path(o.musique).expanduser()
    if o.chansons:
        chansons = [Path(c) for c in o.chansons if Path(c).is_file()]
        if not chansons:
            raise ErreurAutoclip("Les musiques de l'aperçu sont introuvables.")
    elif source_musique.is_dir():
        chansons = choisir_musiques(source_musique, 30 if o.format == "youtube" else o.duree, combien, log)
        if not chansons:
            raise ErreurAutoclip(f"Aucune chanson utilisable dans {source_musique}")
    elif source_musique.is_file():
        chansons = [source_musique]
    else:
        raise ErreurAutoclip(f"Musique introuvable : {source_musique}")
    return chansons


def _nb_chapitres(o: Options) -> int:
    return max(2, round(o.duree / 30))


def _rendu_de(o: Options) -> Rendu:
    if o.format == "youtube":
        W, H = (960, 540) if o.apercu else (1920, 1080)
    else:
        W, H = (540, 960) if o.apercu else (1080, 1920)
    return Rendu(largeur=pair(W), hauteur=pair(H), fps=o.fps, cadrage=o.cadrage,
                 qualite="apercu" if o.apercu else "finale", police=trouver_police(),
                 textes_kills=o.textes_kills, style=o.style, filigrane=o.filigrane,
                 caster=o.caster, instrumental=o.instrumental, gel=o.arret_image, marque=o.marque,
                 look=o.look)


def _afficher_plan(segments, duree, chanson, rythme, fps, log):
    log(f"  Montage de {duree:.1f} s en {len(segments)} plans ({chanson.name}, rythme {rythme}) :")
    for s in segments:
        effets = ", ".join(s.effets) or "-"
        trans = f" ← {s.transition}" if s.transition else (f" ← {s.entree}" if s.entree else "")
        if s.impact:
            effets = "IMPACT " + effets
        extra = f" {s.moment.arme}" if getattr(s.moment, "arme", "") else ""
        log(f"   {s.index + 1:2d}. [{s.role:6}] {s.images / fps:4.2f}s  {s.moment}{extra}{trans}  ({effets})")


def _rendre(o: Options, r: Rendu, segments, musique, debut_musique, duree, sortie_v: Path, log,
            compteur_depart: int | None, outro: str, texte: str) -> None:
    temp = Path(tempfile.mkdtemp(prefix="autoclip_"))
    try:
        fichiers = rendre_segments(segments, temp, r, log)
        fichier_texte = None
        if texte and r.police is not None:
            fichier_texte = temp / "texte.txt"
            fichier_texte.write_text(texte, encoding="utf-8")
        reactions = []
        if o.reactions:
            reactions = trouver_reactions(segments, debuts_plans(segments, o.fps), duree, log,
                                          maximum=12 if o.mode == "fun" else 5)
            if reactions and o.sous_titres:
                sous_titrer(reactions, log, langue=o.langue_sous_titres)
        log("  assemblage final (transitions, musique, voix)…")
        assembler(segments, fichiers, musique.chemin, debut_musique, duree, sortie_v, r,
                  o.son_jeu, texte, r.police, fichier_texte, reactions=reactions, dossier_temp=temp,
                  compteur_depart=compteur_depart, outro=outro, effets_sonores=o.effets_sonores)
        if o.paroles:
            from paroles import appliquer as paroles_dans_le_decor
            paroles_dans_le_decor(sortie_v, musique.chemin, debut_musique, duree, cadrage=r.cadrage,
                                  chanson=o.titre_chanson if Path(o.musique).is_file() else "",
                                  lrc=o.lrc, couleur=o.couleur_paroles or getattr(r, "accent", "") or "", segments=segments, fps=o.fps,
                                  qualite=r.qualite, log=log, decalage=o.decalage_paroles)
        if o.miniature:
            if miniatures(segments, r, sortie_v, texte, temp):
                log(f"  3 miniatures au choix : {sortie_v.stem}_mini1..3.jpg")
    finally:
        if o.garder_temp:
            log(f"  fichiers temporaires gardés dans {temp}")
        else:
            shutil.rmtree(temp, ignore_errors=True)


def noter(segments) -> float:
    """Note sur 10 d'un montage, comme un monteur le jugerait : la force des kills (surtout au drop
    et sur les coups forts), un multi-kill en entier, la variété des clips et des transitions."""
    plans = [s for s in segments if s.role not in ("hook", "rewind", "intro3d", "approche")]
    if not plans:
        return 0.0
    force = sum(s.moment.score - 10 for s in plans) / len(plans)          # 0 à ~5
    drop = next((s.moment.score - 10 for s in segments if s.role == "drop"), 0)
    sur_coups_forts = sum(1 for s in plans if s.impact and s.moment.score >= 12.4) / max(1, sum(1 for s in plans if s.impact))
    sequence = 1.0 if any(s.role == "sequence" for s in segments) else 0.0
    variete = len({s.moment.clip for s in plans}) / len(plans)
    transitions = len({s.entree or s.transition for s in plans}) / max(1, len(plans))
    note = (3.0 * min(1.0, max(0.0, force / 4.5)) + 1.5 * min(1.0, max(0.0, drop / 4.5)) + 1.5 * sur_coups_forts
            + 1.0 * sequence + 2.0 * variete + 1.0 * min(1.0, transitions * 1.5))
    return round(min(10.0, note), 1)


def _sauver_infos(sortie_v: Path, o: Options, graine0: int, v: int, chansons: list[Path], legende_txt: str,
                  note: float | None = None, titres: list[str] | None = None) -> None:
    """À côté de la vidéo : la légende (.txt) et de quoi refaire la même vidéo en qualité finale (.json)."""
    from dataclasses import asdict
    try:
        if legende_txt:
            sortie_v.with_suffix(".txt").write_text(legende_txt, encoding="utf-8")
        d = asdict(o)
        sortie_v.with_suffix(".json").write_text(json.dumps(
            {"graine": graine0, "version": v, "chansons": [str(c) for c in chansons], "cree": time.time(),
             "note": note, "titres": titres or [], "options": d}, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def generer(o: Options, log=print, plan_seulement: bool = False) -> list[Path]:
    """Fait le(s) montage(s). Retourne la liste des vidéos produites (une par version)."""
    from legende import legende
    import licence
    debut_chrono = time.time()
    verifier_ffmpeg()
    o = licence.appliquer(o)            # version gratuite : Valorant, vertical, 1 version, 30 i/s, marque
    if o.paroles:
        # Paroles dans le décor : pas de textes de kill, de compteur ni de sous-titres du micro en plus
        # (l'image respire, et deux textes à la fois se lisent mal).
        o = dc_replace(o, textes_kills=False, compteur=False, sous_titres=False)
    if o.mode == "morts":
        # Coach : plans longs à vitesse normale, le son du jeu bien présent, musique discrète.
        o = dc_replace(o, rythme="fun", hook=False, sequences=False, son_jeu=max(o.son_jeu, 0.8),
                       filtre="tous", format="vertical", ralenti=1.0, style="clean", compteur=False,
                       caster="", arret_image=False, versions=1, look="aucun")
    if o.mode == "fun":
        # Fails et rires : plans longs, pas de teaser, ta voix en avant, plus de son du jeu.
        o = dc_replace(o, rythme="fun", hook=False, sequences=False, reactions=True,
                       son_jeu=max(o.son_jeu, 0.5), filtre="tous", format="vertical", ralenti=1.0)
    sortie = Path(o.sortie).expanduser().resolve()
    sortie.parent.mkdir(parents=True, exist_ok=True)
    ctx = _preparer(o, log)
    moments, chansons, seqs, source_musique = ctx["moments"], ctx["chansons"], ctx["seqs"], ctx["source_musique"]
    r = _rendu_de(o)
    profil = None
    if o.profil:
        from copie_style import charger as charger_profil
        profil = charger_profil(o.profil)
        if profil:
            log(f"  style copié : {profil['nom']} (plans de {profil['duree_plan']} s, {int(profil['flash'] * 100)} % de flashs)")
    graine0 = o.graine if o.graine is not None else random.randrange(1_000_000)
    texte = _texte_sans_emoji(o.texte) if o.texte else ""
    if o.format == "youtube":
        return _generer_youtube(o, ctx, r, graine0, sortie, texte, log, plan_seulement, debut_chrono, profil)

    analyses_musique: dict[Path, object] = {}
    drops_utilises: set = set()
    sorties: list[Path] = []
    notes: list[float] = []
    total = max(1, o.versions)

    for v in range(total):
        rendre_celle_ci = o.version_seule is None or o.version_seule == v
        if total > 1 and rendre_celle_ci:
            log(f"=== Version {v + 1}/{total} ===")
        chanson = chansons[v % len(chansons)]
        if chanson not in analyses_musique:
            analyses_musique[chanson] = analyser_musique(chanson, o.duree, o.debut_musique,
                                                         log if rendre_celle_ci else (lambda *_: None),
                                                         beats_intro=4 if o.hook else 0)
        musique = analyses_musique[chanson]
        rng = random.Random(graine0 + 7919 * v)
        # Chaque version mélange un peu l'ordre des kills pour ne pas refaire le même montage.
        moments_v = moments if v == 0 else [dc_replace(m, score=m.score + rng.uniform(-0.9, 0.9)) for m in moments]
        if v > 0:
            # Chaque version a son propre « moment du drop » : le meilleur kill d'un clip pas encore utilisé.
            candidat = next((m for m in sorted(moments_v, key=lambda m: m.score, reverse=True)
                             if m.clip not in drops_utilises), None)
            if candidat is not None:
                haut = max(m.score for m in moments_v) + 1
                moments_v = [dc_replace(m, score=haut) if m is candidat else m for m in moments_v]
        sequence = seqs[v % len(seqs)] if seqs else None
        # Chaque version a aussi son propre rythme de coupes (ton choix pour la version 1).
        autres = [x for x in ("nerveux", "normal", "calme") if x != o.rythme]
        rythme_v = o.rythme if v == 0 or o.mode == "fun" else autres[(v - 1) % len(autres)]
        segments, debut_musique, duree = planifier(musique, moments_v, o.fps, rythme_v, o.ralenti,
                                                   graine0 + 101 * v, sequence=sequence, hook=o.hook,
                                                   style=o.style, gel=o.arret_image, profil=profil,
                                                   intro_3d=o.intro_3d)
        drops_utilises.update(s.moment.clip for s in segments if s.role == "drop")
        if not rendre_celle_ci:
            continue
        _afficher_plan(segments, duree, chanson, rythme_v, o.fps, log)
        if plan_seulement:
            continue

        sortie_v = _nom_version(sortie, v, total) if o.version_seule is None else sortie
        log("4/4  Rendu vidéo…")
        _rendre(o, r, segments, musique, debut_musique, duree, sortie_v, log,
                compteur_depart=0 if o.compteur else None, outro=o.outro, texte=texte)
        if source_musique.is_dir() and not o.apercu:
            noter_utilisee(source_musique, chanson.name)
        note = noter(segments)
        notes.append(note)
        from legende import titres as titres_possibles
        _sauver_infos(sortie_v, o, graine0, v, [chanson], legende(segments, o) if o.legende else "", note,
                      titres_possibles(segments, o))
        sorties.append(sortie_v)
        log(f"  note du montage : {note}/10")
        log(f"Version prête → {sortie_v}" if total > 1 else f"Vidéo prête → {sortie_v}")

    if len(sorties) > 1:
        # La meilleure version en premier (c'est elle que l'app montre).
        ordre = sorted(range(len(sorties)), key=lambda i: notes[i], reverse=True)
        log(f"Meilleure : version {ordre[0] + 1} ({notes[ordre[0]]}/10)")
        sorties = [sorties[i] for i in ordre]
    log(f"Terminé en {time.time() - debut_chrono:.0f} s → {sorties[0] if sorties else '(plan seulement)'}")
    return sorties


def _generer_youtube(o: Options, ctx: dict, r: Rendu, graine0: int, sortie: Path, texte: str, log,
                     plan_seulement: bool, debut_chrono: float, profil: dict | None = None) -> list[Path]:
    """Version YouTube 16:9 longue : plusieurs « chapitres » de ~30 s (chacun avec sa chanson, son drop
    et ses kills, sans jamais reprendre un kill), enchaînés en fondu, avec le compteur qui continue."""
    from legende import legende
    from planification import Segment  # noqa: F401  (type)
    moments = list(ctx["moments"])
    chansons, seqs, source_musique = ctx["chansons"], ctx["seqs"], ctx["source_musique"]
    nb = _nb_chapitres(o)
    log(f"  version YouTube : {nb} chapitres d'environ 30 s")
    utilises: set = set()
    compte = 0
    chapitres, tous_segments = [], []
    temp_global = Path(tempfile.mkdtemp(prefix="autoclip_yt_"))
    try:
        for c in range(nb):
            dispo = [m for m in moments if (m.clip, round(m.t, 1)) not in utilises]
            if len(dispo) < 6:
                log(f"  plus assez de kills pour un chapitre {c + 1} : on s'arrête à {c}")
                break
            log(f"=== Chapitre {c + 1}/{nb} ===")
            chanson = chansons[c % len(chansons)]
            musique = analyser_musique(chanson, 30, None, log, beats_intro=4 if (o.hook and c == 0) else 0)
            sequence = seqs[c] if c < len(seqs) else None
            if sequence is not None and (sequence.clip, round(sequence.t, 1)) in utilises:
                sequence = None
            rythmes = [o.rythme, "normal", "nerveux", o.rythme, "calme"]
            segments, debut_musique, duree = planifier(musique, dispo, o.fps, rythmes[c % len(rythmes)], o.ralenti,
                                                       graine0 + 101 * c, sequence=sequence,
                                                       hook=o.hook and c == 0, style=o.style,
                                                       gel=o.arret_image, profil=profil,
                                                       intro_3d=o.intro_3d)
            for s in segments:
                if s.role not in ("hook", "rewind", "intro3d"):
                    utilises.add((s.moment.clip, round(s.moment.t, 1)))
            _afficher_plan(segments, duree, chanson, rythmes[c % len(rythmes)], o.fps, log)
            tous_segments += segments
            if plan_seulement:
                continue
            fichier = temp_global / f"chapitre_{c:02d}.mp4"
            dernier = c == nb - 1 or len([m for m in moments if (m.clip, round(m.t, 1)) not in utilises]) < 6
            o_chap = dc_replace(o, miniature=(c == 0))
            _rendre(o_chap, r, segments, musique, debut_musique, duree, fichier, log,
                    compteur_depart=compte if o.compteur else None,
                    outro=o.outro if dernier else "", texte=texte if c == 0 else "")
            from rendu import evenements_montage
            compte += len([t for t in evenements_montage(segments, o.fps)["compteur"] if t < duree])
            chapitres.append((fichier, duree))
            if source_musique.is_dir() and not o.apercu:
                noter_utilisee(source_musique, chanson.name)
            if dernier:
                break
        if plan_seulement or not chapitres:
            return []
        log("  assemblage des chapitres…")
        _enchainer(chapitres, sortie, r)
        mini = temp_global / "chapitre_00.jpg"
        if mini.exists():
            shutil.copy2(mini, sortie.with_suffix(".jpg"))
        for v in range(1, 4):
            f = temp_global / f"chapitre_00_mini{v}.jpg"
            if f.exists():
                shutil.copy2(f, sortie.with_name(f"{sortie.stem}_mini{v}.jpg"))
    finally:
        if not o.garder_temp:
            shutil.rmtree(temp_global, ignore_errors=True)
    from legende import titres as titres_possibles
    _sauver_infos(sortie, o, graine0, 0, chansons, legende(tous_segments, o) if o.legende else "",
                  titres=titres_possibles(tous_segments, o))
    log(f"Vidéo YouTube prête ({sum(d for _, d in chapitres):.0f} s, {compte} kills) → {sortie}")
    log(f"Terminé en {time.time() - debut_chrono:.0f} s → {sortie}")
    return [sortie]


def _enchainer(chapitres: list[tuple[Path, float]], sortie: Path, r: Rendu, fondu: float = 0.5) -> None:
    from outils import executer
    if len(chapitres) == 1:
        shutil.copy2(chapitres[0][0], sortie)
        return
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for f, _ in chapitres:
        cmd += ["-i", str(f)]
    parties, cv, ca, longueur = [], "0:v", "0:a", chapitres[0][1]
    for i in range(1, len(chapitres)):
        parties.append(f"[{cv}][{i}:v]xfade=transition=fadeblack:duration={fondu}:offset={longueur - fondu:.3f}[v{i}]")
        parties.append(f"[{ca}][{i}:a]acrossfade=d={fondu}[a{i}]")
        cv, ca = f"v{i}", f"a{i}"
        longueur += chapitres[i][1] - fondu
    preset, crf = ("veryfast", "23") if r.qualite == "apercu" else ("medium", "18")
    executer(cmd + ["-filter_complex", ";".join(parties), "-map", f"[{cv}]", "-map", f"[{ca}]",
                    "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p", "-r", str(r.fps),
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(sortie)])


def main(argv=None) -> int:
    for flux in (sys.stdout, sys.stderr):
        try:
            flux.reconfigure(errors="replace")   # console Windows : pas de plantage sur « ≈ » ou « → »
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Montage automatique TikTok à partir de tes clips Medal.")
    ap.add_argument("--clips", nargs="+", required=False, help="Dossier(s) de clips Medal ou fichiers vidéo")
    ap.add_argument("--musique", required=False,
                    help="Fichier audio, ou un DOSSIER de musiques (AutoClip choisit la meilleure)")
    ap.add_argument("--sortie", default="montage_autoclip.mp4", help="Fichier vidéo produit")
    ap.add_argument("--duree", type=float, default=20, help="Durée visée en secondes (défaut 20)")
    ap.add_argument("--cadrage", choices=["flou", "crop"], default="flou",
                    help="flou = image entière sur fond flou ; crop = plein écran recadré au centre")
    ap.add_argument("--rythme", choices=list(RYTHMES), default="normal", help="Vitesse des coupes")
    ap.add_argument("--son-jeu", type=float, default=0.3, help="Volume du son du jeu sous la musique (0 à 1)")
    ap.add_argument("--ralenti", type=float, default=0.5, help="Vitesse du ralenti au drop (1 = pas de ralenti)")
    ap.add_argument("--debut-musique", type=float, default=None,
                    help="Début de l'extrait dans la chanson (s). Par défaut : calé avant le drop")
    ap.add_argument("--derniers", type=int, default=25, help="Nombre de clips récents à analyser (0 = tous)")
    ap.add_argument("--max-par-clip", type=int, default=3, help="Moments max retenus par clip")
    ap.add_argument("--texte", default="", help="Texte d'accroche affiché au début")
    ap.add_argument("--filtre", choices=list(FILTRES), default="tous",
                    help="Kills à montrer : tous, onetap, multi (3K+), ace (ace/4K/clutch)")
    ap.add_argument("--jeux", nargs="+", default=None, metavar="JEU",
                    help="Seulement ces jeux (nom du dossier Medal), ex. --jeux Valorant")
    ap.add_argument("--liste-jeux", action="store_true", help="Affiche les jeux trouvés et quitte")
    ap.add_argument("--sans-sequence", action="store_true",
                    help="Ne pas montrer en entier les multi-kills rapides (ex. 4 kills en 3 s)")
    ap.add_argument("--sans-textes", action="store_true", help="Pas de texte ONE TAP / ACE sur les kills")
    ap.add_argument("--versions", type=int, default=1, help="Nombre de montages différents (ex. 3)")
    ap.add_argument("--sans-hook", action="store_true", help="Pas de teaser + rewind au début")
    ap.add_argument("--sans-intro-3d", action="store_true",
                    help="Après le teaser, le rewind classique au lieu de l'intro 3D (cartes de clips)")
    ap.add_argument("--sans-voix", action="store_true", help="Ne pas garder tes réactions au micro")
    ap.add_argument("--sans-sous-titres", action="store_true", help="Pas de sous-titres sur ta voix")
    ap.add_argument("--sans-miniature", action="store_true", help="Pas d'image de couverture")
    ap.add_argument("--depuis", type=int, default=None, metavar="JOURS",
                    help="Seulement les clips des N derniers jours")
    ap.add_argument("--best-of", action="store_true", help="Best-of de la semaine avec les réglages de l'app")
    ap.add_argument("--format", choices=["vertical", "youtube"], default="vertical",
                    help="vertical = TikTok/Shorts ; youtube = 16:9 long (mets --duree 150)")
    ap.add_argument("--style", choices=["auto", "edit", "phonk", "cinematique", "clean"], default="auto",
                    help="edit = façon edits TikTok (flou de mouvement, speed ramps, plans vibe, bloom)")
    ap.add_argument("--look", choices=["auto", "neon", "glace", "chaud", "aucun"], default=None,
                    help="Étalonnage du montage (auto = selon la couleur du skin). Défaut : auto en style edit, aucun sinon")
    ap.add_argument("--sans-compteur", action="store_true", help="Pas de compteur KILLS à l'écran")
    ap.add_argument("--outro", default="", help="Ton pseudo affiché à la fin")
    ap.add_argument("--sans-effets-sonores", action="store_true", help="Pas de whoosh / boom")
    ap.add_argument("--armes", nargs="+", default=None, help="Valorant : seulement ces armes, ex. Operator")
    ap.add_argument("--agents", nargs="+", default=None, help="Valorant : seulement ces agents, ex. Jett")
    ap.add_argument("--indexer", action="store_true", help="Analyse tous les kills (armes, agents, one taps)")
    ap.add_argument("--fps", type=int, default=30, choices=[30, 60])
    ap.add_argument("--apercu", action="store_true", help="Rendu rapide en basse résolution")
    ap.add_argument("--graine", type=int, default=None, help="Fixe le choix aléatoire des effets")
    ap.add_argument("--plan", action="store_true", help="Affiche le plan du montage sans faire le rendu")
    ap.add_argument("--garder-temp", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--paroles", action="store_true",
                    help="Paroles mot à mot dans le décor, derrière l'arme (.lrc à côté de la musique, LRCLIB ou Whisper)")
    ap.add_argument("--titre-chanson", default="", metavar="\"ARTISTE - TITRE\"",
                    help="Chanson à chercher sur LRCLIB (sinon tags ou nom du fichier)")
    ap.add_argument("--lrc", default="", metavar="FICHIER", help="Paroles synchronisées (.lrc) à utiliser")
    ap.add_argument("--decalage-paroles", type=float, default=0.0, metavar="SECONDES",
                    help="Réglage fin des paroles, de -0.5 à +0.5 s (+ = les mots apparaissent plus tard)")
    a = ap.parse_args(argv)
    if a.best_of:
        import sessions
        return 0 if sessions.best_of(log=print) else 1
    if not a.clips:
        ap.error("--clips est obligatoire")
    if a.indexer:
        from index_kills import indexer
        n = indexer(a.clips[0])
        print(f"{n} clip(s) analysé(s)")
        return 0

    o = Options(clips=a.clips, musique=a.musique, sortie=a.sortie, duree=a.duree, cadrage=a.cadrage,
                rythme=a.rythme, son_jeu=a.son_jeu, ralenti=a.ralenti, debut_musique=a.debut_musique,
                derniers=a.derniers, max_par_clip=a.max_par_clip, texte=a.texte, fps=a.fps,
                apercu=a.apercu, graine=a.graine, garder_temp=a.garder_temp, filtre=a.filtre,
                jeux=a.jeux, textes_kills=not a.sans_textes, sequences=not a.sans_sequence,
                versions=max(1, a.versions), hook=not a.sans_hook, intro_3d=not a.sans_intro_3d,
                reactions=not a.sans_voix,
                sous_titres=not a.sans_sous_titres, miniature=not a.sans_miniature, depuis_jours=a.depuis,
                format=a.format, style=a.style, look=a.look or ("auto" if a.style == "edit" else "aucun"), compteur=not a.sans_compteur, outro=a.outro,
                effets_sonores=not a.sans_effets_sonores, armes=a.armes, agents=a.agents,
                paroles=a.paroles or bool(a.lrc), titre_chanson=a.titre_chanson, lrc=a.lrc,
                decalage_paroles=max(-0.5, min(0.5, a.decalage_paroles)))
    if a.liste_jeux:
        for jeu, n in jeux_disponibles(a.clips).items():
            print(f"{jeu:30} {n} clip(s)")
        return 0
    if not a.musique:
        ap.error("--musique est obligatoire")
    try:
        generer(o, plan_seulement=a.plan)
    except ErreurAutoclip as e:
        print(f"\nErreur : {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
