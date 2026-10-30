"""Paroles dans le décor : chaque mot chanté apparaît seul, en énorme, DERRIÈRE l'arme.

1. Trouver les paroles avec leurs temps, dans cet ordre :
   a) un fichier .lrc à côté de la musique (ou du texte LRC donné dans les réglages) ;
   b) LRCLIB (https://lrclib.net), base publique et gratuite de paroles synchronisées
      (titre et artiste : réglage « Artiste - Titre », sinon les tags du fichier, sinon son nom) ;
   c) sinon, LRCLIB est interrogé avec le refrain entendu par Whisper.
   Synchro IA (synchro_paroles.py) : Whisper (faster-whisper, sur le PC) entend chaque mot dans TON
   fichier ; le texte LRC (bonne orthographe) est aligné sur ces mots, d'où un temps PAR MOT, même
   si la version YouTube est décalée ou coupée. Sans texte : les mots de Whisper directement.
   Repli (pas de modèle, pas d'internet) : les lignes LRC sont découpées en mots (durée répartie
   selon les syllabes), puis chaque mot est recalé sur l'attaque (onset) la plus proche.
   Décalage manuel de ±0,5 s possible (réglage fin de secours).
2. Les mots sont regroupés comme dans les edits TikTok (petits mots collés au mot plein :
   « OF THE NIGHT », interjections retirées, mots-clés seulement dans les lignes chargées, vide
   entre les lignes), puis placés à gauche / à droite / en haut, jamais sur le réticule.
   Dessin : FFmpeg drawtext (police Anton, OFL) puis numpy : lettres claires translucides qui
   prennent un peu la texture de la scène, halo doux de la couleur d'accent du look, ombre
   légère ; apparition en fondu avec un flou de profondeur. Styles : neon (défaut), verre, accent.
3. Le viewmodel (arme + mains, fixe à l'écran) est repéré plan par plan : quand la caméra bouge,
   le décor suit le mouvement global alors que l'arme reste à la même place à l'écran. On compare,
   pour chaque pixel, l'erreur « fixe à l'écran » et l'erreur « collée au monde » (mouvement
   estimé par corrélation de phase). Ce qui colle à l'écran dans la moitié basse = premier plan.
   Le masque reçoit une note de fiabilité (taille, position, stabilité, assez d'indices) : s'il
   n'est pas sûr, on ne s'en sert pas (un masque faux découpe les lettres) et le mot va ailleurs.
   Le mot suit doucement le mouvement de la caméra.
4. Passe de post-traitement : la vidéo montée est relue (pipe rawvideo), composée, réencodée ;
   le son est copié tel quel.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from chemins import DONNEES, RESSOURCES, VERSION
from outils import _FLAGS, chemin_filtre

POLICE = RESSOURCES / "polices" / "anton.ttf"
CACHE = DONNEES / "paroles_cache"
API = "https://lrclib.net/api"
DELAI_RESEAU = 4.0          # secondes : l'app ne doit jamais rester bloquée sur internet
NEGATIF_VALIDE = 3 * 86400   # une chanson introuvable n'est pas redemandée avant 3 jours


@dataclass
class Mot:
    texte: str
    debut: float   # secondes (dans la chanson, puis dans le montage)
    fin: float
    ligne: int = 0  # numéro de la ligne (phrase) : sert à grouper les mots et à respirer entre les lignes


# ─────────────────────────────── 1. Trouver les paroles ───────────────────────────────

_TEMPS = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")
_TEMPS_MOT = re.compile(r"<(\d+):(\d+(?:[.:]\d+)?)>")
_OFFSET = re.compile(r"\[offset:\s*([+-]?\d+)\]", re.I)


def _secondes(mn: str, s: str) -> float:
    return int(mn) * 60 + float(s.replace(":", "."))


def lire_lrc(texte: str) -> list[tuple[float, str]]:
    """Lignes (temps, texte) d'un LRC, triées. Gère [offset:], plusieurs temps par ligne
    et le LRC « enrichi » (<mm:ss.xx> devant chaque mot, gardé dans le texte)."""
    decalage = 0.0
    m = _OFFSET.search(texte)
    if m:
        decalage = int(m.group(1)) / 1000.0
    lignes = []
    for brut in texte.splitlines():
        temps = _TEMPS.findall(brut)
        if not temps:
            continue
        contenu = _TEMPS.sub("", brut).strip()
        for mn, s in temps:
            lignes.append((max(0.0, _secondes(mn, s) - decalage), contenu))
    lignes.sort(key=lambda x: x[0])
    return lignes


def _syllabes(mot: str) -> int:
    return max(1, len(re.findall(r"[aeiouyàâäéèêëîïôöûùüœæ]+", mot.lower())))


def _nettoyer_mot(mot: str) -> str:
    mot = mot.replace("’", "'").replace("`", "'")
    mot = re.sub("[\u115f\u1160\u3164\uffa0\u200b-\u200f\u2060\ufeff]", "", mot)   # caractères invisibles (LRCLIB)
    mot = re.sub(r"[^\w'\-!?&$]", "", mot, flags=re.UNICODE).strip("-'")
    return mot.upper()[:16]


def mots_depuis_lignes(lignes: list[tuple[float, str]], fin_chanson: float) -> list[Mot]:
    """Découpe chaque ligne en mots. Durée de la ligne répartie au prorata des syllabes
    (le chanteur ne remplit pas forcément tout l'espace jusqu'à la ligne suivante)."""
    mots: list[Mot] = []
    for i, (t, contenu) in enumerate(lignes):
        suivant = lignes[i + 1][0] if i + 1 < len(lignes) else fin_chanson
        if not contenu or suivant <= t:
            continue
        if _TEMPS_MOT.search(contenu):               # LRC enrichi : temps de chaque mot donnés
            morceaux = re.split(r"(<\d+:\d+(?:[.:]\d+)?>)", contenu)
            tmot = t
            for m in morceaux:
                mt = _TEMPS_MOT.fullmatch(m)
                if mt:
                    tmot = _secondes(*mt.groups())
                    continue
                for w in m.split():
                    w = _nettoyer_mot(w)
                    if w:
                        mots.append(Mot(w, tmot, tmot, i))
                        tmot += 0.01
            continue
        brut = [_nettoyer_mot(w) for w in contenu.split()]
        brut = [w for w in brut if w]
        if not brut:
            continue
        syl = np.array([_syllabes(w) for w in brut], dtype=float)
        duree = min((suivant - t) * 0.92, syl.sum() * 0.30 + 0.25)
        bornes = t + duree * np.concatenate([[0.0], np.cumsum(syl) / syl.sum()])
        for w, a in zip(brut, bornes[:-1]):
            mots.append(Mot(w, float(a), float(a), i))
    # Fin de chaque mot : le mot suivant (ou au plus 0,9 s).
    for i, m in enumerate(mots):
        nxt = mots[i + 1].debut if i + 1 < len(mots) else m.debut + 0.9
        m.fin = float(min(nxt, m.debut + 0.9))
    return [m for m in mots if m.fin - m.debut > 0.03]


def recaler(mots: list[Mot], y: np.ndarray, sr: int, origine: float, rayon: float = 0.11) -> int:
    """Recale chaque début de mot sur l'attaque (voix ou percussion) la plus proche (± rayon).
    `y` commence au temps `origine` de la chanson. Retourne le nombre de mots déplacés."""
    import librosa
    if not mots or len(y) < sr:
        return 0
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", backtrack=False) + origine
    except Exception:
        return 0
    if not len(onsets):
        return 0
    bouges = 0
    precedent = -1e9
    for m in mots:
        j = int(np.searchsorted(onsets, m.debut))
        proches = [onsets[k] for k in (j - 1, j) if 0 <= k < len(onsets)]
        o = min(proches, key=lambda x: abs(x - m.debut)) if proches else None
        if o is not None and abs(o - m.debut) <= rayon and o > precedent + 0.07:
            m.debut = float(o)
            bouges += 1
        m.debut = max(m.debut, precedent + 0.07)
        precedent = m.debut
    for i, m in enumerate(mots):
        nxt = mots[i + 1].debut if i + 1 < len(mots) else m.fin
        m.fin = float(max(m.debut + 0.08, min(m.fin, nxt) if nxt > m.debut else m.fin))
    return bouges


def _tags(musique: Path) -> tuple[str, str]:
    """(artiste, titre) depuis les tags ID3 / MP4 / Vorbis du fichier (ffprobe)."""
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags:stream_tags",
                            "-of", "json", str(musique)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           creationflags=_FLAGS, timeout=10)
        d = json.loads(p.stdout or b"{}")
        tags = {}
        for bloc in [d.get("format", {})] + d.get("streams", []):
            for k, v in (bloc.get("tags") or {}).items():
                tags.setdefault(k.lower(), str(v).strip())
        return tags.get("artist", "") or tags.get("album_artist", ""), tags.get("title", "")
    except Exception:
        return "", ""


def _duree_fichier(musique: Path) -> float:
    try:
        p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                            "default=nw=1:nk=1", str(musique)], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, creationflags=_FLAGS, timeout=10)
        return float(p.stdout.strip() or 0)
    except Exception:
        return 0.0


# Marques ajoutées par les sites de téléchargement (« Elley-Duhe-SnapYT.App », « y2mate.com - … »).
_TELECHARGEURS = (r"snap\s*yt([\s.]*app)?|y2mate(\.com|\.is)?|savefrom(\.net)?|ytmp3(\.cc)?|ssyoutube|"
                  r"yt1s(\.com)?|mp3juices|notube(\.net)?|converter|download(ed)?")


def _propre(s: str) -> str:
    s = re.sub(rf"[\s_\-.]*\b({_TELECHARGEURS})\b[\s_\-.]*", " ", s, flags=re.I)
    s = re.sub(r"[\(\[][^\)\]]*(official|lyrics?|audio|video|clip|remaster|hd|4k|visuali[sz]er|paroles)"
               r"[^\)\]]*[\)\]]", "", s, flags=re.I)
    s = re.sub(r"\s+(ft\.?|feat\.?)\s.*$", "", s, flags=re.I)
    return re.sub(r"[_\s]+", " ", s).strip(" -")


def artiste_titre(musique: Path, chanson: str = "") -> tuple[str, str]:
    """« Artiste - Titre » donné dans l'app, sinon les tags, sinon le nom du fichier."""
    for source in (chanson, None, Path(musique).stem):
        if source is None:
            a, t = _tags(musique)
            if t:
                return _propre(a), _propre(t)
            continue
        if not source:
            continue
        source = re.sub(rf"^[\s_\-.]*\b({_TELECHARGEURS})\b[\s_\-.]*", "", source, flags=re.I)
        morceaux = re.split(r"\s+[-–—]\s+|\s*[–—]\s*", source, maxsplit=1)
        if len(morceaux) == 2 and all(x.strip() for x in morceaux):
            return _propre(morceaux[0]), _propre(morceaux[1])
        if source is chanson:
            return "", _propre(source)
        # Nom de fichier sans « - » clair (ex. « Elley-Duhe ») : recherche libre, la durée tranchera.
        libre = _propre(re.sub(r"[-_.]+", " ", source))
        if libre:
            return "", libre
    return "", ""


def _get_json(url: str):
    req = urllib.request.Request(url, headers={
        "User-Agent": f"AutoClip/{VERSION} (montages Valorant ; paroles synchronisees)",
        "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=DELAI_RESEAU) as rep:
        return json.loads(rep.read().decode("utf-8", "replace"))


def lrclib(artiste: str, titre: str, duree: float = 0.0, log=print) -> str | None:
    """Paroles synchronisées (texte LRC) depuis LRCLIB, avec cache dans DONNEES/paroles_cache."""
    if not titre:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    cle = hashlib.sha1(f"{artiste.lower()}|{titre.lower()}".encode()).hexdigest()[:20]
    fichier = CACHE / f"{cle}.json"
    try:
        c = json.loads(fichier.read_text(encoding="utf-8"))
        if c.get("lrc") or time.time() - c.get("quand", 0) < NEGATIF_VALIDE:
            return c.get("lrc") or None
    except (OSError, ValueError):
        pass
    q = urllib.parse.urlencode
    lrc, joint = None, False
    essais = []
    if artiste:
        essais.append(f"{API}/get?" + q({"artist_name": artiste, "track_name": titre}))
        essais.append(f"{API}/search?" + q({"artist_name": artiste, "track_name": titre}))
    essais.append(f"{API}/search?" + q({"q": f"{artiste} {titre}".strip()}))
    for url in essais:
        try:
            d = _get_json(url)
            joint = True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                joint = True
                continue
            # Erreur serveur (503…) : une 2e tentative, et surtout pas de cache négatif.
            try:
                time.sleep(1.0)
                d = _get_json(url)
                joint = True
            except Exception:
                log(f"  paroles : LRCLIB a répondu {e.code}, réessaie plus tard")
                joint = False
                break
        except Exception as e:
            log(f"  paroles : LRCLIB injoignable ({type(e).__name__})")
            joint = False
            break
        candidats = d if isinstance(d, list) else [d]
        candidats = [x for x in candidats if isinstance(x, dict) and x.get("syncedLyrics")]
        if duree > 0:
            candidats.sort(key=lambda x: abs(float(x.get("duration") or 0) - duree))
            # Recherche libre : refuser une chanson dont la durée ne colle pas (mauvais morceau).
            tolerance = 8.0 if artiste else 4.0
            candidats = [x for x in candidats if not x.get("duration")
                         or abs(float(x["duration"]) - duree) <= tolerance]
        if candidats:
            lrc = candidats[0]["syncedLyrics"]
            break
    if joint:   # on ne met en cache que les vraies réponses (pas les coupures réseau)
        try:
            fichier.write_text(json.dumps({"artiste": artiste, "titre": titre, "lrc": lrc or "",
                                           "quand": time.time()}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return lrc


def _ngrammes_repetes(cles: list[str], n_min: int = 3, n_max: int = 5) -> list[str]:
    """Suites de mots entendues plusieurs fois (le refrain, qui contient souvent le titre)."""
    compte: dict[tuple, int] = {}
    for n in range(n_min, n_max + 1):
        for i in range(len(cles) - n + 1):
            g = tuple(cles[i:i + n])
            compte[g] = compte.get(g, 0) + 1
    rep = sorted((g for g, c in compte.items() if c >= 2), key=lambda g: (-compte[g] * len(g), g))
    res: list[str] = []
    for g in rep:
        s = " ".join(g)
        if not any(s in r for r in res):
            res.append(s)
    return res


def lrclib_ecoute(musique: Path, libre: str, entendus, duree: float, log=print) -> str | None:
    """LRCLIB ne trouve rien avec le nom du fichier (« Elley-Duhe-SnapYT.App ») : on cherche avec ce
    que Whisper a entendu (le refrain contient souvent le titre). Une réponse n'est gardée que si ses
    paroles s'alignent vraiment sur les mots entendus (au moins la moitié)."""
    import synchro_paroles as sp
    cles = [sp.normaliser(w.texte) for w in entendus if w.proba >= 0.3]
    cles = [c for c in cles if c]
    if len(cles) < 6:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    fichier = CACHE / f"ecoute_{sp.empreinte(musique)}.json"
    try:
        c = json.loads(fichier.read_text(encoding="utf-8"))
        if c.get("lrc") or time.time() - c.get("quand", 0) < NEGATIF_VALIDE:
            return c.get("lrc") or None
    except (OSError, ValueError):
        pass
    refrains = _ngrammes_repetes(cles)[:2] or [" ".join(cles[:5])]
    requetes = [f"{libre} {refrains[0]}".strip(), refrains[0]] + [f"{libre} {r}".strip() for r in refrains[1:]]
    vus, meilleur, score_max, joint = set(), None, 0.0, False
    for q in list(dict.fromkeys(requetes))[:3]:
        try:
            d = _get_json(f"{API}/search?" + urllib.parse.urlencode({"q": q}))
            joint = True
        except Exception:
            continue
        for x in (d if isinstance(d, list) else [])[:6]:
            lrc = x.get("syncedLyrics") if isinstance(x, dict) else None
            if not lrc or lrc in vus:
                continue
            vus.add(lrc)
            if duree > 0 and x.get("duration") and abs(float(x["duration"]) - duree) > 25:
                continue
            ref = [(m.texte, m.ligne, m.debut) for m in mots_depuis_lignes(lire_lrc(lrc), duree or 600)]
            _, st = sp.synchroniser(ref, entendus)
            score = st["alignes"] / max(1, len(cles))
            if score > score_max:
                meilleur, score_max = (lrc, f"{x.get('artistName', '')} - {x.get('trackName', '')}"), score
        if score_max >= 0.6:
            break
    lrc = meilleur[0] if meilleur and score_max >= 0.5 else None
    if lrc:
        log(f"  paroles : trouvées sur LRCLIB d'après le refrain entendu ({meilleur[1]}, "
            f"{score_max:.0%} des mots concordent)")
    if joint:
        try:
            fichier.write_text(json.dumps({"lrc": lrc or "", "quand": time.time()}, ensure_ascii=False),
                               encoding="utf-8")
        except OSError:
            pass
    return lrc


def synchro_ia(musique: Path, debut: float, duree: float, lignes: list[tuple[float, str]],
               log=print, reseau: bool = True, libre: str = "") -> tuple[list[Mot] | None, str]:
    """Temps PAR MOT entendus par Whisper dans ce fichier (voir synchro_paroles.py).
    Retourne (mots en temps de la chanson, source) ; (None, "") si Whisper ou son modèle manquent,
    ou si l'alignement n'est pas fiable (l'appelant garde alors l'ancienne méthode)."""
    try:
        import synchro_paroles as sp
    except Exception:
        return None, ""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return None, ""
    from outils import decoder_audio
    fin = debut + duree
    try:
        y = decoder_audio(musique, sp.SR)
        prompt = sp.prompt_paroles(lignes, debut, fin) if lignes else ""
        langue = sp.langue_texte(" ".join(c for _, c in lignes)) if lignes else None
        entendus = sp.transcrire(musique, debut - sp.MARGE, fin + sp.MARGE, prompt, langue, log=log, y=y)
        if entendus is None:
            return None, ""
        if not lignes and reseau:
            lrc = lrclib_ecoute(musique, libre, entendus, len(y) / sp.SR, log)
            if lrc:
                lignes = lire_lrc(lrc)
                # Nouvelle transcription guidée par les paroles trouvées (plus précise), en cache ensuite.
                prompt = sp.prompt_paroles(lignes, debut, fin)
                langue = sp.langue_texte(" ".join(c for _, c in lignes))
                ent2 = sp.transcrire(musique, debut - sp.MARGE, fin + sp.MARGE, prompt, langue, log=log, y=y)
                entendus = ent2 if ent2 else entendus
        if lignes:
            tous = mots_depuis_lignes(lignes, fin_chanson=len(y) / sp.SR + 5)
            ref = [(m.texte, m.ligne, m.debut) for m in tous]
            sync, st = sp.synchroniser(ref, entendus)
            n_ent = sum(1 for w in entendus if debut <= w.debut < fin and sp.normaliser(w.texte))
            dans = [m for m in sync if debut <= m.debut < fin]
            n_al = sum(1 for m in dans if m.source in ("aligne", "substitue"))
            if n_ent >= 4 and n_al < 0.35 * n_ent:
                log(f"  paroles : les paroles trouvées ne collent pas à ce qui est chanté "
                    f"({n_al}/{n_ent} mots) : synchro IA ignorée")
                return None, ""
            dec = st.get("decalage")
            log(f"  paroles : synchro IA, {n_al} mots calés sur le chant, "
                f"{sum(1 for m in dans if m.source == 'interpole')} interpolés"
                + (f" (version décalée de {dec:+.1f} s par rapport à LRCLIB)" if dec is not None and abs(dec) > 0.5 else ""))
            mots = [Mot(m.texte, m.debut, m.fin, m.ligne) for m in sync]
            return mots, "LRCLIB + IA"
        sync = sp.filtrer_entendus(entendus)
        mots = [Mot(_nettoyer_mot(m.texte), m.debut, m.fin, m.ligne) for m in sync]
        mots = [m for m in mots if m.texte]
        return (mots or None), "IA"
    except Exception as e:
        log(f"  paroles : synchro IA impossible ({type(e).__name__}: {str(e)[:160]})")
        return None, ""


def _lrc_source(lrc: str, musique: Path) -> tuple[str | None, str]:
    """Texte LRC à utiliser + d'où il vient. `lrc` : chemin d'un fichier ou texte LRC collé."""
    if lrc:
        if "[" in lrc and "]" in lrc and ("\n" in lrc or _TEMPS.search(lrc)):
            return lrc, "texte LRC donné"
        p = Path(lrc).expanduser()
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace"), p.name
    for p in (Path(musique).with_suffix(".lrc"), Path(musique).with_suffix(".LRC")):
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace"), p.name
    return None, ""


def trouver_mots(musique: Path, debut: float, duree: float, chanson: str = "", lrc: str = "",
                 log=print, reseau: bool = True, decalage: float = 0.0, ia: bool = True) -> list[Mot]:
    """Mots de l'extrait [debut, debut + duree] de la chanson, en temps du MONTAGE (0 = début).
    `decalage` (s) : réglage fin manuel, positif = les mots apparaissent plus tard.
    `ia` : synchro par Whisper (temps par mot) ; sinon / en repli, temps LRC par ligne + prorata."""
    musique = Path(musique)
    texte, source = _lrc_source(lrc, musique)
    mots: list[Mot] | None = None
    a = t = ""
    if texte is None and reseau:
        a, t = artiste_titre(musique, chanson)
        if t:
            texte = lrclib(a, t, _duree_fichier(musique), log)
            source = f"LRCLIB ({a + ' - ' if a else ''}{t})"
    lignes = lire_lrc(texte) if texte else []
    if ia:
        mots, src = synchro_ia(musique, debut, duree, lignes, log, reseau, libre=f"{a} {t}".strip())
        if mots:
            source = src
    if not mots and lignes:
        if ia:
            log("  paroles : synchro IA indisponible, ancienne méthode (temps par ligne, moins précise)")
        mots = mots_depuis_lignes(lignes, fin_chanson=debut + duree + 5)
    if not mots:
        return []
    mots = [m for m in mots if m.debut >= debut - 0.05 and m.debut < debut + duree]   # pas de mot coupé au début
    if not mots:
        log(f"  paroles ({source}) : aucune parole dans l'extrait choisi")
        return []
    if "IA" not in source:
        try:
            from outils import decoder_audio
            sr = 22050
            y = decoder_audio(musique, sr)
            a = int(max(0.0, debut - 0.5) * sr)
            n = recaler(mots, y[a:int((debut + duree + 0.5) * sr)], sr, a / sr)
            log(f"  paroles ({source}) : {len(mots)} mots, {n} recalés sur la musique")
        except Exception:
            log(f"  paroles ({source}) : {len(mots)} mots")
    else:
        log(f"  paroles ({source}) : {len(mots)} mots")
    decalage = float(np.clip(decalage or 0.0, -0.5, 0.5)) if np.isfinite(decalage or 0.0) else 0.0
    if decalage:
        log(f"  paroles : décalage manuel {decalage:+.2f} s")
    return [Mot(m.texte, max(0.0, m.debut - debut + decalage), max(0.05, m.fin - debut + decalage), m.ligne)
            for m in mots]


# ─────────────────────────────── 2. Dessiner les mots ───────────────────────────────

_option_script = None


def _option_filtre_fichier() -> str:
    """Option pour lire un filtre depuis un fichier : « -/filter:v » depuis FFmpeg 7
    (« -filter_script » a été retiré), « -filter_script:v » pour les versions plus anciennes."""
    global _option_script
    if _option_script is None:
        try:
            v = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                               creationflags=_FLAGS, timeout=20).stdout
            m = re.search(r"ffmpeg version n?(\d+)\.", v)
            _option_script = "-filter_script:v" if m and int(m.group(1)) < 7 else "-/filter:v"
        except Exception:
            _option_script = "-/filter:v"
    return _option_script


def _dessiner(mots: list[str], tailles: list[int], largeur: int, hauteur: int, dossier: Path) -> np.ndarray:
    """Rend chaque mot (blanc sur noir) dans une image à part : tableau (N, hauteur, largeur) uint8."""
    largeur, hauteur = largeur + largeur % 2, hauteur + hauteur % 2
    filtres = []
    for i, (m, t) in enumerate(zip(mots, tailles)):
        f = dossier / f"mot_{i:03d}.txt"
        f.write_text(m, encoding="utf-8")
        filtres.append(f"drawtext=fontfile='{chemin_filtre(POLICE)}':textfile='{chemin_filtre(f)}'"
                       f":expansion=none:fontsize={t}:fontcolor=white:x={int(t * 0.15)}:y={int(t * 0.15)}"
                       f":enable='eq(n,{i})'")
    script = dossier / "mots.txt"
    script.write_text(",".join(["format=gray"] + filtres), encoding="utf-8")
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
           f"color=c=black:s={largeur}x{hauteur}:r=1:d={len(mots)}",
           _option_filtre_fichier(), str(script), "-frames:v", str(len(mots)),
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_FLAGS)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "replace")[-400:])
    return np.frombuffer(p.stdout, np.uint8).reshape(len(mots), hauteur, largeur)


def _boite(a: np.ndarray, seuil: int = 8):
    ys = np.flatnonzero(a.max(axis=1) > seuil)
    xs = np.flatnonzero(a.max(axis=0) > seuil)
    if not len(ys) or not len(xs):
        return None
    return ys[0], ys[-1] + 1, xs[0], xs[-1] + 1


def _couleur_hex(c: str) -> np.ndarray | None:
    c = (c or "").strip().lstrip("#")
    if re.fullmatch(r"[0-9a-fA-F]{6}", c):
        return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255
    return None


def couleur_dominante(images: np.ndarray) -> np.ndarray:
    """Couleur vive la plus présente (images RVB uint8 basse définition). Repli : bleu glacé."""
    x = images[::3].reshape(-1, 3).astype(np.float32) / 255
    mx, mn = x.max(1), x.min(1)
    sat = (mx - mn) / (mx + 1e-6)
    poids = sat * mx * ((sat > 0.5) & (mx > 0.35))
    if poids.sum() < 50:
        return np.array([0.55, 0.85, 1.0], np.float32)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    teinte = (np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) + 360) % 360
    h, _ = np.histogram(teinte, bins=24, range=(0, 360), weights=poids)
    h = h + 0.5 * (np.roll(h, 1) + np.roll(h, -1))
    k = int(np.argmax(h))
    sel = (np.abs(((teinte - (k + 0.5) * 15) + 180) % 360 - 180) < 22) & (poids > 0)
    c = (x[sel] * poids[sel, None]).sum(0) / (poids[sel].sum() + 1e-6)
    c = c / (c.max() + 1e-6)                       # couleur pleine et lumineuse
    return np.clip(0.15 + 0.85 * c, 0, 1).astype(np.float32)


# ───────────────────────── 2b. Groupes de mots, importance, place ─────────────────────────

# Petits mots : jamais seuls à l'écran, ils s'accrochent au mot plein qui suit (« OF THE NIGHT »).
PETITS = set("""THE A AN OF IN ON TO AT AND OR BUT I I'M YOU ME MY YOUR IT IT'S IS ARE BE WE SO FOR WITH AS BY
FROM UP OUT IF THAT THIS DO DON'T CAN JUST ALL NO NOT HE SHE HIS HER OUR THEY THEM WAS WILL
LE LA LES L' DE DES DU D' UN UNE ET OU JE J' TU IL ELLE ON NOUS VOUS ILS ELLES À AU AUX EN CE ÇA
SE SA SON SES MA MON MES TA TON TES ME TE QUI QUE QU' NE N' Y S' C' M' T' EST PAS""".split())
# Interjections : on ne les écrit pas (ce sont des respirations).
INTERJECTIONS = set("AH OH OOH UH UM MM HMM HM NA EH HA WOO WOAH WHOA OOOH AAH".split())


@dataclass
class Groupe:
    texte: str
    debut: float
    fin: float
    ligne: int
    importance: float = 0.5    # 0-1 : taille du texte
    place: str = ""
    fiable: bool = False       # masque de l'arme assez sûr pendant ce groupe
    centre_ia: tuple | None = None   # place trouvée derrière l'arme découpée par IA (arme_ia.py)


def _petit(mot: str) -> bool:
    return mot in PETITS or (len(mot) <= 2 and mot not in ("GO", "UP"))


def grouper(mots: list[Mot]) -> list[Groupe]:
    """Mots → groupes affichables : interjections retirées, petits mots collés au mot plein voisin,
    au plus 3 mots / 14 lettres, au moins ~0,4 s à l'écran, et du vide entre les lignes."""
    lignes: dict[int, list[Mot]] = {}
    for m in mots:
        if m.texte.strip("!?") in INTERJECTIONS:
            continue
        lignes.setdefault(m.ligne, []).append(m)
    groupes: list[Groupe] = []
    for num in sorted(lignes):
        ws = lignes[num]
        bruts: list[list[Mot]] = []
        attente: list[Mot] = []
        for w in ws:
            attente.append(w)
            if not _petit(w.texte):
                bruts.append(attente)
                attente = []
        if attente:                                  # petits mots en fin de ligne : avec le précédent
            if bruts and len(bruts[-1]) + len(attente) <= 3:
                bruts[-1] += attente
            elif any(not _petit(w.texte) or len(w.texte) > 2 for w in attente) or not bruts:
                bruts.append(attente)
        # Trop longs → on garde les 3 derniers mots (« IN THE MIDDLE OF » → « THE MIDDLE OF »… rare).
        propres = []
        for g in bruts:
            while len(g) > 3 or len(" ".join(w.texte for w in g)) > 15:
                if len(g) == 1:
                    break
                if not _petit(g[0].texte):
                    propres.append([g[0]])      # mot plein : affiché seul
                g = g[1:]                         # petit mot de tête : sauté
            propres.append(g)
        # Groupes trop courts (< 0,38 s) : fusion avec le suivant si la place le permet.
        fusion: list[list[Mot]] = []
        for g in propres:
            if fusion:
                prec = fusion[-1]
                court = g[0].debut - prec[0].debut < 0.45
                texte = " ".join(w.texte for w in prec + g)
                if court and len(prec + g) <= 3 and len(texte) <= 15:
                    fusion[-1] = prec + g
                    continue
            fusion.append(g)
        for g in fusion:
            if all(_petit(w.texte) for w in g):
                continue                                # « WE WILL », « IN THE » seuls : rien à montrer
            texte = " ".join(w.texte for w in g)
            plein = max((w.texte for w in g if not _petit(w.texte)), key=len, default=texte)
            groupes.append(Groupe(texte, g[0].debut, g[-1].fin, num,
                                  importance=float(np.clip((len(plein) - 3) / 5, 0, 1))))
        if groupes and groupes[-1].ligne == num:
            groupes[-1].importance = min(1.0, groupes[-1].importance + 0.35)   # fin de phrase : accent
    # Respirations : dans une ligne chargée (3 groupes ou plus), on ne garde que les mots forts
    # (importance) et la fin de phrase. Les refs montrent des mots-clés, pas chaque syllabe.
    par_ligne: dict[int, list[Groupe]] = {}
    for g in groupes:
        par_ligne.setdefault(g.ligne, []).append(g)
    gardes = []
    for num, gs in par_ligne.items():
        if len(gs) >= 3:
            forts = [g for k, g in enumerate(gs) if g.importance >= 0.3 or k == len(gs) - 1]
            gs = forts or gs[-1:]
        gardes += gs
    groupes = sorted(gardes, key=lambda g: g.debut)
    # Durées : jusqu'au groupe suivant de la même ligne, au plus 0,25 s après le dernier mot
    # (respiration entre les lignes), et au moins 0,4 s si la place existe.
    for i, g in enumerate(groupes):
        suivant = groupes[i + 1].debut if i + 1 < len(groupes) else g.fin + 1.0
        meme = i + 1 < len(groupes) and groupes[i + 1].ligne == g.ligne
        fin = suivant if meme else min(suivant - 0.2, g.fin + 0.25)
        g.fin = float(max(min(suivant, g.debut + 0.4), min(fin, g.debut + 1.4)))
    return [g for g in groupes if g.fin - g.debut > 0.12]


# Emplacements (centre x, centre y, largeur max, hauteur max), en fraction de la zone de jeu.
# Le réticule et l'ennemi sont au centre : aucun emplacement ne le recouvre.
PLACES_PAYSAGE = {
    "gauche": (0.21, 0.50, 0.32, 0.22),     # sous la minimap, à gauche du réticule
    "haut": (0.50, 0.17, 0.62, 0.24),
    "droite": (0.75, 0.25, 0.44, 0.24),
    "bas": (0.26, 0.74, 0.44, 0.24),        # sous le réticule, côté opposé à l'arme
    "arme": (0.72, 0.60, 0.46, 0.30),       # derrière l'arme : seulement si le masque est fiable
}
PLACES_PORTRAIT = {
    "haut": (0.50, 0.22, 0.88, 0.13),
    "bas": (0.31, 0.69, 0.56, 0.10),
    "haut2": (0.50, 0.33, 0.84, 0.10),
    "arme": (0.62, 0.71, 0.72, 0.12),
}
RETICULE = (0.38, 0.36, 0.62, 0.64)          # x0, y0, x1, y1 : zone interdite (paysage)
RETICULE_PORTRAIT = (0.30, 0.42, 0.70, 0.58)
# Derrière l'arme découpée par IA (Pro) : mot plus gros, centré-bas, juste derrière l'arme (refs).
# Le centre exact est cherché autour de ce point (voir _centre_derriere_arme).
ARME_IA_PAYSAGE = (0.50, 0.78, 0.46, 0.20)
ARME_IA_PORTRAIT = (0.48, 0.68, 0.80, 0.14)


def placer(groupes: list[Groupe], paysage: bool, ia: bool = False) -> None:
    """Alterne les emplacements (jamais deux fois de suite le même) ; « arme » seulement si fiable.
    `ia` : arme découpée par IA, chaque groupe qui a trouvé sa place derrière l'arme y va (refs)."""
    cycle = ["gauche", "droite", "haut", "bas"] if paysage else ["haut", "bas", "haut2"]
    k = 0
    precedent = ""
    for g in groupes:
        if (g.centre_ia is not None) if ia else (g.fiable and precedent != "arme" and g.importance >= 0.5):
            g.place = "arme"
        else:
            g.place = cycle[k % len(cycle)]
            if g.place == precedent:
                k += 1
                g.place = cycle[k % len(cycle)]
            k += 1
        precedent = g.place


# ─────────────────────────────── 2c. Dessin des groupes ───────────────────────────────

STYLES = ("verre", "accent", "neon")


@dataclass
class Sprite:
    a: np.ndarray        # (h, w) lettres 0-1
    glow: np.ndarray     # (h, w) halo 0-1
    ombre: np.ndarray    # (h, w) ombre portée 0-1
    flous: list          # lettres floues (profondeur) pour les premières images
    cx: float
    cy: float


def _flou_bas(x: np.ndarray, sigma: float, q: int = 4) -> np.ndarray:
    """Flou gaussien calculé en basse définition puis agrandi (rapide pour les grands halos)."""
    h, w = x.shape
    if sigma < 2 * q:
        return ndimage.gaussian_filter(x, sigma)
    hq, wq = -(-h // q), -(-w // q)
    p = np.pad(x, ((0, hq * q - h), (0, wq * q - w))).reshape(hq, q, wq, q).mean((1, 3))
    p = ndimage.gaussian_filter(p, sigma / q)
    return ndimage.zoom(p, q, order=1)[:h, :w]


def _preparer(alpha: np.ndarray) -> Sprite:
    hb = alpha.shape[0]
    marge = int(hb * 0.45) + 6
    a = np.pad(alpha.astype(np.float32) / 255, marge)
    lettres = _boite((a * 255).astype(np.uint8), 40)
    glow = _flou_bas(a, hb * 0.18)
    glow = np.clip(glow / (glow.max() + 1e-6) * 1.3, 0, 1)
    dec = max(2, int(hb * 0.05))
    ombre = np.clip(_flou_bas(ndimage.shift(a, (dec, dec * 0.5), order=0), hb * 0.06) * 1.2, 0, 1)
    flous = [np.clip(ndimage.gaussian_filter(a, hb * s) * 1.1, 0, 1) for s in (0.09, 0.045, 0.018)]
    cy, cx = ((lettres[0] + lettres[1]) / 2, (lettres[2] + lettres[3]) / 2) if lettres else (a.shape[0] / 2, a.shape[1] / 2)
    return Sprite(a, glow.astype(np.float32), ombre.astype(np.float32), [f.astype(np.float32) for f in flous], cx, cy)


def preparer_sprites(groupes: list[Groupe], zone_l: int, zone_h: int, dossier: Path,
                     places: dict | None = None) -> dict:
    """Un sprite par (texte, emplacement) : taille selon l'emplacement et l'importance."""
    places = places or (PLACES_PAYSAGE if zone_l > zone_h else PLACES_PORTRAIT)
    cles = sorted({(g.texte, g.place, round(g.importance, 1)) for g in groupes})
    if not cles:
        return {}
    textes = sorted({c[0] for c in cles})
    base = 100
    mesures = _dessiner(textes, [base] * len(textes), base * 12, int(base * 1.6), dossier)
    dims = {}
    for i, t in enumerate(textes):
        bt = _boite(mesures[i])
        dims[t] = (bt[3] - bt[2], max(bt[1] - bt[0], base * 0.7)) if bt else (base, base)
    tailles = []
    for t, p, imp in cles:
        w, h = dims[t]
        _, _, mw, mh = places[p]
        s = min(mw * zone_l / w, mh * zone_h / h) * (0.78 + 0.22 * imp)
        tailles.append(max(14, int(base * s)))
    sprites = {}
    ordre = np.argsort(tailles)
    lot: list[int] = []

    def vider(lot):
        tmax = max(tailles[i] for i in lot)
        L = int(tmax * 0.5 * max(len(cles[i][0]) for i in lot) + tmax * 0.6) + 16
        imgs = _dessiner([cles[i][0] for i in lot], [tailles[i] for i in lot], L, int(tmax * 1.5), dossier)
        for j, i in enumerate(lot):
            bt = _boite(imgs[j], 4)
            if bt:
                sprites[cles[i]] = _preparer(imgs[j][bt[0]:bt[1], bt[2]:bt[3]])

    for i in ordre:
        lot.append(int(i))
        if len(lot) * tailles[i] * tailles[i] * 8 > 60e6:
            vider(lot)
            lot = []
    if lot:
        vider(lot)
    return sprites


def _teintes(style: str, couleur: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """(couleur des lettres, couleur du halo, opacité des lettres, force du halo, fusion avec la scène)."""
    blanc = np.ones(3, np.float32)
    if style == "accent":
        return couleur, 0.6 * couleur + 0.4 * blanc, 0.86, 0.55, 0.45
    if style == "neon":
        return 0.86 * blanc + 0.14 * couleur, couleur, 0.92, 0.7, 0.3
    # verre : blanc légèrement teinté, translucide, la texture de la scène passe à travers
    return 0.9 * blanc + 0.1 * couleur, 0.5 * couleur + 0.5 * blanc, 0.88, 0.5, 0.5


# ─────────────────────────── 3. Arme au premier plan + caméra ───────────────────────────

def zone_jeu(W: int, H: int, cadrage: str = "crop") -> tuple[int, int, int, int]:
    """(x, y, l, h) de l'image du jeu dans la vidéo (en 9:16 « flou », le jeu est une bande)."""
    from outils import pair
    if W < H and cadrage == "flou":
        fgw = pair(W * 1.3)
        fh = min(H, pair(fgw * 9 / 16))
        return 0, (H - fh) // 2, W, fh
    if W < H and cadrage == "ecran_partage":
        return 0, 0, W, pair(W * 9 / 16)
    return 0, 0, W, H


def _correlation_phase(a: np.ndarray, b: np.ndarray, fen: np.ndarray) -> tuple[float, float, float]:
    """Décalage (dx, dy) tel que b(x) ≈ a(x - d), et la netteté du pic (confiance)."""
    A = np.fft.rfft2((a - a.mean()) * fen)
    B = np.fft.rfft2((b - b.mean()) * fen)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-6
    c = np.fft.irfft2(R, s=a.shape)
    h, w = c.shape
    i = int(np.argmax(c))
    py, px = divmod(i, w)
    if (py in (0, 1, h - 1)) and (px in (0, 1, w - 1)):
        # Pic à zéro : souvent le HUD (fixe à l'écran). Si un autre pic presque aussi net existe,
        # c'est le vrai mouvement du décor.
        c2 = c.copy()
        c2[np.ix_([0, 1, h - 1, 2, h - 2], [0, 1, w - 1, 2, w - 2])] = -1
        j = int(np.argmax(c2))
        if c2.flat[j] > 0.45 * c[py, px]:
            py, px = divmod(j, w)
    def fin(v_m, v_0, v_p):
        d = v_m - 2 * v_0 + v_p
        return 0.5 * (v_m - v_p) / d if abs(d) > 1e-9 else 0.0

    sy = fin(c[(py - 1) % h, px], c[py, px], c[(py + 1) % h, px])
    sx = fin(c[py, (px - 1) % w], c[py, px], c[py, (px + 1) % w])
    dy = py + sy if py <= h // 2 else py + sy - h
    dx = px + sx if px <= w // 2 else px + sx - w
    return float(dx), float(dy), float(c[py, px])


def _decaler(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.empty_like(img)
    out[:] = np.nan
    h, w = img.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


@dataclass
class Analyse:
    masques: np.ndarray        # (K, lh, lw) float32 : premier plan (arme) par image-clé
    cle: np.ndarray            # (T,) index du masque de chaque image
    camera: np.ndarray         # (T, 2) mouvement global (dx, dy) en pixels basse déf. depuis l'image précédente
    confiance: np.ndarray      # (T,) confiance du mouvement
    coupes: list[int]          # images où commence un plan
    couleur: np.ndarray
    lumiere: np.ndarray        # (T,) luminosité moyenne
    fiabilite: np.ndarray | None = None   # (K,) confiance 0-1 dans le masque de chaque image-clé


def _nettoyer_masque(brut: np.ndarray, lh: int, lw: int) -> np.ndarray:
    m = ndimage.binary_opening(brut, iterations=1)
    m = ndimage.binary_closing(m, iterations=3)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n:
        tailles = ndimage.sum(m, lab, range(1, n + 1))
        garder = np.zeros(n + 1, bool)
        # Le viewmodel part du bas ou du bord droit (à quelques pixels près : bords sans info).
        bas = set(np.unique(lab[-max(2, lh // 12):, :])) | set(np.unique(lab[int(lh * 0.45):, -max(2, lw // 12):]))
        centres = ndimage.center_of_mass(m, lab, range(1, n + 1))
        for k in range(1, n + 1):
            # Arme + mains : à droite (centre au-delà de 45 % de la largeur), assez grande.
            if tailles[k - 1] >= 0.004 * lh * lw and k in bas and centres[k - 1][1] > 0.45 * lw:
                garder[k] = True
        m = garder[lab]
    if m.mean() > 0.30:          # bien trop grand pour un viewmodel : plan fixe ou image de synthèse
        m[:] = False
    m = ndimage.binary_dilation(m, iterations=1)
    return ndimage.gaussian_filter(m.astype(np.float32), 0.9)


def analyser(images: np.ndarray, coupes_donnees: list[int] | None = None, pas: int = 2,
             demi_fenetre: int = 7) -> Analyse:
    """images : (T, lh, lw, 3) uint8 de la zone de jeu en basse définition."""
    T, lh, lw, _ = images.shape
    Y = images.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    Y = ndimage.gaussian_filter(Y, (0, 0.8, 0.8))
    lumiere = Y.reshape(T, -1).mean(1)
    # Mouvement global : sur le décor (sous le HUD du haut, au-dessus de l'arme).
    r0, hh = int(lh * 0.12), int(lh * 0.62)
    fen = np.outer(np.hanning(hh - r0), np.hanning(lw)).astype(np.float32)
    camera = np.zeros((T, 2), np.float32)
    conf = np.zeros(T, np.float32)
    diff = np.zeros(T, np.float32)
    for t in range(1, T):
        dx, dy, c = _correlation_phase(Y[t - 1, r0:hh], Y[t, r0:hh], fen)
        camera[t] = (dx, dy)
        conf[t] = c
        diff[t] = float(np.abs(Y[t] - Y[t - 1]).mean())
    # Coupes : grosse différence + mouvement incohérent, ou coupes connues du montage.
    coupes = {0}
    seuil = max(18.0, float(np.median(diff[1:])) * 4) if T > 2 else 18.0
    for t in range(1, T):
        if diff[t] > seuil and conf[t] < 0.25:
            coupes.add(t)
    for c in coupes_donnees or []:
        if 0 < c < T:
            coupes.add(int(c))
    coupes = sorted(coupes)
    # Indices « fixe à l'écran » par paire d'images, accumulés (num / den) :
    #  A) caméra qui tourne : erreur « collé au monde » vs erreur « fixe à l'écran » ;
    #  B) décor qui change (on avance, on saute…) : pixel texturé qui change bien moins que le décor.
    num = np.zeros((T, lh, lw), np.float32)
    den = np.zeros((T, lh, lw), np.float32)
    valide = np.zeros(T, np.float32)
    debut_plan = np.zeros(T, int)
    for c in coupes:
        debut_plan[c:] = c
    for t in range(1, T):
        if debut_plan[t] == t:
            continue
        es = np.abs(Y[t] - Y[t - 1])
        dx, dy = camera[t]
        if conf[t] >= 0.08 and np.hypot(dx, dy) >= 0.9 and abs(dx) < lw * 0.3 and abs(dy) < lh * 0.3:
            em = _decaler(Y[t - 1], int(round(dx)), int(round(dy)))
            em = np.abs(Y[t] - em)
            trou = np.isnan(em)
            em[trou] = es[trou]                      # bord sans info : neutre
            num[t] += em - es
            den[t] += em + es
            valide[t] = 1
        niveau = float(np.median(es[r0:hh]))
        if niveau > 4.0:
            gx = np.abs(np.diff(Y[t], axis=1, append=Y[t][:, -1:]))
            gy = np.abs(np.diff(Y[t], axis=0, append=Y[t][-1:]))
            porte = np.clip((gx + gy) / 14.0, 0, 1) * 0.7
            num[t] += (niveau - es) * porte
            den[t] += (niveau + es) * porte
            valide[t] = 1
    cum_m = np.cumsum(num, axis=0)
    cum_e = np.cumsum(den, axis=0)
    del num, den
    cum_v = np.cumsum(valide)
    lignes = np.arange(lh)[:, None]
    bas = (lignes > lh * 0.36)
    masques, cle = [], np.zeros(T, int)
    zero = np.zeros((lh, lw), np.float32)
    global_somme, global_n = zero.copy(), 0
    infos = []
    bornes = coupes + [T]
    # A priori de position : le viewmodel est à droite (le bas-gauche est souvent le sol).
    colonnes = np.arange(lw)[None, :] / lw
    seuil_pos = 0.3 / np.clip((colonnes - 0.15) / 0.4, 0.45, 1.0)

    def somme(u, v, cum):
        return cum[v] - (cum[u - 1] if u > 0 else 0)

    for p in range(len(coupes)):
        a, b = bornes[p], bornes[p + 1]
        n_plan = somme(a, b - 1, cum_v)
        long_ = somme(a, b - 1, cum_m) / (somme(a, b - 1, cum_e) + 1e-3) if n_plan >= 3 else None
        tex_long = somme(a, b - 1, cum_e) / (2 * max(1, n_plan))
        for k in range(a, b, pas):
            u, v = max(a, k - demi_fenetre), min(b - 1, k + demi_fenetre)
            n = somme(u, v, cum_v)
            if n < 3 and long_ is None:
                infos.append((a, b, k, None, 0))
                continue
            if n >= 3:
                se = somme(u, v, cum_e)
                score = somme(u, v, cum_m) / (se + 1e-3)
                texture = se / (2 * n)
                if long_ is not None:   # mélange court terme / tout le plan (l'arme bouge peu)
                    score = 0.6 * score + 0.4 * long_
                    texture = np.maximum(texture, tex_long)
            else:
                score, texture = long_, tex_long
            brut = (score > seuil_pos) & (texture > 2.0) & bas
            m = _nettoyer_masque(brut, lh, lw)
            infos.append((a, b, k, m, n))
            if m.max() > 0.5:
                global_somme += m
                global_n += 1
    repli = (global_somme / global_n > 0.5).astype(np.float32) if global_n else zero
    repli = ndimage.gaussian_filter(repli, 0.9)
    # Images-clés sans info (caméra immobile) : masque voisin du même plan, sinon masque moyen.
    for idx, (a, b, k, m, _n) in enumerate(infos):
        if m is None:
            voisins = [x for x in infos if x[0] == a and x[3] is not None]
            if voisins:
                m = min(voisins, key=lambda x: abs(x[2] - k))[3]
            else:
                m = repli
        masques.append(m)
        cle[k:min(b, k + pas)] = idx
    # Cohérence dans le temps : une zone qui n'apparaît que sur une image-clé isolée est du bruit
    # (l'arme, elle, reste là). Puis lissage léger entre images-clés voisines du même plan.
    M = np.stack(masques) if masques else np.zeros((1, lh, lw), np.float32)
    stable = M.copy()
    for i in range(len(infos)):
        av = M[i - 1] if i > 0 and infos[i - 1][0] == infos[i][0] else None
        ap = M[i + 1] if i + 1 < len(infos) and infos[i + 1][0] == infos[i][0] else None
        voisins = [x for x in (av, ap) if x is not None]
        if voisins:
            stable[i] = np.minimum(M[i], np.maximum.reduce(voisins))
    M = stable
    lisse = M.copy()
    for i in range(1, len(infos) - 1):
        if infos[i - 1][0] == infos[i][0] == infos[i + 1][0]:
            lisse[i] = 0.25 * M[i - 1] + 0.5 * M[i] + 0.25 * M[i + 1]
    # Fiabilité de chaque masque : assez d'indices (la caméra a bougé), une taille d'arme plausible
    # et une forme stable d'une image-clé à l'autre. Un masque faux découpe les lettres au hasard :
    # c'est pire que pas de masque, donc en dessous du seuil on ne s'en sert pas.
    fiab = np.zeros(len(infos), np.float32)
    binaires = lisse > 0.5
    for i, (a, b, k, m, n) in enumerate(infos):
        if m is None or n < 6:
            continue
        aire = float(binaires[i].mean())
        if not 0.015 < aire < 0.22:
            continue
        ys, xs = np.nonzero(binaires[i])
        # Un viewmodel est en bas à droite : centre de masse dans ce coin, et il touche le bord.
        if xs.mean() < 0.58 * lw or ys.mean() < 0.55 * lh:
            continue
        if not (binaires[i][-max(2, lh // 15):, :].any() or binaires[i][:, -max(2, lw // 15):].any()):
            continue
        ious = []
        for j in (i - 2, i - 1, i + 1, i + 2):
            if 0 <= j < len(infos) and infos[j][0] == a:
                inter = np.logical_and(binaires[i], binaires[j]).sum()
                union = np.logical_or(binaires[i], binaires[j]).sum()
                ious.append(inter / max(1, union))
        if not ious:
            continue
        fiab[i] = float(np.clip((np.mean(ious) - 0.55) / 0.3, 0, 1) * min(1.0, n / 10))
    return Analyse(lisse, cle, camera, conf, coupes, couleur_dominante(images), lumiere, fiab)


def _images_groupe(g: Groupe, fps: float, T: int) -> tuple[int, int]:
    """Images [a, b) où le groupe est affiché (au moins 3 images, comme dans incruster)."""
    a = max(0, int(np.ceil(g.debut * fps - 1e-6)))
    b = int(np.ceil(g.fin * fps - 1e-6))
    return a, min(T, max(b, a + 3))


def _arme_ia(video: Path, groupes: list[Groupe], an: Analyse, T: int, fps: float, zone, paysage: bool,
             places: dict, log=print):
    """Découpe l'arme par IA pendant l'affichage des mots (arme_ia.py), puis cherche pour chaque groupe
    une place juste derrière l'arme. `g.fiable` = masque sûr sur TOUTES les images du groupe ;
    `g.centre_ia` = centre trouvé (sinon le groupe garde le placement habituel). None : IA indisponible."""
    try:
        import arme_ia
    except Exception as e:
        log(f"  paroles : découpage de l'arme par IA indisponible ({type(e).__name__})")
        return None
    plan_image = np.zeros(T, int)
    for k, c in enumerate(an.coupes):
        plan_image[c:] = k
    cles: set[int] = set()
    for g in groupes:
        a, b = _images_groupe(g, fps, T)
        cles.update(range(a, b, arme_ia.PAS))
        cles.add(b - 1)
        for c in an.coupes:                    # un plan qui commence pendant le groupe : ses 2 bords
            if a < c < b:
                cles.update((c - 1, c))
    try:
        ia = arme_ia.segmenter(video, zone, paysage, T, sorted(cles), plan_image, log)
    except Exception as e:
        log(f"  paroles : découpage de l'arme par IA impossible ({type(e).__name__}: {str(e)[:160]})")
        return None
    if ia is None:
        return None
    bilan = []
    for g in groupes:
        a, b = _images_groupe(g, fps, T)
        couverts = [ia.couvert(t) for t in range(a, b)]
        g.fiable = all(couverts)
        g.centre_ia = None
        bilan.append([g.texte, round(float(np.mean(couverts)), 2), False])
    ia.stats["groupes"] = bilan
    candidats = [g for g in groupes if g.fiable]
    if not candidats:
        return ia
    for g in candidats:
        g.place = "arme"
    zx, zy, zl, zh = zone
    dossier = Path(tempfile.mkdtemp(prefix="autoclip_paroles_"))
    try:
        sprites = preparer_sprites(candidats, zl, zh, dossier, places)
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    for g in candidats:
        sp = sprites.get((g.texte, g.place, round(g.importance, 1)))
        if sp is not None:
            g.centre_ia = _centre_derriere_arme(sp, g, ia, fps, T, zone, paysage)
            bilan[groupes.index(g)][2] = g.centre_ia is not None
        g.place = ""
    return ia


def _centre_derriere_arme(sp: Sprite, g: Groupe, ia, fps: float, T: int, zone, paysage: bool):
    """Centre (x, y) du mot pour que l'arme passe DEVANT une partie des lettres (environ 15 %) sans
    le rendre illisible ; jamais sur le réticule. None si aucune place ne convient."""
    zx, zy, zl, zh = zone
    a, b = _images_groupe(g, fps, T)
    ts = [int(t) for t in ia.cles if a <= t < b] or [a]
    gl = 192
    gh = max(8, int(round(gl * zh / zl)))
    Ms = [m for m in (ia.masque_bas(t, gl, gh) for t in ts) if m is not None]
    if not Ms:
        return None
    M = np.stack(Ms).astype(np.float32)
    f = gl / zl
    bt = _boite((sp.a * 255).astype(np.uint8), 40)
    if not bt:
        return None
    lettres = sp.a[bt[0]:bt[1], bt[2]:bt[3]]
    hl, wl = lettres.shape
    ag = ndimage.zoom(lettres, (max(1, round(hl * f)) / hl, max(1, round(wl * f)) / wl), order=1)
    ag = np.clip(ag, 0, 1)
    hg, wg = ag.shape
    total = float(ag.sum()) + 1e-6
    # Une tranche par lettre (environ) : aucune lettre ne doit disparaître entièrement derrière l'arme
    # (« CAL » au lieu de « CALL » se lit comme un mot coupé, pas comme un mot posé dans la scène).
    n_tr = max(3, len(g.texte.replace(" ", "")))
    bornes = np.linspace(0, wg, n_tr + 1).astype(int)
    tranches = [(u, v) for u, v in zip(bornes[:-1], bornes[1:]) if v > u and ag[:, u:v].sum() > 0.2 * total / n_tr] or [(0, wg)]
    Mp = np.pad(M, ((0, 0), (hg, hg), (wg, wg)))
    # 16:9 : l'arme est grande et la zone interdite habituelle (réticule + ennemi) très large ; ici
    # on protège seulement le centre, sinon aucun mot ne peut passer derrière l'arme.
    rx0, ry0, rx1, ry1 = (0.42, 0.38, 0.58, 0.62) if paysage else RETICULE_PORTRAIT
    limite_lettre = 0.6 if paysage else 0.5
    px, py = (ARME_IA_PAYSAGE if paysage else ARME_IA_PORTRAIT)[:2]
    xs = np.linspace(0.20, 0.80, 25) if paysage else np.linspace(0.30, 0.70, 17)
    ys = np.linspace(0.50, 0.90, 17) if paysage else np.linspace(0.56, 0.86, 13)
    meilleur = None
    # 1er essai : arme devant ~17 % des lettres, aucune lettre cachée à plus de 50-60 %. Sinon, 2e essai
    # plus permissif (mot long, arme très grande) : toujours lisible, mais plus caché.
    for moy_min, pire_max, lettre_max in ((0.06, 0.38, limite_lettre), (0.04, 0.50, limite_lettre + 0.15)):
        if meilleur is not None:
            break
        for cyn in ys:
            for cxn in xs:
                x0n, x1n = cxn - wl / 2 / zl, cxn + wl / 2 / zl
                y0n, y1n = cyn - hl / 2 / zh, cyn + hl / 2 / zh
                if x0n < 0.02 or x1n > 0.98 or y0n < 0.02 or y1n > 0.97:
                    continue
                if x0n < rx1 and x1n > rx0 and y0n < ry1 and y1n > ry0:
                    continue
                gx, gy = int(round(x0n * gl)) + wg, int(round(y0n * gh)) + hg
                fen = Mp[:, gy:gy + hg, gx:gx + wg] * ag
                occ = fen.sum(axis=(1, 2)) / total
                moy, pire = float(occ.mean()), float(occ.max())
                if moy < moy_min or pire > pire_max or float(occ.min()) < 0.02:
                    continue
                fen_m = fen.mean(axis=0)
                lettre = max(float(fen_m[:, u:v].sum() / (ag[:, u:v].sum() + 1e-6)) for u, v in tranches)
                if lettre > lettre_max:
                    continue
                note = -3.0 * abs(moy - 0.17) - 1.5 * max(0.0, lettre - 0.3) - np.hypot(cxn - px, (cyn - py) * 1.3)
                if meilleur is None or note > meilleur[0]:
                    meilleur = (note, cxn, cyn)
    if meilleur is None:
        return None
    _, cxn, cyn = meilleur
    # Centre du sprite = centre des lettres (comme le placement habituel).
    return (zx + cxn * zl, zy + cyn * zh)


# ─────────────────────────────── 4. Composition ───────────────────────────────

def _matrice_interp(n_out: int, debut: float, pas: float, n_in: int) -> np.ndarray:
    """Matrice (n_out, n_in) d'interpolation linéaire : sortie i ← entrée (debut + i*pas)."""
    u = np.clip(debut + pas * np.arange(n_out), 0, n_in - 1)
    i0 = np.floor(u).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    f = (u - i0).astype(np.float32)
    M = np.zeros((n_out, n_in), np.float32)
    M[np.arange(n_out), i0] += 1 - f
    M[np.arange(n_out), i1] += f
    return M


def _redim_proche(x: np.ndarray, s: float) -> np.ndarray:
    h, w = x.shape[:2]
    H, W = max(1, int(h * s)), max(1, int(w * s))
    iy = np.minimum((np.arange(H) / s).astype(int), h - 1)
    ix = np.minimum((np.arange(W) / s).astype(int), w - 1)
    return x[iy][:, ix]


def _ffprobe_video(chemin: Path) -> tuple[int, int, float, int]:
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
                        "-of", "json", str(chemin)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=_FLAGS)
    s = json.loads(p.stdout)["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(a) / float(b), int(s.get("nb_read_packets") or 0)


def incruster(video: Path, sortie: Path, mots: list[Mot], cadrage: str = "crop", couleur: str = "",
              coupes: list[float] | None = None, exclus: list[tuple[float, float]] | None = None,
              fondu_fin: float = 0.6, qualite: str = "finale", log=print, debug_masque: bool = False,
              style: str = "neon", arme_ia: bool = False) -> dict:
    """Compose les paroles dans la vidéo : groupes de mots placés autour du réticule, derrière
    l'arme quand le masque est fiable. Retourne des stats (temps, groupes affichés).
    `arme_ia` (Pro) : l'arme et les mains sont découpées par IA (arme_ia.py) au lieu de l'heuristique."""
    t0 = time.time()
    W, H, fps, T = _ffprobe_video(video)
    zx, zy, zl, zh = zone_jeu(W, H, cadrage)
    paysage = zl > zh
    lw = int(round(np.sqrt(30000 * zl / zh))) // 2 * 2
    lh = int(round(lw * zh / zl)) // 2 * 2
    p = subprocess.run(["ffmpeg", "-v", "error", "-skip_loop_filter", "all", "-i", str(video), "-vf",
                        f"crop={zl}:{zh}:{zx}:{zy},scale={lw}:{lh}:flags=bilinear", "-f", "rawvideo",
                        "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=_FLAGS)
    images = np.frombuffer(p.stdout, np.uint8)
    T = len(images) // (lw * lh * 3)
    images = images[:T * lw * lh * 3].reshape(T, lh, lw, 3)
    if T < 2:
        raise RuntimeError("vidéo illisible")
    an = analyser(images, [int(round(c * fps)) for c in (coupes or [])])
    t_analyse = time.time() - t0
    col = _couleur_hex(couleur)
    col = col if col is not None else an.couleur
    duree = T / fps
    exclus = exclus or []
    groupes = []
    for g in grouper(mots):
        if g.debut >= duree - 0.6 or g.fin <= 0 or any(a <= g.debut < b for a, b in exclus):
            continue
        g.fin = min([g.fin] + [a for a, b in exclus if g.debut < a < g.fin])   # coupé avant l'intro 3D
        a, b = int(np.ceil(g.debut * fps)), min(T, int(np.ceil(g.fin * fps)))
        if b > a and an.fiabilite is not None:
            g.fiable = bool(np.percentile(an.fiabilite[an.cle[a:b]], 10) >= 0.5)
        groupes.append(g)
    places = PLACES_PAYSAGE if paysage else PLACES_PORTRAIT
    ia = None
    if arme_ia and groupes:
        places = dict(places, arme=ARME_IA_PAYSAGE if paysage else ARME_IA_PORTRAIT)
        ia = _arme_ia(video, groupes, an, T, fps, (zx, zy, zl, zh), paysage, places, log)
    t_ia = time.time() - t0 - t_analyse
    placer(groupes, paysage, ia=ia is not None)
    dossier = Path(tempfile.mkdtemp(prefix="autoclip_paroles_"))
    try:
        sprites = preparer_sprites(groupes, zl, zh, dossier, places)
    finally:
        shutil.rmtree(dossier, ignore_errors=True)
    t_sprites = time.time() - t0 - t_analyse - t_ia
    rx0, ry0, rx1, ry1 = RETICULE if paysage else RETICULE_PORTRAIT
    rx0, rx1 = zx + rx0 * zl, zx + rx1 * zl
    ry0, ry1 = zy + ry0 * zh, zy + ry1 * zh
    # Centre de chaque groupe, écarté du réticule si besoin.
    centres = {}
    for i, g in enumerate(groupes):
        sp = sprites.get((g.texte, g.place, round(g.importance, 1)))
        if sp is None:
            continue
        if g.place == "arme" and g.centre_ia is not None:
            centres[i] = g.centre_ia
            continue
        bt = _boite((sp.a * 255).astype(np.uint8), 40)
        wl, hl = (bt[3] - bt[2], bt[1] - bt[0]) if bt else sp.a.shape[::-1]
        cxn, cyn, _, _ = places[g.place]
        cx, cy = zx + cxn * zl, zy + cyn * zh
        if cx - wl / 2 < rx1 and cx + wl / 2 > rx0 and cy - hl / 2 < ry1 and cy + hl / 2 > ry0:
            if g.place in ("haut", "haut2"):
                cy = ry0 - hl / 2 - 0.01 * zh
            elif cx < (rx0 + rx1) / 2:
                cx = rx0 - wl / 2 - 0.01 * zl
            else:
                cx = rx1 + wl / 2 + 0.01 * zl
        cx = float(np.clip(cx, zx + wl / 2 + 0.02 * zl, zx + zl - wl / 2 - 0.02 * zl))
        cy = float(np.clip(cy, zy + hl / 2 + 0.02 * zh, zy + zh - hl / 2 - 0.02 * zh))
        centres[i] = (cx, cy)
    actif = np.full(T, -1, int)
    for i, g in enumerate(groupes):
        if i in centres:
            a, b = int(np.ceil(g.debut * fps - 1e-6)), int(np.ceil(g.fin * fps - 1e-6))
            actif[max(0, a):min(T, max(b, a + 3))] = i
    teintes = _teintes(style, col)
    echx, echy = zl / lw, zh / lh

    from outils import args_encodage   # libx264 ou carte graphique (Rendu GPU)
    tmp = Path(sortie).with_name(Path(sortie).stem + "_paroles_tmp.mp4")
    lecteur = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, creationflags=_FLAGS, bufsize=W * H * 3)
    ecrivain = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "pipe:0", "-i", str(video),
                                 "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
                                 *args_encodage("rapide_apercu" if qualite == "apercu" else "rapide"),
                                 "-movflags", "+faststart", str(tmp)],
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_FLAGS)
    taille = W * H * 3
    tampon = bytearray(taille)
    vue = memoryview(tampon)
    image = np.frombuffer(tampon, np.uint8).reshape(H, W, 3)
    suivi = np.zeros(2, np.float32)
    dernier = -1
    affiches = set()
    caches: dict[int, list[float]] = {}      # part des lettres cachée par l'arme, image par image
    try:
        for t in range(T):
            lu = 0
            while lu < taille:
                n = lecteur.stdout.readinto(vue[lu:])
                if not n:
                    break
                lu += n
            if lu < taille:
                break
            i = actif[t]
            if i >= 0:
                g = groupes[i]
                sp = sprites[(g.texte, g.place, round(g.importance, 1))]
                debut_i = int(np.ceil(g.debut * fps - 1e-6))
                age = t - debut_i
                if i != dernier or t in an.coupes:
                    suivi[:] = 0
                    dernier = i
                elif an.confiance[t] > 0.12:
                    d = an.camera[t]
                    if np.hypot(*d) < lw * 0.08:     # suivi doux du décor (pas pendant un flick)
                        suivi += 0.5 * d
                suivi = np.clip(suivi, [-lw * 0.05, -lh * 0.05], [lw * 0.05, lh * 0.05])
                # Petite dérive lente (le mot « flotte ») + léger glissement à l'apparition.
                vers_centre = 1 if centres[i][0] < zx + zl / 2 else -1
                dx = suivi[0] * echx + (age / fps) * 0.012 * zl * vers_centre
                dy = suivi[1] * echy + 0.02 * zh * max(0.0, 1 - age / 4)
                opa = min(1.0, (age + 1) / 4)
                reste = int(np.ceil(g.fin * fps - 1e-6)) - t
                if reste < 4:
                    opa *= max(0.0, reste / 4)
                fin_mots = duree - 0.45                 # fondu de fin du montage
                if t / fps > fin_mots - fondu_fin:
                    opa *= max(0.0, (fin_mots - t / fps) / fondu_fin)
                if opa > 0.01:
                    cache = _composer(image, sp, (centres[i][0] + dx, centres[i][1] + dy), age, opa, teintes,
                                      an if g.fiable and ia is None else None, t, (zx, zy, zl, zh), (lw, lh),
                                      debug_masque, ia if g.fiable else None)
                    caches.setdefault(i, []).append(cache)
                    affiches.add(i)
            ecrivain.stdin.write(tampon)
    finally:
        try:
            ecrivain.stdin.close()
        except OSError:
            pass
        lecteur.stdout.close()
        lecteur.wait()
        erreur = ecrivain.stderr.read().decode("utf-8", "replace")
        ecrivain.wait()
    if ecrivain.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(erreur[-400:])
    tmp.replace(sortie)
    # Groupe « derrière l'arme » : au moins 4 % des lettres cachées pendant la moitié de son affichage.
    derriere = [i for i, v in caches.items() if np.mean(np.array(v) >= 0.04) >= 0.5]
    return {"secondes": time.time() - t0, "analyse": t_analyse, "sprites": t_sprites, "duree": duree,
            "ia": t_ia if ia is not None else None, "ia_stats": ia.stats if ia is not None else None,
            "derriere": len(derriere), "derriere_textes": [groupes[i].texte for i in derriere],
            "mots": len(affiches), "groupes": [(g.texte, g.place, g.fiable) for g in groupes],
            "couleur": "#" + "".join(f"{int(c * 255):02x}" for c in col)}


_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _composer(image, sp: Sprite, centre, age, opa, teintes, an: Analyse | None, t, zone, basse,
              debug_masque=False, ia=None) -> float:
    """Halo + ombre + lettres translucides qui prennent la texture de la scène ; derrière l'arme
    seulement si `an` est donné (masque jugé fiable pour ce groupe), ou `ia` (masque découpé par IA,
    contour net en pleine définition). Retourne la part des lettres cachée par l'arme."""
    zx, zy, zl, zh = zone
    lw, lh = basse
    H, W = image.shape[:2]
    fc, gc, opa_lettres, force_glow, fusion = teintes
    a = sp.flous[age] if age < 3 else sp.a                # flou de profondeur à l'apparition
    h, w = a.shape
    x0 = int(round(centre[0] - sp.cx))
    y0 = int(round(centre[1] - sp.cy))
    ax, ay = max(0, x0), max(0, y0)
    bx, by = min(W, x0 + w), min(H, y0 + h)
    if bx <= ax or by <= ay:
        return 0.0
    sy, sx = slice(ay - y0, by - y0), slice(ax - x0, bx - x0)
    a, glow, ombre = a[sy, sx], sp.glow[sy, sx], sp.ombre[sy, sx]
    k = None
    if ia is not None:
        k = ia.masque(t, ax, ay, bx, by)
        an = None
    if an is not None:
        M = an.masques[an.cle[min(t, len(an.cle) - 1)]]
        gy0, gy1 = (ay - zy) * lh / zh, (by - zy) * lh / zh
        gx0, gx1 = (ax - zx) * lw / zl, (bx - zx) * lw / zl
        r0, r1 = max(0, int(gy0) - 1), min(lh, int(np.ceil(gy1)) + 2)
        c0, c1 = max(0, int(gx0) - 1), min(lw, int(np.ceil(gx1)) + 2)
        if r1 > r0 and c1 > c0:
            Mc = M[r0:r1, c0:c1]
            if Mc.max() > 0.02:
                Ay = _matrice_interp(by - ay, (ay - zy + 0.5) * lh / zh - 0.5 - r0, lh / zh, r1 - r0)
                Ax = _matrice_interp(bx - ax, (ax - zx + 0.5) * lw / zl - 0.5 - c0, lw / zl, c1 - c0)
                k = np.clip(Ay @ Mc @ Ax.T, 0, 1)
    f = (1.0 - k) * opa if k is not None else opa
    z = image[ay:by, ax:bx].astype(np.float32)
    lum = z @ _LUMA                                        # texture de la scène (avant le texte)
    z *= (1.0 - 0.30 * ombre * f)[..., None]               # ombre portée légère
    gl = (glow * (force_glow * f))[..., None]
    z += (255.0 - z) * gl * gc                             # halo (mode « écran ») de la couleur
    af = (a * (opa_lettres * f))[..., None]
    mod = (1.0 - fusion) + fusion * (0.55 + 0.9 * lum / 255.0)
    lettres = np.minimum(255.0, (255.0 * fc)[None, None, :] * mod[..., None])
    z = z * (1.0 - af) + lettres * af
    if debug_masque and k is not None:
        z[..., 0] = np.maximum(z[..., 0], k * 150)
    np.clip(z + 0.5, 0, 255, out=z)
    image[ay:by, ax:bx] = z
    return float((a * k).sum() / max(1e-6, float(a.sum()))) if k is not None else 0.0


# ─────────────────────────────── Branchement ───────────────────────────────

def appliquer(video: Path, musique: Path, debut_musique: float, duree: float, cadrage: str = "crop",
              chanson: str = "", lrc: str = "", couleur: str = "", segments=None, fps: int = 30,
              qualite: str = "finale", log=print, decalage: float = 0.0, arme_ia: bool = False) -> bool:
    """Passe « paroles dans le décor » sur un montage fini. Ne lève jamais : sans paroles
    (ou en cas de problème) le montage reste tel quel et on l'explique dans le journal."""
    try:
        if not POLICE.is_file():
            log("  paroles : police introuvable, passe ignorée")
            return False
        mots = trouver_mots(Path(musique), debut_musique, duree, chanson, lrc, log, decalage=decalage)
        if not mots:
            log("  paroles introuvables pour cette chanson : montage sans paroles. "
                "(Astuce : mets un fichier .lrc du même nom à côté de la musique, "
                "ou écris « Artiste - Titre » dans le champ Chanson.)")
            return False
        coupes, exclus = [], []
        if segments:
            from rendu import debuts_plans
            debuts = debuts_plans(segments, fps)
            for s, d in zip(segments, debuts):
                coupes.append(d)
                coupes.append(d + s.images_transition / fps)
                if s.role in ("intro3d", "rewind"):
                    exclus.append((d, d + s.images_totales / fps))
        log(f"  paroles dans le décor : composition de {len(mots)} mots…")
        st = incruster(Path(video), Path(video), mots, cadrage=cadrage, couleur=couleur, coupes=coupes,
                       exclus=exclus, qualite=qualite, log=log, arme_ia=arme_ia)
        derriere = st["derriere"]
        log(f"  paroles : {st['mots']} groupes de mots affichés ({derriere} derrière l'arme), "
            f"couleur {st['couleur']}, {st['secondes']:.1f} s pour {st['duree']:.1f} s de vidéo")
        return True
    except Exception as e:
        log(f"  paroles : passe ignorée ({type(e).__name__}: {str(e)[:200]})")
        return False
