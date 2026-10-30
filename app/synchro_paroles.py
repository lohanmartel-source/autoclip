"""Synchro des paroles par Whisper (faster-whisper, sur le PC) : un temps PAR MOT, entendu dans TON fichier.

Pourquoi : LRCLIB donne des temps par LIGNE pour la version officielle du morceau. Une version
téléchargée de YouTube (intro différente, edit…) décale tout, et le découpage d'une ligne en mots au
prorata des syllabes ne tombe pas sur le chant. Ici :

1. On transcrit l'extrait utilisé dans le montage (avec quelques secondes de marge), mot à mot
   (`word_timestamps=True`), avec des réglages pensés pour du CHANT sur de la musique :
   - pas de filtre VAD : Silero est entraîné sur de la parole et coupe souvent le chant sur la musique ;
   - `condition_on_previous_text=False` : évite les boucles (une ligne répétée à l'infini) ;
   - température 0 et `beam_size=5` ; si Whisper boucle, nouvel essai sans prompt, puis 0,2 / 0,4 ;
   - langue forcée d'après le texte LRCLIB s'il existe, sinon détectée par Whisper ;
   - `initial_prompt` = les paroles LRCLIB autour de l'extrait : Whisper « sait » quoi écouter.
   Les extraits de plus de ~28 s sont coupés en morceaux qui se chevauchent (le prompt ne vaut que
   pour la première fenêtre de 30 s de Whisper).
2. Si LRCLIB donne le texte, on ALIGNE ce texte (bonne orthographe) sur les mots entendus :
   alignement global de séquences (Needleman-Wunsch) avec une ressemblance tolérante (accents,
   ponctuation, distance d'édition), en deux passes : la 1re estime le décalage entre la version
   officielle et ton fichier, la 2e pénalise les correspondances trop loin de ce décalage (refrains
   répétés). Texte affiché = LRCLIB, temps = Whisper. Les mots non alignés sont interpolés entre
   leurs voisins alignés (au prorata des syllabes).
3. Sans LRCLIB : les mots de Whisper directement, sans les hallucinations classiques (« Thank you
   for watching », mots répétés en boucle, confiance faible).
4. Le modèle se télécharge à la 1re utilisation dans DONNEES/modeles (progression dans le journal).
   Pas d'internet et pas de modèle : message clair, et paroles.py garde l'ancienne méthode.
5. Cache : la transcription (empreinte du fichier audio + extrait + modèle + prompt) est gardée dans
   DONNEES/paroles_cache/synchro : la même chanson n'est pas retranscrite à chaque montage.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from chemins import DONNEES

MODELES = DONNEES / "modeles"
CACHE = DONNEES / "paroles_cache" / "synchro"
ALGO = 1                         # à changer si le format du cache change

# Choisi en mesurant sur CPU (voir le rapport de test) : meilleur compromis précision / vitesse.
MODELE = os.environ.get("AUTOCLIP_MODELE_PAROLES", "small")
DEPOTS = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "mobilesam": "Acly/MobileSAM",     # découpage de l'arme (arme_ia.py), MIT / Apache-2.0
}
# Taille du téléchargement (Mo), pour afficher un pourcentage.
TAILLES_MO = {"base": 145, "small": 484, "medium": 1530, "large-v3-turbo": 1620, "mobilesam": 45}
FICHIERS = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]

SR = 16000
MARGE = 3.0          # secondes transcrites en plus avant / après l'extrait
MORCEAU = 28.0       # longueur max d'une fenêtre (Whisper travaille par 30 s)
RECOUVREMENT = 3.0
BOUCLE = 3.0         # taux de compression du texte au-delà duquel Whisper tourne en boucle


@dataclass
class MotEntendu:
    texte: str        # tel que Whisper l'a écrit
    debut: float      # secondes, temps de la CHANSON
    fin: float
    proba: float
    segment: int


# ─────────────────────────────── Modèle ───────────────────────────────

_modele = None
_nom_modele = ""
_echecs: dict[str, float] = {}     # modèle → heure du dernier téléchargement raté
_verrou = threading.Lock()


def _taille_dossier(d: Path) -> int:
    total = 0
    for racine, _, fichiers in os.walk(d):
        for f in fichiers:
            try:
                total += os.path.getsize(os.path.join(racine, f))
            except OSError:
                pass
    return total


def chemin_modele(nom: str = MODELE, log=print, telecharger: bool = True,
                  cache_hf: bool = True, fichiers: list[str] | None = None, revision: str | None = None,
                  quoi: str = "synchro IA") -> str | None:
    """Dossier du modèle CTranslate2. Cherche dans DONNEES/modeles, puis dans le cache HuggingFace
    habituel (déjà là sur un PC de dev), sinon le télécharge dans DONNEES/modeles avec la progression
    dans le journal. Retourne None (avec un message clair) si c'est impossible.
    Sert aussi à d'autres modèles (arme_ia.py) : `fichiers` (motifs à télécharger), `revision`
    (commit figé du dépôt) et `quoi` (nom affiché dans le journal)."""
    fichiers = fichiers or FICHIERS
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")      # pas de console dans l'exe
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log("  paroles : huggingface_hub manquant, synchro IA indisponible")
        return None
    depot = DEPOTS.get(nom, nom)
    endroits = [str(MODELES)] + ([None] if cache_hf else [])
    for cache_dir in endroits:
        try:
            return snapshot_download(depot, cache_dir=cache_dir, allow_patterns=fichiers, revision=revision,
                                     local_files_only=True)
        except Exception:
            pass
    if not telecharger or time.time() - _echecs.get(nom, 0) < 600:
        return None          # pas de nouvel essai réseau avant 10 min (montages en série)
    MODELES.mkdir(parents=True, exist_ok=True)
    total_mo = TAILLES_MO.get(nom, 0)
    log(f"  paroles : téléchargement du modèle de {quoi} « {nom} » (~{total_mo} Mo, une seule fois)…")
    dossier = MODELES / ("models--" + depot.replace("/", "--"))
    fini = threading.Event()
    depart = _taille_dossier(dossier) if dossier.exists() else 0

    def suivre():
        dernier = -1
        while not fini.wait(2.0):
            mo = (_taille_dossier(dossier) - depart) / 1e6 if dossier.exists() else 0.0
            if total_mo:
                pct = int(min(99, 100 * mo / total_mo))
                if pct // 5 != dernier:          # une ligne tous les 5 % (le journal reste lisible)
                    log(f"  paroles : modèle IA {pct} % ({mo:.0f}/{total_mo} Mo)")
                    dernier = pct // 5
            elif int(mo) // 50 != dernier:
                dernier = int(mo) // 50
                log(f"  paroles : modèle IA {mo:.0f} Mo téléchargés")

    th = threading.Thread(target=suivre, daemon=True)
    th.start()
    t0 = time.time()
    try:
        chemin = snapshot_download(depot, cache_dir=str(MODELES), allow_patterns=fichiers, revision=revision)
        fini.set()
        log(f"  paroles : modèle IA téléchargé en {time.time() - t0:.0f} s")
        return chemin
    except Exception as e:
        fini.set()
        _echecs[nom] = time.time()
        log(f"  paroles : impossible de télécharger le modèle de {quoi} "
            f"({type(e).__name__}). Vérifie ta connexion internet : la {quoi} se fera au prochain "
            "montage.")
        return None


def charger_modele(nom: str = MODELE, log=print):
    """WhisperModel gardé en mémoire (int8 sur CPU). None si faster-whisper ou le modèle manque."""
    global _modele, _nom_modele
    with _verrou:
        if _modele is not None and _nom_modele == nom:
            return _modele
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            log("  paroles : faster-whisper n'est pas installé, synchro IA indisponible")
            return None
        chemin = chemin_modele(nom, log)
        if not chemin:
            return None
        try:
            t0 = time.time()
            fils = int(os.environ.get("AUTOCLIP_FILS_WHISPER") or max(1, min(8, (os.cpu_count() or 4) - 2)))
            _modele = WhisperModel(chemin, device="cpu", compute_type="int8", cpu_threads=fils)
            _nom_modele = nom
            log(f"  paroles : modèle IA « {nom} » chargé ({time.time() - t0:.1f} s)")
            return _modele
        except Exception as e:
            log(f"  paroles : modèle IA illisible ({type(e).__name__}: {str(e)[:120]})")
            return None


# ─────────────────────────────── Texte ───────────────────────────────

def normaliser(mot: str) -> str:
    """Clé de comparaison : minuscules, sans accents ni ponctuation (« Don't » → « dont »)."""
    mot = unicodedata.normalize("NFKD", mot.lower())
    mot = "".join(c for c in mot if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", mot)


_EN = set("the you i me my and to a it is in of that your we be on for with all what this don't im i'm "
          "know love oh baby just like when so can no up down got never".split())
_FR = set("je tu il elle nous vous le la les de des du un une et est pas que qui dans pour sur ce ça "
          "mon ma mes ton ta tes on mais avec plus tout moi toi".split())


def langue_texte(texte: str) -> str | None:
    """« en » / « fr » d'après les mots les plus courants, sinon None (Whisper détectera)."""
    mots = re.findall(r"[a-zà-ÿ']+", texte.lower())
    if len(mots) < 6:
        return None
    en = sum(m in _EN for m in mots)
    fr = sum(m in _FR for m in mots)
    if en >= 3 and en > 1.5 * fr:
        return "en"
    if fr >= 3 and fr > 1.5 * en:
        return "fr"
    return None


# Phrases que Whisper invente sur de la musique (vidéos YouTube sous-titrées pendant son entraînement).
_HALLU = re.compile(r"(thank(s| you) for watching|subscribe|subtitles? by|sous-titr|amara\.org|"
                    r"merci d'avoir regard|abonnez|www\.|\.com|♪|music playing|\[music\])", re.I)


# ─────────────────────────────── Transcription ───────────────────────────────

def empreinte(fichier: Path) -> str:
    """Empreinte du CONTENU du fichier audio (pas de son nom) : sha1 du début, du milieu, de la fin + taille."""
    fichier = Path(fichier)
    h = hashlib.sha1()
    n = fichier.stat().st_size
    h.update(str(n).encode())
    with open(fichier, "rb") as f:
        for pos in (0, max(0, n // 2 - 524288), max(0, n - 1048576)):
            f.seek(pos)
            h.update(f.read(1048576))
    return h.hexdigest()[:24]


def _cache_lire(cle: str):
    try:
        return json.loads((CACHE / f"{cle}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_ecrire(cle: str, donnees) -> None:
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / f"{cle}.json").write_text(json.dumps(donnees, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def transcrire(musique: Path, debut: float, fin: float, prompt: str = "", langue: str | None = None,
               nom: str = MODELE, log=print, y: np.ndarray | None = None, cache: bool = True,
               **reglages) -> list[MotEntendu] | None:
    """Mots entendus entre `debut` et `fin` (temps de la chanson). None si Whisper est indisponible.
    `reglages` : remplace des options de décodage (tests)."""
    cle_base = f"{ALGO}|{empreinte(musique)}|{nom}|{langue}|{hashlib.sha1(prompt.encode()).hexdigest()[:10]}"
    # Une transcription déjà faite qui couvre cet extrait sert telle quelle.
    index = (_cache_lire("index") or {}) if cache else {}
    for cle, (a, b) in (index.get(cle_base) or {}).items():
        if a <= debut + 0.01 and b >= fin - 0.01:
            d = _cache_lire(cle)
            if d is not None:
                log(f"  paroles : synchro IA déjà faite pour cet extrait (cache)")
                return [MotEntendu(**m) for m in d if a <= m["debut"] <= b]
    modele = charger_modele(nom, log)
    if modele is None:
        return None
    if y is None:
        from outils import decoder_audio
        y = decoder_audio(musique, SR)
    debut, fin = max(0.0, debut), min(fin, len(y) / SR)
    if fin - debut < 1.0:
        return []
    t0 = time.time()
    mots: list[MotEntendu] = []
    # Fenêtres de MORCEAU secondes qui se chevauchent ; chaque mot est gardé par la fenêtre où il
    # est le plus loin des bords (milieu du recouvrement).
    bornes = []
    a = debut
    while True:
        b = min(fin, a + MORCEAU)
        bornes.append((a, b))
        if b >= fin:
            break
        a = b - RECOUVREMENT
    n_seg = 0
    for k, (a, b) in enumerate(bornes):
        garde_a = a + RECOUVREMENT / 2 if k > 0 else -1e9
        garde_b = b - RECOUVREMENT / 2 if k + 1 < len(bornes) else 1e9
        morceau = y[int(a * SR):int(b * SR)].astype(np.float32)
        options = dict(language=langue, task="transcribe", beam_size=5, best_of=5, temperature=0.0,
                       condition_on_previous_text=False, initial_prompt=prompt or None,
                       word_timestamps=True, vad_filter=False, compression_ratio_threshold=None,
                       no_speech_threshold=0.6, log_prob_threshold=-1.0)
        options.update(reglages)

        def decoder(opts):
            segs_, info_ = modele.transcribe(morceau, **opts)
            segs_ = list(segs_)
            boucle_ = sum(len(x.words or []) for x in segs_ if getattr(x, "compression_ratio", 0) > BOUCLE)
            return segs_, info_, boucle_

        # Décodage glouton-faisceau à température 0 (le plus fidèle aux temps). Si Whisper part en
        # boucle (le prompt de paroles peut l'y pousser), on réessaie SANS prompt, puis en dernier
        # recours avec un peu de température (bien plus lent : 5 tirages).
        segs, info, boucle = decoder(options)
        if boucle and options.get("initial_prompt"):
            essai = decoder({**options, "initial_prompt": None})
            if essai[2] < boucle:
                segs, info, boucle = essai
        if boucle and "temperature" not in reglages:
            essai = decoder({**options, "temperature": [0.2, 0.4], "compression_ratio_threshold": 2.4})
            if essai[2] < boucle:
                segs, info, boucle = essai
        for s in segs:
            n_seg += 1
            if _HALLU.search(s.text or ""):
                continue
            if getattr(s, "compression_ratio", 0) > BOUCLE:     # boucle « la la la la… »
                continue
            for w in (s.words or []):
                t = a + float(w.start)
                if garde_a <= t < garde_b and (w.word or "").strip():
                    mots.append(MotEntendu(w.word.strip(), round(t, 3), round(a + float(w.end), 3),
                                           round(float(w.probability), 3), n_seg))
        if langue is None and getattr(info, "language", None) and info.language_probability > 0.6:
            langue = info.language          # même langue pour les fenêtres suivantes
    duree = time.time() - t0
    log(f"  paroles : synchro IA, {len(mots)} mots entendus en {duree:.1f} s "
        f"({fin - debut:.0f} s de musique, modèle {nom})")
    if not cache:
        return mots
    cle = hashlib.sha1(f"{cle_base}|{debut:.2f}|{fin:.2f}".encode()).hexdigest()[:20]
    _cache_ecrire(cle, [m.__dict__ for m in mots])
    index.setdefault(cle_base, {})[cle] = (debut, fin)
    _cache_ecrire("index", index)
    return mots


# ─────────────────────────────── Alignement ───────────────────────────────

def ressemblance(a: str, b: str) -> float:
    """0 à 1 entre deux clés normalisées (distance d'édition relative, via difflib)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def aligner(ref: list[str], hyp: list[str], t_hyp: list[float] | None = None,
            fenetres: list[tuple[float, float]] | None = None) -> list[tuple[int, int]]:
    """Needleman-Wunsch : paires (i_ref, j_hyp) alignées. Trous gratuits au début et à la fin de la
    référence (on lui donne plus de lignes que l'extrait). Avec `fenetres` (pour chaque mot de la
    référence, l'intervalle de temps où il devrait être entendu), une correspondance hors de son
    intervalle est pénalisée : c'est ce qui place le bon « in the middle of the night » quand le
    refrain se répète."""
    n, m = len(ref), len(hyp)
    if not n or not m:
        return []
    TROU_REF, TROU_HYP = -0.4, -0.6
    sim = np.zeros((n, m))
    for i, r in enumerate(ref):
        for j, h in enumerate(hyp):
            s = ressemblance(r, h)
            # Mots collés / coupés différemment (« gonna » / « going ») : on tolère jusqu'à 0,5.
            sc = 2 * s - 1 if s >= 0.5 else -1.0
            if fenetres is not None and t_hyp is not None:
                lo, hi = fenetres[i]
                ecart = max(lo - t_hyp[j], t_hyp[j] - hi, 0.0)
                sc -= min(2.0, ecart * 1.5)
            sim[i, j] = sc
    D = np.zeros((n + 1, m + 1))
    D[0, 1:] = TROU_HYP * np.arange(1, m + 1)
    D[1:, 0] = 0.0                                     # préfixe de la référence gratuit
    P = np.zeros((n + 1, m + 1), np.int8)              # 0 diag, 1 haut (trou hyp), 2 gauche (trou ref)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = D[i - 1, j - 1] + sim[i - 1, j - 1]
            u = D[i - 1, j] + (0.0 if j == m else TROU_REF)   # suffixe de la référence gratuit
            l = D[i, j - 1] + TROU_HYP
            if d >= u and d >= l:
                D[i, j], P[i, j] = d, 0
            elif u >= l:
                D[i, j], P[i, j] = u, 1
            else:
                D[i, j], P[i, j] = l, 2
    paires = []
    i, j = n, m
    while i > 0 and j > 0:
        if P[i, j] == 0:
            if sim[i - 1, j - 1] > 0:
                paires.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif P[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return paires[::-1]


def _syllabes(mot: str) -> int:
    return max(1, len(re.findall(r"[aeiouyàâäéèêëîïôöûùüœæ]+", mot.lower())))


@dataclass
class MotSynchro:
    texte: str        # texte affiché (LRCLIB si dispo)
    debut: float      # temps de la chanson
    fin: float
    ligne: int
    source: str       # « aligne », « interpole », « lrc », « whisper »


def synchroniser(ref_mots: list[tuple[str, int, float]], entendus: list[MotEntendu]
                 ) -> tuple[list[MotSynchro], dict]:
    """ref_mots : (texte, n° de ligne, temps LRC du mot). Retourne les mots avec les temps de Whisper
    et des statistiques (alignés, décalage estimé)."""
    ref_cles = [normaliser(t) for t, _, _ in ref_mots]
    hyp_cles = [normaliser(w.texte) for w in entendus]
    t_ref = [t for _, _, t in ref_mots]
    t_hyp = [w.debut for w in entendus]
    paires = aligner(ref_cles, hyp_cles)
    decalage = None
    if len(paires) >= 4:
        # 2e passe. Décalage LOCAL entre le LRC (version officielle) et ton fichier : médiane des
        # paires voisines de la 1re passe (une version YouTube coupée change de décalage en route).
        idx = np.array([i for i, _ in paires])
        dec = np.array([t_hyp[j] - t_ref[i] for i, j in paires])
        debut_ligne: dict[int, float] = {}
        for t, (_, ligne, _) in zip(t_ref, ref_mots):
            debut_ligne[ligne] = min(t, debut_ligne.get(ligne, t))
        lignes_triees = sorted(debut_ligne)
        suite = {l: debut_ligne[n] for l, n in zip(lignes_triees, lignes_triees[1:])}
        fenetres = []
        for i, (_, ligne, t) in enumerate(ref_mots):
            proches = np.argsort(np.abs(idx - i))[:9]
            d = float(np.median(dec[proches]))
            ls = debut_ligne[ligne]
            le = min(suite.get(ligne, ls + 8.0), ls + 8.0)
            fenetres.append((ls + d - 0.5, le + d + 0.5))
        paires = aligner(ref_cles, hyp_cles, t_hyp, fenetres)
        if len(paires) >= 4:
            decalage = float(np.median([t_hyp[j] - t_ref[i] for i, j in paires]))
    stats = {"alignes": len(paires), "ref": len(ref_mots), "entendus": len(entendus), "decalage": decalage}
    if len(paires) < 3:
        return [], stats
    temps: list[float | None] = [None] * len(ref_mots)
    fins: list[float | None] = [None] * len(ref_mots)
    for i, j in paires:
        temps[i], fins[i] = entendus[j].debut, entendus[j].fin
    src = ["aligne" if t is not None else "" for t in temps]
    # Substitutions : entre deux paires, autant de mots non alignés des deux côtés (« in » entendu
    # « and ») → on prend les temps entendus, dans l'ordre.
    for (ia, ja), (ib, jb) in zip(paires, paires[1:]):
        if ib - ia == jb - ja > 1 and ib - ia <= 4:
            for k in range(1, ib - ia):
                temps[ia + k], fins[ia + k] = entendus[ja + k].debut, entendus[ja + k].fin
                src[ia + k] = "substitue"
    alignes = [i for i, t in enumerate(temps) if t is not None]
    # Mots non alignés ENTRE deux mots alignés : interpolés au prorata des syllabes.
    for a, b in zip(alignes, alignes[1:]):
        if b - a <= 1:
            continue
        trou = list(range(a + 1, b))
        syl = np.array([_syllabes(ref_mots[k][0]) for k in trou], float)
        t1 = temps[b]
        t0 = min(fins[a] if fins[a] is not None else temps[a], temps[a] + 0.6, t1)
        if t1 - t0 >= 0.08 * len(trou):
            # Les mots manquants se partagent l'espace entre la fin du mot a et le début du mot b.
            avant = np.concatenate([[0.0], np.cumsum(syl)])[:-1] / syl.sum()
        else:
            # Pas de place après le mot a : on partage depuis son DÉBUT (Whisper a pu fusionner).
            t0 = temps[a]
            if t1 - t0 < 0.08 * (len(trou) + 1):
                continue        # aucun temps pour les chanter : coupés dans cette version (edit)
            s_a = _syllabes(ref_mots[a][0])
            avant = (s_a + np.concatenate([[0.0], np.cumsum(syl)])[:-1]) / (s_a + syl.sum())
        for k, p in zip(trou, avant):
            temps[k] = float(t0 + (t1 - t0) * p)
            src[k] = "interpole"
    # Avant le 1er / après le dernier mot aligné : temps LRC + décalage estimé, seulement hors de la
    # zone que Whisper a entendue (sinon c'est que ces mots ne sont pas chantés là).
    if decalage is not None:
        premier, dernier = temps[alignes[0]], temps[alignes[-1]]
        for k in list(range(0, alignes[0])) + list(range(alignes[-1] + 1, len(ref_mots))):
            t = t_ref[k] + decalage
            if t < premier - 0.3 or t > dernier + 0.3:
                temps[k], src[k] = float(t), "lrc"
    res = []
    for k, (texte, ligne, _) in enumerate(ref_mots):
        if temps[k] is None:
            continue
        res.append(MotSynchro(texte, float(temps[k]), float(fins[k] or temps[k]), ligne, src[k]))
    res.sort(key=lambda m: m.debut)
    # Temps strictement croissants (un mot interpolé ne passe pas devant un mot aligné).
    for a, b in zip(res, res[1:]):
        if b.debut < a.debut + 0.04:
            b.debut = a.debut + 0.04
    for k, m in enumerate(res):
        nxt = res[k + 1].debut if k + 1 < len(res) else m.debut + 0.9
        m.fin = float(min(nxt, max(m.fin, m.debut + 0.12), m.debut + 0.9))
    return res, stats


def filtrer_entendus(entendus: list[MotEntendu]) -> list[MotSynchro]:
    """Sans texte de référence : mots de Whisper, sans hallucinations ni mots douteux.
    Les lignes = segments de Whisper, recoupés aux pauses de plus de 0,7 s."""
    res: list[MotSynchro] = []
    ligne, prec = 0, None
    for w in entendus:
        cle = normaliser(w.texte)
        if not cle or w.proba < 0.30 or w.fin - w.debut > 3.0:
            continue
        # « baby baby baby baby » : au-delà de 2 répétitions collées, c'est une boucle de Whisper.
        if len(res) >= 2 and normaliser(res[-1].texte) == cle == normaliser(res[-2].texte):
            continue
        if prec is not None and (w.segment != prec.segment or w.debut - prec.fin > 0.7):
            ligne += 1
        res.append(MotSynchro(w.texte, w.debut, w.fin, ligne, "whisper"))
        prec = w
    for k, m in enumerate(res):
        nxt = res[k + 1].debut if k + 1 < len(res) else m.debut + 0.9
        m.fin = float(min(nxt, max(m.fin, m.debut + 0.12), m.debut + 0.9))
    return res


def prompt_paroles(lignes: list[tuple[float, str]], debut: float, fin: float, marge: float = 25.0) -> str:
    """Paroles LRC autour de l'extrait (marge large : la version YouTube peut être décalée)."""
    txt = [c for t, c in lignes if debut - marge <= t <= fin + marge and c.strip()]
    vus, propres = set(), []
    for c in txt:                      # pas de refrain répété dans le prompt (Whisper le recopierait)
        c = re.sub(r"<\d+:\d+(?:[.:]\d+)?>", "", c).strip()
        c = re.sub(r"(?i)\b(.{8,}?)(?:[,\s]+\1\b)+", r"\1", c)   # « X, X » dans une même ligne → « X »
        if c.lower() not in vus:
            vus.add(c.lower())
            propres.append(c)
    return " ".join(propres)[-600:]
