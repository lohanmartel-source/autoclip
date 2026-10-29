"""Réactions au micro : ta voix juste après les kills (« let's gooo », « ace easy »…), gardée par-dessus
la musique même quand le montage a déjà coupé au plan suivant (L-cut, comme un vrai monteur),
avec des sous-titres mot par mot générés par Whisper.

Medal enregistre le micro sur une piste séparée (« Microphone ») : on la lit directement.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from outils import _FLAGS, flux_audio

SR = 16000
AVANT, APRES = 0.4, 3.6     # fenêtre écoutée autour du kill (s)
CIBLE_VOIX_DB = -11.0       # niveau visé pour ta voix (RMS pendant que tu parles), au-dessus de la musique
SEUIL_VOIX = 0.035          # niveau RMS minimum pour considérer qu'il y a de la voix
MAX_REACTIONS = 5
AMORCE_WHISPER = ("Nice! Let's go! Ace, easy, one tap, clutch, GG, oh my god. "
                  "Sale, malade, c'est fou, let's go, tabarnak.")


@dataclass
class Reaction:
    clip: Path
    piste: int                 # index de la piste micro dans le clip
    src_debut: float           # début dans le clip (s)
    duree: float
    debut: float               # début dans le montage final (s)
    gain_db: float = 0.0       # pour ramener la voix à un niveau constant
    intervalles: list = field(default_factory=list)  # [(début, fin)] où tu parles vraiment (temps du montage)
    mots: list = field(default_factory=list)   # [(texte, début, fin)] en temps du montage


def _lire_micro(clip: Path, piste: int, debut: float, duree: float) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, debut):.3f}", "-t", f"{duree:.3f}", "-i", str(clip),
           "-map", f"0:{piste}", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    try:
        brut = subprocess.run(cmd, capture_output=True, creationflags=_FLAGS, timeout=30).stdout
    except Exception:
        return np.zeros(0, np.float32)
    return np.frombuffer(brut, np.float32).copy()


def _zone_voix(y: np.ndarray):
    """Où est ta voix dans l'extrait : (début, fin, niveau RMS en dB pendant la voix, intervalles parlés).
    None si trop peu de voix."""
    if len(y) < SR * 0.3:
        return None
    fen = int(0.05 * SR)
    rms = np.array([np.sqrt(np.mean(y[i:i + fen] ** 2)) for i in range(0, len(y) - fen, fen)])
    voix = rms > max(SEUIL_VOIX, 4 * float(np.percentile(rms, 15)))   # 4x le bruit de fond
    if voix.sum() < 6:                 # moins de 0,3 s de voix
        return None
    # Intervalles parlés (on recolle les petits trous de moins de 0,35 s entre les mots).
    intervalles = []
    for i in np.flatnonzero(voix):
        t0, t1 = i * 0.05, (i + 1) * 0.05
        if intervalles and t0 - intervalles[-1][1] <= 0.35:
            intervalles[-1][1] = t1
        else:
            intervalles.append([t0, t1])
    intervalles = [(a, b) for a, b in intervalles if b - a >= 0.15]
    if not intervalles:
        return None
    debut = max(0.0, intervalles[0][0] - 0.12)
    fin = min(len(y) / SR, intervalles[-1][1] + 0.25)
    niveau = 20 * np.log10(float(np.sqrt(np.mean(rms[voix] ** 2))) + 1e-9)
    return debut, fin, niveau, intervalles


def trouver_reactions(segments, debuts_plans: list[float], duree_totale: float, log=print,
                      maximum: int = MAX_REACTIONS) -> list[Reaction]:
    """Cherche ta voix après les kills les plus importants du montage.
    debuts_plans : moment où chaque plan commence dans le montage final."""
    candidats = []
    for seg, debut_plan in zip(segments, debuts_plans):
        drole = seg.moment.etiquette in ("Rire", "Cri")
        if (seg.role not in ("drop", "apres", "sequence") and not drole) or seg.t_kill is None:
            continue
        m = seg.moment
        t_src = m.kills[-1][1] if seg.role == "sequence" and m.kills else m.t
        priorite = 4 if drole else {"drop": 3, "sequence": 2.5}.get(seg.role, 2 if seg.impact else 1)
        candidats.append((priorite, seg, debut_plan + seg.t_kill, t_src))
    candidats.sort(key=lambda c: c[0], reverse=True)

    reactions: list[Reaction] = []
    for _, seg, t_global, t_src in candidats:
        if len(reactions) >= maximum:
            break
        piste = flux_audio(seg.moment.clip).get("micro")
        if piste is None:
            continue
        y = _lire_micro(seg.moment.clip, piste, t_src - AVANT, AVANT + APRES)
        zone = _zone_voix(y)
        if zone is None:
            continue
        z0, z1, niveau, parle = zone
        debut = t_global - AVANT + z0
        duree = min(z1 - z0, duree_totale - 0.3 - debut)
        if duree < 0.3 or debut < 0:
            continue
        # Pas deux réactions qui se chevauchent.
        if any(debut < r.debut + r.duree + 0.15 and r.debut < debut + duree + 0.15 for r in reactions):
            continue
        # Gain réglé sur le niveau de ta voix QUAND tu parles (pas sur un pic) : on t'entend toujours.
        gain = float(np.clip(CIBLE_VOIX_DB - niveau, -6, 24))
        origine = t_global - AVANT          # temps du montage qui correspond au début de l'extrait
        intervalles = [(origine + a, min(origine + b, debut + duree)) for a, b in parle
                       if origine + a < debut + duree]
        reactions.append(Reaction(clip=seg.moment.clip, piste=piste, src_debut=t_src - AVANT + z0,
                                  duree=duree, debut=debut, gain_db=gain, intervalles=intervalles))
    reactions.sort(key=lambda r: r.debut)
    if reactions:
        log(f"  {len(reactions)} réaction(s) au micro gardée(s) : "
            + ", ".join(f"{r.debut:.1f} s ({r.duree:.1f} s)" for r in reactions))
    return reactions


_modele = None


def sous_titrer(reactions: list[Reaction], log=print, langue: str = "fr") -> None:
    """Ajoute les mots (avec leur timing) à chaque réaction grâce à Whisper. Sans Whisper : rien."""
    global _modele
    if not reactions:
        return
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log("  sous-titres : Whisper n'est pas installé (pip install faster-whisper)")
        return
    try:
        if _modele is None:
            log("  chargement de Whisper (sous-titres)…")
            # Même dossier de modèles que la synchro des paroles (DONNEES/modeles, progression dans
            # le journal) ; repli : le cache HuggingFace habituel.
            try:
                from synchro_paroles import chemin_modele
                chemin = chemin_modele("small", log) or "small"
            except Exception:
                chemin = "small"
            _modele = WhisperModel(chemin, device="cpu", compute_type="int8")
    except Exception as e:
        log(f"  sous-titres indisponibles ({e})")
        return
    for r in reactions:
        y = _lire_micro(r.clip, r.piste, r.src_debut, r.duree)
        if len(y) < SR * 0.2:
            continue
        try:
            # Testé sur tes clips : beam 5 + filtre de voix = « Nice, nice, nice, six kill ace, easy ».
            # langue « en » : Whisper traduit directement en anglais (pour toucher plus de monde).
            segs, _ = _modele.transcribe(y, beam_size=5, word_timestamps=True, vad_filter=True,
                                         initial_prompt=AMORCE_WHISPER if langue != "en" else None,
                                         condition_on_previous_text=False,
                                         task="translate" if langue == "en" else "transcribe")
            for s in segs:
                if getattr(s, "no_speech_prob", 0) > 0.6:
                    continue
                for w in (s.words or []):
                    texte = w.word.strip()
                    if texte and w.probability > 0.25:
                        r.mots.append((texte, r.debut + float(w.start), r.debut + float(w.end)))
        except Exception:
            continue
    for r in reactions:
        if r.mots:
            log(f"  sous-titres à {r.debut:.1f} s : « {' '.join(m[0] for m in r.mots)} »")


def bonus_voix(moments: list, log=print, maximum: int = 45, bonus: float = 0.8) -> list:
    """Un monteur garde les kills où tu réagis : les moments suivis de ta voix au micro passent devant.
    Retourne la liste des moments (scores ajustés)."""
    from dataclasses import replace
    if not moments:
        return moments
    tries = sorted(moments, key=lambda m: m.score, reverse=True)
    resultat, avec_voix = [], 0
    for i, m in enumerate(tries):
        if i < maximum and m.etiquette:
            piste = flux_audio(m.clip).get("micro")
            t = m.kills[-1][1] if m.kills else m.t
            if piste is not None:
                zone = _zone_voix(_lire_micro(m.clip, piste, t - AVANT, AVANT + APRES))
                if zone is not None and sum(b - a for a, b in zone[3]) >= 0.4:
                    m = replace(m, score=m.score + bonus)
                    avec_voix += 1
        resultat.append(m)
    if avec_voix:
        log(f"  {avec_voix} kill(s) avec ta voix au micro : mis en avant")
    return resultat


# ---------------------------------------------------------------- moments drôles (fails, rires, cris)
from chemins import DONNEES as _DONNEES
CACHE_VOIX = _DONNEES / "voix_fortes.json"


def _analyse_voix(clip: Path) -> list[dict]:
    """Moments où tu parles fort (cris, rires) dans tout le clip.
    Rire : la voix « hache » le son 3 à 8 fois par seconde (ha-ha-ha), on le mesure sur l'enveloppe."""
    piste = flux_audio(clip).get("micro")
    if piste is None:
        return []
    y = _lire_micro(clip, piste, 0.0, 600.0)
    if len(y) < SR:
        return []
    fen = int(0.025 * SR)                      # 40 mesures par seconde
    n = len(y) // fen
    rms = np.sqrt(np.mean(y[:n * fen].reshape(n, fen) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-9)
    bruit = float(np.percentile(rms, 20))
    actif = rms > max(SEUIL_VOIX * 1.6, 5 * bruit)
    moments, i = [], 0
    while i < n:
        if not actif[i]:
            i += 1
            continue
        j, trou = i, 0
        while j < n and trou <= 14:             # on recolle les trous de moins de 0,35 s
            trou = 0 if actif[j] else trou + 1
            j += 1
        j -= trou
        duree = (j - i) * 0.025
        if duree >= 0.6:
            env = rms[i:j] - rms[i:j].mean()
            spectre = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
            freqs = np.fft.rfftfreq(len(env), 0.025)
            total = spectre[(freqs > 0.5) & (freqs < 15)].sum() + 1e-12
            rire = float(spectre[(freqs >= 3) & (freqs <= 8)].sum() / total)
            fort = float(np.percentile(db[i:j], 90))
            score = (fort + 32) / 6 + min(duree, 4.0) * 0.6 + 4 * rire
            moments.append({"debut": round(i * 0.025, 2), "duree": round(duree, 2), "fort": round(fort, 1),
                            "rire": round(rire, 2), "score": round(score, 2)})
        i = j + 1
    moments.sort(key=lambda m: m["score"], reverse=True)
    return moments[:4]


def moments_droles(clips: list[Path], log=print, maximum: int = 40) -> list:
    """Moments où tu cries ou ris au micro, pour un montage « fails et rires ».
    Analyse gardée en cache (une seule fois par clip)."""
    import json
    from analyse_clips import Moment
    from outils import infos_video
    try:
        cache = json.loads(CACHE_VOIX.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    res = []
    for k, clip in enumerate(clips, 1):
        try:
            cle = f"{clip.name}:{int(clip.stat().st_mtime)}"
        except OSError:
            continue
        if cle not in cache:
            cache[cle] = _analyse_voix(clip)
            if k % 10 == 0:
                log(f"  [{k}/{len(clips)}] voix analysée")
        if not cache[cle]:
            continue
        try:
            info = infos_video(clip)
        except Exception:
            continue
        for m in cache[cle][:2]:
            if m["fort"] < -24:          # pas assez fort pour être drôle
                continue
            t = m["debut"] + 0.15
            if not 1.5 < t < info["duree"] - 1.0:
                continue
            etiquette = "Rire" if m["rire"] > 0.35 else "Cri"
            res.append(Moment(clip=clip, t=t, score=10 + m["score"], duree_clip=info["duree"],
                              a_audio=info["a_audio"], largeur=info["largeur"], hauteur=info["hauteur"],
                              etiquette=etiquette))
    try:
        CACHE_VOIX.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass
    res.sort(key=lambda m: m.score, reverse=True)
    log(f"  {len(res)} moment(s) où tu cries ou ris trouvés")
    return res[:maximum]
