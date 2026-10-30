"""Détection des meilleurs moments dans les clips Medal, à partir de l'audio du jeu.

Idée : un kill ou une grosse action produit des sons forts et secs (tirs, impacts,
son de kill) et souvent plusieurs d'affilée. On mesure donc, pour chaque instant :
  - l'attaque des sons (onset strength) : les tirs et impacts ;
  - le volume (RMS) ;
  - la densité de tirs sur 2,5 s : les rafales et multi-kills.
Les pics de ce score deviennent des « moments ».
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from outils import decoder_audio, infos_video

SR = 22050
HOP = 512
FPS_ANALYSE = SR / HOP  # environ 43 mesures par seconde


@dataclass
class Moment:
    clip: Path
    t: float            # instant du pic d'action dans le clip (s)
    score: float
    duree_clip: float
    a_audio: bool
    largeur: int
    hauteur: int
    etiquette: str = ""  # événement Medal (ex. « 3K »), vide si trouvé par le son
    kills: list = field(default_factory=list)  # séquence de multi-kill : [(nom, temps), …]
    arme: str = ""       # Valorant : arme du kill (lue dans le fil des kills), vide si inconnue
    agent: str = ""      # Valorant : agent joué dans ce clip

    def __str__(self) -> str:
        source = self.etiquette or "son"
        return f"{self.clip.name} @ {self.t:5.1f}s [{source}] (score {self.score:.2f})"


def _caracteristiques(y: np.ndarray):
    if len(y) < SR:
        return None
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    n = min(len(onset), len(rms))
    return onset[:n], rms[:n]


def _normaliser(x, med, haut):
    return np.clip((x - med) / (haut - med + 1e-9), 0, 3)


def _mouvement(clip: Path, duree: float, n: int) -> np.ndarray | None:
    """Jeux sans kills : vitesse de la caméra (0-3, comme le score du son) sur les n mesures du son.
    None pour les jeux à kills (le son des tirs suffit) ou si la vidéo est illisible."""
    try:
        from jeux import profil
        if profil(clip).kills:
            return None
        from mouvement import _decalages
        d, fps = _decalages(clip, 0.0, duree, w=96, h=54)
    except Exception:
        return None
    if len(d) < 3:
        return None
    v = uniform_filter1d(np.hypot(d[:, 0], d[:, 1]), size=max(1, int(0.5 * fps)))
    haut = np.percentile(v, 95)
    if haut <= 1e-6:
        return None
    v = np.clip(v / haut, 0, 1.5) * 2
    return np.interp(np.arange(n) / FPS_ANALYSE, np.arange(len(v)) / fps, v)


def trouver_moments(clips: list[Path], max_par_clip: int = 3, log=print) -> list[Moment]:
    """Analyse tous les clips et retourne les moments triés du meilleur au moins bon."""
    analyses = []
    for i, clip in enumerate(clips, 1):
        try:
            info = infos_video(clip)
        except Exception as e:  # fichier corrompu ou illisible : on l'ignore
            log(f"  ignoré ({e}) : {clip.name}")
            continue
        if info["duree"] < 2:
            continue
        y = decoder_audio(clip, SR) if info["a_audio"] else np.zeros(0, np.float32)
        analyses.append((clip, info, _caracteristiques(y)))
        log(f"  [{i}/{len(clips)}] analysé : {clip.name}")

    avec_son = [f for _, _, f in analyses if f is not None]
    if avec_son:
        tous_on = np.concatenate([f[0] for f in avec_son])
        tous_rms = np.concatenate([f[1] for f in avec_son])
        on_med, on_haut = np.median(tous_on), np.percentile(tous_on, 98)
        rms_med, rms_haut = np.median(tous_rms), np.percentile(tous_rms, 98)

    moments: list[Moment] = []
    for clip, info, feats in analyses:
        base = dict(clip=clip, duree_clip=info["duree"], a_audio=info["a_audio"],
                    largeur=info["largeur"], hauteur=info["hauteur"])
        if feats is None:
            # Pas de son : on prend vers la fin, là où Medal place habituellement l'action.
            moments.append(Moment(t=info["duree"] * 0.75, score=0.05, **base))
            continue

        onset, rms = feats
        on_n = _normaliser(onset, on_med, on_haut)
        rms_n = _normaliser(rms, rms_med, rms_haut)
        score = uniform_filter1d(0.55 * on_n + 0.45 * rms_n, size=max(1, int(0.35 * FPS_ANALYSE)))
        mvt = _mouvement(clip, info["duree"], len(score))
        if mvt is not None:
            # Jeux sans kills (Minecraft, courses…) : l'action se voit autant qu'elle s'entend
            # (virage, chute, combat). Le mouvement de la caméra compte pour un tiers du score.
            score = 0.67 * score + 0.33 * mvt

        # Densité de tirs/impacts sur 2,5 s (rafales, multi-kills)
        pics_on, _ = find_peaks(on_n, height=0.8, distance=max(1, int(0.06 * FPS_ANALYSE)))
        impulsions = np.zeros_like(on_n)
        impulsions[pics_on] = 1.0
        fenetre = int(2.5 * FPS_ANALYSE)
        densite = uniform_filter1d(impulsions, size=fenetre) * fenetre
        score = score + 0.35 * np.clip(densite / 8.0, 0, 2)

        # Medal sauvegarde les N dernières secondes APRÈS l'action : la fin compte un peu plus.
        position = np.linspace(0, 1, len(score))
        score = score * (0.85 + 0.30 * position)

        pics, _ = find_peaks(score, distance=max(1, int(4 * FPS_ANALYSE)))
        temps = pics / FPS_ANALYSE
        ok = (temps > 0.6) & (temps < info["duree"] - 0.3)
        pics = pics[ok]
        if len(pics) == 0:
            pics = np.array([int(np.argmax(score))])
        ordre = pics[np.argsort(score[pics])[::-1]]
        if mvt is not None:
            # Jeux sans kills : un pic du son est souvent un clic de menu (inventaire, réglages) ou un
            # écran noir. On garde les pics dont l'image est propre (mêmes règles que le montage).
            from mouvement import image_propre
            propres = [p for p in ordre[:max_par_clip + 4]
                       if image_propre(clip, max(0.0, p / FPS_ANALYSE - 0.9), 1.6)]
            if len(propres) < len(ordre[:max_par_clip + 4]):
                log(f"    {len(ordre[:max_par_clip + 4]) - len(propres)} moment(s) écarté(s) (menu, écran noir) : {clip.name}")
            ordre = np.array(propres, dtype=int)
        meilleurs = ordre[:max_par_clip]
        for p in meilleurs:
            moments.append(Moment(t=float(p / FPS_ANALYSE), score=float(score[p]), **base))

    moments.sort(key=lambda m: m.score, reverse=True)
    return moments


# Les moments venant des événements Medal passent toujours devant ceux trouvés par le son.
SCORE_EVENEMENT = 10.0
FUSION_KILLS = 2.0   # kills à moins de 2 s d'écart = une seule action (ex. double kill rapide)


def _affiner(t: float, onset: np.ndarray | None) -> float:
    """Recale l'événement sur le coup de feu le plus fort juste avant (le bookmark Medal
    peut arriver quelques dixièmes de seconde après le kill)."""
    if onset is None or len(onset) == 0:
        return t
    a = max(0, int((t - 0.4) * FPS_ANALYSE))
    b = min(len(onset), int((t + 0.3) * FPS_ANALYSE) + 1)
    if b - a < 3:
        return t
    fen = onset[a:b]
    i = int(np.argmax(fen))
    if fen[i] < 1.8 * (np.median(onset) + 1e-9):   # pas de coup de feu net : on garde le bookmark
        return t
    return (a + i) / FPS_ANALYSE


FILTRES = {
    "tous": "Tous les kills",
    "onetap": "One taps seulement",
    "multi": "Multi-kills (3K et plus)",
    "ace": "Ace, 4K et clutch",
}


# Zone du compteur de munitions de Valorant (HUD par défaut, image 16:9), en fraction de l'image.
HUD_MUNITIONS = (0.662, 0.692, 0.930, 0.972)


def tirs_avant_kill(clip: Path, t: float, largeur: int, hauteur: int) -> int | None:
    """Compte les balles tirées juste avant un kill en lisant le compteur de munitions à l'écran
    (chaque tir change le chiffre). 1 seule balle dans les 0,8 s avant le kill = one tap.
    None si on ne peut pas lire le HUD (pas du 16:9, vidéo illisible)."""
    import subprocess
    from outils import _FLAGS
    if hauteur <= 0 or abs(largeur / hauteur - 16 / 9) > 0.05:
        return None
    x0, x1, y0, y1 = HUD_MUNITIONS
    W, H, fps = 1920, 1080, 60
    cw, ch = int((x1 - x0) * W), int((y1 - y0) * H)
    debut = max(0.0, t - 0.85)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{debut:.3f}", "-i", str(clip), "-t", f"{t + 0.04 - debut:.3f}",
           "-vf", f"fps={fps},scale={W}:{H},crop={cw}:{ch}:{int(x0 * W)}:{int(y0 * H)},format=gray",
           "-f", "rawvideo", "-"]
    try:
        brut = subprocess.run(cmd, capture_output=True, creationflags=_FLAGS, timeout=30).stdout
    except Exception:
        return None
    n = len(brut) // (cw * ch)
    if n < 10:
        return None
    m = np.frombuffer(brut[:n * cw * ch], np.uint8).reshape(n, ch, cw) > 215
    tirs, dernier = 0, -9
    for i in range(1, n):
        if float(np.mean(m[i] ^ m[i - 1])) > 0.04:
            if i - dernier > 1:
                tirs += 1
            dernier = i
    return tirs


def garde(etiquettes: list[str], filtre: str) -> bool:
    """Le groupe de kills correspond-il au filtre choisi ?"""
    from medal_db import poids
    noms = [e.lower() for e in etiquettes]
    if filtre == "onetap":
        return any("one tap" in n for n in noms)
    if filtre == "multi":
        return any(poids(n) >= 2.4 for n in noms)
    if filtre == "ace":
        return any(("ace" in n) or ("clutch" in n) or poids(n) >= 3.3 for n in noms)
    return True


def _cle(x: str) -> str:
    return "".join(c for c in x.lower() if c.isalnum())


def moments_medal(clips: list[Path], infos: dict, log=print, filtre: str = "tous", index: dict | None = None,
                  armes: list[str] | None = None, agents: list[str] | None = None) -> tuple[list[Moment], set[Path]]:
    """Moments tirés des événements enregistrés par Medal (kills, multi-kills, ace…).
    index : index des kills (one taps, armes, agents déjà analysés) pour aller plus vite.
    armes / agents : garde seulement les kills faits avec ces armes / ces agents (Valorant).
    Retourne (moments, clips qui avaient des événements)."""
    from medal_db import arme_evenement, info_pour, poids
    from index_kills import infos_kill
    armes_v = {_cle(a) for a in armes} if armes else None
    agents_v = {_cle(a) for a in agents} if agents else None

    moments: list[Moment] = []
    couverts: set[Path] = set()
    absolu: dict[int, tuple[str, float]] = {}   # moment -> (partie, heure réelle du kill)
    for k, clip in enumerate(clips, 1):
        ic = info_pour(infos, clip)
        if ic is None or not ic.evenements:
            continue
        try:
            info = infos_video(clip)
        except Exception:
            continue
        duree = info["duree"]
        if duree < 2:
            continue
        # Le bookmark de Medal tombe sur la balle qui tue (mesuré à ±0,05 s) : on le garde tel quel.
        evts = [(nom, f * duree) for nom, f in ic.evenements]
        evts = [(n, t) for n, t in evts if 0.3 < t < duree - 0.2]
        if not evts:
            continue
        couverts.add(clip)
        bonus_clip = 0.25 * max(poids(n) for n, _ in evts)   # un clip d'ace fait briller tous ses kills

        # Regroupe les kills rapprochés : le moment est placé sur le dernier kill du groupe.
        groupes: list[list[tuple[str, float]]] = []
        for n, t in evts:
            if groupes and t - groupes[-1][-1][1] <= FUSION_KILLS:
                groupes[-1].append((n, t))
            else:
                groupes.append([(n, t)])
        base = dict(clip=clip, duree_clip=duree, a_audio=info["a_audio"],
                    largeur=info["largeur"], hauteur=info["hauteur"])
        est_valorant = "valorant" in str(clip).lower()
        connu = (lambda t: infos_kill(index, clip.name, t)) if index else (lambda t: None)
        agent_clip = next((ki.get("agent") for _, t in evts if (ki := connu(t)) and ki.get("agent")), None) or ""
        if agents_v is not None and _cle(agent_clip) not in agents_v:
            log(f"  [{k}/{len(clips)}] ignoré : {clip.name} (agent {agent_clip or 'inconnu'})")
            continue
        trouves = []
        for g in groupes:
            noms = [n for n, _ in g]
            ik = connu(g[-1][1])
            arme = (ik or {}).get("arme") or arme_evenement(ic, g[-1][1] / duree)   # CS2 : Medal donne l'arme
            if armes_v is not None and _cle(arme) not in armes_v:
                continue
            # Medal ne marque les one taps que dans CS2 : dans Valorant, on les détecte en lisant
            # le compteur de munitions (une seule balle tirée avant le kill), ou dans l'index.
            if (filtre in ("onetap", "tous") and est_valorant and len(g) == 1
                    and not any("one tap" in n.lower() for n in noms)
                    and any(x in g[0][0].lower() for x in ("kill", "k", "ace", "headshot"))
                    and "knife" not in g[0][0].lower()):
                if ik is not None and ik.get("one_tap") is not None:
                    un = ik["one_tap"]
                else:
                    un = tirs_avant_kill(clip, g[0][1], info["largeur"], info["hauteur"]) == 1
                if un:
                    noms.append("One Tap")
            if not garde(noms, filtre):
                continue
            meilleur = max(poids(n) for n in noms)
            score = SCORE_EVENEMENT + meilleur + 0.4 * (len(g) - 1) + bonus_clip
            etiquette = "+".join(noms)
            trouves.append(etiquette)
            moments.append(Moment(t=g[-1][1], score=score, etiquette=etiquette, arme=arme, agent=agent_clip, **base))
            if ic.match and ic.debut is not None:
                absolu[id(moments[-1])] = (ic.match, ic.debut.timestamp() + g[-1][1])
        log(f"  [{k}/{len(clips)}] analysé : {clip.name} ({', '.join(trouves) or 'rien pour ce filtre'})")
    moments.sort(key=lambda m: m.score, reverse=True)
    # Medal enregistre parfois deux clips de la même action (à 1 s d'écart) : on garde un seul exemplaire.
    gardes, vus = [], []
    for m in moments:
        a = absolu.get(id(m))
        if a is not None and any(a[0] == b[0] and abs(a[1] - b[1]) < 2.5 for b in vus):
            continue
        if a is not None:
            vus.append(a)
        gardes.append(m)
    if len(gardes) < len(moments):
        log(f"  {len(moments) - len(gardes)} kill(s) en double (clips qui se chevauchent) ignoré(s)")
    return gardes, couverts


def sequences_medal(clips: list[Path], infos: dict, ecart_max: float = 4.0, min_kills: int = 3,
                    log=print) -> list[Moment]:
    """Multi-kills rapides (ex. 4 kills en 3 s) à montrer en entier dans un seul plan.
    Ne lit que la base Medal (rapide) ; la vidéo n'est ouverte que pour les séquences gardées."""
    from medal_db import info_pour, poids

    trouvees = []
    for clip in clips:
        ic = info_pour(infos, clip)
        if ic is None or len(ic.evenements) < min_kills or ic.duree <= 0:
            continue
        # Un clutch, un knock (Apex, Fortnite : l'ennemi est seulement à terre) ou une assistance ne sont
        # pas un kill de plus dans la séquence.
        evts = sorted(((n, f * ic.duree) for n, f in ic.evenements
                       if not any(x in n.lower() for x in ("clutch", "knock", "assist"))), key=lambda e: e[1])
        meilleur, groupe = [], []
        for e in evts:
            groupe = groupe + [e] if groupe and e[1] - groupe[-1][1] <= ecart_max else [e]
            if len(groupe) > len(meilleur) or (len(groupe) == len(meilleur)
                                                and groupe[-1][1] - groupe[0][1] < meilleur[-1][1] - meilleur[0][1]):
                meilleur = list(groupe)
        if len(meilleur) < min_kills:
            continue
        duree_seq = meilleur[-1][1] - meilleur[0][1]
        score = SCORE_EVENEMENT + 2.0 + 1.2 * len(meilleur) + max(poids(n) for n, _ in meilleur) - 0.25 * duree_seq
        trouvees.append((score, clip, meilleur))

    trouvees.sort(key=lambda x: x[0], reverse=True)
    resultat = []
    for score, clip, kills in trouvees[:6]:
        try:
            info = infos_video(clip)
        except Exception:
            continue
        resultat.append(Moment(clip=clip, t=kills[-1][1], score=score, duree_clip=info["duree"],
                               a_audio=info["a_audio"], largeur=info["largeur"], hauteur=info["hauteur"],
                               etiquette="+".join(n for n, _ in kills), kills=kills))
    for m in resultat[:3]:
        log(f"  séquence : {m.clip.name} — {len(m.kills)} kills en {m.kills[-1][1] - m.kills[0][1]:.1f} s")
    return resultat
