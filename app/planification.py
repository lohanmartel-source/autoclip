"""Planification du montage : découpe calée sur les beats, choix des moments et des effets.

Structure d'un montage (inspirée des edits YouTube Shorts / TikTok) :
  - « montée »  : avant le drop, plans plus longs, transitions douces, petits zooms sur chaque beat ;
  - « approche » : juste avant le drop, le même clip que le drop, qui monte en tension (zoom lent) ;
  - « drop »     : coupe PILE sur le kill au moment du drop : flash, impact frame, zoom punch,
                   tremblement, puis ralenti ;
  - « après »    : plans courts sur les beats. Le kill tombe toujours sur un beat, et les meilleurs
                   kills sont placés sur les coups les plus forts de la musique. Entre les plans :
                   transitions « coupées » (whip pan, zoom, spin, glitch, flash).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from analyse_clips import Moment
from analyse_musique import Musique

RYTHMES = {"calme": 1.4, "normal": 1.0, "nerveux": 0.6, "fun": 3.6}  # fun : plans longs (le temps de rire)
TRANSITIONS_DOUCES = ["fade", "smoothleft", "smoothup", "slideleft", "circleopen",
                      "zoomin", "hblur", "radial", "fadewhite", "dissolve", "pixelize"]
# Transitions « coupées » : un effet de sortie sur le plan d'avant + un effet d'entrée sur le suivant,
# avec la coupe exactement sur le beat.
TRANSITIONS_COUPEES = ["whip", "zoom", "spin", "glitch", "flash", "leak", "secousse", "whip", "zoom", "coupe"]
SORTIES = ("whip", "zoom", "spin", "flash")   # transitions qui ont aussi un effet de sortie
EFFETS_KILL = ["punch", "shake", "rgb", "flash", "pulse"]

# Styles de montage : ce qui est interdit, et par quoi on le remplace.
STYLES = {
    "auto": {},
    "phonk": {      # agressif : glitchs, secousses, impacts partout
        "entrees": {"leak": "glitch", "coupe": "secousse", None: None},
        "ajout_kill": ["shake"],
    },
    "cinematique": {  # posé : zooms, light leaks, ralentis, pas de glitch
        "effets_interdits": {"rgb", "impact", "shake"},
        "entrees": {"glitch": "leak", "secousse": "zoom", "spin": "zoom", "flash": "leak"},
    },
    "clean": {      # propre : coupes nettes, zooms simples
        "effets_interdits": {"rgb", "impact", "shake"},
        "entrees": {"glitch": "zoom", "secousse": "whip", "spin": "whip", "leak": "zoom"},
    },
}


@dataclass
class Morceau:
    """Portion d'un clip source jouée à une vitesse donnée."""
    debut: float      # dans le clip source (s)
    vitesse: float    # 1 = normal, 0.5 = ralenti x2
    images: int       # nombre d'images en sortie
    inverse: bool = False  # lu à l'envers (effet rewind)
    gel: bool = False      # arrêt sur image (l'image à « debut » reste figée)

    def duree_source(self, fps: int) -> float:
        return self.images / fps * self.vitesse


@dataclass
class Segment:
    index: int
    role: str                 # montee / approche / drop / apres
    moment: Moment
    images: int               # images entre ce beat et le prochain
    nb_beats: int
    periode: float = 0.5
    accent: float = 0.0       # force du coup de musique où tombe le kill
    impact: bool = False      # kill sur un coup fort : effets maximum + ralenti
    transition: str | None = None   # transition xfade (fondu) avec le plan précédent
    images_transition: int = 0      # chevauchement avec le plan précédent (avant le beat)
    entree: str | None = None       # transition coupée : effet d'entrée
    sortie: str | None = None       # transition coupée : effet de sortie
    effets: list[str] = field(default_factory=list)
    morceaux: list[Morceau] = field(default_factory=list)
    t_kill: float | None = None     # moment du kill dans ce plan (s, depuis le début du plan rendu)
    texte: str = ""                 # texte affiché au kill (ONE TAP, ACE, 4K…)
    kills_sortie: list = field(default_factory=list)  # séquence : [(temps dans le plan, texte), …]

    def temps_kills(self) -> list[float]:
        if self.kills_sortie:
            return [t for t, _ in self.kills_sortie]
        return [self.t_kill] if self.t_kill is not None else []

    @property
    def images_totales(self) -> int:
        return self.images + self.images_transition

    def beats_locaux(self, fps: int) -> list[float]:
        t0 = self.images_transition / fps
        return [t0 + k * self.periode for k in range(self.nb_beats)]


def _beats_par_plan(periode: float, seuil: float, choix=(1, 2, 4, 8)) -> int:
    for n in choix:
        if n * periode >= seuil:
            return n
    return choix[-1]


def _bornes(dernier: int, drop: int, n_pre: int, n_drop: int, n_post: int) -> list[int]:
    """Index des beats où l'on coupe (début et fin inclus)."""
    pre = []
    i = drop
    while i - n_pre >= 0:
        i -= n_pre
        pre.insert(0, i)
    if pre and pre[0] > 0:
        if pre[0] >= n_pre / 2:
            pre.insert(0, 0)
        else:
            pre[0] = 0
    if not pre and drop > 0:
        pre = [0]
    bornes = pre + [drop]
    fin_drop = min(drop + n_drop, dernier)
    if fin_drop > drop:
        bornes.append(fin_drop)
    i = fin_drop
    while i + n_post <= dernier:
        i += n_post
        bornes.append(i)
    if i < dernier:
        if len(bornes) < 2:
            bornes.append(dernier)
        else:
            bornes[-1] = dernier      # le reste est ajouté au dernier plan
    return sorted(set(bornes))


def _beats_sequence(seq: Moment, p: float) -> tuple[int, float]:
    """Nombre de beats entre le 1er et le dernier kill, et vitesse pour que les deux tombent pile
    sur un beat (légèrement accéléré ou ralenti, entre 0,75x et 1,35x)."""
    ecart = seq.kills[-1][1] - seq.kills[0][1]
    meilleur = None
    for m in range(1, 64):
        v = ecart / (m * p)
        if 0.75 <= v <= 1.35 and (meilleur is None or abs(v - 1) < abs(meilleur[1] - 1)):
            meilleur = (m, v)
    return meilleur or (max(1, round(ecart / p)), ecart / (max(1, round(ecart / p)) * p))


def planifier(musique: Musique, moments: list[Moment], fps: int, rythme: str = "normal",
              ralenti: float = 0.5, graine: int | None = None,
              sequence: Moment | None = None, hook: bool = False,
              style: str = "auto", gel: bool = False,
              profil: dict | None = None) -> tuple[list[Segment], float, float]:
    """Retourne (segments, décalage du début de la musique, durée du montage)."""
    if not moments:
        raise ValueError("aucun moment")
    f = RYTHMES.get(rythme, 1.0)
    if profil:                     # style copié d'une vidéo : même durée de plan
        from copie_style import facteur_rythme
        f = facteur_rythme(profil)
    p = musique.periode
    n_pre = _beats_par_plan(p, 1.4 * f)
    n_post = _beats_par_plan(p, 0.7 * f)
    n_drop = _beats_par_plan(p, 2.2, (2, 4, 8))

    beats = musique.beats
    accents = musique.accents if musique.accents is not None else [0.5] * len(beats)
    bornes = _bornes(len(beats) - 1, musique.drop, n_pre, n_drop, n_post)
    i_drop = bornes.index(musique.drop) if musique.drop in bornes else 0
    dernier = len(beats) - 1

    # Séquence de multi-kill montrée en entier, juste après le drop : 1 beat d'élan, les kills
    # (premier et dernier pile sur un beat), puis 1 beat au ralenti sur le dernier kill.
    i_seq = None
    if sequence is not None and i_drop + 1 < len(bornes):
        m_beats, _ = _beats_sequence(sequence, p)
        nb_seq = m_beats + 2
        fin_drop = bornes[i_drop + 1]
        if fin_drop + nb_seq <= dernier:
            nouvelles = bornes[:i_drop + 2] + [fin_drop + nb_seq]
            i = fin_drop + nb_seq
            while i + n_post <= dernier:
                i += n_post
                nouvelles.append(i)
            if i < dernier:
                if len(nouvelles) - 1 > i_drop + 2:
                    nouvelles[-1] = dernier
                else:
                    nouvelles.append(dernier)
            bornes = nouvelles
            i_seq = i_drop + 1
            moments = [m for m in moments if not (m.clip == sequence.clip and
                       sequence.kills[0][1] - 1.5 <= m.t <= sequence.kills[-1][1] + 1.5)]

    # Le plan avant le drop (« approche ») réutilise le clip du drop : il ne consomme pas de moment,
    # et la séquence a son propre clip.
    def nb_moments_requis(bs, idrop):
        return len(bs) - 1 - (1 if idrop >= 1 else 0) - (1 if i_seq is not None else 0)

    # Pas assez de moments ? On raccourcit : d'abord la fin, puis le début de la montée.
    plancher = (i_drop + 3) if i_seq is not None else 2   # on ne coupe jamais la séquence
    while nb_moments_requis(bornes, i_drop) > len(moments) and len(bornes) > plancher:
        nb_apres = len(bornes) - 2 - i_drop
        if nb_apres > i_drop or i_drop == 0:
            bornes.pop()
        else:
            bornes.pop(0)
            i_drop -= 1
            if i_seq is not None:
                i_seq -= 1
    # Plan final plus long (2 coupes fusionnées) pour finir sur un moment fort.
    seq_fin = i_seq + 1 if i_seq is not None else None
    if len(bornes) - 1 >= i_drop + 3 and bornes[-1] - bornes[-2] <= n_post and len(bornes) - 2 != seq_fin:
        bornes.pop(-2)

    # Hook d'intro : 2 beats de teaser sur le meilleur kill, puis 1 beat de « rewind » (retour en
    # arrière accéléré), et le montage commence. Il faut de la place avant l'approche du drop.
    i_hook = None
    if hook and i_drop >= 1 and bornes[i_drop - 1] >= 3:
        val_drop = bornes[i_drop]
        val_seq = bornes[i_seq] if i_seq is not None else None
        bornes = [0, 2, 3] + [b for b in bornes if b > 3]
        i_drop = bornes.index(val_drop)
        i_seq = bornes.index(val_seq) if val_seq is not None else None
        i_hook = 0

    nb = len(bornes) - 1
    roles = []
    for k in range(nb):
        if i_hook is not None and k == 0:
            roles.append("hook")
        elif i_hook is not None and k == 1:
            roles.append("rewind")
        elif k == i_drop:
            roles.append("drop")
        elif k == i_seq:
            roles.append("sequence")
        elif k == i_drop - 1:
            roles.append("approche")
        elif k < i_drop:
            roles.append("montee")
        else:
            roles.append("apres")

    # Beat où tombera le kill dans chaque plan, et force de la musique à cet endroit.
    def beat_kill(k):
        a, b = bornes[k], bornes[k + 1]
        if roles[k] == "drop":
            return a
        return a + ((b - a) // 2 if b - a >= 2 else 0)

    force = [float(accents[min(beat_kill(k), len(accents) - 1)]) for k in range(nb)]

    # Attribution : le meilleur moment au drop ; puis les meilleurs kills sur les coups les plus
    # forts de la musique ; la montée reçoit les moments restants, du plus calme au plus fort.
    pool = _varier(sorted(moments, key=lambda m: m.score, reverse=True), nb_moments_requis(bornes, i_drop))
    attribution: list[Moment | None] = [None] * nb
    attribution[i_drop] = pool.pop(0)
    if i_drop >= 1:
        attribution[i_drop - 1] = attribution[i_drop]
    if i_seq is not None:
        attribution[i_seq] = sequence
    if i_hook is not None:
        attribution[0] = attribution[1] = attribution[i_drop]
    apres = [k for k in range(nb) if roles[k] == "apres"]
    importance = {k: force[k] + (0.35 if k == nb - 1 else 0.0) for k in apres}
    for k in sorted(apres, key=lambda k: importance[k], reverse=True):
        attribution[k] = pool.pop(0) if pool else attribution[i_drop]
    montee = [k for k in range(nb) if roles[k] == "montee"]
    restant = sorted(pool[:len(montee)], key=lambda m: m.score)
    for k, m in zip(montee, restant):
        attribution[k] = m
    for k in range(nb):
        if attribution[k] is None:
            attribution[k] = attribution[i_drop]
    _eviter_repetitions(attribution, roles)

    # Les coups les plus forts (après le drop) deviennent des « impacts ».
    seuil_impact = sorted((force[k] for k in apres), reverse=True)[max(0, len(apres) // 3 - 1)] if apres else 9

    t0 = float(beats[bornes[0]])
    image = lambda b: int(round((float(beats[b]) - t0) * fps))
    rng = random.Random(graine)
    segments: list[Segment] = []
    for k in range(nb):
        a, b = bornes[k], bornes[k + 1]
        seg = Segment(index=k, role=roles[k], moment=attribution[k], images=image(b) - image(a),
                      nb_beats=b - a, periode=p, accent=force[k],
                      impact=roles[k] == "apres" and force[k] >= seuil_impact and force[k] > 0.6)
        if rythme == "fun":
            seg.impact = False          # pas de ralenti : on garde le moment drôle à vitesse normale
        _choisir_effets(seg, segments[-1] if segments else None, rng, fps)
        _appliquer_style(seg, segments[-1] if segments else None, style, rng)
        if rythme == "fun":
            seg.effets = [e for e in seg.effets if e != "velocity"]
        if profil and seg.role == "apres" and segments and rng.random() < profil.get("flash", 0):
            seg.entree = "flash"            # la vidéo copiée met des flashs sur ses coupes
            segments[-1].sortie = None
        segments.append(seg)
    segments[-1].effets.append("fondu_fin")

    for seg in segments:
        _calculer_morceaux(seg, p, fps, ralenti, gel and rythme != "fun")
        if seg.moment.etiquette.startswith("@") and seg.t_kill is not None:
            # Montage d'équipe : le pseudo de ton ami sur son kill.
            pseudo = "".join(c for c in seg.moment.etiquette.split()[0] if c.isalnum() or c in "@_-.")
            seg.texte = pseudo.upper()
            continue
        if seg.moment.etiquette.startswith("Mort") and seg.t_kill is not None:
            seg.texte = seg.moment.etiquette.upper()          # coach : « MORT - OPERATOR »
            continue
        if seg.role in ("drop", "apres", "hook") and seg.t_kill is not None:
            # Comme un monteur : le texte est réservé aux gros moments, pas à chaque kill.
            t = texte_kill(seg.moment.etiquette)
            if seg.role in ("drop", "hook") or seg.impact or t in ("ACE", "CLUTCH", "ONE TAP"):
                seg.texte = t

    duree = image(bornes[-1]) / fps
    return segments, musique.debut + t0, duree


def _varier(moments: list[Moment], besoin: int, max_par_clip: int = 2) -> list[Moment]:
    """Comme un monteur : pas plus de 2 kills du même clip, sauf s'il en manque."""
    choisis, compte, reste = [], {}, []
    for m in moments:
        if compte.get(m.clip, 0) < max_par_clip:
            choisis.append(m)
            compte[m.clip] = compte.get(m.clip, 0) + 1
        else:
            reste.append(m)
    if len(choisis) < besoin:
        choisis += reste[:besoin - len(choisis)]
    return sorted(choisis, key=lambda m: m.score, reverse=True)


TEXTES = [("ace", "ACE"), ("clutch", "CLUTCH"), ("5k", "5K"), ("4k", "4K"), ("one tap", "ONE TAP"),
          ("knife", "KNIFE"), ("3k", "3K")]


def texte_kill(etiquette: str) -> str:
    e = etiquette.lower()
    for cle, texte in TEXTES:
        if cle in e:
            return texte
    return ""


def _eviter_repetitions(attribution: list, roles: list[str]) -> None:
    """Évite deux plans de suite du même clip (sauf approche → drop, voulu)."""
    fixes = ("drop", "approche", "sequence", "hook", "rewind")     # l'approche et le drop partagent volontairement le même clip
    n = len(attribution)

    def clip_de(i):
        return attribution[i].clip if 0 <= i < n else None

    for k in range(1, n):
        if roles[k - 1] == "approche" and roles[k] == "drop":
            continue
        if clip_de(k) != clip_de(k - 1):
            continue
        a_bouger = k if roles[k] not in fixes else (k - 1 if roles[k - 1] not in fixes else None)
        if a_bouger is None:
            continue
        for j in list(range(a_bouger + 1, n)) + list(range(0, a_bouger)):
            if roles[j] in fixes or j == a_bouger:
                continue
            # Après l'échange, ni a_bouger ni j ne doivent toucher un plan du même clip.
            ca, cj = clip_de(a_bouger), clip_de(j)
            autour_a = {clip_de(a_bouger - 1), clip_de(a_bouger + 1)} - {None}
            autour_j = {clip_de(j - 1), clip_de(j + 1)} - {None}
            if cj not in autour_a and ca not in autour_j:
                attribution[a_bouger], attribution[j] = attribution[j], attribution[a_bouger]
                break


def _choisir_effets(seg: Segment, prec: Segment | None, rng: random.Random, fps: int) -> None:
    if seg.role == "hook":
        seg.effets = ["flash", "impact", "punch", "rgb"]
        return
    if seg.role == "rewind":
        seg.effets = ["rewind"]
        return
    if prec is None:
        seg.effets = ["fondu_noir", "pulse"]
        return
    if prec.role == "rewind" and seg.role in ("montee", "approche"):
        # Après le rewind : coupe franche avec flash, le « vrai » montage commence.
        seg.effets = ["pulse"] + (["monte"] if seg.role == "approche" else [])
        seg.entree = "flash"
        return
    if seg.role == "drop":
        # Coupe sèche pile sur le kill, au moment du drop.
        seg.effets = ["flash", "impact", "punch", "shake", "rgb"]
        seg.entree = "flash"
        return
    if seg.role == "sequence":
        # Multi-kill en entier : arrivée en whip/zoom, effet sur chaque kill, gros impact sur le dernier.
        t = rng.choice(["whip", "zoom", "secousse"])
        seg.entree = t
        prec.sortie = t if t in SORTIES else None
        seg.effets = ["multi"]
        return
    if seg.role == "approche":
        # Montée en tension : zoom lent + petits coups de zoom sur chaque beat.
        seg.effets = ["monte", "pulse"]
        if prec.role == "montee":
            seg.transition = rng.choice(["fadewhite", "zoomin", "hblur"])
            seg.images_transition = min(int(round(0.25 * fps)), prec.images // 3, seg.images // 3)
            if seg.images_transition < 3:
                seg.transition, seg.images_transition = None, 0
        return
    if seg.role == "montee":
        transitions = [t for t in TRANSITIONS_DOUCES if t != prec.transition]
        seg.transition = rng.choice(transitions)
        td = min(0.35, 0.3 * seg.images / fps, 0.3 * prec.images / fps)
        seg.images_transition = int(round(td * fps))
        if seg.images_transition < 3:
            seg.transition, seg.images_transition = None, 0
        seg.effets = ["pulse"] + (["zoom_leger"] if rng.random() < 0.3 else [])
        return

    # Après le drop : transition coupée sur le beat + effets déclenchés AU MOMENT DU KILL.
    deja = {prec.entree}
    choix = [t for t in TRANSITIONS_COUPEES if t not in deja] or TRANSITIONS_COUPEES
    t = rng.choice(choix)
    if t != "coupe":
        prec.sortie = t if t in SORTIES else None
        seg.entree = t
    if seg.impact:
        seg.effets = ["flash", "impact", "punch", "shake"]
        if seg.entree == "flash":
            seg.entree = "zoom"
            prec.sortie = "zoom"
    else:
        possibles = [e for e in EFFETS_KILL if e not in prec.effets] or EFFETS_KILL
        seg.effets = [rng.choice(possibles)]
        if rng.random() < 0.4:
            autres = [e for e in possibles if e not in seg.effets]
            if autres:
                seg.effets.append(rng.choice(autres))
        # « Velocity » : accéléré jusqu'au kill, ralenti sec sur le kill, puis vitesse normale.
        if rng.random() < 0.45 and "velocity" not in prec.effets:
            seg.effets.append("velocity")


def _appliquer_style(seg: Segment, prec: Segment | None, style: str, rng: random.Random) -> None:
    regles = STYLES.get(style) or {}
    if not regles or seg.role in ("hook", "rewind"):
        return
    interdits = regles.get("effets_interdits", set())
    garder = [e for e in seg.effets if e not in interdits]
    if seg.role in ("drop", "apres", "sequence") and interdits and not any(
            e in garder for e in ("punch", "flash", "pulse", "multi")):
        garder.append("punch")
    seg.effets = garder
    if seg.role in ("drop",) or seg.impact:
        seg.effets += [e for e in regles.get("ajout_kill", []) if e not in seg.effets]
    remplacement = regles.get("entrees", {})
    if seg.entree in remplacement and seg.role != "drop":
        nouveau = remplacement[seg.entree]
        seg.entree = nouveau
        if prec is not None:
            prec.sortie = nouveau if nouveau in SORTIES else None
    if style == "cinematique" and seg.transition in ("pixelize", "radial", "hblur"):
        seg.transition = rng.choice(["fade", "dissolve", "fadewhite"])


def _caler(debut: float, duree_src: float, duree_clip: float) -> float:
    return max(0.0, min(debut, duree_clip - duree_src - 0.05))


def _calculer_morceaux(seg: Segment, periode: float, fps: int, ralenti: float, gel: bool = False) -> None:
    m = seg.moment
    total = seg.images_totales
    lead_in = seg.images_transition / fps
    duree = total / fps
    ralenti = ralenti if 0 < ralenti < 1 else 1.0

    if seg.role == "drop" and gel and total > int(1.2 * fps):
        # Arrêt sur image pile sur le kill (l'ennemi ressort, zoom), puis le ralenti reprend.
        img_gel = min(int(round(0.45 * fps)), total // 3)
        reste = total - img_gel
        img_lent = int(reste * 0.6) if ralenti < 1 else 0
        img_reel = reste - img_lent
        src_lent = img_lent / fps * ralenti
        debut = m.t + 0.02
        if debut + src_lent + img_reel / fps > m.duree_clip - 0.05:
            debut = max(0.0, m.duree_clip - 0.05 - src_lent - img_reel / fps)
        seg.morceaux = [Morceau(m.t + 0.02, 1.0, img_gel, gel=True)]
        if img_lent:
            seg.morceaux.append(Morceau(debut, ralenti, img_lent))
        seg.morceaux.append(Morceau(debut + src_lent, 1.0, img_reel))
        seg.t_kill = 0.0
        seg.effets.append("gel")
        return

    if seg.role == "drop":
        # Coupe pile sur le kill (calé sur le drop), ralenti pendant ~60 % du plan, puis retour à la
        # vitesse normale (speed ramp en S, typique des edits).
        avant = 0.04
        img_lent = int(total * 0.6) if ralenti < 1 else 0
        img_reel = total - img_lent
        src_lent = img_lent / fps * ralenti
        debut = m.t - avant
        if debut + src_lent + img_reel / fps > m.duree_clip - 0.05:
            debut = max(0.0, m.duree_clip - 0.05 - src_lent - img_reel / fps)
        seg.morceaux = [Morceau(debut, ralenti, img_lent)] if img_lent else []
        seg.morceaux.append(Morceau(debut + src_lent, 1.0, img_reel))
        seg.t_kill = (m.t - debut) / (ralenti if img_lent else 1.0)
        return

    if seg.role == "hook":
        img_avant = min(total // 2, int(round(periode * fps)))
        img_lent = total - img_avant
        lent = ralenti if ralenti < 1 else 0.5
        debut = max(0.0, m.t - img_avant / fps)
        seg.morceaux = [Morceau(debut, 1.0, img_avant), Morceau(debut + img_avant / fps, lent, img_lent)]
        seg.t_kill = (m.t - debut) if m.t - debut <= img_avant / fps else img_avant / fps
        return

    if seg.role == "rewind":
        # On repart en arrière depuis le kill du hook, en accéléré.
        fin = min(m.duree_clip - 0.05, m.t + 0.3)
        debut = max(0.0, fin - 2.6)
        seg.morceaux = [Morceau(debut, max(1.0, (fin - debut) / duree), total, inverse=True)]
        seg.t_kill = None
        return

    if seg.role == "sequence":
        kills = m.kills
        m_beats = max(1, seg.nb_beats - 2)
        v = (kills[-1][1] - kills[0][1]) / (m_beats * periode)
        img_queue = int(round(periode * fps))
        img_tete = total - img_queue
        debut = max(0.0, kills[0][1] - periode * v)
        lent = ralenti if ralenti < 1 else 1.0
        fin_src = debut + img_tete / fps * v + img_queue / fps * lent
        if fin_src > m.duree_clip - 0.05:          # pas assez de vidéo après : on recule un peu
            debut = max(0.0, debut - (fin_src - m.duree_clip + 0.05))
        seg.morceaux = [Morceau(debut, v, img_tete), Morceau(debut + img_tete / fps * v, lent, img_queue)]
        textes = []
        for i, (nom, t) in enumerate(kills):
            textes.append(((t - debut) / v, "1" if i == 0 else texte_kill(nom) or nom.upper()))
        seg.kills_sortie = textes
        seg.t_kill = textes[-1][0]
        return

    if seg.role == "approche":
        # Les secondes juste avant le kill du drop : la vidéo s'arrête là où le drop reprend.
        fin = m.t - 0.04
        vitesse = 1.0
        if fin < duree:                       # pas assez d'images avant le kill : on ralentit
            vitesse = max(0.4, fin / duree) if fin > 0.3 else 1.0
        debut = max(0.0, fin - duree * vitesse)
        seg.morceaux = [Morceau(debut, vitesse, total)]
        seg.t_kill = None
        return

    # Le kill tombe sur un beat : au milieu du plan (ou à 35 % s'il n'y a qu'un beat).
    avance = (seg.nb_beats // 2) * periode if seg.nb_beats >= 2 else 0.35 * (duree - lead_in)
    t_kill_sortie = lead_in + avance

    if seg.impact and ralenti < 1 and duree - t_kill_sortie > 0.4:
        # Vitesse normale jusqu'au kill, puis ralenti.
        img_reel = int(round(t_kill_sortie * fps))
        img_lent = total - img_reel
        src_lent = img_lent / fps * ralenti
        t_k = min(m.t, m.duree_clip - 0.05 - src_lent)
        t_k = max(t_k, img_reel / fps)
        seg.morceaux = [Morceau(t_k - img_reel / fps, 1.0, img_reel), Morceau(t_k, ralenti, img_lent)]
        seg.t_kill = img_reel / fps + (m.t - t_k) / ralenti
        return

    if "velocity" in seg.effets and t_kill_sortie >= 0.3 and duree - t_kill_sortie >= 0.45:
        rapide, lent, d_lent = 1.7, 0.35, 0.3
        img_a = int(round(t_kill_sortie * fps))
        img_b = int(round(d_lent * fps))
        img_c = total - img_a - img_b
        debut = m.t - img_a / fps * rapide
        fin = m.t + img_b / fps * lent + img_c / fps
        if debut >= 0 and fin <= m.duree_clip - 0.05 and img_c > 0:
            seg.morceaux = [Morceau(debut, rapide, img_a), Morceau(m.t, lent, img_b),
                            Morceau(m.t + img_b / fps * lent, 1.0, img_c)]
            seg.t_kill = img_a / fps
            return
        seg.effets.remove("velocity")

    vitesse = 1.0
    if m.duree_clip - 0.1 < duree:   # clip trop court : on ralentit pour remplir le plan
        vitesse = max(0.4, (m.duree_clip - 0.1) / duree)
    debut = _caler(m.t - t_kill_sortie * vitesse, duree * vitesse, m.duree_clip)
    seg.morceaux = [Morceau(debut, vitesse, total)]
    seg.t_kill = (m.t - debut) / vitesse
