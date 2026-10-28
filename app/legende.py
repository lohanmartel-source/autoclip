"""Légende + hashtags prêts à coller dans TikTok / YouTube Shorts / YouTube, d'après ce qu'il y a dans le montage."""
from __future__ import annotations

import random
from collections import Counter

HASHTAGS_JEU = {
    "valorant": ["#valorant", "#valorantclips", "#valorantmontage", "#valorantedit"],
    "counterstrike2": ["#cs2", "#counterstrike2", "#cs2clips", "#csgo"],
    "cs2": ["#cs2", "#counterstrike2", "#cs2clips", "#csgo"],
}
HASHTAGS_BASE = ["#fyp", "#gaming", "#clips"]


def _cle(x: str) -> str:
    return "".join(c for c in x.lower() if c.isalnum())


def titres(segments: list, o, combien: int = 3) -> list[str]:
    """Plusieurs titres au choix (la 1re ligne de la légende)."""
    res = []
    for graine in range(12):
        t = legende(segments, o, graine).splitlines()[0] if segments else ""
        sans_emoji = t.rsplit(" ", 1)[0]
        if t and all(r.rsplit(" ", 1)[0] != sans_emoji for r in res):
            res.append(t)
        if len(res) >= combien:
            break
    return res


def legende(segments: list, o, graine: int = 0) -> str:
    moments = [s.moment for s in segments if s.role not in ("hook", "rewind", "approche")]
    if not moments:
        return ""
    rng = random.Random(len(moments) * 31 + int(sum(m.t for m in moments)) + 7777 * graine)
    etiquettes = " ".join(m.etiquette.lower() for m in moments)
    sequence = next((s.moment for s in segments if s.role == "sequence" and s.moment.kills), None)
    nb_kills = sum(len(s.kills_sortie) if s.kills_sortie else (1 if s.t_kill is not None else 0)
                   for s in segments if s.role not in ("hook", "rewind", "approche"))
    agents = Counter(m.agent for m in moments if getattr(m, "agent", ""))
    armes = Counter(m.arme for m in moments if getattr(m, "arme", ""))
    agent = agents.most_common(1)[0][0] if agents else ""
    arme = armes.most_common(1)[0][0] if armes else ""
    jeux = Counter(m.clip.parent.name for m in moments)
    jeu = jeux.most_common(1)[0][0] if jeux else ""

    if sequence is not None:
        duree = sequence.kills[-1][1] - sequence.kills[0][1]
        d = f"{duree:.1f}".replace(".", ",")
        n = len(sequence.kills)
        titre = rng.choice([f"{n}K en {d} s", f"{n} kills en {d} secondes", f"{n}K… en {d} s ?!",
                            f"{d} secondes, {n} morts"])
    elif "ace" in etiquettes:
        titre = rng.choice(["ACE. Rien à ajouter.", "Un ace propre", "ACE 5/5"])
    elif "4k" in etiquettes:
        titre = rng.choice(["4K, il en manquait un", "4K sans trembler"])
    elif "one tap" in etiquettes:
        titre = rng.choice(["Que des one taps", "One tap, next", "Une balle, un kill"])
    else:
        titre = rng.choice([f"{nb_kills} kills, zéro pitié", "Ça clique tout seul", "Session propre"])
    if o.armes and len(o.armes) == 1:
        titre += f" à la {o.armes[0]}"
    elif agent and agents[agent] >= len(moments) * 0.6:
        titre += f" sur {agent}"
    if getattr(o, "mode", "") == "fun":
        titre = rng.choice(["Mes potes en PLS", "J'en pouvais plus", "Le micro a pas survécu"])
    elif getattr(o, "mode", "") == "morts":
        titre = rng.choice(["Mes pires morts de la semaine", "Comment je suis mort (encore)"])
    emoji = rng.choice(["🔥", "😤", "💀", "🎯", "⚡"])

    tags = list(HASHTAGS_JEU.get(_cle(jeu), [f"#{_cle(jeu)}"] if jeu else []))
    for mot, tag in (("ace", "#ace"), ("4k", "#4k"), ("one tap", "#onetap"), ("clutch", "#clutch")):
        if mot in etiquettes:
            tags.append(tag)
    if agent:
        tags.append("#" + _cle(agent))
    if arme:
        tags.append("#" + _cle(arme))
    if o.style == "phonk":
        tags.append("#phonk")
    tags += HASHTAGS_BASE
    if o.format == "youtube":
        tags = [t for t in tags if t != "#fyp"]
    vus, propres = set(), []
    for t in tags:
        if t not in vus and len(t) > 1:
            vus.add(t)
            propres.append(t)

    lignes = [f"{titre} {emoji}", "", " ".join(propres[:10])]
    if o.format == "youtube":
        lignes = [f"{titre} {emoji} | {jeu or 'Montage'}", "",
                  f"{nb_kills} kills, montés automatiquement sur la musique.", "", " ".join(propres[:12])]
    return "\n".join(lignes) + "\n"
