"""Rendu FFmpeg : chaque plan est rendu à part (cadrage 9:16 + effets), puis tout est assemblé
avec les transitions, la musique et le son du jeu."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import subprocess

import numpy as np

from outils import _FLAGS, chemin_filtre, executer, flux_audio, pair
from planification import Segment


@dataclass
class Rendu:
    largeur: int = 1080
    hauteur: int = 1920
    fps: int = 30
    cadrage: str = "flou"      # flou = image complète sur fond flou ; crop = plein écran recadré
    zoom_avant_plan: float = 1.3
    qualite: str = "finale"    # finale ou apercu
    police: Path | None = None  # police des textes de kill (ONE TAP, ACE…)
    textes_kills: bool = True
    style: str = "auto"        # auto / phonk / cinematique / clean (étalonnage des couleurs)
    filigrane: str = ""        # ton pseudo en petit dans un coin (ou logo.png à côté du programme)
    caster: str = ""           # "" = pas de commentateur, "en" ou "fr"
    instrumental: bool = False  # musique sans paroles pendant que tu parles
    gel: bool = False          # arrêt sur image + contour de l'ennemi au drop
    marque: bool = False       # « Fait avec AutoClip » (version gratuite)

    @property
    def base(self) -> int:
        """Taille de référence des textes : le petit côté de l'image (marche en 9:16 et en 16:9)."""
        return min(self.largeur, self.hauteur)

    @property
    def horizontal(self) -> bool:
        return self.largeur > self.hauteur


# Étalonnage par style : (filtre eq, filtre couleur en plus, force de la vignette ou None)
ETALONNAGES = {
    "auto": ("eq=contrast=1.08:saturation=1.22", "", "PI/5.5"),
    "phonk": ("eq=contrast=1.2:saturation=1.38", "colorbalance=rs=0.07:gs=-0.02:bs=-0.03:rm=0.04", "PI/4.2"),
    "cinematique": ("eq=contrast=1.06:saturation=0.92",
                    "colorbalance=rs=-0.05:bs=0.07:rh=0.07:bh=-0.05", "PI/4.8"),
    "clean": ("eq=contrast=1.04:saturation=1.12", "", None),
}


def _atempo(vitesse: float) -> str:
    """atempo n'accepte que 0.5–2 : on enchaîne si besoin."""
    filtres, v = [], vitesse
    while v < 0.5:
        filtres.append("atempo=0.5")
        v /= 0.5
    while v > 2.0:
        filtres.append("atempo=2.0")
        v /= 2.0
    if abs(v - 1) > 1e-3:
        filtres.append(f"atempo={v:.4f}")
    return ",".join(filtres) or "anull"


def _cadrage(r: Rendu, source_verticale: bool) -> str:
    W, H = r.largeur, r.hauteur
    plein = (f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=bicubic,"
             f"crop={W}:{H},setsar=1")
    if r.cadrage == "crop" or source_verticale or r.horizontal:
        return f"[vc]{plein}[vb]"
    if r.cadrage == "ecran_partage":
        # Écran partagé : en haut le jeu en entier, en bas un gros zoom sur le viseur (centre de l'image).
        ht = pair(W * 9 / 16)
        hb = H - ht
        trait = max(2, H // 320)
        return (f"[vc]split=2[ha0][ba0];"
                f"[ha0]scale={W}:{ht}:flags=bicubic,setsar=1[ha];"
                f"[ba0]crop='iw/4':'iw/4*{hb}/{W}':'(iw-ow)/2':'(ih-oh)/2',scale={W}:{hb}:flags=bicubic,setsar=1[ba];"
                f"[ha][ba]vstack,drawbox=x=0:y={ht - trait // 2}:w=iw:h={trait}:color=black:t=fill,setsar=1[vb]")
    bw, bh = pair(W / 4), pair(H / 4)
    fgw = pair(W * r.zoom_avant_plan)
    return (f"[vc]split=2[bg0][fg0];"
            f"[bg0]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
            f"boxblur=8:2,scale={W}:{H},eq=brightness=-0.12:saturation=1.1[bg];"
            f"[fg0]scale={fgw}:-2:flags=bicubic,crop='min(iw,{W})':ih[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[vb]")


def _n(x: float) -> str:
    return f"{x:.3f}"


def _texte_pop(r: Rendu, texte: str, T: float, fin: float, grand: bool = True) -> str:
    """Texte qui « pop » sur le kill : grossit puis se pose, monte légèrement et s'efface."""
    W, H = r.largeur, r.hauteur
    taille = int(r.base * (0.11 if grand else 0.09))
    fondu = min(0.2, max(0.05, (fin - T) / 3))
    return (f"drawtext=fontfile='{chemin_filtre(r.police)}':text='{texte}'"
            f":fontsize='{taille}*(1+0.6*max(0,1-(t-{_n(T)})/0.12))'"
            f":fontcolor=white:borderw={max(3, taille // 14)}:bordercolor=black@0.85"
            f":shadowx=0:shadowy={max(2, taille // 20)}:shadowcolor=black@0.5"
            f":alpha='min(1,max(0,(t-{_n(T)})/0.04))*max(0,min(1,({_n(fin)}-t)/{_n(fondu)}))'"
            f":x=(w-text_w)/2:y='h*0.66-{int(H * 0.02)}*max(0,t-{_n(T)})/0.5'"
            f":enable='between(t,{_n(T)},{_n(fin)})'")


def _effets(seg: Segment, r: Rendu) -> str:
    """Construit la chaîne d'effets d'un plan.

    Tout le mouvement de caméra (zooms, pulses, whip pan, tremblement) passe par UN seul zoompan :
      z(t)  = 1 + somme des zooms,
      kx/ky = décalage horizontal/vertical entre -1 et 1 (en fraction de la marge dispo).
    Les effets « kill » se déclenchent à Tk (moment du kill, sur un beat), les transitions coupées
    au début (entrée) ou à la fin (sortie) du plan.
    """
    W, H, fps = r.largeur, r.hauteur, r.fps
    T0 = seg.images_transition / fps
    D = seg.images_totales / fps
    Tk = seg.t_kill if seg.t_kill is not None else T0
    z, kx, ky, lum, a_rot, g_r, g_b = [], [], [], [], [], [], []
    filtres_avant, filtres_apres = [], []

    def fenetre(nom, debut, fin, *args):
        filtres_apres.append(f"{nom}=" + ":".join(args) + f":enable='between(t,{_n(debut)},{_n(fin)})'")

    # ---- effets sur le kill
    effets = set(seg.effets)
    if "punch" in effets:
        z.append(f"0.30*pow(max(0,1-(it-{_n(Tk)})/0.35),2)*gte(it,{_n(Tk)})")
    if "zoom_leger" in effets:
        z.append(f"0.10*pow(max(0,1-(it-{_n(T0)})/0.5),2)*gte(it,{_n(T0)})")
    if "pulse" in effets:
        for b in seg.beats_locaux(fps):
            if b < D - 0.05:
                z.append(f"0.045*max(0,1-(it-{_n(b)})/0.2)*gte(it,{_n(b)})")
    if "monte" in effets:
        z.append(f"0.16*pow(it/{_n(D)},2)")
    if "shake" in effets:
        env = f"gte(it,{_n(Tk)})*max(0,1-(it-{_n(Tk)})/0.45)"
        z.append(f"0.10*{env}")
        kx.append(f"0.75*{env}*sin(2*PI*(it-{_n(Tk)})*17)")
        ky.append(f"0.75*{env}*cos(2*PI*(it-{_n(Tk)})*13)")
    if "flash" in effets:
        lum.append(f"0.65*max(0,1-(t-{_n(Tk)})/0.2)*gte(t,{_n(Tk)})")
    if "impact" in effets:   # « impact frame » : couleurs inversées 2 images sur le kill
        filtres_apres.append(f"negate=enable='between(t,{_n(Tk)},{_n(Tk + 1.5 / fps)})'")
    if "rgb" in effets:
        fenetre("rgbashift", Tk, Tk + 0.18, "rh=-16", "bh=16", "edge=smear")
    if "gel" in effets and seg.morceaux and seg.morceaux[0].gel:
        # Pendant l'arrêt sur image : zoom lent vers le viseur, puis on relâche.
        gd = seg.morceaux[0].images / fps
        z.append(f"0.22*min(1,(it-{_n(T0)})/{_n(gd)})*lt(it,{_n(T0 + gd)})*gte(it,{_n(T0)})")
        z.append(f"0.22*max(0,1-(it-{_n(T0 + gd)})/0.2)*gte(it,{_n(T0 + gd)})")
    if "rewind" in effets:
        # Look « cassette qui rembobine » : couleurs délavées, lignes, bruit, décalage RGB.
        filtres_apres.append("eq=saturation=0.55:contrast=1.15")
        filtres_apres.append("noise=alls=30:allf=t")
        filtres_apres.append("rgbashift=rh=-8:bh=8:edge=smear")
        filtres_apres.append("drawgrid=w=iw:h=6:t=2:c=black@0.35")
        z.append("0.06")
        if r.police is not None:
            taille = int(r.base * 0.07)
            filtres_apres.append(f"drawtext=fontfile='{chemin_filtre(r.police)}':text='<< REWIND'"
                                 f":fontsize={taille}:fontcolor=white:borderw=3:bordercolor=black"
                                 f":x=w*0.07:y=h*0.08:alpha='0.5+0.5*gt(mod(t,0.3),0.15)'")
    if "multi" in effets and seg.kills_sortie:
        # Chaque kill de la séquence frappe : petit zoom + flash ; le dernier frappe fort.
        temps = [t for t, _ in seg.kills_sortie]
        for t in temps[:-1]:
            z.append(f"0.14*pow(max(0,1-(it-{_n(t)})/0.25),2)*gte(it,{_n(t)})")
            lum.append(f"0.35*max(0,1-(t-{_n(t)})/0.12)*gte(t,{_n(t)})")
            fenetre("rgbashift", t, t + 0.1, "rh=-10", "bh=10", "edge=smear")
        tl = temps[-1]
        z.append(f"0.30*pow(max(0,1-(it-{_n(tl)})/0.35),2)*gte(it,{_n(tl)})")
        lum.append(f"0.65*max(0,1-(t-{_n(tl)})/0.2)*gte(t,{_n(tl)})")
        env = f"gte(it,{_n(tl)})*max(0,1-(it-{_n(tl)})/0.45)"
        z.append(f"0.10*{env}")
        kx.append(f"0.75*{env}*sin(2*PI*(it-{_n(tl)})*17)")
        ky.append(f"0.75*{env}*cos(2*PI*(it-{_n(tl)})*13)")
        filtres_apres.append(f"negate=enable='between(t,{_n(tl)},{_n(tl + 1.5 / fps)})'")

    # ---- transition d'entrée (début du plan, coupe sur le beat)
    e = seg.entree
    if e == "whip":       # arrive de la droite avec un flou de mouvement
        z.append("0.45*max(0,1-it/0.22)")
        kx.append("pow(max(0,1-it/0.18),2)")
        fenetre("avgblur", 0, 0.07, "sizeX=60", "sizeY=1")
        fenetre("avgblur", 0.07, 0.13, "sizeX=22", "sizeY=1")
    elif e == "zoom":     # dézoom rapide depuis un gros plan flou
        z.append("0.7*pow(max(0,1-it/0.2),2)")
        fenetre("gblur", 0, 0.07, "sigma=12")
        fenetre("gblur", 0.07, 0.13, "sigma=5")
    elif e == "spin":     # rotation qui se remet droite
        a_rot.append("0.3*pow(max(0,1-t/0.2),2)")
        z.append("0.6*max(0,1-it/0.22)")
        fenetre("gblur", 0, 0.08, "sigma=6")
    elif e == "glitch":
        fenetre("rgbashift", 0, 0.12, "rh=-28", "gv=10", "bh=28", "edge=wrap")
        fenetre("noise", 0, 0.09, "alls=45", "allf=t")
        filtres_apres.append(f"negate=enable='between(t,0,{_n(1.5 / fps)})'")
    elif e == "flash":
        lum.append("0.8*max(0,1-t/0.15)")
    elif e == "leak":     # « light leak » : éclat chaud orangé qui s'efface
        lum.append("0.35*max(0,1-t/0.3)")
        g_r.append("0.6*max(0,1-t/0.3)")
        g_b.append("0.35*max(0,1-t/0.3)")
    elif e == "secousse":  # coupe avec gros tremblement de caméra
        env = "max(0,1-it/0.3)"
        z.append(f"0.12*{env}")
        kx.append(f"0.9*{env}*sin(2*PI*it*14)")
        ky.append(f"0.9*{env}*cos(2*PI*it*11)")

    # ---- transition de sortie (fin du plan, juste avant la coupe)
    s = seg.sortie
    if s == "whip":       # part vers la gauche
        z.append(f"0.45*max(0,(it-{_n(D - 0.22)})/0.22)")
        kx.append(f"(0-pow(max(0,(it-{_n(D - 0.18)})/0.18),2))")
        fenetre("avgblur", D - 0.12, D, "sizeX=50", "sizeY=1")
    elif s == "zoom":     # zoom avant rapide vers la coupe
        z.append(f"0.7*pow(max(0,(it-{_n(D - 0.18)})/0.18),2)")
        fenetre("gblur", D - 0.1, D, "sigma=10")
    elif s == "spin":
        a_rot.append(f"(0-0.3*pow(max(0,(t-{_n(D - 0.2)})/0.2),2))")
        z.append(f"0.6*max(0,(it-{_n(D - 0.22)})/0.22)")
        fenetre("gblur", D - 0.08, D, "sigma=6")
    elif s == "flash":
        lum.append(f"0.8*max(0,(t-{_n(D - 0.1)})/0.1)")

    f = []
    if a_rot:
        f.append(f"rotate=a='{'+'.join(a_rot)}':fillcolor=black")
    eq, couleur, vignette = ETALONNAGES.get(r.style, ETALONNAGES["auto"])
    if lum:
        eq += f":brightness='min(0.9,{'+'.join(lum)})'"
    if g_r:
        eq += f":gamma_r='1+{'+'.join(g_r)}':gamma_b='max(0.3,1-({'+'.join(g_b)}))'"
    if lum or g_r:
        eq += ":eval=frame"
    f.append(eq)
    if couleur:
        f.append(couleur)
    f.append("unsharp=5:5:0.5")
    if z or kx or ky:
        zz = "1+" + "+".join(z) if z else "1"
        borne = lambda termes: f"max(-1,min(1,{'+'.join(termes)}))" if termes else "0"
        f.append(f"zoompan=z='{zz}':x='(iw/2-iw/zoom/2)*(1+{borne(kx)})'"
                 f":y='(ih/2-ih/zoom/2)*(1+{borne(ky)})':d=1:s={W}x{H}:fps={fps}")
    f += filtres_apres
    if vignette:
        f.append(f"vignette={vignette}")   # bords légèrement assombris
    if r.style == "cinematique" and r.horizontal:
        # Bandes noires « cinémascope » (2.39:1).
        bande = int((H - W / 2.39) / 2)
        if bande > 0:
            f.append(f"drawbox=x=0:y=0:w=iw:h={bande}:color=black:t=fill,"
                     f"drawbox=x=0:y=ih-{bande}:w=iw:h={bande}:color=black:t=fill")
    if r.textes_kills and r.police is not None:
        if seg.kills_sortie:
            # Compteur qui monte : 1 → 2K → 3K → 4K, chaque texte reste jusqu'au kill suivant.
            ks = seg.kills_sortie
            for i, (t, texte) in enumerate(ks):
                fin = ks[i + 1][0] if i + 1 < len(ks) else t + 0.9
                f.append(_texte_pop(r, texte, t, fin, grand=(i == len(ks) - 1)))
        elif seg.texte and seg.t_kill is not None:
            f.append(_texte_pop(r, seg.texte, seg.t_kill, seg.t_kill + 0.9))
    if "fondu_noir" in effets:
        f.append("fade=t=in:st=0:d=0.35")
    if "fondu_fin" in effets:
        f.append(f"fade=t=out:st={max(0, D - 0.6):.3f}:d=0.6")
    return "[vb]" + ",".join(f) + ",setsar=1,format=yuv420p[vout]"


def rendre_segment(seg: Segment, sortie: Path, r: Rendu) -> Path:
    m = seg.moment
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    images_gel = {}
    for i, mo in enumerate(seg.morceaux):
        if mo.gel:
            png = sortie.with_name(f"gel_{seg.index:02d}.png")
            try:
                import arret_image
                ok = arret_image.image_gelee(m.clip, mo.debut, m.largeur, m.hauteur, png)
            except Exception:
                ok = False
            if ok:
                images_gel[i] = png
                cmd += ["-i", str(png)]
                continue
            cmd += ["-ss", f"{mo.debut:.3f}", "-t", "0.3", "-i", str(m.clip)]
            continue
        cmd += ["-ss", f"{mo.debut:.3f}", "-t", f"{mo.duree_source(r.fps) + 0.25:.3f}", "-i", str(m.clip)]

    parties = []
    # Son du jeu SANS le micro (le micro est ajouté à part, au bon moment, par les réactions).
    piste = flux_audio(m.clip)["jeu"] if m.a_audio else None
    for i, mo in enumerate(seg.morceaux):
        d = mo.images / r.fps
        if mo.gel:
            # Une seule image, répétée : arrêt sur image (sans le son du jeu, le « boom » s'en charge).
            parties.append(f"[{i}:v]trim=end_frame=1,setpts=PTS-STARTPTS,loop=loop={max(0, mo.images - 1)}:size=1:start=0,"
                           f"setpts=N/{r.fps}/TB,fps={r.fps},trim=end_frame={mo.images},setsar=1,"
                           f"scale={m.largeur}:{m.hauteur}[v{i}]")
            parties.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={d:.4f}[a{i}]")
            continue
        inverse_v = ",reverse" if mo.inverse else ""
        inverse_a = ",areverse" if mo.inverse else ""
        parties.append(
            f"[{i}:v]setpts=(PTS-STARTPTS)/{mo.vitesse:.4f},fps={r.fps},"
            f"tpad=stop_mode=clone:stop_duration=2,trim=end_frame={mo.images},setpts=PTS-STARTPTS{inverse_v}[v{i}]")
        if m.a_audio:
            src = f"{i}:{piste}" if piste is not None else f"{i}:a"
            parties.append(
                f"[{src}]asetpts=PTS-STARTPTS,{_atempo(mo.vitesse)},aresample=48000,"
                f"aformat=channel_layouts=stereo,apad,atrim=duration={d:.4f},asetpts=PTS-STARTPTS{inverse_a}[a{i}]")
        else:
            parties.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={d:.4f}[a{i}]")

    n = len(seg.morceaux)
    if n == 1:
        parties.append("[v0]null[vc];[a0]anull[ac]")
    else:
        entrees = "".join(f"[v{i}][a{i}]" for i in range(n))
        parties.append(f"{entrees}concat=n={n}:v=1:a=1[vc][ac]")

    verticale = m.hauteur > m.largeur
    parties.append(_cadrage(r, verticale))
    parties.append(_effets(seg, r))

    cmd += ["-filter_complex", ";".join(parties), "-map", "[vout]", "-map", "[ac]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-r", str(r.fps),
            "-c:a", "pcm_s16le", "-ar", "48000", str(sortie)]
    executer(cmd)
    return sortie


def rendre_segments(segments: list[Segment], dossier: Path, r: Rendu, log=print) -> list[Path]:
    fichiers = [dossier / f"plan_{s.index:02d}.mkv" for s in segments]
    travailleurs = max(1, min(4, (os.cpu_count() or 2) // 2))
    fait = 0

    def tache(args):
        return rendre_segment(*args, r)

    with ThreadPoolExecutor(max_workers=travailleurs) as ex:
        for _ in ex.map(tache, zip(segments, fichiers)):
            fait += 1
            log(f"  plan {fait}/{len(segments)} rendu")
    return fichiers


def debuts_plans(segments: list[Segment], fps: int) -> list[float]:
    """Moment où chaque plan commence dans le montage final (en tenant compte des fondus)."""
    debuts, longueur = [], 0.0
    for i, s in enumerate(segments):
        td = s.images_transition / fps if i else 0.0
        debuts.append(longueur - td)
        longueur += s.images_totales / fps - td
    return debuts


def _nom_police(police: Path | None) -> str:
    nom = (police.stem.lower() if police else "")
    if "impact" in nom:
        return "Impact"
    if "arial" in nom:
        return "Arial"
    if "dejavu" in nom:
        return "DejaVu Sans"
    return "Arial"


def _temps_ass(t: float) -> str:
    t = max(0.0, t)
    h, reste = divmod(t, 3600)
    m, sec = divmod(reste, 60)
    return f"{int(h)}:{int(m):02d}:{sec:05.2f}"


def _lignes_sous_titres(mots: list) -> list[list]:
    """Regroupe les mots en petites lignes (3 mots / 16 lettres max), coupées aux silences."""
    lignes, ligne = [], []
    for m in mots:
        texte = m[0]
        longueur = sum(len(x[0]) + 1 for x in ligne) + len(texte)
        if ligne and (len(ligne) >= 3 or longueur > 16 or m[1] - ligne[-1][2] > 0.45):
            lignes.append(ligne)
            ligne = []
        ligne.append(m)
    if ligne:
        lignes.append(ligne)
    return lignes


def _fichier_sous_titres(reactions, r: Rendu, dossier: Path) -> Path | None:
    """Sous-titres façon TikTok/Shorts (fichier .ass) : 2-3 mots à la fois, gros, contour noir épais,
    le mot prononcé s'allume en jaune et grossit (karaoké). Placés sous l'image du jeu."""
    mots = [m for rea in reactions for m in rea.mots]
    if not mots:
        return None
    W, H = r.largeur, r.hauteur
    taille = int(r.base * (0.105 if not r.horizontal else 0.085))
    contour = max(3, taille // 9)
    marge_bas = int(H * (0.2 if not r.horizontal else 0.08))
    en_tete = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Voix,{_nom_police(r.police)},{taille},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,"
        f"100,100,1,0,1,{contour},{max(2, contour // 2)},2,{int(W * 0.06)},{int(W * 0.06)},{marge_bas},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    evenements = []
    lignes = _lignes_sous_titres(mots)
    for n, ligne in enumerate(lignes):
        fin_ligne = ligne[-1][2] + 0.3
        if n + 1 < len(lignes):
            fin_ligne = min(fin_ligne, lignes[n + 1][0][1])
        for i, (texte, debut, _) in enumerate(ligne):
            fin = ligne[i + 1][1] if i + 1 < len(ligne) else fin_ligne
            if fin - debut < 0.05:
                continue
            morceaux = []
            for j, (t2, _, _) in enumerate(ligne):
                t2 = t2.upper().replace("{", "(").replace("}", ")").strip(" ,.")
                if j == i:   # mot prononcé : jaune, un peu plus gros
                    morceaux.append("{\\c&H4DE1FF&\\fscx112\\fscy112}" + t2 + "{\\r}")
                elif j < i:
                    morceaux.append(t2)
                else:        # mots pas encore dits : un peu transparents
                    morceaux.append("{\\alpha&H40&}" + t2 + "{\\r}")
            pop = "{\\fscx125\\fscy125\\t(0,90,\\fscx100\\fscy100)}" if i == 0 else ""
            evenements.append(f"Dialogue: 0,{_temps_ass(debut)},{_temps_ass(fin)},Voix,,0,0,0,,"
                              + pop + " ".join(morceaux))
    f = Path(dossier) / "sous_titres.ass"
    f.write_text(en_tete + "\n".join(evenements) + "\n", encoding="utf-8")
    return f


def _sous_titres(reactions, r: Rendu, dossier: Path) -> list[str]:
    f = _fichier_sous_titres(reactions, r, dossier)
    if f is None:
        return []
    polices = ""
    if r.police is not None:
        polices = f":fontsdir='{chemin_filtre(r.police.parent)}'"
    return [f"ass='{chemin_filtre(f)}'{polices}"]


CHAINE_VOIX = ("adeclip,highpass=f=110,lowpass=f=9500,afftdn=nr=12:nf=-42,"
               "equalizer=f=250:t=q:w=1:g=-3,equalizer=f=3200:t=q:w=1.3:g=5,"
               "speechnorm=e=3:r=0.0005:l=1:p=0.6")
SR_MIX = 48000


def _decoder_stereo(cmd_entree: list[str]) -> np.ndarray:
    """Décode un son en tableau (échantillons, 2) float32 à 48 kHz."""
    proc = subprocess.run(["ffmpeg", "-v", "error"] + cmd_entree + ["-ac", "2", "-ar", str(SR_MIX), "-f", "f32le", "-"],
                          capture_output=True, creationflags=_FLAGS)
    y = np.frombuffer(proc.stdout, np.float32)
    return y[: len(y) // 2 * 2].reshape(-1, 2).copy()


def _ajuster(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.vstack([x, np.zeros((n - len(x), 2), np.float32)])


def _enveloppe(n: int, intervalles: list, montee: float, descente: float) -> np.ndarray:
    """0 en dehors des intervalles, 1 dedans, avec des rampes douces (en secondes)."""
    t = np.arange(n) / SR_MIX
    env = np.zeros(n, np.float32)
    for a, b in intervalles:
        e = np.clip(np.minimum((t - (a - montee)) / montee, ((b + descente) - t) / descente), 0, 1)
        env = np.maximum(env, e)
    return env


def _limiteur(x: np.ndarray, plafond: float = 0.95) -> np.ndarray:
    """Limiteur simple : réduit doucement le volume là où ça dépasse, sans écrêter."""
    crete = np.max(np.abs(x), axis=1)
    fen = int(0.01 * SR_MIX)
    if len(crete) > fen:
        from scipy.ndimage import maximum_filter1d, uniform_filter1d
        crete = uniform_filter1d(maximum_filter1d(crete, size=fen), size=fen)
    gain = np.minimum(1.0, plafond / np.maximum(crete, 1e-9)).astype(np.float32)
    return np.clip(x * gain[:, None], -1, 1)


def mixer_audio(jeu: np.ndarray, musique: np.ndarray, reactions: list, kills: list[float], duree: float,
                son_jeu: float, dossier: Path, effets: list | None = None,
                instru: np.ndarray | None = None) -> np.ndarray:
    """Mélange final fait en Python (fiable et vérifiable) :
    - musique : baisse d'environ 11 dB seulement pendant que tu parles ;
    - jeu : volume de base, remonte sur chaque kill (le coup de feu claque), baisse quand tu parles ;
    - ta voix : nettoyée, mise à un niveau fort et constant, placée au bon moment."""
    n = int(round(duree * SR_MIX))
    jeu, musique = _ajuster(jeu, n), _ajuster(musique, n)
    t = np.arange(n) / SR_MIX
    # Fondus de la musique (entrée 0,05 s, sortie 0,8 s).
    fondu = np.clip(np.minimum(t / 0.05, (duree - t) / 0.8), 0, 1).astype(np.float32)
    voix = np.zeros((n, 2), np.float32)
    parle = []
    for k, rea in enumerate(reactions):
        wav = Path(dossier) / f"voix_{k:02d}.wav"
        y = _decoder_stereo(["-ss", f"{rea.src_debut:.3f}", "-t", f"{rea.duree:.3f}", "-i", str(rea.clip),
                             "-map", f"0:{rea.piste}", "-af", CHAINE_VOIX])
        if not len(y):
            continue
        # Niveau : ta voix à environ -10 dB RMS pendant que tu parles (au-dessus de la musique baissée).
        actif = np.sqrt(np.mean(y ** 2, axis=1))
        seuil = max(0.01, np.percentile(actif, 60) * 0.5)
        rms = float(np.sqrt(np.mean(y[actif > seuil] ** 2))) if (actif > seuil).any() else 1e-3
        y = y * min(20.0, 10 ** (-10 / 20) / max(rms, 1e-4))
        y = _limiteur(y, 0.9)
        m = min(len(y), int(0.03 * SR_MIX))
        y[:m] *= np.linspace(0, 1, m)[:, None]
        m = min(len(y), int(0.15 * SR_MIX))
        y[-m:] *= np.linspace(1, 0, m)[:, None]
        i0 = int(rea.debut * SR_MIX)
        i1 = min(n, i0 + len(y))
        if i1 > i0:
            voix[i0:i1] += y[: i1 - i0]
        parle += list(rea.intervalles or [(rea.debut, rea.debut + rea.duree)])
    env_voix = _enveloppe(n, parle, 0.08, 0.3)
    coups = _enveloppe(n, [(k, k + 0.3) for k in kills if k < duree], 0.06, 0.02)
    hausse = min(1.0, son_jeu + 0.55)
    gain_jeu = (son_jeu + (hausse - son_jeu) * coups * (1 - 0.7 * env_voix)) * (1 - 0.55 * env_voix)
    gain_mus = fondu * (1 - 0.72 * env_voix)
    if instru is not None:
        # Pendant que tu parles, on passe en douceur sur la version sans paroles de la chanson
        # (la voix du chanteur ne couvre plus la tienne), et la musique baisse un peu moins.
        instru = _ajuster(instru, n)
        musique = musique * (1 - env_voix)[:, None] + instru * env_voix[:, None]
        gain_mus = fondu * (1 - 0.5 * env_voix)
    mix = musique * gain_mus[:, None] + jeu * gain_jeu[:, None] + voix
    if effets:
        from effets_sonores import piste_effets
        mix += piste_effets(effets, n) * (1 - 0.6 * env_voix)[:, None]
    return _limiteur(mix)


def evenements_montage(segments: list[Segment], fps: int) -> dict:
    """Temps (dans le montage final) des kills, des coupes et des moments clés."""
    debuts = debuts_plans(segments, fps)
    kills, compteur, sons = [], [], []
    for s, d in zip(segments, debuts):
        temps = [d + t for t in s.temps_kills()]
        kills += temps
        if s.role not in ("hook", "rewind"):
            compteur += temps
        if s.role == "rewind":
            sons.append(("rewind", d, s.images_totales / fps))
        elif s.role == "approche":
            sons.append(("riser", d + s.images_totales / fps, min(3.0, s.images_totales / fps)))
        if (s.role in ("drop", "hook") or s.moment.etiquette in ("Rire", "Cri")) and s.t_kill is not None:
            sons.append(("boom", d + s.t_kill, 0))
        elif s.role == "sequence" and s.kills_sortie:
            sons.append(("boom", d + s.kills_sortie[-1][0], 0))
        elif s.impact and s.t_kill is not None:
            sons.append(("impact", d + s.t_kill, 0))
        if s.entree in ("whip", "zoom", "spin", "secousse"):
            sons.append(("whoosh", d + s.images_transition / fps, 0))
        elif s.entree == "glitch":
            sons.append(("glitch", d + s.images_transition / fps, 0))
    return {"kills": kills, "compteur": sorted(compteur), "sons": sons}


def _filtre_compteur(r: Rendu, temps: list[float], depart: int, duree: float) -> str:
    """« KILLS 14 » en haut à gauche : le chiffre monte et « pop » à chaque kill."""
    if r.police is None or not temps:
        return ""
    taille = int(r.base * 0.055)
    x = int(r.largeur * 0.05)
    y = int(r.hauteur * (0.105 if not r.horizontal else 0.06))
    if r.cadrage == "ecran_partage" and not r.horizontal:
        y = int(r.largeur * 9 / 16 + r.hauteur * 0.015)     # juste sous le jeu, sur le zoom
    filtres = []
    for i, t in enumerate(temps):
        if t >= duree:
            break
        fin = temps[i + 1] if i + 1 < len(temps) else duree
        n = depart + i + 1
        filtres.append(
            f"drawtext=fontfile='{chemin_filtre(r.police)}':text='KILLS {n}'"
            f":fontsize='{taille}*(1+0.35*max(0,1-(t-{_n(t)})/0.15))'"
            f":fontcolor=white:borderw={max(2, taille // 10)}:bordercolor=black@0.9"
            f":shadowx=0:shadowy=3:shadowcolor=black@0.5:x={x}:y={y}"
            f":enable='between(t,{_n(t)},{_n(fin)})'")
    return ",".join(filtres)


def _filtre_outro(r: Rendu, fichier_pseudo: Path, duree: float) -> str:
    """Fin : ton pseudo en gros + « ABONNE-TOI », qui apparaissent pendant la dernière phrase musicale."""
    if r.police is None:
        return ""
    d = min(1.8, duree * 0.25)
    t0 = duree - d
    taille = int(r.base * 0.1)
    petite = int(r.base * 0.045)
    alpha = f"min(1,max(0,(t-{_n(t0)})/0.25))"
    return (f"drawtext=fontfile='{chemin_filtre(r.police)}':textfile='{chemin_filtre(fichier_pseudo)}'"
            f":fontsize='{taille}*(1.25-0.25*min(1,(t-{_n(t0)})/0.25))':fontcolor=white"
            f":borderw={max(3, taille // 12)}:bordercolor=black:alpha='{alpha}'"
            f":x=(w-text_w)/2:y=h*0.42:enable='gte(t,{_n(t0)})',"
            f"drawtext=fontfile='{chemin_filtre(r.police)}':text='ABONNE-TOI'"
            f":fontsize={petite}:fontcolor=#FFE14D:borderw={max(2, petite // 10)}:bordercolor=black"
            f":alpha='{alpha}':x=(w-text_w)/2:y=h*0.42+{int(taille * 1.25)}:enable='gte(t,{_n(t0 + 0.15)})'")


def assembler(segments: list[Segment], fichiers: list[Path], musique: Path, debut_musique: float,
              duree: float, sortie: Path, r: Rendu, son_jeu: float = 0.3, texte: str = "",
              police: Path | None = None, fichier_texte: Path | None = None,
              reactions: list | None = None, dossier_temp: Path | None = None,
              compteur_depart: int | None = None, outro: str = "", effets_sonores: bool = False) -> Path:
    """Étape 1 (FFmpeg) : image + son du jeu bruts. Étape 2 (Python) : mixage musique/jeu/voix.
    Étape 3 (FFmpeg) : on colle le son mixé sur l'image."""
    reactions = reactions or []
    dossier = Path(dossier_temp or sortie.parent)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for fch in fichiers:
        cmd += ["-i", str(fch)]
    n = len(fichiers)

    parties = [f"[{i}:v]setsar=1,settb=AVTB,fps={r.fps}[vi{i}]" for i in range(n)]
    cv, ca = "vi0", "0:a"
    longueur = segments[0].images_totales / r.fps
    evts = evenements_montage(segments, r.fps)
    kills = evts["kills"]
    for i in range(1, n):
        s = segments[i]
        td = s.images_transition / r.fps
        if s.transition and td > 0:
            parties.append(f"[{cv}][vi{i}]xfade=transition={s.transition}:duration={td:.4f}"
                           f":offset={longueur - td:.4f}[x{i}]")
            parties.append(f"[{ca}][{i}:a]acrossfade=d={td:.4f}:c1=tri:c2=tri[xa{i}]")
        else:
            parties.append(f"[{cv}][vi{i}]concat=n=2:v=1:a=0,settb=AVTB,fps={r.fps}[x{i}]")
            parties.append(f"[{ca}][{i}:a]concat=n=2:v=0:a=1[xa{i}]")
        cv, ca = f"x{i}", f"xa{i}"
        longueur += s.images_totales / r.fps - td

    # Texte d'accroche (2,5 premières secondes)
    video_fin = f"[{cv}]trim=duration={duree:.4f},setpts=PTS-STARTPTS"
    if compteur_depart is not None:
        c = _filtre_compteur(r, evts["compteur"], compteur_depart, duree)
        if c:
            video_fin += "," + c
    if outro and r.police is not None:
        f_pseudo = dossier / "outro.txt"
        f_pseudo.write_text(outro, encoding="utf-8")
        video_fin += "," + _filtre_outro(r, f_pseudo, duree)
    if texte and police and fichier_texte:
        taille = int(r.base * 0.075)
        video_fin += (f",drawtext=fontfile='{chemin_filtre(police)}':textfile='{chemin_filtre(fichier_texte)}'"
                      f":fontsize={taille}:fontcolor=white:borderw={max(2, taille // 12)}:bordercolor=black"
                      f":x=(w-text_w)/2:y=h*0.16:enable='lt(t,2.5)'")
    if reactions:
        video_fin += "".join("," + f for f in _sous_titres(reactions, r, dossier))
    if r.marque and r.police is not None:
        # Version gratuite : petite marque en bas, au-dessus de la zone des boutons de TikTok.
        from licence import MARQUE
        f_marque = dossier / "marque.txt"
        f_marque.write_text(MARQUE, encoding="utf-8")
        taille = int(r.base * 0.034)
        y = "h*0.70" if not r.horizontal else "h*0.92"
        video_fin += (f",drawtext=fontfile='{chemin_filtre(r.police)}':textfile='{chemin_filtre(f_marque)}'"
                      f":fontsize={taille}:fontcolor=white@0.75:borderw=2:bordercolor=black@0.5"
                      f":x=(w-text_w)/2:y={y}")
    from chemins import DONNEES as _DONNEES
    logo = _DONNEES / "logo.png"
    if logo.exists():
        # Ton logo en filigrane (coin en haut à droite : en bas à droite, TikTok met ses boutons).
        cmd += ["-i", str(logo)]
        larg = pair(r.base * 0.16)
        parties.append(f"[{n}:v]scale={larg}:-2,format=rgba,colorchannelmixer=aa=0.75[logo]")
        y = int(r.hauteur * (0.095 if not r.horizontal else 0.04))
        parties.append(video_fin + f"[vpre];[vpre][logo]overlay=W-w-{int(r.largeur * 0.05)}:{y},format=yuv420p[vfinal]")
    else:
        if r.filigrane and r.police is not None:
            f_fil = dossier / "filigrane.txt"
            f_fil.write_text(r.filigrane, encoding="utf-8")
            taille = int(r.base * 0.038)
            y = int(r.hauteur * (0.11 if not r.horizontal else 0.06))
            if r.cadrage == "ecran_partage" and not r.horizontal:
                y = int(r.largeur * 9 / 16 + r.hauteur * 0.02)
            video_fin += (f",drawtext=fontfile='{chemin_filtre(r.police)}':textfile='{chemin_filtre(f_fil)}'"
                          f":fontsize={taille}:fontcolor=white@0.6:borderw=2:bordercolor=black@0.4"
                          f":x=w-text_w-{int(r.largeur * 0.05)}:y={y}")
        parties.append(video_fin + ",format=yuv420p[vfinal]")
    parties.append(f"[{ca}]atrim=duration={duree:.4f},asetpts=PTS-STARTPTS[jeu]")

    image = dossier / "image_et_jeu.mkv"
    preset, crf = ("veryfast", "23") if r.qualite == "apercu" else ("medium", "18")
    executer(cmd + ["-filter_complex", ";".join(parties), "-map", "[vfinal]", "-map", "[jeu]",
                    "-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p",
                    "-r", str(r.fps), "-c:a", "pcm_s16le", "-ar", str(SR_MIX), "-t", f"{duree:.4f}", str(image)])

    jeu = _decoder_stereo(["-i", str(image), "-map", "0:a"])
    mus = _decoder_stereo(["-ss", f"{debut_musique:.3f}", "-t", f"{duree + 0.5:.3f}", "-i", str(musique),
                           "-map", "0:a:0"])
    effets = list(evts["sons"]) if effets_sonores else []
    if r.caster:
        try:
            import caster
            effets += caster.evenements(segments, debuts_plans(segments, r.fps), r.caster)
        except Exception:
            pass
    instru = None
    if r.instrumental and reactions:
        try:
            import separation
            f_instru = separation.instrumental(musique)
            if f_instru is not None:
                instru = _decoder_stereo(["-ss", f"{debut_musique:.3f}", "-t", f"{duree + 0.5:.3f}",
                                          "-i", str(f_instru), "-map", "0:a:0"])
        except Exception:
            instru = None
    mix = mixer_audio(jeu, mus, reactions, kills, duree, son_jeu, dossier, effets=effets or None, instru=instru)
    son = dossier / "son_final.wav"
    import soundfile as sf
    sf.write(str(son), mix, SR_MIX, subtype="PCM_16")

    executer(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(image), "-i", str(son),
              "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
              "-movflags", "+faststart", "-t", f"{duree:.4f}", str(sortie)])
    return sortie


def miniature(segments: list[Segment], r: Rendu, sortie: Path, accroche: str = "",
              dossier_temp: Path | None = None, variante: int = 0) -> Path | None:
    """Image de couverture. variante 0 : le meilleur kill + gros texte jaune ; 1 : un autre gros kill,
    texte cyan en haut ; 2 : gros zoom sur le viseur, texte rouge."""
    from planification import texte_kill
    principal = next((s for s in segments if s.role == "sequence"), None) \
        or next((s for s in segments if s.role == "drop"), None) or segments[0]
    choix = principal
    if variante == 1:
        autres = [s for s in segments if s.role == "apres" and s.moment.clip != principal.moment.clip]
        autres.sort(key=lambda s: (s.impact, s.moment.score), reverse=True)
        choix = autres[0] if autres else principal
    m = choix.moment
    t = (m.kills[-1][1] if choix.role == "sequence" and m.kills else m.t) + 0.06
    texte = (texte_kill(m.etiquette) or ("4K" if len(m.kills) >= 4 else "")) if m.etiquette else ""
    if choix.role == "sequence" and m.kills:
        texte = texte or f"{len(m.kills)}K"
    if variante == 1 and not texte:
        texte = (getattr(m, "agent", "") or "CLEAN").upper()
    if variante == 2:
        texte = texte or "1 BALLE"
    W, H = (1280, 720) if r.horizontal else (1080, 1920)
    rr = Rendu(largeur=W, hauteur=H, fps=r.fps, cadrage=r.cadrage, police=r.police)
    entree = "[0:v]null[vc]"
    if variante == 2:   # zoom x2 sur le viseur
        entree = "[0:v]crop='iw/2':'ih/2':'iw/4':'ih/4',scale=iw*2:ih*2[vc]"
    couleur, y = {0: ("#FFE14D", "h*0.60"), 1: ("#25F4EE", "h*0.20"), 2: ("#FF3D6E", "h*0.64")}[variante % 3]
    filtres = [entree, _cadrage(rr, m.hauteur > m.largeur),
               "[vb]eq=contrast=1.12:saturation=1.35,unsharp=5:5:0.8,vignette=PI/4.5"]
    if r.police is not None and texte:
        taille = int(min(W, H) * (0.26 if variante != 1 else 0.2))
        filtres[-1] += (f",drawtext=fontfile='{chemin_filtre(r.police)}':text='{texte}':fontsize={taille}"
                        f":fontcolor={couleur}:borderw={taille // 12}:bordercolor=black"
                        f":shadowx=0:shadowy={taille // 16}:shadowcolor=black@0.6:x=(w-text_w)/2:y={y}")
    if r.police is not None and accroche and dossier_temp is not None and variante == 0:
        f = dossier_temp / "miniature_texte.txt"
        f.write_text(accroche.upper(), encoding="utf-8")
        taille = int(min(W, H) * 0.085)
        filtres[-1] += (f",drawtext=fontfile='{chemin_filtre(r.police)}':textfile='{chemin_filtre(f)}'"
                        f":fontsize={taille}:fontcolor=white:borderw={taille // 10}:bordercolor=black"
                        f":x=(w-text_w)/2:y=h*0.13")
    filtres[-1] += "[vout]"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(m.clip),
           "-filter_complex", ";".join(filtres), "-map", "[vout]", "-frames:v", "1", "-q:v", "2", str(sortie)]
    try:
        executer(cmd)
        return sortie
    except Exception:
        return None


def miniatures(segments: list[Segment], r: Rendu, video: Path, accroche: str = "",
               dossier_temp: Path | None = None) -> list[Path]:
    """3 miniatures au choix : video_mini1.jpg, _mini2.jpg, _mini3.jpg ; la 1re est aussi video.jpg."""
    res = []
    for v in range(3):
        f = video.with_name(f"{video.stem}_mini{v + 1}.jpg")
        if miniature(segments, r, f, accroche, dossier_temp, v):
            res.append(f)
    if res:
        import shutil
        shutil.copy2(res[0], video.with_suffix(".jpg"))
    return res
