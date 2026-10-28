"""Serveur local d'AutoClip : la fenêtre de l'app (app.py) affiche la page web/index.html
et lui parle par cette petite API. Il n'écoute que sur 127.0.0.1 : rien n'est ouvert sur le réseau.
"""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import re
import secrets
import sys
import threading
import time
import webbrowser
import os
import subprocess
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from analyse_clips import FILTRES
import cartes
import copie_style
import defi
import equipe
import morts
import sessions
import index_kills
import medal_dossier
from autoclip import Options, generer, jeux_disponibles
from outils import ErreurAutoclip
from chemins import DONNEES, RESSOURCES, VERSION
import liens
import licence

CONFIG = DONNEES / "serveur.json"
PAGE = RESSOURCES / "web" / "index.html"
EXT_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".mp4", ".webm"}
TAILLE_MAX_MUSIQUE = 80 * 1024 * 1024


# ---------------------------------------------------------------- configuration
def charger_config() -> dict:
    defaut = {
        "cle": secrets.token_urlsafe(12),
        "clips": "",
        "dossier_sortie": str(Path.home() / "Videos" / "AutoClip"),
        "dossier_musiques": str(Path.home() / "Music" / "AutoClip"),
        "reglages": {},
    }
    try:
        defaut.update(json.loads(CONFIG.read_text(encoding="utf-8")))
    except Exception:
        pass
    if not defaut["clips"] or not Path(defaut["clips"]).is_dir():
        try:
            defaut["clips"] = medal_dossier.trouver()
        except Exception:
            pass
    sauver_config(defaut)
    return defaut


def sauver_config(c: dict) -> None:
    try:
        CONFIG.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


_capacites: dict | None = None


def capacites() -> dict:
    """Fonctions optionnelles disponibles (dépendent des librairies livrées)."""
    global _capacites
    if _capacites is None:
        import importlib.util as u
        _capacites = {"sous_titres": u.find_spec("faster_whisper") is not None,
                      "caster": u.find_spec("edge_tts") is not None,
                      "instrumental": u.find_spec("demucs") is not None}
    return _capacites


def choisir_dossier_tk(depart: str = "") -> str:
    """Fenêtre « Choisir un dossier » de Windows (remplacée par celle de la fenêtre de l'app si dispo)."""
    import tkinter as tk
    from tkinter import filedialog
    racine = tk.Tk()
    racine.withdraw()
    racine.attributes("-topmost", True)
    try:
        return filedialog.askdirectory(initialdir=depart or str(Path.home()), title="Dossier de tes clips Medal") or ""
    finally:
        racine.destroy()


def nom_sur(nom: str) -> str:
    """Nom de fichier sans caractères dangereux (pas de ../ ni de chemins)."""
    nom = Path(unquote(nom)).name
    nom = re.sub(r"[^\w\-. ()\[\]]", "_", nom, flags=re.UNICODE).strip(" .")
    return nom[:120] or "fichier"


# ---------------------------------------------------------------- tâche de rendu
class Surveillant:
    """En arrière-plan : analyse les nouveaux clips (index des kills) et, si activé, fait tout seul
    le montage de ta session de jeu quand elle est finie (45 min sans nouveau clip)."""

    def __init__(self, config: dict, tache: "Tache"):
        self.config, self.tache = config, tache
        self.arret = threading.Event()
        self.en_cours = False
        self.message = ""
        self.reveil = threading.Event()

    def demarrer(self):
        threading.Thread(target=self._boucle, daemon=True).start()

    def _log(self, msg: str):
        self.message = msg.strip()

    def _boucle(self):
        time.sleep(5)
        while not self.arret.is_set():
            try:
                if self.config.get("clips") and not self.tache.etat()["en_cours"]:
                    self.en_cours = True
                    # Par petits paquets : un montage lancé depuis l'app passe toujours en premier.
                    n = index_kills.indexer(self.config["clips"], log=self._log, arret=self.arret, max_clips=8)
                    self.en_cours = False
                    if n:
                        continue
                    # Puis tes morts (coach), aussi par petits paquets.
                    self.en_cours = True
                    n = morts.indexer(self.config["clips"], log=self._log, arret=self.arret, max_clips=6)
                    self.en_cours = False
                    if n:
                        continue
                    self._session_auto()
                self._defi()
            except Exception as e:
                self.message = f"index : {type(e).__name__} : {e}"
            finally:
                self.en_cours = False
            self.reveil.wait(120)
            self.reveil.clear()

    def _defi(self):
        """Publie tes stats de la semaine pour le défi entre amis (au plus une fois par heure)."""
        d = self.config.get("defi") or {}
        if not d.get("code") or not self.config.get("pseudo"):
            return
        if time.time() - float(d.get("publie") or 0) < 3600:
            if time.time() - float(d.get("recu") or 0) > 900:     # clips des amis : toutes les 15 min
                d["recu"] = time.time()
                try:
                    equipe.recevoir(self.config, self._log)
                except Exception:
                    pass
            return
        s = defi.mes_scores(index_kills.charger_index(), self.config["pseudo"],
                            (self.config.get("reglages") or {}).get("jeux"))
        if defi.publier(d["code"], s):
            d["publie"] = time.time()
            self.config["defi"] = d
            # Montage d'équipe : envoie tes meilleurs kills (si activé) et récupère ceux de tes amis.
            try:
                equipe.partager(self.config, index_kills.charger_index(), self._log)
                equipe.recevoir(self.config, self._log)
            except Exception as e:
                self.message = f"équipe : {e}"
            sauver_config(self.config)

    def _session_auto(self):
        if not self.config.get("auto_session"):
            return
        s = sessions.derniere_session(index_kills.charger_index(), (self.config.get("reglages") or {}).get("jeux"))
        if s is None:
            return
        debut, fin, kills = s
        if time.time() - fin < 45 * 60 or fin <= float(self.config.get("auto_session_faite") or 0) or kills < 8:
            return
        if self.tache.lancer(lambda log: sessions.montage_session(log, debut, fin)):
            self.config["auto_session_faite"] = fin
            sauver_config(self.config)

    def etat(self) -> dict:
        index = index_kills.charger_index()
        return {"en_cours": self.en_cours, "message": self.message, "clips": len(index["clips"])}


class Tache:
    """Un seul montage à la fois ; l'état est lu par le téléphone toutes les secondes."""

    def __init__(self):
        self.verrou = threading.Lock()
        self.en_cours = False
        self.journal: list[str] = []
        self.progression = 0.0
        self.etape = ""
        self.resultat: str | None = None
        self.erreur: str | None = None
        self.debut = 0.0
        self.lien: str | None = None     # lien YouTube après une publication

    def etat(self) -> dict:
        with self.verrou:
            return {"en_cours": self.en_cours, "journal": self.journal[-60:], "progression": self.progression,
                    "etape": self.etape, "resultat": self.resultat, "erreur": self.erreur, "lien": self.lien,
                    "secondes": int(time.time() - self.debut) if self.debut else 0}

    def _log(self, msg: str):
        with self.verrou:
            self.journal.append(msg)
            m = re.match(r"=== (Version|Chapitre) (\d+)/(\d+)", msg)
            if m:
                self.etape = f"{m.group(1)} {m.group(2)}/{m.group(3)}"
            m = re.match(r"(\d)/4\s+(.*)", msg)
            if m:
                n = int(m.group(1))
                self.progression = {1: 0.02, 2: 0.05, 3: 0.30, 4: 0.35}[n]
                self.etape = m.group(2).rstrip("…")
            m = re.search(r"\[(\d+)/(\d+)\] analysé", msg)
            if m:
                self.progression = 0.05 + 0.25 * int(m.group(1)) / int(m.group(2))
            m = re.search(r"plan (\d+)/(\d+) rendu", msg)
            if m:
                self.progression = 0.35 + 0.50 * int(m.group(1)) / int(m.group(2))
                self.etape = f"Rendu des plans ({m.group(1)}/{m.group(2)})"
            if "assemblage final" in msg:
                self.progression, self.etape = 0.88, "Assemblage final"

    def lancer(self, travail) -> bool:
        """travail : Options d'un montage, ou une fonction(log) qui renvoie la liste des vidéos."""
        with self.verrou:
            if self.en_cours:
                return False
            self.en_cours, self.journal, self.progression = True, [], 0.0
            self.etape, self.resultat, self.erreur, self.debut = "Démarrage", None, None, time.time()
            self.lien = None
        threading.Thread(target=self._travail, args=(travail,), daemon=True).start()
        return True

    def _travail(self, travail):
        try:
            if isinstance(travail, Options):
                videos = generer(travail, log=self._log)
            else:
                videos = travail(self._log)
            if not videos:
                raise ErreurAutoclip("Aucune vidéo produite (voir les détails).")
            with self.verrou:
                self.resultat, self.progression, self.etape = Path(videos[0]).name, 1.0, "Terminé"
        except ErreurAutoclip as e:
            with self.verrou:
                self.erreur = str(e)
        except Exception as e:
            with self.verrou:
                self.erreur = f"{type(e).__name__} : {e}"
        finally:
            with self.verrou:
                self.en_cours = False


# ---------------------------------------------------------------- serveur HTTP
class Gestionnaire(BaseHTTPRequestHandler):
    server_version = "AutoClip"
    config: dict = {}
    choisir_dossier = staticmethod(choisir_dossier_tk)   # app.py met celle de pywebview
    tache: Tache = None  # type: ignore
    surveillant: Surveillant = None  # type: ignore

    def log_message(self, fmt, *args):  # journal HTTP silencieux
        pass

    # -- utilitaires
    def _local(self) -> bool:
        # Seulement ce PC, et l'adresse demandée doit être 127.0.0.1 (protège du « DNS rebinding »).
        hote = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return self.client_address[0] in ("127.0.0.1", "::1") and hote in ("127.0.0.1", "localhost")

    def _autorise(self) -> bool:
        if not self._local():
            return False
        cle = self.config["cle"]
        q = parse_qs(urlparse(self.path).query)
        if q.get("cle", [""])[0] == cle or self.headers.get("X-Cle") == cle:
            return True
        c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        return "autoclip" in c and secrets.compare_digest(c["autoclip"].value, cle)

    def _json(self, data, code=200, extra_headers=None):
        corps = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corps)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(corps)

    def _texte(self, contenu: str, type_="text/html; charset=utf-8", code=200, extra_headers=None):
        corps = contenu.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", type_)
        self.send_header("Content-Length", str(len(corps)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(corps)

    def _lire_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 100_000:
            raise ValueError("requête trop grosse")
        return json.loads(self.rfile.read(n) or b"{}")

    def _refuser(self):
        self._json({"erreur": "Accès refusé. Ferme puis rouvre AutoClip."}, 401)

    # -- routes
    def do_GET(self):
        url = urlparse(self.path)
        chemin = url.path

        if chemin in ("/", "/index.html"):
            # La clé passée par la fenêtre de l'app est mémorisée dans un cookie, puis retirée de l'adresse.
            if parse_qs(url.query).get("cle", [""])[0] == self.config["cle"]:
                self.send_response(303)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie", f"autoclip={self.config['cle']}; Path=/; Max-Age=31536000; SameSite=Strict; HttpOnly")
                self.end_headers()
                return
            if not self._autorise():
                return self._texte(PAGE_REFUS, code=401)
            return self._texte(PAGE.read_text(encoding="utf-8"))
        if chemin == "/manifest.json":
            return self._texte(MANIFEST, "application/manifest+json")
        if chemin == "/icone.svg":
            return self._texte(ICONE, "image/svg+xml")

        if not self._autorise():
            return self._refuser()
        if chemin == "/api/etat":
            return self._json(self._etat_complet())
        if chemin.startswith("/videos/"):
            return self._fichier(Path(self.config["dossier_sortie"]), chemin[len("/videos/"):],
                                 telecharger="dl" in parse_qs(url.query))
        if chemin.startswith("/musiques/"):
            return self._fichier(Path(self.config["dossier_musiques"]), chemin[len("/musiques/"):])
        if chemin.startswith("/vignettes/"):
            return self._fichier(index_kills.VIGNETTES, chemin[len("/vignettes/"):])
        q = parse_qs(url.query)
        if chemin == "/api/stats":
            return self._json(index_kills.statistiques(index_kills.charger_index(), self._jeux_reglages(),
                                                       int(q.get("jours", ["30"])[0] or 30)))
        if chemin == "/api/kills":
            return self._kills(q)
        if chemin == "/api/filtres":
            return self._filtres()
        if chemin == "/api/menage":
            return self._menage_liste()
        if chemin == "/api/menage/vignette":
            return self._menage_vignette(q.get("nom", [""])[0])
        if chemin == "/api/coach":
            depuis = time.time() - float(q.get("jours", ["30"])[0] or 30) * 86400
            return self._json(morts.resume(depuis))
        if chemin == "/api/styles":
            return self._json({"profils": copie_style.profils()})
        if chemin == "/api/matchs":
            ms = cartes.matchs_recents(index_kills.charger_index())
            return self._json({"matchs": [{k: v for k, v in m.items() if k != "fond"} for m in ms]})
        if chemin == "/api/carte":
            return self._carte(q.get("match", [""])[0], "dl" in q)
        if chemin == "/api/defi":
            return self._defi_etat()
        if chemin == "/api/legende":
            f = Path(self.config["dossier_sortie"]) / nom_sur(q.get("nom", [""])[0])
            t = f.with_suffix(".txt")
            return self._json({"texte": t.read_text(encoding="utf-8") if t.is_file() else ""})
        self._json({"erreur": "introuvable"}, 404)

    def do_POST(self):
        chemin = urlparse(self.path).path
        if not self._autorise():
            return self._refuser()
        try:
            if chemin == "/api/generer":
                return self._generer(self._lire_json())
            if chemin == "/api/style":
                return self._recevoir_style()
            if chemin == "/api/style/supprimer":
                d = self._lire_json()
                f = copie_style.DOSSIER / (copie_style._slug(str(d.get("id", ""))) + ".json")
                f.unlink(missing_ok=True)
                return self._json({"ok": True})
            if chemin == "/api/choix":
                return self._choix(self._lire_json())
            if chemin == "/api/musique":
                return self._recevoir_musique()
            if chemin == "/api/config":
                return self._maj_config(self._lire_json())
            if chemin == "/api/supprimer":
                return self._supprimer(self._lire_json())
            if chemin == "/api/pro":
                d = self._lire_json()
                if d.get("action") == "desactiver":
                    licence.desactiver()
                    return self._json({"ok": True, "pro": licence.etat()})
                try:
                    return self._json({"ok": True, "pro": licence.activer(str(d.get("code", "")))})
                except ValueError as e:
                    return self._json({"erreur": str(e)}, 400)
            if chemin == "/api/bestof":
                if not self.tache.lancer(lambda log: sessions.best_of(log=log)):
                    return self._json({"erreur": "Un montage est déjà en cours."}, 409)
                return self._json({"ok": True})
            if chemin == "/api/style/importer":
                return self._importer_styles()
            if chemin == "/api/dossier":
                return self._choisir_dossier()
            if chemin == "/api/detecter":
                trouve = medal_dossier.trouver()
                if not trouve:
                    return self._json({"erreur": "Dossier Medal introuvable. Choisis-le à la main."}, 404)
                self.config["clips"] = trouve
                sauver_config(self.config)
                return self._json({"ok": True, "clips": trouve})
            if chemin == "/api/ouvrir":
                return self._ouvrir(self._lire_json())
            if chemin == "/api/lien":
                url = liens.LIENS.get(str(self._lire_json().get("quoi", "")))
                if not url:
                    return self._json({"erreur": "Lien pas encore configuré."}, 400)
                webbrowser.open(url)
                return self._json({"ok": True})
            if chemin == "/api/defi":
                return self._defi_action(self._lire_json())
            if chemin == "/api/finaliser":
                return self._finaliser(self._lire_json())
            if chemin == "/api/menage/corbeille":
                return self._menage_corbeille(self._lire_json())
            if chemin == "/api/session":
                if not self.tache.lancer(lambda log: sessions.montage_session(log)):
                    return self._json({"erreur": "Un montage est déjà en cours."}, 409)
                return self._json({"ok": True})
        except (ValueError, json.JSONDecodeError) as e:
            return self._json({"erreur": str(e)}, 400)
        self._json({"erreur": "introuvable"}, 404)

    # -- actions
    _cache_jeux: tuple[float, str, dict] = (0.0, "", {})

    def _jeux(self) -> dict:
        """Jeux trouvés dans le dossier Medal (mis en cache 60 s)."""
        quand, dossier, jeux = Gestionnaire._cache_jeux
        if dossier != self.config["clips"] or time.time() - quand > 60:
            try:
                jeux = jeux_disponibles([self.config["clips"]]) if self.config["clips"] else {}
            except Exception:
                jeux = {}
            Gestionnaire._cache_jeux = (time.time(), self.config["clips"], jeux)
        return jeux

    def _etat_complet(self) -> dict:
        sortie = Path(self.config["dossier_sortie"])
        musiques = Path(self.config["dossier_musiques"])
        videos = sorted((f for f in sortie.glob("*.mp4") if not f.stem.endswith("_notif")),
                        key=lambda f: f.stat().st_mtime, reverse=True) if sortie.is_dir() else []
        sons = sorted([f for f in musiques.iterdir() if f.suffix.lower() in EXT_AUDIO],
                      key=lambda f: f.stat().st_mtime, reverse=True) if musiques.is_dir() else []
        return {
            "tache": self.tache.etat(),
            "clips": self.config["clips"],
            "clips_ok": bool(self.config["clips"]) and Path(self.config["clips"]).expanduser().is_dir(),
            "dossier_sortie": str(sortie),
            "musiques": [f.name for f in sons],
            "videos": [{"nom": f.name, "taille": f.stat().st_size, "date": int(f.stat().st_mtime),
                        "miniature": f.with_suffix(".jpg").name if f.with_suffix(".jpg").exists() else None}
                       for f in videos[:30]],
            "infos_videos": self._infos_videos(videos[:30]),
            "defi": {"code": (self.config.get("defi") or {}).get("code", "")},
            "reglages": self.config.get("reglages", {}),
            "pseudo": self.config.get("pseudo", ""),
            "auto_session": bool(self.config.get("auto_session")),
            "version": VERSION,
            "liens": {k: bool(v) for k, v in liens.LIENS.items()},
            "capacites": capacites(),
            "pro": licence.etat(),
            "mise_a_jour": liens.mise_a_jour_dispo(),
            "index": self.surveillant.etat() if self.surveillant else {},
            "jeux": [{"nom": j, "clips": n, "pro": not licence.est_jeu_gratuit(j)} for j, n in self._jeux().items()],
        }

    def _infos_videos(self, videos: list[Path]) -> dict:
        res = {}
        for f in videos:
            info = {}
            try:
                info = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
            except Exception:
                pass
            minis = [m.name for m in (f.with_name(f"{f.stem}_mini{i}.jpg") for i in (1, 2, 3)) if m.exists()]
            res[f.name] = {"legende": f.with_suffix(".txt").exists(), "note": info.get("note"),
                           "finalisable": "_apercu" in f.stem and bool(info), "meilleure": False,
                           "titres": info.get("titres") or [], "miniatures": minis}
        # Parmi les versions d'un même montage (_v1, _v2, _v3), la mieux notée.
        groupes: dict[str, list[str]] = {}
        for nom in res:
            m = re.match(r"(.+)_v\d+(_final)?\.mp4$", nom)
            if m:
                groupes.setdefault(m.group(1), []).append(nom)
        for noms in groupes.values():
            notes = [(res[n]["note"] or 0, n) for n in noms]
            if len(noms) > 1 and max(notes)[0] > 0:
                res[max(notes)[1]]["meilleure"] = True
        return res

    def _jeux_reglages(self):
        return (self.config.get("reglages") or {}).get("jeux") or None

    def _kills(self, q: dict):
        index = index_kills.charger_index()
        jours = float(q.get("jours", ["7"])[0] or 7)
        depuis = time.time() - jours * 86400 if jours > 0 else None
        kills = index_kills.kills_plats(index, self._jeux_reglages(), depuis)
        kills = [k for k in kills if index_kills.est_kill(k["nom"])]
        champs = ("id", "clip", "t", "nom", "one_tap", "arme", "agent", "date", "vignette", "jeu")
        return self._json({"kills": [{c: k.get(c) for c in champs} for k in kills[:400]],
                           "total": len(kills), "clips_analyses": len(index["clips"])})

    def _filtres(self):
        from collections import Counter
        index = index_kills.charger_index()
        armes, agents = Counter(), Counter()
        for c in index["clips"].values():
            for k in c["kills"]:
                if k.get("arme"):
                    armes[k["arme"]] += 1
            if c.get("agent"):
                agents[c["agent"]] += 1
        return self._json({"armes": armes.most_common(), "agents": agents.most_common(),
                           "clips_analyses": len(index["clips"])})

    _menage_cache: tuple[float, list] = (0.0, [])

    def _menage_liste(self):
        if not self.config.get("clips"):
            return self._json({"clips": [], "taille": 0})
        liste = index_kills.clips_sans_kill(self.config["clips"])
        Gestionnaire._menage_cache = (time.time(), liste)
        return self._json({"clips": [{k: c[k] for k in ("nom", "jeu", "taille", "date")} for c in liste],
                           "taille": sum(c["taille"] for c in liste)})

    def _menage_vignette(self, nom: str):
        _, liste = Gestionnaire._menage_cache
        c = next((c for c in liste if c["nom"] == nom), None)
        if c is None:
            return self._json({"erreur": "introuvable"}, 404)
        v = index_kills.vignette_menage(Path(c["chemin"]))
        if v is None:
            return self._json({"erreur": "introuvable"}, 404)
        return self._fichier(index_kills.VIGNETTES, v.name)

    def _menage_corbeille(self, d: dict):
        """Rien n'est supprimé sans ton accord : il faut cocher les clips ET confirmer.
        Les clips vont dans la Corbeille de Windows (récupérables)."""
        if d.get("confirmation") != "OUI":
            return self._json({"erreur": "Confirmation manquante."}, 400)
        noms = {str(n) for n in (d.get("noms") or [])}
        # Seulement des clips de la liste « sans kill » (jamais un fichier arbitraire).
        liste = index_kills.clips_sans_kill(self.config["clips"]) if self.config.get("clips") else []
        cibles = [Path(c["chemin"]) for c in liste if c["nom"] in noms]
        if not cibles:
            return self._json({"erreur": "Aucun clip sélectionné."}, 400)
        ok, erreurs = index_kills.mettre_a_la_corbeille(cibles)
        for p in cibles:
            (index_kills.VIGNETTES / f"{p.stem}_menage.jpg").unlink(missing_ok=True)
        return self._json({"ok": ok, "erreurs": erreurs})

    def _carte(self, match: str, telecharger: bool):
        partie = next((m for m in cartes.matchs_recents(index_kills.charger_index(), 50) if m["match"] == match), None)
        if partie is None:
            return self._json({"erreur": "Partie introuvable."}, 404)
        from autoclip import trouver_police
        f = cartes.faire_carte(partie, self.config.get("pseudo", ""), trouver_police())
        if f is None:
            return self._json({"erreur": "Impossible de faire la carte."}, 500)
        return self._fichier(cartes.DOSSIER, f.name, telecharger=telecharger)

    def _defi_etat(self):
        d = self.config.get("defi") or {}
        if not d.get("code"):
            return self._json({"code": "", "classement": []})
        connus = defi.lire(d["code"], d.get("scores"))
        if self.config.get("pseudo"):
            connus[self.config["pseudo"]] = defi.mes_scores(index_kills.charger_index(), self.config["pseudo"],
                                                           self._jeux_reglages())
        d["scores"] = connus
        self.config["defi"] = d
        sauver_config(self.config)
        semaine, _ = defi.semaine_actuelle()
        return self._json({"code": d["code"], "semaine": semaine, "moi": self.config.get("pseudo", ""),
                           "classement": defi.classement(connus), "partager_clips": bool(d.get("partager_clips")),
                           "clips_amis": len(equipe.clips_amis())})

    def _defi_action(self, d: dict):
        action = d.get("action")
        if action in ("creer", "rejoindre"):
            if not self.config.get("pseudo"):
                return self._json({"erreur": "Écris d'abord ton pseudo (Réglages du PC)."}, 400)
            code = defi.nouveau_code() if action == "creer" else str(d.get("code", "")).strip().lower()
            if not defi.code_valide(code):
                return self._json({"erreur": "Code invalide (il commence par defi-)."}, 400)
            self.config["defi"] = {"code": code, "scores": {}, "publie": 0}
            sauver_config(self.config)
            self.surveillant.reveil.set()          # publie tes stats tout de suite
            return self._json({"ok": True, "code": code})
        if action == "partager_clips":
            d0 = self.config.get("defi") or {}
            if not d0.get("code"):
                return self._json({"erreur": "Rejoins d'abord un défi."}, 400)
            d0["partager_clips"] = bool(d.get("actif"))
            d0["publie"] = 0                        # envoie au prochain passage
            self.config["defi"] = d0
            sauver_config(self.config)
            self.surveillant.reveil.set()
            return self._json({"ok": True})
        if action == "quitter":
            self.config.pop("defi", None)
            sauver_config(self.config)
            return self._json({"ok": True})
        return self._json({"erreur": "action inconnue"}, 400)

    def _choisir_dossier(self):
        try:
            choisi = Gestionnaire.choisir_dossier(self.config.get("clips", ""))
        except Exception as e:
            return self._json({"erreur": f"Fenêtre de choix impossible : {e}"}, 500)
        if not choisi:
            return self._json({"ok": False})
        self.config["clips"] = str(Path(choisi))
        sauver_config(self.config)
        Gestionnaire._cache_jeux = (0.0, "", {})
        self.surveillant.reveil.set()
        return self._json({"ok": True, "clips": self.config["clips"]})

    def _ouvrir(self, d: dict):
        """Montre une vidéo dans l'Explorateur (ou ouvre le dossier des montages)."""
        sortie = Path(self.config["dossier_sortie"])
        sortie.mkdir(parents=True, exist_ok=True)
        cible = sortie / nom_sur(str(d["nom"])) if d.get("nom") else sortie
        if not cible.exists():
            return self._json({"erreur": "Introuvable."}, 404)
        try:
            if sys.platform == "win32":
                if cible.is_file():
                    subprocess.Popen(["explorer", "/select,", str(cible)])
                else:
                    os.startfile(str(cible))  # type: ignore[attr-defined]
            else:
                webbrowser.open(cible.parent.as_uri() if cible.is_file() else cible.as_uri())
        except Exception as e:
            return self._json({"erreur": str(e)}, 500)
        return self._json({"ok": True})

    def _importer_styles(self):
        """Pack de styles (.json) : un style, ou {"styles": [...]} (ex. les styles des membres Patreon)."""
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 500_000:
            return self._json({"erreur": "Fichier vide ou trop gros."}, 400)
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8-sig"))
        except Exception:
            return self._json({"erreur": "Ce fichier n'est pas un pack de styles AutoClip."}, 400)
        liste = data.get("styles") if isinstance(data, dict) and "styles" in data else [data]
        pack = str(data.get("pack") or "")[:30] if isinstance(data, dict) else ""
        ajoutes = []
        for st in liste if isinstance(liste, list) else []:
            if not isinstance(st, dict) or not st.get("nom"):
                continue
            try:
                propre = {"nom": str(st["nom"])[:30],
                          "duree_plan": round(min(4.0, max(0.15, float(st.get("duree_plan", 0.7)))), 2),
                          "beats_par_plan": round(min(16.0, max(0.25, float(st.get("beats_par_plan", 1)))), 2),
                          "sur_beat": round(min(1.0, max(0.0, float(st.get("sur_beat", 0.8)))), 2),
                          "flash": round(min(1.0, max(0.0, float(st.get("flash", 0.1)))), 2),
                          "coupes": int(st.get("coupes", 0)), "tempo": int(st.get("tempo", 0)),
                          "duree_video": float(st.get("duree_video", 0)), "pack": pack}
            except (TypeError, ValueError):
                continue
            copie_style.DOSSIER.mkdir(parents=True, exist_ok=True)
            (copie_style.DOSSIER / f"{copie_style._slug(propre['nom'])}.json").write_text(
                json.dumps(propre, ensure_ascii=False, indent=1), encoding="utf-8")
            ajoutes.append(propre["nom"])
        if not ajoutes:
            return self._json({"erreur": "Aucun style valide dans ce fichier."}, 400)
        return self._json({"ok": True, "styles": ajoutes})

    def _finaliser(self, d: dict):
        """Refait un aperçu en qualité finale : même musique, mêmes kills, mêmes effets."""
        f = Path(self.config["dossier_sortie"]) / nom_sur(str(d.get("nom", "")))
        info = f.with_suffix(".json")
        if not info.is_file():
            return self._json({"erreur": "Cet aperçu ne peut pas être refait (infos manquantes)."}, 404)
        data = json.loads(info.read_text(encoding="utf-8"))
        champs = {k: v for k, v in data["options"].items() if k in Options.__dataclass_fields__}
        sortie = f.with_name(f.stem.replace("_apercu", "") + "_final.mp4")
        champs.update(apercu=False, graine=data["graine"], chansons=data["chansons"], sortie=str(sortie),
                      version_seule=data.get("version", 0), jusqu_a_ts=champs.get("jusqu_a_ts") or data.get("cree"))
        if champs.get("format") == "youtube":
            champs["version_seule"] = None
        if not self.tache.lancer(Options(**champs)):
            return self._json({"erreur": "Un montage est déjà en cours."}, 409)
        return self._json({"ok": True})

    def _generer(self, d: dict):
        if not self.config["clips"]:
            return self._json({"erreur": "Choisis d'abord le dossier de tes clips Medal (onglet Plus)."}, 400)
        if d.get("musique") == "__auto__":
            musique = Path(self.config["dossier_musiques"])
            if not any(f.suffix.lower() in EXT_AUDIO for f in musique.iterdir()):
                return self._json({"erreur": "Ajoute au moins une musique."}, 400)
        else:
            musique = Path(self.config["dossier_musiques"]) / nom_sur(str(d.get("musique", "")))
            if not musique.is_file():
                return self._json({"erreur": "Choisis une musique."}, 400)

        def nombre(cle, defaut, mini, maxi, type_=float):
            try:
                return type_(min(maxi, max(mini, type_(d.get(cle, defaut)))))
            except (TypeError, ValueError):
                return defaut

        debut = d.get("debut_musique")
        try:
            debut = float(str(debut).replace(",", ".")) if debut not in (None, "") else None
        except ValueError:
            debut = None
        rythme = d.get("rythme") if d.get("rythme") in ("calme", "normal", "nerveux") else "normal"
        filtre = d.get("filtre") if d.get("filtre") in FILTRES else "tous"
        connus = set(self._jeux())
        jeux = [j for j in (d.get("jeux") or []) if isinstance(j, str) and j in connus] or None
        cadrage = d.get("cadrage") if d.get("cadrage") in ("flou", "crop", "ecran_partage") else "flou"
        apercu = bool(d.get("apercu"))
        format_ = "youtube" if d.get("format") == "youtube" else "vertical"
        style = d.get("style") if d.get("style") in ("auto", "phonk", "cinematique", "clean") else "auto"
        horodatage = time.strftime("%Y-%m-%d_%H-%M-%S")
        prefixe = "youtube" if format_ == "youtube" else "autoclip"
        sortie = Path(self.config["dossier_sortie"]) / f"{prefixe}_{horodatage}{'_apercu' if apercu else ''}.mp4"
        # Période : « session » (ta dernière session de jeu), « jour », « semaine », ou tout.
        depuis_ts = jusqu_a_ts = None
        periode = d.get("periode")
        if periode == "session":
            s = sessions.derniere_session(index_kills.charger_index(), d.get("jeux") or None)
            if s:
                depuis_ts, jusqu_a_ts = s[0], s[1]
        elif periode in ("jour", "semaine"):
            depuis_ts = time.time() - (86400 if periode == "jour" else 7 * 86400)
        liste = lambda cle: [str(x)[:40] for x in (d.get(cle) or []) if isinstance(x, str)][:20] or None
        selection = [{"clip": nom_sur(str(k.get("clip", ""))), "t": float(k.get("t", 0))}
                     for k in (d.get("selection") or []) if isinstance(k, dict)][:200] or None
        drop = d.get("drop") if isinstance(d.get("drop"), dict) else None
        if drop:
            drop = {"clip": nom_sur(str(drop.get("clip", ""))), "t": float(drop.get("t", 0))}
        if d.get("pseudo") is not None:
            self.config["pseudo"] = str(d.get("pseudo"))[:30]
        duree_max = 200 if format_ == "youtube" else 90

        o = Options(clips=[self.config["clips"]], musique=str(musique), sortie=str(sortie),
                    duree=nombre("duree", 150 if format_ == "youtube" else 20, 8, duree_max), cadrage=cadrage, rythme=rythme,
                    son_jeu=nombre("son_jeu", 30, 0, 100) / 100, ralenti=0.5 if d.get("ralenti", True) else 1.0,
                    debut_musique=debut, derniers=nombre("derniers", 25, 0, 500, int),
                    texte=str(d.get("texte", ""))[:60], fps=60 if str(d.get("fps")) == "60" else 30,
                    apercu=apercu, filtre=filtre, jeux=jeux, textes_kills=bool(d.get("textes", True)),
                    sequences=bool(d.get("sequences", True)),
                    versions=3 if str(d.get("versions")) == "3" else 1, hook=bool(d.get("hook", True)),
                    reactions=bool(d.get("voix", True)), sous_titres=bool(d.get("sous_titres", True)),
                    format=format_, style=style, compteur=bool(d.get("compteur", True)),
                    outro=self.config.get("pseudo", "") if d.get("outro", True) else "",
                    effets_sonores=bool(d.get("effets_sonores", True)), armes=liste("armes"), agents=liste("agents"),
                    selection=selection, drop=drop, depuis_ts=depuis_ts, jusqu_a_ts=jusqu_a_ts,
                    mode=d.get("mode") if d.get("mode") in ("fun", "morts", "equipe") else "kills",
                    caster=d.get("caster") if d.get("caster") in ("en", "fr") else "",
                    instrumental=bool(d.get("instrumental", True)), arret_image=bool(d.get("arret_image", True)),
                    profil=str(d.get("profil") or "")[:40],
                    langue_sous_titres="en" if d.get("langue_sous_titres") == "en" else "fr",
                    filigrane=("@" + self.config["pseudo"]) if d.get("filigrane") and self.config.get("pseudo") else "")
        if not licence.est_pro():
            pro = licence.fonctions_pro_demandees(o)
            if pro:
                return self._json({"erreur": "Réservé à AutoClip Pro : " + ", ".join(pro)
                                   + ". Deviens membre Patreon pour le débloquer.", "pro": True}, 402)
            if not o.jeux:
                if not any(licence.est_jeu_gratuit(j) for j in connus):
                    return self._json({"erreur": "Aucun clip Valorant trouvé. Les autres jeux sont dans "
                                       "AutoClip Pro (Patreon).", "pro": True}, 402)
                o.jeux = [j for j in connus if licence.est_jeu_gratuit(j)]
        if not self.tache.lancer(o):
            return self._json({"erreur": "Un montage est déjà en cours."}, 409)
        # Mémorise les réglages pour la prochaine fois.
        if not selection:   # un montage « à la main » ne change pas les réglages habituels
            self.config["reglages"] = {k: d.get(k) for k in (
                "musique", "duree", "cadrage", "rythme", "filtre", "jeux", "textes", "sequences", "versions", "hook",
                "voix", "sous_titres", "son_jeu", "ralenti", "derniers", "texte", "fps", "apercu", "format", "style",
                "compteur", "outro", "effets_sonores", "armes", "agents", "mode", "langue_sous_titres", "filigrane",
                "caster", "instrumental", "arret_image", "profil")}
        sauver_config(self.config)
        self._json({"ok": True})

    def _recevoir_style(self):
        """Une vidéo (TikTok que tu aimes) choisie dans l'app : on mesure son style puis on l'efface."""
        import tempfile
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 300 * 1024 * 1024:
            return self._json({"erreur": "Vidéo vide ou trop grosse (300 Mo max)."}, 400)
        nom = unquote(self.headers.get("X-Nom-Style", "") or "Mon style").strip()[:30] or "Mon style"
        temp = Path(tempfile.mkdtemp(prefix="autoclip_style_")) / "video.mp4"
        try:
            restant = n
            with open(temp, "wb") as f:
                while restant > 0:
                    bloc = self.rfile.read(min(1 << 20, restant))
                    if not bloc:
                        break
                    f.write(bloc)
                    restant -= len(bloc)
            if restant:
                return self._json({"erreur": "Envoi interrompu."}, 400)
            profil = copie_style.analyser(temp, nom)
        except Exception as e:
            return self._json({"erreur": f"Analyse impossible : {e}"}, 400)
        finally:
            import shutil
            shutil.rmtree(temp.parent, ignore_errors=True)
        return self._json({"ok": True, "profil": profil})

    def _choix(self, d: dict):
        """Choisir le titre ou la miniature d'un montage parmi les 3 proposés."""
        f = Path(self.config["dossier_sortie"]) / nom_sur(str(d.get("nom", "")))
        if f.suffix.lower() != ".mp4" or not f.is_file():
            return self._json({"erreur": "Vidéo introuvable."}, 404)
        if "miniature" in d:
            i = int(d["miniature"])
            m = f.with_name(f"{f.stem}_mini{i}.jpg")
            if i not in (1, 2, 3) or not m.exists():
                return self._json({"erreur": "Miniature introuvable."}, 404)
            import shutil
            shutil.copy2(m, f.with_suffix(".jpg"))
        if "titre" in d:
            titre = str(d["titre"]).replace("\n", " ").strip()[:100]
            t = f.with_suffix(".txt")
            lignes = t.read_text(encoding="utf-8").splitlines() if t.exists() else [""]
            lignes[0] = titre
            t.write_text("\n".join(lignes) + "\n", encoding="utf-8")
        return self._json({"ok": True})

    def _recevoir_musique(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > TAILLE_MAX_MUSIQUE:
            return self._json({"erreur": "Fichier vide ou trop gros (80 Mo max)."}, 400)
        nom = nom_sur(self.headers.get("X-Nom-Fichier", "musique.mp3"))
        if Path(nom).suffix.lower() not in EXT_AUDIO:
            return self._json({"erreur": "Format audio non supporté."}, 400)
        dossier = Path(self.config["dossier_musiques"])
        dossier.mkdir(parents=True, exist_ok=True)
        cible = dossier / nom
        temp = cible.with_suffix(cible.suffix + ".part")
        restant = n
        with open(temp, "wb") as f:
            while restant > 0:
                bloc = self.rfile.read(min(1 << 20, restant))
                if not bloc:
                    break
                f.write(bloc)
                restant -= len(bloc)
        if restant:
            temp.unlink(missing_ok=True)
            return self._json({"erreur": "Envoi interrompu."}, 400)
        temp.replace(cible)
        self._json({"ok": True, "nom": nom})

    def _maj_config(self, d: dict):
        if "clips" in d:
            p = Path(str(d["clips"]).strip().strip('"')).expanduser()
            if not p.is_dir():
                return self._json({"erreur": f"Dossier introuvable sur le PC : {p}"}, 400)
            self.config["clips"] = str(p)
        if "pseudo" in d:
            self.config["pseudo"] = str(d["pseudo"]).strip()[:30]
        if "auto_session" in d:
            self.config["auto_session"] = bool(d["auto_session"])
            if d["auto_session"] and not self.config.get("auto_session_faite"):
                # On ne refait pas les vieilles sessions : seulement les prochaines.
                self.config["auto_session_faite"] = time.time()
        sauver_config(self.config)
        self._json({"ok": True})

    def _supprimer(self, d: dict):
        if self.tache.etat()["en_cours"]:
            return self._json({"erreur": "Attends la fin du montage en cours."}, 409)
        f = Path(self.config["dossier_sortie"]) / nom_sur(str(d.get("nom", "")))
        if f.suffix.lower() != ".mp4" or not f.is_file():
            return self._json({"erreur": "Vidéo introuvable."}, 404)
        f.unlink()
        for ext in (".jpg", ".txt", ".json", "_mini1.jpg", "_mini2.jpg", "_mini3.jpg"):
            f.with_name(f.stem + ext).unlink(missing_ok=True)
        self._json({"ok": True})

    def _fichier(self, dossier: Path, nom: str, telecharger: bool = False):
        """Sert un fichier avec support des plages (nécessaire pour lire la vidéo sur Android)."""
        f = dossier / nom_sur(nom)
        if not f.is_file():
            return self._json({"erreur": "introuvable"}, 404)
        taille = f.stat().st_size
        type_ = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        debut, fin = 0, taille - 1
        plage = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", plage or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                debut = int(m.group(1))
                fin = int(m.group(2)) if m.group(2) else fin
            else:
                debut = max(0, taille - int(m.group(2)))
            fin = min(fin, taille - 1)
            if debut > fin:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{taille}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {debut}-{fin}/{taille}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", type_)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(fin - debut + 1))
        if telecharger:
            self.send_header("Content-Disposition", f'attachment; filename="{f.name}"')
        self.end_headers()
        try:
            with open(f, "rb") as fh:
                fh.seek(debut)
                restant = fin - debut + 1
                while restant > 0:
                    bloc = fh.read(min(1 << 20, restant))
                    if not bloc:
                        break
                    self.wfile.write(bloc)
                    restant -= len(bloc)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass  # le téléphone a arrêté la lecture

# ---------------------------------------------------------------- pages fixes
PAGE_REFUS = """<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AutoClip</title>
<body style="background:#0e0f13;color:#e9eaf0;font:16px system-ui;padding:32px;text-align:center">
<h2>AutoClip</h2><p>Ouvre AutoClip depuis son raccourci (menu Démarrer ou bureau).</p></body></html>"""

MANIFEST = json.dumps({
    "name": "AutoClip", "short_name": "AutoClip", "start_url": "/", "display": "standalone",
    "background_color": "#0e0f13", "theme_color": "#0e0f13",
    "icons": [{"src": "/icone.svg", "sizes": "any", "type": "image/svg+xml"}],
})

ICONE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" rx="112" fill="#0e0f13"/>
<rect x="140" y="96" width="232" height="320" rx="36" fill="none" stroke="#ff3d6e" stroke-width="28"/>
<path d="M226 196v120l100-60z" fill="#25f4ee"/></svg>"""


def demarrer(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Lance le serveur en arrière-plan sur 127.0.0.1. Retourne (serveur, adresse à ouvrir)."""
    config = charger_config()
    config["cle"] = secrets.token_urlsafe(16)          # nouvelle clé à chaque lancement
    sauver_config(config)
    for d in ("dossier_sortie", "dossier_musiques"):
        Path(config[d]).mkdir(parents=True, exist_ok=True)
    Gestionnaire.config = config
    Gestionnaire.tache = Tache()
    Gestionnaire.surveillant = Surveillant(config, Gestionnaire.tache)
    Gestionnaire.surveillant.demarrer()
    serveur = ThreadingHTTPServer(("127.0.0.1", port), Gestionnaire)
    serveur.daemon_threads = True
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    return serveur, f"http://127.0.0.1:{serveur.server_address[1]}/?cle={config['cle']}"


def main() -> int:
    """Sans fenêtre d'app : ouvre AutoClip dans le navigateur (utile pour tester)."""
    ap = argparse.ArgumentParser(description="AutoClip dans le navigateur.")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    try:
        serveur, url = demarrer(a.port)
    except OSError:
        print(f"Le port {a.port} est déjà utilisé. AutoClip est peut-être déjà ouvert.")
        return 1
    print(f"AutoClip : {url.split('?')[0]}  (Ctrl+C pour arrêter)")
    webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
