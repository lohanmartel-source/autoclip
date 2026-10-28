"""AutoClip : point d'entrée de l'app Windows.

Lance le serveur local puis ouvre la fenêtre de l'app (pywebview, moteur Edge WebView2).
Si la fenêtre ne peut pas s'ouvrir, AutoClip s'ouvre dans le navigateur.
"""
from __future__ import annotations

import multiprocessing
import sys
import traceback
import webbrowser

import chemins


def _journal_erreur() -> None:
    try:
        (chemins.DONNEES / "erreur.log").write_text(traceback.format_exc(), encoding="utf-8")
    except OSError:
        pass


def main() -> int:
    import serveur

    try:
        _, url = serveur.demarrer(0)
    except Exception:
        _journal_erreur()
        raise

    try:
        import webview
    except ImportError:
        webview = None

    if webview is None:
        webbrowser.open(url)
        print("AutoClip est ouvert dans ton navigateur. Ferme cette fenêtre pour quitter.")
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        return 0

    webview.settings["ALLOW_DOWNLOADS"] = True
    fenetre = webview.create_window(
        f"AutoClip {chemins.VERSION}", url, width=1280, height=820, min_size=(420, 600),
        background_color="#0c0e0b", text_select=True)

    def choisir(depart: str = "") -> str:
        res = fenetre.create_file_dialog(webview.FOLDER_DIALOG, directory=depart or "")
        return res[0] if res else ""

    serveur.Gestionnaire.choisir_dossier = staticmethod(choisir)
    webview.start(private_mode=False, storage_path=str(chemins.DONNEES / "webview"))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except Exception:
        _journal_erreur()
        sys.exit(1)
