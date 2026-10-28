# PyInstaller : construit dist/AutoClip/AutoClip.exe (dossier complet, sans console).
# Lancer depuis le dossier build/ :  pyinstaller --noconfirm autoclip.spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

APP = Path(SPECPATH).parent / "app"
ICONE = Path(SPECPATH) / "autoclip.ico"

datas = [
    (str(APP / "web"), "web"),
    (str(APP / "modeles_valorant.npz"), "."),
]
# librosa charge ses modules à la demande (lazy_loader + fichiers .pyi) : il faut tout embarquer.
datas += collect_data_files("librosa")
datas += collect_data_files("edge_tts")
hidden = collect_submodules("librosa") + ["sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes"]

a = Analysis(
    [str(APP / "app.py")],
    pathex=[str(APP)],
    datas=datas,
    hiddenimports=hidden,
    excludes=["torch", "demucs", "faster_whisper", "matplotlib", "IPython", "pytest", "tkinter.test",
              "pandas", "lxml", "cryptography", "pydantic", "pydantic_core", "PIL", "playwright", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AutoClip",
    console=False,
    icon=str(ICONE) if ICONE.exists() else None,
    version=str(Path(SPECPATH) / "version_info.txt"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="AutoClip")
