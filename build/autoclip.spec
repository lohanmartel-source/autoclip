# PyInstaller : construit dist/AutoClip/AutoClip.exe (dossier complet, sans console).
# Lancer depuis le dossier build/ :  pyinstaller --noconfirm autoclip.spec
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

APP = Path(SPECPATH).parent / "app"
ICONE = Path(SPECPATH) / "autoclip.ico"

datas = [
    (str(APP / "web"), "web"),
    (str(APP / "modeles_valorant.npz"), "."),
    (str(APP / "polices"), "polices"),   # Anton (OFL) : paroles dans le décor
]
# librosa charge ses modules à la demande (lazy_loader + fichiers .pyi) : il faut tout embarquer.
datas += collect_data_files("librosa")
datas += collect_data_files("edge_tts")
hidden = collect_submodules("librosa") + ["sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes"]

# Whisper local (faster-whisper) : synchro des paroles mot à mot + sous-titres du micro.
# Pièges PyInstaller : le modèle VAD (faster_whisper/assets/silero_vad_v6.onnx), les DLL de
# CTranslate2 et d'onnxruntime, les modules Cython de PyAV, et les métadonnées (importlib.metadata)
# que huggingface_hub lit pour savoir quelles librairies sont là. Le modèle Whisper lui-même n'est
# PAS embarqué : il se télécharge à la 1re utilisation dans DONNEES/modeles.
datas += collect_data_files("faster_whisper")
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")
hidden += collect_submodules("faster_whisper") + collect_submodules("ctranslate2") + collect_submodules("av")
hidden += ["tokenizers", "onnxruntime", "huggingface_hub", "tqdm"]
for paquet in ("faster_whisper", "ctranslate2", "huggingface_hub", "tokenizers", "onnxruntime", "av", "tqdm",
               "hf_xet", "filelock", "fsspec", "packaging", "pyyaml", "requests", "httpx"):
    try:
        datas += copy_metadata(paquet)
    except Exception:
        pass            # paquet absent : rien à copier
# Paroles derrière l'arme (Pro, arme_ia.py) : MobileSAM en ONNX, lancé par le même onnxruntime.
# Importé à la demande par paroles.py ; le modèle (~45 Mo) n'est PAS embarqué (DONNEES/modeles).
hidden += ["arme_ia"]
if find_spec("hf_xet") is not None:
    hidden.append("hf_xet")   # téléchargement rapide des modèles (importé à la demande par huggingface_hub)

a = Analysis(
    [str(APP / "app.py")],
    pathex=[str(APP)],
    datas=datas,
    binaries=binaries,
    hiddenimports=hidden,
    excludes=["torch", "demucs", "matplotlib", "IPython", "pytest", "tkinter.test",
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
