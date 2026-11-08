# PyInstaller spec: one-folder Windows build. Build with: python ../tools/build_release.py
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

a = Analysis(
    ["run_engine_trainer.py"],
    pathex=["."],
    datas=[("engine_trainer/inference.py", "engine_trainer")] + collect_data_files("lightgbm") + collect_data_files("optuna"),
    hiddenimports=collect_submodules("engine_trainer") + ["sklearn.ensemble._hist_gradient_boosting", "threadpoolctl"]
    + collect_submodules("optuna") + collect_submodules("lightgbm"),
    excludes=["torch", "tensorflow", "matplotlib", "tkinter", "IPython", "pytest", "notebook", "jupyter", "sphinx",
              "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtMultimedia"],
    noarchive=False,
)
pyz = PYZ(a.pure)
# GUI launcher (no console window) and a console launcher for the CLI / --selftest, sharing one set of libraries
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Epoch", console=False, disable_windowed_traceback=False)
cli = EXE(pyz, a.scripts, [], exclude_binaries=True, name="EpochCLI", console=True)
coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name="Epoch")
