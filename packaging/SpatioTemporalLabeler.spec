# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_submodules


project_root = Path.cwd()
project_hidden_imports = collect_submodules("spatiotemporal_labeler")
application_icon = project_root / "packaging" / "assets" / "app-icon.ico"
application_icon_png = (
    project_root / "src" / "spatiotemporal_labeler" / "assets" / "app-icon.png"
)

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[(str(application_icon_png), "spatiotemporal_labeler/assets")],
    hiddenimports=project_hidden_imports + [
        "vtkmodules.vtkInteractionStyle",
        "vtkmodules.vtkRenderingOpenGL2",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / "packaging" / "runtime_hook.py")],
    excludes=[
        "PyQt5",
        "PyQt6",
        "PySide2",
        "IPython",
        "OpenGL",
        "PIL",
        "cupy",
        "cupyx",
        "h5py",
        "jedi",
        "lxml",
        "matplotlib",
        "numba",
        "pandas",
        "pytest",
        "sklearn",
        "tensorflow",
        "tkinter",
        "torch",
        "zmq",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SpatioTemporalLabeler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(application_icon) if sys.platform == "win32" else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SpatioTemporalLabeler",
)
