# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

project_root = Path(SPECPATH).resolve()
block_cipher = None


def _collect_package(name: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    datas, binaries, hiddenimports = collect_all(name)
    return list(datas), list(binaries), list(hiddenimports)


datas: list[tuple[str, str]] = [(str(project_root / "config"), "config")]
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = []

# Open3D is needed at runtime, but its optional ML integrations trigger noisy
# PyInstaller warnings when submodules are scanned wholesale. We only collect its
# data files and native libraries here; regular import analysis handles the code
# paths actually used by scan2plan.
datas += collect_data_files("open3d")
binaries += collect_dynamic_libs("open3d")

for package_name in ("pye57", "lazrs", "matplotlib", "cv2"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = _collect_package(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

hiddenimports = sorted(set(hiddenimports))

a = Analysis(
    [str(project_root / "src" / "scan2plan" / "cli.py")],
    pathex=[str(project_root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="scan2plan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

