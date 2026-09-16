# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — onedir folder you can zip and run on a clean Windows PC."""
from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata
import os

ROOT = os.path.abspath(SPECPATH)

datas = [
    (os.path.join(ROOT, "app.py"), "."),
    (os.path.join(ROOT, "cascade_engine.py"), "."),
    (os.path.join(ROOT, "apex_flow.py"), "."),
    (os.path.join(ROOT, "poc_future.py"), "."),
    (os.path.join(ROOT, "hybrid_screener.py"), "."),
    (os.path.join(ROOT, "storm_watch_tab.py"), "."),
    (os.path.join(ROOT, "storm_watch_engine.py"), "."),
    (os.path.join(ROOT, "ignition_scanner.py"), "."),
    (os.path.join(ROOT, "mw_paths.py"), "."),
    (os.path.join(ROOT, "mw_log.py"), "."),
    (os.path.join(ROOT, "mw_secrets.py"), "."),
    (os.path.join(ROOT, "mw_yf.py"), "."),
    (os.path.join(ROOT, "macro_simulator.html"), "."),
    (os.path.join(ROOT, ".streamlit", "config.toml"), ".streamlit"),
]
for extra in ("LICENSE", "THIRD_PARTY_NOTICES.txt"):
    p = os.path.join(ROOT, extra)
    if os.path.isfile(p):
        datas.append((p, "."))
_assets = os.path.join(ROOT, "assets")
if os.path.isdir(_assets):
    datas.append((_assets, "assets"))
_bt = os.path.join(ROOT, "data", "storm_watch_backtest.json")
if os.path.isfile(_bt):
    datas.append((_bt, "data"))

hiddenimports = [
    "mw_paths", "mw_log", "mw_secrets", "mw_yf",
    "cascade_engine", "apex_flow", "poc_future", "hybrid_screener",
    "storm_watch_tab", "storm_watch_engine", "ignition_scanner",
    "streamlit", "streamlit.web.cli", "streamlit.runtime.scriptrunner",
    "yfinance", "plotly", "plotly.graph_objects", "plotly.subplots",
    "pandas", "numpy", "pyarrow", "requests", "tzdata",
    "altair", "tornado", "watchdog", "blinker", "click",
    "ijson", "tomli",
]
binaries = []

for pkg in ("streamlit", "altair", "plotly", "yfinance", "tzdata"):
    try:
        t, b, h = collect_all(pkg)
        datas += t
        binaries += b
        hiddenimports += h
    except Exception:
        pass
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

for pkg in ("pyarrow", "pandas", "numpy"):
    try:
        datas += collect_data_files(pkg, excludes=["**/tests/*", "**/tests/**", "**/*test*"])
    except Exception:
        pass
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

try:
    datas += collect_data_files("tzdata")
except Exception:
    pass

block_cipher = None
_icon = os.path.join(ROOT, "assets", "moneyweather.ico")
_ver = os.path.join(ROOT, "file_version_info.txt")

a = Analysis(
    [os.path.join(ROOT, "desktop_main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(ROOT, "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "tkinter", "IPython", "notebook",
        "pandas.tests", "numpy.tests", "numpy.testing.tests",
        "pyarrow.tests", "pytest", "setuptools.tests",
        "curl_cffi",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe_kw = dict(
    exclude_binaries=True,
    name="MoneyWeather",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
if os.path.isfile(_icon):
    exe_kw["icon"] = _icon
if os.path.isfile(_ver):
    exe_kw["version"] = _ver

exe = EXE(
    pyz,
    a.scripts,
    [],
    **exe_kw,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MoneyWeather",
)
