"""Bundle vs user-data paths for Cloud, local, and the Windows desktop build.

Cloud / `streamlit run` keep writing next to the repo (`data/`).
The frozen desktop exe writes caches to %LOCALAPPDATA%\\MoneyWeather so a
copy under Program Files stays read-only.
"""
from __future__ import annotations
import os
import sys


def frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def desktop_mode() -> bool:
    return frozen() or os.environ.get("MW_DESKTOP", "").strip() in ("1", "true", "yes")


def bundle_dir() -> str:
    """Read-only files shipped with the app (app.py, HTML, assets)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if frozen() and meipass:
        return meipass
    return os.path.dirname(os.path.abspath(__file__))


def exe_dir() -> str:
    """Folder that contains MoneyWeather.exe (or the source tree)."""
    if frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def user_data_dir() -> str:
    override = (os.environ.get("MW_DATA_DIR") or "").strip()
    if override:
        return os.path.abspath(override)
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(root, "MoneyWeather")
    return os.path.join(os.path.expanduser("~"), ".moneyweather")


def data_dir() -> str:
    """Writable cache: nightly dump, history.parquet, watchlist."""
    if desktop_mode():
        d = os.path.join(user_data_dir(), "data")
    else:
        d = os.path.join(bundle_dir(), "data")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d
