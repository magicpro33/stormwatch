"""Process-wide logging for desktop and Cloud.

Desktop writes to %LOCALAPPDATA%\\MoneyWeather\\moneyweather.log.
Cloud / `streamlit run` logs to stderr (Streamlit captures it).
"""
from __future__ import annotations

import logging
import os
import sys

LOG_NAME = "moneyweather"
_configured = False


def log_path() -> str:
    try:
        from mw_paths import user_data_dir, desktop_mode
        if desktop_mode():
            return os.path.join(user_data_dir(), "moneyweather.log")
    except Exception:
        pass
    override = (os.environ.get("MW_DATA_DIR") or "").strip()
    if override:
        return os.path.join(override, "moneyweather.log")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "moneyweather.log")


def setup_logging() -> logging.Logger:
    """Idempotent. Safe to call from desktop_main and from the Streamlit script."""
    global _configured
    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(logging.INFO)
    if _configured or logger.handlers:
        _configured = True
        return logger
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    path = log_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    _configured = True
    return logger


def get_logger() -> logging.Logger:
    if not _configured:
        return setup_logging()
    return logging.getLogger(LOG_NAME)


def log_exc(where: str, exc: BaseException) -> None:
    get_logger().warning("%s: %s: %s", where, type(exc).__name__, exc)
