"""Alpaca / ntfy secrets from Streamlit secrets, env, or desktop secrets.toml."""
from __future__ import annotations

import os

_ALPACA_PAIRS = (
    ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
    ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
    ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
)


def get_secret(name: str) -> str:
    """st.secrets first (Cloud + copied desktop file), then os.environ."""
    try:
        import streamlit as st
        v = st.secrets.get(name, "")
        if v:
            return str(v).strip().strip('"').strip("'")
    except Exception:
        pass
    v = os.environ.get(name, "")
    return str(v).strip().strip('"').strip("'") if v else ""


def alpaca_key_pair() -> tuple[str, str]:
    for a, b in _ALPACA_PAIRS:
        k, s = get_secret(a), get_secret(b)
        if k and s and k not in ("your-key",) and s not in ("your-secret",):
            return k, s
    return "", ""
