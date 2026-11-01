"""Single-thread yfinance executor — curl/libcurl handles are not thread-safe.

Import this instead of calling yf.Ticker from Streamlit worker threads or
from Ignition's 12-way scan pool.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")

_YF_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yfinance-io")
_YF_BLOCK_UNTIL = 0.0
_YF_BLOCK_SEC = 300


def cooling() -> bool:
    """True while Yahoo has asked us to back off."""
    return time.time() < _YF_BLOCK_UNTIL


def _is_rate_limit(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return ("ratelimit" in name or "too many requests" in msg
            or "rate limited" in msg or "cooling off" in msg)


def yf_call(fn, *args, **kwargs):
    """Run a yfinance-touching callable on the dedicated I/O thread."""
    global _YF_BLOCK_UNTIL
    if cooling():
        raise RuntimeError("YFRateLimitError: cooling off")
    try:
        return _YF_EXECUTOR.submit(fn, *args, **kwargs).result()
    except Exception as exc:
        if _is_rate_limit(exc):
            _YF_BLOCK_UNTIL = time.time() + _YF_BLOCK_SEC
        raise
