"""Single-thread yfinance executor — curl/libcurl handles are not thread-safe.

Import this instead of calling yf.Ticker from Streamlit worker threads or
from Ignition's 12-way scan pool.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")

_YF_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yfinance-io")


def yf_call(fn, *args, **kwargs):
    """Run a yfinance-touching callable on the dedicated I/O thread."""
    return _YF_EXECUTOR.submit(fn, *args, **kwargs).result()
