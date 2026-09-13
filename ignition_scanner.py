"""
IGNITION - Momentum Ignition Scanner
====================================
Detects the earliest minutes of a momentum move by combining:

  IGNITION signals (live, intraday):
    - Relative volume vs 20-day pace (RVOL)
    - Bar-level volume surge (last 3 bars vs session average)
    - Price velocity and acceleration (1-minute bars)
    - VWAP reclaim / cross
    - High-of-day breakout
    - RSI(14) thrust on 5-minute bars
    - MACD bullish cross on 5-minute bars

  FUEL signals (daily, "is this stock primed to move"):
    - Short percent of float (squeeze fuel)
    - Insider net buying, last 90 days
    - Fresh news flow with keyword sentiment
    - Float size (small float = explosive)
    - Distance to 52-week high

Each ticker gets an Ignition Score (0-100). When live confirmation
conditions all fire at once, the ticker is flagged IGNITING and logged
to the alert feed with a timestamp.

Run:  streamlit run ignition_scanner.py
Data: Yahoo Finance via yfinance (free, ~15 min delayed on some feeds).
"""

import time
import random
import math
import re
import gzip
import json
import decimal
import threading
import os

# ── CRITICAL: disable curl_cffi BEFORE yfinance is imported ──────────
# curl_cffi 0.15.0 (yfinance's default HTTP backend) has documented
# segfault-class bugs (memory corruption in Curl.reset(), upstream
# issue #677) that crashed this app repeatedly in production — killing
# the interpreter with no Python traceback. Serializing access (v3.7),
# removing deprecated APIs (v3.8), and even pinning every yfinance call
# to a single dedicated thread (v3.9) did not stop the crashes, proving
# the fault is inside the C extension itself, not our threading.
#
# yfinance provides an official escape hatch: this env var makes it use
# plain `requests` and never import curl_cffi at all. Trade-off: no
# browser TLS impersonation, so Yahoo may rate-limit more aggressively —
# but a 429 is a catchable Python exception our data_issues reporting
# already handles; a segfault kills the whole app. Alpaca remains the
# primary intraday source and the nightly dump + seed cache backstop
# fundamentals, so degraded yfinance access is survivable by design.
# This MUST run before `import yfinance` (checked at module import).
os.environ["YF_DISABLE_CURL_CFFI"] = "1"

# Belt-and-suspenders: yfinance also has secondary, env-var-unaware
# `from curl_cffi...` imports (e.g. session type-checking helpers) that
# are guarded only by `except ImportError`. Poisoning the sys.modules
# entry makes ANY `import curl_cffi` anywhere in this process raise
# ImportError, so those guarded imports skip cleanly and the crashing
# C extension can never be loaded by any code path at all.
import sys as _sys
_sys.modules["curl_cffi"] = None

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── yfinance thread-affinity guard ────────────────────────────────────
# yfinance's HTTP layer (curl_cffi) has caused interpreter segfaults in
# production even when access was serialized with a plain Lock. A Lock
# only guarantees one thread touches yfinance AT A TIME — it does not
# guarantee the SAME thread does it every time. With a multi-worker pool,
# a different OS thread acquires the lock on each call. If curl_cffi (or
# the SSL/curl session it caches) has thread-affinity — a known category
# of bug in C extensions wrapping libcurl, where a handle created on one
# thread corrupts state when reused from another — that alone can crash
# the interpreter with zero contention and zero warning.
#
# The fix: route EVERY yfinance call, for the entire server process
# lifetime, through a single-worker executor. max_workers=1 guarantees
# the exact same OS thread executes every submitted call, eliminating
# thread-affinity crashes structurally rather than just reducing their
# odds. Alpaca calls (plain `requests`, thread-safe, no known affinity
# issues) are NOT routed through this and keep full 12-way parallelism.
_YF_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yfinance-io")


def _yf(fn, *args, **kwargs):
    """Run any yfinance-touching call on the single dedicated yfinance
    thread and block for its result. Use for every yf.Ticker/.info/
    .history/.calendar/.news/.insider_transactions/.earnings_* access
    and yf.Search — never call these directly from a worker thread."""
    return _YF_EXECUTOR.submit(fn, *args, **kwargs).result()

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ----------------------------------------------------------------------------
# Page config and theme
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="IGNITION - Momentum Scanner",
        layout="wide",
        initial_sidebar_state="expanded",
    )

CUSTOM_CSS = """
<style>
/* =====================================================================
   AI UPSCALE BRAND THEME
   Fonts  : Rajdhani (headings) + Plus Jakarta Sans (body) + Space Mono (data)
   Palette:
     --bg-deep   #07111f   deepest background (page)
     --bg-card   #0d1e33   card / row surfaces
     --bg-mid    #122540   borders / dividers
     --bg-hover  #1a3050   hover / selection
     --amber     #f5a623   primary accent (amber)
     --amber-dim #c47d0e   dimmed amber
     --amber-glow rgba(245,166,35,0.35)
     --text-hi   #e8f0fa   high-contrast text
     --text-mid  #8baac8   medium text / labels
     --text-lo   #4a6a8a   low / placeholder text
     --green     #3ddc84   ignition positive
     --red       #e05555   alert / reversal
     --teal      #29b6c8   secondary accent
     --navy-lt   #1e3a5f   light navy highlight
===================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"]  { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp                       { background: #07111f; }

h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important;
             letter-spacing: 0.8px; color: #ffffff; }

.metric-mono, .stDataFrame, code { font-family: 'Space Mono', monospace !important; }

/* ── Streamlit widget overrides ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #07111f; border-radius: 10px; gap: 4px;
    padding: 4px; border: 1px solid #1e3a5f; }
.stTabs [data-baseweb="tab"] {
    color: #7a9ab8; font-family: 'Rajdhani', sans-serif;
    font-weight: 700; font-size: 15px; letter-spacing: 1px;
    text-transform: uppercase; border-radius: 8px;
    padding: 8px 20px; border: 1px solid transparent;
    transition: all 0.15s; }
.stTabs [data-baseweb="tab"]:hover {
    color: #f5a623; background: #0d1e33; border-color: #1e3a5f; }
.stTabs [aria-selected="true"] {
    color: #f5a623 !important; background: #0d1e33 !important;
    border: 1px solid #f5a623 !important;
    box-shadow: 0 0 10px rgba(245,166,35,0.2); }
.stTabs [data-baseweb="tab-panel"] { background: transparent; }

/* ── IGNITING banner ─────────────────────────────────────────────── */
.ignite-banner {
    background: linear-gradient(90deg, #1a1000, #261800);
    border: 1px solid #f5a623;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'Space Mono', monospace;
    color: #f5c96a;
    animation: amber-pulse 1.6s infinite;
}
@keyframes amber-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(245,166,35,0.45); }
    70%  { box-shadow: 0 0 0 12px rgba(245,166,35,0); }
    100% { box-shadow: 0 0 0 0 rgba(245,166,35,0); }
}

/* ── GAP REVERSAL banner ─────────────────────────────────────────── */
.ignite-banner.rev {
    background: linear-gradient(90deg, #0e1a2a, #152030);
    border-color: #29b6c8;
    color: #7dd8e4;
    animation: none;
}
.crow.rev  { border-color: #29b6c8; }
.cflag.rev { color: #29b6c8; border-color: #29b6c8; }

/* ── Alert feed rows ─────────────────────────────────────────────── */
.alert-row {
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    color: #ffffff;
    padding: 6px 10px;
    border-left: 3px solid #f5a623;
    background: #0d1e33;
    margin-bottom: 4px;
    border-radius: 4px;
}

/* ── Plain info pills ────────────────────────────────────────────── */
.fuel-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
    background: #0d1e33;
    border: 1px solid #1e3a5f;
    color: #b0c8e8;
}
.fuel-tag a       { color: inherit; text-decoration: none; }
.fuel-tag a:hover { text-decoration: underline; opacity: 0.85; }

/* ── Catalyst tag color classes (navy-amber palette) ─────────────── */
.ct-earnings     { background:#0d2215; border-color:#1e6b35; color:#4dd880; }
.ct-fda          { background:#150d22; border-color:#6b35a0; color:#c07ae0; }
.ct-buyout       { background:#211800; border-color:#c47d0e; color:#f5c040; }
.ct-legal        { background:#220d0d; border-color:#a03535; color:#ff4444; }
.ct-partnership  { background:#07111f; border-color:#1e6a8a; color:#29b6c8; }
.ct-squeeze      { background:#1f1200; border-color:#c47d0e; color:#f5a623; }
.ct-breakout     { background:#071a10; border-color:#1a8040; color:#3ddc84; }
.ct-geopolitical { background:#0d1020; border-color:#3a5090; color:#7090d0; }
.ct-rate         { background:#1a1500; border-color:#907020; color:#d0b040; }
.ct-bimodal      { background:#221800; border-color:#f5a623; color:#f5a623; }
.ct-earn-growth  { background:#071a10; border-color:#1e8a3a; color:#4dd880; }
.ct-dtc          { background:#0a1828; border-color:#1e4a7a; color:#5090d0; }

/* ── DTC fuel gauge pill ──────────────────────────────────────────── */
.dtc-gauge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 2px 8px 2px 7px;
    border-radius: 4px;
    margin-right: 6px;
    background: #0a1828;
    border: 1px solid #1e4a7a;
    color: #b0c8e8;
    vertical-align: middle;
}
.dtc-gauge a { color: inherit; text-decoration: none; display: inline-flex; align-items: center; gap: 5px; }
.dtc-gauge a:hover { opacity: 0.85; }
.dtc-bar-track {
    display: inline-block;
    width: 36px; height: 5px;
    background: #122540;
    border-radius: 3px;
    overflow: hidden;
    vertical-align: middle;
}
.dtc-bar-fill {
    display: block;
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}

/* ── Card icon indicators ─────────────────────────────────────────── */
.cicon { flex-shrink:0; display:flex; align-items:center; justify-content:center; }
.cicon svg { display:block; }
/* ── Stat grid cards (Option 2) ──────────────────────────────────── */
.crow {
    background: #0d1e33;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 11px 13px 12px 13px;
    margin-bottom: 8px;
}
.crow.hot { border-color: #f5a623;
            box-shadow: 0 0 10px rgba(245,166,35,0.15); }
.crow.rev { border-color: #29b6c8; }
.chead {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.ctick  { font-size: 18px; font-weight: 700; color: #ffffff;
          letter-spacing: 0.3px; font-family: 'Rajdhani', sans-serif; }
.cflag  { font-size: 10px; color: #f5a623; border: 1px solid #f5a623;
          border-radius: 4px; padding: 1px 6px;
          margin-left: 8px; vertical-align: middle;
          font-family: 'Space Mono', monospace; }
.cflag.rev { color: #29b6c8; border-color: #29b6c8; }
.cbar   { height: 6px; background: #122540; border-radius: 3px;
          margin: 0 0 9px 0; overflow: hidden; }
.cfill  { height: 100%; border-radius: 3px; }
.cgrid  { display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
          margin-bottom: 8px; }
.ctile  { background: #07111f; border-radius: 6px; padding: 7px 9px; }
.ctile-lbl { font-family: 'Space Mono', monospace; font-size: 10px;
             letter-spacing: .6px; text-transform: uppercase;
             color: #7a9ab8; margin-bottom: 3px; }
.ctile-val { font-family: 'Space Mono', monospace; font-size: 13px;
             font-weight: 700; color: #ffffff; line-height: 1.2; }
.ctile-sub { font-family: 'Space Mono', monospace; font-size: 11px;
             margin-top: 1px; }
.ref-table { width: 100%; border-collapse: collapse; font-size: 13.5px;
             font-family: 'Plus Jakarta Sans', sans-serif; }
.ref-table td  { padding: 8px 10px; border-bottom: 1px solid #122540;
                 color: #ffffff; vertical-align: top; line-height: 1.5; }
.ref-table .grp td { padding-top: 20px; padding-bottom: 6px;
                     font-family: 'Rajdhani', sans-serif; font-weight: 700;
                     font-size: 14px; letter-spacing: 0.8px;
                     border-bottom: 1px solid #1e3a5f; text-transform: uppercase; }
.ref-table .trm { font-family: 'Space Mono', monospace; font-weight: 700;
                  white-space: nowrap; color: #ffffff; width: 120px; }
.ref-table .lvl { font-family: 'Space Mono', monospace; color: #7a9ab8;
                  font-size: 12px; white-space: nowrap; width: 170px; }
.ref-table .mng { color: #b0c8e8; }

/* ── Card sub-elements ───────────────────────────────────────────── */
.cline  { display: flex; align-items: baseline;
          justify-content: space-between;
          font-family: 'Space Mono', monospace; }
.ctick  { font-size: 17px; font-weight: 700; color: #ffffff;
          letter-spacing: 0.5px; font-family: 'Rajdhani', sans-serif; }
.cflag  { font-size: 10px; color: #f5a623; border: 1px solid #f5a623;
          border-radius: 4px; padding: 1px 6px;
          margin-left: 8px; vertical-align: middle;
          font-family: 'Space Mono', monospace; }
.cscore { font-size: 20px; font-weight: 700;
          font-family: 'Rajdhani', sans-serif; }
.cbar   { height: 5px; background: #122540; border-radius: 3px;
          margin: 6px 0 5px 0; overflow: hidden; }
.cfill  { height: 100%; border-radius: 3px; }
.csub   { font-family: 'Space Mono', monospace; font-size: 11.5px;
          color: #7a9ab8; display: flex; justify-content: space-between;
          flex-wrap: wrap; gap: 4px; }
/* ── Sidebar brand header ────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #07111f; border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] .stMarkdown p { color: #b0c8e8; font-size: 13px; }
[data-testid="stSidebar"] label { color: #ffffff !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 13px; }
[data-testid="stSidebar"] .stSlider span { color: #b0c8e8 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { color: #ffffff !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #ffffff !important; }
[data-testid="stSidebar"] .stTextArea label { color: #ffffff !important; }
[data-testid="stSidebar"] .stNumberInput label { color: #ffffff !important; }
[data-testid="stSidebar"] .stToggle label { color: #ffffff !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
    border-color: #1e3a5f; margin: 10px 0; }
.sidebar-section {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #7a9ab8;
    margin: 14px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e3a5f;
}
/* ── Streamlit caption color override ───────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] { color: #4a6a8a !important; }
/* ── Progress bar brand color ────────────────────────────────────── */
.stProgress > div > div { background-color: #f5a623 !important; }
/* ── Button brand style ──────────────────────────────────────────── */
.stButton button {
    background: #0d1e33; border: 1px solid #1e3a5f; color: #b0c8e8;
    font-family: 'Plus Jakarta Sans', sans-serif;
    border-radius: 6px; transition: all 0.2s;
}
.stButton button:hover { border-color: #f5a623; color: #f5a623; }
/* ── Success / info / error boxes ────────────────────────────────── */
.stSuccess { background: #071a10 !important; border-color: #3ddc84 !important;
             color: #3ddc84 !important; }
.stInfo    { background: #07111f !important; border-color: #1e3a5f !important;
             color: #b0c8e8 !important; }
.stError   { background: #220d0d !important; border-color: #ff3333 !important;
             color: #ff3333 !important; }
/* ── Selectbox / text area backgrounds ───────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stTextArea"] textarea {
    background: #0d1e33 !important; border-color: #1e3a5f !important;
    color: #ffffff !important;
}
</style>
"""

# ----------------------------------------------------------------------------
# Universe presets
# ----------------------------------------------------------------------------
PRESETS = {
    "Precious Metals / Mining": (
        "HL,AG,FSM,EXK,CDE,FCX,HBM,TECK,RIO,BHP,"
        "SCCO,ALB,GLDG,LAC,TMQ,NISTF,CGAU,CRML,AA,"
        "CENX,XME,COPX,SLVR,SILJ,PSLV,GORO,NVA,USAR,CMP,MP"
    ),
    "Energy / Oil / Gas / Uranium": (
        "CCJ,UROY,EU,LEU,NNE,SMR,OKLO,TXNM,GEV,CEG,"
        "NEE,XE,LNG,XOM,CVX,OXY,EQT,DVN,VLO,SHEL,"
        "BP,DTM,CNQ,AROC,FCG,XLE,XLU,UUUU,UUUG,SMUP"
    ),
    "Defense / Aerospace / Space": (
        "NOC,LMT,RTX,AVAV,ASTS,RKLB,RDW,LASR,SKYT,"
        "MOG.A,PKE,LPTH,LHX,GD,HII,ATRO,MRCY,VSAT,"
        "SPCX,BBAI,BKSY,PLTR,SYM,TER,HEI"
    ),
    "Semiconductors / Technology": (
        "AMD,MU,TSM,TSEM,INTC,SWKS,QRVO,SNXX,"
        "TTMI,VICR,SMH,SOXX,XSD,IGV,AMZN,MSFT,"
        "GOOGL,GOOG,NOW,ORCL,DT,TENB,CVLT,EXFY,"
        "INFQ,BOTZ,MAGS,ARKK,FTXR,TCAI"
    ),
    "Quantum / AI / Biotech": (
        "IONQ,QBTS,RGTI,QMCO,BBAI,CMPS,BMEA,ADPT,"
        "TXG,CTKB,BETA,MARA,IBIT,DTCR,ASTS,RDW,"
        "LPTH,SKYT,LASR,SMR,NNE,OKLO,XE,STNG,SEA"
    ),
    "Industrials / Infrastructure": (
        "PWR,MTZ,AGX,POWL,STRL,MWA,MLI,TEX,HLIO,NPO,"
        "GTES,KAI,FELE,CSW,AIT,SXI,MIDD,TRS,SNX,TX,"
        "VICR,XTN,XLI,NEWT,KFRC,OXM,EPC,DBI,GIS,KMB"
    ),
    "Income / Dividends / ETFs": (
        "T,VZ,MO,PM,BTI,ENB,ET,EPD,GLAD,AGD,"
        "OTF,FSK,ITUB,OCCI,LTC,O,MAIN,ARCC,NLY,AGNC,"
        "BWET,STNG,DTCR,SEA,TX,STNG,OXLC,PFLT,CSWC,GAIN,HTGC"
    ),
    "Critical Minerals / Materials": (
        "ALB,LAC,TMQ,MP,CRML,USAR,CMP,CGAU,AA,CENX,"
        "SCCO,FCX,TECK,RIO,BHP,XME,COPX,TMC,FLKR,EWY,"
        "TX,NVA,GORO,UUUU,EU,LEU,NNE,CCJ,UROY,CRMX"
    ),
}

# ETFs excluded from All Presets scan — funds don't have ignition signals
# (no insider buying, no earnings catalyst, no squeeze) and slow the scan.
# Individual stocks only for All Presets mode.
ALL_PRESETS_ETF_EXCLUDE = {
    "ARKK","MAGS","SPY","IWM","XLF","XLI","XLU","XLY","XLE","XME",
    "SOXX","SMH","IGV","BOTZ","DTCR","FTXR","XSD","XTN","COPX","SILJ",
    "SLVR","PSLV","JEPI","QYLD","SPHD","SRET","DIV","KBWD","PGX",
    "VTI","VOO","VUG","VEU","VT","VXUS","VTIAX","VTSAX","VTWAX","VFIAX",
    "VIGAX","ITA","XAR","KDEF","SPCX","EWY","FLKR","IBIT","FCG",
    "BWET","SEA","TCAI","AIPO","SMUP","UUUG","CRMX",
}

POSITIVE_WORDS = [
    "beat", "beats", "surge", "record", "upgrade", "upgraded", "raises",
    "contract", "award", "awarded", "partnership", "approval", "approved",
    "buyback", "acquisition", "acquire", "breakthrough", "expands", "wins",
    "guidance raised", "outperform", "buy rating", "patent", "milestone",
    "merger", "takeover", "deal", "agreement", "selected", "chosen",
    "fda approved", "cleared", "accelerated approval", "positive results",
    "positive data", "phase 3", "exceeded", "top-line", "revenue growth",
]
NEGATIVE_WORDS = [
    "miss", "misses", "downgrade", "downgraded", "cuts", "offering",
    "dilution", "lawsuit", "investigation", "recall", "halts", "delay",
    "bankruptcy", "warning", "sell rating", "underperform", "resigns",
    "rejected", "fda rejection", "clinical hold", "adverse", "fraud",
    "subpoena", "default", "going concern", "lowered guidance",
]

# Catalyst keyword buckets — each maps to a catalyst type for tagging
CATALYST_KEYWORDS = {
    "earnings":    ["earnings", "eps", "revenue beat", "quarterly results",
                    "q1", "q2", "q3", "q4", "fiscal", "guidance", "outlook",
                    "profit", "loss", "surprise"],
    "fda":         ["fda", "food and drug", "pdufa", "nda", "bla", "inda",
                    "clinical trial", "phase 1", "phase 2", "phase 3",
                    "approval", "approved", "clearance", "510k", "drug",
                    "biologics", "clinical hold"],
    "legal":       ["lawsuit", "settlement", "verdict", "litigation",
                    "court", "ruling", "judgment", "class action", "sued",
                    "damages", "injunction", "doj", "sec investigation",
                    "subpoena", "antitrust"],
    "buyout":      ["acquisition", "acquire", "merger", "takeover", "buyout",
                    "going private", "lbo", "strategic review", "sale process",
                    "offer to acquire", "bid for", "deal with", "m&a"],
    "partnership": ["partnership", "collaboration", "joint venture", "alliance",
                    "agreement", "contract", "mou", "supply agreement",
                    "licensing deal", "strategic agreement", "selected by"],
    "squeeze":     ["short squeeze", "short interest", "most shorted",
                    "short seller", "short covering", "days to cover"],
    "breakout":    ["52-week high", "all-time high", "breakout", "new high",
                    "technical breakout", "resistance broken", "record high"],
    "geopolitical":["tariff", "sanction", "trade war", "geopolitical",
                    "supply chain", "export ban", "china", "russia", "ukraine",
                    "energy crisis", "oil", "opec", "nato", "war", "conflict",
                    "defense contract", "pentagon"],
    "rate":        ["fed", "federal reserve", "interest rate", "rate hike",
                    "rate cut", "fomc", "powell", "inflation", "cpi", "ppi",
                    "hawkish", "dovish", "treasury yield"],
    "earn_growth": ["record earnings", "earnings growth", "eps growth",
                    "profit surge", "earnings beat", "record profit",
                    "blowout quarter", "record quarter", "beat estimates",
                    "exceeded expectations", "top-line beat"],
}

# ── Option 2: minimum keyword hits required before a catalyst tag fires ──
# Prevents a single passing mention from triggering a tag.
# Catalysts with high false-positive risk need more evidence.
CATALYST_MIN_HITS = {
    "earnings":    1,   # very common, 1 hit ok in context
    "fda":         2,   # needs 2 FDA-specific terms (e.g. "fda" + "approval")
    "legal":       2,   # needs 2 legal terms to distinguish from passing refs
    "buyout":      2,   # needs 2 M&A terms (e.g. "acquire" + "merger")
    "partnership": 2,   # "agreement" alone fires too easily
    "squeeze":     1,   # squeeze keywords are specific enough
    "breakout":    1,   # technical terms are specific
    "geopolitical":2,   # "oil" or "china" alone is too broad
    "rate":        2,   # "fed" alone appears in too many general articles
    "earn_growth": 1,   # "record earnings" is specific — 1 hit is sufficient
}

# ── Option 1: sector whitelist per catalyst ──────────────────────────────────
# Catalysts only fire for sectors where they are actually meaningful.
# None = fires for ALL sectors (no restriction).
# List = only fires if the stock's sector contains one of these strings.
CATALYST_SECTOR_WHITELIST = {
    "earnings":    None,   # universal — all companies report earnings
    "fda":         [       # only healthcare/pharma/biotech/medical devices
                    "health", "pharma", "biotech", "drug", "life science",
                    "medical", "clinical", "therapeut", "diagnostic",
                    "biolog", "genomic",
                   ],
    "legal":       None,   # any company can face litigation
    "buyout":      None,   # any company can be acquired
    "partnership": None,   # any company can sign deals
    "squeeze":     None,   # short squeeze is universal
    "breakout":    None,   # technical signal, universal
    "geopolitical":[       # sectors directly exposed to macro/trade events
                    "energy", "material", "defense", "industrial",
                    "semiconductor", "technology", "mining", "oil",
                    "chemical", "aerospace", "transport",
                   ],
    "rate":        [       # rate-sensitive sectors only
                    "financial", "bank", "real estate", "reit", "utility",
                    "insurance", "mortgage", "savings", "trust",
                   ],
    "earn_growth": None,   # universal — any company can post record earnings
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clamp(x, lo=0.0, hi=100.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


# Typical U-shaped intraday volume distribution: fraction of a full day's
# volume that trades in each 30-minute bucket from 9:30 to 16:00 ET.
# Heavy at the open and close, quiet over lunch. Using this instead of a
# linear pace stops RVOL from reading 3-5x on every stock at 9:45 AM.
_VOL_CURVE = [0.13, 0.09, 0.075, 0.065, 0.06, 0.055, 0.05,
              0.05, 0.055, 0.06, 0.07, 0.09, 0.15]
_VOL_CURVE_CUM = []
_acc = 0.0
for _b in _VOL_CURVE:
    _acc += _b
    _VOL_CURVE_CUM.append(_acc)
del _acc, _b


def expected_vol_fraction(minutes_elapsed: int) -> float:
    """Cumulative fraction of a typical day's volume expected by N minutes
    into the session, following the U-shaped curve above."""
    m = max(1, min(int(minutes_elapsed), 390))
    full_buckets = m // 30
    frac = _VOL_CURVE_CUM[full_buckets - 1] if full_buckets > 0 else 0.0
    if full_buckets < len(_VOL_CURVE):
        frac += _VOL_CURVE[full_buckets] * (m - full_buckets * 30) / 30.0
    return max(frac, 0.01)


def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price for intraday bars."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol = df["Volume"].cumsum().replace(0, np.nan)
    return (tp * df["Volume"]).cumsum() / cum_vol


# ----------------------------------------------------------------------------
# Data fetch (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def alpaca_keys():
    """Read Alpaca keys once per server session (cache_resource)."""
    try:
        k = st.secrets.get("ALPACA_API_KEY", "")
        s = st.secrets.get("ALPACA_SECRET_KEY", "")
        if k and s:
            return k, s
    except Exception:
        pass
    return None


@st.cache_resource
def ntfy_config():
    """Read ntfy settings once per server session (cache_resource)."""
    try:
        topic = st.secrets.get("NTFY_TOPIC", "")
        server = st.secrets.get("NTFY_SERVER", "https://ntfy.sh")
        if topic:
            return server.rstrip("/"), topic
    except Exception:
        pass
    return None


def send_ntfy(title: str, message: str, priority: str = "default", tags: str = "chart_with_upwards_trend"):
    """Push a notification to the phone via ntfy.sh. Fire-and-forget:
    a notification failure must never break the scan loop."""
    cfg = ntfy_config()
    if not cfg:
        return False
    server, topic = cfg
    try:
        requests.post(
            f"{server}/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            timeout=5,
        )
        return True
    except Exception as _ntfy_err:
        st.session_state["ntfy_last_error"] = str(_ntfy_err)
        return False


def _alpaca_bars(ticker: str, timeframe: str, start_iso: str, key: str, secret: str):
    """Fetch bars from Alpaca Market Data v2 (IEX feed works on free plans)."""
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {
        "timeframe": timeframe,
        "start": start_iso,
        "limit": 10000,
        "adjustment": "raw",
        "feed": "iex",
        "sort": "asc",
    }
    rows = []
    _page_limit = 50  # safety guard: max 50 pages (~500k bars) before bail
    for _ in range(_page_limit):
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("t").rename(columns={
        "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume",
    })[["Open", "High", "Low", "Close", "Volume"]]
    # Keep regular session only (09:30-16:00 ET) for intraday bars.
    # Daily bars are stamped pre-market by Alpaca, so skip the filter for them.
    if "Min" in timeframe:
        df = df.between_time("09:30", "16:00")
    return df if len(df) else None


def _bars_ok(df, n):
    return df is not None and len(df) >= n


@st.cache_data(ttl=55, show_spinner=False)
def fetch_intraday(ticker: str):
    """1-minute bars for today plus 5-minute bars for ~5 days.
    Tries Alpaca (real-time) first when keys are configured, but validates the
    result: thin tickers can come back nearly empty on the free IEX feed, so
    anything insufficient falls back to Yahoo, and the better source wins."""
    m1_a = m5_a = None
    _ak = alpaca_keys()
    if _ak:
        try:
            k, s = _ak
            today = datetime.now(timezone.utc) - timedelta(hours=24)
            week = datetime.now(timezone.utc) - timedelta(days=7)
            m1_a = _alpaca_bars(ticker, "1Min", today.strftime("%Y-%m-%dT%H:%M:%SZ"), k, s)
            m5_a = _alpaca_bars(ticker, "5Min", week.strftime("%Y-%m-%dT%H:%M:%SZ"), k, s)
            if m1_a is not None:
                last_day = m1_a.index[-1].date()
                m1_a = m1_a[m1_a.index.date == last_day]
        except Exception:
            m1_a = m5_a = None
    if _bars_ok(m1_a, 6):
        return m1_a, m5_a

    m1_y = m5_y = None
    try:
        def _do():
            tk = yf.Ticker(ticker)
            a = flatten_cols(tk.history(period="1d", interval="1m", prepost=False))
            b = flatten_cols(tk.history(period="5d", interval="5m", prepost=False))
            return a, b
        m1_y, m5_y = _yf(_do)
    except Exception:
        m1_y = m5_y = None
    if _bars_ok(m1_y, 6):
        return m1_y, m5_y

    # Neither source is sufficient: return whichever has the most bars
    a_n = len(m1_a) if m1_a is not None else 0
    y_n = len(m1_y) if m1_y is not None else 0
    return (m1_a, m5_a) if a_n >= y_n else (m1_y, m5_y)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily(ticker: str):
    """~3 months of daily bars for the RVOL baseline and previous close.
    Tries Alpaca first when keys are configured, validates, falls back to
    Yahoo, and keeps whichever source has more history."""
    d_a = None
    _ak = alpaca_keys()
    if _ak:
        try:
            k, s = _ak
            start = datetime.now(timezone.utc) - timedelta(days=100)
            d_a = _alpaca_bars(ticker, "1Day", start.strftime("%Y-%m-%dT%H:%M:%SZ"), k, s)
        except Exception:
            d_a = None
    if _bars_ok(d_a, 21):
        return d_a
    d_y = None
    try:
        d_y = _yf(lambda: flatten_cols(yf.Ticker(ticker).history(period="3mo", interval="1d")))
    except Exception:
        d_y = None
    if _bars_ok(d_y, 21):
        return d_y
    a_n = len(d_a) if d_a is not None else 0
    y_n = len(d_y) if d_y is not None else 0
    return d_a if a_n >= y_n else d_y


# ----------------------------------------------------------------------------
# Nightly screener dump (magicpro33/stock pipeline)
# ----------------------------------------------------------------------------
SCREENER_URL_DEFAULT = (
    "https://raw.githubusercontent.com/magicpro33/stock/main/data/stock_data.json.gz"
)


def _strip_heavy(d: dict) -> dict:
    """object_hook: drop numeric/scalar arrays (OHLC history blobs) as each
    object is parsed, so the 100MB+ dump doesn't blow Streamlit Cloud memory.
    Lists of dicts (record collections) and dict values are preserved; any
    residual dict-valued fields are stripped later when rows are built."""
    out = {}
    for k, v in d.items():
        if isinstance(v, list) and v and not isinstance(v[0], dict):
            continue  # numeric/scalar array, e.g. price history
        out[k] = v
    return out


@st.cache_data(ttl=21600, show_spinner="Loading nightly screener dump...")
def load_screener_dump(url: str) -> pd.DataFrame:
    """Download and parse the nightly stock_data.json.gz into a slim,
    scalar-only DataFrame indexed by ticker.

    Memory-safe path: the response is streamed, decompressed on the fly, and
    parsed record-by-record with ijson, so the multi-hundred-MB decompressed
    JSON never exists in RAM at once. This matters on Streamlit Cloud, which
    kills the container (-> 'no response from server') around ~1 GB."""
    records = []
    try:
        import ijson
        with requests.get(url, timeout=300, stream=True) as resp:
            resp.raise_for_status()
            stream = resp.raw
            if url.endswith(".gz"):
                stream = gzip.GzipFile(fileobj=resp.raw)
            for tkr, fields in ijson.kvitems(stream, ""):
                if not isinstance(fields, dict):
                    continue
                row = {}
                for k, v in fields.items():
                    if isinstance(v, decimal.Decimal):
                        row[k] = float(v)
                    elif isinstance(v, (str, int, float, bool)) or v is None:
                        row[k] = v
                row.setdefault("ticker", str(tkr))
                records.append(row)
    except Exception:
        records = []

    if not records:
        # Fallback: in-memory parse (handles list-of-records formats too)
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        blob = resp.content
        if url.endswith(".gz") or blob[:2] == b"\x1f\x8b":
            blob = gzip.decompress(blob)
        data = json.loads(blob, object_hook=_strip_heavy)
        del blob
        if isinstance(data, dict):
            inner = None
            for key in ("stocks", "data", "results", "records"):
                if key in data and isinstance(data.get(key), (list, dict)):
                    inner = data[key]
                    break
            src = inner if inner is not None else data
            if isinstance(src, dict):
                for tkr, fields in src.items():
                    if isinstance(fields, dict):
                        row = {k: v for k, v in fields.items() if not isinstance(v, dict)}
                        row.setdefault("ticker", tkr)
                        records.append(row)
            elif isinstance(src, list):
                records = [
                    {k: v for k, v in r.items() if not isinstance(v, dict)}
                    for r in src if isinstance(r, dict)
                ]
        elif isinstance(data, list):
            records = [
                {k: v for k, v in r.items() if not isinstance(v, dict)}
                for r in data if isinstance(r, dict)
            ]

    df = pd.DataFrame(records)
    # Find the ticker column
    tcol = None
    for c in df.columns:
        if str(c).lower() in ("ticker", "symbol", "sym"):
            tcol = c
            break
    if tcol is None:
        raise ValueError("Could not find a ticker/symbol column in the dump.")
    df[tcol] = df[tcol].astype(str).str.upper().str.strip()
    df = df[df[tcol].str.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}")]
    df = df.set_index(tcol)
    return df


@st.cache_data(ttl=86400, show_spinner="Building today's watchlist from nightly dump...")
def screener_watchlist(url: str, pool_size: int, min_price: float, day_key: str):
    """Pin the candidate watchlist ONCE per calendar day from the nightly dump.
    day_key (today's date) is part of the cache key, so this recomputes only
    when the date changes or the settings change. After this, all live data
    comes from the real-time feed - the dump is not touched again."""
    dump = load_screener_dump(url)
    pre = prescore_screener(dump)
    pcol = _find_col(dump.columns, "price") or _find_col(dump.columns, "close")
    if pcol is not None:
        prices = pd.to_numeric(dump[pcol], errors="coerce")
        pre = pre[prices >= float(min_price)]
    candidates = pre.sort_values(ascending=False).head(pool_size)
    meta = {"n_tickers": len(dump), "n_fields": len(dump.columns),
            "fields": [str(c) for c in dump.columns]}
    return list(candidates.index), candidates, meta


def _find_col(cols, *needles):
    """Fuzzy column lookup: first column whose lowercase name contains all
    needles. Lets this survive renames in the nightly pipeline."""
    for c in cols:
        name = str(c).lower().replace(" ", "_")
        if all(n in name for n in needles):
            return c
    return None


def prescore_screener(df: pd.DataFrame) -> pd.Series:
    """Rank the whole dump 0-100 using its own nightly signals. Scale-agnostic:
    numeric signals become percentile ranks, booleans become 0/100."""
    cols = list(df.columns)
    parts = []   # (weight, series 0-100)

    def pct(series):
        s = pd.to_numeric(series, errors="coerce")
        return s.rank(pct=True) * 100.0

    def boolean(series):
        s = series.map(lambda v: 1.0 if v in (True, 1, "1", "true", "True", "Y", "YES", "Yes") else 0.0)
        return s * 100.0

    c = _find_col(cols, "squeeze")
    if c is not None:
        parts.append((0.22, pct(df[c]) if pd.to_numeric(df[c], errors="coerce").notna().any() else boolean(df[c])))

    c = _find_col(cols, "score")
    if c is not None:
        parts.append((0.20, pct(df[c])))

    c = _find_col(cols, "rsi")
    if c is not None:
        r = pd.to_numeric(df[c], errors="coerce")
        # thrust zone 50-70 best; oversold/overbought worth less
        parts.append((0.14, r.map(lambda x: 100.0 if 50 <= x <= 70 else (60.0 if 40 <= x < 50 or 70 < x <= 80 else 20.0) if pd.notna(x) else 0.0)))

    c = _find_col(cols, "golden")
    if c is not None:
        parts.append((0.12, boolean(df[c])))

    c = _find_col(cols, "mfi")
    if c is not None:
        parts.append((0.10, boolean(df[c]) if not pd.to_numeric(df[c], errors="coerce").notna().any() else pct(df[c])))

    c = _find_col(cols, "obv")
    if c is not None:
        parts.append((0.08, pct(df[c])))

    c = _find_col(cols, "volume") or _find_col(cols, "vol")
    if c is not None:
        parts.append((0.08, pct(df[c])))

    c = _find_col(cols, "piotroski")
    if c is not None:
        parts.append((0.06, pct(df[c])))

    if not parts:
        # Nothing recognized: fall back to any numeric columns averaged
        num = df.select_dtypes(include=[np.number])
        return num.rank(pct=True).mean(axis=1).fillna(0) * 100.0

    total_w = sum(w for w, _ in parts)
    out = sum(w * s.fillna(0.0) for w, s in parts) / total_w
    return out


def sector_allows(catalyst: str, sector: str) -> bool:
    """Return True if this catalyst is valid for the stock's sector.
    Option 1: sector whitelist check. None whitelist = universal."""
    whitelist = CATALYST_SECTOR_WHITELIST.get(catalyst)
    if whitelist is None:
        return True  # no restriction
    s = sector.lower()
    return any(w in s for w in whitelist)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fuel(ticker: str) -> dict:
    """Slow-moving 'primed to move' data: short interest, float,
    insider transactions, news, 52-week levels, earnings date,
    sector, institutional ownership, and full catalyst detection."""
    out = {
        # existing
        "short_pct_float": None, "float_shares": None,
        "insider_net_buy_usd": 0.0, "insider_buys": 0, "insider_sells": 0,
        "news_count_48h": 0, "news_sentiment": 0, "latest_headline": "",
        "high_52w": None, "name": ticker,
        "target_mean": None,   # analyst mean price target
        # new catalyst fields
        "earnings_days": None,       # days until next earnings (negative = past)
        "days_to_cover": None,       # short interest / avg daily vol
        "inst_pct": None,            # institutional ownership %
        "sector": "",                # sector string
        "catalyst_tags": [],         # list of detected catalyst types
        "catalyst_score": 0.0,       # 0-100 composite catalyst sub-score
        "bimodal_event": False,      # binary event within ±3 days
        "catalyst_suppressed": [],   # tags detected but filtered out (sector/threshold)
        "earnings_surprise": None,    # last quarter EPS beat vs estimates (%)
        "data_issues": [],            # reasons any data could not be fetched
    }
    try:
        tk = _yf(lambda: yf.Ticker(ticker))
        info = {}
        try:
            info = _yf(lambda: tk.info or {})
            if not info or len(info) < 3:
                out["data_issues"].append(
                    "fundamentals: yfinance returned empty (rate-limited or unknown symbol)")
        except Exception as _ie:
            info = {}
            _msg = str(_ie)[:80]
            if "404" in _msg or "Not Found" in _msg:
                out["data_issues"].append(
                    "fundamentals: not available (ETF/fund or delisted symbol)")
            elif "429" in _msg or "rate" in _msg.lower():
                out["data_issues"].append(
                    "fundamentals: yfinance rate limit hit — retry in a few seconds")
            else:
                out["data_issues"].append(f"fundamentals: {_msg}")

        out["short_pct_float"] = info.get("shortPercentOfFloat")
        out["float_shares"] = info.get("floatShares")
        out["high_52w"] = info.get("fiftyTwoWeekHigh")
        out["name"] = info.get("shortName") or ticker
        out["target_mean"] = info.get("targetMeanPrice")
        out["sector"] = info.get("sector") or ""

        # Days to cover (short interest / avg daily volume)
        shares_short = info.get("sharesShort")
        avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")
        if shares_short and avg_vol and avg_vol > 0:
            out["days_to_cover"] = round(shares_short / avg_vol, 1)

        # Institutional ownership
        inst = info.get("heldPercentInstitutions")
        if inst is not None:
            out["inst_pct"] = round(float(inst) * 100, 1)

        # Next earnings date
        try:
            cal = _yf(lambda: tk.calendar)
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date") or cal.get("earningsDate")
                if ed is not None:
                    if isinstance(ed, (list, tuple)):
                        ed = ed[0]
                    ed_dt = pd.to_datetime(ed, errors="coerce")
                    if pd.notna(ed_dt):
                        out["earnings_days"] = (ed_dt.date() - datetime.now().date()).days
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for lbl in ["Earnings Date", "earningsDate"]:
                    if lbl in cal.index:
                        val = cal.loc[lbl].iloc[0]
                        ed_dt = pd.to_datetime(val, errors="coerce")
                        if pd.notna(ed_dt):
                            out["earnings_days"] = (ed_dt.date() - datetime.now().date()).days
                        break
        except Exception:
            pass

        # Insider transactions (last ~90 days)
        try:
            ins = _yf(lambda: tk.insider_transactions)
            if ins is not None and len(ins) > 0:
                ins = ins.copy()
                date_col = None
                for c in ["Start Date", "startDate", "Date"]:
                    if c in ins.columns:
                        date_col = c
                        break
                if date_col is not None:
                    ins[date_col] = pd.to_datetime(ins[date_col], errors="coerce")
                    cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
                    ins = ins[ins[date_col] >= cutoff]
                text_col = "Text" if "Text" in ins.columns else None
                val_col = "Value" if "Value" in ins.columns else None
                for _, row in ins.iterrows():
                    txt = str(row.get(text_col, "")).lower() if text_col else ""
                    tr = str(row.get("Transaction", "")).lower()
                    val = row.get(val_col, 0) if val_col else 0
                    try:
                        val = float(val) if pd.notna(val) else 0.0
                    except Exception:
                        val = 0.0
                    is_buy = ("purchase" in txt) or ("buy" in tr) or ("purchase" in tr)
                    is_sell = ("sale" in txt) or ("sell" in tr) or ("sale" in tr)
                    if is_buy:
                        out["insider_buys"] += 1
                        out["insider_net_buy_usd"] += abs(val)
                    elif is_sell:
                        out["insider_sells"] += 1
                        out["insider_net_buy_usd"] -= abs(val)
        except Exception:
            pass

        # News flow + catalyst detection
        try:
            news = _yf(lambda: tk.news or [])
            now = datetime.now(timezone.utc)
            sent = 0
            count = 0
            latest = ""
            cat_hits: dict[str, int] = {k: 0 for k in CATALYST_KEYWORDS}
            for item in news:
                content = item.get("content", item)
                title = content.get("title", "") or ""
                summary = content.get("summary", "") or ""
                full_text = (title + " " + summary).lower()
                pub = content.get("pubDate") or content.get("providerPublishTime")
                ts = None
                if isinstance(pub, (int, float)):
                    ts = datetime.fromtimestamp(pub, tz=timezone.utc)
                elif isinstance(pub, str):
                    try:
                        ts = pd.to_datetime(pub, utc=True).to_pydatetime()
                    except Exception:
                        ts = None
                if ts is not None and (now - ts) <= timedelta(hours=48):
                    count += 1
                    if not latest:
                        latest = title
                    sent += sum(1 for w in POSITIVE_WORDS if w in full_text)
                    sent -= sum(1 for w in NEGATIVE_WORDS if w in full_text)
                # Catalyst scan across all recent news (7 days for catalyst tagging)
                if ts is not None and (now - ts) <= timedelta(days=7):
                    for cat, kws in CATALYST_KEYWORDS.items():
                        cat_hits[cat] += sum(1 for w in kws if w in full_text)
            out["news_count_48h"] = count
            out["news_sentiment"] = sent
            out["latest_headline"] = latest
            # Tag any catalyst with at least 1 keyword hit
            # ── Earnings growth fundamental check ─────────────────────────
            # Fires when: last quarter beat estimates by 10%+ OR YoY EPS growth
            # exceeds 25%. Adds 2 keyword hits so tag fires on fundamentals
            # alone even without a matching news headline.
            try:
                _eg  = float(info.get("earningsGrowth") or 0.0)
                _esp = float(info.get("earningsSurprise") or info.get("earningsSurprisePercent") or 0.0)
                if abs(_esp) > 5: _esp = _esp / 100.0  # pct → fraction
                if _esp >= 0.10 or _eg >= 0.25:
                    cat_hits["earn_growth"] = cat_hits.get("earn_growth", 0) + 2
                out["earnings_surprise"] = round(_esp * 100, 1) if _esp else None
            except Exception:
                pass

            # Apply Option 1 (sector whitelist) + Option 2 (min keyword hits)
            sector = out.get("sector", "")
            out["catalyst_tags"] = [
                c for c, h in cat_hits.items()
                if h >= CATALYST_MIN_HITS.get(c, 1)          # Option 2: threshold
                and sector_allows(c, sector)                   # Option 1: sector
            ]
            # Track which catalysts were suppressed for transparency
            out["catalyst_suppressed"] = [
                c for c, h in cat_hits.items()
                if h > 0 and c not in out["catalyst_tags"]
            ]
        except Exception:
            pass

        # Bimodal event: earnings within ±3 days OR fda/legal catalyst in news
        ed = out["earnings_days"]
        bimodal_news = any(t in out["catalyst_tags"] for t in ("fda", "legal", "buyout"))
        out["bimodal_event"] = (
            (ed is not None and -1 <= ed <= 3) or bimodal_news
        )

        # ------------------------------------------------------------------
        # Catalyst sub-score (0-100)
        # Weights reflect how reliably each catalyst produces a sharp move
        # ------------------------------------------------------------------
        cat_sc = 0.0
        tags = out["catalyst_tags"]

        # Earnings proximity: peak score 2 days before, decays fast after
        if ed is not None:
            if 0 <= ed <= 2:
                cat_sc += 30.0   # imminent earnings = binary event
            elif ed == 3:
                cat_sc += 20.0
            elif 4 <= ed <= 7:
                cat_sc += 12.0
            elif -1 <= ed < 0:
                cat_sc += 15.0   # just reported, still in motion

        if "fda" in tags:
            cat_sc += 25.0       # FDA binary event, often explosive
        if "buyout" in tags:
            cat_sc += 25.0       # M&A premium = instant catalyst
        if "partnership" in tags:
            cat_sc += 15.0
        if "legal" in tags:
            cat_sc += 10.0       # verdict risk, can go either way
        if "squeeze" in tags:
            cat_sc += 10.0
        if "breakout" in tags:
            cat_sc += 8.0
        if "geopolitical" in tags:
            cat_sc += 6.0
        if "rate" in tags:
            cat_sc += 5.0
        if "earnings" in tags and ed is None:
            cat_sc += 8.0       # earnings headlines without a clear date
        if "earn_growth" in tags:
            cat_sc += 20.0      # fundamental beat + news = strong momentum fuel

        # Days-to-cover bonus: >= 5 days = serious squeeze setup
        dtc = out["days_to_cover"]
        if dtc and dtc >= 10:
            cat_sc += 15.0
        elif dtc and dtc >= 5:
            cat_sc += 8.0

        # Bimodal event multiplier: elevates catalyst score across the board
        if out["bimodal_event"]:
            cat_sc = min(cat_sc * 1.25, 100.0)

        out["catalyst_score"] = clamp(cat_sc)

    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------
# Signal engine
# ----------------------------------------------------------------------------
def compute_signals(ticker: str) -> dict:
    """Returns a dict of raw signals, sub-scores and the composite score."""
    s = {
        "ticker": ticker, "price": None, "chg_pct": None,
        "rvol": 0.0, "surge": 0.0, "velocity": 0.0, "accel": 0.0,
        "above_vwap": False, "vwap_cross": False, "new_hod": False,
        "near_hod": False, "rsi5": None, "macd_cross": False,
        "macd_bull": False, "ignition_score": 0.0, "fuel_score": 0.0,
        "score": 0.0, "igniting": False, "gap_reversal": False,
        "gap_pct": 0.0, "reasons": [], "error": None,
    }

    m1, m5 = fetch_intraday(ticker)
    daily = fetch_daily(ticker)
    fuel = fetch_fuel(ticker)
    s["fuel"] = fuel

    if m1 is None or len(m1) < 6 or daily is None or len(daily) < 5:
        s["error"] = "no data"
        return s

    close = m1["Close"]
    s["price"] = float(close.iloc[-1])

    # --- Date-aware previous close ------------------------------------
    # Alpaca and Yahoo differ on whether the last daily bar is today's
    # partial bar. Select the last bar strictly BEFORE the intraday
    # session date instead of blindly using iloc[-2].
    session_date = m1.index[-1].date()
    try:
        prior = daily[daily.index.date < session_date]
    except Exception:
        prior = daily.iloc[:-1] if len(daily) >= 2 else daily
    if len(prior) >= 1:
        prev_close = float(prior["Close"].iloc[-1])
    else:
        prev_close = float(daily["Close"].iloc[-1])
    s["chg_pct"] = (s["price"] / prev_close - 1.0) * 100.0 if prev_close else None

    # --- RVOL: cumulative volume today vs trailing average, curve-adjusted ---
    # Baseline uses only completed prior days (date-aware, never today's
    # partial bar). Adaptive window: up to 20 days, fewer for new listings.
    base = prior if len(prior) >= 5 else daily.iloc[:-1]
    win = min(20, len(base))
    avg_daily_vol = float(base["Volume"].iloc[-win:].mean()) if win > 0 else 0.0
    cum_vol = float(m1["Volume"].sum())
    # True elapsed minutes from bar timestamps — illiquid tickers have gaps
    # in their 1m bars, so len(m1) undercounts and would inflate RVOL.
    try:
        elapsed = int((m1.index[-1] - m1.index[0]).total_seconds() // 60) + 1
        elapsed = max(elapsed, len(m1), 1)
    except Exception:
        elapsed = max(len(m1), 1)
    expected = avg_daily_vol * expected_vol_fraction(elapsed)
    s["rvol"] = cum_vol / expected if expected > 0 else 0.0

    # --- Opening gap vs yesterday's close ---
    open_px = float(m1["Open"].iloc[0])
    s["gap_pct"] = (open_px / prev_close - 1.0) * 100.0 if prev_close else 0.0

    # --- Bar-level surge: last 3 one-minute bars vs session average bar ---
    avg_bar = float(m1["Volume"].iloc[:-3].mean()) if len(m1) > 6 else float(m1["Volume"].mean())
    last3 = float(m1["Volume"].iloc[-3:].mean())
    s["surge"] = last3 / avg_bar if avg_bar > 0 else 0.0

    # --- Velocity and acceleration on 1m closes ---
    if len(close) >= 11:
        vel_now = (close.iloc[-1] / close.iloc[-6] - 1.0) * 100.0
        vel_prev = (close.iloc[-6] / close.iloc[-11] - 1.0) * 100.0
        s["velocity"] = float(vel_now)
        s["accel"] = float(vel_now - vel_prev)
    else:
        s["velocity"] = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0

    # --- VWAP ---
    vw = vwap(m1)
    s["above_vwap"] = bool(close.iloc[-1] > vw.iloc[-1])
    if len(close) >= 6:
        was_below = bool((close.iloc[-6:-1] < vw.iloc[-6:-1]).any())
        s["vwap_cross"] = s["above_vwap"] and was_below

    # --- High of day ---
    hod = float(m1["High"].max())
    recent_high = float(m1["High"].iloc[-3:].max())
    s["new_hod"] = recent_high >= hod * 0.9999
    s["near_hod"] = s["price"] >= hod * 0.995

    # --- RSI / MACD on 5m bars ---
    if m5 is not None and len(m5) > 30:
        r = rsi(m5["Close"])
        s["rsi5"] = float(r.iloc[-1])
        macd_line, sig_line = macd(m5["Close"])
        s["macd_bull"] = bool(macd_line.iloc[-1] > sig_line.iloc[-1])
        if len(macd_line) >= 4:
            s["macd_cross"] = s["macd_bull"] and bool(
                (macd_line.iloc[-4:-1] <= sig_line.iloc[-4:-1]).any()
            )

    # ------------------------------------------------------------------
    # IGNITION sub-score (0-100)
    # ------------------------------------------------------------------
    rvol_sc = clamp(s["rvol"] / 5.0 * 100.0)
    surge_sc = clamp(s["surge"] / 4.0 * 100.0)
    vel_sc = clamp(s["velocity"] / 1.0 * 100.0) if s["velocity"] > 0 else 0.0
    acc_sc = clamp(s["accel"] / 0.7 * 100.0) if s["accel"] > 0 else 0.0
    if s["vwap_cross"]:
        vwap_sc = 100.0
    elif s["above_vwap"]:
        vwap_sc = 60.0
    else:
        vwap_sc = 0.0
    if s["new_hod"]:
        hod_sc = 100.0
    elif s["near_hod"]:
        hod_sc = 70.0
    else:
        hod_sc = 0.0
    if s["rsi5"] is None:
        rsi_sc = 0.0
    elif 55 <= s["rsi5"] <= 75:
        rsi_sc = 100.0
    elif s["rsi5"] > 75:
        rsi_sc = 70.0  # strong but stretched
    elif s["rsi5"] >= 50:
        rsi_sc = 60.0
    else:
        rsi_sc = clamp(s["rsi5"])
    if s["macd_cross"]:
        macd_sc = 100.0
    elif s["macd_bull"]:
        macd_sc = 60.0
    else:
        macd_sc = 0.0

    s["ignition_score"] = (
        0.25 * rvol_sc + 0.15 * surge_sc + 0.15 * vel_sc + 0.10 * acc_sc
        + 0.10 * vwap_sc + 0.10 * hod_sc + 0.075 * rsi_sc + 0.075 * macd_sc
    )

    # ------------------------------------------------------------------
    # FUEL sub-score (0-100)
    # ------------------------------------------------------------------
    spf = fuel.get("short_pct_float")
    short_sc = clamp((spf or 0) * 100.0 / 20.0 * 100.0) if spf else 0.0

    nb = fuel.get("insider_net_buy_usd", 0.0)
    buys = fuel.get("insider_buys", 0)
    if nb > 1_000_000:
        ins_sc = 100.0
    elif nb > 100_000:
        ins_sc = 75.0
    elif nb > 0 or buys > 0:
        ins_sc = 50.0
    elif nb < -1_000_000:
        ins_sc = 0.0
    else:
        ins_sc = 20.0

    nc = fuel.get("news_count_48h", 0)
    ns = fuel.get("news_sentiment", 0)
    news_sc = clamp(min(nc, 6) / 6.0 * 70.0 + max(ns, 0) * 10.0)

    fl = fuel.get("float_shares")
    if fl is None:
        float_sc = 30.0
    elif fl < 20e6:
        float_sc = 100.0
    elif fl < 50e6:
        float_sc = 80.0
    elif fl < 150e6:
        float_sc = 60.0
    elif fl < 500e6:
        float_sc = 40.0
    else:
        float_sc = 20.0

    h52 = fuel.get("high_52w")
    if h52 and s["price"]:
        dist = (h52 - s["price"]) / h52 * 100.0
        if dist <= 5:
            h52_sc = 100.0
        elif dist <= 15:
            h52_sc = 70.0
        elif dist <= 30:
            h52_sc = 40.0
        else:
            h52_sc = 15.0
    else:
        h52_sc = 30.0

    s["fuel_score"] = (
        0.18 * short_sc + 0.18 * ins_sc + 0.18 * news_sc
        + 0.08 * float_sc + 0.12 * h52_sc
        + 0.26 * clamp(fuel.get("catalyst_score", 0.0))
    )

    s["score"] = 0.6 * s["ignition_score"] + 0.4 * s["fuel_score"]

    # ------------------------------------------------------------------
    # Direction-aware flags. The same live footprint (RVOL + surge +
    # velocity + HOD/VWAP trigger) means different things depending on the
    # day's tape:
    #   IGNITING     = footprint fires on a flat/up day -> fresh momentum leg
    #   GAP REVERSAL = footprint fires while the stock is down hard on the
    #                  day or gapped down big -> a bounce attempt inside a
    #                  selloff (tradable, but a different and riskier trade)
    # ------------------------------------------------------------------
    footprint = (
        s["rvol"] >= 2.0
        and s["surge"] >= 2.0
        and s["velocity"] > 0
        and (s["new_hod"] or s["vwap_cross"])
    )
    chg = s["chg_pct"] if s["chg_pct"] is not None else 0.0
    gap = s.get("gap_pct", 0.0) or 0.0
    down_tape = chg <= -4.0 or gap <= -4.0
    s["igniting"] = footprint and not down_tape
    s["gap_reversal"] = footprint and down_tape

    # Human-readable reasons
    if s["rvol"] >= 2:
        s["reasons"].append(f"RVOL {s['rvol']:.1f}x")
    if s["surge"] >= 2:
        s["reasons"].append(f"vol surge {s['surge']:.1f}x")
    if s["vwap_cross"]:
        s["reasons"].append("VWAP reclaim")
    if s["new_hod"]:
        s["reasons"].append("new HOD")
    if s["macd_cross"]:
        s["reasons"].append("MACD cross")
    if s["velocity"] > 0.5:
        s["reasons"].append(f"+{s['velocity']:.2f}% / 5min")
    if s["gap_reversal"]:
        s["reasons"].insert(0, f"day {chg:+.1f}% - bounce attempt")
    if spf and spf >= 0.15:
        s["reasons"].append(f"short float {spf*100:.0f}%")
    if nb > 0:
        s["reasons"].append("insider buying")
    if nc >= 2:
        s["reasons"].append(f"{nc} headlines 48h")

    # Catalyst tags as readable reasons
    tag_labels = {
        "earnings": f"earnings in {fuel.get('earnings_days')}d" if fuel.get("earnings_days") is not None else "earnings news",
        "fda": "FDA event",
        "legal": "legal catalyst",
        "buyout": "M&A/buyout",
        "partnership": "partnership",
        "squeeze": "squeeze setup",
        "breakout": "breakout news",
        "geopolitical": "geopolitical",
        "rate": "rate catalyst",
    }
    for tag in (fuel.get("catalyst_tags") or []):
        label = tag_labels.get(tag)
        if label:
            s["reasons"].append(label)
    if fuel.get("bimodal_event"):
        s["reasons"].append("BIMODAL EVENT")
    dtc = fuel.get("days_to_cover")
    if dtc and dtc >= 10:
        s["reasons"].append(f"squeeze extreme ({dtc}d)")
    elif dtc and dtc >= 7:
        s["reasons"].append(f"short fuel high ({dtc}d)")
    elif dtc and dtc >= 5:
        s["reasons"].append(f"short fuel mod ({dtc}d)")

    return s


def scan_tickers_parallel(ticker_list, progress_cb=None, max_workers=12):
    """Scan many tickers concurrently using a thread pool.

    compute_signals is a pure function whose results are cached by
    @st.cache_data — which is thread-safe — so running it across threads
    is safe. The slow part of each call is network I/O (Alpaca + yfinance),
    which releases the GIL, so threads give a near-linear speedup for the
    network-bound scan. Cache hits return instantly.

    progress_cb(done_count, total, last_ticker) is called as each result
    arrives so the UI can show live progress. Results preserve no order
    here — the caller sorts by score afterwards anyway.
    """
    total = len(ticker_list)
    if total == 0:
        return []
    results = []
    # Cap workers so we don't hammer the API past its rate limits.
    workers = min(max_workers, total)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(compute_signals, t): t for t in ticker_list}
        done = 0
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({
                    "ticker": t, "error": str(e), "score": 0.0,
                    "igniting": False, "gap_reversal": False,
                    "reasons": [], "fuel": {},
                })
            done += 1
            if progress_cb:
                progress_cb(done, total, t)
    return results



def render_ignition_scanner_tab():
    """Scan Hub tab — IGNITION momentum scanner.

    Controls render immediately. The live scan (Alpaca/Yahoo per ticker)
    runs only after you hit **Scan**. Results stay until the next Scan.
    """
    # Card/banner CSS only — skip page/tab chrome that would restyle Money Weather.
    _css = CUSTOM_CSS
    for _drop in (
        "html, body, [class*=\"css\"]  { font-family: 'Plus Jakarta Sans', sans-serif; }\n.stApp                       { background: #07111f; }\n\n"
        "h1, h2, h3 { font-family: 'Rajdhani', sans-serif !important;\n"
        "             letter-spacing: 0.8px; color: #ffffff; }\n",
    ):
        _css = _css.replace(_drop, "")
    st.markdown(_css, unsafe_allow_html=True)

    screener_mode = False
    st.markdown(
        "<div style='padding:12px 4px 8px;'>"
        "<div style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;"
        "color:#f5a623;letter-spacing:1px'><a href='https://aiupscalellc.netlify.app/' "
        "target='_blank' rel='noopener' style='color:inherit;text-decoration:none'"
        ">AI UPSCALE</a></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("<div class='sidebar-section'>Watchlist</div>", unsafe_allow_html=True)

    # ── Top-N results slider (defined here — used by all watchlist modes) ──
    top_n = st.slider(
        "Results to show", 5, 30, 10,
        key="ig_top_n",
        help="How many stocks to display after the scan, ranked by Score. "
             "Set to 5 for the hottest only, 30 for a full board.",
    )

    # ── Scan ALL presets toggle ──────────────────────────────────────────
    all_presets_mode = st.toggle(
        "Scan ALL presets",
        value=False,
        key="ig_all_presets_mode",
        help="Combines every ticker across all 8 sector presets into one scan "
             "and shows the top N results by Score. Great for finding the single "
             "best opportunity across your entire watchlist right now.",
    )

    source = st.radio(
        "Watchlist source",
        ["Preset / Custom", "Scan over 5000 stocks"],
        index=0,
        disabled=all_presets_mode,
        key="ig_source",
        help="Preset / Custom: scan a hand-picked sector list. "
             "Nightly Screener Top 10: uses your nightly dump to pre-rank candidates. "
             "Disabled when Scan ALL presets is on.",
    )

    screener_mode = False
    if all_presets_mode:
        # Build deduplicated mega-list across all 8 presets, excluding ETFs
        seen, all_combined = set(), []
        for tkrs in PRESETS.values():
            for t in [x.strip().upper() for x in tkrs.split(",") if x.strip()]:
                if t not in seen and t not in ALL_PRESETS_ETF_EXCLUDE:
                    seen.add(t)
                    all_combined.append(t)
        tickers = all_combined
        st.caption(
            f"ALL PRESETS: {len(tickers)} stocks across "
            f"{len(PRESETS)} sectors (ETFs excluded). "
            f"Top {top_n} shown after scan."
        )
    elif source == "Scan over 5000 stocks":
        screener_mode = True
        screener_url = st.text_input("Dump URL", value=SCREENER_URL_DEFAULT,
                                     key="ig_screener_url")
        pool_size = st.slider(
            "Candidate pool (pre-ranked from dump)", 20, 80, 40,
            key="ig_pool_size",
            help="The whole dump is pre-ranked by its own nightly signals; the top "
                 "N candidates get the live ignition scan.",
        )
        min_price = st.number_input(
            "Min price filter", value=1.0, step=0.5,
            key="ig_min_price",
            help="Excludes stocks below this price from the candidate pool.",
        )
        tickers = []
        st.caption("Dump pre-rank + live scan run only when you hit Scan.")
    else:
        preset_names = ["Custom"] + list(PRESETS.keys())
        # Pick a random sector on very first load (not on every rerun)
        if "ig_initial_preset_idx" not in st.session_state:
            st.session_state["ig_initial_preset_idx"] = random.randint(1, len(preset_names) - 1)
        preset = st.selectbox(
            "Sector preset",
            preset_names,
            index=st.session_state["ig_initial_preset_idx"],
            help="Select a sector to scan, or choose Custom to enter your own tickers.",
            key="ig_sector_preset",
        )
        # Update stored index so switching presets doesn't snap back on rerun
        st.session_state["ig_initial_preset_idx"] = preset_names.index(preset)
        if preset == "Custom":
            tickers_raw = st.text_area(
                "Tickers (comma separated)",
                value="CCJ,SMR,NNE,OKLO,LEU,EU,UUUU,UROY,GEV,CEG",
                height=90,
                key="ig_custom_tickers",
            )
            tickers = [t.strip().upper() for t in re.split(r"[,\s]+", tickers_raw) if t.strip()][:250]
        else:
            tickers = [t.strip().upper() for t in PRESETS[preset].split(",") if t.strip()]

    st.markdown("---")
    st.markdown("<div class='sidebar-section'>Scanner controls</div>", unsafe_allow_html=True)

    # top_n defined earlier in Watchlist section

    alert_threshold = st.slider(
        "Alert score threshold", 40, 95, 65,
        key="ig_alert_th",
        help="A ticker triggers an alert (feed entry + phone push) the first time "
             "its overall Score crosses this line each day. IGNITING alerts fire "
             "regardless of this threshold when all live conditions confirm at once.",
    )
    view_mode = "Compact (phone)"
    show_all_cols = False
    st.markdown("---")
    st.markdown("<div class='sidebar-section'>Notifications</div>", unsafe_allow_html=True)
    popup_alerts_on = st.toggle(
        "Show popup alerts",
        value=False,
        key="ig_popup_alerts_on",
        help="When off (default) the scanner runs silently after each scan. "
             "Turn on to see in-app toast popups for every alert that fires.",
    )
    if ntfy_config():
        notify_on = st.toggle("Phone notifications (ntfy)", value=True, key="ig_notify")
        if st.button("Send test notification", key="ig_ntfy_test"):
            ok_test = send_ntfy("IGNITION test", "If you can read this, alerts are wired up.", tags="white_check_mark")
            if ok_test:
                st.success("Test sent - check your phone.")
            else:
                st.error("Send failed - check NTFY_TOPIC in secrets.")
    else:
        notify_on = False
        st.info("Phone alerts off: add NTFY_TOPIC to secrets to enable ntfy push.")

    st.markdown("---")
    st.markdown("<div class='sidebar-section'>Data feed</div>", unsafe_allow_html=True)
    if alpaca_keys():
        st.success("Data feed: Alpaca (real-time IEX)")
    else:
        st.info("Data feed: Yahoo (may lag ~15 min). Add Alpaca keys in secrets for real-time.")
    st.caption(
        "This tool detects momentum early; it does not predict the future. "
        "Not financial advice."
    )

    if "ig_alerts" not in st.session_state:
        st.session_state["ig_alerts"] = []
    if "ig_alerted" not in st.session_state:
        st.session_state["ig_alerted"] = set()

    # ----------------------------------------------------------------------------
    # Header
    # ----------------------------------------------------------------------------
    APP_VERSION = "v3.10"
    last_scan = st.session_state.get("ig_last_scan_time", "--:--:--")
    st.markdown(
        "<div style='display:flex;align-items:center;gap:14px;margin-bottom:2px'>"
        "<svg width='44' height='44' viewBox='0 0 72 72' fill='none' "
        "xmlns='http://www.w3.org/2000/svg'>"
        "<polyline points='8,36 18,36 24,16 30,56 38,26 44,46 50,36 64,36' "
        "stroke='#f5a623' stroke-width='4' stroke-linecap='round' "
        "stroke-linejoin='round' fill='none'/>"
        "<circle cx='36' cy='36' r='4' fill='#f5a623'/>"
        "</svg>"
        f"<div>"
        f"<div style='font-family:Space Mono,monospace;font-size:12px;color:#7a9ab8'>{APP_VERSION}</div>"
        f"<div style='font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;"
        f"color:#ffffff;letter-spacing:1px;line-height:1.1'>STOCKS IN THE MONEY ZONE</div>"
        f"</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------------
    # Scan — only when you hit the Scan button
    # ----------------------------------------------------------------------------

    col_refresh, col_status = st.columns([1, 4])
    with col_refresh:
        refresh_clicked = st.button(
            "🚀 Scan",
            type="primary",
            key="ig_scan",
            help="Run the live ignition scan. Nothing is fetched until you hit this.",
            use_container_width=True,
        )
    with col_status:
        scan_count = len(st.session_state.get("ig_last_results", []))
        if scan_count:
            st.markdown(
                f"<div style='padding:8px 0;font-family:Space Mono,monospace;font-size:12px;"
                f"color:#7a9ab8'>{scan_count} tickers scanned</div>",
                unsafe_allow_html=True,
            )

    should_scan = bool(refresh_clicked)

    if should_scan:
        if screener_mode:
            try:
                day_key = datetime.now().strftime("%Y-%m-%d")
                tickers, candidates, meta = screener_watchlist(
                    screener_url, pool_size, float(min_price), day_key
                )
                st.session_state["ig_screener_pre"] = candidates
                st.caption(
                    f"Watchlist pinned for {day_key} from dump "
                    f"({meta['n_tickers']} tickers, {meta['n_fields']} fields). "
                    f"Pool: {len(tickers)}."
                )
                with st.expander("Detected dump fields"):
                    st.write(", ".join(meta["fields"]))
            except Exception as e:
                st.error(f"Could not load nightly dump: {e}")
                tickers = []
        current_watchlist_key = ",".join(sorted(tickers))
        progress = st.progress(0.0, text="")
        _scan_label = st.empty()

        def _scan_progress(done, total, last_t):
            progress.progress(done / max(total, 1))
            _scan_label.markdown(
                f"<span style='color:#cc0000;font-family:Space Mono,monospace;"
                f"font-size:13px;font-weight:500'>Scanning… ({done}/{total})</span>",
                unsafe_allow_html=True,
            )

        results = scan_tickers_parallel(tickers, progress_cb=_scan_progress)

        progress.empty()
        _scan_label.empty()
        st.session_state["ig_last_results"] = results
        st.session_state["ig_last_scan_time"] = datetime.now().strftime("%H:%M:%S")
        st.session_state["ig_last_watchlist_key"] = current_watchlist_key
    else:
        results = st.session_state.get("ig_last_results") or []

    ok = [r for r in results if not r.get("error")]
    failed = [r["ticker"] for r in results if r.get("error")]
    ok.sort(key=lambda r: r["score"], reverse=True)

    # Apply top-N limit (screener mode, all-presets mode, and normal mode all use it)
    never_scanned = "ig_last_results" not in st.session_state
    if results:
        if screener_mode:
            scanned_n = len(ok)
            pre_ranks = st.session_state.get("ig_screener_pre")
            if pre_ranks is not None:
                ok = [dict(r) for r in ok[:top_n]]
                for r in ok:
                    if r["ticker"] in pre_ranks.index:
                        r["pre_rank"] = float(pre_ranks[r["ticker"]])
            else:
                ok = ok[:top_n]
            st.caption(
                f"Nightly Screener mode: pre-ranked the full dump, live-scanned the top "
                f"{scanned_n} candidates, showing the top {top_n} by ignition score."
            )
        else:
            ok = ok[:top_n]
            if all_presets_mode:
                st.caption(
                    f"All Presets: scanned {len(results)} stocks across "
                    f"{len(PRESETS)} sectors (ETFs excluded), showing top {top_n} by Score."
                )

    # Register new alerts (only on a fresh scan, not when reusing cached results)
    session_key = datetime.now().strftime("%Y%m%d")
    for r in (ok if should_scan else []):
        key = f"{session_key}:{r['ticker']}"
        if (r["igniting"] or r["gap_reversal"] or r["score"] >= alert_threshold) and key not in st.session_state["ig_alerted"]:
            st.session_state["ig_alerted"].add(key)
            kind = "igniting" if r["igniting"] else ("reversal" if r["gap_reversal"] else "alert")
            st.session_state["ig_alerts"].insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "ticker": r["ticker"],
                "score": round(r["score"]),
                "price": r["price"],
                "why": ", ".join(r["reasons"][:4]) or "score threshold",
                "kind": kind,
            })
            if popup_alerts_on:
                st.toast(f"{r['ticker']} {kind.upper()} {r['score']:.0f} — {', '.join(r['reasons'][:3])}")
            if notify_on:
                why = ", ".join(r["reasons"][:4]) or "score threshold"
                price_txt = f" @ ${r['price']:.2f}" if r.get("price") else ""
                if kind == "igniting":
                    send_ntfy(
                        f"IGNITING: {r['ticker']}{price_txt}",
                        f"Score {r['score']:.0f} | {why}",
                        priority="urgent", tags="fire",
                    )
                elif kind == "reversal":
                    send_ntfy(
                        f"GAP REVERSAL: {r['ticker']}{price_txt}",
                        f"Bounce attempt inside a selloff. Score {r['score']:.0f} | {why}",
                        priority="high", tags="warning",
                    )
                else:
                    send_ntfy(
                        f"{r['ticker']} alert{price_txt}",
                        f"Score {r['score']:.0f} | {why}",
                        priority="default",
                    )


    # ----------------------------------------------------------------------------
    # Stock Analyzer helpers (module-level to avoid re-definition on every rerun)
    # ----------------------------------------------------------------------------
    def az_pill(label, good):
        bg = "#0d2215" if good is True else ("#220d0d" if good is False else "#1a1500")
        col = "#4dd880" if good is True else ("#ff4444" if good is False else "#d0b040")
        bc  = "#1e6b35" if good is True else ("#a03535" if good is False else "#907020")
        return (f"<span style='display:inline-block;background:{bg};color:{col};"
                f"border:1px solid {bc};border-radius:4px;font-family:Space Mono,monospace;"
                f"font-size:11px;font-weight:500;padding:2px 9px;margin:2px 4px 2px 0'>{label}</span>")


    def az_tag(v, hi, lo, fmt="{:.2f}", suffix=""):
        try:
            fv = float(v)
            col = "#4dd880" if fv >= hi else ("#d0b040" if fv >= lo else "#ff4444")
            return f"<span style='font-family:Space Mono,monospace;color:{col}'>{fmt.format(fv)}{suffix}</span>"
        except Exception:
            return "--"


    def mrow(label, tooltip, value):
        return (f"<tr style='border-bottom:1px solid #122540'>"
                f"<td style='padding:8px 10px;font-size:13px;color:#b0c8e8;width:40%'>"
                f"<span title='{tooltip}' style='cursor:help;border-bottom:1px dashed #1e3a5f'>{label}</span></td>"
                f"<td style='padding:8px 10px;font-size:13px;font-family:Space Mono,monospace;color:#ffffff'>{value}</td>"
                f"</tr>")


    def az_section(title):
        st.markdown(f"<div style='font-family:Rajdhani,sans-serif;font-weight:700;font-size:13px;"
                    f"letter-spacing:1px;text-transform:uppercase;color:#f5a623;"
                    f"border-bottom:1px solid #1e3a5f;padding-bottom:4px;margin:14px 0 8px'>{title}</div>",
                    unsafe_allow_html=True)


    def pct_color(v):
        if v is None: return "--"
        col = "#4dd880" if v >= 0 else "#ff4444"
        return f"<span style='font-family:Space Mono,monospace;color:{col}'>{'+' if v >= 0 else ''}{v:.2f}%</span>"


    def render_eps_trend(eps_history, eps_forward=None, why=""):
        """SVG bar chart of quarterly EPS (actual vs estimate) plus forward
        analyst projections. eps_history = oldest→newest historicals; eps_forward
        = forward consensus estimates (next qtr, qtr after, this yr, next yr)."""
        eps_forward = eps_forward or []
        if not eps_history and not eps_forward:
            st.caption(f"Quarterly EPS history unavailable — {why or 'no earnings records returned (thin coverage, ETF, or rate limit)'}")
            return

        # Build a combined timeline: historicals (bars) then forward (projected bars)
        # Each entry: {label, value, kind} where kind ∈ {actual, estimate_miss, projected}
        timeline = []
        for q in eps_history:
            if q["actual"] is not None:
                beat = q["surprise"]
                kind = "beat" if (beat is None or beat >= 0) else "miss"
                timeline.append({"label": (q["quarter"].split()[0] if q["quarter"] else ""),
                                 "value": q["actual"], "kind": kind, "est": q["estimate"]})
        # Forward — only quarterly periods on the chart (years would distort scale)
        fwd_quarters = [f for f in eps_forward if "Qtr" in f["period"]]
        for f in fwd_quarters:
            short = "Next Q" if f["period"] == "Next Qtr" else "Q+2"
            timeline.append({"label": short, "value": f["estimate"],
                             "kind": "projected", "est": None})

        if not timeline:
            st.caption(f"Quarterly EPS history unavailable — {why or 'no earnings records returned (thin coverage, ETF, or rate limit)'}")
            return

        n = len(timeline)
        W, H = 320, 110
        pad_l, pad_r, pad_t, pad_b = 8, 8, 10, 22
        plot_w = W - pad_l - pad_r
        plot_h = H - pad_t - pad_b
        slot   = plot_w / n
        bar_w  = min(slot * 0.55, 26)

        vals = [t["value"] for t in timeline if t["value"] is not None]
        vals += [t["est"] for t in timeline if t.get("est") is not None]
        if not vals:
            st.caption(f"Quarterly EPS history unavailable — {why or 'no earnings records returned (thin coverage, ETF, or rate limit)'}")
            return
        vmax = max(max(vals), 0.0)
        vmin = min(min(vals), 0.0)
        vrange = (vmax - vmin) or 1.0

        def y_of(v):
            return pad_t + plot_h * (1 - (v - vmin) / vrange)

        zero_y = y_of(0.0)
        bars_svg, labels_svg = [], []
        for i, t in enumerate(timeline):
            cx = pad_l + slot * i + slot / 2
            # estimate tick for historicals
            if t.get("est") is not None:
                ey = y_of(t["est"])
                bars_svg.append(
                    f"<line x1='{cx-bar_w/2-2:.1f}' y1='{ey:.1f}' x2='{cx+bar_w/2+2:.1f}' "
                    f"y2='{ey:.1f}' stroke='#7a9ab8' stroke-width='1.5' stroke-dasharray='2,2'/>"
                )
            if t["value"] is not None:
                ay = y_of(t["value"])
                top = min(ay, zero_y)
                height = abs(ay - zero_y)
                if t["kind"] == "beat":
                    fill = "fill='#4dd880'"
                elif t["kind"] == "miss":
                    fill = "fill='#ff4444'"
                else:  # projected — amber, hatched/translucent to signal "future"
                    fill = "fill='#f5a623' fill-opacity='0.55' stroke='#f5a623' stroke-width='1' stroke-dasharray='3,2'"
                bars_svg.append(
                    f"<rect x='{cx-bar_w/2:.1f}' y='{top:.1f}' width='{bar_w:.1f}' "
                    f"height='{max(height,1):.1f}' rx='2' {fill}/>"
                )
            lbl_col = "#f5a623" if t["kind"] == "projected" else "#7a9ab8"
            labels_svg.append(
                f"<text x='{cx:.1f}' y='{H-6}' text-anchor='middle' "
                f"font-family='Space Mono,monospace' font-size='7' fill='{lbl_col}'>{t['label']}</text>"
            )

        zero_line = (
            f"<line x1='{pad_l}' y1='{zero_y:.1f}' x2='{W-pad_r}' y2='{zero_y:.1f}' "
            f"stroke='#1e3a5f' stroke-width='1'/>"
        )
        svg = (
            f"<svg width='100%' height='{H}' viewBox='0 0 {W} {H}' "
            f"preserveAspectRatio='xMidYMid meet' style='max-width:360px'>"
            f"{zero_line}{''.join(bars_svg)}{''.join(labels_svg)}</svg>"
        )

        legend = (
            "<div style='display:flex;gap:12px;flex-wrap:wrap;font-family:Space Mono,monospace;"
            "font-size:10px;color:#7a9ab8;margin:2px 0 6px'>"
            "<span><span style='color:#4dd880'>█</span> beat</span>"
            "<span><span style='color:#ff4444'>█</span> miss</span>"
            "<span><span style='color:#7a9ab8'>╌</span> estimate</span>"
            "<span><span style='color:#f5a623'>▦</span> projected</span>"
            "</div>"
        )
        st.markdown(legend + svg, unsafe_allow_html=True)

        # Historical table — newest first
        if eps_history:
            rows = []
            for q in reversed(eps_history):
                a = f"${q['actual']:.2f}"   if q["actual"]   is not None else "--"
                e = f"${q['estimate']:.2f}" if q["estimate"] is not None else "--"
                if q["surprise"] is not None:
                    sc = "#4dd880" if q["surprise"] >= 0 else "#ff4444"
                    s = f"<span style='color:{sc}'>{q['surprise']:+.1f}%</span>"
                else:
                    s = "--"
                rows.append(
                    f"<tr>"
                    f"<td style='padding:3px 8px;font-family:Space Mono,monospace;font-size:11px;color:#b0c8e8'>{q['quarter']}</td>"
                    f"<td style='padding:3px 8px;font-family:Space Mono,monospace;font-size:11px;color:#ffffff;text-align:right'>{a}</td>"
                    f"<td style='padding:3px 8px;font-family:Space Mono,monospace;font-size:11px;color:#7a9ab8;text-align:right'>{e}</td>"
                    f"<td style='padding:3px 8px;font-family:Space Mono,monospace;font-size:11px;text-align:right'>{s}</td>"
                    f"</tr>"
                )
            st.markdown(
                "<table style='width:100%;border-collapse:collapse'>"
                "<thead><tr>"
                "<th style='padding:3px 8px;font-family:Space Mono,monospace;font-size:9px;color:#4a6a8a;text-align:left;text-transform:uppercase'>Quarter</th>"
                "<th style='padding:3px 8px;font-family:Space Mono,monospace;font-size:9px;color:#4a6a8a;text-align:right;text-transform:uppercase'>Actual</th>"
                "<th style='padding:3px 8px;font-family:Space Mono,monospace;font-size:9px;color:#4a6a8a;text-align:right;text-transform:uppercase'>Est</th>"
                "<th style='padding:3px 8px;font-family:Space Mono,monospace;font-size:9px;color:#4a6a8a;text-align:right;text-transform:uppercase'>Surprise</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
                unsafe_allow_html=True,
            )

        # Forward projections table — the analyst consensus going out
        if eps_forward:
            st.markdown(
                "<div style='font-family:Space Mono,monospace;font-size:10px;color:#f5a623;"
                "letter-spacing:.5px;text-transform:uppercase;margin:10px 0 2px'>"
                "Forward Projections (analyst consensus)</div>",
                unsafe_allow_html=True,
            )
            frows = []
            for f in eps_forward:
                est = f"${f['estimate']:.2f}"
                na  = f"{f['n_analysts']}" if f.get("n_analysts") else "--"
                ecol = "#4dd880" if f["estimate"] >= 0 else "#ff4444"
                frows.append(
                    f"<tr>"
                    f"<td style='padding:3px 8px;font-family:Space Mono,monospace;font-size:11px;color:#f5a623'>{f['period']}</td>"
                    f"<td style='padding:3px 8px;font-family:Space Mono,monospace;font-size:11px;color:{ecol};text-align:right;font-weight:500'>{est}</td>"
                    f"<td style='padding:3px 8px;font-family:Space Mono,monospace;font-size:11px;color:#7a9ab8;text-align:right'>{na}</td>"
                    f"</tr>"
                )
            st.markdown(
                "<table style='width:100%;border-collapse:collapse'>"
                "<thead><tr>"
                "<th style='padding:3px 8px;font-family:Space Mono,monospace;font-size:9px;color:#4a6a8a;text-align:left;text-transform:uppercase'>Period</th>"
                "<th style='padding:3px 8px;font-family:Space Mono,monospace;font-size:9px;color:#4a6a8a;text-align:right;text-transform:uppercase'>Est EPS</th>"
                "<th style='padding:3px 8px;font-family:Space Mono,monospace;font-size:9px;color:#4a6a8a;text-align:right;text-transform:uppercase'>Analysts</th>"
                "</tr></thead><tbody>" + "".join(frows) + "</tbody></table>",
                unsafe_allow_html=True,
            )


    def render_dividend_info(info):
        """Render dividend details: payout per share, yield %, ex-date, pay date,
        future payout date, frequency, and payout ratio. info is yfinance .info."""
        rate     = info.get("dividendRate")              # annual $ per share
        yield_   = info.get("dividendYield")             # decimal fraction
        last_div = info.get("lastDividendValue")         # last single payment $
        ex_date  = info.get("exDividendDate")            # unix ts — next ex-date
        # yfinance dividendDate is the UPCOMING pay date; lastDividendDate is historical
        next_pay = info.get("dividendDate")
        last_pay = info.get("lastDividendDate")
        payout   = info.get("payoutRatio")               # decimal fraction
        freq_raw = info.get("dividendFrequency")         # rarely present

        # No dividend at all
        if not rate and not yield_ and not last_div:
            if info.get("_data_issues"):
                st.caption(f"Dividend data unavailable — {'; '.join(info['_data_issues'])}")
            else:
                st.caption("This stock does not currently pay a dividend.")
            return

        def _fmt_date(ts):
            if ts is None:
                return None
            try:
                if isinstance(ts, (int, float)):
                    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d, %Y")
                d = pd.to_datetime(ts, errors="coerce")
                return d.strftime("%b %d, %Y") if pd.notna(d) else None
            except Exception:
                return None

        def _to_dt(ts):
            if ts is None:
                return None
            try:
                if isinstance(ts, (int, float)):
                    return datetime.fromtimestamp(ts, tz=timezone.utc)
                d = pd.to_datetime(ts, errors="coerce")
                return d.to_pydatetime() if pd.notna(d) else None
            except Exception:
                return None

        # Normalise yield — yfinance returns it as a fraction (0.035) OR percent (3.5)
        yld_pct = None
        if yield_ is not None:
            yld_pct = yield_ * 100 if yield_ < 1 else yield_

        # Infer frequency from annual rate / single payment
        freq_label = "--"
        per_payment = last_div
        freq_months = None   # months between payments, for projecting the next date
        if freq_raw:
            freq_label = str(freq_raw).title()
        elif rate and last_div and last_div > 0:
            ratio = round(rate / last_div)
            freq_label = {1: "Annual", 2: "Semi-annual", 4: "Quarterly",
                          12: "Monthly"}.get(ratio, f"{ratio}×/yr")
            freq_months = {1: 12, 2: 6, 4: 3, 12: 1}.get(ratio)

        # ── Determine the next (future) payout date ──────────────────────
        # Priority: 1) yfinance dividendDate if it's in the future,
        #           2) project from ex-date + frequency,
        #           3) project from last pay date + frequency
        now = datetime.now(timezone.utc)
        next_pay_dt   = _to_dt(next_pay)
        next_pay_est  = False
        if next_pay_dt is None or next_pay_dt < now:
            # Project forward from the ex-dividend date (pay usually ~2-4 wks after)
            ex_dt = _to_dt(ex_date)
            if ex_dt is not None and freq_months:
                projected = ex_dt
                # Roll forward until it's in the future
                while projected < now:
                    m = projected.month - 1 + freq_months
                    projected = projected.replace(
                        year=projected.year + m // 12, month=m % 12 + 1
                    )
                # Pay date typically lands ~3 weeks after ex-date
                next_pay_dt  = projected + timedelta(days=21)
                next_pay_est = True
            elif last_pay is not None and freq_months:
                lp = _to_dt(last_pay)
                if lp is not None:
                    projected = lp
                    while projected < now:
                        m = projected.month - 1 + freq_months
                        projected = projected.replace(
                            year=projected.year + m // 12, month=m % 12 + 1
                        )
                    next_pay_dt  = projected
                    next_pay_est = True

        rows = []
        if rate is not None:
            rows.append(mrow("Annual Payout", "Total dollars paid per share per year.",
                             f"<span style='font-family:Space Mono,monospace;color:#4dd880;font-weight:500'>${rate:.2f}</span>/share"))
        if per_payment is not None:
            rows.append(mrow("Per Payment", "Dollar amount of each individual dividend payment.",
                             f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>${per_payment:.2f}</span>"))
        if yld_pct is not None:
            yc = "#4dd880" if yld_pct >= 4 else ("#d0b040" if yld_pct >= 2 else "#7a9ab8")
            rows.append(mrow("Dividend Yield", "Annual payout as % of current price. 4%+ = high yield.",
                             f"<span style='font-family:Space Mono,monospace;color:{yc};font-weight:500'>{yld_pct:.2f}%</span>"))
        if freq_label != "--":
            rows.append(mrow("Frequency", "How often the dividend is paid.",
                             f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{freq_label}</span>"))
        _ex = _fmt_date(ex_date)
        if _ex:
            rows.append(mrow("Ex-Dividend Date", "Buy BEFORE this date to receive the next dividend. Sell on/after and still get paid.",
                             f"<span style='font-family:Space Mono,monospace;color:#f5a623'>{_ex}</span>"))
        # ── Future payout date ───────────────────────────────────────────
        if next_pay_dt is not None:
            _np = next_pay_dt.strftime("%b %d, %Y")
            est_tag = " <span style='font-size:9px;color:#7a9ab8'>(est.)</span>" if next_pay_est else ""
            tip = ("Projected next payment date based on ex-dividend date and payment frequency."
                   if next_pay_est else
                   "Next scheduled dividend payment date from the company.")
            rows.append(mrow("Next Payout Date", tip,
                             f"<span style='font-family:Space Mono,monospace;color:#4dd880;font-weight:500'>{_np}</span>{est_tag}"))
        _lp = _fmt_date(last_pay)
        if _lp:
            rows.append(mrow("Last Pay Date", "When the most recent dividend was actually paid out.",
                             f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{_lp}</span>"))
        if payout is not None:
            pr_pct = payout * 100 if payout < 1 else payout
            pc = "#4dd880" if pr_pct < 60 else ("#d0b040" if pr_pct < 90 else "#ff4444")
            rows.append(mrow("Payout Ratio", "% of earnings paid as dividends. Under 60% = sustainable. Over 90% = at risk.",
                             f"<span style='font-family:Space Mono,monospace;color:{pc}'>{pr_pct:.1f}%</span>"))

        if rows:
            st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(rows)}</tbody></table>",
                        unsafe_allow_html=True)
        else:
            st.caption("Dividend data unavailable for this ticker.")


    @st.cache_data(ttl=900, show_spinner="Loading analysis…")
    def fetch_analyzer(ticker):
        """Fetch full analyst data for the Stock Analyzer section.

        Data priority chain:
          1. Alpaca daily bars for price history (real-time, if keys configured)
          2. yfinance for fundamentals, info fields, and history fallback
          3. Nightly scan dump (stock_data.json.gz) for any fields still missing
        """
        # ── Step 1: Price history — Alpaca first, Yahoo fallback ─────────
        hist = pd.DataFrame()
        _ak = alpaca_keys()
        if _ak:
            try:
                k, s = _ak
                start = (datetime.now(timezone.utc) - timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%SZ")
                h_alp = _alpaca_bars(ticker, "1Day", start, k, s)
                if h_alp is not None and len(h_alp) >= 50:
                    hist = h_alp
            except Exception:
                pass

        # Yahoo fallback for history
        tk = _yf(lambda: yf.Ticker(ticker))
        if hist.empty:
            try:
                h = _yf(lambda: tk.history(period="1y", interval="1d"))
                if h is not None and not h.empty:
                    if isinstance(h.columns, pd.MultiIndex):
                        h.columns = h.columns.get_level_values(0)
                    hist = h
            except Exception:
                pass

        # ── Step 2: Fundamentals from yfinance ───────────────────────────
        info = {}
        _issues = []
        if hist.empty:
            _issues.append("price history: no bars from Alpaca or Yahoo (new listing, delisted, or feed outage)")
        try:
            info = _yf(lambda: tk.info or {})
            if not info or len(info) < 3:
                _issues.append("fundamentals: yfinance returned empty (rate-limited or symbol has no profile)")
        except Exception as _ie:
            _msg = str(_ie)[:80]
            if "404" in _msg or "Not Found" in _msg:
                _issues.append("fundamentals: not published for this symbol (ETFs/funds have no fundamentals)")
            elif "429" in _msg or "rate" in _msg.lower():
                _issues.append("fundamentals: yfinance rate limit — wait a few seconds and press Refresh")
            else:
                _issues.append(f"fundamentals: {_msg}")

        # ── Step 3: Nightly scan dump fallback for missing fields ────────
        # Fields we want that yfinance often omits for smaller/thinner tickers
        SCAN_FIELD_MAP = {
            # scan dump key → info key we'd populate
            "rsi":            "_scan_rsi",
            "short_squeeze":  "_scan_squeeze_score",
            "golden_cross":   "_scan_golden_cross",
            "mfi_sweet_spot": "_scan_mfi",
            "obv":            "_scan_obv",
            "score":          "_scan_composite_score",
            "piotroski":      "_scan_piotroski",
            "price":          "_scan_price",
            "short_pct_float":"shortPercentOfFloat",
            "short_ratio":    "shortRatio",
            "beta":           "beta",
            "pe_ratio":       "trailingPE",
            "forward_pe":     "forwardPE",
            "profit_margin":  "profitMargins",
            "operating_margin":"operatingMargins",
            "roe":            "returnOnEquity",
            "roa":            "returnOnAssets",
            "revenue_growth": "revenueGrowth",
            "earnings_growth":"earningsGrowth",
            "current_ratio":  "currentRatio",
            "market_cap":     "marketCap",
            "float_shares":   "floatShares",
            "target_mean":    "targetMeanPrice",
            "target_low":     "targetLowPrice",
            "target_high":    "targetHighPrice",
        }
        # Fields we still need after yfinance
        missing = [ik for ik in SCAN_FIELD_MAP.values()
                   if not ik.startswith("_scan_") and not info.get(ik)]
        if missing:
            try:
                dump = load_screener_dump(SCREENER_URL_DEFAULT)
                sym = ticker.upper()
                if sym in dump.index:
                    row = dump.loc[sym]
                    for scan_key, info_key in SCAN_FIELD_MAP.items():
                        val = row.get(scan_key) if hasattr(row, "get") else (
                            row[scan_key] if scan_key in row.index else None
                        )
                        if val is not None and (info_key.startswith("_scan_") or not info.get(info_key)):
                            info[info_key] = val
                    info["_from_scan_dump"] = True
            except Exception:
                pass

        # ── Step 4: Quarterly EPS history (actual vs estimate) ───────────
        # Returns a list of dicts: [{quarter, actual, estimate, surprise_pct}, ...]
        # ordered oldest → newest. Empty list if unavailable.
        eps_history = []
        try:
            # yfinance >= 0.2 exposes earnings_history (actual vs estimate per quarter)
            eh = _yf(lambda: getattr(tk, "earnings_history", None))
            if eh is not None and hasattr(eh, "empty") and not eh.empty:
                eh = eh.copy()
                # Columns vary by version: epsActual/epsEstimate or epsactual/epsestimate
                cols = {c.lower(): c for c in eh.columns}
                act_col = cols.get("epsactual") or cols.get("eps_actual")
                est_col = cols.get("epsestimate") or cols.get("eps_estimate")
                sur_col = cols.get("surprisepercent") or cols.get("surprise_percent")
                # Index is usually the quarter date
                for idx, row in eh.iterrows():
                    actual   = row.get(act_col) if act_col else None
                    estimate = row.get(est_col) if est_col else None
                    surprise = row.get(sur_col) if sur_col else None
                    if actual is None and estimate is None:
                        continue
                    # Surprise % — compute if not provided
                    if surprise is None and actual is not None and estimate not in (None, 0):
                        try:
                            surprise = (float(actual) - float(estimate)) / abs(float(estimate)) * 100
                        except Exception:
                            surprise = None
                    # Quarter label
                    q_label = str(idx)
                    try:
                        q_dt = pd.to_datetime(idx, errors="coerce")
                        if pd.notna(q_dt):
                            q_label = q_dt.strftime("%b %Y")
                    except Exception:
                        pass
                    eps_history.append({
                        "quarter":  q_label,
                        "actual":   float(actual) if actual is not None else None,
                        "estimate": float(estimate) if estimate is not None else None,
                        "surprise": float(surprise) if surprise is not None else None,
                    })
                # Keep most recent 8 quarters, oldest → newest
                eps_history = eps_history[-8:]
        except Exception:
            eps_history = []
            _issues.append("EPS history: earnings data not returned (thin analyst coverage or rate limit)")

        # Fallback: derive quarterly EPS from the income statement if
        # earnings_history was empty. We deliberately do NOT use
        # tk.quarterly_earnings / tk.earnings here — yfinance has deprecated
        # that property (it warns "not available via API, look for Net Income
        # in Ticker.income_stmt") and the underlying scraper it falls back to
        # is unmaintained and has caused interpreter-level crashes in this
        # environment. income_stmt is the modern, supported replacement.
        if not eps_history:
            try:
                qis = _yf(lambda: getattr(tk, "quarterly_income_stmt", None))
                if qis is not None and hasattr(qis, "empty") and not qis.empty:
                    # Rows are line items (e.g. "Net Income", "Diluted EPS"),
                    # columns are quarter-end dates, newest first.
                    row_idx = {str(i).strip().lower(): i for i in qis.index}
                    eps_row = row_idx.get("diluted eps") or row_idx.get("basic eps")
                    if eps_row is not None:
                        for col in qis.columns:
                            val = qis.loc[eps_row, col]
                            if pd.isna(val):
                                continue
                            try:
                                q_dt = pd.to_datetime(col, errors="coerce")
                                q_label = q_dt.strftime("%b %Y") if pd.notna(q_dt) else str(col)
                            except Exception:
                                q_label = str(col)
                            eps_history.append({
                                "quarter":  q_label,
                                "actual":   float(val),
                                "estimate": None,
                                "surprise": None,
                            })
                        # Statement columns are newest→oldest; reverse to oldest→newest
                        eps_history = list(reversed(eps_history))[-8:]
            except Exception:
                pass
            if not eps_history:
                _issues.append("EPS history: income statement EPS line item not available for this symbol")

        # ── Forward EPS estimates (analyst consensus, as far out as available) ─
        # Sources, in priority order:
        #   1. tk.earnings_estimate  — next quarter (0q), current quarter (+1q),
        #      current year (0y), next year (+1y) consensus avg
        #   2. info forwardEps / epsCurrentYear / epsNextYear  — annual estimates
        # Returns oldest→newest list: [{period, estimate, n_analysts, is_forward}]
        eps_forward = []
        try:
            ee = _yf(lambda: getattr(tk, "earnings_estimate", None))
            if ee is not None and hasattr(ee, "empty") and not ee.empty:
                # Rows indexed by period: '0q','+1q','0y','+1y' (current/next qtr/yr)
                period_labels = {
                    "0q":  "Next Qtr",
                    "+1q": "Qtr After",
                    "0y":  "This Year",
                    "+1y": "Next Year",
                }
                cols = {c.lower(): c for c in ee.columns}
                avg_col = cols.get("avg") or cols.get("average")
                num_col = cols.get("numberofanalysts") or cols.get("numberofanalystopinions")
                # Keep a logical order: quarters first, then years
                for pk in ["0q", "+1q", "0y", "+1y"]:
                    if pk in ee.index:
                        row = ee.loc[pk]
                        est = row.get(avg_col) if avg_col else None
                        nan = row.get(num_col) if num_col else None
                        if est is not None and pd.notna(est):
                            eps_forward.append({
                                "period":     period_labels.get(pk, pk),
                                "estimate":   float(est),
                                "n_analysts": int(nan) if (nan is not None and pd.notna(nan)) else None,
                                "is_forward": True,
                            })
        except Exception:
            pass

        # Fallback / supplement: annual forward EPS from info fields
        if not eps_forward:
            try:
                cy = info.get("epsCurrentYear")
                ny = info.get("epsNextYear") or info.get("forwardEps")
                fwd = info.get("forwardEps")
                if cy is not None:
                    eps_forward.append({"period": "This Year (est)", "estimate": float(cy),
                                        "n_analysts": info.get("numberOfAnalystOpinions"), "is_forward": True})
                if ny is not None and ny != cy:
                    eps_forward.append({"period": "Next Year (est)", "estimate": float(ny),
                                        "n_analysts": info.get("numberOfAnalystOpinions"), "is_forward": True})
                elif fwd is not None and not eps_forward:
                    eps_forward.append({"period": "Forward (est)", "estimate": float(fwd),
                                        "n_analysts": info.get("numberOfAnalystOpinions"), "is_forward": True})
            except Exception:
                pass

        if _issues:
            info["_data_issues"] = _issues
        return info, hist, eps_history, eps_forward


    # ----------------------------------------------------------------------------
    # Reference key (rendered in its own tab)
    # ----------------------------------------------------------------------------
    REFERENCE_KEY = [
        ("Stock card layout", "#f5a623", [
            ("Arc gauge", "Semicircular dial in the top-right of every card. Sweeps from teal (cold) through green, amber, orange to red (igniting, score 100). Score number sits at the center bottom in the matching zone color — read it like a speedometer.", "sweep = heat level"),
            ("IGNITING badge", "Orange badge on the ticker when all four live conditions confirm at once on a flat/up tape: RVOL >=2x, volume surge >=2x, positive velocity, new HOD or VWAP reclaim.", "orange border pulse"),
            ("GAP REV badge", "Teal badge. Same four conditions fire but stock is down 4%+ or gapped down 4%+ — bounce in a selloff. Tradable but higher failure rate than clean ignition.", "teal border"),
            ("Price tile", "Current price in amber, change % in green (up) or red (down). Alpaca real-time when API keys set, else Yahoo Finance.", "amber = current"),
            ("Target tile", "Analyst consensus mean price target. Green = 10%+ upside, amber = 0-10%, red = trading above target. Sourced from yfinance or nightly dump fallback.", "color = upside"),
            ("RVOL tile", "Relative Volume vs 20-day average. Amber = 3x+ explosive, yellow = 1.5x+ high, grey = normal/low. Plain-English label: explosive / high / normal / low.", "3x+ = explosive"),
            ("Catalyst pills", "Colored clickable badges below the tiles. Each opens the most relevant source for that catalyst. See Catalyst signals section.", "tap to research"),
        ]),
        ("Catalyst signals - inside Fuel score", "#f5a623", [
            ("Earnings", "Days until next earnings. 0-2 days = binary event, peak score +30. Score decays with distance.", "0-2d = peak score"),
            ("FDA", "Healthcare/Pharma/Biotech only. Requires 2+ FDA-specific keywords. Sector-gated to prevent false positives on unrelated stocks like hotels.", "sector-gated, 2 hits"),
            ("M&A / Buyout", "Any sector. Requires 2+ M&A keywords to avoid single-word false positives. Links to Google News.", "2 hits required"),
            ("Partnership", "Any sector. Requires 2+ partnership keywords — single words like agreement are too common alone.", "2 hits required"),
            ("Legal", "Any sector. Requires 2+ legal keywords to confirm a real litigation event.", "2 hits required"),
            ("Squeeze", "Any sector. 1 specific squeeze keyword sufficient — precise terms. Backed by DTC gauge. Links to Finviz.", "1 hit, see DTC"),
            ("Breakout", "Any sector. 1 keyword sufficient — 52-week high and breakout terms are unambiguous. Links to Finviz chart.", "1 hit"),
            ("Geopolitical", "Energy, Materials, Defense, Industrials, Semis only. Requires 2+ keywords — oil or china alone appears too broadly.", "sector-gated, 2 hits"),
            ("Rate", "Financials, Banks, REITs, Utilities, Insurance only. Requires 2+ keywords — fed/inflation appear in nearly every article.", "sector-gated, 2 hits"),
            ("BIMODAL", "Binary event within 3 days (earnings, FDA, legal) = elevated volatility expected. Catalyst score multiplied 1.25x. Amber BIMODAL badge on card.", "1.25x multiplier"),
            ("DTC", "Days to Cover gauge: shares short / avg daily volume. Red = 10d+ extreme, amber = 7-10d high, yellow = 5-7d moderate. Links to Finviz.", "10d+ = extreme"),
            ("Filtered", "Catalysts detected but blocked by sector whitelist or keyword threshold appear struck-through in the Stock Analyzer, showing what was caught and why it was removed.", "transparency"),
            ("EARN ↑", "Fires when last quarter beat analyst estimates by 10%+ OR YoY earnings growth exceeds 25%, confirmed by at least 1 news keyword. Universal — all sectors. Worth 20 pts in Fuel score. Links to Yahoo Finance financials.", "10% beat or 25% YoY growth"),
        ]),
        ("The scores", "#ffffff", [
            ("Score", "Overall grade 0-100: 60% Ignition + 40% Fuel. Displayed as the arc gauge sweep and center number on every card.", "arc gauge = score"),
            ("Ignition", "Is money flowing in RIGHT NOW? Rebuilt from live bars every scan. RVOL 25%, surge 15%, velocity 15%, acceleration 10%, VWAP 10%, HOD 10%, RSI5 7.5%, MACD 7.5%.", "live, every scan"),
            ("Fuel", "Is this stock READY to make a big move? Updates hourly. Short float 18%, insider buying 18%, news/sentiment 18%, catalyst score 26%, float 8%, 52W position 12%.", "hourly update"),
            ("Catalyst Score", "Sub-score inside Fuel (26% weight). FDA event +25, M&A +25, earnings imminent +30, DTC 10d+ +15, partnership +15, legal +10, squeeze news +10, breakout +8, geo +6, rate +5. Bimodal multiplier 1.25x.", "26% of Fuel"),
            ("NightlyRank", "How the stock scored in last night's screener dump. Used in Nightly Screener mode to pick the candidate pool.", "pool selection"),
        ]),
        ("Live ignition signals - 60% of Score", "#f5a623", [
            ("RVOL", "Today's cumulative volume vs 20-day average, adjusted for the U-shaped intraday curve (heavy at open/close, quiet at lunch). 2x = unusual, 5x = explosive.", "2x unusual, 5x explosive"),
            ("Surge", "Last 3 one-minute bars vs the session average bar. Catches the exact minutes buying pressure hits. 2x+ = surge in progress.", "2x+ = surge live"),
            ("Vel %/5m", "Price move over the last 5 minutes. Positive and growing = price thrust in progress.", "positive + growing"),
            ("Accel", "Is the 5-minute velocity itself speeding up? Positive acceleration on top of positive velocity = momentum building not fading.", "positive = building"),
            ("VWAP", "Price above the volume-weighted average price = buyers controlling the session. A reclaim from below is one of the four IGNITING triggers.", "reclaim = trigger"),
            ("HOD", "High of Day. NEW = just broke to a new session high, classic ignition trigger. Near = within 0.5%, coiling under resistance.", "NEW = trigger"),
            ("RSI5", "RSI(14) on 5-minute bars. 55-75 = momentum thrust zone. Above 75 = stretched. Below 50 = no momentum.", "55-75 sweet spot"),
            ("MACD", "Bullish cross on 5-minute bars: MACD line crosses above signal line. Confirms the trend flip.", "cross = confirmation"),
        ]),
        ("Fuel signals - 40% of Score", "#3ddc84", [
            ("Short%Flt", "Short interest as % of float. 15%+ means heavy bets against the stock — forced short covering adds buying pressure if price rises.", "15%+ = squeeze fuel"),
            ("InsiderNet$", "Net dollar value of insider buys minus sells, last 90 days (SEC Form 4). Positive = informed accumulation.", "positive = accumulating"),
            ("News48h", "Headlines in the last 48 hours. Momentum needs a catalyst. Zero news = no sustained move likely.", "0 = no catalyst"),
            ("Float", "Tradable shares outstanding. Under 50M = explosive potential on any volume spike.", "under 50M = explosive"),
            ("52wk dist", "Position within the 52-week range. Momentum regimes cluster near highs. Within 5% of 52W high = approaching breakout zone.", "within 5% = best"),
        ]),
        ("Direction filters", "#29b6c8", [
            ("IGNITING", "All four conditions confirm on a flat/up tape: RVOL >=2x, surge >=2x, positive velocity, new HOD or VWAP reclaim. Orange card border. Phone push: urgent priority.", "4 conditions, up tape"),
            ("GAP REVERSAL", "Same four conditions on a down tape (-4%+ day or -4%+ gap open). Teal card border + GAP REV badge. Bounce in a selloff — higher failure rate than clean ignition.", "bounce in selloff"),
            ("Direction check", "Gap calculated from first bar open vs yesterday close. High RVOL on a -17% earnings flush is a completely different trade than high RVOL on a +3% day.", "-4% threshold"),
        ]),
        ("Data source chain", "#b0c8e8", [
            ("Alpaca API", "Real-time price history (1-min, 5-min, daily bars) via IEX feed. Powers live Ignition signals and Stock Analyzer history. Set ALPACA_API_KEY + ALPACA_SECRET_KEY in Streamlit secrets.", "fastest, real-time"),
            ("yfinance", "Yahoo Finance fallback for price history when Alpaca has insufficient bars. Primary source for fundamentals: P/E, margins, analyst targets, short interest, insider transactions, news.", "15 min delay"),
            ("Nightly dump", "stock_data.json.gz from magicpro33/stock pipeline. Third-tier fallback for missing fundamental fields. Also powers Nightly Screener pre-ranking mode.", "nightly fallback"),
            ("Data source badge", "Below Stock Analyzer header: green = Alpaca active, grey = Yahoo history, amber = nightly dump filled missing fields.", "badge shows source"),
        ]),
        ("Watchlist modes", "#d0b040", [
            ("Sector presets", "8 sector watchlists from your Webull list: Precious Metals, Energy/Uranium, Defense/Space, Semis/Tech, Quantum/AI/Biotech, Industrials, Income/ETFs, Critical Minerals. Random sector on first app load.", "random on startup"),
            ("Custom", "Select Custom to show the ticker entry box. Hidden for named presets to keep the sidebar clean.", "hidden unless Custom"),
            ("Scan ALL presets", "Combines all 8 presets into one scan (187 unique stocks), ETFs excluded. Shows top N by Score.", "ETFs excluded"),
            ("Nightly Screener", "Downloads your nightly dump once per calendar day, pre-ranks all tickers, picks top N candidates, live-scans them. Watchlist pinned for the day.", "dump once/day"),
            ("Results slider", "5-30: how many stocks to display after the scan, ranked by Score.", "5-30 stocks"),
        ]),
        ("Price range analysis", "#29b6c8", [
            ("52W Channel", "Visual bar: where current price sits within the year low-to-high range. Red = near 52W low, amber = mid-range, green = near 52W high (momentum regime).", "green near high"),
            ("Bollinger Bands", "20-day MA +/- 2 standard deviations. Near upper band = overbought. Near lower = oversold/bounce potential. Inside bands = neutral.", "2 std dev band"),
            ("ATR", "Average True Range 14-day: typical daily price range in dollars. Use for stop-loss sizing — place stops 1-1.5x ATR below entry.", "stop-loss sizing"),
            ("Price Performance", "Returns over 1 day, 5 days, 1 month, 3 months. Green = positive, red = negative.", "4 timeframes"),
        ]),
        ("Technical indicators - stock analyzer", "#3ddc84", [
            ("RSI 14d", "Daily RSI (14-period). Below 30 = oversold. Above 70 = overbought. 45-70 = momentum sweet spot. Different from RSI5 (5-minute bars in live signals).", "45-70 sweet spot"),
            ("MACD daily", "MACD on daily bars. Above signal line = daily uptrend. Used for multi-week trend direction — separate from 5-minute MACD in live signals.", "daily trend"),
            ("50-Day MA", "Price just above 50MA = ideal entry zone. Just below = watch for reclaim. Golden Cross (50MA > 200MA) = major buy signal.", "just above = entry"),
            ("200-Day MA", "Long-term trend. Golden Cross: 50MA crosses above 200MA = major bull signal. Death Cross: 50MA falls below 200MA = caution.", "golden vs death"),
            ("Volume ratio", "Today's volume vs 20-day average. Above 1.5x = high participation confirms moves. Below 0.5x = low conviction.", "1.5x+ = confirms"),
        ]),
        ("Valuation metrics", "#b0c8e8", [
            ("Market Cap", "Price x shares. Micro <$300M, small <$2B, mid <$10B, large >$10B. Affects volatility and institutional eligibility.", "size tier"),
            ("P/E Ratio", "Price / trailing earnings. High = growth priced in. Low = value or declining business. Compare within sector only.", "sector-relative"),
            ("Forward P/E", "P/E using next 12 months estimates. Forward < Trailing = analysts expect earnings growth.", "fwd < trail = growth"),
            ("P/B Ratio", "Price / book value. Below 1.0 = trading below assets. Above 3.0 = paying for brand/growth.", "below 1 = cheap"),
            ("P/S Ratio", "Price / revenue. Below 2x = reasonable. Above 10x = high growth premium.", "below 2x = fair"),
            ("Beta", "Volatility vs S&P 500. 1.5 = 50% more volatile. High beta = bigger swings both directions.", "1.5 = high vol"),
            ("Float", "Tradeable shares. Under 20M = explosive on volume.", "under 20M = explosive"),
            ("Analyst target", "Consensus mean price target on every card: green = 10%+ upside, amber = 0-10%, red = above target.", "on every card"),
        ]),
        ("Financial health metrics", "#ffffff", [
            ("Profit Margin", "Net income / revenue. Expanding = pricing power. Shrinking = rising costs or competition.", "expanding = strong"),
            ("Operating Margin", "EBIT / revenue. Core business efficiency before interest/taxes.", "core efficiency"),
            ("ROE", "Return on Equity. Above 15% = strong. Buffett's primary durable-advantage metric.", "15%+ = strong"),
            ("ROA", "Return on Assets. Above 5% = solid. Not distorted by debt level.", "5%+ = solid"),
            ("D/E Ratio", "Debt / equity. High D/E amplifies gains and losses. Compare within sector.", "watch the trend"),
            ("Current Ratio", "Current assets / current liabilities. Above 1.5 = comfortable short-term liquidity.", "1.5+ = comfortable"),
            ("Revenue Growth", "YoY revenue change. Double-digit growth = attractive to institutions.", "10%+ = strong"),
            ("Earnings Growth", "YoY EPS change. Growing faster than revenue = margin expansion.", "10%+ = strong"),
        ]),
        ("Flags and alerts", "#ff4444", [
            ("IGNITING", "Orange card border + IGNITING badge. Phone push: urgent priority (punches through Do Not Disturb). Fires once per ticker per day.", "urgent push"),
            ("GAP REV", "Teal card border + GAP REV badge. Phone push: high priority. Labeled separately from IGNITING so you know the context immediately.", "high push"),
            ("Score alert", "Score crossed the sidebar threshold (default 65). Phone push: default priority. Each ticker alerts once per day.", "default push"),
            ("Popup alerts toggle", "Off by default. When on, in-app toast popups fire for every alert. When off, scanner runs silently — ntfy phone pushes still fire independently.", "off by default"),
            ("ntfy.sh", "Free phone push service. Set NTFY_TOPIC in Streamlit secrets. Topic name is the only password — make it unguessable. Test button in sidebar.", "free, instant"),
        ]),
    ]

    def render_reference_key():
        """Color-coded glossary of every term in the indicator, as its own tab."""
        rows = ["<table class='ref-table'>"]
        for group, color, terms in REFERENCE_KEY:
            rows.append(
                f"<tr class='grp'><td colspan='3' style='color:{color}'>{group}</td></tr>"
            )
            for term, meaning, level in terms:
                rows.append(
                    f"<tr><td class='trm' style='border-left:3px solid {color};"
                    f"padding-left:8px'>{term}</td>"
                    f"<td class='mng'>{meaning}</td><td class='lvl'>{level}</td></tr>"
                )
        rows.append("</table>")
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption(
            "The relationship that matters: Fuel tells you WHICH stocks to watch, "
            "Ignition tells you WHEN. A name at Fuel 70 / Ignition 25 is a loaded "
            "spring doing nothing - yet. When Ignition jumps 30 points in one "
            "refresh, that is the moment this tool exists for. Not financial advice."
        )


    # ----------------------------------------------------------------------------
    # Tabs: live scanner + reference key
    # ----------------------------------------------------------------------------
    tab_scan, tab_ref, tab_lookup = st.tabs(["Scanner", "Reference key", "Stock Lookup"])

    with tab_ref:
        render_reference_key()

    # Everything below renders inside the Scanner tab. The tab context is entered
    # explicitly so the long display section keeps its flat indentation.
    tab_scan.__enter__()

    # Banners removed — IGNITING and GAP REV status shown via card icons instead
    igniting_now = [r for r in ok if r["igniting"]]
    reversals_now = [r for r in ok if r["gap_reversal"]]

    # Catalyst tag definitions: label, CSS class, URL builder (lambda ticker -> url)
    CATALYST_TAG_META = {
        "earnings":     ("EARNINGS",  "ct-earnings",     lambda t: f"https://finance.yahoo.com/calendar/earnings?symbol={t}"),
        "fda":          ("FDA",       "ct-fda",          lambda t: f"https://www.google.com/search?q={t}+FDA+approval+news&tbm=nws"),
        "buyout":       ("M&A",       "ct-buyout",       lambda t: f"https://www.google.com/search?q={t}+merger+acquisition+buyout&tbm=nws"),
        "legal":        ("LEGAL",     "ct-legal",        lambda t: f"https://www.google.com/search?q={t}+lawsuit+settlement+verdict&tbm=nws"),
        "partnership":  ("PARTNER",   "ct-partnership",  lambda t: f"https://finance.yahoo.com/quote/{t}/news/"),
        "squeeze":      ("SQUEEZE",   "ct-squeeze",      lambda t: f"https://finviz.com/quote.ashx?t={t}"),
        "breakout":     ("BREAKOUT",  "ct-breakout",     lambda t: f"https://finviz.com/quote.ashx?t={t}&ty=c&ta=1&p=d"),
        "geopolitical": ("GEO/MACRO", "ct-geopolitical", lambda t: f"https://www.google.com/search?q={t}+tariff+geopolitical+news&tbm=nws"),
        "rate":         ("FED/RATES", "ct-rate",         lambda t: "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"),
        "earn_growth":  ("EARN ↑",    "ct-earn-growth",  lambda t: f"https://finance.yahoo.com/quote/{t}/financials/"),
    }

    BIMODAL_TAG = ("BIMODAL", "ct-bimodal", lambda t: f"https://finance.yahoo.com/calendar/earnings?symbol={t}")
    DTC_TAG_CLASS = "ct-dtc"


    def catalyst_pill(label: str, css_class: str, url: str) -> str:
        """Return an HTML span pill that is a clickable colored link."""
        return (
            f"<span class='fuel-tag {css_class}'>"
            f"<a href='{url}' target='_blank' rel='noopener'>{label}</a>"
            f"</span>"
        )


    def dtc_gauge_pill(ticker: str, dtc: float) -> str:
        """Render DTC as a labeled fuel gauge bar pill.
        Scale: 0d = empty, 15d+ = full. Color shifts low→mid→high."""
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        pct = min(dtc / 15.0, 1.0) * 100
        if dtc >= 10:
            color = "#ff3333"   # red  = extreme squeeze fuel
            label = "SQUEEZE"
            intensity = "EXTREME"
        elif dtc >= 7:
            color = "#f5a623"   # amber = high squeeze fuel
            label = "SHORT"
            intensity = "HIGH"
        elif dtc >= 5:
            color = "#d0b040"   # yellow = moderate
            label = "SHORT"
            intensity = "MOD"
        else:
            color = "#5090d0"   # blue = low
            label = "SHORT"
            intensity = "LOW"
        bar = (
            f"<span class='dtc-bar-track'>"
            f"<span class='dtc-bar-fill' style='width:{pct:.0f}%;background:{color}'></span>"
            f"</span>"
        )
        return (
            f"<span class='dtc-gauge' style='border-color:{color};color:{color}'>"
            f"<a href='{url}' target='_blank' rel='noopener'>"
            f"{label} {bar} {intensity} <span style='color:#7a9ab8;font-size:10px'>({dtc}d)</span>"
            f"</a></span>"
        )


    def build_catalyst_tags_html(ticker: str, cat_tags: list, bimodal: bool,
                                  dtc=None, earnings_days=None) -> str:
        """Build the full row of colored clickable catalyst pills for a ticker."""
        parts = []
        if bimodal:
            label, css, url_fn = BIMODAL_TAG
            parts.append(catalyst_pill(label, css, url_fn(ticker)))
        for tag in cat_tags:
            if tag not in CATALYST_TAG_META:
                continue
            label, css, url_fn = CATALYST_TAG_META[tag]
            if tag == "earnings" and earnings_days is not None:
                label = f"EARN {earnings_days}d" if earnings_days >= 0 else f"EARN -{abs(earnings_days)}d"
            parts.append(catalyst_pill(label, css, url_fn(ticker)))
        if dtc and dtc >= 5:
            parts.append(dtc_gauge_pill(ticker, dtc))
        return "".join(parts)


    def _score_icon(sc, igniting, gap_rev):
        """Return an SVG icon that communicates signal strength without a number.

        IGNITING  → animated pulsing flame  (amber)
        GAP REV   → wave/bounce arc         (teal)
        Score 70+ → lightning bolt          (amber, solid)
        Score 50+ → trending arrow up       (amber, outline)
        Score <50 → flat dash               (grey)
        """
        if igniting:
            # Flame — three teardrop paths
            return (
                "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
                "xmlns='http://www.w3.org/2000/svg' aria-label='Igniting'>"
                "<style>@keyframes fp{0%,100%{opacity:1}50%{opacity:.55}}"
                ".fp{animation:fp 1.4s ease-in-out infinite}</style>"
                "<path class='fp' d='M14 3C14 3 8 9 8 15a6 6 0 0012 0c0-2.5-1.5-5-2.5-6.5"
                "C17 10 16 11.5 14 12c1-2.5.5-6-0-9z' fill='#f5a623'/>"
                "<path class='fp' style='animation-delay:.2s' d='M14 14c0 0-3 1.5-3 4a3 3 0 006 0"
                "c0-1.5-1-2.8-1.5-3.5C15.5 15.5 15 16.5 14 17c.5-1.2.3-2.5 0-3z' fill='#ff9500'/>"
                "</svg>"
            )
        elif gap_rev:
            # Bounce arc — wave up from bottom
            return (
                "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
                "xmlns='http://www.w3.org/2000/svg' aria-label='Gap reversal'>"
                "<path d='M4 20 Q9 8 14 16 Q19 24 24 12' stroke='#29b6c8' "
                "stroke-width='2.5' stroke-linecap='round' fill='none'/>"
                "<path d='M20 10 L24 12 L22 16' stroke='#29b6c8' "
                "stroke-width='2' stroke-linecap='round' stroke-linejoin='round' fill='none'/>"
                "</svg>"
            )
        elif sc >= 70:
            # Lightning bolt — solid amber
            return (
                "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
                "xmlns='http://www.w3.org/2000/svg' aria-label='High score'>"
                "<path d='M16 3L7 16h7l-2 9 10-13h-7l2-9z' fill='#f5a623'/>"
                "</svg>"
            )
        elif sc >= 50:
            # Trending up arrow — amber outline
            return (
                "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
                "xmlns='http://www.w3.org/2000/svg' aria-label='Rising score'>"
                "<path d='M4 20L11 13l5 4 8-10' stroke='#f5a623' "
                "stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/>"
                "<path d='M19 7h5v5' stroke='#f5a623' "
                "stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/>"
                "</svg>"
            )
        else:
            # Flat dash — grey, no momentum
            return (
                "<svg width='28' height='28' viewBox='0 0 28 28' fill='none' "
                "xmlns='http://www.w3.org/2000/svg' aria-label='Low score'>"
                "<path d='M6 14h16' stroke='#4a6a8a' "
                "stroke-width='2.5' stroke-linecap='round'/>"
                "</svg>"
            )


    def render_compact(rows_data):
        """Option 2 stat-grid cards: 2×2 metric tiles, icon indicator, no score number."""
        html = []
        for r in rows_data:
            sc   = r["score"]
            ign  = r["igniting"]
            grev = r.get("gap_reversal", False)
            color = "#f5a623" if sc >= 50 else "#4a6a8a"
            hot   = " hot" if ign else (" rev" if grev else "")

            # ── Header flag badge ────────────────────────────────────────
            if ign:
                flag = "<span class='cflag'>IGNITING</span>"
            elif grev:
                flag = "<span class='cflag rev'>GAP REV</span>"
            else:
                flag = ""

            # ── Icon indicator (replaces score number) ───────────────────
            icon_svg = _score_icon(sc, ign, grev)

            # ── Arc gauge (replaces gradient bar + icon) ─────────────────
            # SVG semicircle: arc length of the half-circle path =
            # π × r = π × 40 ≈ 125.7. We use stroke-dasharray=125.7
            # and stroke-dashoffset = 125.7 × (1 - sc/100) to fill.
            arc_len   = 125.7
            fill_len  = arc_len * (sc / 100.0)
            dash_off  = arc_len - fill_len
            # Score color matches the gradient zones
            if sc >= 80:   gc = "#ff2200"
            elif sc >= 65: gc = "#ff6600"
            elif sc >= 50: gc = "#f5a623"
            elif sc >= 35: gc = "#d0b040"
            elif sc >= 20: gc = "#3ddc84"
            else:          gc = "#29b6c8"
            # Gradient id must be unique per card to avoid SVG bleed
            gid = f"ag{abs(hash(r['ticker']))%9999}"
            arc_svg = (
                f"<svg width='80' height='46' viewBox='0 0 90 50' fill='none' "
                f"aria-label='Score {sc:.0f}'>"
                f"<defs><linearGradient id='{gid}' x1='0%' y1='0%' x2='100%' y2='0%'>"
                f"<stop offset='0%' stop-color='#29b6c8'/>"
                f"<stop offset='30%' stop-color='#3ddc84'/>"
                f"<stop offset='60%' stop-color='#f5a623'/>"
                f"<stop offset='100%' stop-color='#ff2200'/>"
                f"</linearGradient></defs>"
                f"<path d='M5 45 A40 40 0 0 1 85 45' stroke='#122540' "
                f"stroke-width='9' stroke-linecap='round' fill='none'/>"
                f"<path d='M5 45 A40 40 0 0 1 85 45' stroke='url(#{gid})' "
                f"stroke-width='9' stroke-linecap='round' fill='none' "
                f"stroke-dasharray='{arc_len:.1f}' stroke-dashoffset='{dash_off:.1f}'/>"
                f"<text x='45' y='43' text-anchor='middle' "
                f"font-family='Space Mono,monospace' font-size='16' font-weight='700' "
                f"fill='{gc}'>{sc:.0f}</text>"
                f"</svg>"
            )

            # ── Price tile ───────────────────────────────────────────────
            price  = r.get("price")
            chg    = r.get("chg_pct")
            if price:
                chg_col = "#3ddc84" if (chg or 0) >= 0 else "#ff3333"
                chg_str = f"<span class='ctile-sub' style='color:{chg_col}'>{chg:+.1f}%</span>" if chg is not None else ""
                price_tile = (
                    f"<div class='ctile'>"
                    f"<div class='ctile-lbl'>Price</div>"
                    f"<div class='ctile-val' style='color:#f5a623'>${price:.2f}</div>"
                    f"{chg_str}</div>"
                )
            else:
                price_tile = "<div class='ctile'><div class='ctile-lbl'>Price</div><div class='ctile-val'>--</div></div>"

            # ── Target tile ──────────────────────────────────────────────
            fuel      = r["fuel"]
            tgt       = fuel.get("target_mean")
            if tgt and price and price > 0:
                upside  = (tgt - price) / price * 100
                tgt_col = "#3ddc84" if upside >= 10 else ("#d0b040" if upside >= 0 else "#ff3333")
                sign    = "+" if upside >= 0 else ""
                target_tile = (
                    f"<div class='ctile'>"
                    f"<div class='ctile-lbl'>Target</div>"
                    f"<div class='ctile-val' style='color:{tgt_col}'>${tgt:.2f}</div>"
                    f"<div class='ctile-sub' style='color:{tgt_col}'>{sign}{upside:.1f}% upside</div>"
                    f"</div>"
                )
            else:
                target_tile = "<div class='ctile'><div class='ctile-lbl'>Target</div><div class='ctile-val' style='color:#4a6a8a'>--</div></div>"

            # ── RVOL tile ────────────────────────────────────────────────
            rvol = r.get("rvol", 0)
            if rvol:
                rvol_col = "#f5a623" if rvol >= 3 else ("#d0b040" if rvol >= 1.5 else "#7a9ab8")
                rvol_lbl = "explosive" if rvol >= 5 else ("high" if rvol >= 2 else ("normal" if rvol >= 0.8 else "low"))
                rvol_tile = (
                    f"<div class='ctile'>"
                    f"<div class='ctile-lbl'>RVOL</div>"
                    f"<div class='ctile-val' style='color:{rvol_col}'>{rvol:.1f}x</div>"
                    f"<div class='ctile-sub' style='color:{rvol_col}'>{rvol_lbl}</div>"
                    f"</div>"
                )
            else:
                rvol_tile = "<div class='ctile'><div class='ctile-lbl'>RVOL</div><div class='ctile-val' style='color:#4a6a8a'>--</div></div>"

            # ── Catalyst pills ───────────────────────────────────────────
            cat_tags = fuel.get("catalyst_tags") or []
            bimodal  = fuel.get("bimodal_event", False)
            ed       = fuel.get("earnings_days")
            dtc      = fuel.get("days_to_cover")
            tag_html = build_catalyst_tags_html(r["ticker"], cat_tags, bimodal, dtc, ed)

            # ── Assemble card ────────────────────────────────────────────
            parts = [
                f"<div class='crow{hot}'>",
                # Header: ticker + flag left, arc gauge right
                f"<div class='chead'>",
                f"<span><span class='ctick'>{r['ticker']}</span>{flag}</span>",
                f"<div class='cicon' style='margin-top:-6px'>{arc_svg}</div>",
                f"</div>",
                # Stat grid (no bar below header — gauge replaces it)
                f"<div class='cgrid' style='grid-template-columns:1fr 1fr 1fr'>",
                price_tile,
                target_tile,
                rvol_tile,
                f"</div>",
            ]
            if tag_html:
                parts.append(f"<div style='margin-top:2px'>{tag_html}</div>")
            parts.append("</div>")
            html.append("".join(parts))

        st.markdown("".join(html), unsafe_allow_html=True)


    # ----------------------------------------------------------------------------
    # Leaderboard
    # ----------------------------------------------------------------------------
    if ok and view_mode == "Compact (phone)":
        render_compact(ok)
    elif ok:
        HELP = {
            "Ticker": "Stock symbol. Click a row's ticker in the Chart selector below to inspect it.",
            "Score": "Overall Ignition Score, 0-100. Weighted blend: 60% Ignition (live, is it moving NOW) + 40% Fuel (is it primed to move). Higher = stronger momentum setup.",
            "NightlyRank": "How this stock scored in last night's screener dump pre-ranking (0-100), based on your nightly signals: Short Squeeze, composite score, RSI zone, Golden Cross, MFI, OBV, volume, Piotroski. This is yesterday's homework; Score is today's live grade.",
            "Ignition": "Live momentum sub-score, 0-100, recalculated every refresh. Built from: relative volume vs 20-day pace (25%), last-3-bar volume surge (15%), 5-min price velocity (15%), acceleration (10%), VWAP position/reclaim (10%), high-of-day breakout (10%), RSI thrust (7.5%), MACD cross (7.5%).",
            "Fuel": "Primed-to-move sub-score, 0-100, refreshed hourly. Built from: short % of float (25%), insider net buying 90d (25%), news flow + sentiment 48h (25%), float size (10%), distance to 52-week high (15%). High Fuel + rising Ignition is the setup you want.",
            "Price": "Last traded price from the live feed (Alpaca real-time when keys are set, otherwise Yahoo, which can lag ~15 min).",
            "Chg %": "Percent change vs yesterday's close.",
            "RVOL": "Relative Volume: today's cumulative volume vs the 20-day average, adjusted for the U-shaped intraday volume curve (heavy at open and close, quiet at lunch). 1.0 = normal. 2x+ = unusual money flowing in. 5x+ = explosive. The single most reliable ignition tell.",
            "Surge": "Last 3 one-minute bars' average volume vs this session's average bar. Catches the exact minutes buying pressure hits. 2x+ = surge in progress.",
            "Vel %/5m": "Velocity: percent price move over the last 5 minutes. Positive and growing = thrust.",
            "VWAP+": "Y = price is above VWAP (volume-weighted average price). Above VWAP means buyers in control of the session; a reclaim from below is a classic ignition trigger.",
            "HOD": "High of day status. NEW = just broke to a new session high (breakout trigger). near = within 0.5% of the high, coiling under it.",
            "RSI5": "RSI(14) on 5-minute bars. 55-75 is the momentum thrust zone (strong but not exhausted). Above 75 = stretched, chase risk rises. Below 50 = no momentum yet.",
            "Short%Flt": "Short interest as a percent of float. 15%+ means heavy bets against the stock - squeeze fuel if price starts running and shorts are forced to cover.",
            "InsiderNet$": "Net dollar value of insider buys minus sells over the last 90 days (SEC Form 4 filings). Positive = the people who know the company best are accumulating.",
            "News48h": "Number of news headlines in the last 48 hours. Momentum needs a catalyst; no news usually means no sustained move.",
            "Signals": "Plain-English list of which triggers are currently firing for this ticker.",
        }
        rows = []
        for r in ok:
            f = r["fuel"]
            spf = f.get("short_pct_float")
            rows.append({
                "Ticker": r["ticker"],
                "Score": round(r["score"], 1),
                "NightlyRank": round(r["pre_rank"], 0) if r.get("pre_rank") is not None else None,
                "Ignition": round(r["ignition_score"], 1),
                "Fuel": round(r["fuel_score"], 1),
                "Price": round(r["price"], 2) if r["price"] else None,
                "Chg %": round(r["chg_pct"], 2) if r["chg_pct"] is not None else None,
                "RVOL": round(r["rvol"], 2),
                "Surge": round(r["surge"], 2),
                "Vel %/5m": round(r["velocity"], 2),
                "VWAP+": "Y" if r["above_vwap"] else "",
                "HOD": "NEW" if r["new_hod"] else ("near" if r["near_hod"] else ""),
                "RSI5": round(r["rsi5"], 0) if r["rsi5"] is not None else None,
                "Short%Flt": round(spf * 100, 1) if spf else None,
                "InsiderNet$": f.get("insider_net_buy_usd", 0.0),
                "News48h": f.get("news_count_48h", 0),
                "Signals": ", ".join(r["reasons"][:5]),
            })
        df = pd.DataFrame(rows)
        if not show_all_cols:
            keep = ["Ticker", "Score", "NightlyRank", "Ignition", "Fuel"]
            df = df[[c for c in keep if c in df.columns]]
            if "NightlyRank" in df.columns and df["NightlyRank"].isna().all():
                df = df.drop(columns=["NightlyRank"])
        col_cfg = {
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f", help=HELP["Score"]),
            "Ignition": st.column_config.ProgressColumn("Ignition", min_value=0, max_value=100, format="%.0f", help=HELP["Ignition"]),
            "Fuel": st.column_config.ProgressColumn("Fuel", min_value=0, max_value=100, format="%.0f", help=HELP["Fuel"]),
            "InsiderNet$": st.column_config.NumberColumn("InsiderNet$", format="$%.0f", help=HELP["InsiderNet$"]),
        }
        for col in df.columns:
            if col not in col_cfg:
                col_cfg[col] = st.column_config.Column(col, help=HELP.get(col, ""))
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=col_cfg,
        )
    else:
        if never_scanned:
            st.info("Hit **Scan** to run. Nothing is fetched until then.")
        else:
            st.warning("No data returned. Market may be closed, or tickers invalid.")

    if failed:
        st.caption(f"No data for: {', '.join(failed)}")

    with st.expander("Metric guide - what everything means"):
        st.markdown("""
    **The two halves of the Score**

    - **Ignition (60% of Score)** answers *"is money flowing in right now?"* It is rebuilt
      from live bars every refresh. Its loudest inputs are **RVOL** (today's volume vs the
      20-day norm for this time of day) and the **volume Surge** in the last 3 minutes.
      Price **Velocity/Acceleration**, a **VWAP** reclaim, a **new High of Day**, RSI in the
      55-75 thrust zone, and a fresh MACD cross round it out.
    - **Fuel (40% of Score)** answers *"is this stock primed to make a big move?"* High
      **short % of float** means forced buyers if price runs. **Insider net buying** means
      informed accumulation. **Fresh news** provides the catalyst. **Small float** makes
      moves violent. **Near 52-week highs** is where momentum lives.

    **IGNITING NOW banner / urgent phone alert**

    Fires only when ALL of these confirm on the same refresh - the footprint of the first
    minutes of a real momentum leg:
    - RVOL at least 2x normal pace (pace follows the real U-shaped intraday volume curve,
      so 9:45 AM readings are no longer inflated)
    - Last-3-bar volume surge at least 2x the session average
    - Positive 5-minute velocity
    - A new high of day OR a VWAP reclaim within the last few bars
    - AND the stock is NOT down hard on the day (no big gap-down, day change above -4%)

    **GAP REVERSAL banner (teal) / high-priority phone alert**

    The same live footprint firing while the stock is down 4%+ on the day or gapped down
    4%+ at the open - usually a post-earnings flush being bought. This is a bounce attempt
    inside a selloff: a real, tradable pattern, but a different and riskier trade than
    fresh ignition. Bounces in crushed stocks fail more often than breakouts in strong
    ones, which is why it gets its own label instead of the IGNITING banner.

    **Alert feed** logs each ticker once per day, the first time it crosses your score
    threshold or ignites. Phone pushes mirror the feed when ntfy is configured.

    **Chart**: candles are 1-minute bars for the current session; the amber line is VWAP -
    price above it means buyers control the session. Volume bars underneath confirm whether
    a move has real participation.

    **NightlyRank vs Score**: NightlyRank is how the stock graded in last night's screener
    dump (yesterday's homework). Score is the live grade. A high NightlyRank with a surging
    Ignition number is the combination this tool exists to catch.

    *Detects momentum early; does not predict the future. Not financial advice.*
    """)

    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    # Stock Analyzer  (replaces chart + alert feed)
    # ----------------------------------------------------------------------------
    st.markdown(
        "<div style='font-family:Rajdhani,sans-serif;font-size:22px;font-weight:700;"
        "color:#ffffff;letter-spacing:.6px;margin:18px 0 4px'>Stock Analyzer</div>",
        unsafe_allow_html=True,
    )

    if not ok:
        st.info("Run a scan first — then select a stock to analyze.")
    else:
        az_col1, az_col2 = st.columns([2, 3])
        with az_col1:
            az_ticker = st.selectbox(
                "Select stock",
                [r["ticker"] for r in ok],
                index=0,
                key="ig_az_ticker",
                help="Choose any stock from the current scan results to drill into full analysis.",
                label_visibility="collapsed",
            )

        # ── helper renderers ────────────────────────────────────────────
        # helpers and fetch_analyzer defined at module level

        info, hist, eps_history, eps_forward = fetch_analyzer(az_ticker)

        # ── Resolve scan result first — needed for fuel merge and px fallback ───
        scan_r  = next((r for r in ok if r["ticker"] == az_ticker), None)
        ign_sc  = scan_r["ignition_score"] if scan_r else None
        fuel_sc = scan_r["fuel_score"]     if scan_r else None
        rvol    = scan_r["rvol"]           if scan_r else None
        reasons = scan_r.get("reasons", []) if scan_r else []

        # ── Merge fetch_fuel into info (both hit yfinance separately; when one is
        # rate-limited the other may succeed — merging guarantees best available data)
        az_fuel = scan_r["fuel"] if scan_r else fetch_fuel(az_ticker)
        _AZ_FUEL_MAP = {
            "short_pct_float": "shortPercentOfFloat", "float_shares": "floatShares",
            "high_52w": "fiftyTwoWeekHigh", "days_to_cover": "shortRatio",
            "target_mean": "targetMeanPrice", "sector": "sector", "name": "shortName",
        }
        for fk, ik in _AZ_FUEL_MAP.items():
            fv = az_fuel.get(fk)
            if fv is not None and fv != "" and not info.get(ik):
                info[ik] = fv

        # ── Parse core fields with multi-source fallbacks ─────────────────
        px    = float(info.get("currentPrice") or info.get("regularMarketPrice") or
                      info.get("previousClose") or
                      (scan_r["price"] if scan_r and scan_r.get("price") else 0) or 0)
        name  = info.get("shortName") or info.get("longName") or az_fuel.get("name") or az_ticker
        sec   = info.get("sector")   or az_fuel.get("sector") or ""
        ind   = info.get("industry") or ""
        sec_display = f"{sec} / {ind}" if (sec and ind) else (sec or ind or "--")
        mcap  = info.get("marketCap")
        pe    = info.get("trailingPE")
        fwpe  = info.get("forwardPE")
        pb    = info.get("priceToBook")
        ps    = info.get("priceToSalesTrailing12Months")
        beta  = info.get("beta")
        hi52  = info.get("fiftyTwoWeekHigh") or az_fuel.get("high_52w") or 0
        lo52  = info.get("fiftyTwoWeekLow") or 0
        spf   = info.get("shortPercentOfFloat") or az_fuel.get("short_pct_float")
        sratio= info.get("shortRatio") or az_fuel.get("days_to_cover")
        am    = info.get("targetMeanPrice") or az_fuel.get("target_mean")
        al    = info.get("targetLowPrice");  ahigh = info.get("targetHighPrice")
        nana  = info.get("numberOfAnalystOpinions") or 0
        recky = info.get("recommendationKey") or ""
        rg    = info.get("revenueGrowth");   eg  = info.get("earningsGrowth")
        pm    = info.get("profitMargins");   om  = info.get("operatingMargins")
        roe   = info.get("returnOnEquity");  roa = info.get("returnOnAssets")
        deq   = info.get("debtToEquity");    cr  = info.get("currentRatio")
        fl    = info.get("floatShares") or az_fuel.get("float_shares")
        inst_pct = az_fuel.get("inst_pct")
        ins_buys = az_fuel.get("insider_buys", 0)
        ins_net  = az_fuel.get("insider_net_buy_usd", 0.0)
        news_cnt = az_fuel.get("news_count_48h", 0)
        aus   = ((am - px) / px * 100) if (am and px and px > 0) else None
        rng52 = ((px - lo52) / (hi52 - lo52) * 100) if (hi52 and lo52 and hi52 != lo52) else None

        # ── Technicals from history ──────────────────────────────────────
        rsi_v = ma50_v = ma200_v = macd_v = macd_s = vol_avg = vol_td = None
        pct1d = pct5d = pct1m = pct3m = atr = None
        bb_upper = bb_mid = bb_lower = None

        if not hist.empty and len(hist) >= 35:  # MACD needs 26+9=35 minimum
            cl = hist["Close"].dropna()
            vl = hist["Volume"].dropna()
            try:
                dlt = cl.diff(); g = dlt.clip(lower=0).rolling(14).mean()
                ls = (-dlt.clip(upper=0)).rolling(14).mean()
                rs3 = (100 - (100 / (1 + g / ls.replace(0, np.nan)))).dropna()
                rsi_v = float(rs3.iloc[-1]) if not rs3.empty else None
            except Exception: pass
            try:
                e12 = cl.ewm(span=12, adjust=False).mean()
                e26 = cl.ewm(span=26, adjust=False).mean()
                ml = e12 - e26; sl = ml.ewm(span=9, adjust=False).mean()
                macd_v = float(ml.iloc[-1]); macd_s = float(sl.iloc[-1])
            except Exception: pass
            try:
                if len(cl) >= 50:  ma50_v  = float(cl.rolling(50).mean().iloc[-1])
                if len(cl) >= 200: ma200_v = float(cl.rolling(200).mean().iloc[-1])
            except Exception: pass
            try:
                if len(vl) >= 21:
                    vol_avg = float(vl.iloc[-21:-1].mean())  # exclude today's partial bar
                    # Derive today's vol from RVOL scan signal to avoid false 'Low' mid-day
                    if scan_r and scan_r.get("rvol") and vol_avg > 0:
                        vol_td = vol_avg * scan_r["rvol"] * expected_vol_fraction(390)
                    else:
                        vol_td = None
            except Exception: pass
            try:
                if len(cl) >= 2:  pct1d = (float(cl.iloc[-1]) - float(cl.iloc[-2])) / float(cl.iloc[-2]) * 100
                if len(cl) >= 6:  pct5d = (float(cl.iloc[-1]) - float(cl.iloc[-6])) / float(cl.iloc[-6]) * 100
                if len(cl) >= 22: pct1m = (float(cl.iloc[-1]) - float(cl.iloc[-22])) / float(cl.iloc[-22]) * 100
                if len(cl) >= 66: pct3m = (float(cl.iloc[-1]) - float(cl.iloc[-66])) / float(cl.iloc[-66]) * 100
            except Exception: pass
            try:
                # ATR (14-day Average True Range)
                hi = hist["High"].dropna(); lo = hist["Low"].dropna()
                tr = pd.concat([hi - lo,
                                (hi - cl.shift()).abs(),
                                (lo - cl.shift()).abs()], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])
            except Exception: pass
            try:
                # Bollinger Bands (20-day, 2 std)
                if len(cl) >= 20:
                    bm = cl.rolling(20).mean()
                    bstd = cl.rolling(20).std()
                    bb_mid = float(bm.iloc[-1])
                    bb_upper = float((bm + 2 * bstd).iloc[-1])
                    bb_lower = float((bm - 2 * bstd).iloc[-1])
            except Exception: pass

        # ── Signal pills ─────────────────────────────────────────────────
        pills_html = ""
        if rsi_v is not None:
            if rsi_v < 30:   pills_html += az_pill("RSI Oversold", True)
            elif rsi_v > 70: pills_html += az_pill("RSI Overbought", False)
            elif 45 < rsi_v < 65: pills_html += az_pill("RSI Sweet Spot", True)
            else:            pills_html += az_pill("RSI Neutral", None)
        if macd_v is not None and macd_s is not None:
            pills_html += az_pill("MACD Bullish" if macd_v > macd_s else "MACD Bearish", macd_v > macd_s)
        if ma50_v and ma200_v:
            pills_html += az_pill("Golden Cross" if ma50_v > ma200_v else "Death Cross", ma50_v > ma200_v)
        if vol_avg and vol_td:
            if vol_td > vol_avg * 1.5: pills_html += az_pill("High Volume", True)
            elif vol_td < vol_avg * 0.5: pills_html += az_pill("Low Volume", None)
        if spf and spf > 0.15: pills_html += az_pill("High Short Interest", None)
        if aus is not None and aus > 15:
            pills_html += az_pill(f"Analyst Upside {aus:.0f}%", True)
        elif aus is not None and aus < -10:
            pills_html += az_pill(f"Above Target {aus:.0f}%", False)
        if scan_r and scan_r.get("igniting"): pills_html += az_pill("IGNITING NOW", True)
        if scan_r and scan_r.get("gap_reversal"): pills_html += az_pill("GAP REVERSAL", False)
        if scan_r and scan_r.get("new_hod"): pills_html += az_pill("New HOD", True)

        # ── Header metrics ───────────────────────────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Price", f"${px:.2f}" if px else "--")
        m2.metric("Ignition", f"{ign_sc:.0f}" if ign_sc is not None else "--")
        m3.metric("Fuel", f"{fuel_sc:.0f}" if fuel_sc is not None else "--")
        m4.metric("RVOL", f"{rvol:.1f}x" if rvol else "--")
        m5.metric("52W Pos", f"{rng52:.0f}%" if rng52 is not None else "--")
        m6.metric("Target", f"${am:.2f}" if am else "--", delta=f"{aus:.1f}%" if aus else None)

        st.markdown(
            f"<div style='margin:6px 0 4px;font-family:Plus Jakarta Sans,sans-serif'>"
            f"<strong style='color:#ffffff'>{name}</strong>"
            f"  <span style='color:#7a9ab8;font-size:13px'>{sec_display}</span></div>",
            unsafe_allow_html=True,
        )
        if pills_html:
            st.markdown(f"<div style='margin:6px 0 14px'>{pills_html}</div>", unsafe_allow_html=True)

        # Data source badge
        src_parts = []
        _ak_present = alpaca_keys() is not None
        if _ak_present and not hist.empty:
            src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#4dd880;border:1px solid #1e6b35;border-radius:3px;padding:1px 7px'>Alpaca history</span>")
        else:
            src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#7a9ab8;border:1px solid #1e3a5f;border-radius:3px;padding:1px 7px'>Yahoo history</span>")
        if info.get("_from_scan_dump"):
            src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#d0b040;border:1px solid #907020;border-radius:3px;padding:1px 7px'>nightly dump fallback</span>")
        st.markdown("<div style='margin:0 0 10px;display:flex;gap:6px;flex-wrap:wrap'>" + "".join(src_parts) + "</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0'>", unsafe_allow_html=True)

        # ── Three column layout ──────────────────────────────────────────
        colA, colB, colC = st.columns(3)

        # ── COLUMN A: Price Range Analysis ──────────────────────────────
        with colA:
            az_section("Price Range Analysis")

            # 52-week channel visual
            if hi52 and lo52 and px:
                pct_pos = max(0.0, min(1.0, (px - lo52) / (hi52 - lo52))) if hi52 != lo52 else 0.5
                bar_pct = int(pct_pos * 100)
                # Color: near low=red, mid=amber, near high=green
                bar_col = "#4dd880" if pct_pos > 0.7 else ("#d0b040" if pct_pos > 0.35 else "#ff4444")
                st.markdown(
                    f"<div style='background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;"
                    f"padding:14px 16px;margin-bottom:12px'>"
                    f"<div style='display:flex;justify-content:space-between;font-family:Space Mono,monospace;"
                    f"font-size:11px;color:#7a9ab8;margin-bottom:6px'>"
                    f"<span>52W Low ${lo52:.2f}</span><span>52W High ${hi52:.2f}</span></div>"
                    f"<div style='background:#122540;border-radius:4px;height:8px;position:relative;overflow:visible'>"
                    f"<div style='background:{bar_col};width:{bar_pct}%;height:100%;border-radius:4px'></div>"
                    f"<div style='position:absolute;top:-18px;left:calc({bar_pct}% - 1px);"
                    f"font-family:Space Mono,monospace;font-size:10px;color:{bar_col}'>"
                    f"${px:.2f}</div></div>"
                    f"<div style='margin-top:10px;font-family:Space Mono,monospace;font-size:11px;color:#b0c8e8'>"
                    f"Position: <strong style='color:{bar_col}'>{rng52:.1f}% of 52W range</strong></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Bollinger Bands channel
            if bb_upper and bb_lower and bb_mid and px:
                bw = bb_upper - bb_lower
                bpos = max(0.0, min(1.0, (px - bb_lower) / bw)) if bw > 0 else 0.5
                bpct = int(bpos * 100)
                bcol = "#ff4444" if bpos > 0.85 else ("#4dd880" if bpos < 0.15 else "#8baac8")
                bb_label = "Near upper band (overbought)" if bpos > 0.85 else ("Near lower band (oversold)" if bpos < 0.15 else "Inside bands (neutral)")
                st.markdown(
                    f"<div style='background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;"
                    f"padding:14px 16px;margin-bottom:12px'>"
                    f"<div style='display:flex;justify-content:space-between;font-family:Space Mono,monospace;"
                    f"font-size:11px;color:#7a9ab8;margin-bottom:6px'>"
                    f"<span>BB Lower ${bb_lower:.2f}</span><span>BB Upper ${bb_upper:.2f}</span></div>"
                    f"<div style='background:#122540;border-radius:4px;height:8px;position:relative;overflow:visible'>"
                    f"<div style='position:absolute;left:50%;width:1px;height:100%;background:#1e3a5f'></div>"
                    f"<div style='background:{bcol};width:{bpct}%;height:100%;border-radius:4px'></div>"
                    f"<div style='position:absolute;top:-18px;left:calc({bpct}% - 1px);"
                    f"font-family:Space Mono,monospace;font-size:10px;color:{bcol}'>${px:.2f}</div></div>"
                    f"<div style='margin-top:10px;font-family:Space Mono,monospace;font-size:11px;color:#b0c8e8'>"
                    f"Bollinger: <strong style='color:{bcol}'>{bb_label}</strong></div>"
                    f"<div style='margin-top:4px;font-family:Space Mono,monospace;font-size:11px;color:#7a9ab8'>"
                    f"Mid (20MA): ${bb_mid:.2f} · Width: ${bw:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            az_section("Price Performance")
            perf_rows = [
                mrow("1 Day", "Daily price change vs yesterday's close.", pct_color(pct1d)),
                mrow("5 Day", "Five trading days — roughly one calendar week.", pct_color(pct5d)),
                mrow("1 Month", "~22 trading days. Shows the near-term trend.", pct_color(pct1m)),
                mrow("3 Month", "~66 trading days (one quarter). Used by institutions.", pct_color(pct3m)),
            ]
            st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(perf_rows)}</tbody></table>", unsafe_allow_html=True)

        # ── COLUMN B: Technical Signals + Short/Growth ───────────────────
        with colB:
            az_section("Technical Signals")
            tech_rows = []
            if rsi_v is not None:
                if rsi_v < 30:    ri, rc = "Oversold — potential bounce", "#4dd880"
                elif rsi_v < 45:  ri, rc = "Weak — losing momentum", "#ff4444"
                elif rsi_v < 55:  ri, rc = "Neutral — no clear direction", "#8baac8"
                elif rsi_v < 70:  ri, rc = "Strong — uptrend confirmed", "#4dd880"
                else:             ri, rc = "Overbought — pullback possible", "#ff4444"
                tech_rows.append(mrow("RSI (14d)", "Relative Strength Index 0-100. Below 30 = oversold. Above 70 = overbought. 45-70 = momentum sweet spot.", f"<span style='color:{rc};font-family:Space Mono,monospace'>{rsi_v:.1f}</span> <span style='font-size:11px;color:#7a9ab8'>{ri}</span>"))
            if macd_v is not None:
                mi = "Bullish — momentum building" if macd_v > macd_s else "Bearish — momentum fading"
                mc2 = "#4dd880" if macd_v > macd_s else "#ff4444"
                tech_rows.append(mrow("MACD", "Moving Average Convergence Divergence. MACD above signal line = buyers in control.", f"<span style='color:{mc2};font-family:Space Mono,monospace'>{macd_v:.4f}</span> <span style='font-size:11px;color:#7a9ab8'>{mi}</span>"))
            if ma50_v and px:
                pvs = (px - ma50_v) / ma50_v * 100
                m5i = "Extended — may be overbought" if pvs > 5 else ("Just above — ideal zone" if pvs > 0 else ("Just below — watch reclaim" if pvs > -5 else "Well below — downtrend"))
                m5c = "#4dd880" if 0 < pvs < 5 else ("#d0b040" if pvs > 5 else ("#d0b040" if pvs > -5 else "#ff4444"))
                tech_rows.append(mrow("50-Day MA", "50-Day Moving Average. Price just above = support. Just below = watch for reclaim.", f"${ma50_v:.2f} <span style='color:{m5c};font-size:11px'>({'+' if pvs >= 0 else ''}{pvs:.1f}%)</span>"))
            if ma200_v:
                m2i = "Golden Cross — long-term uptrend" if (ma50_v and ma50_v > ma200_v) else "Death Cross — long-term downtrend"
                m2c = "#4dd880" if (ma50_v and ma50_v > ma200_v) else "#ff4444"
                tech_rows.append(mrow("200-Day MA", "200-Day Moving Average. Golden Cross (50MA above 200MA) = major bull signal.", f"${ma200_v:.2f} <span style='font-size:11px;color:{m2c}'>{m2i}</span>"))
            if vol_avg and vol_td:
                vr = vol_td / vol_avg
                vi = "Very high" if vr > 2 else ("High" if vr > 1.5 else ("Normal" if vr > 0.5 else "Low"))
                vc = "#4dd880" if vr > 1.5 else ("#8baac8" if vr > 0.5 else "#d0b040")
                tech_rows.append(mrow("Volume", "Today's volume vs 20-day average. High volume confirms moves.", f"<span style='color:{vc};font-family:Space Mono,monospace'>{vr:.2f}x avg</span> <span style='font-size:11px;color:#7a9ab8'>{vi}</span>"))
            if ign_sc is not None:
                ic = "#f5a623" if ign_sc >= 70 else ("#c47d0e" if ign_sc >= 50 else "#4a6a8a")
                tech_rows.append(mrow("Ignition Score", "Live momentum score from RVOL, volume surge, velocity, VWAP, HOD, RSI, MACD (0-100).", f"<span style='color:{ic};font-family:Space Mono,monospace;font-size:15px;font-weight:500'>{ign_sc:.0f}</span>"))
            if tech_rows:
                st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(tech_rows)}</tbody></table>", unsafe_allow_html=True)

            az_section("Short Interest & Growth")
            si_rows = []
            if spf is not None:
                si_rows.append(mrow("Short % Float",  "% of float sold short. 15%+ = squeeze fuel if price runs.",          az_tag(spf * 100, 20, 10, "{:.1f}", "%")))
            if sratio is not None:
                si_rows.append(mrow("Days to Cover",   "Shares short / avg vol. High = forced shorts if price rises.",        az_tag(sratio, 5, 3, "{:.1f}", "d")))
            if ins_buys > 0:
                ins_col = "#4dd880" if ins_net > 0 else "#ff4444"
                si_rows.append(mrow("Insider Buying",  "Net insider buys last 90d (SEC Form 4).", f"<span style='font-family:Space Mono,monospace;color:{ins_col}'>{ins_buys} buys · ${ins_net/1e3:.0f}K net</span>"))
            elif ins_net < 0:
                si_rows.append(mrow("Insider Activity","Net insider selling last 90d.", "<span style='font-family:Space Mono,monospace;color:#ff4444'>net selling</span>"))
            if news_cnt:
                si_rows.append(mrow("News 48h",        "Headlines in last 48 hours. Momentum needs a catalyst.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{news_cnt} headlines</span>"))
            if rg is not None:
                si_rows.append(mrow("Revenue Growth",  "YoY revenue growth. Double-digit = attractive to institutions.",      az_tag(rg * 100, 10, 3, "{:.1f}", "%")))
            if eg is not None:
                si_rows.append(mrow("Earnings Growth", "YoY EPS growth. Growing faster than revenue = margin expansion.",     az_tag(eg * 100, 10, 3, "{:.1f}", "%")))
            _esp_pct = az_fuel.get("earnings_surprise")
            if _esp_pct is not None:
                _esp_col = "#4dd880" if _esp_pct >= 10 else ("#d0b040" if _esp_pct >= 0 else "#ff4444")
                si_rows.append(mrow("Earnings Surprise", "Last quarter EPS vs analyst estimates. 10%+ beat = strong momentum fuel.", f"<span style='font-family:Space Mono,monospace;color:{_esp_col}'>{_esp_pct:+.1f}%</span> vs estimates"))
            if fuel_sc is not None:
                fc = "#4dd880" if fuel_sc >= 70 else ("#d0b040" if fuel_sc >= 50 else "#4a6a8a")
                si_rows.append(mrow("Fuel Score", "Primed-to-move score (0-100).", f"<span style='color:{fc};font-family:Space Mono,monospace;font-size:15px;font-weight:500'>{fuel_sc:.0f}</span>"))
            if si_rows:
                st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(si_rows)}</tbody></table>", unsafe_allow_html=True)
            else:
                _why = "; ".join(info.get("_data_issues", [])) or "short interest not published for this symbol (common for ETFs and foreign listings)"
                st.caption(f"Short interest unavailable — {_why}")

            az_section("Live Signals")
            if reasons:
                sig_html = "".join(
                    f"<span style='display:inline-block;background:#0d1e33;color:#f5a623;"
                    f"border:1px solid #1e3a5f;border-radius:4px;font-family:Space Mono,monospace;"
                    f"font-size:11px;padding:2px 8px;margin:2px 3px 2px 0'>{r}</span>"
                    for r in reasons
                )
                st.markdown(sig_html, unsafe_allow_html=True)
            else:
                st.caption("No active signals this scan.")

            az_section("Earnings Breakdown (EPS Trend)")
            render_eps_trend(eps_history, eps_forward, why='; '.join(info.get('_data_issues', [])))

            az_section("Dividend")
            render_dividend_info(info)

        # ── COLUMN C: Valuation + Financial Health + Analyst ────────────
        with colC:
            az_section("Valuation")
            val_rows = []
            if mcap is not None:
                mc_str = f"${mcap/1e9:.2f}B" if mcap >= 1e9 else f"${mcap/1e6:.1f}M"
                val_rows.append(mrow("Market Cap", "Total market value = share price × shares outstanding. Determines if this is micro/small/mid/large cap.", mc_str))
            if pe is not None:
                val_rows.append(mrow("P/E Ratio", "Price-to-Earnings. How much investors pay per dollar of earnings. High P/E = growth expectations.", az_tag(pe, 0, 25, "{:.1f}", "x") if pe < 100 else f"<span style='font-family:Space Mono,monospace;color:#d0b040'>{pe:.1f}x</span>"))
            if fwpe is not None:
                val_rows.append(mrow("Forward P/E", "P/E based on next 12 months estimated earnings. Lower than trailing P/E = growth expected.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{fwpe:.1f}x</span>"))
            if pb is not None:
                val_rows.append(mrow("P/B Ratio", "Price-to-Book. Below 1.0 = trading below asset value. Above 3.0 = premium for brand/growth.", az_tag(pb, 0, 3, "{:.2f}", "x")))
            if ps is not None:
                val_rows.append(mrow("P/S Ratio", "Price-to-Sales. Below 2x = reasonable. Above 10x = high growth premium.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{ps:.2f}x</span>"))
            if beta is not None:
                val_rows.append(mrow("Beta", "Volatility vs S&P 500. 1.0 = market. 1.5 = 50% more volatile. Under 0.6 = defensive.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{beta:.2f}</span>"))
            if fl is not None:
                fl_str = f"{fl/1e9:.2f}B" if fl >= 1e9 else f"{fl/1e6:.0f}M"
                val_rows.append(mrow("Float", "Tradeable shares. <20M = explosive on volume.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{fl_str}</span>"))
            if inst_pct is not None:
                val_rows.append(mrow("Institutional", "% held by institutions.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{inst_pct:.1f}%</span>"))
            if val_rows:
                st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(val_rows)}</tbody></table>", unsafe_allow_html=True)
            else:
                _why = "; ".join(info.get("_data_issues", [])) or "yfinance returned no valuation fields — likely rate-limited; press Refresh"
                st.caption(f"Valuation unavailable — {_why}")

            az_section("Financial Health")
            hlth_rows = []
            if pm  is not None: hlth_rows.append(mrow("Profit Margin", "Net income / revenue. Expanding = pricing power.", az_tag(pm * 100, 15, 5, "{:.1f}", "%")))
            if om  is not None: hlth_rows.append(mrow("Operating Margin", "EBIT / revenue. Core business efficiency before interest and taxes.", az_tag(om * 100, 15, 5, "{:.1f}", "%")))
            if roe is not None: hlth_rows.append(mrow("Return on Equity", "Net income / shareholders equity. Above 15% = strong.", az_tag(roe * 100, 15, 8, "{:.1f}", "%")))
            if roa is not None: hlth_rows.append(mrow("Return on Assets", "Net income / total assets. Above 5% = solid.", az_tag(roa * 100, 8, 3, "{:.1f}", "%")))
            if deq is not None:
                hlth_rows.append(mrow("D/E Ratio", "Total debt / equity. High D/E amplifies gains and losses. Capital-intensive sectors carry more.", f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{deq:.1f}%</span>"))
            if cr is not None:
                hlth_rows.append(mrow("Current Ratio", "Current assets / current liabilities. Above 1.5 = comfortable. Below 1.0 = short-term risk.", az_tag(cr, 1.5, 1.0, "{:.2f}")))
            if hlth_rows:
                st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(hlth_rows)}</tbody></table>", unsafe_allow_html=True)

            if am:
                az_section("Analyst Consensus")
                rcol = "#4dd880" if "buy" in recky.lower() else ("#ff4444" if "sell" in recky.lower() else "#d0b040")
                rdisp = recky.replace("_", " ").title() if recky else "--"
                an_rows = [
                    mrow("Recommendation", "Wall Street consensus: Strong Buy / Buy / Hold / Sell. Aggregates all analyst ratings.", f"<span style='color:{rcol};font-weight:500'>{rdisp}</span> <span style='font-size:11px;color:#7a9ab8'>({nana} analysts)</span>"),
                    mrow("Mean Target", "Average 12-month analyst price target. Implies expected upside/downside from current price.", f"${am:.2f}" + (f" <span style='font-size:11px;color:{'#4dd880' if aus and aus > 0 else '#ff4444'}'>({'+' if aus and aus >= 0 else ''}{aus:.1f}%)</span>" if aus else "")),
                    mrow("Target Range", "Low-to-high analyst target spread. Wide range = high uncertainty. Narrow = strong consensus.", f"${al:.2f} – ${ahigh:.2f}" if (al and ahigh) else "--"),
                ]
                st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(an_rows)}</tbody></table>", unsafe_allow_html=True)

            # Catalyst tags from scan
            f_data = scan_r["fuel"] if scan_r else {}
            cat_tags = f_data.get("catalyst_tags") or []
            bimodal = f_data.get("bimodal_event", False)
            dtc_val = f_data.get("days_to_cover")
            ed = f_data.get("earnings_days")
            if cat_tags or bimodal or (dtc_val and dtc_val >= 5):
                az_section("Catalysts Detected")
                cat_html = build_catalyst_tags_html(az_ticker, cat_tags, bimodal, dtc_val, ed)
                suppressed = f_data.get("catalyst_suppressed") or []
                if suppressed:
                    for s in suppressed:
                        cat_html += (
                            f"<span style='font-family:Space Mono,monospace;font-size:10px;"
                            f"color:#7a9ab8;border:1px dashed #1e3a5f;border-radius:4px;"
                            f"padding:1px 6px;margin:2px 3px 2px 0;text-decoration:line-through;"
                            f"display:inline-block'>{s.upper()}</span>"
                        )
                if f_data.get("latest_headline"):
                    cat_html += f"<div style='margin-top:8px;font-size:11px;color:#7a9ab8'>{f_data['latest_headline']}</div>"
                st.markdown(cat_html, unsafe_allow_html=True)



    # Close the Scanner tab context entered above
    tab_scan.__exit__(None, None, None)

    # ----------------------------------------------------------------------------
    # Stock Lookup tab — enter any ticker or company name, get full analysis
    # ----------------------------------------------------------------------------
    with tab_lookup:
        st.markdown(
            "<div style='font-family:Rajdhani,sans-serif;font-size:20px;font-weight:700;"
            "color:#ffffff;letter-spacing:.6px;margin-bottom:4px'>Stock Lookup</div>"
            "<div style='font-family:Plus Jakarta Sans,sans-serif;font-size:13px;"
            "color:#7a9ab8;margin-bottom:14px'>Enter any ticker symbol or company name. "
            "Full ignition scan + stock analyzer — Alpaca → Yahoo → nightly dump.</div>",
            unsafe_allow_html=True,
        )

        lk_col1, lk_col2 = st.columns([3, 1])
        with lk_col1:
            lk_input = st.text_input(
                "Ticker or company name",
                placeholder="e.g.  NVDA   or   Nvidia Corp",
                label_visibility="collapsed",
                key="ig_lk_query",
            )
        with lk_col2:
            lk_run = st.button("Analyze ▶", type="primary", use_container_width=True,
                               key="ig_lk_run")

        @st.cache_data(ttl=3600, show_spinner=False)
        def resolve_ticker(query: str):
            import re as _re
            q = query.strip().upper()
            if _re.fullmatch(r"[A-Z]{1,5}([.\-][A-Z]{1,2})?", q):
                try:
                    info = _yf(lambda: yf.Ticker(q).info or {})
                    name = info.get("shortName") or info.get("longName") or q
                    if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"):
                        return q, name
                except Exception:
                    pass
            try:
                results = _yf(lambda: yf.Search(query.strip(), max_results=5).quotes)
                if results:
                    best = results[0]
                    return best.get("symbol", q), best.get("shortname") or best.get("longname") or q
            except Exception:
                pass
            return q, q

        lk_ticker = None
        if lk_input and (lk_run or st.session_state.get("ig_lk_last_input") == lk_input):
            st.session_state["ig_lk_last_input"] = lk_input
            with st.spinner("Resolving ticker…"):
                lk_ticker, lk_name = resolve_ticker(lk_input)
            if lk_ticker:
                st.markdown(
                    f"<div style='font-family:Space Mono,monospace;font-size:12px;"
                    f"color:#7a9ab8;margin:4px 0 12px'>Analyzing "
                    f"<strong style='color:#f5a623'>{lk_ticker}</strong> — {lk_name}</div>",
                    unsafe_allow_html=True,
                )

        if lk_ticker:
            # ── Step 1: Live ignition signals (Alpaca → Yahoo) ────────────
            with st.spinner(f"Running live ignition scan on {lk_ticker}…"):
                lk_sig  = compute_signals(lk_ticker)
                lk_fuel = fetch_fuel(lk_ticker)
                lk_sig["fuel"] = lk_fuel

            # ── Step 2: Deep fundamentals (Alpaca history + Yahoo + dump) ─
            with st.spinner("Loading fundamentals…"):
                lk_info, lk_hist, lk_eps_history, lk_eps_forward = fetch_analyzer(lk_ticker)

            # ── Merge fuel dict into info to fill gaps ───────────────────────
            # fetch_fuel and fetch_analyzer both hit yfinance but cache independently.
            # Merge the fuel fields so we never display "Unknown" when fuel has the data.
            FUEL_TO_INFO = {
                "short_pct_float":   "shortPercentOfFloat",
                "float_shares":      "floatShares",
                "high_52w":          "fiftyTwoWeekHigh",
                "days_to_cover":     "shortRatio",
                "target_mean":       "targetMeanPrice",
                "sector":            "sector",
                "name":              "shortName",
            }
            for fk, ik in FUEL_TO_INFO.items():
                fv = lk_fuel.get(fk)
                if fv and not lk_info.get(ik):
                    lk_info[ik] = fv

            # ── Parse all fields ──────────────────────────────────────────
            px    = float(lk_info.get("currentPrice") or lk_info.get("regularMarketPrice") or
                          lk_info.get("previousClose") or lk_sig.get("price") or 0)
            name  = lk_info.get("shortName") or lk_info.get("longName") or lk_name or lk_ticker
            sec   = lk_info.get("sector")   or ""
            ind   = lk_info.get("industry") or ""
            mcap  = lk_info.get("marketCap") or 0
            pe    = lk_info.get("trailingPE")
            fwpe  = lk_info.get("forwardPE")
            pb    = lk_info.get("priceToBook")
            ps    = lk_info.get("priceToSalesTrailing12Months")
            beta  = lk_info.get("beta") or lk_fuel.get("_scan_beta")
            hi52  = lk_info.get("fiftyTwoWeekHigh") or lk_fuel.get("high_52w") or 0
            lo52  = lk_info.get("fiftyTwoWeekLow")  or 0
            spf   = lk_info.get("shortPercentOfFloat") or lk_fuel.get("short_pct_float")
            sratio= lk_info.get("shortRatio") or lk_fuel.get("days_to_cover")
            am    = lk_info.get("targetMeanPrice") or lk_fuel.get("target_mean")
            al    = lk_info.get("targetLowPrice")
            ahigh = lk_info.get("targetHighPrice")
            nana  = lk_info.get("numberOfAnalystOpinions") or 0
            recky = lk_info.get("recommendationKey") or ""
            rg    = lk_info.get("revenueGrowth")
            eg    = lk_info.get("earningsGrowth")
            pm    = lk_info.get("profitMargins")
            om    = lk_info.get("operatingMargins")
            roe   = lk_info.get("returnOnEquity")
            roa   = lk_info.get("returnOnAssets")
            deq   = lk_info.get("debtToEquity")
            cr    = lk_info.get("currentRatio")
            fl    = lk_info.get("floatShares") or lk_fuel.get("float_shares")
            ins_buys = lk_fuel.get("insider_buys", 0)
            ins_net  = lk_fuel.get("insider_net_buy_usd", 0.0)
            news_cnt = lk_fuel.get("news_count_48h", 0)
            inst_pct = lk_fuel.get("inst_pct")
            aus   = ((am - px) / px * 100) if (am and px > 0) else None
            rng52 = ((px - lo52) / (hi52 - lo52) * 100) if (hi52 and lo52 and hi52 != lo52) else None
            # Sector/industry display — show "--" if empty rather than "Unknown"
            sec_display = f"{sec} / {ind}" if (sec and ind) else (sec or ind or "--")

            # ── Technicals from history ────────────────────────────────────
            rsi_v = ma50_v = ma200_v = macd_v = macd_s = vol_avg = vol_td = None
            pct1d = pct5d = pct1m = pct3m = atr = bb_upper = bb_mid = bb_lower = None

            if not lk_hist.empty and len(lk_hist) >= 26:
                cl = lk_hist["Close"].dropna()
                vl = lk_hist["Volume"].dropna() if "Volume" in lk_hist.columns else pd.Series(dtype=float)
                try:
                    dlt = cl.diff(); g = dlt.clip(lower=0).rolling(14).mean()
                    ls  = (-dlt.clip(upper=0)).rolling(14).mean()
                    rs3 = (100 - (100 / (1 + g / ls.replace(0, np.nan)))).dropna()
                    rsi_v = float(rs3.iloc[-1]) if not rs3.empty else None
                except Exception: pass
                try:
                    e12 = cl.ewm(span=12, adjust=False).mean(); e26 = cl.ewm(span=26, adjust=False).mean()
                    ml  = e12 - e26; sl = ml.ewm(span=9, adjust=False).mean()
                    macd_v = float(ml.iloc[-1]); macd_s = float(sl.iloc[-1])
                except Exception: pass
                try:
                    if len(cl) >= 50:  ma50_v  = float(cl.rolling(50).mean().iloc[-1])
                    if len(cl) >= 200: ma200_v = float(cl.rolling(200).mean().iloc[-1])
                except Exception: pass
                try:
                    if len(vl) >= 20: vol_avg = float(vl.iloc[-20:].mean()); vol_td = float(vl.iloc[-1])
                except Exception: pass
                try:
                    if len(cl) >= 2:  pct1d = (float(cl.iloc[-1]) - float(cl.iloc[-2]))  / float(cl.iloc[-2])  * 100
                    if len(cl) >= 6:  pct5d = (float(cl.iloc[-1]) - float(cl.iloc[-6]))  / float(cl.iloc[-6])  * 100
                    if len(cl) >= 22: pct1m = (float(cl.iloc[-1]) - float(cl.iloc[-22])) / float(cl.iloc[-22]) * 100
                    if len(cl) >= 66: pct3m = (float(cl.iloc[-1]) - float(cl.iloc[-66])) / float(cl.iloc[-66]) * 100
                except Exception: pass
                try:
                    lhi = lk_hist["High"].dropna(); llo = lk_hist["Low"].dropna()
                    tr  = pd.concat([lhi - llo, (lhi - cl.shift()).abs(), (llo - cl.shift()).abs()], axis=1).max(axis=1)
                    atr = float(tr.rolling(14).mean().iloc[-1])
                except Exception: pass
                try:
                    if len(cl) >= 20:
                        bm = cl.rolling(20).mean(); bstd = cl.rolling(20).std()
                        bb_mid = float(bm.iloc[-1]); bb_upper = float((bm + 2*bstd).iloc[-1]); bb_lower = float((bm - 2*bstd).iloc[-1])
                except Exception: pass

            # ── Ignition/Fuel from live scan ───────────────────────────────
            ign_sc  = lk_sig.get("ignition_score")
            fuel_sc = lk_sig.get("fuel_score")
            rvol_v  = lk_sig.get("rvol")
            reasons = lk_sig.get("reasons", [])
            is_ign  = lk_sig.get("igniting", False)
            is_grev = lk_sig.get("gap_reversal", False)

            # ── Signal pills (same logic as Scanner Analyzer) ─────────────
            pills_html = ""
            if rsi_v is not None:
                if rsi_v < 30:            pills_html += az_pill("RSI Oversold", True)
                elif rsi_v > 70:          pills_html += az_pill("RSI Overbought", False)
                elif 45 < rsi_v < 65:     pills_html += az_pill("RSI Sweet Spot", True)
                else:                      pills_html += az_pill("RSI Neutral", None)
            if macd_v is not None and macd_s is not None:
                pills_html += az_pill("MACD Bullish" if macd_v > macd_s else "MACD Bearish", macd_v > macd_s)
            if ma50_v and ma200_v:
                pills_html += az_pill("Golden Cross" if ma50_v > ma200_v else "Death Cross", ma50_v > ma200_v)
            if vol_avg and vol_td:
                if vol_td > vol_avg * 1.5: pills_html += az_pill("High Volume", True)
                elif vol_td < vol_avg * 0.5: pills_html += az_pill("Low Volume", None)
            if spf and spf > 0.15:        pills_html += az_pill("High Short Interest", None)
            if aus and aus > 15:           pills_html += az_pill(f"Analyst Upside {aus:.0f}%", True)
            if is_ign:                     pills_html += az_pill("IGNITING NOW", True)
            if is_grev:                    pills_html += az_pill("GAP REVERSAL", False)
            if lk_sig.get("new_hod"):     pills_html += az_pill("New HOD", True)
            if lk_sig.get("vwap_cross"):  pills_html += az_pill("VWAP Reclaim", True)

            # ── Arc gauge ──────────────────────────────────────────────────
            sc = lk_sig["score"]
            gc = "#ff2200" if sc >= 80 else ("#ff6600" if sc >= 65 else "#f5a623" if sc >= 50 else "#d0b040" if sc >= 35 else "#29b6c8")
            gid = f"lk{abs(hash(lk_ticker)) % 9999}"
            arc_len = 125.7; dash_off = arc_len * (1 - min(sc, 100) / 100)
            arc_svg = (
                f"<svg width='100' height='58' viewBox='0 0 90 50' fill='none'>"
                f"<defs><linearGradient id='{gid}' x1='0%' y1='0%' x2='100%' y2='0%'>"
                f"<stop offset='0%' stop-color='#29b6c8'/><stop offset='30%' stop-color='#3ddc84'/>"
                f"<stop offset='60%' stop-color='#f5a623'/><stop offset='100%' stop-color='#ff2200'/>"
                f"</linearGradient></defs>"
                f"<path d='M5 45 A40 40 0 0 1 85 45' stroke='#122540' stroke-width='9' stroke-linecap='round' fill='none'/>"
                f"<path d='M5 45 A40 40 0 0 1 85 45' stroke='url(#{gid})' stroke-width='9' stroke-linecap='round' fill='none' "
                f"stroke-dasharray='{arc_len:.1f}' stroke-dashoffset='{dash_off:.1f}'/>"
                f"<text x='45' y='43' text-anchor='middle' font-family='Space Mono,monospace' font-size='18' font-weight='700' fill='{gc}'>{sc:.0f}</text>"
                f"</svg>"
            )

            # ── 6-column header metrics ────────────────────────────────────
            hm1, hm2, hm3, hm4, hm5, hm6 = st.columns(6)
            hm1.metric("Price",    f"${px:.2f}" if px else "--")
            hm2.metric("Ignition", f"{ign_sc:.0f}" if ign_sc is not None else "--")
            hm3.metric("Fuel",     f"{fuel_sc:.0f}" if fuel_sc is not None else "--")
            hm4.metric("RVOL",     f"{rvol_v:.1f}x" if rvol_v else "--")
            hm5.metric("52W Pos",  f"{rng52:.0f}%" if rng52 is not None else "--")
            hm6.metric("Target",   f"${am:.2f}" if am else "--", delta=f"{aus:.1f}%" if aus else None)

            st.markdown(
                f"<div style='margin:6px 0 4px;font-family:Plus Jakarta Sans,sans-serif'>"
                f"<strong style='color:#ffffff'>{name}</strong>"
                f"  <span style='color:#7a9ab8;font-size:13px'>{sec_display}</span></div>",
                unsafe_allow_html=True,
            )

            # Gauge + pills side by side
            cat_html = build_catalyst_tags_html(
                lk_ticker, lk_fuel.get("catalyst_tags") or [],
                lk_fuel.get("bimodal_event", False),
                lk_fuel.get("days_to_cover"), lk_fuel.get("earnings_days"),
            )
            if pills_html or cat_html:
                st.markdown(
                    f"<div style='display:flex;gap:16px;align-items:flex-start;margin:8px 0 14px'>"
                    f"<div>{arc_svg}</div>"
                    f"<div style='flex:1'><div style='margin-bottom:6px'>{pills_html}</div>{cat_html}</div>"
                    f"</div>", unsafe_allow_html=True,
                )
            else:
                st.markdown(arc_svg, unsafe_allow_html=True)

            # Suppressed catalysts
            suppressed = lk_fuel.get("catalyst_suppressed") or []
            if suppressed:
                sup_html = " ".join(
                    f"<span style='font-family:Space Mono,monospace;font-size:10px;color:#4a6a8a;"
                    f"border:1px dashed #1e3a5f;border-radius:4px;padding:1px 6px;margin-right:4px;"
                    f"text-decoration:line-through;display:inline-block'>{s.upper()}</span>"
                    for s in suppressed
                )
                st.markdown(
                    f"<div style='margin-bottom:12px;font-family:Space Mono,monospace;font-size:10px;color:#4a6a8a'>"
                    f"filtered (sector: {lk_fuel.get('sector','unknown')}): {sup_html}</div>",
                    unsafe_allow_html=True,
                )

            # Data source badge
            _ak_lk = alpaca_keys() is not None
            src_parts = []
            if _ak_lk and not lk_hist.empty:
                src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#4dd880;border:1px solid #1e6b35;border-radius:3px;padding:1px 7px'>Alpaca history</span>")
            else:
                src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#7a9ab8;border:1px solid #1e3a5f;border-radius:3px;padding:1px 7px'>Yahoo history</span>")
            if lk_info.get("_from_scan_dump"):
                src_parts.append("<span style='font-family:Space Mono,monospace;font-size:10px;color:#d0b040;border:1px solid #907020;border-radius:3px;padding:1px 7px'>nightly dump fallback</span>")
            st.markdown("<div style='margin:0 0 12px;display:flex;gap:6px;flex-wrap:wrap'>" + "".join(src_parts) + "</div>", unsafe_allow_html=True)

            st.markdown("<hr style='border-color:#1e3a5f;margin:12px 0'>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-family:Rajdhani,sans-serif;font-size:18px;font-weight:700;"
                "color:#ffffff;letter-spacing:.5px;margin-bottom:10px'>Stock Analyzer</div>",
                unsafe_allow_html=True,
            )

            # ── Three column layout — exact match of Scanner Stock Analyzer ─
            lk_colA, lk_colB, lk_colC = st.columns(3)

            with lk_colA:
                az_section("Price Range Analysis")
                if hi52 and lo52 and px:
                    pct_pos = max(0.0, min(1.0, (px - lo52) / (hi52 - lo52))) if hi52 != lo52 else 0.5
                    bar_pct = int(pct_pos * 100)
                    bar_col = "#4dd880" if pct_pos > 0.7 else ("#d0b040" if pct_pos > 0.35 else "#ff4444")
                    st.markdown(
                        f"<div style='background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;padding:14px 16px;margin-bottom:12px'>"
                        f"<div style='display:flex;justify-content:space-between;font-family:Space Mono,monospace;font-size:11px;color:#7a9ab8;margin-bottom:6px'><span>52W Low ${lo52:.2f}</span><span>52W High ${hi52:.2f}</span></div>"
                        f"<div style='background:#122540;border-radius:4px;height:8px;position:relative;overflow:visible'>"
                        f"<div style='background:{bar_col};width:{bar_pct}%;height:100%;border-radius:4px'></div>"
                        f"<div style='position:absolute;top:-18px;left:calc({bar_pct}% - 1px);font-family:Space Mono,monospace;font-size:10px;color:{bar_col}'>${px:.2f}</div></div>"
                        f"<div style='margin-top:10px;font-family:Space Mono,monospace;font-size:11px;color:#b0c8e8'>Position: <strong style='color:{bar_col}'>{rng52:.1f}% of 52W range</strong></div>"
                        f"</div>", unsafe_allow_html=True,
                    )
                if bb_upper and bb_lower and bb_mid and px:
                    bw   = bb_upper - bb_lower
                    bpos = max(0.0, min(1.0, (px - bb_lower) / bw)) if bw > 0 else 0.5
                    bpct = int(bpos * 100)
                    bcol = "#e05555" if bpos > 0.85 else ("#4dd880" if bpos < 0.15 else "#b0c8e8")
                    bbl  = "Near upper band (overbought)" if bpos > 0.85 else ("Near lower band (oversold)" if bpos < 0.15 else "Inside bands (neutral)")
                    st.markdown(
                        f"<div style='background:#0d1e33;border:1px solid #1e3a5f;border-radius:8px;padding:14px 16px;margin-bottom:12px'>"
                        f"<div style='display:flex;justify-content:space-between;font-family:Space Mono,monospace;font-size:11px;color:#7a9ab8;margin-bottom:6px'><span>BB Lower ${bb_lower:.2f}</span><span>BB Upper ${bb_upper:.2f}</span></div>"
                        f"<div style='background:#122540;border-radius:4px;height:8px;position:relative;overflow:visible'>"
                        f"<div style='position:absolute;left:50%;width:1px;height:100%;background:#1e3a5f'></div>"
                        f"<div style='background:{bcol};width:{bpct}%;height:100%;border-radius:4px'></div>"
                        f"<div style='position:absolute;top:-18px;left:calc({bpct}% - 1px);font-family:Space Mono,monospace;font-size:10px;color:{bcol}'>${px:.2f}</div></div>"
                        f"<div style='margin-top:10px;font-family:Space Mono,monospace;font-size:11px;color:#b0c8e8'>Bollinger: <strong style='color:{bcol}'>{bbl}</strong></div>"
                        f"<div style='margin-top:4px;font-family:Space Mono,monospace;font-size:11px;color:#7a9ab8'>Mid (20MA): ${bb_mid:.2f} · Width: ${bw:.2f}</div>"
                        f"</div>", unsafe_allow_html=True,
                    )
                az_section("Price Performance")
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse'><tbody>"
                    f"{''.join([mrow('1 Day','Daily change.',pct_color(pct1d)),mrow('5 Day','Five trading days.',pct_color(pct5d)),mrow('1 Month','~22 trading days.',pct_color(pct1m)),mrow('3 Month','~66 trading days.',pct_color(pct3m))])}"
                    f"</tbody></table>", unsafe_allow_html=True,
                )

            with lk_colB:
                az_section("Technical Signals")
                tech_rows = []
                if rsi_v is not None:
                    ri,rc = (("Oversold","#4dd880") if rsi_v<30 else ("Weak","#ff4444") if rsi_v<45 else ("Neutral","#b0c8e8") if rsi_v<55 else ("Strong","#4dd880") if rsi_v<70 else ("Overbought","#ff4444"))
                    tech_rows.append(mrow("RSI (14d)","RSI 0-100. 45-70 = sweet spot.",f"<span style='color:{rc};font-family:Space Mono,monospace'>{rsi_v:.1f}</span> <span style='font-size:11px;color:#7a9ab8'>{ri}</span>"))
                if macd_v is not None and macd_s is not None:
                    mc2 = "#4dd880" if macd_v > macd_s else "#ff4444"
                    tech_rows.append(mrow("MACD","Above signal = buyers in control.",f"<span style='color:{mc2};font-family:Space Mono,monospace'>{macd_v:.4f}</span> <span style='font-size:11px;color:#7a9ab8'>{'Bullish' if macd_v>macd_s else 'Bearish'}</span>"))
                if ma50_v and px:
                    pvs = (px - ma50_v) / ma50_v * 100
                    m5c2 = "#4dd880" if 0<pvs<5 else ("#d0b040" if pvs>=5 else ("#d0b040" if pvs>-5 else "#ff4444"))
                    tech_rows.append(mrow("50-Day MA","Short-term trend anchor.",f"${ma50_v:.2f} <span style='color:{m5c2};font-size:11px'>({pvs:+.1f}%)</span>"))
                if ma200_v:
                    m2c2 = "#4dd880" if (ma50_v and ma50_v>ma200_v) else "#ff4444"
                    tech_rows.append(mrow("200-Day MA","Long-term trend.",f"${ma200_v:.2f} <span style='font-size:11px;color:{m2c2}'>{'Golden Cross ↑' if (ma50_v and ma50_v>ma200_v) else 'Death Cross ↓'}</span>"))
                if vol_avg and vol_td:
                    vr = vol_td/vol_avg
                    vc = "#4dd880" if vr>1.5 else ("#b0c8e8" if vr>0.5 else "#d0b040")
                    tech_rows.append(mrow("Volume","Today vs 20-day avg.",f"<span style='color:{vc};font-family:Space Mono,monospace'>{vr:.2f}x avg</span> <span style='font-size:11px;color:#7a9ab8'>{'High' if vr>1.5 else 'Normal' if vr>0.5 else 'Low'}</span>"))
                if atr and px:
                    tech_rows.append(mrow("ATR (14d)","Avg daily range. Use for stop sizing.",f"${atr:.2f} <span style='color:#7a9ab8;font-size:11px'>({atr/px*100:.1f}% of price)</span>"))
                if ign_sc is not None:
                    ic = "#f5a623" if ign_sc>=70 else ("#d0b040" if ign_sc>=50 else "#4a6a8a")
                    tech_rows.append(mrow("Ignition Score","Live momentum score (0-100).",f"<span style='color:{ic};font-family:Space Mono,monospace;font-size:15px;font-weight:500'>{ign_sc:.0f}</span>"))
                if tech_rows:
                    st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(tech_rows)}</tbody></table>", unsafe_allow_html=True)

                az_section("Short Interest & Growth")
                si_rows = []
                if spf:    si_rows.append(mrow("Short % Float","% of float sold short. 15%+ = squeeze setup.",az_tag(spf*100,20,10,"{:.1f}","%")))
                if sratio: si_rows.append(mrow("Days to Cover","Shares short / avg volume.",az_tag(sratio,5,3,"{:.1f}","d")))
                if rg:     si_rows.append(mrow("Revenue Growth","YoY revenue change.",az_tag(rg*100,10,3,"{:.1f}","%")))
                if eg:     si_rows.append(mrow("Earnings Growth","YoY EPS change.",az_tag(eg*100,10,3,"{:.1f}","%")))
                if ins_buys and ins_net > 0:
                    si_rows.append(mrow("Insider Buying","Net insider buys last 90d (SEC Form 4).",f"<span style='font-family:Space Mono,monospace;color:#4dd880'>{ins_buys} buys · ${ins_net/1e3:.0f}K net</span>"))
                elif ins_net < 0:
                    si_rows.append(mrow("Insider Activity","Net insider selling last 90d.",f"<span style='font-family:Space Mono,monospace;color:#ff4444'>selling</span>"))
                if news_cnt:
                    si_rows.append(mrow("News 48h","Headlines in last 48 hours — catalyst coverage.",f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{news_cnt} headlines</span>"))
                if fuel_sc is not None:
                    fc = "#4dd880" if fuel_sc>=70 else ("#d0b040" if fuel_sc>=50 else "#4a6a8a")
                    si_rows.append(mrow("Fuel Score","Primed-to-move score (0-100).",f"<span style='color:{fc};font-family:Space Mono,monospace;font-size:15px;font-weight:500'>{fuel_sc:.0f}</span>"))
                if si_rows:
                    st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(si_rows)}</tbody></table>", unsafe_allow_html=True)

                az_section("Live Signals")
                if reasons:
                    st.markdown("".join(
                        f"<span style='display:inline-block;background:#0d1e33;color:#f5a623;"
                        f"border:1px solid #1e3a5f;border-radius:4px;font-family:Space Mono,monospace;"
                        f"font-size:11px;padding:2px 8px;margin:2px 3px 2px 0'>{r}</span>"
                        for r in reasons
                    ), unsafe_allow_html=True)
                else:
                    st.caption("No active live signals.")

                az_section("Earnings Breakdown (EPS Trend)")
                render_eps_trend(lk_eps_history, lk_eps_forward, why='; '.join(lk_info.get('_data_issues', [])))

                az_section("Dividend")
                render_dividend_info(lk_info)

            with lk_colC:
                az_section("Valuation")
                val_rows = []
                if mcap:
                    mc_str = f"${mcap/1e9:.2f}B" if mcap>=1e9 else f"${mcap/1e6:.1f}M"
                    val_rows.append(mrow("Market Cap","Total market value.",mc_str))
                if pe is not None:
                    pe_col = "#b0c8e8" if pe < 40 else ("#d0b040" if pe < 80 else "#ff4444")
                    val_rows.append(mrow("P/E Ratio","Price / trailing earnings. <25 = value, 25-40 = fair, 40+ = growth premium.",f"<span style='font-family:Space Mono,monospace;color:{pe_col}'>{pe:.1f}x</span>"))
                if fwpe is not None: val_rows.append(mrow("Forward P/E","P/E on next 12m estimates. Lower than trailing = earnings growth expected.",f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{fwpe:.1f}x</span>"))
                if pb is not None:   val_rows.append(mrow("P/B Ratio","Price / book value. <1.0 = trading below assets. >3.0 = growth premium.",f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{pb:.2f}x</span>"))
                if ps is not None:   val_rows.append(mrow("P/S Ratio","Price / revenue. <2x = reasonable, >10x = high growth premium.",f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{ps:.2f}x</span>"))
                if beta is not None: val_rows.append(mrow("Beta","Volatility vs S&P 500. 1.5 = 50% more volatile.",f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{beta:.2f}</span>"))
                if fl:
                    fl_str = f"{fl/1e9:.2f}B" if fl>=1e9 else f"{fl/1e6:.0f}M"
                    val_rows.append(mrow("Float","Tradeable shares. <20M = explosive on volume.",f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{fl_str}</span>"))
                if inst_pct is not None:
                    val_rows.append(mrow("Institutional","% held by institutions.",f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{inst_pct:.1f}%</span>"))
                if val_rows:
                    st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(val_rows)}</tbody></table>", unsafe_allow_html=True)
                else:
                    _why = "; ".join(lk_info.get("_data_issues", [])) or "yfinance returned no valuation fields — likely rate-limited; retry"
                    st.caption(f"Valuation unavailable — {_why}")

                az_section("Financial Health")
                hlth_rows = []
                if pm  is not None: hlth_rows.append(mrow("Profit Margin","Net income / revenue. Expanding = pricing power.",az_tag(pm*100,15,5,"{:.1f}","%")))
                if om  is not None: hlth_rows.append(mrow("Operating Margin","EBIT / revenue. Core business efficiency.",az_tag(om*100,15,5,"{:.1f}","%")))
                if roe is not None: hlth_rows.append(mrow("ROE","Return on equity. 15%+ = strong.",az_tag(roe*100,15,8,"{:.1f}","%")))
                if roa is not None: hlth_rows.append(mrow("ROA","Return on assets. 5%+ = solid.",az_tag(roa*100,8,3,"{:.1f}","%")))
                if deq is not None: hlth_rows.append(mrow("D/E Ratio","Total debt / equity. Compare within sector.",f"<span style='font-family:Space Mono,monospace;color:#b0c8e8'>{deq:.1f}%</span>"))
                if cr  is not None: hlth_rows.append(mrow("Current Ratio","Current assets / liabilities. 1.5+ = healthy.",az_tag(cr,1.5,1.0,"{:.2f}")))
                if hlth_rows:
                    st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(hlth_rows)}</tbody></table>", unsafe_allow_html=True)

                if am:
                    az_section("Analyst Consensus")
                    rcol  = "#4dd880" if "buy" in recky.lower() else ("#ff4444" if "sell" in recky.lower() else "#d0b040")
                    rdisp = recky.replace("_"," ").title() if recky else "--"
                    an_rows = [
                        mrow("Recommendation","Wall Street consensus.",f"<span style='color:{rcol};font-weight:500'>{rdisp}</span> <span style='font-size:11px;color:#7a9ab8'>({nana} analysts)</span>"),
                        mrow("Mean Target","Average 12-month target.",f"${am:.2f}" + (f" <span style='font-size:11px;color:{'#4dd880' if aus and aus>0 else '#ff4444'}'>({aus:+.1f}%)</span>" if aus else "")),
                        mrow("Target Range","Low to high analyst target.",f"${al:.2f} – ${ahigh:.2f}" if (al and ahigh) else "--"),
                    ]
                    st.markdown(f"<table style='width:100%;border-collapse:collapse'><tbody>{''.join(an_rows)}</tbody></table>", unsafe_allow_html=True)

                # Catalysts from scan
                if lk_fuel.get("catalyst_tags") or lk_fuel.get("bimodal_event") or (lk_fuel.get("days_to_cover") and lk_fuel["days_to_cover"] >= 5):
                    az_section("Catalysts Detected")
                    full_cat = build_catalyst_tags_html(
                        lk_ticker, lk_fuel.get("catalyst_tags") or [],
                        lk_fuel.get("bimodal_event", False),
                        lk_fuel.get("days_to_cover"), lk_fuel.get("earnings_days"),
                    )
                    if suppressed:
                        for s in suppressed:
                            full_cat += f"<span style='font-family:Space Mono,monospace;font-size:10px;color:#4a6a8a;border:1px dashed #1e3a5f;border-radius:4px;padding:1px 6px;margin:2px 3px 2px 0;text-decoration:line-through;display:inline-block'>{s.upper()}</span>"
                    if lk_fuel.get("latest_headline"):
                        full_cat += f"<div style='margin-top:8px;font-size:11px;color:#7a9ab8'>{lk_fuel['latest_headline']}</div>"
                    st.markdown(full_cat, unsafe_allow_html=True)



if __name__ == "__main__":
    render_ignition_scanner_tab()
