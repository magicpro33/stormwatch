"""Hybrid Screener — the magicpro33/stock filter suite, on the nightly dump.

Filters, metric weights, and presets are the same ones the standalone Hybrid
Stock Screener uses. Screening runs against the dump Money Weather already
downloads (no live yfinance scan). Price / MA50 / range width are refreshed
from the dump panel so the range lookback slider is honest.
"""
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import cascade_engine as ce


METRICS = {
    "OE_Yield": {
        "label": "OE Yield",
        "weight": 3,
        "desc": "Owner Earnings Yield = (Net Income + Depreciation − CapEx) ÷ Market Cap. "
                "Higher is better — above 5% is considered good value.",
    },
    "ROIC": {
        "label": "ROIC",
        "weight": 2,
        "desc": "Return on Invested Capital. Above 15% suggests a durable moat.",
    },
    "ROIC_Trend": {
        "label": "ROIC Trend",
        "weight": 2,
        "desc": "Year-over-year change in ROIC. Positive = the moat is widening.",
    },
    "RevenueGrowth": {
        "label": "Revenue Growth",
        "weight": 1,
        "desc": "Year-over-year revenue growth. 5%+ is healthy for a mature company.",
    },
    "EarningsGrowth": {
        "label": "Earnings Growth",
        "weight": 1,
        "desc": "Year-over-year EPS growth. Ideally faster than revenue.",
    },
    "GrossMargin": {
        "label": "Gross Margin (Moat)",
        "weight": 4,
        "desc": "Gross profit ÷ revenue. 60%+ is pricing power; below 30% is a commodity fight.",
    },
    "Piotroski": {
        "label": "Piotroski Score",
        "weight": 1,
        "desc": "Accounting health (F-Score). 7–9 strong; 0–2 weak or distressed.",
    },
    "OBV": {
        "label": "OBV (On-Balance Volume)",
        "weight": 1,
        "desc": "1.0 if the 20-day OBV slope is rising (accumulation), else 0.0.",
    },
    "MFI": {
        "label": "MFI (Money Flow Index)",
        "weight": 1,
        "desc": "Volume-weighted RSI scaled 0–1. Below 0.3 outflow; above 0.6 buying pressure.",
    },
    "PCV": {
        "label": "PCV (Price-Confirmed Volume)",
        "weight": 1,
        "desc": "Share of recent volume on up-days. Scales from 0 (50/50) to 1.0 (all up-day).",
    },
    "RangePosScore": {
        "label": "Range Position Score",
        "weight": 2,
        "desc": "1 − RangePos. 1.0 = sitting on range support; 0.0 = already at the highs.",
    },
    "RSI": {
        "label": "RSI (Momentum Zone)",
        "weight": 2,
        "desc": "1.0 = RSI 55–70 (trend, not overbought). 0.0 = RSI >80 or <50.",
    },
    "MACD": {
        "label": "MACD Momentum",
        "weight": 2,
        "desc": "1.0 = histogram positive and growing. 0.0 = histogram negative.",
    },
    "GoldenCross": {
        "label": "Golden Cross",
        "weight": 3,
        "desc": "1.0 = 50MA > 200MA. 0.5 = within 2%. 0.0 = death cross.",
    },
    "MFISweetSpot": {
        "label": "MFI Sweet Spot",
        "weight": 2,
        "desc": "1.0 = MFI 55–75 (inflow without overbought). 0.0 = >90 or <50.",
    },
    "NoBearDiv": {
        "label": "No Bearish Divergence",
        "weight": 2,
        "desc": "1.0 = price and MFI both higher highs. 0.0 = price high, MFI did not confirm.",
    },
    "MA50Proximity": {
        "label": "MA50 Proximity",
        "weight": 1,
        "desc": "1.0 = 0–5% above MA50. 0.0 = >20% extended or below the average.",
    },
    "ShortSqueeze": {
        "label": "Short Squeeze Score",
        "weight": 3,
        "desc": "High short interest plus upward price. 1.0 = >20% short + DTC >5 + above MA50.",
    },
    "CleanSetupScore": {
        "label": "Clean Setup (Bull Patterns)",
        "weight": 5,
        "desc": "Trend + bull flag + higher-low + RSI 40–60 + volume. 0 if price <$5 or thin volume.",
    },
    "DividendScore": {
        "label": "Dividend Score",
        "weight": 4,
        "desc": "Yield + payout sustainability + payment frequency. 0 = no dividend.",
    },
}

ALL_SECTORS = [
    "All Sectors",
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Unknown",
]

_SECTOR_ALIASES = {
    "basic materials": "Basic Materials",
    "communication services": "Communication Services",
    "consumer cyclical": "Consumer Cyclical",
    "consumer defensive": "Consumer Defensive",
    "energy": "Energy",
    "financial services": "Financial Services",
    "healthcare": "Healthcare",
    "industrials": "Industrials",
    "real estate": "Real Estate",
    "technology": "Technology",
    "utilities": "Utilities",
    "materials": "Basic Materials",
    "consumer discretionary": "Consumer Cyclical",
    "consumer staples": "Consumer Defensive",
    "financials": "Financial Services",
    "finance": "Financial Services",
    "health care": "Healthcare",
    "information technology": "Technology",
    "it": "Technology",
}

EXCHANGES = {
    "All Stocks": "all",
    "S&P 500": "sp500",
    "NYSE": "nyse",
    "NASDAQ": "nasdaq",
}

REV_GROWTH_STEPS = [0, 15, 30, 45, 60, 75, 90]

_METRIC_GROUPS = [
    ("Fundamental",
     ["OE_Yield", "ROIC", "ROIC_Trend", "GrossMargin",
      "RevenueGrowth", "EarningsGrowth", "Piotroski"]),
    ("Volume & Buying Pressure", ["OBV", "MFI", "PCV"]),
    ("Technical & Momentum",
     ["RSI", "MACD", "GoldenCross", "MFISweetSpot", "NoBearDiv", "MA50Proximity"]),
    ("Pattern Setups", ["CleanSetupScore"]),
    ("Short Interest", ["ShortSqueeze"]),
    ("Dividend Income", ["DividendScore"]),
    ("Range & Breakout Setup", ["RangePosScore"]),
]


def _k(name: str) -> str:
    return f"hs_{name}"


def _normalise_sector(raw) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "Unknown"
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "n/a"):
        return "Unknown"
    return _SECTOR_ALIASES.get(s.lower(), s)


def _ensure_defaults() -> None:
    defaults = {
        "max_price": 250,
        "min_score": 2.0,
        "top_n": 30,
        "ma50": "below",
        "range": False,
        "range_days": 30,
        "range_pct": 10.0,
        "pe_filter": False,
        "pe_range": (0, 50),
        "rev_filter": False,
        "rev_min": 0,
        "exchange": "All Stocks",
        "sector": "All Sectors",
    }
    for name, val in defaults.items():
        defaults_key = _k(name)
        if defaults_key not in st.session_state:
            st.session_state[defaults_key] = val
    # Same first-load defaults as the standalone Hybrid Stock Screener:
    # fundamentals on, technical / pattern / squeeze / dividend / range off.
    _off = {
        "RangePosScore", "RSI", "MACD", "GoldenCross", "MFISweetSpot",
        "NoBearDiv", "MA50Proximity", "ShortSqueeze", "DividendScore",
        "CleanSetupScore", "GrossMargin",
    }
    for key, cfg in METRICS.items():
        tog, wt = _k(f"tog_{key}"), _k(f"wt_{key}")
        if tog not in st.session_state:
            st.session_state[tog] = key not in _off
        if wt not in st.session_state:
            st.session_state[wt] = float(cfg["weight"])


def _clear_metric_weights() -> None:
    for key in METRICS:
        st.session_state[_k(f"tog_{key}")] = False
        st.session_state[_k(f"wt_{key}")] = 0.0


def _apply_preset(name: str) -> None:
    """Same weight maps as the standalone Hybrid Stock Screener."""
    st.session_state[_k("max_price")] = 500
    st.session_state[_k("min_score")] = 0.0
    st.session_state[_k("ma50")] = "off"
    st.session_state[_k("range")] = False
    st.session_state[_k("range_days")] = 30
    st.session_state[_k("range_pct")] = 10.0
    st.session_state[_k("pe_filter")] = False
    st.session_state[_k("rev_filter")] = False
    _clear_metric_weights()

    if name == "clean":
        st.session_state[_k("tog_CleanSetupScore")] = True
        st.session_state[_k("wt_CleanSetupScore")] = 5.0
        st.session_state[_k("tog_MACD")] = True
        st.session_state[_k("wt_MACD")] = 4.0
        st.session_state[_k("tog_RSI")] = True
        st.session_state[_k("wt_RSI")] = 3.0
        st.session_state[_k("tog_MA50Proximity")] = True
        st.session_state[_k("wt_MA50Proximity")] = 2.0
    elif name == "felix":
        st.session_state[_k("max_price")] = 1000
        st.session_state[_k("pe_filter")] = True
        st.session_state[_k("pe_range")] = (0, 50)
        st.session_state[_k("tog_ROIC")] = True
        st.session_state[_k("wt_ROIC")] = 5.0
        st.session_state[_k("tog_GrossMargin")] = True
        st.session_state[_k("wt_GrossMargin")] = 5.0
        st.session_state[_k("tog_OE_Yield")] = True
        st.session_state[_k("wt_OE_Yield")] = 4.0
        st.session_state[_k("tog_Piotroski")] = True
        st.session_state[_k("wt_Piotroski")] = 4.0
        st.session_state[_k("tog_ROIC_Trend")] = True
        st.session_state[_k("wt_ROIC_Trend")] = 2.0
        st.session_state[_k("tog_RevenueGrowth")] = True
        st.session_state[_k("wt_RevenueGrowth")] = 1.0
        st.session_state[_k("tog_EarningsGrowth")] = True
        st.session_state[_k("wt_EarningsGrowth")] = 1.0
    elif name == "squeeze":
        st.session_state[_k("ma50")] = "above"
        st.session_state[_k("tog_ShortSqueeze")] = True
        st.session_state[_k("wt_ShortSqueeze")] = 5.0
        st.session_state[_k("tog_Piotroski")] = True
        st.session_state[_k("wt_Piotroski")] = 5.0
        st.session_state[_k("tog_RSI")] = True
        st.session_state[_k("wt_RSI")] = 2.0
        st.session_state[_k("tog_NoBearDiv")] = True
        st.session_state[_k("wt_NoBearDiv")] = 2.0
        st.session_state[_k("tog_PCV")] = True
        st.session_state[_k("wt_PCV")] = 2.0
        st.session_state[_k("tog_EarningsGrowth")] = True
        st.session_state[_k("wt_EarningsGrowth")] = 1.0
        st.session_state[_k("tog_MACD")] = True
        st.session_state[_k("wt_MACD")] = 1.0
    elif name == "lowpos":
        st.session_state[_k("range")] = True
        st.session_state[_k("range_days")] = 30
        st.session_state[_k("range_pct")] = 20.0
        st.session_state[_k("tog_RangePosScore")] = True
        st.session_state[_k("wt_RangePosScore")] = 5.0
        st.session_state[_k("tog_OE_Yield")] = True
        st.session_state[_k("wt_OE_Yield")] = 5.0
        st.session_state[_k("tog_RSI")] = True
        st.session_state[_k("wt_RSI")] = 5.0
        st.session_state[_k("tog_ROIC")] = True
        st.session_state[_k("wt_ROIC")] = 3.0
        st.session_state[_k("tog_NoBearDiv")] = True
        st.session_state[_k("wt_NoBearDiv")] = 2.0
        st.session_state[_k("tog_OBV")] = True
        st.session_state[_k("wt_OBV")] = 1.0
        st.session_state[_k("tog_EarningsGrowth")] = True
        st.session_state[_k("wt_EarningsGrowth")] = 1.0
        st.session_state[_k("tog_GoldenCross")] = True
        st.session_state[_k("wt_GoldenCross")] = 1.0
    elif name == "volume":
        st.session_state[_k("ma50")] = "above"
        st.session_state[_k("tog_OBV")] = True
        st.session_state[_k("wt_OBV")] = 5.0
        st.session_state[_k("tog_PCV")] = True
        st.session_state[_k("wt_PCV")] = 5.0
        st.session_state[_k("tog_MACD")] = True
        st.session_state[_k("wt_MACD")] = 5.0
        st.session_state[_k("tog_Piotroski")] = True
        st.session_state[_k("wt_Piotroski")] = 5.0
        st.session_state[_k("tog_NoBearDiv")] = True
        st.session_state[_k("wt_NoBearDiv")] = 4.0
        st.session_state[_k("tog_RSI")] = True
        st.session_state[_k("wt_RSI")] = 3.0
        st.session_state[_k("tog_MA50Proximity")] = True
        st.session_state[_k("wt_MA50Proximity")] = 3.0
        st.session_state[_k("tog_MFI")] = True
        st.session_state[_k("wt_MFI")] = 2.0
        st.session_state[_k("tog_MFISweetSpot")] = True
        st.session_state[_k("wt_MFISweetSpot")] = 1.0
        st.session_state[_k("tog_GoldenCross")] = True
        st.session_state[_k("wt_GoldenCross")] = 1.0
        st.session_state[_k("tog_EarningsGrowth")] = True
        st.session_state[_k("wt_EarningsGrowth")] = 1.0
        st.session_state[_k("tog_OE_Yield")] = True
        st.session_state[_k("wt_OE_Yield")] = 1.0
    elif name == "breakout":
        st.session_state[_k("range")] = True
        st.session_state[_k("range_days")] = 30
        st.session_state[_k("range_pct")] = 15.0
        st.session_state[_k("tog_RangePosScore")] = True
        st.session_state[_k("wt_RangePosScore")] = 5.0
        st.session_state[_k("tog_OBV")] = True
        st.session_state[_k("wt_OBV")] = 5.0
        st.session_state[_k("tog_MACD")] = True
        st.session_state[_k("wt_MACD")] = 5.0
        st.session_state[_k("tog_NoBearDiv")] = True
        st.session_state[_k("wt_NoBearDiv")] = 5.0
        st.session_state[_k("tog_OE_Yield")] = True
        st.session_state[_k("wt_OE_Yield")] = 4.0
        st.session_state[_k("tog_ROIC")] = True
        st.session_state[_k("wt_ROIC")] = 4.0
        st.session_state[_k("tog_PCV")] = True
        st.session_state[_k("wt_PCV")] = 3.0
        st.session_state[_k("tog_MA50Proximity")] = True
        st.session_state[_k("wt_MA50Proximity")] = 3.0
        st.session_state[_k("tog_GoldenCross")] = True
        st.session_state[_k("wt_GoldenCross")] = 3.0
        st.session_state[_k("tog_MFISweetSpot")] = True
        st.session_state[_k("wt_MFISweetSpot")] = 3.0
        st.session_state[_k("tog_RSI")] = True
        st.session_state[_k("wt_RSI")] = 2.0
        st.session_state[_k("tog_EarningsGrowth")] = True
        st.session_state[_k("wt_EarningsGrowth")] = 1.0
    elif name == "insider":
        st.session_state[_k("ma50")] = "above"
        st.session_state[_k("range_days")] = 60
        st.session_state[_k("range_pct")] = 15.0
        st.session_state[_k("tog_OBV")] = True
        st.session_state[_k("wt_OBV")] = 5.0
        st.session_state[_k("tog_Piotroski")] = True
        st.session_state[_k("wt_Piotroski")] = 5.0
        st.session_state[_k("tog_MFISweetSpot")] = True
        st.session_state[_k("wt_MFISweetSpot")] = 4.0
        st.session_state[_k("tog_RangePosScore")] = True
        st.session_state[_k("wt_RangePosScore")] = 3.0
        st.session_state[_k("tog_MACD")] = True
        st.session_state[_k("wt_MACD")] = 3.0
        st.session_state[_k("tog_PCV")] = True
        st.session_state[_k("wt_PCV")] = 2.0
        st.session_state[_k("tog_EarningsGrowth")] = True
        st.session_state[_k("wt_EarningsGrowth")] = 2.0
        st.session_state[_k("tog_NoBearDiv")] = True
        st.session_state[_k("wt_NoBearDiv")] = 2.0
        st.session_state[_k("tog_GoldenCross")] = True
        st.session_state[_k("wt_GoldenCross")] = 2.0
        st.session_state[_k("tog_RSI")] = True
        st.session_state[_k("wt_RSI")] = 1.0
    st.rerun()


def _load_universe() -> tuple[pd.DataFrame, str]:
    """Nightly dump rows minus bulky _hist / _analyzer packs."""
    try:
        ce.load_dump_panel()
    except Exception as exc:
        return pd.DataFrame(), f"Nightly dump unavailable: {exc}"
    raw = ce._dump_records_cache()
    if not raw:
        return pd.DataFrame(), "Nightly dump is empty — try Refresh on Cascade Map."
    drop = {"_hist", "_analyzer"}
    rows = [{k: v for k, v in rec.items() if k not in drop}
            for rec in raw.values() if rec.get("Ticker")]
    df = pd.DataFrame(rows)
    df.replace(["N/A", "None", "-", ""], pd.NA, inplace=True)
    if "Sector" in df.columns:
        df["Sector"] = df["Sector"].fillna("Unknown").apply(_normalise_sector)
    else:
        df["Sector"] = "Unknown"
    zero_cols = [
        "OBV", "MFI", "PCV", "RSI", "MACD", "GoldenCross",
        "MFISweetSpot", "NoBearDiv", "MA50Proximity",
        "OE_Yield", "ROIC", "ROIC_Trend", "Piotroski",
        "RevenueGrowth", "EarningsGrowth", "RangePct",
        "RangePos", "RangeHigh", "RangeLow", "MA50",
        "ShortSqueeze", "ShortPctFloat", "ShortPctFloatRaw",
        "DaysToCover", "ShortChange",
        "DividendYieldPct", "DividendRate", "DividendPayoutRatio", "DividendScore",
        "CleanSetupScore", "GrossMargin",
    ]
    for col in zero_cols:
        if col not in df.columns:
            df[col] = 0.0
    for col in ["Price", "MarketCap", "P/E", "OwnerEarnings"]:
        if col not in df.columns:
            df[col] = np.nan
    numeric = zero_cols + ["Price", "MarketCap", "P/E", "OwnerEarnings"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, ""


def _overlay_panel(df: pd.DataFrame, range_days: int) -> pd.DataFrame:
    """Refresh Price / MA50 / range from the dump panel (vectorized)."""
    try:
        panel, tickers, _sectors, _mdv, _dates = ce.load_dump_panel()
    except Exception:
        df = df.copy()
        df["RangePosScore"] = (1 - df["RangePos"].fillna(0.5)).round(4)
        return df
    tix = {str(t).upper(): i for i, t in enumerate(tickers)}
    c, h, l = panel["c"], panel["h"], panel["l"]
    look = max(5, min(int(range_days), c.shape[0]))
    last = c[-1]
    hi = np.nanmax(h[-look:], axis=0)
    lo = np.nanmin(l[-look:], axis=0)
    ma = np.nanmean(c[-50:], axis=0) if c.shape[0] >= 50 else np.full(c.shape[1], np.nan)
    width = hi - lo
    pct = np.full_like(last, np.nan, dtype=np.float64)
    pos = np.full_like(last, np.nan, dtype=np.float64)
    ok_px = np.isfinite(last) & (last > 0)
    ok_w = np.isfinite(width) & (width > 0)
    np.divide(width, last, out=pct, where=ok_px)
    pct = np.where(ok_px, pct * 100.0, np.nan)
    np.divide(last - lo, width, out=pos, where=ok_w)

    n = len(df)
    price = np.full(n, np.nan)
    ma50 = np.full(n, np.nan)
    rhi = np.full(n, np.nan)
    rlo = np.full(n, np.nan)
    rpct = np.full(n, np.nan)
    rpos = np.full(n, np.nan)
    for i, tk in enumerate(df["Ticker"].astype(str).str.upper()):
        j = tix.get(tk)
        if j is None:
            continue
        price[i] = last[j]
        ma50[i] = ma[j]
        rhi[i] = hi[j]
        rlo[i] = lo[j]
        rpct[i] = pct[j]
        rpos[i] = pos[j]
    out = df.copy()
    m = np.isfinite(price)
    out.loc[m, "Price"] = price[m]
    m = np.isfinite(ma50)
    out.loc[m, "MA50"] = np.round(ma50[m], 2)
    m = np.isfinite(rhi)
    out.loc[m, "RangeHigh"] = rhi[m]
    out.loc[m, "RangeLow"] = rlo[m]
    out.loc[m, "RangePct"] = rpct[m]
    out.loc[m, "RangePos"] = rpos[m]
    out["RangePosScore"] = (1 - pd.to_numeric(out["RangePos"], errors="coerce").fillna(0.5)).round(4)
    return out


def _in_exchange(row_exchanges, key: str) -> bool:
    if key == "all":
        return True
    if not isinstance(row_exchanges, (list, tuple, set)):
        return False
    return key in row_exchanges


def _mfi_signal(v) -> str:
    if pd.isna(v):
        return "—"
    mfi = float(v) * 100
    if mfi >= 80:
        return "🔥 Overbought"
    if mfi >= 60:
        return "📈 Buying"
    if mfi >= 40:
        return "➡️ Neutral"
    if mfi >= 20:
        return "📉 Selling"
    return "🧊 Oversold"


def _color_score(val):
    try:
        v = float(val)
        if v >= 8:
            return "background-color: #1a7a1a; color: white"
        if v >= 5:
            return "background-color: #4caf50; color: white"
        if v >= 3:
            return "background-color: #ff9800; color: white"
        return "background-color: #f44336; color: white"
    except Exception:
        return ""


def _format_display(screened: pd.DataFrame, enabled: dict) -> pd.DataFrame:
    display = screened.copy()
    if "MarketCap" in display.columns:
        display["MarketCap"] = display["MarketCap"].apply(
            lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
    if "OwnerEarnings" in display.columns:
        display["OwnerEarnings"] = display["OwnerEarnings"].apply(
            lambda x: f"${x:,.0f}" if pd.notnull(x) else "")
    if "RevenueGrowth" in display.columns:
        display["RevenueGrowth"] = display["RevenueGrowth"].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    if "EarningsGrowth" in display.columns:
        display["EarningsGrowth"] = display["EarningsGrowth"].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    if "RangePct" in display.columns:
        display["RangePct"] = display["RangePct"].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) else "")
    if "RangePos" in display.columns:
        bars = "▁▂▃▄▅▆▇█"

        def _bar(x):
            if pd.isna(x):
                return "—"
            return f"{bars[min(int(float(x) * 8), 7)]} {float(x):.0%}"

        display["RangePos"] = display["RangePos"].apply(_bar)
    if "ShortPctFloatRaw" in display.columns:
        display["Short % Float"] = display["ShortPctFloatRaw"].apply(
            lambda x: f"{x:.1f}%" if pd.notnull(x) and x else "—")
    if "DaysToCover" in display.columns:
        display["Days to Cover"] = display["DaysToCover"].apply(
            lambda x: f"{x:.1f}" if pd.notnull(x) and x else "—")
    if "DividendYieldPct" in display.columns:
        display["Div Yield"] = display["DividendYieldPct"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) and x else "—")
    if "DividendRate" in display.columns:
        display["Div Rate"] = display["DividendRate"].apply(
            lambda x: f"${x:.2f}" if pd.notnull(x) and x else "—")
    if "MFI" in display.columns:
        display["MFI_Signal"] = display["MFI"].apply(_mfi_signal)

    skip = {"Ticker", "Sector", "MarketCap", "OwnerEarnings",
            "RevenueGrowth", "EarningsGrowth", "RangePct", "RangePos",
            "MFI_Signal", "_exchanges"}
    for col in display.columns:
        if col not in skip and pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

    hidden = [k for k in METRICS if not enabled.get(k) and k in display.columns]
    always = {
        "ROIC", "OBV", "PCV", "RSI", "MACD", "GoldenCross", "MFI",
        "MFISweetSpot", "NoBearDiv", "ShortPctFloat", "ShortPctFloatRaw",
        "DaysToCover", "ShortChange", "ShortSqueeze",
        "DividendYieldPct", "DividendRate", "DividendPayoutRatio",
        "DividendFrequency", "DividendScore", "CleanSetupScore",
        "GrossMargin", "_exchanges", "_exchange",
    }
    hidden += [c for c in always if c in display.columns and c not in hidden]
    if not enabled.get("MFI") and "MFI_Signal" in display.columns:
        hidden.append("MFI_Signal")
    if not enabled.get("ShortSqueeze"):
        hidden += [c for c in ("Short % Float", "Days to Cover") if c in display.columns]
    if not enabled.get("DividendScore"):
        hidden += [c for c in ("Div Yield", "Div Rate") if c in display.columns]
    return display.drop(columns=hidden, errors="ignore")


def _to_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _run_screen(settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df, err = _load_universe()
    if df.empty:
        raise RuntimeError(err or "No dump rows to screen.")

    exch = settings["exchange_key"]
    if exch != "all" and "_exchanges" in df.columns:
        df = df[df["_exchanges"].apply(lambda x: _in_exchange(x, exch))].copy()

    sector = settings["sector"]
    if sector != "All Sectors":
        df = df[df["Sector"].apply(_normalise_sector).str.lower() == sector.strip().lower()]

    df = _overlay_panel(df, settings["range_days"])

    enabled, weights = settings["enabled"], settings["weights"]
    score = pd.Series(0.0, index=df.index)
    active = [k for k, on in enabled.items() if on]
    for key in active:
        if key in df.columns:
            score += pd.to_numeric(df[key], errors="coerce").fillna(0) * weights.get(key, 0)
    df["Score"] = score.round(2)

    under_price = df["Price"].isna() | (df["Price"] <= settings["max_price"])
    above_score = df["Score"] >= settings["min_score"] if active else pd.Series(True, index=df.index)
    ma50 = pd.to_numeric(df.get("MA50", pd.Series(np.nan, index=df.index)), errors="coerce")
    mode = settings["ma50"]
    if mode == "below":
        ma_ok = df["Price"].isna() | ma50.isna() | (df["Price"] <= ma50)
    elif mode == "above":
        ma_ok = df["Price"].isna() | ma50.isna() | (df["Price"] >= ma50)
    else:
        ma_ok = pd.Series(True, index=df.index)
    if settings["use_range"]:
        rng = pd.to_numeric(df.get("RangePct", pd.Series(np.nan, index=df.index)), errors="coerce")
        in_range = rng.isna() | (rng <= settings["range_pct"])
    else:
        in_range = pd.Series(True, index=df.index)
    if settings["use_pe"]:
        pe = pd.to_numeric(df.get("P/E", pd.Series(np.nan, index=df.index)), errors="coerce")
        lo, hi = settings["pe_range"]
        in_pe = pe.notna() & (pe >= lo) & (pe <= hi)
    else:
        in_pe = pd.Series(True, index=df.index)
    if settings["use_rev"]:
        rev = pd.to_numeric(df.get("RevenueGrowth", pd.Series(np.nan, index=df.index)), errors="coerce")
        in_rev = rev.notna() & (rev >= settings["rev_min"] / 100.0)
    else:
        in_rev = pd.Series(True, index=df.index)

    sort_col = "Score" if active else "RangePct"
    sort_asc = not bool(active)
    screened = (
        df[under_price & ma_ok & above_score & in_range & in_pe & in_rev]
        .sort_values(sort_col, ascending=sort_asc, na_position="last")
        .head(int(settings["top_n"]))
        .reset_index(drop=True)
    )
    diag = {
        "scanned": int(len(df)),
        "passed": int(len(screened)),
        "active": active,
        "price": int(under_price.sum()),
        "ma50": int(ma_ok.sum()),
        "score": int(above_score.sum()),
        "range": int(in_range.sum()),
        "pe": int(in_pe.sum()),
        "rev": int(in_rev.sum()),
        "err": "",
    }
    return df, screened, diag


def render_hybrid_screener() -> None:
    """Filters + results. Call from the Hybrid Screener tab."""
    _ensure_defaults()

    st.caption(
        "The Hybrid Stock Screener from the nightly dump — same presets, "
        "metric weights, and gates. No live Yahoo scan: scores are the ones "
        "computed overnight, with Price / MA50 / range refreshed from the dump panel."
    )

    presets = st.columns(7)
    _preset_btns = [
        ("clean", "📐 Clean Setup", "Trend + flag + RSI band. Strongest preset in the suite."),
        ("felix", "🎩 Felix", "ROIC, moat, cash, Piotroski, P/E ≤ 50."),
        ("squeeze", "🎯 Short Squeeze", "High short interest + quality + above MA50."),
        ("lowpos", "📉 Low Price Position", "Range low + OE Yield + ROIC."),
        ("volume", "⚡ Magic Volume", "OBV / PCV surge with MACD confirmation."),
        ("breakout", "🚀 Breakout Setup", "Tight coil at range low, ready to break."),
        ("insider", "🕵️ Insider Buying", "Accumulation footprints in quiet ranges."),
    ]
    for col, (key, label, help_) in zip(presets, _preset_btns):
        if col.button(label, width="stretch", key=_k(f"preset_{key}"), help=help_):
            _apply_preset(key)

    with st.expander("⚙️ Filters", expanded=True):
        r1 = st.columns([1.2, 1.2, 1, 1, 1])
        exchange = r1[0].selectbox(
            "Universe", list(EXCHANGES), key=_k("exchange"),
            help="All Stocks = the full nightly dump. S&P / NYSE / NASDAQ use "
                 "the exchange tags written by the scan.")
        sector = r1[1].selectbox("Sector", ALL_SECTORS, key=_k("sector"))
        max_price = r1[2].slider("Max price ($)", 10, 1000, step=10, key=_k("max_price"))
        min_score = r1[3].slider("Min hybrid score", 0.0, 20.0, step=0.5, key=_k("min_score"))
        top_n = r1[4].slider("Max results", 5, 100, step=5, key=_k("top_n"))

        r2 = st.columns([1.4, 1, 1, 1.2])
        ma50_mode = r2[0].radio(
            "50-day MA", ["off", "below", "above"], horizontal=True, key=_k("ma50"),
            format_func=lambda x: {
                "off": "No MA50 filter",
                "below": "Below MA50 (pullbacks)",
                "above": "Above MA50 (uptrends)",
            }[x],
        )
        use_range = r2[1].toggle("Tight range", key=_k("range"),
                                 help="Keep names whose high–low spread is under Max range %.")
        range_days = r2[2].slider("Range lookback (days)", 5, 180, step=5, key=_k("range_days"))
        range_pct = r2[3].slider("Max range width (%)", 1.0, 30.0, step=0.5,
                                 key=_k("range_pct"), disabled=not use_range)

        r3 = st.columns([1, 1.4, 1, 1.2])
        use_pe = r3[0].toggle("P/E filter", key=_k("pe_filter"))
        pe_range = r3[1].slider("P/E range", 0, 200, step=1, key=_k("pe_range"),
                                disabled=not use_pe)
        use_rev = r3[2].toggle("Revenue growth", key=_k("rev_filter"))
        rev_min = r3[3].select_slider(
            "Min TTM revenue growth", options=REV_GROWTH_STEPS,
            format_func=lambda x: f"{x}%+", key=_k("rev_min"), disabled=not use_rev)

    enabled, weights = {}, {}
    with st.expander("📊 Metric weights", expanded=False):
        st.caption("Toggle a factor and set its weight in the hybrid score (0–5).")
        for title, keys in _METRIC_GROUPS:
            st.markdown(f"**{title}**")
            for key in keys:
                cfg = METRICS[key]
                a, b = st.columns([1, 2])
                enabled[key] = a.toggle(cfg["label"], key=_k(f"tog_{key}"))
                weights[key] = b.slider(
                    "Weight", 0.0, 5.0, step=0.5, key=_k(f"wt_{key}"),
                    disabled=not enabled[key], label_visibility="collapsed")
                st.caption(cfg["desc"])
    # Expander skipped on a later rerun still needs current toggle values
    for key in METRICS:
        enabled[key] = bool(st.session_state.get(_k(f"tog_{key}"), False))
        weights[key] = float(st.session_state.get(_k(f"wt_{key}"), 0.0) or 0.0)

    run = st.button("🚀 Run Screener", type="primary", width="stretch", key=_k("run"))

    if run:
        settings = {
            "exchange_key": EXCHANGES[exchange],
            "sector": sector,
            "max_price": max_price,
            "min_score": min_score,
            "top_n": top_n,
            "ma50": ma50_mode,
            "use_range": use_range,
            "range_days": range_days,
            "range_pct": range_pct,
            "use_pe": use_pe,
            "pe_range": pe_range,
            "use_rev": use_rev,
            "rev_min": rev_min,
            "enabled": enabled,
            "weights": weights,
        }
        with st.spinner("Screening the nightly dump…"):
            try:
                _all, screened, diag = _run_screen(settings)
            except Exception as exc:
                st.error(str(exc))
                return
        display = _format_display(screened, enabled) if not screened.empty else screened
        st.session_state["_hs_display"] = display
        st.session_state["_hs_raw"] = screened
        st.session_state["_hs_diag"] = diag
        st.session_state["_hs_sector"] = sector
        st.session_state.pop(_k("inline"), None)

    display = st.session_state.get("_hs_display")
    diag = st.session_state.get("_hs_diag") or {}
    if display is None:
        st.info("Pick a preset or set your own weights, then click **Run Screener**.")
        return

    active = diag.get("active") or []
    if active:
        st.success("**Active metrics:** " + " · ".join(METRICS[k]["label"] for k in active if k in METRICS))
    else:
        st.info("No metrics active — sorted by tightest stored range.")

    m1, m2, m3, m4 = st.columns(4)
    raw = st.session_state.get("_hs_raw")
    m1.metric("Tickers scanned", f"{diag.get('scanned', 0):,}")
    m2.metric("Passed filters", f"{diag.get('passed', 0):,}")
    if raw is not None and not raw.empty and "Score" in raw.columns:
        m3.metric("Avg score", f"{raw['Score'].mean():.2f}")
        m4.metric("Top score", f"{raw['Score'].max():.2f}")
    else:
        m3.metric("Avg score", "—")
        m4.metric("Top score", "—")

    if display is None or display.empty:
        st.warning("No stocks passed. Loosen the score threshold, turn off MA50, or pick another sector.")
        with st.expander("Filter diagnostics", expanded=True):
            st.write(
                f"After price: **{diag.get('price', 0)}** → "
                f"MA50: **{diag.get('ma50', 0)}** → "
                f"score: **{diag.get('score', 0)}** → "
                f"range: **{diag.get('range', 0)}** → "
                f"P/E: **{diag.get('pe', 0)}** → "
                f"revenue: **{diag.get('rev', 0)}**"
            )
        return

    label = st.session_state.get("_hs_sector", "All Sectors")
    st.subheader(f"Top {len(display)} — {label}")
    st.caption("Click a ticker to open the Money Weather analysis under the table.")

    tickers = list(display["Ticker"]) if "Ticker" in display.columns else []
    if tickers:
        cols = st.columns(min(10, len(tickers)))
        for i, tk in enumerate(tickers):
            with cols[i % len(cols)]:
                sel = st.session_state.get(_k("inline")) == tk
                if st.button(tk, key=f"hs_tk_{tk}_{i}", width="stretch",
                             type="primary" if sel else "secondary"):
                    st.session_state[_k("inline")] = tk
                    st.session_state["lk_tk"] = tk

    order = [c for c in [
        "Ticker", "Sector", "Price", "MarketCap", "P/E",
        "OwnerEarnings", "MA50", "RangeHigh", "RangeLow", "RangePos",
        "MFI_Signal", "Score", "Short % Float", "Days to Cover",
        "Div Yield", "Div Rate",
    ] if c in display.columns]
    try:
        styled = display.style.map(_color_score, subset=["Score"]) if "Score" in display.columns else display
        st.dataframe(styled, width="stretch", height=560, column_order=order or None)
    except Exception:
        st.dataframe(display, width="stretch", height=560, column_order=order or None)

    st.download_button(
        "⬇️ Download CSV",
        data=_to_csv(display),
        file_name=f"hybrid_screener_{datetime.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width="stretch",
        key=_k("dl_csv"),
    )
