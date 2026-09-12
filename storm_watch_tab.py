"""
Storm Watch tab — shakeout coils that tend to lift.

Drop-in for any Streamlit app (this screener, or another Storm Watch app):

    from storm_watch_tab import render_storm_watch_tab
    with tab_storm:
        render_storm_watch_tab()

Computes from data/stock_data.json.gz. Backtest scoreboard is read from
data/storm_watch_backtest.json when present (written by storm_watch_engine.py).
"""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

from storm_watch_engine import LIVE_FACTORS, STOP_PCT, score_from_data, score_from_panel

RAW_BASE = "https://raw.githubusercontent.com/magicpro33/stock/main/data"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "data")
DUMP_NAME = "stock_data.json.gz"
BACKTEST_NAME = "storm_watch_backtest.json"


def _in_mw() -> bool:
    return "cascade_engine" in sys.modules

POS = ["#9FE1CB", "#5DCAA5", "#1D9E75"]
NEG = ["#F09595", "#E24B4A", "#A32D2D"]
TEAL = "#5DCAA5"
CREAM = "#F6F4E9"


def _dump_paths():
    paths = [os.path.join(LOCAL_DIR, DUMP_NAME)]
    try:
        import mw_paths
        paths.append(os.path.join(mw_paths.data_dir(), DUMP_NAME))
        paths.append(os.path.join(mw_paths.bundle_dir(), "data", DUMP_NAME))
    except Exception:
        pass
    try:
        import cascade_engine as ce
        paths.append(getattr(ce, "LOCAL_DUMP_GZ", ""))
        reader = getattr(ce, "_data_read_path", None)
        if reader and getattr(ce, "LOCAL_DUMP_GZ", None):
            paths.append(reader(ce.LOCAL_DUMP_GZ))
    except Exception:
        pass
    out, seen = [], set()
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _load_dump():
    for path in _dump_paths():
        if os.path.exists(path):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
    import io
    import urllib.request
    with urllib.request.urlopen(f"{RAW_BASE}/{DUMP_NAME}", timeout=120) as r:
        return json.load(io.TextIOWrapper(
            gzip.GzipFile(fileobj=io.BytesIO(r.read())), encoding="utf-8"))


def _panel_from_ce(ce):
    panel, tickers, sectors, mdv, dts = ce.load_dump_panel()
    fa = {}
    try:
        fa = ce.dump_fundamentals_all()
    except Exception:
        pass
    dates = np.array([str(pd.Timestamp(d).date()) for d in dts])
    fund = pd.DataFrame(index=list(tickers))
    fund.index.name = "Ticker"
    for col in ("Piotroski", "ROIC", "OE_Yield", "ShortPctFloat",
                "DaysToCover", "MarketCap", "P/E", "GrossMargin"):
        arr = fa.get(col) if isinstance(fa, dict) else None
        if arr is not None and len(arr) == len(tickers):
            fund[col] = arr
    return dict(
        O=np.asarray(panel["o"], dtype=np.float32),
        H=np.asarray(panel["h"], dtype=np.float32),
        L=np.asarray(panel["l"], dtype=np.float32),
        C=np.asarray(panel["c"], dtype=np.float32),
        V=np.nan_to_num(panel["v"]).astype(np.float32),
        sectors=np.array(sectors),
        tickers=np.array(tickers),
        dates=dates,
        fund=fund,
    )


def _dump_mtime() -> float:
    try:
        import cascade_engine as ce
        return float(ce._dump_mtime() or 0.0)
    except Exception:
        return 0.0


@st.cache_data(ttl=3600, show_spinner="Scanning for shakeout coils…")
def _live(n: int = 80, dump_mtime: float = 0.0):
    if _in_mw():
        import cascade_engine as ce
        return score_from_panel(_panel_from_ce(ce), n=n)
    data = _load_dump()
    picks, info = score_from_data(data, n=n)
    del data
    return picks, info


def _backtest() -> dict | None:
    candidates = [os.path.join(LOCAL_DIR, BACKTEST_NAME)]
    try:
        import mw_paths
        candidates.append(os.path.join(mw_paths.data_dir(), BACKTEST_NAME))
        candidates.append(os.path.join(mw_paths.bundle_dir(), "data", BACKTEST_NAME))
    except Exception:
        pass
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def _pct(v, digits=1, signed=True):
    if v is None or not np.isfinite(v):
        return "—"
    fmt = f"{{:{'+' if signed else ''}.{digits}%}}"
    return fmt.format(v)


def _num(v, digits=2):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def _setup_bits(r) -> list[str]:
    bits = []
    sh = r.get("Shakeout")
    if sh is not None and np.isfinite(sh):
        bits.append(f"shakeout {int(sh)}/5")
    if r.get("Ret5") is not None and np.isfinite(r.Ret5):
        bits.append(f"5d {_pct(r.Ret5)}")
    if r.get("RangePos") is not None and np.isfinite(r.RangePos):
        bits.append(f"range pos {r.RangePos:.0%}")
    if r.get("RSI") is not None and np.isfinite(r.RSI):
        bits.append(f"RSI {r.RSI:.0f}")
    if r.get("Resid21") is not None and np.isfinite(r.Resid21):
        bits.append(f"vs sector 21d {_pct(r.Resid21)}")
    if r.get("Quality"):
        bits.append(f"Piotroski {r.get('Piotroski') or '—'} ({r.Quality})")
    if not r.get("AboveMA50", True):
        bits.append("below 50d MA (washout)")
    return bits


def _leaderboard_html(factors: list, univ_hw: float) -> str:
    rows = []
    rows.append(
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr style='opacity:.7;text-align:left;'>"
        "<th style='padding:6px 8px;'>Method</th>"
        "<th style='padding:6px 8px;'>Kind</th>"
        "<th style='padding:6px 8px;'>IS IC</th>"
        "<th style='padding:6px 8px;'>OOS IC</th>"
        "<th style='padding:6px 8px;'>OOS excess 10d</th>"
        "<th style='padding:6px 8px;'>OOS +3% high-water</th>"
        "</tr></thead><tbody>"
    )
    for i, f in enumerate(factors[:12]):
        ic_oos = f.get("ic10_oos")
        edge = f.get("oos_edge")
        hw = f.get("oos_hw")
        ic_c = TEAL if (ic_oos or 0) > 0 else NEG[1]
        bg = "rgba(29,158,117,0.08)" if i == 0 else "transparent"
        hw_s = "—" if hw is None else f"{hw:.0%}  ({(hw - univ_hw)*100:+.0f}pp vs univ)"
        rows.append(
            f"<tr style='background:{bg};'>"
            f"<td style='padding:6px 8px;font-weight:600;'>{f.get('factor')}</td>"
            f"<td style='padding:6px 8px;opacity:.7;'>{f.get('kind')}</td>"
            f"<td style='padding:6px 8px;'>{_num(f.get('ic10_is'), 3)}</td>"
            f"<td style='padding:6px 8px;color:{ic_c};font-weight:600;'>{_num(ic_oos, 3)}</td>"
            f"<td style='padding:6px 8px;'>{_pct(edge, 2)}</td>"
            f"<td style='padding:6px 8px;'>{hw_s}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "".join(rows)


def _render_backtest(bt: dict | None):
    with st.expander("How this was chosen — 40 methods, IS / OOS split", expanded=False):
        st.markdown(
            """
**Question:** which signal identifies a name *right before* it goes up?

**Test:** 5,527 names, 252 sessions (2025-09-11 → 2026-09-11). Point-in-time
OHLCV only. Warmup 80 days, split **2026-04-24**. Ranked factors: Spearman IC
vs next 5/10/21d close-to-close, top-decile excess, and P(10d high ≥ +3%).
Binary events use the same forward window. Liquidity: price ≥ $5, 21d median
dollar volume ≥ $2M.

**What failed out of sample** (worked in H1, reversed in H2): 63d momentum,
residual momentum, golden cross, Donchian breakout, MACD cross, VCP / silent
accumulation / storm-coil continuation. Chasing strength was the trap.

**What held up:** short-term reversal, sitting in the *lower* part of the 63d
range, RSI oversold, stretch below the lower Bollinger, and *peer catch-up*
(lagged 5d vs a still-constructive 21d sector residual). Combining those into
a **shakeout coil** — sold off, coiled at lows, vol compressed, sector hasn't
abandoned it — was the #1 OOS IC.

**Outside the box:** the "calm before the storm" in *this* tape is not a
high-tight flag. It is a **washout that has stopped going down**. A
trailing-momentum regime switch was tested; it did not beat a straight
reversal book while the tape is in reversal mode (current reading).

**Limits:** one year, one regime flip, survivorship in a current-universe
dump, close-to-close fills, no borrow/cost model beyond the −8% stop used as
a risk sketch. Snapshot fundamentals (Piotroski, short interest) are a live
overlay only — they are not dated, so they were not backtested.
            """
        )
        if bt:
            meta = bt.get("meta") or {}
            univ_hw = float(meta.get("univ_hw10_p3") or 0.70)
            st.caption(
                f"Split {meta.get('split_date')} · IS {meta.get('is_days')}d / "
                f"OOS {meta.get('oos_days')}d · universe 10d return "
                f"{_pct(meta.get('univ_fwd10'), 2, signed=False)} · "
                f"universe P(high-water +3% in 10d) {univ_hw:.0%}."
            )
            st.markdown(_leaderboard_html(bt.get("factors") or [], univ_hw),
                        unsafe_allow_html=True)
            wk = meta.get("weekly_topn_oos") or {}
            if wk:
                st.caption(
                    f"Composite weekly top-25, OOS: avg 5d {_pct(wk.get('avg_5d'), 2)} vs "
                    f"universe {_pct(wk.get('univ_5d'), 2)} "
                    f"(excess {_pct(wk.get('excess_5d'), 2)}, "
                    f"{wk.get('n')} rebalances)."
                )
        else:
            st.info("Run `python storm_watch_engine.py data/stock_data.json.gz` to refresh the scoreboard.")


def render_storm_watch_tab():
    st.subheader("🌩 Shakeout coils")
    st.caption(
        "Finds names coiling at range lows after a short-term shakeout — "
        "the setup that actually preceded upside in the held-out half of the "
        "nightly dump. Classic breakouts, MACD crosses, and 63-day momentum "
        "all flipped negative out of sample. Research tool, not investment advice."
    )

    # Money Weather already has the dump panel — scan automatically.
    # Standalone Hybrid Screener keeps an explicit button so other tabs stay fast.
    go = True
    if not _in_mw():
        go = st.button("Scan shakeout coils", type="primary", key="sw_scan") or st.session_state.get("_sw_ran")
        if not go:
            st.info("Scan ranks the liquid universe (~15s the first time, then cached for an hour).")
            _render_backtest(_backtest())
            return
        st.session_state["_sw_ran"] = True

    try:
        picks, info = _live(80, _dump_mtime())
    except Exception as e:
        st.error(f"Could not score Storm Watch: {e}")
        return
    if picks is None or picks.empty:
        st.warning("No tradeable names passed the liquidity filter.")
        return

    bt = _backtest()
    regime = info.get("regime", "reversal")
    as_of = info.get("as_of", "—")
    n_tr = info.get("n_tradeable", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("As of", as_of)
    c2.metric("Regime", regime)
    c3.metric("Tradeable names", f"{n_tr:,}")
    if bt and bt.get("composite"):
        ex = bt["composite"].get("top_ex10_oos")
        c4.metric("OOS top-decile excess / 10d", _pct(ex, 2) if ex is not None else "—")
    else:
        c4.metric("Live factors", str(len(LIVE_FACTORS)))

    st.caption(
        f"Universe filter: price ≥ $5, 21d median dollar volume ≥ $2M. "
        f"Score = equal-weight cross-sectional rank of {', '.join(LIVE_FACTORS)}. "
        f"Shakeout checklist is 0–5 (5d down, lower 40% of 63d range, RSI < 40, "
        f"sector residual > −3%, vol/range compressed). Stop = entry {STOP_PCT:.0%}."
    )

    sectors = ["All"] + sorted(picks.Sector.dropna().unique().tolist())
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        min_sh = st.slider("Min shakeout (0–5)", 0, 5, 3, key="sw_min_shakeout")
    with fc2:
        sector = st.selectbox("Sector", sectors, key="sw_sector")
    with fc3:
        n_show = 30 if st.toggle("Show 30 picks", value=False, key="sw_show30") else 12

    view = picks[picks.Shakeout >= min_sh].copy()
    if sector != "All":
        view = view[view.Sector == sector]
    view = view.sort_values("StormScore", ascending=False)

    if view.empty:
        st.info("No names match those filters — lower the shakeout minimum.")
        _render_backtest(bt)
        return

    st.subheader(f"Shakeout coils · {len(view.head(n_show))} of {len(view)}")
    for _, r in view.head(n_show).iterrows():
        sh = int(r.Shakeout) if np.isfinite(r.Shakeout) else 0
        dots = "●" * sh + "○" * (5 - sh)
        title = f"{r.Ticker}  ·  ${r.Price:,.2f}  ·  score {r.StormScore:.2f}  ·  {dots}"
        with st.expander(title):
            a, b, c, d = st.columns(4)
            a.metric("5-day return", _pct(r.Ret5))
            b.metric("Range position", _pct(r.RangePos, 0, signed=False) if r.RangePos is not None else "—")
            c.metric("RSI", _num(r.RSI, 0))
            d.metric("Suggested stop", f"${r.SuggestedStop:,.2f}" if np.isfinite(r.SuggestedStop) else "—")
            e, f_, g, h = st.columns(4)
            e.metric("Vs sector 21d", _pct(r.Resid21))
            f_.metric("21d range", _pct(r.Range21, 0, signed=False) if r.Range21 is not None else "—")
            g.metric("Vol ratio 10/63", _num(r.VolRatio, 2))
            h.metric("Peer catch-up", _num(r.PeerCatchup, 3))
            st.caption(" · ".join(_setup_bits(r)))

    st.download_button(
        "Download Shakeout CSV",
        view.head(n_show).to_csv(index=False).encode(),
        file_name=f"shakeout_{as_of}.csv",
        mime="text/csv",
        key="sw_download",
    )

    _render_backtest(bt)
