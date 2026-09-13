"""
Shakeout tab — coils at range lows after a short-term washout.

Money Weather:
    from storm_watch_tab import render_storm_watch_tab
    with tab_shakeout:
        render_storm_watch_tab(asof=asof, closes=closes, gauge=GAUGE)

Standalone Hybrid Screener still works with render_storm_watch_tab().
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

ACCENT = "#E87722"
GREEN = "#3fbf7f"
RED = "#e05252"
DIM = "#9aa8bd"
TEAL = "#5DCAA5"

BASIC_MIN_SHAKEOUT = 3
BASIC_N_SHOW = 12
SCAN_UNIVERSE = 250

HELP = {
    "mode": "Basic uses the backtest defaults and hides the rest. "
            "Advanced shows every filter (macro lens, hot sectors, shakeout gate, list length).",
    "macro_lens": "A playbook overlaid on the ranking. Off = no sector tilt. "
                  "Auto = the regime Money Weather detects live. A named lens re-orders "
                  "names toward that scenario's winners — it does not change the shakeout "
                  "checklist. Full playbooks live in the Lenses tab.",
    "hot_sectors": "Replace the sector list with the sectors that received the most money "
                   "in the chosen session window. Same hot-sector read the Top 20 tab uses.",
    "sectors": "Limit the scan to these sectors. The top-N cut is applied inside your "
               "selection, so you still get a full list from the names you picked.",
    "flow_window": "Which session(s) the hot-sector reading uses. Single day = one rotation. "
                   "Range = combined money flow over the last N sessions.",
    "min_shakeout": "Checklist 0–5: 5d down, lower 40% of the 63d range, RSI < 40, "
                    "sector residual > −3%, vol/range compressed. 3 is the validated default.",
    "n_show": "How many coils to list after filters.",
    "scan": "Nothing ranks until you hit this. First scan takes ~15s, then cached for an hour.",
    "watchlist": "Saves the ticker to the same Money Weather watchlist as Stock Lookup.",
    "chart": "Last ~180 sessions from the nightly dump. SMA50 / SMA150 overlaid.",
    "score": "Equal-weight cross-sectional rank of 5d reversal, range-low coil, peer catch-up, "
             "lower Bollinger stretch, and RSI oversold — then multiplied by the macro-lens sector fit.",
    "ret5": "Five-session return. For this setup a mild selloff (−2% to −12%) is the shakeout; "
            "a crash worse than −22% is filtered out as distress.",
    "rangepos": "Where price sits in the 63-day high-low range. Low = coiled at support (good here).",
    "rsi": "14-day RSI. Below 35 is the oversold zone this coil wants; above 50 is not a washout.",
    "resid21": "21-day return vs the equal-weight sector. Near zero or slightly positive means "
               "the sector has not abandoned the name.",
    "volratio": "10-day realized vol vs 63-day. Below ~0.85 means the range has compressed.",
    "macrofit": "Sector multiplier from the chosen macro playbook. Above 1.00 = a sector that "
                "regime historically favors.",
}


def _in_mw() -> bool:
    return "cascade_engine" in sys.modules


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
def _live(n: int = SCAN_UNIVERSE, dump_mtime: float = 0.0):
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


def _css(good: bool | None):
    if good is None:
        return f"color:{DIM}"
    return f"color:{GREEN};font-weight:600" if good else f"color:{RED};font-weight:600"


def _band_ret5(v):
    if v is None or not np.isfinite(v):
        return ""
    if -0.12 <= v <= -0.02:
        return _css(True)
    if v < -0.18 or v > 0:
        return _css(False)
    return f"color:{DIM}"


def _band_range(v):
    if v is None or not np.isfinite(v):
        return ""
    if v <= 0.25:
        return _css(True)
    if v >= 0.50:
        return _css(False)
    return f"color:{DIM}"


def _band_rsi(v):
    if v is None or not np.isfinite(v):
        return ""
    if v <= 35:
        return _css(True)
    if v >= 50:
        return _css(False)
    return f"color:{DIM}"


def _band_shake(v):
    if v is None or not np.isfinite(v):
        return ""
    if v >= 4:
        return _css(True)
    if v >= 3:
        return f"color:{DIM}"
    return _css(False)


def _band_score(v):
    if v is None or not np.isfinite(v):
        return ""
    if v >= 0.80:
        return _css(True)
    if v >= 0.70:
        return f"color:{DIM}"
    return _css(False)


def _band_resid(v):
    if v is None or not np.isfinite(v):
        return ""
    if v >= -0.03:
        return _css(True)
    if v <= -0.12:
        return _css(False)
    return f"color:{DIM}"


def _band_vol(v):
    if v is None or not np.isfinite(v):
        return ""
    if v <= 0.85:
        return _css(True)
    if v >= 1.20:
        return _css(False)
    return f"color:{DIM}"


def _band_fit(v):
    if v is None or not np.isfinite(v):
        return ""
    if abs(v - 1.0) < 0.02:
        return f"color:{DIM}"
    return _css(v > 1.0)


def _chip(label, value, css, tip=""):
    title = f' title="{tip}"' if tip else ""
    return (f'<div style="background:#0c1829;border:1px solid #1d2b40;border-radius:8px;'
            f'padding:8px 10px;min-width:110px;flex:1;"{title}>'
            f'<div style="font-size:11px;color:{DIM};letter-spacing:.4px;">{label}</div>'
            f'<div style="font-size:16px;{css}">{value}</div></div>')


def _flow_window(advanced: bool):
    """Returns (lookback, offset, label) — last session in Basic."""
    if not advanced or not _in_mw():
        return 1, 0, "last session"
    import cascade_engine as ce
    try:
        sess = ce.flow_sessions()
    except Exception:
        sess = []
    c1, c2 = st.columns([1, 2])
    mode = c1.radio(
        "Window", ["Single day", "Range"], horizontal=True,
        key="sw_flowmode", label_visibility="collapsed",
        help=HELP["flow_window"])
    if mode == "Single day":
        if not sess:
            return 1, 0, "last session"
        pick = c2.selectbox(
            "Session", sess, index=0, key="sw_flowday",
            label_visibility="collapsed",
            help="Any of the last 15 trading days — same picker as Top 20.")
        off = sess.index(pick)
        return 1, off, ("last session" if off == 0 else f"session of {pick}")
    nmax = int(getattr(ce, "SECTOR_FLOW_MAX_BACK", 15))
    n = c2.slider(
        "Sessions", 2, nmax, 5, key="sw_flowrange",
        label_visibility="collapsed",
        help="Combined money flow over this many recent sessions.")
    return int(n), 0, f"last {int(n)} sessions"


def _resolve_lens(label, closes, gauge):
    import cascade_engine as ce
    if label == "off":
        return None, "no macro lens"
    if label is None:
        try:
            key = ce.macro_regime(closes, pressure_gauge=gauge)["regime"]
        except Exception:
            key = "base"
        return key, f"auto · {ce.REGIME_NAMES.get(key, key)}"
    return label, ce.REGIME_NAMES.get(label, label)


def _apply_macro(picks: pd.DataFrame, regkey: str | None) -> pd.DataFrame:
    out = picks.copy()
    if not regkey or not _in_mw():
        out["MacroFit"] = 1.0
        return out
    import cascade_engine as ce
    tilts = ce.SECTOR_TILTS.get(regkey, {}) or {}
    out["MacroFit"] = out["Sector"].map(lambda s: float(tilts.get(s, 1.0)))
    out["StormScore"] = out["StormScore"] * out["MacroFit"]
    return out.sort_values("StormScore", ascending=False)


def _leaderboard_html(factors: list, univ_hw: float) -> str:
    rows = [
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        "<thead><tr style='opacity:.7;text-align:left;'>"
        "<th style='padding:6px 8px;'>Method</th>"
        "<th style='padding:6px 8px;'>Kind</th>"
        "<th style='padding:6px 8px;'>IS IC</th>"
        "<th style='padding:6px 8px;'>OOS IC</th>"
        "<th style='padding:6px 8px;'>OOS excess 10d</th>"
        "<th style='padding:6px 8px;'>OOS +3% high-water</th>"
        "</tr></thead><tbody>"
    ]
    for i, f in enumerate(factors[:12]):
        ic_oos = f.get("ic10_oos")
        edge = f.get("oos_edge")
        hw = f.get("oos_hw")
        ic_c = GREEN if (ic_oos or 0) > 0 else RED
        bg = "rgba(63,191,127,0.08)" if i == 0 else "transparent"
        hw_s = "—" if hw is None else f"{hw:.0%}  ({(hw - univ_hw)*100:+.0f}pp vs univ)"
        rows.append(
            f"<tr style='background:{bg};'>"
            f"<td style='padding:6px 8px;font-weight:600;'>{f.get('factor')}</td>"
            f"<td style='padding:6px 8px;opacity:.7;'>{f.get('kind')}</td>"
            f"<td style='padding:6px 8px;'>{_num(f.get('ic10_is'), 3)}</td>"
            f"<td style='padding:6px 8px;color:{ic_c};font-weight:600;'>{_num(ic_oos, 3)}</td>"
            f"<td style='padding:6px 8px;'>{_pct(edge, 2)}</td>"
            f"<td style='padding:6px 8px;'>{hw_s}</td></tr>"
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

**What failed out of sample:** 63d momentum, golden cross, Donchian breakout,
MACD cross, VCP / continuation coils. Chasing strength was the trap.

**What held up:** short-term reversal, sitting in the *lower* part of the 63d
range, RSI oversold, stretch below the lower Bollinger, and peer catch-up.
The **shakeout coil** stack was the #1 OOS IC.

**Limits:** one year, one regime flip, survivorship in a live dump, close-to-close
fills. Snapshot fundamentals are a live overlay only. Research tool, not advice.
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


def _draw_chart(tk: str):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    df = pd.DataFrame()
    src = ""
    if _in_mw():
        import cascade_engine as ce
        try:
            df = ce.dump_ohlcv(tk)
            src = "nightly dump"
        except Exception:
            df = pd.DataFrame()
    if df is None or df.empty:
        st.caption("No history for that ticker in the dump.")
        return
    d = df.tail(180)
    has_ohlc = {"Open", "High", "Low"}.issubset(d.columns)
    has_vol = "Volume" in d.columns and d["Volume"].notna().any()
    fig = make_subplots(rows=2 if has_vol else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25] if has_vol else [1.0],
                        vertical_spacing=0.03)
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close,
            increasing_line_color=GREEN, decreasing_line_color=RED, name=tk),
            row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=d.index, y=d.Close, mode="lines",
                                 line=dict(color=ACCENT, width=2), name=tk),
                      row=1, col=1)
    c_full = df["Close"]
    fig.add_trace(go.Scatter(
        x=d.index, y=c_full.rolling(150).mean().reindex(d.index),
        name="SMA150", line=dict(color="#7fb2ff", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=d.index, y=c_full.rolling(50).mean().reindex(d.index),
        name="SMA50", line=dict(color=ACCENT, width=1)), row=1, col=1)
    if has_vol:
        vcol = np.where(d.Close.diff().fillna(0) >= 0, GREEN, RED)
        fig.add_trace(go.Bar(x=d.index, y=d.Volume, marker_color=vcol,
                             name="Volume", opacity=0.6), row=2, col=1)
    fig.update_layout(height=420, template="plotly_dark",
                      paper_bgcolor="#081325", plot_bgcolor="#0c1829",
                      font_color="#F6F4E9", showlegend=True,
                      legend=dict(orientation="h", y=1.05),
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch", key=f"sw_chart_{tk}")
    last = float(df["Close"].iloc[-1])
    sma50 = float(c_full.rolling(50).mean().iloc[-1]) if len(c_full) >= 50 else np.nan
    if np.isfinite(sma50):
        trend = "above" if last > sma50 else "below"
        st.caption(f"{tk} is **{trend}** its 50-session average · last 180 sessions · {src}. "
                   "Research view, not investment advice.")
    else:
        st.caption(f"Last 180 sessions · {src}. Research view, not investment advice.")


def render_storm_watch_tab(asof: str | None = None, closes=None, gauge=None):
    st.subheader("🌩 Shakeout coils")
    st.caption(
        "Names coiling at range lows after a short-term shakeout — the setup that "
        "preceded upside in the held-out half of the nightly dump. Nothing ranks "
        "until you hit Scan. Research tool, not investment advice."
    )

    mw = _in_mw()
    mode = st.radio(
        "Mode", ["Basic", "Advanced"], horizontal=True, key="sw_ui_mode",
        help=HELP["mode"])
    advanced = mode == "Advanced"

    # ── defaults (Basic) ────────────────────────────────────────────
    min_sh = BASIC_MIN_SHAKEOUT
    n_show = BASIC_N_SHOW
    lens_token = None          # Auto
    sec_filter = None
    lb, off, flbl = 1, 0, "last session"
    hot_now: list[str] = []

    if advanced:
        if mw:
            import cascade_engine as ce
            LENS_OFF = "🚫 Off — no macro tilt"
            LENS_AUTO = "📡 Auto — detect the live regime"
            lens_opts = {LENS_OFF: "off", LENS_AUTO: None}
            lens_opts.update({f"{v}": k for k, v in ce.REGIME_NAMES.items()})
            st.markdown("**Macro lens**")
            lens_label = st.selectbox(
                "Macro lens", list(lens_opts), key="sw_lens",
                label_visibility="collapsed", help=HELP["macro_lens"])
            lens_token = lens_opts.get(lens_label)
            if lens_token == "off":
                st.caption("No sector tilt. Pure shakeout ranking.")
            elif lens_token is None:
                st.caption("Uses whichever regime is detected now. 🔭 Lenses has the playbooks.")
            else:
                card = ce.REGIME_CARDS.get(lens_token, {})
                st.caption(card.get("thesis") or card.get("leads") or "")

            lb, off, flbl = _flow_window(True)
            try:
                all_secs = sorted({
                    str(s) for s in ce.load_dump_panel()[2]
                    if s and str(s) != "Unknown"
                })
            except Exception:
                all_secs = []
            hs1, hs2 = st.columns([1, 3])
            if hs1.button("🔥 Use today's hot sectors", key="sw_hot_btn",
                          width="stretch", help=HELP["hot_sectors"]):
                try:
                    hot = ce.hot_sectors(5, lookback=lb, offset=off)
                    if hot:
                        st.session_state["_sw_sectors_pending"] = hot
                        st.rerun()
                except Exception as he:
                    st.caption(f"Hot sectors unavailable: {he}")
            try:
                hot_now = ce.hot_sectors(5, lookback=lb, offset=off)
                if hot_now:
                    hs2.caption(f"🔥 Hottest ({flbl}): " + " · ".join(hot_now))
            except Exception:
                hot_now = []
            pending = st.session_state.pop("_sw_sectors_pending", None)
            if pending:
                st.session_state["sw_sectors"] = pending
            with st.expander("🔥 Where the money went in the last session"):
                try:
                    fl = ce.sector_flow(lookback=lb, offset=off)
                except Exception as fe:
                    fl = pd.DataFrame()
                    st.caption(f"Sector flow unavailable: {fe}")
                if fl is None or fl.empty:
                    st.caption("No sector-flow reading available yet.")
                else:
                    st.caption(
                        f"{flbl} · a sector is hot when money-weighted return, "
                        "breadth and turnover all lean the same way.")
                    fs = fl.copy()
                    mark = set(hot_now) if hot_now else set(str(s) for s in fs.Sector.head(5))
                    fs["Hot"] = ["🔥" if str(s) in mark else "" for s in fs.Sector]
                    st.dataframe(
                        fs[["Rank", "Hot", "Sector", "Ret", "RS", "Breadth", "VolSurge", "Names"]]
                        .style.format({"Ret": "{:+.2%}", "RS": "{:+.2%}",
                                       "Breadth": "{:.0%}", "VolSurge": "{:.2f}x"}),
                        width="stretch", hide_index=True,
                        column_config={
                            "Ret": st.column_config.Column(help="Dollar-weighted sector return."),
                            "RS": st.column_config.Column(help="Sector return minus the market."),
                            "Breadth": st.column_config.Column(help="Share of names in the sector that rose."),
                            "VolSurge": st.column_config.Column(help="Median dollar-volume vs its 63-day average."),
                        })
            ms_kw = {} if "sw_sectors" in st.session_state else {"default": all_secs}
            picked = st.multiselect(
                "Sectors", all_secs, key="sw_sectors", **ms_kw, help=HELP["sectors"])
            sec_filter = (None if (not picked or (all_secs and len(picked) == len(all_secs)))
                          else list(picked))
            if sec_filter:
                st.caption(f"🎯 Scanning {len(sec_filter)} of {len(all_secs)} sectors.")
        fc1, fc2 = st.columns(2)
        min_sh = fc1.slider(
            "Min shakeout (0–5)", 0, 5, BASIC_MIN_SHAKEOUT, key="sw_min_shakeout",
            help=HELP["min_shakeout"])
        n_show = fc2.selectbox(
            "How many stocks", list(range(8, 44, 4)),
            index=1, key="sw_n_show", help=HELP["n_show"])
    else:
        st.caption(
            f"Basic defaults: shakeout ≥ {BASIC_MIN_SHAKEOUT}/5 · top {BASIC_N_SHOW} · "
            "auto macro lens · every sector. Switch to Advanced to change them."
        )

    # recipe line
    if mw:
        import cascade_engine as ce
        regkey, lens_txt = _resolve_lens(lens_token, closes, gauge)
    else:
        regkey, lens_txt = None, "no macro lens"
    sec_txt = (f", limited to {len(sec_filter)} sectors ({flbl})"
               if sec_filter else "")
    st.markdown(
        f"""<div style="background:#0c1829;border-left:3px solid {ACCENT};
        padding:9px 14px;margin:8px 0 10px;font-size:13.5px;">
        Ranking shakeout coils by <b style="color:{ACCENT};">washout + coil at range lows</b>,
        through <b style="color:{ACCENT};">{lens_txt}</b>{sec_txt},
        shakeout ≥ <b>{min_sh}/5</b>.</div>""",
        unsafe_allow_html=True)

    b1, b2 = st.columns([2, 1])
    if b1.button("🚀 Scan shakeout coils", type="primary", key="sw_run",
                 width="stretch", help=HELP["scan"]):
        st.session_state["sw_go"] = True
        st.session_state.pop("sw_sel_tk", None)
    if b2.button("Clear results", key="sw_clear", width="stretch",
                 help="Drop the last scan so nothing is ranked until you scan again."):
        st.session_state["sw_go"] = False
        st.session_state.pop("sw_sel_tk", None)
        st.rerun()

    if not st.session_state.get("sw_go"):
        st.info("Hit **Scan shakeout coils** to rank the universe. Nothing runs until you do.")
        _render_backtest(_backtest())
        return

    try:
        picks, info = _live(SCAN_UNIVERSE, _dump_mtime())
    except Exception as e:
        st.error(f"Could not score Shakeout: {e}")
        return
    if picks is None or picks.empty:
        st.warning("No tradeable names passed the liquidity filter.")
        return

    ranked = _apply_macro(picks, regkey)
    if sec_filter:
        ranked = ranked[ranked.Sector.isin(sec_filter)]
    ranked = ranked[ranked.Shakeout >= min_sh]
    view = ranked.head(int(n_show)).copy()

    bt = _backtest()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("As of", info.get("as_of", "—"),
              help="Last session in the nightly dump used for this ranking.")
    c2.metric("Tape regime", info.get("regime", "—"),
              help="Trailing momentum-vs-reversal spread. Reversal = fade the dip; "
                   "momentum = the continuation playbook (which failed OOS this year).")
    c3.metric("Tradeable names", f"{info.get('n_tradeable', 0):,}",
              help="Price ≥ $5 and 21d median dollar volume ≥ $2M.")
    if bt and bt.get("composite"):
        ex = bt["composite"].get("top_ex10_oos")
        c4.metric("OOS top-decile excess / 10d", _pct(ex, 2) if ex is not None else "—",
                  help="Held-out excess return of the composite's top decile over 10 sessions.")
    else:
        c4.metric("Live factors", str(len(LIVE_FACTORS)))

    if view.empty:
        st.info("No names match those filters — lower the shakeout minimum or add sectors.")
        _render_backtest(bt)
        return

    show = view[[
        "Ticker", "Sector", "Price", "StormScore", "Shakeout", "Ret5", "RangePos",
        "RSI", "Resid21", "VolRatio", "MacroFit", "Piotroski", "Quality",
        "SuggestedStop",
    ]].copy() if "MacroFit" in view.columns else view.copy()

    styler = (
        show.style.format({
            "Price": "${:,.2f}", "StormScore": "{:.3f}", "Shakeout": "{:.0f}",
            "Ret5": "{:+.1%}", "RangePos": "{:.0%}", "RSI": "{:.0f}",
            "Resid21": "{:+.1%}", "VolRatio": "{:.2f}", "MacroFit": "{:.2f}",
            "Piotroski": "{:.0f}", "SuggestedStop": "${:,.2f}",
        }, na_rep="—")
        .map(lambda v: _band_score(v), subset=["StormScore"])
        .map(lambda v: _band_shake(v), subset=["Shakeout"])
        .map(lambda v: _band_ret5(v), subset=["Ret5"])
        .map(lambda v: _band_range(v), subset=["RangePos"])
        .map(lambda v: _band_rsi(v), subset=["RSI"])
        .map(lambda v: _band_resid(v), subset=["Resid21"])
        .map(lambda v: _band_vol(v), subset=["VolRatio"])
    )
    if "MacroFit" in show.columns:
        styler = styler.map(lambda v: _band_fit(v), subset=["MacroFit"])

    st.subheader(f"Shakeout coils · {len(show)} of {len(ranked)}")
    st.caption("Green = in the zone this setup wants · red = against it · dim = noise. "
               "Tap a row for the chart and watchlist.")
    sel = st.dataframe(
        styler, width="stretch", hide_index=True, height=min(740, 80 + 34 * len(show)),
        on_select="rerun", selection_mode="single-row", key="sw_table",
        column_config={
            "StormScore": st.column_config.Column(help=HELP["score"]),
            "Shakeout": st.column_config.Column(help=HELP["min_shakeout"]),
            "Ret5": st.column_config.Column(help=HELP["ret5"]),
            "RangePos": st.column_config.Column(help=HELP["rangepos"]),
            "RSI": st.column_config.Column(help=HELP["rsi"]),
            "Resid21": st.column_config.Column(help=HELP["resid21"]),
            "VolRatio": st.column_config.Column(help=HELP["volratio"]),
            "MacroFit": st.column_config.Column(help=HELP["macrofit"]),
            "Piotroski": st.column_config.Column(help="9-point fundamental health from the dump. Overlay only — not in the backtest."),
            "SuggestedStop": st.column_config.Column(help=f"Entry {STOP_PCT:.0%} — the stop that improved expectancy in the dump backtest."),
        })
    rows = sel.selection.rows if sel and getattr(sel, "selection", None) else []
    if rows:
        st.session_state["sw_sel_tk"] = str(show.iloc[rows[0]].Ticker)
        st.session_state["lk_tk"] = st.session_state["sw_sel_tk"]

    tk = st.session_state.get("sw_sel_tk")
    if tk and tk in set(show.Ticker.astype(str)):
        row = show[show.Ticker.astype(str) == tk].iloc[0]
        chips = "".join([
            _chip("5-day return", _pct(row.Ret5), _band_ret5(row.Ret5), HELP["ret5"]),
            _chip("Range position", _pct(row.RangePos, 0, signed=False),
                  _band_range(row.RangePos), HELP["rangepos"]),
            _chip("RSI", _num(row.RSI, 0), _band_rsi(row.RSI), HELP["rsi"]),
            _chip("Vs sector 21d", _pct(row.Resid21), _band_resid(row.Resid21), HELP["resid21"]),
            _chip("Vol ratio", _num(row.VolRatio, 2), _band_vol(row.VolRatio), HELP["volratio"]),
            _chip("Macro fit", _num(row.MacroFit, 2) if "MacroFit" in show.columns else "—",
                  _band_fit(row.MacroFit) if "MacroFit" in show.columns else "", HELP["macrofit"]),
        ])
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px;">{chips}</div>',
            unsafe_allow_html=True)

        w1, w2 = st.columns([1, 2])
        on_list = False
        if mw:
            import cascade_engine as ce
            try:
                on_list = any(w.get("ticker") == tk for w in ce.watchlist_load())
            except Exception:
                on_list = False
            if on_list:
                w1.caption(f"⭐ **{tk}** is on your watchlist")
            elif w1.button(f"⭐ Add {tk} to watchlist", key=f"sw_wl_{tk}",
                           width="stretch", help=HELP["watchlist"]):
                try:
                    ce.watchlist_add(tk, float(row.Price), note="shakeout coil")
                    st.toast(f"⭐ {tk} saved to your watchlist at ${float(row.Price):,.2f}")
                    st.rerun()
                except Exception as we:
                    st.error(f"Watchlist add failed: {we}")
        w2.caption("Row also loads **Stock Lookup** — open that tab for the full analyzer.")
        st.markdown(f"**{tk}** · ${float(row.Price):,.2f} · shakeout {int(row.Shakeout)}/5")
        _draw_chart(tk)

    st.download_button(
        "⬇️ Download Shakeout CSV",
        view.to_csv(index=False).encode(),
        file_name=f"shakeout_{info.get('as_of', 'scan')}.csv",
        mime="text/csv",
        key="sw_download",
        width="stretch",
    )
    _render_backtest(bt)
