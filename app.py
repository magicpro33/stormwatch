"""
🌩 MONEY WEATHER — the Global Flow Cascade map.

Standalone Streamlit app. Money propagates through the world's assets like
weather fronts: fast frictionless nodes first, slow heavy ones last. This app
tracks the pressure system (global liquidity), the sentinels (24/7 early-
warning assets), the cascade graph (empirical storm tracks between ~50 global
nodes), and the forced-flow calendar (scheduled, mechanical money movement).

Run:  streamlit run app.py
First load downloads ~3 years of daily data for ~50 tickers (1-2 min), then
caches to data/history.parquet.
"""
import os
os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")   # curl_cffi segfault guard

import numpy as np
import pandas as pd
import streamlit as st

import cascade_engine as ce

st.set_page_config(page_title="Money Weather", page_icon="🌩", layout="wide")

ACCENT = "#E87722"
GREEN = "#3fbf7f"
RED = "#e05252"

st.markdown(
    f"""
    <div style="padding:6px 0 2px;">
      <span style="font-size:30px;font-weight:700;">🌩 Money Weather</span>
      <span style="color:{ACCENT};font-size:15px;margin-left:10px;">
        the global flow cascade map</span>
    </div>
    <div style="color:#9aa8bd;font-size:13px;margin-bottom:10px;">
      Money doesn't teleport — it propagates. Track the pressure, watch the
      sentinels, follow the storm tracks. Probability tilts, not prophecy.
    </div>
    """, unsafe_allow_html=True)


# ── data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading 3 years of global node history…")
def _history():
    return ce.fetch_history()


@st.cache_data(ttl=3600, show_spinner=False)
def _edges(_key: str):
    return ce.estimate_edges(_history())


@st.cache_data(ttl=1800, show_spinner="Reading the pressure system…")
def _pressure():
    return ce.pressure_system()


try:
    closes = _history()
except Exception as e:
    st.error(f"Could not load market history: {e}")
    st.stop()

asof = str(closes.index[-1].date())
tab_map, tab_pressure, tab_sentinels, tab_forced, tab_lab = st.tabs(
    ["🌊 Cascade Map", "🌡 Pressure", "🛰 Sentinels", "📅 Forced Flows", "🔬 Validation Lab"])


# ── 🌊 cascade map ───────────────────────────────────────────────────
with tab_map:
    st.caption(f"Data through {asof} · edges re-estimated on the trailing "
               f"{ce.EDGE_TRAIN} sessions · forecasts look {ce.EDGE_HORIZON} "
               f"sessions ahead.")
    edges = _edges(asof)
    waves = ce.active_waves(closes, edges)

    c1, c2, c3 = st.columns(3)
    c1.metric("Storm tracks (edges)", len(edges))
    c2.metric("Active waves now", int(waves.source.nunique()) if not waves.empty else 0)
    c3.metric("Downstream forecasts", len(waves))

    if waves.empty:
        st.info("🌤 Calm skies — no node's flow impulse exceeds the wave "
                "threshold right now. Check the Sentinels tab for early "
                "twitches, or lower the threshold in cascade_engine.py.")
    else:
        st.subheader("Active fronts → downstream forecasts")
        show = waves.copy()
        show["confidence"] = (show.hit_rate.fillna(0.5) * 100).round(0).astype(int).astype(str) + "%"
        show = show[["source_name", "source_z", "call", "target_name",
                     "horizon_days", "edge_ic", "confidence"]]
        show.columns = ["Wave source", "Impulse z", "Forecast", "Target",
                        "Days", "Edge IC", "Hist. hit rate"]
        st.dataframe(
            show.style.map(
                lambda v: f"color:{GREEN};font-weight:600" if v == "📈 UP"
                else (f"color:{RED};font-weight:600" if v == "📉 DOWN" else ""),
                subset=["Forecast"]),
            width="stretch", hide_index=True, height=430)
        st.caption("Read it like a weather report: *\"a front entered "
                   "[source]; it historically reaches [target] within "
                   "[days] sessions, [hit rate] of the time.\"* "
                   "Research tool — size positions like forecasts can be wrong, "
                   "because they can.")

    with st.expander("🗺 Strongest storm tracks (full edge list)"):
        e = edges.copy()
        e["hit"] = (e.hit_rate * 100).round(0)
        st.dataframe(
            e[["source_name", "target_name", "ic", "hit"]].rename(columns={
                "source_name": "Leads", "target_name": "Follows",
                "ic": "IC", "hit": "Hit %"}).head(60),
            width="stretch", hide_index=True)


# ── 🌡 pressure ──────────────────────────────────────────────────────
with tab_pressure:
    st.caption("The upstream source of every wave: global net liquidity. "
               "Rising pressure = waves travel far. Draining = fade the rallies.")
    try:
        p = _pressure()
        st.markdown(f"### {p['gauge_label']}")
        cols = st.columns(min(4, max(1, len(p["components"]))))
        for (k, v), c in zip(p["components"].items(), cols * 3):
            c.metric(k, f"{v:,.1f}")
        g1, g2 = st.columns(2)
        if "netliq" in p:
            g1.markdown("**Net US Liquidity ($tn)** — Fed BS − TGA − RRP")
            g1.line_chart(p["netliq"].tail(500), height=220)
        if "stables" in p:
            g2.markdown("**Stablecoin supply ($bn)** — crypto dry powder")
            g2.line_chart(p["stables"].tail(500), height=220)
        if "hy_oas" in p:
            st.markdown("**High-yield credit spread (%)** — the market's blood pressure (down = risk-on)")
            st.line_chart(p["hy_oas"].tail(500), height=200)
        for err in p.get("errors", []):
            st.caption(f"⚠️ {err}")
    except Exception as e:
        st.warning(f"Pressure feeds unavailable right now: {e}")


# ── 🛰 sentinels ─────────────────────────────────────────────────────
with tab_sentinels:
    st.caption("The 24/7 early-warning line — fast, frictionless assets that "
               "react to pressure changes first. Crypto trades all weekend; "
               "Saturday knows things about Monday.")
    sb = ce.sentinel_board(closes)
    if sb.empty:
        st.info("No sentinel data loaded.")
    else:
        st.dataframe(
            sb.style.format({"Last": "{:,.2f}", "5d %": "{:+.1f}%",
                             "21d %": "{:+.1f}%", "ImpulseZ": "{:+.2f}"})
            .map(lambda v: f"color:{GREEN}" if isinstance(v, str) and "surging" in v
                 else (f"color:{RED}" if isinstance(v, str) and "dumping" in v else ""),
                 subset=["Signal"]),
            width="stretch", hide_index=True)
        hot = sb[sb.ImpulseZ.abs() >= ce.WAVE_Z]
        if not hot.empty:
            names = ", ".join(hot.Sentinel)
            st.markdown(f"⚡ **Sentinels firing:** {names} — check the Cascade "
                        "Map for where these waves historically travel next.")


# ── 📅 forced flows ──────────────────────────────────────────────────
with tab_forced:
    st.caption("The closest thing to prophecy that legally exists: flows that "
               "are scheduled and price-insensitive. They don't care what the "
               "chart looks like — they have to trade.")
    ff = ce.forced_flows()
    for _, ev in ff.iterrows():
        days = (ev.Date - pd.Timestamp.today().date()).days
        st.markdown(
            f"""<div style="background:#0c1829;border:1px solid #1d2b40;
            border-radius:10px;padding:10px 14px;margin-bottom:8px;">
            <span style="color:{ACCENT};font-weight:700;">{ev.Date:%a %b %d}</span>
            <span style="color:#9aa8bd;"> · in {days}d</span><br>
            <span style="font-weight:600;">{ev.Event}</span><br>
            <span style="color:#9aa8bd;font-size:13px;">{ev['Why it moves money']}</span>
            </div>""", unsafe_allow_html=True)


# ── 🔬 validation lab ────────────────────────────────────────────────
with tab_lab:
    st.caption("Trust nothing you haven't walk-forward tested. This re-runs "
               "the honest experiment: weekly, re-estimate the graph on "
               "trailing data only, follow the top-5 wave forecasts, compare "
               "to the equal-weight universe.")
    if st.button("🔬 Run walk-forward validation", type="primary"):
        with st.spinner("Walking forward through history — a few minutes…"):
            bt, summ = ce.backtest(closes)
        if not summ:
            st.warning("Not enough history to validate — need ~1.5 years of data.")
        else:
            c = st.columns(4)
            c[0].metric("Periods", summ["periods"])
            c[1].metric("Mean 10d fwd", f"{summ['mean_fwd']:+.2%}",
                        f"{summ['mean_excess']:+.2%} vs universe")
            c[2].metric("Beat universe", f"{summ['hit_vs_bench']:.0%}")
            c[3].metric("Sharpe (strat vs univ)",
                        f"{summ['sharpe']:.2f} / {summ['bench_sharpe']:.2f}")
            st.markdown(
                f"**Honesty split** — 1st half excess: `{summ['h1_excess']:+.2%}` · "
                f"2nd half excess: `{summ['h2_excess']:+.2%}` "
                + ("✅ holds up out-of-sample" if summ['h2_excess'] > 0 else
                   "⚠️ second half is weak — treat forecasts skeptically"))
            eq = (1 + bt.set_index("date")[["strat", "bench"]]).cumprod()
            st.line_chart(eq, height=260)
            st.dataframe(bt.tail(20).style.format(
                {"strat": "{:+.2%}", "bench": "{:+.2%}", "excess": "{:+.2%}"}),
                width="stretch", hide_index=True)
    st.caption("One year is one regime. Cascade edges break when regimes "
               "flip — that is why they are re-estimated every week and why "
               "this tab exists. Not investment advice.")
