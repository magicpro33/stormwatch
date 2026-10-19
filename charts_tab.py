"""Charts — official FRED / Treasury macro series.

M2: Federal Reserve H.6 via FRED M2SL (seasonally adjusted, monthly).
Debt / GDP: Treasury total public debt ÷ BEA GDP via FRED GFDEGDQ188S.
Dollar since 2020: CPI-U purchasing power via FRED CPIAUCSL.
Debt-clock pills: Treasury Debt to the Penny + FRED population, spending,
interest, and deficit series.
"""
from __future__ import annotations

import html as html_mod

import numpy as np
import pandas as pd
import streamlit as st

ACCENT = "#E87722"
GREEN = "#3fbf7f"
RED = "#e05252"
GOLD = "#e8c34a"
BLUE = "#7FD4FF"
DIM = "#9aa8bd"
INK = "#F6F4E9"
NAVY = "#0c1829"
PAPER = "#081325"
LINE = "#3d6a94"

M2_SID = "M2SL"
DEBT_SID = "GFDEGDQ188S"
CPI_SID = "CPIAUCSL"

# Income-taxpayers (not all returns). Same ~140M denominator debt clocks use.
TAXPAYERS = 140_000_000


def _esc(s) -> str:
    return html_mod.escape("" if s is None else str(s), quote=True)


def _html(markup: str) -> None:
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup.replace("$", "&#36;"), unsafe_allow_html=True)


def _sec(title: str, color: str) -> None:
    """Simple section title only — no source line, series id, or subtitle."""
    _html(f'<div class="hub-sec lg" style="--sc:{color}">{_esc(title)}</div>')


@st.cache_data(ttl=6 * 3600, show_spinner="Loading FRED series…")
def _load_series(sid: str) -> pd.Series:
    import cascade_engine as ce
    return ce._fred(sid)


def _try_series(sid: str) -> pd.Series:
    try:
        s = _load_series(sid)
        return s if s is not None else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=6 * 3600, show_spinner="Loading Treasury debt…")
def _treasury_debt_daily() -> pd.Series:
    """Daily total public debt outstanding. Dollars, not millions."""
    import requests
    url = ("https://api.fiscaldata.treasury.gov/services/api/"
           "fiscal_service/v2/accounting/od/debt_to_penny")
    r = requests.get(url, params={"sort": "-record_date", "page[size]": "40",
                                  "format": "json"}, timeout=20)
    r.raise_for_status()
    rows = r.json().get("data") or []
    s = pd.Series({pd.Timestamp(x["record_date"]):
                   float(x["tot_pub_debt_out_amt"]) for x in rows})
    return s.sort_index().dropna()


def _at_or_before(s: pd.Series, ts) -> float | None:
    if s is None or s.empty:
        return None
    hit = s[s.index <= pd.Timestamp(ts)]
    if hit.empty:
        return None
    v = float(hit.iloc[-1])
    return v if np.isfinite(v) else None


def _last(s: pd.Series) -> float | None:
    if s is None or s.empty:
        return None
    v = float(s.iloc[-1])
    return v if np.isfinite(v) else None


def _yoy(s: pd.Series, periods: int) -> float | None:
    if s is None or len(s) <= periods:
        return None
    a, b = float(s.iloc[-1]), float(s.iloc[-1 - periods])
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return None
    return (a / b - 1.0) * 100.0


def _delta(s: pd.Series, periods: int) -> float | None:
    if s is None or len(s) <= periods:
        return None
    a, b = float(s.iloc[-1]), float(s.iloc[-1 - periods])
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    return a - b


def _month_label(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %Y")


def _quarter_label(ts) -> str:
    t = pd.Timestamp(ts)
    return f"Q{(t.month - 1) // 3 + 1} {t.year}"


def _fmt_pct(v, signed=False, digits=1) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:+.{digits}f}%" if signed else f"{v:.{digits}f}%"


def _fmt_usd(n, digits=0) -> str:
    if n is None or not np.isfinite(n):
        return "—"
    return f"${n:,.{digits}f}"


def _fmt_scale(n, digits=1) -> str:
    if n is None or not np.isfinite(n):
        return "—"
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1e12:
        return f"{sign}${n / 1e12:.{digits}f}T"
    if n >= 1e9:
        return f"{sign}${n / 1e9:.{digits}f}B"
    if n >= 1e6:
        return f"{sign}${n / 1e6:.{digits}f}M"
    return f"{sign}${n:,.0f}"


def _pills(items: list[dict], grid: str = "ch-pills") -> None:
    cells = []
    for it in items:
        color = it.get("color") or ACCENT
        ico = it.get("icon") or ""
        feat = " feature" if it.get("feature") else ""
        ico_html = f"<div class='ch-ico'>{_esc(ico)}</div>" if ico else ""
        cells.append(
            f"<div class='ch-pill{feat}' style='--mc:{color}'>"
            f"{ico_html}"
            f"<div class='ch-mlbl'>{_esc(it.get('label', ''))}</div>"
            f"<div class='ch-mval'>{it.get('value', '—')}</div>"
            f"<div class='ch-msub'>{_esc(it.get('sub', ''))}</div></div>"
        )
    _html(f"<div class='{grid}'>{''.join(cells)}</div>")


def _metrics(items: list[tuple[str, str, str, str]]) -> None:
    _pills([{"label": a, "value": b, "sub": c, "color": d}
            for a, b, c, d in items])


def _line_fig(s: pd.Series, *, color: str, y_title: str, hover: str,
              y_prefix: str = "", y_suffix: str = "", y_mult: float = 1.0):
    import plotly.graph_objects as go

    y = s.astype(float) * y_mult
    fill = ("rgba(232,119,34,0.12)" if color == ACCENT
            else "rgba(127,212,255,0.10)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values,
        mode="lines",
        line=dict(color=color, width=2.2),
        fill="tozeroy",
        fillcolor=fill,
        hovertemplate=hover,
        name=y_title,
    ))
    fig.update_layout(
        height=420,
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=NAVY,
        font=dict(color=INK, family="JetBrains Mono, monospace", size=12),
        showlegend=False,
        margin=dict(l=12, r=12, t=28, b=8),
        xaxis=dict(
            gridcolor="#1e3a5f",
            zeroline=False,
            rangeselector=dict(
                buttons=[
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(count=10, label="10Y", step="year", stepmode="backward"),
                    dict(count=20, label="20Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#122a42",
                activecolor=color,
                font=dict(color=INK, size=11),
                x=0, y=1.08, xanchor="left",
            ),
            rangeslider=dict(visible=True, thickness=0.08,
                             bgcolor="#0a1624", bordercolor=LINE),
        ),
        yaxis=dict(
            title=y_title,
            gridcolor="#1e3a5f",
            zeroline=False,
            tickprefix=y_prefix,
            ticksuffix=y_suffix,
            separatethousands=True,
        ),
        hovermode="x unified",
    )
    return fig


def _css() -> None:
    _html(
        f"""<style>
.ch-pills{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:8px 0 12px}}
.ch-hero{{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:10px;margin:8px 0 18px}}
@media(max-width:900px){{
  .ch-pills{{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .ch-hero{{grid-template-columns:1fr}}
}}
.ch-pill{{background:{NAVY};border:1px solid {LINE};border-radius:14px;
  padding:14px 16px 12px;position:relative;min-height:92px}}
.ch-pill.feature{{border-left:5px solid var(--mc,{RED});padding:16px 18px}}
.ch-ico{{position:absolute;top:12px;right:14px;font-size:16px;opacity:.55}}
.ch-mlbl{{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:{DIM};padding-right:22px}}
.ch-mval{{font-family:ui-monospace,JetBrains Mono,monospace;font-size:26px;font-weight:800;
  color:var(--mc,{INK});margin-top:6px;line-height:1.1;letter-spacing:0;
  font-variant-numeric:tabular-nums}}
.ch-pill.feature .ch-mval{{font-size:38px}}
.ch-msub{{font-size:12px;color:{DIM};margin-top:4px;line-height:1.35}}
.st-key-ch_m2{{border-left:5px solid {ACCENT} !important;border-radius:10px}}
.st-key-ch_debt{{border-left:5px solid {BLUE} !important;border-radius:10px}}
</style>"""
    )


def _dollar_since_2020() -> None:
    _sec("Your dollar since 2020", RED)
    cpi = _try_series(CPI_SID)
    m2 = _try_series(M2_SID)
    c0 = _at_or_before(cpi, "2020-01-31")
    c1 = _last(cpi)
    lost = left = infl = None
    if c0 and c1 and c1 > 0:
        infl = (c1 / c0 - 1.0) * 100.0
        left = (c0 / c1) * 100.0
        lost = 100.0 - left
    m0 = _at_or_before(m2, "2020-01-31")
    m1 = _last(m2)
    m2_up = ((m1 / m0 - 1.0) * 100.0) if (m0 and m1 and m0 > 0) else None
    asof = _month_label(cpi.index[-1]) if cpi is not None and len(cpi) else "—"
    cents = f"{left:.0f}¢" if left is not None else "—"
    _pills([
        {"label": "Purchasing power lost",
         "value": _fmt_pct(lost),
         "sub": f"What $1 from Jan 2020 has lost through {asof}. "
                f"It now buys {cents} of the same goods.",
         "color": RED, "feature": True},
        {"label": "$1 today buys",
         "value": cents if left is not None else "—",
         "sub": "of a January 2020 dollar",
         "color": ACCENT},
        {"label": "Prices since 2020",
         "value": _fmt_pct(infl, signed=True),
         "sub": (f"M2 {m2_up:+.0f}%" if m2_up is not None
                 else "Jan 2020 → now"),
         "color": GOLD},
    ], grid="ch-hero")


def _debt_snapshot() -> dict:
    """Live Treasury debt plus FRED fiscal series for the clock pills."""
    daily = pd.Series(dtype=float)
    try:
        daily = _treasury_debt_daily()
    except Exception:
        daily = pd.Series(dtype=float)
    gfde = _try_series("GFDEBTN")
    pop = _try_series("POPTHM")
    gdp = _try_series("GDP")
    spend = _try_series("FGEXPND")
    interest = _try_series("A091RC1Q027SBEA")
    deficit_m = _try_series("MTSDS133FMS")
    fy_def = _try_series("FYFSD")
    fy_out = _try_series("FYONET")
    d2g = _try_series(DEBT_SID)

    debt = _last(daily)
    debt_asof = daily.index[-1] if len(daily) else None
    if debt is None:
        v = _last(gfde)
        if v is not None:
            debt = v * 1_000_000.0
            debt_asof = gfde.index[-1]

    added = None
    if len(daily) >= 2:
        added = float(daily.iloc[-1] - daily.iloc[-2])

    people = _last(pop)
    if people is not None:
        people = people * 1000.0  # POPTHM is thousands

    per_cit = (debt / people) if (debt and people) else None
    per_tax = (debt / TAXPAYERS) if debt else None

    gdp_d = _last(gdp)
    ratio = None
    if debt and gdp_d and gdp_d > 0:
        ratio = debt / (gdp_d * 1e9) * 100.0
    elif _last(d2g) is not None:
        ratio = _last(d2g)

    spend_d = _last(spend)
    if spend_d is not None:
        spend_d *= 1e9  # FGEXPND is $ billions, SAAR
    elif _last(fy_out) is not None:
        spend_d = _last(fy_out) * 1e6  # FYONET is $ millions

    int_yr = _last(interest)
    int_day = (int_yr * 1e9 / 365.0) if int_yr else None

    def_ttm = None
    if deficit_m is not None and len(deficit_m) >= 12:
        def_ttm = float(deficit_m.tail(12).sum()) * 1e6  # millions
    elif _last(fy_def) is not None:
        def_ttm = _last(fy_def) * 1e6

    return dict(
        debt=debt, debt_asof=debt_asof, added=added, people=people,
        per_cit=per_cit, per_tax=per_tax, ratio=ratio,
        spend=spend_d, int_day=int_day, deficit=def_ttm,
        spend_asof=spend.index[-1] if spend is not None and len(spend) else None,
        int_asof=interest.index[-1] if interest is not None and len(interest) else None,
        def_asof=deficit_m.index[-1] if deficit_m is not None and len(deficit_m) else None,
        d2g=d2g,
    )


def _debt_pills(snap: dict) -> None:
    _sec("Federal debt snapshot", BLUE)
    added = snap.get("added")
    added_col = GREEN if (added is not None and added < 0) else RED
    def_abs = snap.get("deficit")
    if def_abs is not None:
        def_abs = abs(def_abs)
    debt_day = (pd.Timestamp(snap["debt_asof"]).strftime("%b %d, %Y")
                if snap.get("debt_asof") is not None else "—")
    pop_m = (snap["people"] / 1e6) if snap.get("people") else None
    _pills([
        {"label": "Debt per citizen",
         "value": _fmt_usd(snap.get("per_cit")),
         "sub": f"Every American's share · {pop_m:.0f}M people" if pop_m else "Every American's share",
         "color": ACCENT, "icon": "👤"},
        {"label": "Debt per taxpayer",
         "value": _fmt_usd(snap.get("per_tax")),
         "sub": "Share per taxpayer (~140M)",
         "color": ACCENT, "icon": "$"},
        {"label": "Added today (est.)",
         "value": (f"${added / 1e9:+.1f}B" if added is not None and np.isfinite(added) else "—"),
         "sub": f"vs prior day · {debt_day}",
         "color": added_col, "icon": "↗"},
        {"label": "Interest per day",
         "value": _fmt_scale(snap.get("int_day"), digits=1),
         "sub": (_quarter_label(snap["int_asof"]) if snap.get("int_asof") is not None else ""),
         "color": RED, "icon": "🏦"},
        {"label": "Federal spending",
         "value": _fmt_scale(snap.get("spend"), digits=2),
         "sub": (_quarter_label(snap["spend_asof"]) if snap.get("spend_asof") is not None else ""),
         "color": RED, "icon": "📜"},
        {"label": "Annual deficit",
         "value": _fmt_scale(def_abs, digits=2),
         "sub": (f"Last 12 months · {_month_label(snap['def_asof'])}"
                 if snap.get("def_asof") is not None else "Last 12 months"),
         "color": RED, "icon": "↘"},
        {"label": "Debt-to-GDP ratio",
         "value": _fmt_pct(snap.get("ratio")),
         "sub": "Latest print",
         "color": RED, "icon": "📊"},
        {"label": "Total public debt",
         "value": _fmt_scale(snap.get("debt"), digits=2),
         "sub": debt_day,
         "color": BLUE, "icon": "🇺🇸"},
    ])


def render_charts_tab() -> None:
    _css()
    _dollar_since_2020()

    snap = _debt_snapshot()
    _debt_pills(snap)

    _sec("M2", ACCENT)
    m2 = _try_series(M2_SID)
    if m2 is None or m2.empty:
        st.error("Could not load M2.")
    else:
        last = float(m2.iloc[-1])
        tn = last / 1000.0
        yoy = _yoy(m2, 12)
        d1 = _delta(m2, 1)
        _metrics([
            ("Latest", f"${tn:,.2f}T", _month_label(m2.index[-1]), ACCENT),
            ("YoY", _fmt_pct(yoy, signed=True), "12-month change", ACCENT),
            ("Month Δ",
             f"${(d1 or 0) / 1000.0:+,.2f}T" if d1 is not None else "—",
             "vs prior month", ACCENT),
            ("History", f"{len(m2)} months",
             f"{_month_label(m2.index[0])} → {_month_label(m2.index[-1])}", DIM),
        ])
        fig = _line_fig(
            m2, color=ACCENT, y_title="Trillions of dollars",
            y_prefix="$", y_suffix="T", y_mult=0.001,
            hover="%{x|%b %Y}<br>$%{y:.2f}T<extra></extra>",
        )
        with st.container(border=True, key="ch_m2"):
            st.plotly_chart(fig, width="stretch", key="ch_m2_fig")

    _sec("Total public debt / GDP", BLUE)
    debt = snap.get("d2g")
    if debt is None or debt.empty:
        st.error("Could not load debt / GDP.")
        return
    last = float(debt.iloc[-1])
    yoy_pp = _delta(debt, 4)
    d1 = _delta(debt, 1)
    yoy_col = RED if (yoy_pp or 0) >= 0 else GREEN
    d1_col = RED if (d1 or 0) >= 0 else GREEN
    peak = float(debt.max())
    _metrics([
        ("Quarterly print", f"{last:.1f}%", _quarter_label(debt.index[-1]), BLUE),
        ("YoY", f"{yoy_pp:+.1f} pp" if yoy_pp is not None else "—",
         "vs same quarter last year", yoy_col),
        ("Quarter Δ", f"{d1:+.1f} pp" if d1 is not None else "—",
         "vs prior quarter", d1_col),
        ("Peak", f"{peak:.1f}%", _quarter_label(debt.idxmax()), GOLD),
    ])
    fig = _line_fig(
        debt, color=BLUE, y_title="Percent of GDP",
        y_suffix="%", y_mult=1.0,
        hover="%{x|%b %Y}<br>%{y:.1f}% of GDP<extra></extra>",
    )
    with st.container(border=True, key="ch_debt"):
        st.plotly_chart(fig, width="stretch", key="ch_debt_fig")
