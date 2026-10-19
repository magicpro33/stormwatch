"""Charts — official FRED macro series.

M2: Federal Reserve H.6 Money Stock Measures via FRED M2SL
    (seasonally adjusted, monthly, billions of dollars).

Total public debt / GDP: Treasury GFDEBTN ÷ BEA GDP via FRED GFDEGDQ188S
    (quarterly, percent of GDP). This is the standard U.S. "debt-to-GDP"
    chart: federal total public debt, not IMF general-government (which
    also folds in state and local).
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
M2_URL = f"https://fred.stlouisfed.org/series/{M2_SID}"
DEBT_URL = f"https://fred.stlouisfed.org/series/{DEBT_SID}"


def _esc(s) -> str:
    return html_mod.escape("" if s is None else str(s), quote=True)


def _html(markup: str) -> None:
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup.replace("$", "&#36;"), unsafe_allow_html=True)


@st.cache_data(ttl=6 * 3600, show_spinner="Loading FRED series…")
def _load_series(sid: str) -> pd.Series:
    import cascade_engine as ce
    return ce._fred(sid)


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
    t = pd.Timestamp(ts)
    return t.strftime("%b %Y")


def _quarter_label(ts) -> str:
    t = pd.Timestamp(ts)
    return f"Q{(t.month - 1) // 3 + 1} {t.year}"


def _metrics(items: list[tuple[str, str, str, str]]) -> None:
    cells = []
    for label, value, sub, color in items:
        cells.append(
            f"<div class='ch-met' style='--mc:{color}'>"
            f"<div class='ch-mlbl'>{_esc(label)}</div>"
            f"<div class='ch-mval'>{value}</div>"
            f"<div class='ch-msub'>{_esc(sub)}</div></div>"
        )
    _html(f"<div class='ch-metrics'>{''.join(cells)}</div>")


def _line_fig(s: pd.Series, *, color: str, y_title: str, hover: str,
              y_prefix: str = "", y_suffix: str = "", y_mult: float = 1.0):
    import plotly.graph_objects as go

    y = s.astype(float) * y_mult
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values,
        mode="lines",
        line=dict(color=color, width=2.2),
        fill="tozeroy",
        fillcolor="rgba(232,119,34,0.12)" if color == ACCENT else "rgba(127,212,255,0.10)",
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
.ch-metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:8px 0 10px}}
@media(max-width:720px){{.ch-metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.ch-met{{background:{NAVY};border:1px solid {LINE};border-left:4px solid var(--mc,{ACCENT});
  border-radius:10px;padding:10px 14px}}
.ch-mlbl{{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{DIM}}}
.ch-mval{{font-family:ui-monospace,monospace;font-size:24px;font-weight:800;
  color:var(--mc,{INK});margin-top:2px}}
.ch-msub{{font-size:12px;color:{DIM};font-family:ui-monospace,monospace}}
.ch-src{{color:{DIM};font-size:12px;margin:2px 0 14px}}
.ch-src a{{color:{BLUE}}}
.st-key-ch_m2{{border-left:5px solid {ACCENT} !important;border-radius:10px}}
.st-key-ch_debt{{border-left:5px solid {BLUE} !important;border-radius:10px}}
</style>"""
    )


def render_charts_tab() -> None:
    _css()
    st.caption("Official U.S. series from FRED (St. Louis Fed). Same CSV feed "
               "the pressure gauge already uses — no extra API key.")

    # ── M2 ───────────────────────────────────────────────────────────
    _html(
        f'<div class="hub-sec lg" style="--sc:{ACCENT}">M2 Money Stock'
        f"<small> Federal Reserve H.6 · FRED {M2_SID}</small></div>")
    try:
        m2 = _load_series(M2_SID)
    except Exception as e:
        st.error(f"Could not load M2 ({M2_SID}): {e}")
        m2 = None
    if m2 is not None and len(m2):
        last = float(m2.iloc[-1])
        tn = last / 1000.0
        yoy = _yoy(m2, 12)
        d1 = _delta(m2, 1)
        yoy_col = GREEN if (yoy or 0) >= 0 else RED
        d1_col = GREEN if (d1 or 0) >= 0 else RED
        _metrics([
            ("Latest", f"${tn:,.2f}T", _month_label(m2.index[-1]), ACCENT),
            ("YoY", f"{yoy:+.1f}%" if yoy is not None else "—",
             "12-month change", yoy_col),
            ("Month Δ", f"${(d1 or 0) / 1000.0:+,.2f}T" if d1 is not None else "—",
             "vs prior month", d1_col),
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
        _html(
            f"<div class='ch-src'>Source: Board of Governors of the Federal Reserve "
            f"System, M2 <a href='{M2_URL}' target='_blank' rel='noopener'>{M2_SID}</a> "
            f"— seasonally adjusted, monthly, billions of dollars (shown in trillions). "
            f"Retrieved from FRED, Federal Reserve Bank of St. Louis.</div>"
        )

    # ── Debt / GDP ───────────────────────────────────────────────────
    _html(
        f'<div class="hub-sec lg" style="--sc:{BLUE}">Total Public Debt / GDP'
        f"<small> Treasury + BEA · FRED {DEBT_SID}</small></div>")
    try:
        debt = _load_series(DEBT_SID)
    except Exception as e:
        st.error(f"Could not load debt/GDP ({DEBT_SID}): {e}")
        debt = None
    if debt is not None and len(debt):
        last = float(debt.iloc[-1])
        yoy_pp = _delta(debt, 4)
        d1 = _delta(debt, 1)
        yoy_col = RED if (yoy_pp or 0) >= 0 else GREEN
        d1_col = RED if (d1 or 0) >= 0 else GREEN
        peak = float(debt.max())
        _metrics([
            ("Latest", f"{last:.1f}%", _quarter_label(debt.index[-1]), BLUE),
            ("YoY", f"{yoy_pp:+.1f} pp" if yoy_pp is not None else "—",
             "vs same quarter last year", yoy_col),
            ("Quarter Δ", f"{d1:+.1f} pp" if d1 is not None else "—",
             "vs prior quarter", d1_col),
            ("Peak", f"{peak:.1f}%",
             _quarter_label(debt.idxmax()), GOLD),
        ])
        fig = _line_fig(
            debt, color=BLUE, y_title="Percent of GDP",
            y_suffix="%", y_mult=1.0,
            hover="%{x|%b %Y}<br>%{y:.1f}% of GDP<extra></extra>",
        )
        with st.container(border=True, key="ch_debt"):
            st.plotly_chart(fig, width="stretch", key="ch_debt_fig")
        _html(
            f"<div class='ch-src'>Source: U.S. Treasury total public debt "
            f"(GFDEBTN) ÷ BEA nominal GDP, published as "
            f"<a href='{DEBT_URL}' target='_blank' rel='noopener'>{DEBT_SID}</a> "
            f"— quarterly, percent of GDP. Federal debt (not IMF general government, "
            f"which also includes state and local). Retrieved from FRED.</div>"
        )
