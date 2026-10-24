"""Charts — official FRED / Treasury macro series.

M2: Federal Reserve H.6 via FRED M2SL (seasonally adjusted, monthly).
Debt / GDP: Treasury total public debt ÷ BEA GDP via FRED GFDEGDQ188S.
Dollar since 2020: CPI-U purchasing power via FRED CPIAUCSL.
Dollar over 100 years: CPI-U NSA (CPIAUCNS) from 1913.
Debt-clock pills: Treasury Debt to the Penny + FRED population, spending,
interest, and deficit series.
Foreign holders: Treasury TIC SLT Table 5 (major foreign holders).
Gold / silver / coins: Yahoo futures and spot (GC=F, SI=F, BTC/ETH/BNB/SOL/XRP).
"""
from __future__ import annotations

import html as html_mod
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

ACCENT = "#E87722"
GREEN = "#3fbf7f"
RED = "#e05252"
GOLD = "#e8c34a"
SILVER = "#C5D4FF"
BLUE = "#7FD4FF"
DIM = "#9aa8bd"
PURPLE = "#A78BFA"
INK = "#F6F4E9"
NAVY = "#0c1829"
PAPER = "#081325"
LINE = "#3d6a94"

M2_SID = "M2SL"
DEBT_SID = "GFDEGDQ188S"
CPI_SID = "CPIAUCSL"
CPI_LONG_SID = "CPIAUCNS"
METAL_TICKERS = ("GC=F", "SI=F")
CRYPTO_FIVE = (
    ("BTC-USD", "Bitcoin", ACCENT),
    ("ETH-USD", "Ethereum", BLUE),
    ("BNB-USD", "BNB", "#F0A6FF"),
    ("SOL-USD", "Solana", "#2DD4BF"),
    ("XRP-USD", "XRP", "#A78BFA"),
)
ALL_TICKERS = METAL_TICKERS + tuple(t for t, _, _ in CRYPTO_FIVE)
LIVE_EVERY = 30
TIC_SLT5 = ("https://ticdata.treasury.gov/resource-center/"
            "data-chart-center/tic/Documents/slt_table5.txt")
TIC_HIST = ("https://ticdata.treasury.gov/resource-center/"
            "data-chart-center/tic/Documents/mfhhis01.txt")
TIC_SKIP = {
    "all other", "grand total", "of which: foreign official",
    "of which: foreign official treasury bills",
    "of which: foreign official t-bonds & notes",
    "of which: foreign official t-bonds and notes",
}

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


def _hex_rgba(color: str, a: float = 0.12) -> str:
    h = str(color or "").lstrip("#")
    if len(h) != 6:
        return f"rgba(232,119,34,{a})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _futures_session_open(now=None) -> bool:
    """COMEX metals: Sun 6pm ET – Fri 5pm ET, halt 5–6pm ET each day."""
    if now is None:
        try:
            now = pd.Timestamp.now(tz="America/New_York")
        except Exception:
            now = pd.Timestamp.now()
    else:
        now = pd.Timestamp(now)
    if now.tzinfo is None:
        try:
            now = now.tz_localize("America/New_York")
        except Exception:
            pass
    else:
        now = now.tz_convert("America/New_York")
    wd = int(now.weekday())
    hm = now.hour * 60 + now.minute
    halt = (17 * 60) <= hm < (18 * 60)
    if wd == 5:
        return False
    if wd == 6:
        return hm >= 18 * 60
    if wd == 4:
        return hm < 17 * 60
    return not halt


def _naive_index(s: pd.Series) -> pd.Series:
    if s is None or s.empty:
        return pd.Series(dtype=float)
    out = s.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    out.index = idx
    return out.sort_index()


def _stamp_last(s: pd.Series, last) -> pd.Series:
    """Write a live last onto today's bar (or append if the daily tape lags)."""
    if last is None or not np.isfinite(last):
        return s if s is not None else pd.Series(dtype=float)
    last = float(last)
    try:
        et = pd.Timestamp.now(tz="America/New_York")
        today = pd.Timestamp(et.strftime("%Y-%m-%d"))
        now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    except Exception:
        now = pd.Timestamp.now()
        today = now.normalize()
    if s is None or s.empty:
        return pd.Series([last], index=[now], dtype=float)
    out = _naive_index(s)
    last_i = pd.Timestamp(out.index[-1])
    if last_i.normalize() == today:
        out = out.copy()
        out.iloc[-1] = last
        return out
    extra = pd.Series([last], index=[now])
    return pd.concat([out, extra]).sort_index()


@st.cache_data(ttl=1800, show_spinner="Loading gold, silver, and coins…")
def _yf_history(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Daily adjusted close for metals and coins. Cached half an hour."""
    tickers = tuple(tickers)
    if not tickers:
        return pd.DataFrame()

    def _pull():
        import yfinance as yf
        return yf.download(
            list(tickers), period="max", auto_adjust=True,
            progress=False, threads=False, group_by="ticker",
        )

    try:
        from mw_yf import yf_call
        df = yf_call(_pull)
    except Exception:
        df = _pull()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=20, show_spinner=False)
def _yf_last(tickers: tuple[str, ...]) -> dict:
    """Latest print only — 1-minute bar, ~20s cache. Empty dict on failure."""
    tickers = tuple(tickers)
    if not tickers:
        return {}

    def _pull():
        import yfinance as yf
        return yf.download(
            list(tickers), period="1d", interval="1m",
            auto_adjust=True, progress=False, threads=False,
            group_by="ticker",
        )

    try:
        from mw_yf import yf_call
        df = yf_call(_pull)
    except Exception:
        df = pd.DataFrame()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    out = {}
    for t in tickers:
        s = _yf_close(df, t)
        if s is None or s.empty:
            continue
        v = float(s.iloc[-1])
        if np.isfinite(v):
            out[t] = v
    return out


def _pull_history() -> pd.DataFrame:
    try:
        return _yf_history(ALL_TICKERS)
    except Exception:
        return pd.DataFrame()


def _pull_last(tickers: tuple[str, ...]) -> dict:
    try:
        return _yf_last(tuple(tickers)) or {}
    except Exception:
        return {}


@st.cache_data(ttl=45, show_spinner=False)
def _yf_minutes(tickers: tuple[str, ...], period: str = "2d") -> pd.DataFrame:
    """1-minute closes for live last + 1h / 24h crypto moves."""
    tickers = tuple(tickers)
    if not tickers:
        return pd.DataFrame()

    def _pull():
        import yfinance as yf
        return yf.download(
            list(tickers), period=period, interval="1m", auto_adjust=True,
            progress=False, threads=False, group_by="ticker",
        )

    try:
        from mw_yf import yf_call
        df = yf_call(_pull)
    except Exception:
        df = _pull()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("no minute crypto bars")
    return df


def _pull_minutes(tickers: tuple[str, ...]) -> pd.DataFrame:
    try:
        return _yf_minutes(tuple(tickers), "2d")
    except Exception:
        return pd.DataFrame()


def _yf_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = None
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if ticker in lvl0:
            sub = df[ticker]
            s = sub["Close"] if isinstance(sub, pd.DataFrame) and "Close" in sub.columns else sub
        elif "Close" in lvl0 and ticker in df["Close"].columns:
            s = df["Close"][ticker]
    elif ticker in df.columns:
        s = df[ticker]
    if s is None:
        return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    out = pd.to_numeric(s, errors="coerce").dropna()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _pct_since(s: pd.Series, **delta) -> float | None:
    """Percent change from the last print at or before now − delta."""
    s = _naive_index(s)
    if s is None or s.empty:
        return None
    end = pd.Timestamp(s.index[-1])
    hit = s[s.index <= end - pd.Timedelta(**delta)]
    if hit.empty:
        return None
    b = float(hit.iloc[-1])
    a = float(s.iloc[-1])
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return None
    return (a / b - 1.0) * 100.0


def _pct_ago(s: pd.Series, days: int) -> float | None:
    s = _naive_index(s)
    if s is None or s.empty:
        return None
    pct = _pct_since(s, days=int(days))
    if pct is not None:
        return pct
    if days <= 2 and len(s) >= 2:
        b = float(s.iloc[-2])
        a = float(s.iloc[-1])
        if np.isfinite(a) and np.isfinite(b) and b != 0:
            return (a / b - 1.0) * 100.0
    return None


def _fmt_px(n, digits=None) -> str:
    if n is None or not np.isfinite(n):
        return "—"
    n = float(n)
    if digits is None:
        if abs(n) >= 1000:
            digits = 0
        elif abs(n) >= 100:
            digits = 2
        elif abs(n) >= 1:
            digits = 2
        else:
            digits = 4
    return f"${n:,.{digits}f}"


def _spark_svg(s: pd.Series, color: str, w: int = 220, h: int = 44) -> str:
    """Tiny price path for a snapshot card. Last ~90 prints as CSS bars.

    Streamlit's st.html sanitizer strips <svg>, so this stays in div/span.
    """
    if s is None or len(s.dropna()) < 2:
        return ""
    ys = pd.to_numeric(s, errors="coerce").dropna().values[-90:]
    if len(ys) < 2:
        return ""
    n = 48
    if len(ys) > n:
        idx = np.linspace(0, len(ys) - 1, n).astype(int)
        ys = ys[idx]
    mn, mx = float(ys.min()), float(ys.max())
    span = mx - mn or 1.0
    fill = _hex_rgba(color, 0.88)
    bars = []
    for v in ys:
        pct = 10.0 + (float(v) - mn) / span * 90.0
        bars.append(f"<span style='height:{pct:.1f}%;background:{fill}'></span>")
    return f"<div class='ch-spark' aria-hidden='true'>{''.join(bars)}</div>"


@st.cache_data(ttl=3600, show_spinner="Loading Treasury debt…")
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


def _tic_name(raw: str) -> str:
    name = " ".join(str(raw or "").replace(",", " ").split())
    if name.lower().startswith("china"):
        return "China"
    if name.lower() in ("korea south", "korea, south"):
        return "South Korea"
    return name


@st.cache_data(ttl=6 * 3600, show_spinner="Loading foreign holders…")
def _tic_foreign_holders() -> dict:
    """Treasury TIC SLT Table 5 — major foreign holders, billions of dollars."""
    import requests
    r = requests.get(TIC_SLT5, timeout=20, headers={"User-Agent": "MoneyWeather/nerd-hut"})
    r.raise_for_status()
    lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
    header_i = next((i for i, ln in enumerate(lines) if ln.lower().startswith("country")), None)
    if header_i is None:
        return {}
    countries = []
    total = official = None
    asof = None
    header = lines[header_i].split()
    if len(header) >= 2:
        asof = header[1]
    for ln in lines[header_i + 1:]:
        low = ln.lower()
        if low.startswith("notes") or low.startswith("the data") or low.startswith("link:"):
            break
        tokens = ln.replace(",", " ").split()
        nums, name_parts = [], []
        for tok in tokens:
            try:
                nums.append(float(tok))
            except ValueError:
                name_parts.append(tok)
        if not nums or not name_parts:
            continue
        name = " ".join(name_parts)
        val = nums[0]
        key = name.lower()
        if key == "grand total":
            total = val
            continue
        if key.startswith("of which: foreign official") and "bill" not in key and "bond" not in key:
            official = val
            continue
        if key in TIC_SKIP or key.startswith("of which"):
            continue
        countries.append((_tic_name(name), val))
    countries.sort(key=lambda kv: kv[1], reverse=True)
    return {"asof": asof, "total": total, "official": official, "countries": countries}


def _tic_december(text: str) -> dict:
    """Year-end (December) holdings in billions from a TIC monthly table."""
    out = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        if not lines[i].lower().startswith("country"):
            i += 1
            continue
        years = []
        for tok in lines[i].split()[1:]:
            try:
                years.append(int(tok))
            except ValueError:
                years.append(None)
        i += 1
        if i < len(lines) and set(lines[i]) <= set("- "):
            i += 1
        block = {}
        while i < len(lines):
            ln = lines[i]
            low = ln.lower()
            if low.startswith("country"):
                break
            if low.startswith("notes") or low.startswith("the data") or low.startswith("of which"):
                i += 1
                continue
            tokens = ln.replace('"', " ").replace(",", " ").split()
            nums, name_parts = [], []
            for tok in tokens:
                try:
                    nums.append(float(tok))
                except ValueError:
                    name_parts.append(tok)
            if nums and name_parts and years:
                block[" ".join(name_parts)] = nums
            i += 1
        if not years:
            continue
        dec_i = 0
        year = years[dec_i]
        if year is None:
            continue
        slot = out.setdefault(year, {})
        for raw, nums in block.items():
            key = raw.lower()
            val = nums[dec_i] if dec_i < len(nums) else None
            if val is None:
                continue
            if key == "grand total":
                slot["total"] = val
            elif key.startswith("japan"):
                slot["japan"] = val
            elif key.startswith("china"):
                slot["china"] = val
        if slot:
            slot["asof"] = f"{year}-12"
    return out


@st.cache_data(ttl=6 * 3600, show_spinner="Loading foreign-holder history…")
def _tic_year_ends() -> dict:
    """December TIC snapshots: {year: {total, japan, china, asof}} in $ billions."""
    import requests
    r = requests.get(TIC_HIST, timeout=20, headers={"User-Agent": "MoneyWeather/nerd-hut"})
    r.raise_for_status()
    return _tic_december(r.text)


_NATION_COLOR = {
    "Japan": ACCENT,
    "United Kingdom": BLUE,
    "China": RED,
    "Belgium": GOLD,
    "Canada": GREEN,
    "Cayman Islands": "#F0A6FF",
    "Luxembourg": "#2DD4BF",
    "France": "#FF8FA3",
    "Ireland": "#C8F542",
    "Taiwan": "#A78BFA",
    "Switzerland": "#C5D4FF",
    "Singapore": "#FFB020",
    "Hong Kong": "#6EE7B7",
    "India": "#9B7FE0",
    "Norway": "#4aa3df",
}
_NATION_FALLBACK = ("#7FD4FF", "#FF9B6A", "#6EE7B7", "#F0A6FF", "#E8C34A",
                    "#2DD4BF", "#FF8FA3", "#C8F542")


def _live_project(snap: dict) -> dict:
    """Walk the last official debt print forward using the last daily change."""
    out = dict(snap)
    debt = snap.get("debt")
    added = snap.get("added")
    asof = snap.get("debt_asof")
    if debt is None or added is None or asof is None:
        return out
    if not np.isfinite(debt) or not np.isfinite(added):
        return out
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = pd.Timestamp(asof).tz_localize(None)
    days = max(0.0, (now - start).total_seconds() / 86400.0)
    live = debt + added * days
    out["debt"] = live
    people = snap.get("people")
    if people:
        out["per_cit"] = live / people
    out["per_tax"] = live / TAXPAYERS
    return out


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
        extra = it.get("extra") or ""
        cells.append(
            f"<div class='ch-pill{feat}' style='--mc:{color}'>"
            f"{ico_html}"
            f"<div class='ch-mlbl'>{_esc(it.get('label', ''))}</div>"
            f"<div class='ch-mval'>{it.get('value', '—')}</div>"
            f"<div class='ch-msub'>{_esc(it.get('sub', ''))}</div>"
            f"{extra}</div>"
        )
    _html(f"<div class='{grid}'>{''.join(cells)}</div>")


def _metrics(items: list[tuple[str, str, str, str]]) -> None:
    _pills([{"label": a, "value": b, "sub": c, "color": d}
            for a, b, c, d in items])


def _plotly(fig, key: str) -> None:
    """Keep Plotly hover labels — Streamlit's default theme can hide them."""
    kw = dict(
        width="stretch", key=key,
        config={"displayModeBar": False, "responsive": True},
    )
    try:
        st.plotly_chart(fig, theme=None, **kw)
    except TypeError:
        st.plotly_chart(fig, **kw)


def _hoverlabel(color: str) -> dict:
    return dict(
        bgcolor="#0c1829",
        bordercolor=color,
        font=dict(color=INK, family="JetBrains Mono, monospace", size=12),
    )


def _line_fig(s: pd.Series, *, color: str, y_title: str, hover: str,
              y_prefix: str = "", y_suffix: str = "", y_mult: float = 1.0,
              height: int = 420, buttons=None, slider: bool = True,
              fill: bool = True):
    import plotly.graph_objects as go

    y = s.astype(float) * y_mult
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y.index, y=y.values,
        mode="lines",
        line=dict(color=color, width=2.2),
        fill="tozeroy" if fill else None,
        fillcolor=_hex_rgba(color, 0.14) if fill else None,
        hovertemplate=hover,
        name=y_title,
    ))
    if buttons is None:
        buttons = [
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(count=10, label="10Y", step="year", stepmode="backward"),
            dict(count=20, label="20Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=NAVY,
        font=dict(color=INK, family="JetBrains Mono, monospace", size=12),
        showlegend=False,
        margin=dict(l=12, r=12, t=28, b=8),
        hovermode="x unified",
        hoverlabel=_hoverlabel(color),
        xaxis=dict(
            gridcolor="#1e3a5f",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor=DIM,
            spikethickness=1,
            spikedash="dot",
            rangeselector=dict(
                buttons=buttons,
                bgcolor="#122a42",
                activecolor=color,
                font=dict(color=INK, size=11),
                x=0, y=1.08, xanchor="left",
            ),
            rangeslider=dict(visible=slider, thickness=0.08,
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
    )
    return fig


def _css() -> None:
    _html(
        f"""<style>
.ch-pills{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:8px 0 12px}}
.ch-hero{{display:grid;grid-template-columns:1.5fr 1fr 1fr;gap:10px;margin:8px 0 18px}}
.ch-hero-2{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0 12px}}
.ch-pills-5{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:8px 0 12px}}
.ch-spark{{display:flex;align-items:flex-end;gap:1.5px;height:46px;margin-top:10px;
  padding:6px 4px 0;border-radius:8px;background:rgba(6,16,28,.55)}}
.ch-spark span{{flex:1 1 0;min-width:0;border-radius:2px 2px 0 0;display:block}}
@media(max-width:1100px){{
  .ch-pills-5{{grid-template-columns:repeat(3,minmax(0,1fr))}}
}}
@media(max-width:900px){{
  .ch-pills{{grid-template-columns:repeat(2,minmax(0,1fr))}}
  .ch-hero,.ch-hero-2,.ch-pills-5{{grid-template-columns:1fr}}
}}
.ch-pill{{background:#06101c;border:1px solid #2a4a6a;border-radius:14px;
  padding:14px 16px 12px;position:relative;min-height:92px}}
.ch-pill:has(.ch-spark){{min-height:138px}}
.ch-pill.feature{{border-left:5px solid var(--mc,{RED});padding:16px 18px;min-height:148px}}
.ch-pill.feature:has(.ch-spark){{min-height:196px}}
.ch-ico{{position:absolute;top:12px;right:14px;font-size:16px;opacity:.55}}
.ch-mlbl{{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:{DIM};padding-right:22px}}
.ch-mval{{font-family:ui-monospace,JetBrains Mono,monospace;font-size:26px;font-weight:800;
  color:var(--mc,{INK});margin-top:6px;line-height:1.1;letter-spacing:0;
  font-variant-numeric:tabular-nums}}
.ch-pill.feature .ch-mval{{font-size:38px}}
.ch-msub{{font-size:12px;color:{DIM};margin-top:4px;line-height:1.35}}
.st-key-ch_metal{{border-left:5px solid {GOLD} !important;border-radius:10px;
  border-color:color-mix(in srgb,{GOLD} 45%,#3d6a94) !important}}
.st-key-ch_debt_snap{{border-left:5px solid {BLUE} !important;border-radius:10px;
  border-color:color-mix(in srgb,{BLUE} 45%,#3d6a94) !important}}
.st-key-ch_dollar{{border-left:5px solid {RED} !important;border-radius:10px;
  border-color:color-mix(in srgb,{RED} 45%,#3d6a94) !important}}
.st-key-ch_dollar100{{border-left:5px solid {RED} !important;border-radius:10px;
  border-color:color-mix(in srgb,{RED} 45%,#3d6a94) !important}}
.st-key-ch_crypto{{border-left:5px solid {PURPLE} !important;border-radius:10px;
  border-color:color-mix(in srgb,{PURPLE} 45%,#3d6a94) !important}}
.st-key-ch_m2{{border-left:5px solid {ACCENT} !important;border-radius:10px;
  border-color:color-mix(in srgb,{ACCENT} 45%,#3d6a94) !important}}
.st-key-ch_d2g,.st-key-ch_debt{{border-left:5px solid {BLUE} !important;border-radius:10px;
  border-color:color-mix(in srgb,{BLUE} 45%,#3d6a94) !important}}
.st-key-ch_foreign{{border-left:5px solid {GOLD} !important;border-radius:10px;
  border-color:color-mix(in srgb,{GOLD} 45%,#3d6a94) !important}}
.ch-trend-h{{text-align:center;font-size:12px;font-weight:800;letter-spacing:.12em;
  text-transform:uppercase;color:{DIM};margin:16px 0 8px}}
.ch-trend-note{{text-align:center;color:{RED};font-size:13px;line-height:1.5;
  margin:10px 4px 4px}}
.ch-coin-wrap{{overflow-x:auto;margin:8px 0 14px;border:1px solid #2a4a6a;
  border-radius:14px;background:#06101c}}
.ch-coin-tbl{{min-width:640px}}
.ch-coin-head,.ch-coin-row{{display:grid;
  grid-template-columns:44px minmax(140px,1.6fr) minmax(110px,1fr) 92px 92px 92px;
  gap:8px;align-items:center;padding:12px 16px}}
.ch-coin-head{{color:{DIM};font-size:11px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;border-bottom:1px solid #1d2b40}}
.ch-coin-head > :nth-child(n+3),.ch-coin-px,.ch-coin-chg{{text-align:right}}
.ch-coin-row{{border-bottom:1px solid #1d2b40}}
.ch-coin-row:last-child{{border-bottom:0}}
.ch-coin-num{{color:{DIM};font-variant-numeric:tabular-nums;font-weight:700}}
.ch-coin-name{{font-weight:800;font-size:15px;letter-spacing:.01em}}
.ch-coin-sym{{color:{DIM};font-size:12px;font-weight:700;letter-spacing:.08em;
  margin-left:8px}}
.ch-coin-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;
  margin-right:8px;vertical-align:middle}}
.ch-coin-px{{font-family:ui-monospace,JetBrains Mono,monospace;font-weight:800;
  font-variant-numeric:tabular-nums;font-size:15px;color:{INK}}}
.ch-coin-chg{{font-family:ui-monospace,JetBrains Mono,monospace;font-weight:700;
  font-variant-numeric:tabular-nums;font-size:13px}}
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


def _chg_color(pct) -> str:
    if pct is None:
        return DIM
    return GREEN if pct >= 0 else RED


def _spot_when(live: bool) -> str:
    now = pd.Timestamp.now()
    if live:
        return now.strftime("%b %d, %Y · %H:%M")
    return now.strftime("%b %d, %Y")


def _metal_pills(gold: pd.Series, silver: pd.Series, live: bool) -> None:
    """Gold and silver clock — same snapshot language as federal debt."""
    g, s = _last(gold), _last(silver)
    if g is None and s is None:
        st.error("Could not load gold or silver.")
        return
    g_day, s_day = _pct_ago(gold, 1), _pct_ago(silver, 1)
    g_yr, s_yr = _pct_ago(gold, 365), _pct_ago(silver, 365)
    g_hi = float(gold.max()) if len(gold) else None
    s_hi = float(silver.max()) if len(silver) else None
    g_off = ((g / g_hi - 1.0) * 100.0) if (g and g_hi) else None
    s_off = ((s / s_hi - 1.0) * 100.0) if (s and s_hi) else None
    when = _spot_when(live)
    _pills([
        {"label": "Gold",
         "value": _fmt_px(g, 0),
         "sub": f"Dollar per ounce · {when}",
         "color": GOLD, "feature": True, "icon": "Au",
         "extra": _spark_svg(gold, GOLD)},
        {"label": "Silver",
         "value": _fmt_px(s, 2),
         "sub": f"Dollar per ounce · {when}",
         "color": SILVER, "feature": True, "icon": "Ag",
         "extra": _spark_svg(silver, SILVER)},
    ], grid="ch-hero-2")
    _pills([
        {"label": "Gold today",
         "value": _fmt_pct(g_day, signed=True),
         "sub": "vs prior session",
         "color": _chg_color(g_day), "icon": "↗" if (g_day or 0) >= 0 else "↘"},
        {"label": "Silver today",
         "value": _fmt_pct(s_day, signed=True),
         "sub": "vs prior session",
         "color": _chg_color(s_day), "icon": "↗" if (s_day or 0) >= 0 else "↘"},
        {"label": "Gold, 1 year",
         "value": _fmt_pct(g_yr, signed=True),
         "sub": "same week last year",
         "color": _chg_color(g_yr)},
        {"label": "Silver, 1 year",
         "value": _fmt_pct(s_yr, signed=True),
         "sub": "same week last year",
         "color": _chg_color(s_yr)},
        {"label": "Gold vs high",
         "value": _fmt_pct(g_off, signed=True),
         "sub": (_fmt_px(g_hi, 0) + " · " + _month_label(gold.idxmax())
                 if g_hi is not None else ""),
         "color": _chg_color(g_off)},
        {"label": "Silver vs high",
         "value": _fmt_pct(s_off, signed=True),
         "sub": (_fmt_px(s_hi, 2) + " · " + _month_label(silver.idxmax())
                 if s_hi is not None else ""),
         "color": _chg_color(s_off)},
        {"label": "Gold high",
         "value": _fmt_px(g_hi, 0) if g_hi is not None else "—",
         "sub": (_month_label(gold.idxmax()) if len(gold) else ""),
         "color": GOLD},
        {"label": "Silver high",
         "value": _fmt_px(s_hi, 2) if s_hi is not None else "—",
         "sub": (_month_label(silver.idxmax()) if len(silver) else ""),
         "color": SILVER},
    ])


def _metal_charts(gold: pd.Series, silver: pd.Series) -> None:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        if gold is not None and len(gold):
            fig = _line_fig(
                gold, color=GOLD, y_title="Gold, $ / oz",
                y_prefix="$", hover="%{x|%b %d, %Y}<br>$%{y:,.0f} / oz<extra></extra>",
                height=340, slider=False,
            )
            _plotly(fig, "ch_gold_fig")
    with c2:
        if silver is not None and len(silver):
            fig = _line_fig(
                silver, color=SILVER, y_title="Silver, $ / oz",
                y_prefix="$", hover="%{x|%b %d, %Y}<br>$%{y:,.2f} / oz<extra></extra>",
                height=340, slider=False,
            )
            _plotly(fig, "ch_silver_fig")


def _dollar_century() -> None:
    """Purchasing power of $1 over the last hundred years."""
    _sec("A century of the dollar", RED)
    cpi = _try_series(CPI_LONG_SID)
    if cpi is None or cpi.empty:
        st.error("Could not load a century of prices.")
        return
    now = pd.Timestamp.now().tz_localize(None)
    start = now - pd.DateOffset(years=100)
    cpi = cpi[cpi.index >= start]
    if cpi.empty:
        st.error("Could not load a century of prices.")
        return
    base = float(cpi.iloc[0])
    last = float(cpi.iloc[-1])
    if not np.isfinite(base) or not np.isfinite(last) or last <= 0 or base <= 0:
        st.error("Could not load a century of prices.")
        return
    power = base / cpi
    cents = 100.0 * float(power.iloc[-1])
    takes = last / base
    infl = (last / base - 1.0) * 100.0
    asof = _month_label(cpi.index[-1])
    began = _month_label(cpi.index[0])
    _pills([
        {"label": f"$1 from {cpi.index[0].year} buys",
         "value": f"{cents:.0f}¢",
         "sub": f"of the same goods in {asof}. The rest was inflation.",
         "color": RED, "feature": True, "icon": "¢"},
        {"label": "It takes today",
         "value": f"${takes:,.0f}",
         "sub": f"to match one {cpi.index[0].year} dollar",
         "color": ACCENT},
        {"label": "Prices since then",
         "value": _fmt_pct(infl, signed=True, digits=0),
         "sub": f"{began} → {asof}",
         "color": GOLD},
    ], grid="ch-hero")
    fig = _line_fig(
        power, color=RED, y_title="Purchasing power of $1",
        y_prefix="$",
        hover="%{x|%b %Y}<br>$%{y:.3f} of that first dollar<extra></extra>",
        height=420, slider=True,
        buttons=[
            dict(count=25, label="25Y", step="year", stepmode="backward"),
            dict(count=50, label="50Y", step="year", stepmode="backward"),
            dict(step="all", label="100Y"),
        ],
    )
    _plotly(fig, "ch_dollar100_fig")
    _html(
        f"<div class='ch-trend-note'>"
        f"A dollar parked in {cpi.index[0].year} still spends like "
        f"{cents:.0f} cents today. The line is that dollar's buying power, "
        f"not the size of a paycheck."
        f"</div>"
    )


def _crypto_series(px: pd.DataFrame, last: dict | None = None) -> list:
    rows = []
    for ticker, name, color in CRYPTO_FIVE:
        s = _stamp_last(_yf_close(px, ticker), (last or {}).get(ticker))
        rows.append((ticker, name, color, s))
    return rows


def _chg_cell(pct) -> str:
    if pct is None or not np.isfinite(pct):
        return f"<span class='ch-coin-chg' style='color:{DIM}'>—</span>"
    col = GREEN if pct >= 0 else RED
    arrow = "▲" if pct >= 0 else "▼"
    return (f"<span class='ch-coin-chg' style='color:{col}'>"
            f"{arrow}{abs(pct):.2f}%</span>")


def _crypto_pills(rows: list, minutes: pd.DataFrame | None = None) -> None:
    """Top 5 crypto coins — live price plus 1h / 24h / 7d, no stables."""
    body = []
    alive = False
    for i, (ticker, name, color, daily) in enumerate(rows, start=1):
        last = _last(daily)
        if last is not None:
            alive = True
        tape = _stamp_last(_yf_close(minutes, ticker), last)
        h1 = _pct_since(tape, hours=1)
        h24 = _pct_since(tape, hours=24)
        d7 = _pct_ago(daily, 7)
        if h24 is None:
            h24 = _pct_ago(daily, 1)
        sym = ticker.replace("-USD", "")
        px_digits = 2 if (last is None or abs(last) >= 1) else 4
        body.append(
            f"<div class='ch-coin-row'>"
            f"<div class='ch-coin-num'>{i}</div>"
            f"<div><span class='ch-coin-dot' style='background:{color}'></span>"
            f"<span class='ch-coin-name' style='color:{color}'>{_esc(name)}</span>"
            f"<span class='ch-coin-sym'> {_esc(sym)}</span></div>"
            f"<div class='ch-coin-px'>{_fmt_px(last, px_digits)}</div>"
            f"{_chg_cell(h1)}{_chg_cell(h24)}{_chg_cell(d7)}"
            f"</div>"
        )
    if not alive:
        st.error("Could not load coin prices.")
        return
    _html(
        "<div class='ch-coin-wrap'><div class='ch-coin-tbl'>"
        "<div class='ch-coin-head'>"
        "<div>#</div><div>Name</div><div>Price</div>"
        "<div>1h %</div><div>24h %</div><div>7d %</div>"
        "</div>"
        f"{''.join(body)}</div></div>"
    )


def _crypto_chart(px: pd.DataFrame) -> None:
    import plotly.graph_objects as go

    now = pd.Timestamp.now().tz_localize(None)
    cut = now - pd.DateOffset(years=5)
    fig = go.Figure()
    any_trace = False
    x0 = x1 = None
    for ticker, name, color in CRYPTO_FIVE:
        s = _naive_index(_yf_close(px, ticker)).dropna()
        if len(s) < 2:
            continue
        any_trace = True
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines",
            line=dict(color=color, width=2.2),
            name=name,
            hovertemplate=name + "<br>%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
        ))
        first, last = s.index[0], s.index[-1]
        x0 = first if x0 is None or first < x0 else x0
        x1 = last if x1 is None or last > x1 else x1
    if not any_trace:
        return
    view0 = cut if (x0 is None or cut > x0) else x0
    fig.update_layout(
        height=460,
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=NAVY,
        font=dict(color=INK, family="JetBrains Mono, monospace", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.08,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=12, r=12, t=72, b=8),
        hovermode="x unified",
        hoverlabel=_hoverlabel(PURPLE),
        xaxis=dict(
            gridcolor="#1e3a5f", zeroline=False,
            range=[view0, x1 or now],
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor=DIM, spikethickness=1, spikedash="dot",
            rangeselector=dict(
                buttons=[
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#122a42",
                activecolor=PURPLE,
                font=dict(color=INK, size=11),
                x=1, y=1.24, xanchor="right",
            ),
            rangeslider=dict(visible=True, thickness=0.10,
                             bgcolor="#0a1624", bordercolor=LINE),
        ),
        yaxis=dict(title="Price", type="log", tickprefix="$",
                   gridcolor="#1e3a5f", zeroline=False),
    )
    _plotly(fig, "ch_crypto_fig")
    _html(
        "<div class='ch-trend-h'>Price over five years</div>"
        "<div class='ch-trend-note' style='color:#9aa8bd'>"
        "Log scale so a one-dollar coin and an eighty-thousand-dollar coin "
        "can share the same tape. Drag the bar under the tape, or the 3M–All "
        "buttons, to change the window. Hover a date for the dollar price."
        "</div>"
    )


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


def _nation_color(name: str, i: int = 0) -> str:
    return _NATION_COLOR.get(name) or _NATION_FALLBACK[i % len(_NATION_FALLBACK)]


def _holder_bars(rows: list[tuple[str, float]]) -> None:
    import plotly.graph_objects as go
    names = [n for n, _ in rows][::-1]
    vals = [v / 1000.0 for _, v in rows][::-1]
    colors = [_nation_color(n, i) for i, n in enumerate(names)]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>$%{x:.2f}T<extra></extra>",
    ))
    fig.update_layout(
        height=max(280, 28 * len(rows) + 48),
        template="plotly_dark",
        paper_bgcolor=PAPER,
        plot_bgcolor=NAVY,
        font=dict(color=INK, family="JetBrains Mono, monospace", size=12),
        showlegend=False,
        margin=dict(l=8, r=12, t=8, b=8),
        xaxis=dict(title="Trillions of dollars", gridcolor="#1e3a5f",
                   zeroline=False, ticksuffix="T", tickprefix="$"),
        yaxis=dict(gridcolor="#1e3a5f", zeroline=False),
    )
    _plotly(fig, "ch_foreign_fig")


def _foreign_holders(snap: dict) -> None:
    _sec("Foreign holders of U.S. debt", GOLD)
    tic = {}
    try:
        tic = _tic_foreign_holders() or {}
    except Exception:
        tic = {}
    countries = list(tic.get("countries") or [])
    total_b = tic.get("total")
    total = (total_b * 1e9) if total_b is not None else None
    debt = snap.get("debt")
    share = (total / debt * 100.0) if (total and debt) else None
    by_name = {n.lower(): v for n, v in countries}

    def _c(name: str):
        v = by_name.get(name.lower())
        return None if v is None else v * 1e9

    japan, uk, china = _c("Japan"), _c("United Kingdom"), _c("China")
    asof = tic.get("asof") or ""
    month = ""
    if asof:
        try:
            month = pd.Timestamp(asof).strftime("%b %Y")
        except Exception:
            month = str(asof)
    lead = "Japan leads" if japan else "Top holders"
    follow = []
    if uk:
        follow.append("UK")
    if china:
        follow.append("China")
    trail = ""
    if follow:
        trail = f", followed by {' and '.join(follow)}" if len(follow) <= 2 \
            else f", followed by {', '.join(follow[:-1])}, and {follow[-1]}"
    sub = f"{lead}{trail}."
    if month:
        sub = f"{sub} {month}."
    _pills([
        {"label": "Foreign holders",
         "value": _fmt_scale(total, digits=1),
         "sub": (f"{_fmt_pct(share)} of U.S. debt. {sub}"
                 if share is not None else sub),
         "color": GOLD, "feature": True, "icon": "🌍"},
        {"label": "Japan",
         "value": _fmt_scale(japan, digits=1),
         "sub": "Largest foreign holder",
         "color": ACCENT, "icon": "🇯🇵"},
        {"label": "United Kingdom",
         "value": _fmt_scale(uk, digits=1),
         "sub": "Second",
         "color": BLUE, "icon": "🇬🇧"},
        {"label": "China",
         "value": _fmt_scale(china, digits=1),
         "sub": "Third",
         "color": RED, "icon": "🇨🇳"},
    ])
    top = countries[:10]
    if top:
        _holder_bars(top)
    elif total is None:
        st.error("Could not load foreign holders.")

    hist = {}
    try:
        hist = _tic_year_ends() or {}
    except Exception:
        hist = {}
    gfde = _try_series("GFDEBTN")
    years = [y for y in (2020, 2022, 2024, 2025) if y in hist and hist[y].get("total")]
    if 2025 not in years:
        latest_y = max((y for y in hist if hist[y].get("total")), default=None)
        if latest_y and latest_y not in years:
            years.append(latest_y)
            years.sort()
    trend = []
    share_2020 = share_now = None
    for y in years:
        tot_b = hist[y].get("total")
        if tot_b is None:
            continue
        tot = tot_b * 1e9
        debt_m = _at_or_before(gfde, f"{y}-12-31")
        pct = (tot / (debt_m * 1e6) * 100.0) if debt_m else None
        if y == 2020:
            share_2020 = pct
        share_now = pct
        trend.append({
            "label": str(y),
            "value": _fmt_pct(pct),
            "sub": _fmt_scale(tot, digits=1),
            "color": BLUE,
        })
    if trend:
        _html('<div class="ch-trend-h">Foreign ownership trend</div>')
        _pills(trend)

    jap_b = (japan / 1e9) if japan else None
    chi_b = (china / 1e9) if china else None
    if jap_b is None and hist:
        last = hist.get(max(hist)) or {}
        jap_b, chi_b = last.get("japan"), last.get("china")
    gap = None
    if jap_b is not None and chi_b is not None:
        gap = (jap_b - chi_b) * 1e9
    china_until = None
    for y in sorted(hist):
        j, ch = hist[y].get("japan"), hist[y].get("china")
        if j is None or ch is None:
            continue
        if ch > j:
            china_until = y
    if share is not None:
        share_now = share
    bits = []
    if china_until is not None:
        bits.append(f"China was the #1 foreign holder until {china_until + 1}.")
    if gap is not None and gap > 0:
        bits.append(f"Japan now holds {_fmt_scale(gap, digits=0)} more than China.")
    if share_2020 is not None and share_now is not None:
        bits.append(
            f"Foreign ownership has dropped from {_fmt_pct(share_2020)} "
            f"to {_fmt_pct(share_now)} since 2020."
        )
    if bits:
        _html(f'<div class="ch-trend-note">{_esc(" ".join(bits))}</div>')


def _live_deco(seconds: int):
    """Fragment that refreshes in place. Falls back to a no-op on old Streamlit."""
    if not hasattr(st, "fragment"):
        return lambda fn: fn
    return st.fragment(run_every=timedelta(seconds=int(seconds)))


@_live_deco(LIVE_EVERY)
def _metal_live() -> None:
    """Spot pills. Yahoo last only while COMEX is in session."""
    hist = _pull_history()
    live = _futures_session_open()
    last = _pull_last(METAL_TICKERS) if live else {}
    gold = _stamp_last(_yf_close(hist, "GC=F"), last.get("GC=F"))
    silver = _stamp_last(_yf_close(hist, "SI=F"), last.get("SI=F"))
    _metal_pills(gold, silver, live=bool(last))


@_live_deco(LIVE_EVERY)
def _crypto_live() -> None:
    """Coin tape. Crypto prints 24/7 so this always hits a short-lived last."""
    hist = _pull_history()
    tickers = tuple(t for t, _, _ in CRYPTO_FIVE)
    minutes = _pull_minutes(tickers)
    last = {}
    for t in tickers:
        v = _last(_yf_close(minutes, t))
        if v is not None:
            last[t] = v
    if not last:
        last = _pull_last(tickers)
    _crypto_pills(_crypto_series(hist, last), minutes)


@_live_deco(LIVE_EVERY)
def _debt_live() -> None:
    """Debt clock. Walks the last daily print forward; no extra HTTP."""
    with st.container(border=True, key="ch_debt_snap"):
        _debt_pills(_live_project(_debt_snapshot()))


def _render_nerd_body() -> None:
    tape = _pull_history()

    with st.container(border=True, key="ch_metal"):
        _sec("Gold and silver snapshot", GOLD)
        _metal_live()
        _metal_charts(_yf_close(tape, "GC=F"), _yf_close(tape, "SI=F"))

    with st.container(border=True, key="ch_crypto"):
        _sec("Top 5 Crypto Coins", PURPLE)
        _crypto_live()
        _crypto_chart(tape)

    _debt_live()

    with st.container(border=True, key="ch_dollar"):
        _dollar_since_2020()

    with st.container(border=True, key="ch_dollar100"):
        _dollar_century()

    with st.container(border=True, key="ch_m2"):
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
            _plotly(fig, "ch_m2_fig")

    snap = _debt_snapshot()
    with st.container(border=True, key="ch_d2g"):
        _sec("Total public debt / GDP", BLUE)
        debt = snap.get("d2g")
        if debt is None or debt.empty:
            st.error("Could not load debt / GDP.")
        else:
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
            _plotly(fig, "ch_debt_fig")

    with st.container(border=True, key="ch_foreign"):
        _foreign_holders(snap)


def render_charts_tab() -> None:
    _css()
    _render_nerd_body()
