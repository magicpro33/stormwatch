"""Native Streamlit Macro Simulator — same playbook as macro_simulator.html.

The HTML file is the source of the playbook lists. This tab runs the sliders,
gkey regime, dump/watchlist scoring, charts, and inflate math inside Streamlit
so nothing opens in an iframe portal.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

ACCENT = "#E87722"
GREEN = "#3fbf7f"
RED = "#e05252"
DIM = "#9aa8bd"
CREAM = "#F6F4E9"
NAVY = "#081325"
PANEL = "#16344f"
BORDER = "#3d6a94"
GOLD = "#e8c36a"

HERE = os.path.dirname(os.path.abspath(__file__))
PLAYBOOK_PATH = os.path.join(HERE, "macro_sim_playbook.json")

SCALE = {"oil": 1, "fed": 100, "iran": 1, "cpi": 10, "yield": 100, "y2": 100,
         "dxy": 10, "qe": 1, "slr": 1, "boj": 100, "usdjpy": 10}

SCEN_OPTS = [
    ("live", "● Live Market"),
    ("base", "Base Case"),
    ("bull", "Bull: Deal Works"),
    ("bear", "Bear: Deal Fails"),
    ("qe", "QE Machine"),
    ("stag", "Stagflation"),
    ("strong", "Strong Dollar"),
    ("carry", "Yen Carry Unwind"),
    ("inflate", "Inflate $"),
]
SCEN_LABELS = {k: lab for k, lab in SCEN_OPTS}

SLIDERS = [
    dict(key="oil", label="WTI Oil", min=20.0, max=200.0, step=0.5,
         fmt="$%.1f", help="West Texas Intermediate. Above $90 activates Alt Energy."),
    dict(key="fed", label="Fed Funds", min=1.0, max=7.0, step=0.01,
         fmt="%.2f%%", help="Effective federal funds. Every 25bps cut lifts forward multiples."),
    dict(key="iran", label="Iran Deal %", min=0.0, max=100.0, step=1.0,
         fmt="%.0f%%", help="0% = Hormuz shock. 100% = deal signed, oil floods the market."),
    dict(key="cpi", label="CPI Inflation", min=1.0, max=12.0, step=0.1,
         fmt="%.1f%%", help="Headline CPI YoY. Above 3.5% keeps the Fed on hold."),
    dict(key="y10", label="10-Year Yield", min=1.5, max=8.0, step=0.01,
         fmt="%.2f%%", help="Long anchor of the curve. Rising = inflation fear; falling = easing."),
    dict(key="y2", label="2-Year Yield", min=1.0, max=8.0, step=0.01,
         fmt="%.2f%%", help="The market's Fed-funds forecast. 2Y > 10Y is inverted."),
    dict(key="dxy", label="DXY Dollar", min=80.0, max=120.0, step=0.1,
         fmt="%.1f", help="USD vs a six-currency basket. Below 100 helps gold and commodities."),
    dict(key="qe", label="QE Probability", min=0.0, max=100.0, step=1.0,
         fmt="%.0f%%", help="Odds the Fed resumes bond-buying / covert QE via SLR."),
    dict(key="slr", label="SLR Deregulation", min=0.0, max=100.0, step=1.0,
         fmt="%.0f%%", help="Lets banks exclude Treasuries from leverage — covert QE."),
    dict(key="boj", label="BOJ Rate", min=0.0, max=3.0, step=0.01,
         fmt="%.2f%%", help="Japan policy rate. 1.50%+ with USD/JPY ≤145 is carry-unwind."),
    dict(key="usdjpy", label="USD/JPY", min=100.0, max=180.0, step=0.1,
         fmt="%.1f", help="Yen per dollar. A fast drop toward 145 is the unwind tell."),
]

LIVE_KEYS = ("oil", "fed", "cpi", "y10", "y2", "dxy", "usdjpy")
JUDGMENT_KEYS = ("iran", "qe", "slr", "boj")

BUY_SECTIONS = [
    ("gold", "Gold & precious metals", "base,bull,bear,qe,stag,carry,inflate", GOLD),
    ("banks", "Financials — SLR + QE", "base,bull,qe,strong", ACCENT),
    ("tech", "AI & tech", "base,bull,qe", "#7eb6d9"),
    ("energy", "Energy — tankers, E&P, refiners", "bear,stag,inflate", ACCENT),
    ("pricing", "Pricing power / necessities", "base,stag,inflate", GREEN),
    ("etf", "ETF broad exposure", "base,bull,bear,qe,stag,strong,carry,inflate", DIM),
    ("scan", "Top screener picks", "base,bull,qe", "#5DCAA5"),
]

ASSET_RET = {
    "bull":    [16, 20, 58, -22, 12, -10],
    "bear":    [-8, -5, -42, 38, -20, 8],
    "qe":      [22, 32, 72, -5, 16, -14],
    "stag":    [-10, 22, -28, 42, -32, 8],
    "base":    [6, 8, 12, -5, -3, 2],
    "strong":  [-5, -15, -30, -8, -10, 18],
    "carry":   [-12, 10, -35, -8, 20, -8],
    "inflate": [12, 38, 55, 30, -22, -16],
}
ASSET_NAMES = ["S&P 500", "Gold", "Bitcoin", "Oil", "10Y Bond", "USD"]

SECTOR_SCORE = {
    "bull":    [80, 48, 88, 55, 62, 50, 75, 60],
    "bear":    [38, 88, 45, 82, 60, 65, 25, 52],
    "qe":      [90, 52, 72, 68, 68, 50, 88, 58],
    "stag":    [25, 80, 35, 88, 65, 72, 18, 52],
    "base":    [62, 58, 62, 56, 52, 56, 50, 55],
    "strong":  [72, 42, 48, 32, 65, 72, 55, 68],
    "carry":   [32, 42, 18, 38, 68, 72, 58, 66],
    "inflate": [60, 88, 58, 90, 55, 58, 78, 50],
}
SECTOR_NAMES = ["Financials", "Energy", "Technology", "Materials",
                "Utilities", "Healthcare", "Real Estate", "Cons Def"]

OBS_MOVES = [
    ("FRO", "Frontline Tankers",
     dict(bull=-20, bear=55, qe=8, stag=50, base=5, strong=-5, carry=-8, inflate=20)),
    ("NEM", "Newmont Gold",
     dict(bull=18, bear=-8, qe=32, stag=22, base=5, strong=-20, carry=15, inflate=38)),
    ("LMT", "Lockheed Martin",
     dict(bull=5, bear=28, qe=6, stag=22, base=10, strong=30, carry=-3, inflate=8)),
    ("NVDA", "NVIDIA",
     dict(bull=55, bear=-28, qe=45, stag=-22, base=18, strong=-12, carry=-35, inflate=8)),
    ("JPM", "JPMorgan",
     dict(bull=32, bear=-15, qe=40, stag=-18, base=12, strong=14, carry=-12, inflate=10)),
    ("IBIT", "Bitcoin ETF",
     dict(bull=58, bear=-42, qe=72, stag=-28, base=10, strong=-30, carry=-35, inflate=50)),
    ("XLE", "Energy ETF",
     dict(bull=-12, bear=38, qe=6, stag=35, base=0, strong=-5, carry=-8, inflate=28)),
    ("CEG", "Constellation Nuke",
     dict(bull=18, bear=20, qe=18, stag=18, base=14, strong=15, carry=-10, inflate=14)),
    ("GDX", "Gold Miners ETF",
     dict(bull=25, bear=-10, qe=40, stag=20, base=5, strong=-22, carry=18, inflate=40)),
    ("KRE", "Regional Banks ETF",
     dict(bull=22, bear=-22, qe=38, stag=-30, base=5, strong=20, carry=-20, inflate=8)),
    ("UUP", "USD Bullish ETF",
     dict(bull=-8, bear=8, qe=-14, stag=5, base=2, strong=18, carry=-8, inflate=-16)),
    ("PLTR", "Palantir",
     dict(bull=60, bear=-18, qe=40, stag=-8, base=18, strong=5, carry=-18, inflate=10)),
]

TIMELINE = {
    "base": [("Now", "Warsh hawkish"), ("Q3 2026", "Iran deal limbo"),
             ("Q3 2026", "New PCE data"), ("Q4 2026", "First cut"),
             ("Q1 2027", "SLR dereg"), ("Q2 2027", "Covert QE fires")],
    "bull": [("Now", "Warsh hawkish"), ("M+1", "Deal signed"),
             ("M+2", "Oil $60s"), ("M+3", "PCE: 2.4%"),
             ("M+4", "Cut 75bps"), ("EOY", "S&P +20%")],
    "bear": [("Now", "Warsh hawkish"), ("M+1", "Deal collapses"),
             ("M+2", "Oil $100+"), ("M+3", "CPI 6%+"),
             ("M+4", "S&P -20%"), ("Q4", "Bond revolt")],
    "qe": [("Now", "Warsh hawkish"), ("M+2", "SLR passed"),
           ("M+3", "Banks buy bonds"), ("M+4", "New PCE fires"),
           ("M+5", "Cut 100bps"), ("M+6", "BTC ATH")],
    "stag": [("Now", "Warsh hawkish"), ("M+2", "Oil $115+"),
             ("M+3", "Growth stalls"), ("M+4", "Fed trapped"),
             ("M+6", "S&P -25%"), ("EOY", "Gold $5,000+")],
    "strong": [("Now", "DXY grinding up"), ("M+1", "27 nations seek USD"),
               ("M+2", "DXY breaks 107"), ("M+3", "Multinational EPS cut"),
               ("M+4", "Small caps lead"), ("Q4", "DXY 110 target")],
    "carry": [("Now", "BOJ hawkish tilt"), ("Day 1", "BOJ shock hike"),
              ("Day 2-3", "Weak US data"), ("Wk 1", "USD/JPY breaks 150"),
              ("Wk 2-3", "Margin call cascade"), ("M+1", "Fed liquidity backstop")],
    "inflate": [("Now", "$8T rollover wall"), ("Q3 2026", "30Y yield strains"),
                ("Q4 2026", "Fed caps long end"), ("2027", "Real rate goes -3%"),
                ("2027-30", "Gold/silver run"), ("5-8 yrs", "Debt/GDP inflated down")],
}

YIELD_SHIFTS = {
    "bull":    [-0.80, -0.80, -0.70, -0.65, -0.55, -0.40, -0.25, -0.15, -0.10],
    "bear":    [0.30, 0.30, 0.25, 0.20, 0.15, 0.25, 0.35, 0.45, 0.50],
    "qe":      [-1.20, -1.20, -1.10, -1.00, -0.90, -0.65, -0.45, -0.30, -0.20],
    "stag":    [0.05, 0.05, 0.08, 0.05, 0.00, 0.60, 1.20, 1.60, 1.80],
    "base":    [-0.20, -0.18, -0.15, -0.12, -0.10, -0.05, 0.00, 0.05, 0.08],
    "strong":  [0.10, 0.10, 0.10, 0.08, 0.05, 0.15, 0.30, 0.40, 0.45],
    "carry":   [-0.60, -0.60, -0.55, -0.55, -0.50, -0.45, -0.40, -0.30, -0.25],
    "inflate": [-1.30, -1.30, -1.20, -1.05, -0.85, -0.55, -0.30, -0.15, 0.00],
}

BULL_PATH = [1, 1.0267, 1.0644, 1.1022, 1.1422, 1.1911, 1.2356,
             1.2711, 1.3067, 1.3378, 1.3756, 1.4156]
BEAR_PATH = [1, 0.9756, 0.9422, 0.9089, 0.8756, 0.8422, 0.8244,
             0.8156, 0.8200, 0.8333, 0.8467, 0.8644]
SPX_LABELS = ["Now", "M+1", "M+2", "M+3", "M+4", "M+5",
              "M+6", "M+7", "M+8", "M+9", "M+10", "M+11"]
YC_LABELS = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]

BASE_IDX = dict(spx=6850, gold=4100, btc=118000, vix=16.5,
                copper=5.05, hyg=82.40, ief=95.30)
IDX_MAP = {
    "spx": "^GSPC", "gold": "GC=F", "btc": "BTC-USD",
    "vix": "^VIX", "copper": "HG=F", "hyg": "HYG", "ief": "IEF",
}

DANGER = [
    ("avBonds", "Long-duration bonds — rate-risk traps",
     "Long bonds are a bet the Fed cuts soon. If QE is delayed or inflation "
     "re-accelerates, duration gets hit. Wait for a confirmed cut."),
    ("avTech", "Unprofitable tech / high-burn EV",
     "Negative ROIC names that need cheap capital. Without QE the funding "
     "window closes."),
    ("avCons", "Consumer cyclical — discretionary",
     "High oil eats household budgets. Furniture, autos, restaurants compress "
     "earnings and multiples together."),
    ("avReit", "Real estate / REITs",
     "Bond proxies with leverage. Every 25bps on the 10Y compresses REIT "
     "valuations ~3–5%. Office demand is structurally weaker."),
    ("avStory", "Story stocks / no moat",
     "ZIRP-era stories without economics. Structural shorts in tight money."),
    ("avInd", "Small-cap industrials, negative ROIC trend",
     "Input-cost squeeze plus no pricing power. Margin compression from both sides."),
]


def _load_playbook() -> dict:
    if not os.path.isfile(PLAYBOOK_PATH):
        return dict(db={}, scens={}, narrs={}, angles=[], base_px={})
    with open(PLAYBOOK_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _playbook() -> dict:
    return _load_playbook()


def _decode_scen(raw: dict) -> dict:
    out = {}
    for k, scale in SCALE.items():
        if k not in raw:
            continue
        dest = "y10" if k == "yield" else k
        out[dest] = float(raw[k]) / float(scale)
    return out


def _default_vals(prints: dict | None = None) -> dict:
    pb = _playbook()
    vals = _decode_scen((pb.get("scens") or {}).get("base") or {
        "oil": 68, "fed": 363, "iran": 62, "cpi": 42, "yield": 448,
        "y2": 424, "dxy": 1013, "qe": 45, "slr": 40, "boj": 100, "usdjpy": 1638,
    })
    disp = (prints or {}).get("display") or {}
    mapping = {"oil": "oil", "fed": "fed", "cpi": "cpi", "yield": "y10",
               "y2": "y2", "dxy": "dxy", "usdjpy": "usdjpy"}
    for src, dest in mapping.items():
        if src not in disp:
            continue
        try:
            vals[dest] = float(disp[src])
        except (TypeError, ValueError):
            pass
    return vals


def _scen_vals(key: str, prints: dict | None = None) -> dict:
    if key == "live":
        return _default_vals(prints)
    pb = _playbook()
    raw = (pb.get("scens") or {}).get(key)
    if not raw:
        return _default_vals(prints)
    return _decode_scen(raw)


def gkey(v: dict) -> str:
    oil, fed, iran = v["oil"], v["fed"], v["iran"]
    cpi, dxy, qe, slr = v["cpi"], v["dxy"], v["qe"], v["slr"]
    boj, usdjpy = v["boj"], v["usdjpy"]
    if boj >= 1.5 and usdjpy <= 145:
        return "carry"
    if qe > 80 and slr > 80:
        return "qe"
    if oil > 95 and cpi > 7.5:
        return "stag"
    if iran < 20 and oil > 90:
        return "bear"
    if oil > 95 and cpi > 5.5:
        return "stag"
    if iran > 75 and oil < 70:
        return "bull"
    if dxy >= 107 and fed >= 4.0 and qe < 30:
        return "strong"
    if (cpi - fed) >= 1.5 and qe >= 70 and dxy < 98:
        return "inflate"
    return "base"


def engine_key(k: str) -> str:
    return "repress" if k == "inflate" else k


def _rewire_html(html: str) -> str:
    html = html or ""
    for a, b in (
        ("var(--ac3)", GREEN), ("var(--ac2)", ACCENT), ("var(--ac)", ACCENT),
        ("var(--red)", RED), ("var(--mu)", DIM), ("var(--gold)", GOLD),
        ("var(--pur)", "#7eb6d9"), ("var(--tx)", CREAM),
    ):
        html = html.replace(a, b)
    return html


def _sign_color(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return DIM
    if n > 0:
        return GREEN
    if n < 0:
        return RED
    return GOLD


def _fmt_ret(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(n):
        return "—"
    return f"{n:+.0f}%"


def scenario_pct(ticker: str, scen: str, row: dict, bundle: dict) -> float | None:
    if scen == "live":
        scen = gkey(st.session_state.get("ms_vals") or _default_vals())
    if row and row.get(scen) is not None:
        try:
            return float(row[scen])
        except (TypeError, ValueError):
            pass
    pb = playbook_of(ticker)
    if pb and pb.get(scen) is not None:
        try:
            return float(pb[scen])
        except (TypeError, ValueError):
            pass
    hit = _bundle_score(ticker, scen, bundle)
    if hit and hit.get("macrofit") is not None:
        try:
            mf = float(hit["macrofit"])
            qual = 50.0
            if hit.get("quality") is not None:
                qual = float(hit["quality"])
            elif hit.get("fit") is not None and mf:
                qual = float(hit["fit"]) / mf
            return round(6 + (mf - 1) * 80 + (qual - 50) * 0.24)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def playbook_of(ticker: str) -> dict | None:
    tk = str(ticker or "").upper()
    for arr in (_playbook().get("db") or {}).values():
        if not isinstance(arr, list):
            continue
        for s in arr:
            if str(s.get("t") or "").upper() == tk:
                return s
    return None


def _bundle_score(ticker: str, scen: str, bundle: dict) -> dict | None:
    if not bundle or not ticker:
        return None
    ek = engine_key(scen)
    tk = str(ticker).upper()
    cross = ((bundle.get("cross") or {}).get(tk) or {}).get(ek)
    if cross and cross.get("fit") is not None:
        return cross
    for r in (bundle.get("watch") or {}).get(ek) or []:
        if str(r.get("t") or "").upper() == tk and r.get("fit") is not None:
            return r
    for r in ((bundle.get("scans") or {}).get(ek) or {}).get("rows") or []:
        if str(r.get("t") or "").upper() == tk:
            return r
    return cross


def favored(arr, key: str) -> list:
    out = []
    for s in arr or []:
        v = s.get(key)
        if v is None:
            v = s.get("base")
        try:
            if v is not None and float(v) > 0:
                out.append(s)
        except (TypeError, ValueError):
            continue
    return out


def _spx_paths(v: dict, k: str, spx0: float):
    bull = [round(spx0 * p) for p in BULL_PATH]
    bear = [round(spx0 * p) for p in BEAR_PATH]
    m = v["iran"] / 100.0
    dm = max(1 - (v["oil"] - 60) / 100.0, 0.28)
    sp = v["y10"] - v["y2"]
    spread_m = 1 + max(min(sp * 0.35, 0.25), -0.20)
    proj = [round((b * m * dm * 0.5 + spx0 * 1.0156 * 0.35
                   + bear[i] * (1 - m) * 0.15) * spread_m)
            for i, b in enumerate(bull)]
    if k == "carry":
        proj = [round(p * 0.55 + bear[i] * 0.45) for i, p in enumerate(proj)]
    elif k == "strong":
        proj = [round(p * 0.965) for p in proj]
    elif k == "inflate":
        proj = [round(p * (1 + 0.015 * i / 11 + 0.06)) for i, p in enumerate(proj)]
    return bull, proj, bear


def _yield_curves(v: dict, k: str):
    y2, y10, fed = v["y2"], v["y10"], v["fed"]
    sp = y10 - y2
    yc = [
        round(fed + 0.05, 2), round(fed + 0.10, 2), round(fed + 0.15, 2),
        round(y2 - 0.35, 2), y2, round(y2 + sp * 0.55, 2),
        y10, round(y10 + 0.25, 2), round(y10 + 0.45, 2),
    ]
    sh = YIELD_SHIFTS.get(k) or YIELD_SHIFTS["base"]
    yp = [round(y + s, 2) for y, s in zip(yc, sh)]
    return yc, yp


def _inflate_math(v: dict, debt, dgdp, roll, oldc, grow, defc, tgt, yrs, path: str):
    cpi, fed, y10 = v["cpi"], v["fed"], v["y10"]
    extra_int_b = roll * max(y10 - oldc, 0) / 100.0 * 1000.0
    real_rate = fed - cpi
    eff_nom = oldc * 0.7 + y10 * 0.3
    r_real = (eff_nom - cpi) / 100.0
    g_real = grow / 100.0
    prim = defc / 100.0
    den = 1 + g_real if abs(1 + g_real) > 1e-9 else 1.0

    def step(x):
        return x * (1 + r_real) / den + prim

    years_to = None
    d = dgdp / 100.0
    for t in range(1, 41):
        d = step(d)
        if years_to is None and d <= tgt / 100.0:
            years_to = t
            break
    d_h = dgdp / 100.0
    for _ in range(int(yrs)):
        d_h = step(d_h)
    cash_real = (1 / (1 + cpi / 100.0)) ** yrs
    if path == "1946":
        assets = [
            ("Cash", 0.0),
            ("Long bonds (pegged)", 0.025),
            ("S&P 500", cpi / 100.0 + 0.05),
            ("Gold (fixed $35)", 0.0),
            ("Real assets / housing", cpi / 100.0 + 0.02),
        ]
    else:
        assets = [
            ("Cash", 0.0),
            ("Long bonds (nom)", y10 / 100.0 * 0.6),
            ("S&P 500", cpi / 100.0 + 0.01),
            ("Gold", 0.27),
            ("Silver", 0.30),
        ]
    rows = []
    for name, rate in assets:
        nominal = 10000 * (1 + rate) ** yrs
        real = nominal * cash_real
        rows.append(dict(Asset=name, Rate=f"{rate * 100:.0f}%/yr",
                         Nominal=round(nominal), Real=round(real)))
    return dict(
        extra_int_b=extra_int_b, real_rate=real_rate, debt_gdp=d_h * 100,
        years_to=years_to, assets=rows, r_real=r_real * 100,
    )


def _panel(html: str) -> str:
    return (
        f'<div style="background:{PANEL};border:1px solid {BORDER};'
        f'border-radius:12px;padding:12px 14px;margin:0 0 12px 0;">{html}</div>'
    )


def _stock_cards(rows: list, scen: str, html_fn, extra: str = "") -> None:
    if not rows:
        html_fn(f'<div style="color:{DIM};font-size:12px;padding:6px 0;">'
                f'No playbook names favored in this scenario.</div>')
        return
    cells = []
    for s in rows:
        tk = s.get("t") or ""
        val = s.get(scen)
        if val is None:
            val = s.get("base")
        col = _sign_color(val)
        chips = []
        if s.get("roic") and str(s["roic"]) not in ("--", "—", "N/A"):
            chips.append(str(s["roic"]))
        if s.get("gc") is not None:
            chips.append("GC✓" if float(s["gc"] or 0) >= 1 else "GC✗")
        if s.get("piotr") is not None:
            chips.append(f"P:{int(s['piotr'])}")
        if s.get("div") and str(s["div"]) not in ("0.0%", "—", "--"):
            chips.append(str(s["div"]))
        chip_html = "".join(
            f'<span style="background:#122a42;border:1px solid {BORDER};'
            f'border-radius:4px;padding:1px 6px;font-size:10px;color:{DIM};'
            f'margin-right:4px;">{c}</span>'
            for c in chips)
        why = s.get("r") or s.get("why") or ""
        cells.append(
            f'<div style="background:#122a42;border:1px solid {BORDER};'
            f'border-radius:10px;padding:10px 12px;">'
            f'<div style="font-weight:800;color:{col};font-size:14px;">{tk}'
            f'{extra}</div>'
            f'<div style="color:{DIM};font-size:11px;margin:2px 0 4px;">'
            f'{s.get("n") or ""}</div>'
            f'<div style="font-weight:800;color:{col};font-size:18px;">'
            f'{_fmt_ret(val)}</div>'
            f'<div style="margin:6px 0;">{chip_html}</div>'
            f'<div style="color:{DIM};font-size:11px;line-height:1.4;">{why}</div>'
            f'</div>'
        )
    html_fn(
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,'
        'minmax(180px,1fr));gap:8px;margin:6px 0 14px 0;">'
        + "".join(cells) + "</div>"
    )


def _fit_table(rows: list, scen: str, bundle: dict) -> pd.DataFrame:
    out = []
    for r in rows or []:
        tk = r.get("t") or ""
        ret = scenario_pct(tk, scen, playbook_of(tk) or {}, bundle)
        out.append({
            "Ticker": tk,
            "Sector": r.get("sec") or "—",
            "Return": None if ret is None else float(ret),
            "Outcome": r.get("outcome") or "—",
            "Fit": r.get("fit"),
            "MacroFit": r.get("macrofit"),
            "Quality": r.get("quality"),
            "ROIC": r.get("roic") or "—",
            "Why": r.get("why") or "",
            "Source": r.get("source") or r.get("src") or "",
        })
    return pd.DataFrame(out)


def _signals(v: dict, idx: dict) -> list:
    sp = v["y10"] - v["y2"]
    carry_gap = v["fed"] - v["boj"]
    carry_bull = carry_gap > 2.0 and v["usdjpy"] > 155
    carry_bear = v["boj"] >= 1.5 and v["usdjpy"] <= 145
    li, bi = idx, BASE_IDX

    def pctv(a, b):
        if a is None or not b:
            return None
        return (a - b) / b * 100.0

    gold_m = pctv(li.get("gold"), bi["gold"])
    btc_m = pctv(li.get("btc"), bi["btc"])
    cg_now = (li["copper"] / li["gold"] * 1000
              if li.get("copper") is not None and li.get("gold") else None)
    cg_base = bi["copper"] / bi["gold"] * 1000
    cg_m = None if cg_now is None else (cg_now - cg_base) / cg_base * 100
    cr_now = (li["hyg"] / li["ief"]
              if li.get("hyg") is not None and li.get("ief") else None)
    cr_base = bi["hyg"] / bi["ief"]
    cr_m = None if cr_now is None else (cr_now - cr_base) / cr_base * 100
    vix = li.get("vix")

    def sub(n, suffix="%"):
        if n is None:
            return "offline"
        return f"{n:+.1f}{suffix}"

    return [
        ("Dollar (DXY)", v["dxy"] < 98, v["dxy"] > 104, f"{v['dxy']:.1f}"),
        ("Stocks (Risk)", v["qe"] > 60 and v["iran"] > 60, v["oil"] > 95, ""),
        ("10Y Yield", v["y10"] < 4.0, v["y10"] > 4.8, f"{v['y10']:.2f}%"),
        ("Gold", (v["dxy"] < 98 and v["qe"] > 50) or (gold_m is not None and gold_m > 2),
         v["dxy"] > 105 and not (gold_m is not None and gold_m > 2), sub(gold_m)),
        ("Bitcoin", (v["qe"] > 65 and v["dxy"] < 100) or (btc_m is not None and btc_m > 3),
         v["qe"] < 20 and not (btc_m is not None and btc_m > 3), sub(btc_m)),
        ("2Y–10Y Spread", sp > 0.50, sp < 0.00, f"{sp:+.2f}%"),
        ("Yen Carry", carry_bull, carry_bear, f"¥{v['usdjpy']:.0f}"),
        ("VIX Fear", vix is not None and vix < 15, vix is not None and vix > 25,
         f"{vix:.1f}" if vix is not None else "offline"),
        ("Copper/Gold", cg_m is not None and cg_m > 3, cg_m is not None and cg_m < -3,
         sub(cg_m)),
        ("Credit Risk", cr_m is not None and cr_m > 1, cr_m is not None and cr_m < -2,
         sub(cr_m)),
    ]


def _idx_live(ce, nonce: int = 0) -> dict:
    cache_key = f"_ms_idx_{int(nonce)}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    live = {k: None for k in BASE_IDX}
    try:
        px = ce.yahoo_prices(list(IDX_MAP.values())) or {}
    except Exception:
        st.session_state[cache_key] = live
        return live
    for k, tk in IDX_MAP.items():
        val = px.get(tk)
        try:
            live[k] = float(val) if val is not None else None
        except (TypeError, ValueError):
            live[k] = None
    st.session_state[cache_key] = live
    return live


def _plotly_layout(fig, h=260):
    fig.update_layout(
        height=h, template="plotly_dark",
        paper_bgcolor=NAVY, plot_bgcolor="#122a42",
        font_color=CREAM, margin=dict(l=8, r=8, t=28, b=8),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def render_macro_sim_tab(*, ce, asof: str, live_regime: dict | None,
                         get_bundle, get_prints, esc, html,
                         closes=None, colors: dict | None = None):
    """Full simulator UI inside the Streamlit tab."""
    accent = (colors or {}).get("accent", ACCENT)
    green = (colors or {}).get("green", GREEN)
    red = (colors or {}).get("red", RED)
    dim = (colors or {}).get("dim", DIM)

    st.caption("Preset buttons scope the playbook, dump Top 20, and your "
               "watchlist to that scenario. Live Market follows the sliders; "
               "the dump ranking is the same quality × sector-tilt math as "
               "Scan Hub's Macro-only scan.")

    if live_regime:
        drivers = " · ".join(live_regime.get("drivers") or [])
        bond = ""
        try:
            if closes is not None:
                bond = ce.bond_master_switch(closes).get("warning", "")
        except Exception:
            bond = ""
        html(_panel(
            f'<span style="font-weight:700;">📡 Live regime (app-detected): '
            f'{esc(live_regime.get("label") or "—")}</span><br>'
            f'<span style="color:{dim};font-size:12px;">From: {esc(drivers)} — '
            f'Live Market uses this weather. Other presets are what-ifs.</span>'
            + (f'<br><span style="color:{dim};font-size:12px;">🏛 <b>Bond check:</b> '
               f'{esc(bond)}</span>' if bond else "")
        ))

    nonce = int(st.session_state.get("ms_print_nonce") or 0)
    try:
        prints = get_prints(nonce)
    except TypeError:
        prints = get_prints()
    except Exception:
        prints = {}
    prints = prints if isinstance(prints, dict) else {}

    if "ms_vals" not in st.session_state:
        st.session_state.ms_vals = _default_vals(prints)
        st.session_state.ms_applied = "live"
        st.session_state.ms_epoch = 0

    labs = [lab for _, lab in SCEN_OPTS]
    keys = [k for k, _ in SCEN_OPTS]
    applied = st.session_state.get("ms_applied", "live")
    idx0 = keys.index(applied) if applied in keys else 0
    pick_lab = st.radio("Scenario", labs, horizontal=True, index=idx0,
                        key="ms_scen_radio", label_visibility="collapsed")
    pick = keys[labs.index(pick_lab)]
    if pick != applied:
        st.session_state.ms_applied = pick
        st.session_state.ms_vals = _scen_vals(pick, prints)
        st.session_state.ms_epoch = int(st.session_state.get("ms_epoch") or 0) + 1
        st.rerun()

    src = prints.get("src") or {}
    disp = prints.get("display") or {}
    src_bits = [f"{k} {disp[k]:g} ({src.get(k, 'live')})"
                for k in ("oil", "fed", "yield", "y2", "dxy", "usdjpy", "cpi")
                if k in disp or (k == "yield" and "yield" in disp)]
    live_line, refresh = st.columns([5, 1], vertical_alignment="bottom")
    with live_line:
        st.markdown("Live prints · " + (" · ".join(src_bits) if src_bits
                                        else "server feed unavailable — using last slider values"))
    with refresh:
        if st.button("↻ Refresh", key="ms_refresh"):
            st.session_state.ms_print_nonce = nonce + 1
            if pick == "live":
                st.session_state.ms_vals = _scen_vals("live", None)
                try:
                    fresh = ce.macro_live_prints()
                except Exception:
                    fresh = prints
                st.session_state.ms_vals = _scen_vals("live", fresh)
                st.session_state.ms_epoch = int(st.session_state.ms_epoch) + 1
            st.rerun()

    epoch = int(st.session_state.get("ms_epoch") or 0)
    seed = dict(st.session_state.ms_vals)
    read = {}
    for i in range(0, len(SLIDERS), 4):
        cols = st.columns(4)
        for j, spec in enumerate(SLIDERS[i:i + 4]):
            with cols[j]:
                k = spec["key"]
                val = float(seed.get(k, spec["min"]))
                val = min(spec["max"], max(spec["min"], val))
                read[k] = st.slider(
                    spec["label"], spec["min"], spec["max"], val,
                    step=spec["step"], format=spec["fmt"],
                    help=spec["help"], key=f"ms_{k}_{epoch}")
    v = dict(read)
    v["spread"] = round(v["y10"] - v["y2"], 2)
    st.session_state.ms_vals = dict(v)
    k = gkey(v)
    whatif = pick != "live"

    spread_lbl = ("STEEP ✓" if v["spread"] > 0.5
                  else ("~ FLAT" if v["spread"] > 0 else "⚠ INVERTED"))
    spread_c = green if v["spread"] > 0.5 else (GOLD if v["spread"] > 0 else red)
    gap = v["fed"] - v["boj"]
    carry_state = ("unwind" if (v["boj"] >= 1.5 and v["usdjpy"] <= 145)
                   else ("watch" if (gap < 1.5 or v["usdjpy"] < 155) else "safe"))
    carry_lbl = {"safe": "CARRY INTACT ✓", "watch": "~ NARROWING",
                 "unwind": "⚠ UNWIND RISK"}[carry_state]
    carry_c = {"safe": green, "watch": GOLD, "unwind": red}[carry_state]
    badge = f"{'WHAT-IF' if whatif else 'LIVE'}: {k.upper()}"
    html(_panel(
        f'<div style="display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;">'
        f'<span style="font-weight:800;color:{accent};letter-spacing:1px;">{badge}</span>'
        f'<span style="color:{spread_c};font-size:12px;">Spread {v["spread"]:+.2f}% — {spread_lbl}</span>'
        f'<span style="color:{carry_c};font-size:12px;">Fed–BOJ +{gap:.2f}% — {carry_lbl}</span>'
        f'</div>'
    ))

    narr = (_playbook().get("narrs") or {}).get(k) or ""
    if narr:
        html(_panel(
            f'<div style="border-left:3px solid {accent};padding:2px 0 2px 12px;'
            f'font-size:13px;line-height:1.65;color:{CREAM};">{_rewire_html(narr)}</div>'
        ))
    if v["oil"] >= 90:
        st.warning("High oil (WTI $90+): Alt Energy below is in play — solar, "
                   "nuclear, wind and storage get a strategic tailwind.")

    idx = _idx_live(ce, nonce)
    sigs = _signals(v, idx)
    scols = st.columns(5)
    for i, (name, bu, be, sub) in enumerate(sigs):
        tone = green if bu else (red if be else GOLD)
        word = "BULLISH" if bu else ("BEARISH" if be else "NEUTRAL")
        with scols[i % 5]:
            html(
                f'<div style="background:#122a42;border:1px solid {tone};'
                f'border-radius:10px;padding:8px 6px;text-align:center;margin-bottom:8px;">'
                f'<div style="color:{dim};font-size:10px;">{name}</div>'
                f'<div style="color:{tone};font-weight:800;font-size:12px;">{word}</div>'
                f'<div style="color:{CREAM};font-size:11px;">{sub}</div></div>'
            )

    chips = [
        (f"⛽ ${v['oil']:.0f}", v["oil"] < 80),
        (f"Fed {v['fed']:.2f}%", v["fed"] < 4.5),
        (f"2Y {v['y2']:.2f}%", v["y2"] < 4.0),
        (f"10Y {v['y10']:.2f}%", v["y10"] < 4.2),
        (f"Spread {v['spread']:+.2f}%", v["spread"] > 0.5),
        (f"CPI {v['cpi']:.1f}%", v["cpi"] < 3.5),
        (f"DXY {v['dxy']:.1f}", v["dxy"] < 100),
        (f"BOJ {v['boj']:.2f}%", v["boj"] < 1.5),
        (f"USD/JPY {v['usdjpy']:.1f}", v["usdjpy"] > 150),
        (f"Carry +{gap:.2f}%", gap > 2.0),
        (f"Iran {v['iran']:.0f}%", v["iran"] > 70),
        (f"QE {v['qe']:.0f}%", v["qe"] > 60),
        (f"SLR {v['slr']:.0f}%", v["slr"] > 60),
    ]
    html('<div style="display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 14px 0;">' +
         "".join(
             f'<span style="border:1px solid {green if ok else red};color:{green if ok else red};'
             f'border-radius:100px;padding:3px 10px;font-size:11px;">{lab}</span>'
             for lab, ok in chips) + "</div>")

    import plotly.graph_objects as go
    spx0 = idx.get("spx") or BASE_IDX["spx"]
    bull, proj, bear = _spx_paths(v, k, float(spx0))
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=SPX_LABELS, y=bull, name="Bull",
                                 line=dict(color=green, width=2)))
        fig.add_trace(go.Scatter(x=SPX_LABELS, y=proj, name="Projected",
                                 line=dict(color=accent, width=2.5)))
        fig.add_trace(go.Scatter(x=SPX_LABELS, y=bear, name="Bear",
                                 line=dict(color=red, width=2)))
        fig.update_layout(title="S&P path (model, not a forecast)")
        st.plotly_chart(_plotly_layout(fig, 280), width="stretch", key=f"ms_spx_{k}")
    with c2:
        rets = ASSET_RET.get(k) or ASSET_RET["base"]
        cols = [green if n >= 0 else red for n in rets]
        fig = go.Figure(go.Bar(x=rets, y=ASSET_NAMES, orientation="h",
                               marker_color=cols, text=[f"{n:+d}%" for n in rets],
                               textposition="outside"))
        fig.update_layout(title="12-month analog returns")
        st.plotly_chart(_plotly_layout(fig, 280), width="stretch", key=f"ms_asset_{k}")
    c3, c4 = st.columns(2)
    with c3:
        sd = SECTOR_SCORE.get(k) or SECTOR_SCORE["base"]
        fig = go.Figure(go.Scatterpolar(
            r=sd + sd[:1], theta=SECTOR_NAMES + SECTOR_NAMES[:1],
            fill="toself", line=dict(color=accent)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                                     gridcolor=BORDER),
                                     bgcolor="#122a42"),
                          title="Sector score")
        st.plotly_chart(_plotly_layout(fig, 300), width="stretch", key=f"ms_sec_{k}")
    with c4:
        yc, yp = _yield_curves(v, k)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=YC_LABELS, y=yc, name="Current",
                                 line=dict(color=accent, width=2)))
        fig.add_trace(go.Scatter(x=YC_LABELS, y=yp, name="12M proj",
                                 line=dict(color=green, width=2, dash="dash")))
        fig.update_layout(title="Yield curve")
        st.plotly_chart(_plotly_layout(fig, 300), width="stretch", key=f"ms_yc_{k}")

    st.markdown("##### Projected moves — key instruments")
    for t, nm, ret in OBS_MOVES:
        r = float(ret.get(k) or 0)
        pct = min(abs(r) / 90.0 * 100.0, 100.0)
        col = green if r >= 0 else red
        fill = green if r >= 0 else red
        html(
            f'<div style="margin:0 0 8px 0;"><div style="display:flex;'
            f'justify-content:space-between;font-size:12px;">'
            f'<span><b>{t}</b> <span style="color:{dim};">{nm}</span></span>'
            f'<span style="color:{col};font-weight:700;">{r:+.0f}%</span></div>'
            f'<div style="height:7px;background:#122a42;border-radius:4px;overflow:hidden;">'
            f'<div style="width:{pct:.0f}%;height:100%;background:{fill};"></div>'
            f'</div></div>'
        )

    st.markdown("##### Macro event timeline")
    steps = TIMELINE.get(k) or TIMELINE["base"]
    tcols = st.columns(len(steps))
    for i, ((when, ev), col) in enumerate(zip(steps, tcols)):
        on = i == 0
        with col:
            html(
                f'<div style="text-align:center;padding:4px;">'
                f'<div style="width:22px;height:22px;border-radius:50%;margin:0 auto 6px;'
                f'background:{accent if on else "#122a42"};color:{NAVY if on else dim};'
                f'border:2px solid {accent if on else BORDER};font-size:11px;'
                f'line-height:18px;font-weight:800;">{"✓" if on else "○"}</div>'
                f'<div style="color:{dim};font-size:10px;">{when}</div>'
                f'<div style="color:{CREAM};font-size:11px;">{ev}</div></div>'
            )

    wl = []
    try:
        wl = ce.watchlist_load()
    except Exception:
        wl = []
    import json as _json
    live_reg = str((live_regime or {}).get("regime") or "base")
    bundle = {}
    if get_bundle:
        try:
            bundle = get_bundle(asof or "live",
                                _json.dumps(wl, default=str, sort_keys=True),
                                live_reg)
        except Exception:
            bundle = {}
    bundle = bundle if isinstance(bundle, dict) else {}
    ek = engine_key(k)

    st.markdown("##### Dump Top 20 — quality × this scenario's tilt")
    pack = (bundle.get("scans") or {}).get(ek) or (bundle.get("scans") or {}).get("base") or {}
    rows = pack.get("rows") or []
    asof_d = bundle.get("dump_asof")
    st.caption(
        (f"Dump as of {asof_d} · " if asof_d else "")
        + f"{pack.get('label') or ek} — {len(rows)} names from "
        + f"{pack.get('eligible') or '—'} eligible. Same math as Scan Hub Macro-only."
    )
    dump_df = _fit_table(rows, k, bundle)
    if dump_df.empty:
        st.info("Dump ranking loads from the nightly scan. If this is empty, "
                "the dump is still warming up.")
    else:
        st.dataframe(
            dump_df, hide_index=True, width="stretch",
            column_config={
                "Return": st.column_config.NumberColumn(format="%+.0f%%"),
                "Fit": st.column_config.NumberColumn(format="%.1f"),
                "MacroFit": st.column_config.NumberColumn(format="%.2f"),
                "Quality": st.column_config.NumberColumn(format="%.0f"),
                "Why": st.column_config.Column(width="large"),
            },
        )

    st.markdown("##### Your watchlist — outcome under this scenario")
    wrows = list((bundle.get("watch") or {}).get(ek) or [])
    rank = {"Favored": 0, "Supported": 1, "Neutral": 2, "Headwind": 3,
            "Not in dump": 4, "Unknown": 5}
    wrows.sort(key=lambda r: (rank.get(r.get("outcome"), 9),
                              -(r.get("fit") or 0)))
    fav_n = sum(1 for r in wrows if r.get("outcome") in ("Favored", "Supported"))
    head_n = sum(1 for r in wrows if r.get("outcome") == "Headwind")
    if not wrows:
        st.caption("Watchlist is empty. Add names from Scan Hub or Stock Lookup.")
    else:
        st.caption(f"{len(wrows)} name{'s' if len(wrows) != 1 else ''} — "
                   f"{fav_n} favored/supported, {head_n} headwind.")
        st.dataframe(
            _fit_table(wrows, k, bundle), hide_index=True, width="stretch",
            column_config={
                "Return": st.column_config.NumberColumn(format="%+.0f%%"),
                "Fit": st.column_config.NumberColumn(format="%.1f"),
                "MacroFit": st.column_config.NumberColumn(format="%.2f"),
                "Quality": st.column_config.NumberColumn(format="%.0f"),
                "Why": st.column_config.Column(width="large"),
            },
        )

    db = _playbook().get("db") or {}
    st.markdown("##### Buy list — playbook names for this scenario")
    any_buy = False
    for key, title, tags, col in BUY_SECTIONS:
        if k not in tags.split(","):
            continue
        names = favored(db.get(key), k)
        if not names:
            continue
        any_buy = True
        st.markdown(f"**{title}**")
        _stock_cards(names, k, html)
    if not any_buy:
        st.caption("No core buy-list sleeve is tagged for this scenario. "
                   "Check the special playbooks below.")

    if k in ("bull", "bear", "qe", "stag") or v["oil"] >= 90:
        st.markdown("##### Alt energy — high-oil activation")
        st.caption("When oil stays above $90: subsidies, corporate PPAs, and "
                   "retail rotation all accelerate. Nuclear is the AI-baseload sleeper.")
        for key, title in (("nuke", "Nuclear"), ("solar", "Solar"),
                           ("wind", "Wind / grid / storage")):
            names = favored(db.get(key), k)
            if names:
                st.markdown(f"**{title}**")
                _stock_cards(names, k, html)

    if k == "carry":
        active = v["boj"] >= 1.5 and v["usdjpy"] <= 145
        st.markdown("##### Yen carry trade unwind")
        if active:
            st.warning("ACTIVE: BOJ 1.50%+ and USD/JPY 145 or below. "
                       "Unwind conditions are live in the sliders.")
        else:
            st.caption("Trigger: BOJ 1.50%+ / USDJPY 145−. Safe havens: FXY, "
                       "TLT/IEF, gold miners. Losers: leveraged tech, BTC, EM, KRE.")
        safe = favored(
            [s for s in (db.get("carrytrade") or [])
             if s.get("t") in ("FXY", "TLT", "IEF", "GDX", "SHY")]
            + [s for s in (db.get("etf") or []) if s.get("t") in ("GDX", "GLD")],
            k)
        loss = [s for s in (db.get("carrytrade") or []) if s.get("t") == "EEM"]
        loss += [s for s in (db.get("tech") or []) if s.get("t") == "NVDA"]
        loss += [s for s in (db.get("etf") or []) if s.get("t") in ("IBIT", "KRE")]
        st.markdown("**Safe havens**")
        _stock_cards(safe, k, html)
        st.markdown("**Carry-funded losers**")
        _stock_cards(loss, k, html)

    if k == "strong":
        active = v["dxy"] >= 107 and v["fed"] >= 4.0 and v["qe"] < 30
        st.markdown("##### Strong dollar playbook")
        if active:
            st.warning("ACTIVE: DXY 107+ with a hawkish hold. Bretton Woods 2.0 is live.")
        sd = db.get("strongdollar") or []
        st.markdown("**Defense**")
        _stock_cards(favored([s for s in sd if s.get("t") in
                              ("LMT", "RTX", "NOC", "GD", "LHX")], k), k, html)
        st.markdown("**Domestic revenue**")
        _stock_cards(favored([s for s in sd if s.get("t") in
                              ("IWM", "WMT", "CEG", "COST")], k), k, html)
        st.markdown("**Dollar-linked**")
        _stock_cards(favored([s for s in sd if s.get("t") in
                              ("UUP", "KRE", "MCO", "V")], k), k, html)

    if k in ("bear", "stag"):
        active = v["oil"] >= 90 and v["dxy"] >= 100
        st.markdown("##### High oil + high dollar")
        if active:
            st.warning("ACTIVE: WTI $90+ and DXY 100+.")
        ho = db.get("hioil") or []
        st.markdown("**Energy**")
        _stock_cards(favored([s for s in ho if s.get("t") in
                              ("XOM", "CVX", "OXY", "MPC", "EOG")], k), k, html)
        st.markdown("**Consumer necessities**")
        _stock_cards(favored([s for s in ho if s.get("t") in
                              ("WMT", "COST", "PG", "KO", "MKC")], k), k, html)
        st.markdown("**Defense**")
        _stock_cards(favored([s for s in ho if s.get("t") in
                              ("LMT", "RTX")], k), k, html)

    if k == "inflate":
        st.markdown("##### Inflate-the-debt calculator")
        st.caption("Educational model of financial repression — not a forecast.")
        active = (v["cpi"] - v["fed"]) >= 1.5 and v["qe"] >= 70 and v["dxy"] < 98
        if active:
            st.warning("ACTIVE: CPI − Fed ≥ 1.5% with QE 70+ and DXY under 98.")
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            debt = st.number_input("Total debt ($T)", 1.0, 80.0, 39.8, 0.1, key="ms_ic_debt")
            dgdp = st.number_input("Debt / GDP (%)", 20.0, 250.0, 122.0, 1.0, key="ms_ic_dgdp")
        with i2:
            roll = st.number_input("Rollover wall ($T, 12mo)", 0.0, 20.0, 8.0, 0.5, key="ms_ic_roll")
            oldc = st.number_input("Old avg coupon (%)", 0.0, 10.0, 2.4, 0.1, key="ms_ic_oldc")
        with i3:
            grow = st.number_input("Real GDP growth (%)", -2.0, 6.0, 2.0, 0.1, key="ms_ic_grow")
            defc = st.number_input("Primary deficit (% GDP)", 0.0, 12.0, 3.5, 0.1, key="ms_ic_def")
        with i4:
            tgt = st.number_input("Target debt/GDP (%)", 10.0, 150.0, 90.0, 1.0, key="ms_ic_tgt")
            yrs = st.number_input("Horizon (yrs)", 1, 40, 10, 1, key="ms_ic_yrs")
        path = st.radio("Asset-path analogue", ["1970s Stagflation", "1946–51 Repression"],
                        horizontal=True, key="ms_ic_path")
        ic = _inflate_math(v, debt, dgdp, roll, oldc, grow, defc, tgt, yrs,
                           "1946" if path.startswith("1946") else "1970s")
        a, b, c = st.columns(3)
        extra = ic["extra_int_b"]
        extra_s = (f"${extra / 1000:.2f}T/yr" if abs(extra) >= 1000
                   else f"${extra:.0f}B/yr")
        a.metric("Rollover interest shock", f"+{extra_s}")
        b.metric("Real policy rate", f"{ic['real_rate']:+.2f}%")
        c.metric(f"Debt/GDP in {int(yrs)} yrs", f"{ic['debt_gdp']:.0f}%")
        if ic["years_to"]:
            st.caption(f"Hits the {tgt:.0f}% target in about {ic['years_to']} years "
                       f"at a {ic['r_real']:+.1f}% real cost of carry.")
        else:
            st.caption(f"Target {tgt:.0f}% is not reached within 40 years at these inputs.")
        st.caption("What $10,000 does over the horizon (illustrative, not a prediction).")
        adf = pd.DataFrame(ic["assets"])
        st.dataframe(
            adf, hide_index=True, width="stretch",
            column_config={
                "Nominal": st.column_config.NumberColumn(format="$%d"),
                "Real": st.column_config.NumberColumn(format="$%d"),
            },
        )

    if k in ("bear", "stag", "carry"):
        st.markdown("##### Danger zone — avoid / short by sector")
        for key, title, blurb in DANGER:
            names = db.get(key) or []
            if not names:
                continue
            with st.expander(title, expanded=False):
                st.caption(blurb)
                _stock_cards(names, "bear" if k != "stag" else "stag", html)

    st.markdown("##### Investment angles for this scenario")
    tag = k.upper()
    angles = [a for a in (_playbook().get("angles") or []) if a.get("s") == tag]
    if not angles:
        angles = [a for a in (_playbook().get("angles") or []) if a.get("s") == "BASE"]
    acols = st.columns(min(2, max(1, len(angles))))
    for i, a in enumerate(angles):
        with acols[i % len(acols)]:
            tks = " · ".join(a.get("tks") or [])
            html(_panel(
                f'<div style="color:{accent};font-size:10px;letter-spacing:1px;">'
                f'{esc(a.get("s") or tag)} ANGLE</div>'
                f'<div style="font-weight:800;margin:4px 0;">{esc(a.get("t") or "")}</div>'
                f'<div style="color:{dim};font-size:12px;line-height:1.55;">'
                f'{esc(a.get("b") or "")}</div>'
                f'<div style="margin-top:8px;color:{accent};font-size:12px;">{esc(tks)}</div>'
            ))

    st.caption("Educational simulator — not financial advice. Projected returns "
               "are hypothetical model outputs. Screened from the nightly dump "
               "using ROIC, Piotroski, Golden Cross, and sector-tilt scoring.")
