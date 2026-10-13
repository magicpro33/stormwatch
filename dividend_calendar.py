"""Dividend Calender — dump-backed capture calendar and income calculator.

Reads the nightly dump Stormwatch already has (no GitHub fetch, no yfinance
on first paint). Calendar = last NYSE trading day before ex-date (T+1).
"""
from __future__ import annotations

import calendar as cal_mod
import datetime
import gzip
import html as html_mod
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

try:
    from mw_paths import data_dir as _mw_data_dir
except Exception:
    def _mw_data_dir() -> str:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(d, exist_ok=True)
        return d

ACCENT = "#E87722"
GREEN = "#3fbf7f"
RED = "#e05252"
BLUE = "#4aa3df"
GOLD = "#e8c34a"
PURPLE = "#9b7fe0"
ORANGE_ALT = "#E87722"
DIM = "#9aa8bd"
INK = "#F6F4E9"
NAVY = "#0c1829"
STEEL = "#1a4060"
LINE = "#3d6a94"

_STOCK_GZ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "stock", "data", "stock_data.json.gz",
)
_HOL = {}


def _esc(s) -> str:
    return html_mod.escape("" if s is None else str(s), quote=True)


def _html(markup: str) -> None:
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup.replace("$", "&#36;"), unsafe_allow_html=True)


def _css() -> None:
    _html(
        f"""<style>
.dc-badge{{background:{NAVY};border:1px solid {LINE};border-left:4px solid {ACCENT};
  border-radius:10px;padding:8px 14px;margin:6px 0 12px;font-size:13px;color:{INK}}}
.dc-grid{{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:5px;margin-top:8px}}
.dc-hdr{{text-align:center;font-size:11px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:{DIM};padding:4px 0}}
.dc-day{{background:{NAVY};border:1px solid {LINE};border-radius:8px;min-height:86px;padding:7px 6px}}
.dc-day.today{{border:2px solid {ACCENT}}}
.dc-day.empty{{background:transparent;border:none;min-height:0}}
.dc-num{{font-size:11px;font-weight:600;color:{DIM};margin-bottom:4px}}
.dc-day.today .dc-num{{color:{ACCENT};font-weight:800}}
.dc-chip{{display:block;border-radius:4px;padding:2px 5px;margin-bottom:3px;
  font-size:11px;font-weight:700;font-family:ui-monospace,monospace;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-left:3px solid}}
.dc-t1{{background:rgba(63,191,127,.16);color:{GREEN};border-color:{GREEN}}}
.dc-t2{{background:rgba(63,191,127,.10);color:#6fd49a;border-color:#6fd49a}}
.dc-t3{{background:rgba(232,119,34,.14);color:{ACCENT};border-color:{ACCENT}}}
.dc-t4{{background:rgba(232,119,34,.08);color:#f0a05a;border-color:#f0a05a}}
.dc-t5{{background:rgba(224,82,82,.12);color:{RED};border-color:{RED}}}
.dc-leg{{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 4px}}
.dc-leg span{{padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;color:{INK}}}
.dc-month{{text-align:center;font-size:20px;font-weight:800;color:{INK};margin:4px 0;
  background:{NAVY};border:1px solid {LINE};border-left:4px solid {ORANGE_ALT};
  border-right:4px solid {ORANGE_ALT};border-radius:10px;padding:6px 10px}}
.dc-calc{{background:{NAVY};border:1px solid {LINE};border-left:4px solid {ACCENT};
  border-radius:10px;padding:16px 18px;color:{INK}}}
.dc-res{{background:#081325;border:1px solid {LINE};border-left:4px solid {GREEN};
  border-radius:10px;padding:14px 16px}}
.dc-sechead{{background:{NAVY};border:1px solid {LINE};border-left:5px solid var(--sc,{ACCENT});
  border-radius:10px;padding:9px 14px;margin:18px 0 8px;font-size:15px;font-weight:800;
  color:{INK};letter-spacing:.02em}}
.dc-sechead small{{font-weight:500;font-size:12px;color:{DIM};margin-left:8px}}
.dc-title{{background:linear-gradient(90deg,{STEEL},{NAVY});border:1px solid {LINE};
  border-left:6px solid {ACCENT};border-radius:12px;padding:12px 18px;margin:4px 0 10px;
  font-size:20px;font-weight:800;color:{INK}}}
.dc-metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:8px 0}}
@media(max-width:720px){{.dc-metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.dc-met{{background:{NAVY};border:1px solid {LINE};border-left:4px solid var(--mc,{ACCENT});
  border-radius:10px;padding:10px 14px}}
.dc-mlbl{{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:{DIM}}}
.dc-mval{{font-family:ui-monospace,monospace;font-size:24px;font-weight:800;color:var(--mc,{INK});margin-top:2px}}
.dc-msub{{font-size:12px;color:{DIM};font-family:ui-monospace,monospace}}
.dc-box{{background:{NAVY};border:1px solid {LINE};border-left:4px solid var(--bc,{BLUE});
  border-radius:10px;padding:12px 16px;color:{INK};font-size:14px;line-height:1.55}}
.dc-box ul{{margin:0;padding-left:20px}}
.dc-box li{{margin-bottom:6px}}
.dc-wrap.pj{{border-left:4px solid {PURPLE}}}
.dc-wrap.tb{{border-left:4px solid {GREEN}}}
.dc-cal{{background:{NAVY};border:1px solid {LINE};border-left:4px solid {ACCENT};
  border-radius:10px;padding:10px 12px}}
.dc-leg{{background:{NAVY};border:1px solid {LINE};border-left:4px solid {GOLD};
  border-radius:10px;padding:8px 12px}}
.st-key-dc_filters{{border-left:4px solid {BLUE};border-radius:10px}}
[class*="st-key-dc_calcbox"]{{border-left:4px solid {GREEN};border-radius:10px}}
.dc-row{{display:flex;justify-content:space-between;align-items:baseline;
  padding:6px 0;border-bottom:1px solid #1d2b40}}
.dc-row:last-child{{border-bottom:none}}
.dc-lbl{{font-size:13px;color:{DIM}}}
.dc-val{{font-family:ui-monospace,monospace;font-size:15px;font-weight:700;color:{INK}}}
.dc-val.big{{font-size:20px;color:{GREEN}}}
.dc-buy{{background:{RED};color:{INK};padding:2px 8px;border-radius:4px;
  font-size:11px;font-weight:800}}
.dc-tmr{{background:{GREEN};color:#081325;padding:2px 8px;border-radius:4px;
  font-size:11px;font-weight:800}}
.dc-wrap{{max-height:480px;overflow-y:auto;border:1px solid {LINE};border-radius:8px;background:{NAVY}}}
.dc-stbl{{width:100%;border-collapse:collapse}}
.dc-stbl thead th{{position:sticky;top:0;z-index:2;background:{STEEL};color:{DIM};
  font-size:11px;letter-spacing:.08em;text-transform:uppercase;padding:10px;
  text-align:left;white-space:nowrap;border-bottom:1px solid {LINE}}}
.dc-stbl tbody td{{padding:10px;border-bottom:1px solid #1d2b40;font-size:13px;color:{INK};background:{NAVY}}}
.dc-stbl tbody tr:last-child td{{border-bottom:none}}
.dc-stbl tbody tr.dc-trow:hover td{{background:{STEEL}}}
.dc-stbl tbody tr.dc-sel td{{background:#214c70;border-left:3px solid {ACCENT}}}
.dc-tk{{font-family:ui-monospace,monospace;font-weight:800;color:{INK}}}
.dc-sec{{font-size:12px;color:{DIM}}}
.dc-tdn{{font-family:ui-monospace,monospace;color:{INK}}}
.dc-fr{{font-size:12px;color:{DIM}}}
.dc-dt{{font-family:ui-monospace,monospace;color:{DIM}}}
.dc-cd{{font-family:ui-monospace,monospace;color:{INK};white-space:nowrap}}
.dc-yb{{padding:2px 9px;border-radius:100px;font-family:ui-monospace,monospace;
  font-size:12px;font-weight:700;background:rgba(255,255,255,.06);color:var(--yc);
  border:1px solid var(--yc)}}
</style>"""
    )


def _sechead(title, color=ACCENT, sub=""):
    s = f"<small>{_esc(sub)}</small>" if sub else ""
    _html(f'<div class="dc-sechead" style="--sc:{color}">{_esc(title)}{s}</div>')


def _metric_cards(cards):
    """cards = [(label, value, sub, color), ...] rendered as bordered cards."""
    out = ['<div class="dc-metrics">']
    for label, value, sub, color in cards:
        out.append(
            f'<div class="dc-met" style="--mc:{color}">'
            f'<div class="dc-mlbl">{_esc(label)}</div>'
            f'<div class="dc-mval">{_esc(value)}</div>'
            f'<div class="dc-msub">{_esc(sub) if sub else "&nbsp;"}</div></div>'
        )
    out.append("</div>")
    _html("".join(out))


def _simple_table(headers, rows, cls="pj"):
    th = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr class=\"dc-trow\">" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>"
        for r in rows
    )
    _html(
        f'<div class="dc-wrap {cls}"><table class="dc-stbl"><thead><tr>{th}'
        f"</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _section(options, key):
    if hasattr(st, "segmented_control"):
        sel = st.segmented_control("Section", options, key=key,
                                   label_visibility="collapsed")
        return sel if sel in options else options[0]
    return st.radio("Section", options, horizontal=True, key=key,
                    label_visibility="collapsed")


def _ycolor(y):
    if y >= 8:
        return GREEN
    if y >= 6:
        return "#6fd49a"
    if y >= 4:
        return ACCENT
    if y >= 2.5:
        return "#f0a05a"
    return RED


def _render_dividend_table(df_show, today, show_buy_cols=True, selected=None):
    rows_html = []
    sel = str(selected).upper() if selected else ""
    for row in df_show.to_dict("records"):
        yc = _ycolor(row["yield_pct"])
        bd = safe_date(row.get("buy_date"))
        ex = safe_date(row.get("ex_date"))
        raw_da = row.get("days_away")
        try:
            da = int(raw_da) if raw_da is not None and not (isinstance(raw_da, float) and pd.isna(raw_da)) else None
        except (TypeError, ValueError):
            da = None
        dr = float(row.get("div_rate") or 0)
        mp = row.get("monthly_pay")
        px = float(row.get("price") or 0)
        pr = row.get("payout")
        freq = str(row.get("frequency") or "—")
        mp_str = f"${mp:.4f}" if mp else "—"
        pr_str = f"{pr:.0f}%" if pr else "—"
        bd_str = bd.strftime("%b %d, %Y") if bd else "—"
        ex_str = ex.strftime("%b %d, %Y") if ex else "—"
        alert = ""
        if show_buy_cols and da is not None:
            if da == 0:
                alert = '<span class="dc-buy">BUY TODAY</span>'
            elif da == 1:
                alert = '<span class="dc-tmr">BUY TOMORROW</span>'
        buy_cols = ""
        if show_buy_cols:
            cd = f"{da}d {alert}" if da is not None else "—"
            buy_cols = (
                f'<td class="dc-dt">{bd_str}</td>'
                f'<td class="dc-dt">{ex_str}</td>'
                f'<td class="dc-cd">{cd}</td>'
            )
        is_sel = str(row.get("ticker") or "").upper() == sel
        rows_html.append(
            f'<tr class="dc-trow{" dc-sel" if is_sel else ""}">'
            f'<td class="dc-tk">{_esc(row["ticker"])}</td>'
            f'<td class="dc-sec">{_esc(row["sector"])}</td>'
            f'<td><span class="dc-yb" style="--yc:{yc}">{row["yield_pct"]}%</span></td>'
            f'<td class="dc-tdn">${dr:.4f}</td>'
            f'<td class="dc-tdn">{mp_str}</td>'
            f'<td class="dc-tdn">{pr_str}</td>'
            f'<td class="dc-fr">{_esc(freq)}</td>'
            f'<td class="dc-tdn">${px:.2f}</td>'
            + buy_cols +
            "</tr>"
        )
    hdr_buy = "<th>Buy Before</th><th>Ex-Date</th><th>Countdown</th>" if show_buy_cols else ""
    _html(
        '<div class="dc-wrap tb"><table class="dc-stbl"><thead><tr>'
        "<th>Ticker</th><th>Sector</th><th>Yield</th><th>Div/Share</th>"
        '<th title="Annual dividend divided by 12">Monthly/Share</th>'
        "<th>Payout</th><th>Frequency</th><th>Price</th>"
        + hdr_buy +
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table></div>"
    )


def safe_date(v):
    if v is None:
        return None
    if isinstance(v, datetime.datetime):
        return v.date()
    if isinstance(v, datetime.date):
        return v
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        ts = pd.to_datetime(v, errors="coerce")
        if ts is not pd.NaT and not pd.isna(ts):
            return ts.date()
    except Exception:
        return None
    return None


def _epoch_to_date(ts):
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if ts < 1e9:
        return None
    try:
        return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).date()
    except Exception:
        return None


def _parse_ex_date(ex_ts):
    if not ex_ts:
        return None
    try:
        ts = float(ex_ts)
        if ts > 1e9:
            return _epoch_to_date(ts)
    except (TypeError, ValueError):
        pass
    return safe_date(ex_ts)


def _easter(y):
    a, b, c = y % 19, y // 100, y % 100
    d, e, f = b // 4, b % 4, (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    return datetime.date(y, (h + l - 7 * m + 114) // 31, ((h + l - 7 * m + 114) % 31) + 1)


def _nth_weekday(y, month, weekday, n):
    d = datetime.date(y, month, 1)
    return d + datetime.timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(y, month, weekday):
    nxt = datetime.date(y + 1, 1, 1) if month == 12 else datetime.date(y, month + 1, 1)
    d = nxt - datetime.timedelta(days=1)
    while d.weekday() != weekday:
        d -= datetime.timedelta(days=1)
    return d


def _holidays(y):
    if y in _HOL:
        return _HOL[y]

    def obs(d):
        if d.weekday() == 5:
            return d - datetime.timedelta(days=1)
        if d.weekday() == 6:
            return d + datetime.timedelta(days=1)
        return d

    hs = {
        obs(datetime.date(y, 1, 1)),
        _nth_weekday(y, 1, 0, 3),
        _nth_weekday(y, 2, 0, 3),
        _easter(y) - datetime.timedelta(days=2),
        _last_weekday(y, 5, 0),
        obs(datetime.date(y, 6, 19)),
        obs(datetime.date(y, 7, 4)),
        _nth_weekday(y, 9, 0, 1),
        _nth_weekday(y, 11, 3, 4),
        obs(datetime.date(y, 12, 25)),
    }
    _HOL[y] = hs
    return hs


def _prev_trading_day(d):
    if d is None:
        return None
    d = d - datetime.timedelta(days=1)
    for _ in range(10):
        if d.weekday() < 5 and d not in _holidays(d.year):
            return d
        d -= datetime.timedelta(days=1)
    return d


def _in_buy_window(bd, start, end):
    d = safe_date(bd)
    return d is not None and start <= d <= end


def _freq_to_pays(freq_label):
    f = (freq_label or "").lower()
    if "month" in f:
        return 12
    if "semi" in f:
        return 2
    if "annual" in f:
        return 1
    return 4


def _tier(y):
    if y >= 8:
        return "dc-t1"
    if y >= 6:
        return "dc-t2"
    if y >= 4:
        return "dc-t3"
    if y >= 2.5:
        return "dc-t4"
    return "dc-t5"


def _gz_paths():
    paths = [
        os.path.join(_mw_data_dir(), "stock_data.json.gz"),
        os.path.normpath(_STOCK_GZ),
        os.path.join(os.path.expanduser("~"), "Documents", "GitHub",
                     "stock", "data", "stock_data.json.gz"),
    ]
    seen, out = set(), []
    for p in paths:
        if p and p not in seen and os.path.exists(p):
            seen.add(p)
            out.append(p)
    return out


def _fresh_gz():
    """Newest local dump by mtime — Stormwatch cache first, then the stock repo."""
    found = _gz_paths()
    if not found:
        return None, 0.0
    path = max(found, key=os.path.getmtime)
    return path, os.path.getmtime(path)


def _div_cache_paths(mtime: float):
    root = _mw_data_dir()
    tag = str(int(mtime))
    return (os.path.join(root, f"div_cal_{tag}.parquet"),
            os.path.join(root, "div_cal_stamp.txt"))


def _rows_from_gz(path: str) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tk = str(item.get("Ticker") or "").upper()
        if not tk:
            continue
        try:
            yp = float(item.get("DividendYieldPct") or 0)
        except (TypeError, ValueError):
            continue
        if yp <= 0 or yp > 50:
            continue
        try:
            dr = float(item.get("DividendRate") or 0) or None
        except (TypeError, ValueError):
            dr = None
        try:
            pr = float(item.get("DividendPayoutRatio") or 0) or None
            if pr is not None and (pr < 0 or pr > 500):
                pr = None
        except (TypeError, ValueError):
            pr = None
        try:
            px = float(item.get("Price") or 0) or None
        except (TypeError, ValueError):
            px = None
        rows.append({
            "ticker": tk,
            "sector": item.get("Sector") or "Unknown",
            "price": px,
            "yield_pct": round(yp, 2),
            "div_rate": dr,
            "monthly_pay": round(dr / 12, 4) if dr else None,
            "payout": pr,
            "frequency": item.get("DividendFrequency") or "—",
            "ex_date": _parse_ex_date(item.get("ExDividendDate")),
        })
    return rows


@st.cache_data(ttl=3600, show_spinner=False)
def _load_dividend_df(stamp: str) -> pd.DataFrame:
    path, mt = _fresh_gz()
    if not path:
        return pd.DataFrame()
    pq, stamp_p = _div_cache_paths(mt)
    if os.path.exists(pq):
        try:
            df = pd.read_parquet(pq)
            if "ex_date" in df.columns:
                df["ex_date"] = df["ex_date"].map(safe_date)
            return df
        except Exception:
            pass
    rows = _rows_from_gz(path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("yield_pct", ascending=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.reset_index(drop=True)
    try:
        save = df.copy()
        save["ex_date"] = pd.to_datetime(save["ex_date"])
        save.to_parquet(pq, index=False)
        for old in os.listdir(_mw_data_dir()):
            if old.startswith("div_cal_") and old.endswith(".parquet") and old != os.path.basename(pq):
                try:
                    os.remove(os.path.join(_mw_data_dir(), old))
                except OSError:
                    pass
        with open(stamp_p, "w", encoding="ascii") as f:
            f.write(stamp)
    except Exception:
        pass
    return df


def _dump_stamp() -> str:
    path, mt = _fresh_gz()
    return f"{int(mt)}" if path else "none"


def _render_calendar(df, year, month):
    today = datetime.date.today()
    day_map = {}
    for row in df.itertuples(index=False):
        bd = safe_date(getattr(row, "buy_date", None))
        if bd and bd.year == year and bd.month == month:
            day_map.setdefault(bd.day, []).append(row)
    parts = ['<div class="dc-cal"><div class="dc-grid">']
    for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"):
        parts.append(f'<div class="dc-hdr">{d}</div>')
    for week in cal_mod.monthcalendar(year, month):
        for day in week:
            if day == 0:
                parts.append('<div class="dc-day empty"></div>')
                continue
            is_td = year == today.year and month == today.month and day == today.day
            parts.append(f'<div class="{"dc-day today" if is_td else "dc-day"}">')
            parts.append(f'<div class="dc-num">{day}</div>')
            for row in day_map.get(day, []):
                t = _tier(row.yield_pct)
                ex = safe_date(row.ex_date)
                mp = row.monthly_pay
                px = row.price or 0
                tip = (
                    f"BUY {row.ticker} before {ex} | Yield: {row.yield_pct}% | "
                    f"Monthly: ${mp:.4f}" if mp else
                    f"BUY {row.ticker} before {ex} | Yield: {row.yield_pct}%"
                ) + f" | Price: ${px:.2f}"
                parts.append(
                    f'<span class="dc-chip {t}" title="{_esc(tip)}">'
                    f"{_esc(row.ticker)} {row.yield_pct}%</span>"
                )
            parts.append("</div>")
    parts.append("</div></div>")
    _html("".join(parts))


def _calc_from_row(row) -> dict:
    return {
        "ticker": row["ticker"],
        "price": float(row.get("price") or 0),
        "yield_pct": float(row["yield_pct"]),
        "div_rate": float(row.get("div_rate") or 0),
        "monthly_pay": row.get("monthly_pay"),
        "frequency": row.get("frequency") or "—",
        "payout": row.get("payout"),
        "sector": row.get("sector") or "Unknown",
    }


def _render_calc_panel(info: dict, key: str = "dc_inv") -> None:
    if not info or not info.get("price", 0):
        if info:
            st.warning("No usable price on this name.")
        return
    left, right = st.columns(2)
    with left:
        _html(
            f'<div class="dc-calc"><b>{_esc(info["ticker"])}</b> — '
            f'{_esc(info["sector"])}<br>'
            f'Yield <b>{info["yield_pct"]}%</b> · '
            f'Price <b>${info["price"]:.2f}</b> · '
            f'{_esc(info["frequency"])}</div>'
        )
        with st.container(border=True, key=f"dc_calcbox_{key}"):
            inv = st.number_input("Investment ($)", min_value=1.0,
                                  max_value=10_000_000.0, value=1000.0,
                                  step=100.0, format="%.2f", key=key)
    with right:
        px = info["price"]
        dr = info["div_rate"] or 0
        shrs = inv / px if px > 0 else 0
        annd = shrs * dr
        mthd = annd / 12
        n = _freq_to_pays(info["frequency"])
        spay = annd / n if n else annd
        slbl = {12: "Per monthly payment", 4: "Per quarterly payment",
                2: "Per semi-annual payment", 1: "Per annual payment"}[n]
        _html(
            '<div class="dc-res">'
            f'<div class="dc-row"><span class="dc-lbl">Shares</span>'
            f'<span class="dc-val">{shrs:.4f}</span></div>'
            f'<div class="dc-row"><span class="dc-lbl">{_esc(slbl)}</span>'
            f'<span class="dc-val">${spay:.2f}</span></div>'
            f'<div class="dc-row"><span class="dc-lbl">Monthly income</span>'
            f'<span class="dc-val">${mthd:.2f}</span></div>'
            f'<div class="dc-row"><span class="dc-lbl">Annual income</span>'
            f'<span class="dc-val big">${annd:.2f}</span></div>'
            f'<div class="dc-row"><span class="dc-lbl">Yield</span>'
            f'<span class="dc-val">{info["yield_pct"]}%</span></div>'
            f'<div class="dc-row"><span class="dc-lbl">Weekly income</span>'
            f'<span class="dc-val">${annd / 52:.2f}</span></div>'
            "</div>"
        )
    _sechead("Holding-period projections", PURPLE,
             "dividends only, no price change")
    _simple_table(
        ["Hold", "Total dividends", "Return", "Monthly income"],
        [
            (f"{m} month" + ("s" if m > 1 else ""),
             f"${mthd * m:.2f}",
             f"{(mthd * m / inv) * 100:.2f}%",
             f"${mthd:.2f}")
            for m in (1, 3, 6, 12, 24, 36, 60)
        ],
    )


def _yield_list_and_calc(src, today, *, show_buy_cols, table_key, inv_key):
    """Original ranked table, plus a ticker pick that loads the calculator under it."""
    src = src.reset_index(drop=True).copy()
    if show_buy_cols and "days_away" not in src.columns:
        src["days_away"] = [
            (safe_date(bd) - today).days if safe_date(bd) else None
            for bd in src["buy_date"]
        ]
    tickers = list(dict.fromkeys(str(t) for t in src["ticker"].tolist()))
    placeholder = "— select a stock —"
    opts = [placeholder] + tickers
    sel_key = f"{table_key}_sel"

    # Seed the selectbox from the last pick (e.g. set on the Calculator tab),
    # and drop a stale value that is no longer in the filtered list.
    if sel_key not in st.session_state:
        last = st.session_state.get("dc_pick")
        st.session_state[sel_key] = last if last in tickers else placeholder
    elif st.session_state[sel_key] not in opts:
        st.session_state[sel_key] = placeholder

    # Read the selection BEFORE drawing the table so the highlight is the
    # stock currently loaded in the calculator, not the previous one.
    chosen = st.session_state[sel_key]
    pick = chosen if chosen != placeholder else None
    st.session_state["dc_pick"] = pick

    _render_dividend_table(src, today, show_buy_cols, selected=pick)
    _sechead("Load calculator", BLUE, "pick a stock to highlight it above")
    st.selectbox("Load calculator for", opts, key=sel_key,
                 label_visibility="collapsed")
    if pick is None:
        return
    row = src[src["ticker"] == pick].iloc[0]
    _sechead(f"Calculator — {pick}", GREEN)
    _render_calc_panel(_calc_from_row(row), key=inv_key)


def render_dividend_calendar_tab():
    _css()
    _html('<div class="dc-title">Dividend Calender</div>')
    view = _section(["Calendar", "Calculator", "Help"], "dc_view")

    stamp = _dump_stamp()
    with st.spinner("Reading dividend names from the dump…"):
        df_all = _load_dividend_df(stamp)
    if df_all.empty:
        st.info("No dividend names in the nightly dump. Load stock info in the header first.")
        return

    today = datetime.date.today()
    if "dc_cy" not in st.session_state:
        st.session_state["dc_cy"] = today.year
    if "dc_cm" not in st.session_state:
        st.session_state["dc_cm"] = today.month

    secs = ["All sectors"] + sorted(
        s for s in df_all["sector"].dropna().unique().tolist() if s)
    if view != "Help":
        _sechead("Filters", BLUE, "narrow the universe")
        with st.container(border=True, key="dc_filters"):
            f1, f2, f3 = st.columns(3)
            min_y = f1.slider("Min yield (%)", 0.0, 25.0, 0.0, 0.5, key="dc_min_y")
            max_y = f2.slider("Max yield (%)", 5.0, 50.0, 25.0, 1.0, key="dc_max_y")
            days = f3.slider("Days ahead", 30, 180, 90, key="dc_days")
            f4, f5, f6 = st.columns(3)
            max_px = f4.number_input("Max price ($)", min_value=1, max_value=100_000,
                                     value=1000, step=1, key="dc_max_px")
            freq = f5.selectbox("Frequency",
                                ["All", "Monthly", "Quarterly", "Semi-Annual", "Annual"],
                                key="dc_freq")
            sector = f6.selectbox("Sector", secs, key="dc_sec")
    else:
        min_y = float(st.session_state.get("dc_min_y", 0.0))
        max_y = float(st.session_state.get("dc_max_y", 25.0))
        days = int(st.session_state.get("dc_days", 90))
        max_px = float(st.session_state.get("dc_max_px", 1000))
        freq = st.session_state.get("dc_freq", "All")
        sector = st.session_state.get("dc_sec", "All sectors")

    df = df_all[
        (df_all["yield_pct"] >= min_y)
        & (df_all["yield_pct"] <= max_y)
        & (df_all["price"].fillna(0) <= max_px)
    ].copy()
    if freq != "All":
        df = df[df["frequency"].str.contains(freq, case=False, na=False)]
    if sector != "All sectors":
        df = df[df["sector"] == sector]

    df["buy_date"] = df["ex_date"].map(lambda d: _prev_trading_day(safe_date(d)))
    cutoff = today + datetime.timedelta(days=int(days))
    df_cal = df[df["buy_date"].map(lambda bd: _in_buy_window(bd, today, cutoff))].copy()
    df_cal = df_cal.sort_values("yield_pct", ascending=False)

    n_ex = int(df["ex_date"].notna().sum())
    if view != "Help":
        _html(
            f'<div class="dc-badge">{len(df_all):,} dividend stocks in the dump · '
            f"{n_ex:,} with ex-dates in this filter · {len(df_cal):,} buy signals "
            f"in the next {int(days)} days</div>"
        )

    if view == "Calendar":
        df_sorted = (
            df_cal.sort_values(["buy_date", "yield_pct"], ascending=[True, False])
            .reset_index(drop=True) if not df_cal.empty else df_cal
        )
        nxt = df_sorted.iloc[0] if not df_cal.empty else None
        avg = (df_cal["yield_pct"].mean() if not df_cal.empty
               else (df["yield_pct"].mean() if not df.empty else np.nan))
        hi_src = df_cal if not df_cal.empty else df
        if not hi_src.empty:
            top = hi_src.sort_values("yield_pct", ascending=False).iloc[0]
            hi_val, hi_sub = f"{top['yield_pct']:.1f}%", str(top["ticker"])
        else:
            hi_val, hi_sub = "—", ""
        nd = safe_date(nxt["buy_date"]) if nxt is not None else None
        _sechead("Snapshot", GREEN, "what the filters are showing")
        _metric_cards([
            ("Buy signals ahead", f"{len(df_cal):,}", "", GREEN),
            ("Avg yield", f"{avg:.1f}%" if np.isfinite(avg) else "—", "", ACCENT),
            ("Highest yield", hi_val, hi_sub, GOLD),
            ("Next buy date", nd.strftime("%b %d") if nd else "—",
             str(nxt["ticker"]) if nxt is not None else "", BLUE),
        ])

        _sechead("Buy calendar", ACCENT, "last trading day before ex-date")
        cp, cc, cn = st.columns([1, 5, 1])
        with cp:
            if st.button("Prev", key="dc_prev", width="stretch"):
                if st.session_state["dc_cm"] == 1:
                    st.session_state["dc_cy"] -= 1
                    st.session_state["dc_cm"] = 12
                else:
                    st.session_state["dc_cm"] -= 1
                st.rerun()
        with cn:
            if st.button("Next", key="dc_next", width="stretch"):
                if st.session_state["dc_cm"] == 12:
                    st.session_state["dc_cy"] += 1
                    st.session_state["dc_cm"] = 1
                else:
                    st.session_state["dc_cm"] += 1
                st.rerun()
        with cc:
            _html(
                f'<div class="dc-month">{cal_mod.month_name[st.session_state["dc_cm"]]} '
                f'{st.session_state["dc_cy"]}</div>'
            )
        _html(
            '<div class="dc-leg">'
            f'<span style="background:{GREEN}">8%+</span>'
            '<span style="background:#2d8a58">6%+</span>'
            f'<span style="background:{ACCENT}">4%+</span>'
            '<span style="background:#b86a20">2.5%+</span>'
            f'<span style="background:{RED}">below 2.5%</span>'
            "</div>"
        )
        _render_calendar(df_cal, st.session_state["dc_cy"], st.session_state["dc_cm"])

        _sechead("Upcoming buy signals", GREEN, "ranked by yield")
        if df_cal.empty:
            if df.empty:
                st.info("No dividend stocks match these filters.")
            else:
                st.info(f"No buy signals in the next {int(days)} days. "
                        "Widen the window or check that the dump has ex-dates.")
                show = df.sort_values("yield_pct", ascending=False).head(200)
                _yield_list_and_calc(show, today, show_buy_cols=False,
                                     table_key="dc_yield_table", inv_key="dc_inv")
        else:
            _yield_list_and_calc(df_sorted, today, show_buy_cols=True,
                                 table_key="dc_yield_table", inv_key="dc_inv")

    elif view == "Calculator":
        _sechead("Dividend income calculator", GREEN, "estimate payouts")
        _sechead("Stock source", BLUE)
        mode = st.radio("Stock source", ["Pick from the list", "Enter a ticker"],
                        horizontal=True, key="dc_calc_mode")
        info = None
        if mode == "Pick from the list":
            src = df_cal if not df_cal.empty else df
            if src.empty:
                st.warning("No dividend stocks match the filters.")
            else:
                src = src.sort_values("yield_pct", ascending=False).head(400)
                opts = [f"{r.ticker}  —  {r.yield_pct}%  |  {r.frequency}"
                        for r in src.itertuples(index=False)]
                pick = st.session_state.get("dc_pick")
                idx = 0
                if pick:
                    for i, r in enumerate(src.itertuples(index=False)):
                        if r.ticker == pick:
                            idx = i
                            break
                sel = st.selectbox("Select stock", opts, index=idx, key="dc_calc_sel")
                tk = sel.split("  —  ")[0].strip()
                row = src[src["ticker"] == tk].iloc[0]
                info = _calc_from_row(row)
        else:
            ctk = st.text_input("Ticker",
                                value=st.session_state.get("dc_pick") or "",
                                placeholder="ET, EPD, DOC",
                                key="dc_calc_tk").strip().upper()
            if ctk:
                hit = df_all[df_all["ticker"] == ctk]
                if not hit.empty:
                    info = _calc_from_row(hit.iloc[0])
                else:
                    st.warning(f"{ctk} is not a dividend stock in the nightly dump.")
        _render_calc_panel(info, key="dc_inv_tab")

    elif view == "Help":
        _sechead("How dividend capture works", GOLD)
        help_items = [
            (ACCENT, "Price drop",
             "The price typically drops by about the dividend on the ex-date. "
             "It is a transfer from share price to cash, not free money."),
            (GREEN, "Buy date",
             "Last day to buy is the previous <b>trading</b> day (T+1 settlement)."),
            (BLUE, "Taxes",
             "Qualified-dividend tax treatment needs a 60-day hold in the "
             "121-day window around the ex-date. Immediate capture is usually "
             "ordinary income plus a short-term gain or loss."),
            (RED, "High yields",
             "Yields above ~12% often mean a cut, return of capital, or a "
             "collapsed price. Costs and the ex-date gap usually eat a pure capture."),
            (PURPLE, "Calendar chips",
             "Chips mark the last session you can buy and still be on the "
             "books. Highest yield first."),
        ]
        for color, head, body in help_items:
            _html(
                f'<div class="dc-box" style="--bc:{color};margin-bottom:8px">'
                f'<b style="color:{color}">{head}</b><br>{body}</div>'
            )
        _sechead("Full filtered universe", GREEN,
                 f"{len(df):,} of {len(df_all):,}")
        if df.empty:
            st.info("No dividend stocks match the current filters. Set filters on Calendar first.")
        else:
            show = df.sort_values("yield_pct", ascending=False).copy()
            show["days_away"] = [
                (safe_date(bd) - today).days if safe_date(bd) else None
                for bd in show["buy_date"]]
            _render_dividend_table(show, today, show_buy_cols=True)


