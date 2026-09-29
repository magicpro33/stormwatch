"""
Crypto Copycat — live Hyperliquid buys and sells from top earners.

Public data only (no key, no orders):
  - Leaderboard snapshot: https://stats-data.hyperliquid.xyz/Mainnet/leaderboard
  - Vault list (the exchange's copy product): .../Mainnet/vaults
  - Live positions and fills: POST https://api.hyperliquid.xyz/info

Watching a fill means the trade already happened. Vaults are how Hyperliquid
lets someone else participate in a leader's book: deposit USDC, the leader
trades the pool. This tab links there. It does not place trades.
"""
from __future__ import annotations

import html as _html
import json
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

INFO = "https://api.hyperliquid.xyz/info"
LEADERBOARD_URL = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
VAULTS_URL = "https://stats-data.hyperliquid.xyz/Mainnet/vaults"
EXPLORER = "https://app.hyperliquid.xyz/explorer/address/"
VAULT_PAGE = "https://app.hyperliquid.xyz/vaults/"
BOARD_PAGE = "https://app.hyperliquid.xyz/leaderboard"
HYPURRSCAN = "https://hypurrscan.io/address/"

# Builder-deployed perp markets. The default clearinghouse call misses them.
HIP3_DEXES = ("xyz", "flx", "vntl", "hyna", "km", "abcd", "cash", "para", "mkts", "io")

WINDOW_KEYS = {
    "Day": "day",
    "Week": "week",
    "Month": "month",
    "All-time": "allTime",
}
LOOKBACKS = {
    "15 minutes": 15,
    "1 hour": 60,
    "4 hours": 240,
    "24 hours": 1440,
}
MIN_ACCOUNTS = {
    "$100k": 100_000,
    "$250k": 250_000,
    "$1M": 1_000_000,
    "$5M": 5_000_000,
}

ACCENT = "#E87722"
GREEN = "#3fbf7f"
RED = "#e05252"
DIM = "#9aa8bd"
INK = "#F6F4E9"
ET = ZoneInfo("America/New_York")
ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

_HYPER: dict[tuple[str, str], float] = {}
_HYPER_LOCK = threading.Lock()
_UA = {"User-Agent": "MoneyWeather/crypto-copycat"}


def _esc(value) -> str:
    return _html.escape("" if value is None else str(value), quote=True)


def _html_block(markup: str) -> None:
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup.replace("$", "&#36;"), unsafe_allow_html=True)


def _num(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _money(value, signed=False) -> str:
    v = _num(value)
    if v is None:
        return "—"
    sign = ""
    if signed:
        sign = "+" if v > 0 else ("−" if v < 0 else "")
    a = abs(v)
    if a >= 1e9:
        body = f"${a / 1e9:.2f}B"
    elif a >= 1e6:
        body = f"${a / 1e6:.2f}M"
    elif a >= 1e3:
        body = f"${a / 1e3:.1f}k"
    else:
        body = f"${a:.2f}"
    return sign + body


def _pct(frac) -> str:
    v = _num(frac)
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%"


def _px(value) -> str:
    v = _num(value)
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.2f}"
    if a >= 1:
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    text = f"{v:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _size(value) -> str:
    v = _num(value)
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.2f}"
    if a >= 1:
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _short(addr: str) -> str:
    a = (addr or "").lower()
    if len(a) < 12:
        return a or "—"
    return f"{a[:6]}…{a[-4:]}"


def _coin_parts(coin: str) -> tuple[str, str]:
    raw = str(coin or "")
    if ":" in raw:
        market, sym = raw.split(":", 1)
        return sym or raw, market or "perp"
    return raw or "—", "perp"


def _fmt_time(ms: int) -> str:
    try:
        dt = datetime.fromtimestamp(int(ms) / 1000, ET)
    except (TypeError, ValueError, OSError):
        return "—"
    return dt.strftime("%b %d %H:%M:%S ET")


def _age_label(ms) -> str:
    created = _num(ms)
    if not created:
        return "—"
    days = (time.time() * 1000 - created) / 86_400_000
    if days < 1:
        return "<1d"
    if days < 365:
        return f"{days:.0f}d"
    return f"{days / 365:.1f}y"


def _info(body: dict, timeout: int = 18):
    r = requests.post(INFO, json=body, headers=_UA, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError("Hyperliquid rate limit — live refresh will retry shortly.")
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data


def _get_json(url: str, timeout: int = 60):
    last = None
    for i in range(2):
        try:
            r = requests.get(url, headers=_UA, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if i == 0:
                time.sleep(0.6)
    raise last


def _window_map(pairs) -> dict:
    out = {}
    for pair in pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        stats = pair[1] if isinstance(pair[1], dict) else {}
        out[str(pair[0])] = {
            "pnl": _num(stats.get("pnl")) or 0.0,
            "roi": _num(stats.get("roi")) or 0.0,
            "vlm": _num(stats.get("vlm")) or 0.0,
        }
    return out


def load_leaderboard() -> dict:
    """Slim the ~40MB leaderboard down to the fields this tab ranks on."""
    payload = _get_json(LEADERBOARD_URL, timeout=60)
    rows_in = payload.get("leaderboardRows") if isinstance(payload, dict) else None
    if not isinstance(rows_in, list):
        raise RuntimeError("Leaderboard response had no leaderboardRows.")
    rows = []
    for row in rows_in:
        if not isinstance(row, dict):
            continue
        addr = str(row.get("ethAddress") or "").lower()
        if not ADDR_RE.match(addr):
            continue
        windows = _window_map(row.get("windowPerformances"))
        item = {
            "address": addr,
            "name": str(row.get("displayName") or "").strip(),
            "account": _num(row.get("accountValue")) or 0.0,
        }
        for key in ("day", "week", "month", "allTime"):
            stats = windows.get(key) or {}
            item[f"{key}_pnl"] = stats.get("pnl") or 0.0
            item[f"{key}_roi"] = stats.get("roi") or 0.0
            item[f"{key}_vlm"] = stats.get("vlm") or 0.0
        rows.append(item)
    return {"rows": rows, "fetched_at": time.time()}


def load_vaults() -> dict:
    payload = _get_json(VAULTS_URL, timeout=60)
    if not isinstance(payload, list):
        raise RuntimeError("Vault list was not a JSON array.")
    rows = []
    for vault in payload:
        if not isinstance(vault, dict):
            continue
        summary = vault.get("summary") if isinstance(vault.get("summary"), dict) else {}
        if summary.get("isClosed"):
            continue
        rel = summary.get("relationship") if isinstance(summary.get("relationship"), dict) else {}
        kind = str(rel.get("type") or "normal")
        if kind == "child":
            continue
        addr = str(summary.get("vaultAddress") or "").lower()
        if not ADDR_RE.match(addr):
            continue
        pnls = vault.get("pnls") if isinstance(vault.get("pnls"), list) else []
        series = {}
        for pair in pnls:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2 and pair[1]:
                series[str(pair[0])] = _num(pair[1][-1]) or 0.0
        rows.append({
            "name": str(summary.get("name") or "").strip() or _short(addr),
            "address": addr,
            "leader": str(summary.get("leader") or "").lower(),
            "tvl": _num(summary.get("tvl")) or 0.0,
            "apr": _num(vault.get("apr")) or 0.0,
            "created": _num(summary.get("createTimeMillis")) or 0.0,
            "kind": "protocol" if kind == "parent" else "leader",
            "day_pnl": series.get("day") or 0.0,
            "week_pnl": series.get("week") or 0.0,
            "month_pnl": series.get("month") or 0.0,
            "all_pnl": series.get("allTime") or 0.0,
        })
    return {"rows": rows, "fetched_at": time.time()}


def rank_earners(rows: list, window: str, by: str, min_account: float,
                 max_turn: float, n: int) -> list:
    """Top positive-PnL wallets, after dropping dust and market-maker turnover."""
    key = WINDOW_KEYS.get(window, "week")
    ranked = []
    for row in rows:
        account = row.get("account") or 0.0
        pnl = row.get(f"{key}_pnl") or 0.0
        vlm = row.get(f"{key}_vlm") or 0.0
        if account < min_account or pnl <= 0 or vlm < 10_000:
            continue
        turn = vlm / account if account else 999.0
        if turn > max_turn:
            continue
        ranked.append({**row, "turn": turn, "rank_pnl": pnl, "rank_roi": row.get(f"{key}_roi") or 0.0})
    metric = "rank_roi" if by == "ROI %" else "rank_pnl"
    ranked.sort(key=lambda r: (r[metric], r["rank_pnl"], r["address"]), reverse=True)
    return ranked[: max(1, int(n))]


def _book_active(state) -> bool:
    if not isinstance(state, dict):
        return False
    equity = _num((state.get("marginSummary") or {}).get("accountValue")) or 0.0
    if equity > 1:
        return True
    for item in state.get("assetPositions") or []:
        szi = _num(((item or {}).get("position") or {}).get("szi"))
        if szi and abs(szi) > 0:
            return True
    return False


def _positions_from(state, market: str) -> tuple[list, float]:
    if not isinstance(state, dict):
        return [], 0.0
    equity = _num((state.get("marginSummary") or {}).get("accountValue")) or 0.0
    out = []
    for item in state.get("assetPositions") or []:
        pos = (item or {}).get("position") or {}
        szi = _num(pos.get("szi"))
        if not szi:
            continue
        sym, embedded = _coin_parts(str(pos.get("coin") or ""))
        lev = pos.get("leverage") if isinstance(pos.get("leverage"), dict) else {}
        out.append({
            "coin": sym,
            "market": embedded if embedded != "perp" else (market or "perp"),
            "szi": szi,
            "side": "Long" if szi > 0 else "Short",
            "entry": _num(pos.get("entryPx")),
            "notional": abs(_num(pos.get("positionValue")) or 0.0),
            "upnl": _num(pos.get("unrealizedPnl")),
            "liq": _num(pos.get("liquidationPx")),
            "lev": lev.get("value"),
        })
    return out, equity


def _pull_fills(addr: str, lookback_ms: int, dex: str = "") -> tuple[list, bool]:
    """Recent fills. A 2,000-row cap is the oldest slice, so hyperactive books are dropped."""
    key = (addr, dex or "")
    now_ms = int(time.time() * 1000)
    window = lookback_ms
    with _HYPER_LOCK:
        if _HYPER.get(key, 0) > time.time():
            window = min(window, 10 * 60 * 1000)

    def pull(ms: int):
        body = {
            "type": "userFillsByTime",
            "user": addr,
            "startTime": now_ms - ms,
            "aggregateByTime": True,
        }
        if dex:
            body["dex"] = dex
        data = _info(body)
        return data if isinstance(data, list) else []

    data = pull(window)
    if len(data) >= 2000 and window > 10 * 60 * 1000:
        data = pull(10 * 60 * 1000)
    if len(data) >= 2000:
        with _HYPER_LOCK:
            _HYPER[key] = time.time() + 1800
        return [], True
    return data, False


def _parse_fill(raw: dict, addr: str, market: str) -> dict | None:
    if not isinstance(raw, dict):
        return None
    px = _num(raw.get("px"))
    sz = _num(raw.get("sz"))
    if px is None or sz is None:
        return None
    sym, embedded = _coin_parts(str(raw.get("coin") or ""))
    buy = raw.get("side") == "B"
    return {
        "time": int(raw.get("time") or 0),
        "tid": raw.get("tid"),
        "address": addr,
        "coin": sym,
        "market": embedded if embedded != "perp" else (market or "perp"),
        "side": "Buy" if buy else "Sell",
        "dir": str(raw.get("dir") or ""),
        "px": px,
        "sz": sz,
        "notional": abs(px * sz),
        "closed_pnl": _num(raw.get("closedPnl")),
    }


def _one_wallet(addr: str, extra_dexes: tuple, lookback_ms: int) -> dict:
    positions = []
    equity = 0.0
    fills = []
    truncated = False
    errors = []
    markets = [""] + [d for d in extra_dexes if d]
    for dex in markets:
        label = dex or "perp"
        body = {"type": "clearinghouseState", "user": addr}
        if dex:
            body["dex"] = dex
        try:
            state = _info(body)
            pos, eq = _positions_from(state, label)
            positions.extend(pos)
            equity += eq
        except Exception as exc:
            errors.append(f"{label} book: {exc}")
        try:
            raw_fills, cut = _pull_fills(addr, lookback_ms, dex)
            truncated = truncated or cut
            for raw in raw_fills:
                parsed = _parse_fill(raw, addr, label)
                if parsed:
                    fills.append(parsed)
        except Exception as exc:
            errors.append(f"{label} fills: {exc}")
    deduped = []
    seen = set()
    for fill in fills:
        key = (fill.get("tid"), fill["coin"], fill["time"], fill["side"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fill)
    deduped.sort(key=lambda f: f["time"])
    if len(deduped) > 60:
        deduped = deduped[-60:]
    return {
        "address": addr,
        "positions": positions,
        "equity": equity,
        "fills": deduped,
        "truncated": truncated,
        "error": "; ".join(errors[:3]),
    }


def scan_extra_dexes(addresses: tuple) -> dict:
    """Which HIP-3 markets actually hold risk for these wallets. Cached by the caller."""
    def scan(addr: str):
        hit = []
        for dex in HIP3_DEXES:
            try:
                state = _info({"type": "clearinghouseState", "user": addr, "dex": dex}, timeout=12)
            except Exception:
                continue
            if _book_active(state):
                hit.append(dex)
        return addr, tuple(hit)

    found = {addr: tuple() for addr in addresses}
    if not addresses:
        return found
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(scan, addr) for addr in addresses]
        for fut in as_completed(futures):
            addr, dexes = fut.result()
            found[addr] = dexes
    return found


def fetch_books(addresses: tuple, dex_map: dict, lookback_min: int) -> dict:
    lookback_ms = int(lookback_min) * 60 * 1000
    books = {}
    if not addresses:
        return books
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_one_wallet, addr, tuple(dex_map.get(addr) or ()), lookback_ms): addr
            for addr in addresses
        }
        for fut in as_completed(futures):
            addr = futures[fut]
            try:
                books[addr] = fut.result()
            except Exception as exc:
                books[addr] = {
                    "address": addr, "positions": [], "equity": 0.0,
                    "fills": [], "truncated": False, "error": str(exc),
                }
    return books


@st.cache_data(ttl=3600, show_spinner="Loading the Hyperliquid leaderboard…")
def _leaderboard(nonce: int) -> dict:
    _ = nonce
    return load_leaderboard()


@st.cache_data(ttl=1800, show_spinner="Loading Hyperliquid vaults…")
def _vaults(nonce: int) -> dict:
    _ = nonce
    return load_vaults()


@st.cache_data(ttl=600, show_spinner="Checking other Hyperliquid markets…")
def _dex_map(addresses: tuple) -> dict:
    return scan_extra_dexes(addresses)


@st.cache_data(ttl=20, show_spinner=False)
def _books(addresses: tuple, dex_sig: str, lookback_min: int) -> dict:
    dex_map = json.loads(dex_sig) if dex_sig else {}
    return fetch_books(addresses, dex_map, lookback_min)


def _trader_label(row: dict) -> str:
    name = (row.get("name") or "").strip()
    return name if name else _short(row.get("address") or "")


def _sign_color(value, pos_good=True) -> str:
    v = _num(value)
    if v is None or abs(v) < 1e-9:
        return DIM
    good = v > 0 if pos_good else v < 0
    return GREEN if good else RED


def _chips(items: list[tuple[str, str, str]]) -> None:
    cells = []
    for label, value, color in items:
        cells.append(
            "<div style='background:#0c1829;border:1px solid #1d2b40;border-radius:10px;"
            "padding:8px 12px;min-width:120px;'>"
            f"<div style='color:{DIM};font-size:11px;letter-spacing:.04em;'>{_esc(label)}</div>"
            f"<div style='color:{color};font-weight:750;font-size:18px;'>{_esc(value)}</div>"
            "</div>"
        )
    _html_block(
        "<div style='display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px;'>"
        + "".join(cells) + "</div>"
    )


def _table(headers: list[str], body_rows: list[str], max_h: int = 560) -> None:
    ths = "".join(
        f"<th style='position:sticky;top:0;background:#122a42;color:{DIM};text-align:left;"
        f"padding:8px;font-weight:650;white-space:nowrap;'>{_esc(h)}</th>"
        for h in headers
    )
    _html_block(
        f"<div style='max-height:{int(max_h)}px;overflow:auto;border:1px solid #1d2b40;"
        f"border-radius:10px;margin:6px 0 12px;'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        f"<thead><tr>{ths}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _td(text, color=INK, align="left", bold=False) -> str:
    weight = "700" if bold else "500"
    return (
        f"<td style='padding:6px 8px;border-top:1px solid #1d2b40;color:{color};"
        f"text-align:{align};font-weight:{weight};white-space:nowrap;'>{text}</td>"
    )


def _flow_chart(fills: list, title: str) -> None:
    nets = defaultdict(float)
    for fill in fills:
        signed = fill["notional"] if fill["side"] == "Buy" else -fill["notional"]
        nets[fill["coin"]] += signed
    if not nets:
        return
    ranked = sorted(nets.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
    coins = [c for c, _ in ranked]
    vals = [v for _, v in ranked]
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=coins,
        y=vals,
        marker_color=[GREEN if v >= 0 else RED for v in vals],
        hovertemplate="%{x}<br>%{y:$,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=INK)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        height=300,
        margin=dict(l=8, r=8, t=40, b=8),
        yaxis=dict(gridcolor="#1d2b40", zerolinecolor="#4a7aa0", tickprefix="$", separatethousands=True),
        xaxis=dict(gridcolor="#1d2b40"),
    )
    st.plotly_chart(fig, width="stretch")


def _position_nets(positions: list) -> list[tuple[str, float, int, int]]:
    """(coin, signed notional, n_long, n_short) across followed wallets."""
    bucket = defaultdict(lambda: {"net": 0.0, "long": 0, "short": 0})
    for pos in positions:
        signed = pos["notional"] if pos["side"] == "Long" else -pos["notional"]
        slot = bucket[pos["coin"]]
        slot["net"] += signed
        if pos["side"] == "Long":
            slot["long"] += 1
        else:
            slot["short"] += 1
    rows = [(coin, v["net"], v["long"], v["short"]) for coin, v in bucket.items()]
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    return rows


def _section(options: list[str], key: str) -> str:
    if hasattr(st, "segmented_control"):
        sel = st.segmented_control("Section", options, key=key, label_visibility="collapsed")
        return sel if sel in options else options[0]
    return st.radio("Section", options, horizontal=True, key=key, label_visibility="collapsed")


def _header() -> None:
    _html_block(
        f"""
        <div style="background:#0c1829;border:1px solid #1d2b40;border-left:4px solid {ACCENT};
            border-radius:10px;padding:12px 14px;margin-bottom:10px;">
          <div style="font-weight:750;font-size:18px;color:{INK};">Crypto Copycat</div>
          <div style="color:{DIM};font-size:13px;margin-top:4px;line-height:1.45;">
            Live buys and sells from Hyperliquid wallets that are actually up, plus the
            vaults Hyperliquid uses for real copy-trading. The leaderboard snapshot
            refreshes about hourly. Positions and fills refresh about every 25 seconds.
            A fill you can see has already printed — you do not get their entry.
            <br>Sources:
            <a href="{BOARD_PAGE}" target="_blank" rel="noopener" style="color:{ACCENT};">leaderboard</a>
            ·
            <a href="https://app.hyperliquid.xyz/vaults" target="_blank" rel="noopener" style="color:{ACCENT};">vaults</a>
            · public info API, no key.
            Research only. Not investment advice.
          </div>
        </div>
        """
    )


def _tape_controls() -> None:
    c1, c2, c3, c4, c5 = st.columns([1.1, 1, 1.1, 1.2, 0.7])
    c1.selectbox(
        "Rank window", list(WINDOW_KEYS), key="cc_window",
        help="Whose PnL counts as a top earner. Day is who is hot right now. "
             "Week and month are the slower 'proven' cut. The leaderboard itself is an hourly snapshot.",
    )
    c2.selectbox(
        "Rank by", ["PnL $", "ROI %"], key="cc_by",
        help="PnL $ is dollars earned in that window. ROI % is the return on account value — "
             "a smaller account can outrank a larger one.",
    )
    c3.selectbox(
        "Min account", list(MIN_ACCOUNTS), index=0, key="cc_min_acct",
        help="Drops tiny accounts whose percentage gain is one lucky fill.",
    )
    c4.selectbox(
        "Tape window", list(LOOKBACKS), index=1, key="cc_lookback",
        help="How far back the buy/sell tape looks. Hyperliquid returns at most 2,000 fills "
             "per request, and that cap is the oldest slice — hyperactive wallets are left off the tape.",
    )
    c5.toggle("Live", value=True, key="cc_live", help="Refresh positions and fills about every 25 seconds.")

    d1, d2, d3, d4, d5 = st.columns([0.9, 1.3, 0.8, 1.1, 1.3])
    d1.slider("Wallets", 3, 8, 6, key="cc_n", help="How many ranked wallets to follow. Capped at 8 to stay inside Hyperliquid's public rate limit.")
    d2.slider(
        "Max volume / equity", 0.5, 15.0, 3.0, step=0.5, key="cc_turn",
        help="Drops market makers. A wallet whose window volume is many times its equity "
             "is scalping both sides — those fills are not a copyable book.",
    )
    d3.selectbox("Side", ["All", "Buys", "Sells"], key="cc_side",
                 help="Filters the tape. Open positions stay listed either way.")
    d4.text_input("Coin filter", key="cc_coin", placeholder="BTC, ETH…",
                  help="Optional. Limits the tape and the open-book list to coins whose name contains this text.")
    d5.text_input("Also watch", key="cc_extra", placeholder="0x wallet",
                  help="One extra address, followed alongside the ranked list.")
    b1, b2 = st.columns([1, 4])
    if b1.button("Reload leaderboard", key="cc_reload_board"):
        st.session_state["cc_board_nonce"] = int(st.session_state.get("cc_board_nonce") or 0) + 1


def _vault_controls() -> None:
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 1])
    c1.selectbox("Min TVL", ["$100k", "$250k", "$1M", "$5M"], index=1, key="cc_v_tvl")
    c2.slider("Min age (days)", 0, 180, 45, key="cc_v_age",
              help="A 2,000% APR on a vault opened last week is not a track record.")
    c3.selectbox("Sort by", ["Month PnL", "APR", "TVL"], key="cc_v_sort")
    c4.slider("Show", 10, 40, 20, key="cc_v_n")
    p1, p2 = st.columns([1.4, 3])
    p1.checkbox("Only positive month PnL", value=True, key="cc_v_pos")
    if p2.button("Reload vaults", key="cc_reload_vaults"):
        st.session_state["cc_vault_nonce"] = int(st.session_state.get("cc_vault_nonce") or 0) + 1


def _as_earner(row: dict, window: str, watched: bool = False) -> dict:
    key = WINDOW_KEYS.get(window, "day")
    account = row.get("account") or 0.0
    vlm = row.get(f"{key}_vlm") or 0.0
    return {
        **row,
        "turn": (vlm / account) if account else 0.0,
        "rank_pnl": row.get(f"{key}_pnl") or 0.0,
        "rank_roi": row.get(f"{key}_roi") or 0.0,
        "watched": watched,
    }


def _blank_earner(addr: str) -> dict:
    item = {
        "address": addr, "name": "", "account": 0.0, "watched": True, "turn": 0.0,
        "rank_pnl": 0.0, "rank_roi": 0.0,
    }
    for key in ("day", "week", "month", "allTime"):
        item[f"{key}_pnl"] = 0.0
        item[f"{key}_roi"] = 0.0
        item[f"{key}_vlm"] = 0.0
    return item


def _selected_addresses(earners: list) -> list[str]:
    addrs = [r["address"] for r in earners]
    extra = str(st.session_state.get("cc_extra") or "").strip().lower()
    if extra:
        if not ADDR_RE.match(extra):
            st.caption("Also-watch needs a 42-character 0x address. That field is being ignored.")
        elif extra not in addrs:
            addrs.append(extra)
    return addrs


def _build_live() -> dict:
    nonce = int(st.session_state.get("cc_board_nonce") or 0)
    board = _leaderboard(nonce)
    window = st.session_state.get("cc_window") or "Day"
    by = st.session_state.get("cc_by") or "PnL $"
    min_account = MIN_ACCOUNTS.get(st.session_state.get("cc_min_acct") or "$100k", 100_000)
    max_turn = float(st.session_state.get("cc_turn") or 3.0)
    n = int(st.session_state.get("cc_n") or 6)
    earners = [_as_earner(r, window) for r in rank_earners(board["rows"], window, by, min_account, max_turn, n)]
    addresses = _selected_addresses(earners)
    known = {r["address"] for r in earners}
    extra = addresses[-1] if addresses and addresses[-1] not in known else ""
    if extra:
        by_addr = {r["address"]: r for r in board["rows"]}
        src = by_addr.get(extra)
        earners = earners + ([_as_earner(src, window, watched=True)] if src else [_blank_earner(extra)])
    dexes = _dex_map(tuple(sorted(addresses))) if addresses else {}
    lookback_label = st.session_state.get("cc_lookback") or "1 hour"
    lookback_min = LOOKBACKS.get(lookback_label, 60)
    books = _books(tuple(addresses), json.dumps(dexes, sort_keys=True), lookback_min) if addresses else {}
    return {
        "board_at": board["fetched_at"],
        "n_board": len(board["rows"]),
        "window": window,
        "earners": earners,
        "books": books,
        "lookback": lookback_label,
        "fetched_at": time.time(),
    }


def _coin_ok(coin: str) -> bool:
    needle = str(st.session_state.get("cc_coin") or "").strip().upper()
    if not needle:
        return True
    return needle in str(coin or "").upper()


def _fill_ok(fill: dict) -> bool:
    if not _coin_ok(fill.get("coin")):
        return False
    side = st.session_state.get("cc_side") or "All"
    if side == "Buys":
        return fill.get("side") == "Buy"
    if side == "Sells":
        return fill.get("side") == "Sell"
    return True


def _draw_roster(payload: dict) -> None:
    key = WINDOW_KEYS.get(payload["window"], "day")
    rows = []
    for i, earner in enumerate(payload["earners"], start=1):
        book = payload["books"].get(earner["address"]) or {}
        fills = [f for f in book.get("fills") or [] if _fill_ok(f)]
        positions = [p for p in book.get("positions") or [] if _coin_ok(p["coin"])]
        if book.get("truncated"):
            tape = "suppressed"
            tape_color = ACCENT
        elif book.get("error") and not fills:
            tape = "error"
            tape_color = RED
        else:
            tape = str(len(fills))
            tape_color = INK
        label = _esc(_trader_label(earner))
        if earner.get("watched"):
            label += f" <span style='color:{ACCENT};font-size:11px;'>watched</span>"
        addr = earner["address"]
        links = (
            f"<a href='{EXPLORER}{addr}' target='_blank' rel='noopener' style='color:{ACCENT};'>explorer</a>"
            f" · <a href='{HYPURRSCAN}{addr}' target='_blank' rel='noopener' style='color:{DIM};'>hypurrscan</a>"
        )
        pnl = earner.get(f"{key}_pnl") or 0.0
        roi = earner.get(f"{key}_roi") or 0.0
        rows.append(
            "<tr>"
            + _td(str(i), DIM)
            + _td(label, INK, bold=True)
            + _td(links)
            + _td(_esc(_money(earner.get("account"))), align="right")
            + _td(_esc(_money(book.get("equity"))), align="right")
            + _td(_esc(_money(pnl, signed=True)), _sign_color(pnl), "right", True)
            + _td(_esc(_pct(roi)), _sign_color(roi), "right")
            + _td(_esc(_money(earner.get("month_pnl"), signed=True)), _sign_color(earner.get("month_pnl")), "right")
            + _td(_esc(f"{earner.get('turn') or 0:.2f}×"), DIM, "right")
            + _td(_esc(tape), tape_color, "right")
            + _td(str(len(positions)), INK, "right")
            + "</tr>"
        )
    if not rows:
        st.info("No wallets passed these filters. Lower the minimum account, or raise max volume / equity.")
        return
    _table(
        ["#", "Trader", "Links", "Board equity", "Live perp equity",
         f"{payload['window']} PnL", f"{payload['window']} ROI", "Month PnL",
         "Vol/equity", "Fills", "Open"],
        rows,
        max_h=360,
    )


def _draw_tape(payload: dict) -> None:
    names = {e["address"]: _trader_label(e) for e in payload["earners"]}
    fills = []
    suppressed = []
    for earner in payload["earners"]:
        book = payload["books"].get(earner["address"]) or {}
        if book.get("truncated"):
            suppressed.append(_trader_label(earner))
        for fill in book.get("fills") or []:
            if _fill_ok(fill):
                fills.append(fill)
    fills.sort(key=lambda f: f["time"], reverse=True)
    buy_n = sum(1 for f in fills if f["side"] == "Buy")
    sell_n = len(fills) - buy_n
    buy_not = sum(f["notional"] for f in fills if f["side"] == "Buy")
    sell_not = sum(f["notional"] for f in fills if f["side"] == "Sell")
    _chips([
        ("Wallets", str(len(payload["earners"])), INK),
        ("Buys", f"{buy_n} · {_money(buy_not)}", GREEN),
        ("Sells", f"{sell_n} · {_money(sell_not)}", RED),
        ("Net buy−sell", _money(buy_not - sell_not, signed=True), _sign_color(buy_not - sell_not)),
        ("Tape", payload["lookback"], DIM),
    ])
    if suppressed:
        st.caption(
            "Tape suppressed for "
            + ", ".join(suppressed)
            + " — more than 2,000 fills in a few minutes, so the public API would return stale prints. Their open book is still listed."
        )
    if fills:
        _flow_chart(fills, f"Net buy minus sell · {payload['lookback']}")
    else:
        st.caption("No fills in this window for the current filters. They may be holding. Open books shows the positions.")
    body = []
    for fill in fills[:250]:
        color = GREEN if fill["side"] == "Buy" else RED
        closed = fill.get("closed_pnl")
        body.append(
            "<tr>"
            + _td(_esc(_fmt_time(fill["time"])), DIM)
            + _td(_esc(names.get(fill["address"]) or _short(fill["address"])), INK, bold=True)
            + _td(_esc(fill["coin"]), INK, bold=True)
            + _td(_esc(fill["market"]), DIM)
            + _td(_esc(fill["side"]), color, bold=True)
            + _td(_esc(fill["dir"] or "—"), DIM)
            + _td(_esc(_size(fill["sz"])), align="right")
            + _td(_esc(_px(fill["px"])), align="right")
            + _td(_esc(_money(fill["notional"])), align="right")
            + _td(_esc(_money(closed, signed=True)), _sign_color(closed), "right")
            + "</tr>"
        )
    if body:
        _table(
            ["Time", "Trader", "Coin", "Market", "Side", "Action", "Size", "Price", "Notional", "Closed PnL"],
            body,
            max_h=640,
        )
        frame = pd.DataFrame([{
            "time_et": _fmt_time(f["time"]),
            "trader": names.get(f["address"]) or _short(f["address"]),
            "address": f["address"],
            "coin": f["coin"],
            "market": f["market"],
            "side": f["side"],
            "action": f["dir"],
            "size": f["sz"],
            "price": f["px"],
            "notional": f["notional"],
            "closed_pnl": f["closed_pnl"],
        } for f in fills[:250]])
        st.download_button(
            "Download tape CSV",
            frame.to_csv(index=False),
            "crypto_copycat_tape.csv",
            "text/csv",
            key="cc_tape_csv",
        )


def _draw_books(payload: dict, limit: int | None = None, chart: bool = True) -> None:
    positions = []
    names = {e["address"]: _trader_label(e) for e in payload["earners"]}
    for earner in payload["earners"]:
        book = payload["books"].get(earner["address"]) or {}
        for pos in book.get("positions") or []:
            if _coin_ok(pos["coin"]):
                positions.append({**pos, "address": earner["address"]})
    nets = _position_nets(positions)
    if chart and nets:
        _flow_chart(
            [{"coin": c, "side": "Buy" if v >= 0 else "Sell", "notional": abs(v)} for c, v, _, _ in nets],
            "Net open notional · long positive, short negative",
        )
    if nets:
        agree = []
        for coin, net, n_long, n_short in nets[:8]:
            if n_long + n_short < 2:
                continue
            bias = "long" if net >= 0 else "short"
            agree.append(f"{coin} {n_long} long / {n_short} short, net {_money(net, signed=True)} {bias}")
        if agree:
            st.caption("Agreement (2+ wallets): " + " · ".join(agree))
    if not positions:
        if chart:
            st.info("None of the followed wallets have an open perp on the markets we can see.")
        else:
            st.caption("No open perps on the markets we can see for this filter.")
        return
    positions.sort(key=lambda p: p["notional"], reverse=True)
    hidden = 0
    if limit is not None and len(positions) > limit:
        hidden = len(positions) - limit
        positions = positions[:limit]
    body = []
    for pos in positions:
        color = GREEN if pos["side"] == "Long" else RED
        lev = pos.get("lev")
        lev_txt = "—" if lev in (None, "") else f"{lev}×"
        body.append(
            "<tr>"
            + _td(_esc(names.get(pos["address"]) or _short(pos["address"])), INK, bold=True)
            + _td(_esc(pos["coin"]), INK, bold=True)
            + _td(_esc(pos["market"]), DIM)
            + _td(_esc(pos["side"]), color, bold=True)
            + _td(_esc(_size(abs(pos["szi"]))), align="right")
            + _td(_esc(_px(pos["entry"])), align="right")
            + _td(_esc(_money(pos["notional"])), align="right")
            + _td(_esc(_money(pos["upnl"], signed=True)), _sign_color(pos["upnl"]), "right", True)
            + _td(_esc(lev_txt), DIM, "right")
            + _td(_esc(_px(pos["liq"])), align="right")
            + "</tr>"
        )
    _table(
        ["Trader", "Coin", "Market", "Side", "Size", "Entry", "Notional", "uPnL", "Lev", "Liq"],
        body,
        max_h=420 if limit else 640,
    )
    if hidden:
        st.caption(f"{hidden} smaller positions are on the Open books view.")


def _render_live() -> None:
    live = bool(st.session_state.get("cc_live", True))
    if live or "cc_payload" not in st.session_state:
        try:
            payload = _build_live()
        except Exception as exc:
            st.error(f"Hyperliquid data did not load: {exc}")
            payload = st.session_state.get("cc_payload")
            if not payload:
                return
        else:
            st.session_state["cc_payload"] = payload
    else:
        payload = st.session_state["cc_payload"]

    age_min = max(0, int((time.time() - payload["board_at"]) / 60))
    stamp = _fmt_time(int(payload["fetched_at"] * 1000))
    st.caption(
        f"Leaderboard: {payload['n_board']:,} wallets, snapshot {age_min} min ago. "
        f"Books read at {stamp}."
        + ("" if live else " Live refresh is paused.")
    )
    errors = [b.get("error") for b in payload["books"].values() if b.get("error")]
    if errors:
        st.caption("Partial read: " + " · ".join(errors[:3]))

    view = st.session_state.get("cc_view") or "Earner tape"
    _draw_roster(payload)
    if view == "Open books":
        st.subheader("Open books")
        _draw_books(payload)
    else:
        st.subheader("Buys and sells")
        _draw_tape(payload)
        st.subheader("Largest open bets")
        _draw_books(payload, limit=8, chart=False)


@st.fragment(run_every=timedelta(seconds=25))
def _live_fragment() -> None:
    _render_live()


def _render_vaults() -> None:
    _html_block(
        f"""
        <div style="background:#0c1829;border:1px solid #1d2b40;border-radius:10px;
            padding:10px 14px;margin:4px 0 10px;color:{DIM};font-size:13px;line-height:1.45;">
          <span style="color:{INK};font-weight:700;">Better copy path: Hyperliquid vaults.</span>
          Depositing into a vault is the exchange's copy product — the leader trades the
          pooled margin and your share follows the vault PnL, after the fee disclosed on
          the vault page. That beats mirroring a fill that already happened.
          Filters below drop closed vaults, brand-new vaults, and tiny TVL.
          APR is what Hyperliquid publishes; a huge APR on a young vault is a hot streak, not proof.
          The protocol vault (HLP) is Hyperliquid's own market-making pool, tagged when it passes the filters.
        </div>
        """
    )
    nonce = int(st.session_state.get("cc_vault_nonce") or 0)
    try:
        blob = _vaults(nonce)
    except Exception as exc:
        st.error(f"Vault list did not load: {exc}")
        return
    min_tvl = {"$100k": 100_000, "$250k": 250_000, "$1M": 1_000_000, "$5M": 5_000_000}.get(
        st.session_state.get("cc_v_tvl") or "$250k", 250_000)
    min_age = float(st.session_state.get("cc_v_age") or 0)
    only_pos = bool(st.session_state.get("cc_v_pos", True))
    sort = st.session_state.get("cc_v_sort") or "Month PnL"
    limit = int(st.session_state.get("cc_v_n") or 20)
    now_ms = time.time() * 1000
    picked = []
    for row in blob["rows"]:
        if row["tvl"] < min_tvl:
            continue
        age_days = (now_ms - row["created"]) / 86_400_000 if row["created"] else 0
        if age_days < min_age:
            continue
        if only_pos and row["month_pnl"] <= 0:
            continue
        picked.append({**row, "age_days": age_days, "month_roi": (row["month_pnl"] / row["tvl"]) if row["tvl"] else 0.0})
    sort_key = {"APR": "apr", "TVL": "tvl"}.get(sort, "month_pnl")
    picked.sort(key=lambda r: r[sort_key], reverse=True)
    shown = picked[:limit]
    age_min = max(0, int((time.time() - blob["fetched_at"]) / 60))
    st.caption(f"{len(picked):,} vaults passed the filters. List refreshed {age_min} min ago. Showing {len(shown)}.")
    if not shown:
        st.info("No vaults passed. Lower the minimum TVL or age, or turn off the positive-month filter.")
        return
    body = []
    for row in shown:
        kind = "protocol" if row["kind"] == "protocol" else "leader"
        name = _esc(row["name"])
        if kind == "protocol":
            name += f" <span style='color:{ACCENT};font-size:11px;'>protocol</span>"
        link = (
            f"<a href='{VAULT_PAGE}{row['address']}' target='_blank' rel='noopener' "
            f"style='color:{ACCENT};'>open vault</a>"
        )
        body.append(
            "<tr>"
            + _td(name, INK, bold=True)
            + _td(link)
            + _td(_esc(kind), DIM)
            + _td(_esc(_money(row["tvl"])), align="right")
            + _td(_esc(_money(row["month_pnl"], signed=True)), _sign_color(row["month_pnl"]), "right", True)
            + _td(_esc(_pct(row["month_roi"])), _sign_color(row["month_roi"]), "right")
            + _td(_esc(f"{row['apr'] * 100:.1f}%"), _sign_color(row["apr"]), "right")
            + _td(_esc(_money(row["all_pnl"], signed=True)), _sign_color(row["all_pnl"]), "right")
            + _td(_esc(_age_label(row["created"])), DIM, "right")
            + _td(_esc(_short(row["leader"])), DIM)
            + "</tr>"
        )
    _table(
        ["Vault", "Link", "Type", "TVL", "Month PnL", "Month / TVL", "APR", "All-time PnL", "Age", "Leader"],
        body,
        max_h=680,
    )
    frame = pd.DataFrame([{
        "name": r["name"],
        "vault": r["address"],
        "leader": r["leader"],
        "type": r["kind"],
        "tvl": r["tvl"],
        "month_pnl": r["month_pnl"],
        "month_roi": r["month_roi"],
        "apr": r["apr"],
        "all_time_pnl": r["all_pnl"],
        "age_days": r["age_days"],
        "url": VAULT_PAGE + r["address"],
    } for r in shown])
    st.download_button(
        "Download vault CSV",
        frame.to_csv(index=False),
        "crypto_copycat_vaults.csv",
        "text/csv",
        key="cc_vault_csv",
    )


def render_crypto_copycat_tab() -> None:
    """Money Maker entry point."""
    _header()
    view = _section(["Earner tape", "Open books", "Copy vaults"], "cc_view")
    if view == "Copy vaults":
        _vault_controls()
        _render_vaults()
        return
    _tape_controls()
    _live_fragment()
