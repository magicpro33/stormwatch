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


def _as_dt(ms):
    try:
        return datetime.fromtimestamp(int(ms) / 1000, ET)
    except (TypeError, ValueError, OSError):
        return pd.NaT


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


def _ncol(help_text: str, fmt: str = "dollar"):
    return st.column_config.NumberColumn(help=help_text, format=fmt)


def _tcol(help_text: str):
    return st.column_config.TextColumn(help=help_text)


def _lcol(help_text: str, display: str):
    return st.column_config.LinkColumn(help=help_text, display_text=display)


def _dcol(help_text: str):
    return st.column_config.DatetimeColumn(help=help_text, format="MMM D, HH:mm:ss")


def _cell_sign(value) -> str:
    return f"color:{_sign_color(value)}"


def _side_css(value) -> str:
    if value in ("Buy", "Long"):
        return f"color:{GREEN};font-weight:600"
    if value in ("Sell", "Short"):
        return f"color:{RED};font-weight:600"
    return ""


def _grid(
    frame: pd.DataFrame,
    config: dict,
    *,
    height: int,
    key: str,
    default: str,
    default_asc: bool,
    sign_cols: tuple = (),
    side_col: str | None = None,
    column_order: list | None = None,
) -> pd.DataFrame:
    """Sortable table. Hover a heading for that column's description."""
    order = list(column_order or list(frame.columns))
    sort_key = f"{key}_sort"
    asc_key = f"{key}_asc"
    if st.session_state.get(sort_key) not in order:
        st.session_state[sort_key] = default if default in order else order[0]
    if asc_key not in st.session_state:
        st.session_state[asc_key] = bool(default_asc)
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox(
        "Sort by",
        order,
        key=sort_key,
        help="Orders this table, and the choice stays put when the live data refreshes. "
             "Hover a column heading to read what that column means.",
    )
    ascending = c2.toggle(
        "Ascending",
        key=asc_key,
        help="On puts the smallest, oldest, or A-to-Z first. Off puts the largest or newest first.",
    )
    ordered = frame.sort_values(
        choice, ascending=bool(ascending), na_position="last", kind="mergesort",
    )
    visible = ordered.loc[:, order]
    styled = visible.style
    paint = [c for c in sign_cols if c in visible.columns]
    if paint:
        styled = styled.map(_cell_sign, subset=paint)
    if side_col and side_col in visible.columns:
        styled = styled.map(_side_css, subset=[side_col])
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=min(int(height), 48 + 36 * max(len(visible), 1)),
        column_config=config,
        column_order=order,
        key=f"{key}_grid",
    )
    return ordered


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
    window = payload["window"]
    pnl_col = f"{window} PnL"
    roi_col = f"{window} ROI"
    rows = []
    for i, earner in enumerate(payload["earners"], start=1):
        book = payload["books"].get(earner["address"]) or {}
        fills = [f for f in book.get("fills") or [] if _fill_ok(f)]
        positions = [p for p in book.get("positions") or [] if _coin_ok(p["coin"])]
        if book.get("truncated") or (book.get("error") and not fills):
            fill_n = None
        else:
            fill_n = len(fills)
        label = _trader_label(earner)
        if earner.get("watched"):
            label += " · watched"
        addr = earner["address"]
        rows.append({
            "#": i,
            "Trader": label,
            "Explorer": EXPLORER + addr,
            "Hypurrscan": HYPURRSCAN + addr,
            "Board equity": earner.get("account"),
            "Live perp equity": book.get("equity"),
            pnl_col: earner.get(f"{key}_pnl") or 0.0,
            roi_col: (earner.get(f"{key}_roi") or 0.0) * 100,
            "Month PnL": earner.get("month_pnl") or 0.0,
            "Vol/equity": earner.get("turn") or 0.0,
            "Fills": fill_n,
            "Open": len(positions),
        })
    if not rows:
        st.info("No wallets passed these filters. Lower the minimum account, or raise max volume / equity.")
        return
    _grid(
        pd.DataFrame(rows),
        {
            "#": _ncol("Rank after the window, the PnL or ROI choice, the account minimum, and the volume filter. 1 is the top of that list.", "%d"),
            "Trader": _tcol("Display name, or a shortened address. A watched suffix means this wallet was added in Also watch."),
            "Explorer": _lcol("Hyperliquid's own page for this address.", "explorer"),
            "Hypurrscan": _lcol("The same address on Hypurrscan, for a deeper look at the wallet.", "hypurrscan"),
            "Board equity": _ncol("Account value on the leaderboard snapshot. That snapshot is about an hour old."),
            "Live perp equity": _ncol("Perp equity from the live position read. It can differ from board equity when margin sits on a book that read misses. Spot USDC is not added on top."),
            pnl_col: _ncol(f"Dollars gained or lost in the {window.lower()} leaderboard window."),
            roi_col: _ncol(f"That {window.lower()} profit divided by account value.", "+%.1f%%"),
            "Month PnL": _ncol("Dollars gained or lost over the last month, whichever rank window you picked."),
            "Vol/equity": _ncol("Window volume divided by account value. A high number is two-sided scalping. The max-volume filter uses this.", "%.2f×"),
            "Fills": _ncol("Buys and sells in the tape window that passed the coin and side filters. Blank means the tape was hidden because the wallet traded too much, or the read failed.", "%.0f"),
            "Open": _ncol("Open perp positions that passed the coin filter.", "%d"),
        },
        height=360,
        key="cc_roster",
        default="#",
        default_asc=True,
        sign_cols=(pnl_col, roi_col, "Month PnL"),
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
    shown = fills[:250]
    if not shown:
        return
    frame = pd.DataFrame([{
        "Time": _as_dt(f["time"]),
        "Trader": names.get(f["address"]) or _short(f["address"]),
        "Address": f["address"],
        "Coin": f["coin"],
        "Market": f["market"],
        "Side": f["side"],
        "Action": f["dir"] or "—",
        "Size": f["sz"],
        "Price": f["px"],
        "Notional": f["notional"],
        "Closed PnL": f["closed_pnl"],
    } for f in shown])
    ordered = _grid(
        frame,
        {
            "Time": _dcol("When the fill printed, New York time."),
            "Trader": _tcol("Which followed wallet printed this fill."),
            "Coin": _tcol("Market symbol, such as BTC or the name on a builder market."),
            "Market": _tcol("perp is the main book. Anything else is a builder market, such as xyz."),
            "Side": _tcol("Buy or sell. Buy is green, sell is red."),
            "Action": _tcol("Whether the fill opened a position, closed one, or flipped it. This is Hyperliquid's own label."),
            "Size": _ncol("Contracts or coins in the fill.", "plain"),
            "Price": _ncol("Fill price.", "plain"),
            "Notional": _ncol("Size times price, in dollars. Always positive. Side says which way."),
            "Closed PnL": _ncol("Realized profit on the part of the position this fill closed. Blank when the fill only opened."),
        },
        height=640,
        key="cc_tape",
        default="Time",
        default_asc=False,
        sign_cols=("Closed PnL",),
        side_col="Side",
        column_order=["Time", "Trader", "Coin", "Market", "Side", "Action", "Size", "Price", "Notional", "Closed PnL"],
    )
    st.download_button(
        "Download tape CSV",
        ordered.to_csv(index=False),
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
    frame = pd.DataFrame([{
        "Trader": names.get(pos["address"]) or _short(pos["address"]),
        "Coin": pos["coin"],
        "Market": pos["market"],
        "Side": pos["side"],
        "Size": abs(pos["szi"]) if pos.get("szi") is not None else None,
        "Entry": pos.get("entry"),
        "Notional": pos.get("notional"),
        "uPnL": pos.get("upnl"),
        "Lev": _num(pos.get("lev")),
        "Liq": pos.get("liq"),
    } for pos in positions])
    _grid(
        frame,
        {
            "Trader": _tcol("Which followed wallet holds this position."),
            "Coin": _tcol("Market symbol."),
            "Market": _tcol("perp is the main book. Anything else is a builder market, such as xyz."),
            "Side": _tcol("Long or short. Long is green, short is red."),
            "Size": _ncol("Absolute position size, in contracts or coins.", "plain"),
            "Entry": _ncol("Average entry price.", "plain"),
            "Notional": _ncol("Absolute position value in dollars. Side says long or short."),
            "uPnL": _ncol("Unrealized profit on the open position."),
            "Lev": _ncol("Leverage on this position.", "%.2f×"),
            "Liq": _ncol("Estimated liquidation price. Blank if the exchange did not send one.", "plain"),
        },
        height=420 if limit else 640,
        key="cc_books_top" if limit else "cc_books",
        default="Notional",
        default_asc=False,
        sign_cols=("uPnL",),
        side_col="Side",
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
    frame = pd.DataFrame([{
        "Vault": (row["name"] + " · protocol") if row["kind"] == "protocol" else row["name"],
        "Link": VAULT_PAGE + row["address"],
        "Type": "protocol" if row["kind"] == "protocol" else "leader",
        "TVL": row["tvl"],
        "Month PnL": row["month_pnl"],
        "Month / TVL": (row["month_roi"] or 0.0) * 100,
        "APR": (row["apr"] or 0.0) * 100,
        "All-time PnL": row["all_pnl"],
        "Age": row["age_days"],
        "Leader": _short(row["leader"]),
        "Vault address": row["address"],
        "Leader address": row["leader"],
    } for row in shown])
    vault_default = {"APR": "APR", "TVL": "TVL"}.get(sort, "Month PnL")
    ordered = _grid(
        frame,
        {
            "Vault": _tcol("Vault name. A protocol suffix is Hyperliquid's own pool (HLP)."),
            "Link": _lcol("The vault page. A deposit is made there, on Hyperliquid, not in this app.", "open vault"),
            "Type": _tcol("protocol is Hyperliquid's own market-making pool. leader is someone else's vault."),
            "TVL": _ncol("Dollars currently in the vault."),
            "Month PnL": _ncol("Dollars the vault made over the last month."),
            "Month / TVL": _ncol("Month profit divided by the money in the vault.", "+%.1f%%"),
            "APR": _ncol("Annualized return Hyperliquid publishes. A huge number on a young vault is a short hot streak.", "+%.1f%%"),
            "All-time PnL": _ncol("Profit since the vault opened."),
            "Age": _ncol("Days since the vault opened.", "%.0fd"),
            "Leader": _tcol("Shortened address of the wallet that trades the vault."),
        },
        height=680,
        key="cc_vaults",
        default=vault_default,
        default_asc=False,
        sign_cols=("Month PnL", "Month / TVL", "APR", "All-time PnL"),
        column_order=["Vault", "Link", "Type", "TVL", "Month PnL", "Month / TVL", "APR", "All-time PnL", "Age", "Leader"],
    )
    st.download_button(
        "Download vault CSV",
        ordered.to_csv(index=False),
        "crypto_copycat_vaults.csv",
        "text/csv",
        key="cc_vault_csv",
    )


def _render_helper() -> None:
    st.markdown("### ₿ Crypto Helper")
    st.caption(
        "How to read this screen. It does not place trades. "
        "On each table, Sort by orders any column and that order sticks when the data refreshes. "
        "Hover a column heading for what that column means."
    )
    st.markdown(
        "Crypto Copycat uses Hyperliquid's public data. No key, no wallet "
        "connection, and no orders. A fill you can already see has printed, "
        "so you do not get that entry. The way to actually follow a leader "
        "on this exchange is a **vault**: you deposit USDC, the leader trades "
        "the pool, and your share follows the book from then on.\n\n"
        "Third-party sites such as Hypurrscan and Coinglass repackage the "
        "same public tape. The wallet links on the roster open Hypurrscan "
        "when you want a deeper look at one address. The ranking, the tape, "
        "and the vault list here are the same Hyperliquid feeds.")

    st.markdown("#### The three views")
    st.markdown(
        "- **Earner tape** — who is up, and the buys and sells they printed "
        "in the tape window. A bar chart shows net buy versus sell notional. "
        "Under that, the largest open bets (eight of them). The rest of the "
        "book is on Open books.\n"
        "- **Open books** — the live perp positions for those wallets, "
        "including Hyperliquid's other markets (the `xyz` books and the rest). "
        "The default perp endpoint misses those, and a lot of the big books "
        "sit there. When two or more wallets are on the same coin, the line "
        "above the table is their net agreement.\n"
        "- **Copy vaults** — Hyperliquid's own copy-trading product. Closed "
        "vaults and child vaults are left out. The protocol vault (HLP) is "
        "tagged. Each row links to the vault page, where a deposit is made "
        "on Hyperliquid, not in this app.")

    st.markdown("#### Who counts as a top earner")
    st.markdown(
        "- **Rank window** — Day is who is hot right now, and it is the "
        "default so the tape is not a list of people sitting still. Week, "
        "month, and all-time are the slower cut. Month PnL stays on the "
        "roster either way.\n"
        "- **Rank by** — dollar PnL is money earned in that window. ROI % is "
        "the return on account value, so a smaller account can outrank a "
        "larger one.\n"
        "- **Min account** — drops tiny accounts whose percentage is one "
        "lucky fill. Default is 100k.\n"
        "- **Max volume / equity** — drops market makers. Default is 3×. "
        "A wallet whose volume in the window is many times its equity is "
        "scalping both sides. Those fills are not a book you can copy.\n"
        "- **Wallets** — how many ranked addresses to follow, from 3 to 8. "
        "The cap keeps the public API inside its rate limit.\n"
        "- **Also watch** — one extra 0x address, 42 characters, followed "
        "alongside the ranked list.")

    st.markdown("#### The tape")
    st.markdown(
        "Each row is one fill: time (New York), trader, coin, market, side, "
        "whether it opened or closed, size, price, notional, and closed PnL.\n\n"
        "- **Tape window** — 15 minutes, 1 hour, 4 hours, or 24 hours. "
        "Default is 1 hour.\n"
        "- **Side** — All, Buys, or Sells. This filters the tape only. "
        "Open positions stay listed either way.\n"
        "- **Coin filter** — limits both the tape and the open-book list "
        "to coins whose name contains that text.\n"
        "- **Live** — positions and fills refresh about every 25 seconds. "
        "The leaderboard snapshot itself refreshes about hourly. "
        "**Reload leaderboard** pulls a new snapshot without waiting.\n\n"
        "Hyperliquid returns at most 2,000 fills per request, and a full "
        "2,000 is the oldest slice in the window, not the newest. If a "
        "wallet is still at that cap on a shorter retry, its tape is hidden "
        "for a while and its positions still show. An empty tape next to a "
        "large open book means they are holding, not that the feed failed. "
        "Day leaders often sit like that.\n\n"
        "Board equity is the leaderboard's account value. Live perp equity "
        "is what the position endpoint reports right now. They can differ "
        "when margin sits somewhere the perp call does not count. Spot USDC "
        "is not added on top of perp equity, because that double-counts "
        "margin.")

    st.markdown("#### Copy vaults")
    st.markdown(
        "Defaults keep vaults with at least 250k locked, 45 days of history, "
        "and a positive month, sorted by month PnL, showing 20. A four-digit "
        "APR on a vault opened last week is not a track record, which is "
        "what the age and size floors are for.\n\n"
        "- **Month PnL** — dollars the vault made over the last month.\n"
        "- **Month / TVL** — that profit relative to the money in the vault.\n"
        "- **APR** — the exchange's annualized figure. Treat a huge number "
        "on a young vault as a short hot streak.\n"
        "- **All-time PnL** and **Age** — the longer record sitting next to "
        "the month.\n"
        "- **Reload vaults** refreshes the list. The cached copy is about "
        "30 minutes old otherwise.\n\n"
        "Open the vault link, read who the leader is, and decide there. "
        "This app will not deposit for you.")
    st.caption("Research only. Probability tilts, not prophecy. "
               "Not investment advice.")


def render_crypto_copycat_tab() -> None:
    """Money Maker entry point."""
    _header()
    view = _section(["Earner tape", "Open books", "Copy vaults", "Crypto Helper"], "cc_view")
    if view == "Crypto Helper":
        _render_helper()
        return
    if view == "Copy vaults":
        _vault_controls()
        _render_vaults()
        return
    _tape_controls()
    _live_fragment()
