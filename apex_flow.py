"""
APEX FLOW — screener port of the APEX FLOW TradingView indicator.

Scores every stock on the same 0-100 conviction scale the Pine indicator uses,
so a name that ranks here looks identical on the chart.

    risk gate            35 pts   20-bar realized volatility (the validated core)
    range position       25 pts   where price sits in its 20-bar Donchian range
    value-area structure 15 pts   price vs the 50-bar volume profile
    regime               15 pts   5-factor trend tally
    trend quality        10 pts   price vs the structural MA

Plus the two filters the research supported:
    * CALM volatility only (thresholds auto-scaled to the bar size)
    * relative strength within +/-3% of the benchmark's return

TIMEFRAME HONESTY
-----------------
Everything here was validated on DAILY bars: score >= 80 + CALM + RS gave a
60.2% win rate and a 4.28% chance of a >10% loss over 20 days, versus a 50.7%
baseline, on 984 held-out tickers.

The volatility thresholds (2.5% / 4.0%) are daily numbers. Volatility scales
with sqrt(time), so on a 5-minute chart typical 20-bar vol is ~0.23% — every
bar would read CALM and the gate would silently stop filtering. TF_SCALE below
rescales the thresholds by 1/sqrt(bars per day) so the gate keeps working on
any bar size. That keeps it mathematically honest; it does NOT make a
daily-swing edge apply to a 2-minute bar. Intraday results are unvalidated.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

# ── timeframe registry ───────────────────────────────────────────────
# bars_per_day drives the volatility rescaling; alpaca is the API's name for it
TIMEFRAMES = {
    "1 Day":   dict(alpaca="1Day",   yf="1d",  bars_per_day=1.0,    validated=True,
                    lookback_days=420),
    "4 Hour":  dict(alpaca="4Hour",  yf="1h",  bars_per_day=1.625,  validated=False,
                    lookback_days=400),
    "1 Hour":  dict(alpaca="1Hour",  yf="1h",  bars_per_day=6.5,    validated=False,
                    lookback_days=120),
    "30 Min":  dict(alpaca="30Min",  yf="30m", bars_per_day=13.0,   validated=False,
                    lookback_days=60),
    "5 Min":   dict(alpaca="5Min",   yf="5m",  bars_per_day=78.0,   validated=False,
                    lookback_days=25),
    "2 Min":   dict(alpaca="2Min",   yf="2m",  bars_per_day=195.0,  validated=False,
                    lookback_days=10),
}

VOL_CALM_D = 2.5      # daily-bar CALM threshold
VOL_HIGH_D = 4.0      # daily-bar HIGH threshold
MIN_BARS = 210        # need the structural MA
BENCHMARK = "SPY"

# The CALM gate rewards low volatility — which a stock that barely trades will
# satisfy trivially. Left unguarded the screener fills up with dead microcaps
# and SPACs printing 0.2% daily vol: "calm" only because nothing happens. These
# floors keep the gate measuring genuine quiet rather than illiquidity.
VOL_FLOOR_D = 0.60    # daily-equivalent vol below this = not calm, just dead
MIN_COVERAGE = 0.92   # share of the last 200 bars that must be real prints


def tf_scale(bars_per_day: float) -> float:
    """Volatility scales with sqrt(time). Rescale daily thresholds to a bar size."""
    return 1.0 / np.sqrt(max(bars_per_day, 1e-6))


# ── vectorised score components (arrays are T x N, newest row last) ──
def _rolling_last(arr: np.ndarray, win: int, fn) -> np.ndarray:
    """fn applied over the last `win` rows -> (N,)"""
    return fn(arr[-win:], axis=0)


def score_panel(o: np.ndarray, h: np.ndarray, l: np.ndarray,
                c: np.ndarray, v: np.ndarray, bars_per_day: float = 1.0,
                bench_ret: float | None = None, rs_band: float = 3.0,
                rs_len: int = 20) -> pd.DataFrame:
    """Score the LAST bar of every column. Arrays are (T, N) float.

    Mirrors the Pine indicator component for component.
    """
    T, N = c.shape
    out = {}

    # --- 1. risk gate (35) ------------------------------------------
    rets = np.diff(c[-21:], axis=0) / c[-21:-1]
    vol20 = np.nanstd(rets, axis=0, ddof=1) * 100.0
    sc = tf_scale(bars_per_day)
    calm_th, high_th = VOL_CALM_D * sc, VOL_HIGH_D * sc
    s_risk = np.where(vol20 <= calm_th, 35.0,
                      np.where(vol20 <= high_th, 20.0, 0.0))
    risk_state = np.where(vol20 <= calm_th, 2,
                          np.where(vol20 <= high_th, 1, 0))

    # --- 2. range position (25) -------------------------------------
    d_hi = np.nanmax(h[-20:], axis=0)
    d_lo = np.nanmin(l[-20:], axis=0)
    rng = d_hi - d_lo
    px = c[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        chan_pos = np.where(rng > 0, (px - d_lo) / rng * 100.0, 50.0)
    chan_pos = np.clip(np.nan_to_num(chan_pos, nan=50.0), 0, 100)
    s_range = np.clip((100.0 - chan_pos) / 100.0 * 25.0, 0, 25)

    # --- 3. value-area structure (15) -------------------------------
    LB, BINS, VA = 50, 12, 0.70
    w_lo = np.nanmin(l[-LB:], axis=0)
    w_hi = np.nanmax(h[-LB:], axis=0)
    step = np.where(w_hi > w_lo, (w_hi - w_lo) / BINS, np.nan)
    tp = (h[-LB:] + l[-LB:] + c[-LB:]) / 3.0
    vol_w = np.nan_to_num(v[-LB:])
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = np.floor((tp - w_lo) / step)
    idx = np.clip(np.nan_to_num(idx, nan=0.0), 0, BINS - 1).astype(np.int16)
    binvol = np.zeros((BINS, N), dtype=np.float64)
    for b in range(BINS):
        binvol[b] = np.where(idx == b, vol_w, 0.0).sum(axis=0)

    total = binvol.sum(axis=0)
    poc_i = binvol.argmax(axis=0)
    # expand out from the POC until VA% of volume is enclosed
    lo_i = poc_i.copy()
    hi_i = poc_i.copy()
    acc = binvol[poc_i, np.arange(N)].astype(np.float64)
    target = total * VA
    cols = np.arange(N)
    for _ in range(BINS):
        need = acc < target
        if not need.any():
            break
        can_dn = lo_i > 0
        can_up = hi_i < BINS - 1
        v_dn = np.where(can_dn, binvol[np.maximum(lo_i - 1, 0), cols], -1.0)
        v_up = np.where(can_up, binvol[np.minimum(hi_i + 1, BINS - 1), cols], -1.0)
        go_up = need & can_up & (v_up >= v_dn)
        go_dn = need & can_dn & ~go_up
        hi_i = np.where(go_up, hi_i + 1, hi_i)
        lo_i = np.where(go_dn, lo_i - 1, lo_i)
        acc = acc + np.where(go_up, np.maximum(v_up, 0), 0) \
                  + np.where(go_dn, np.maximum(v_dn, 0), 0)

    val = w_lo + lo_i * step
    vah = w_lo + (hi_i + 1) * step
    poc = w_lo + (poc_i + 0.5) * step
    below = px < val
    inside = (px >= val) & (px <= vah)
    s_struct = np.where(below, 15.0, np.where(inside, 10.0, 3.0))
    va_state = np.where(below, "BELOW VALUE",
                        np.where(inside, "IN VALUE", "ABOVE VALUE"))

    # --- 4. regime (15) ---------------------------------------------
    sma20 = np.nanmean(c[-20:], axis=0)
    sma50 = np.nanmean(c[-50:], axis=0)
    sma200 = np.nanmean(c[-200:], axis=0)
    tally = (np.where(px > sma20, 1, -1) + np.where(px > sma50, 1, -1)
             + np.where(sma20 > sma50, 1, -1) + np.where(px > sma200, 1, -1))
    regime = np.where(tally >= 3, 2, np.where(tally >= 1, 1,
                      np.where(tally <= -3, -2, np.where(tally <= -1, -1, 0))))
    reg_pts = {2: 15.0, 1: 12.0, 0: 8.0, -1: 5.0, -2: 2.0}
    s_regime = np.vectorize(reg_pts.get)(regime).astype(float)
    reg_name = {2: "STRONG BULL", 1: "MILD BULL", 0: "NEUTRAL",
                -1: "MILD BEAR", -2: "STRONG BEAR"}
    regime_txt = np.vectorize(reg_name.get)(regime)

    # --- 5. trend quality (10) --------------------------------------
    s_trendq = np.where(px > sma200, 10.0, 4.0)

    score = s_risk + s_range + s_struct + s_regime + s_trendq

    # --- relative strength ------------------------------------------
    base = c[-(rs_len + 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        stock_ret = (px / base - 1.0) * 100.0
    if bench_ret is None:
        rs_rel = np.full(N, np.nan)
        rs_ok = np.ones(N, dtype=bool)
    else:
        rs_rel = stock_ret - bench_ret
        rs_ok = np.abs(rs_rel) <= rs_band

    out = pd.DataFrame({
        "Score": np.round(score, 1),
        "Risk": np.where(risk_state == 2, "CALM",
                         np.where(risk_state == 1, "NORMAL", "HIGH")),
        "Vol%": np.round(vol20, 2),
        "RangePos": np.round(chan_pos, 0),
        "ValueArea": va_state,
        "Regime": regime_txt,
        "RS": np.round(rs_rel, 2),
        "RS_OK": rs_ok,
        "Price": np.round(px, 2),
        "POC": np.round(poc, 2),
        "VAL": np.round(val, 2),
        "VAH": np.round(vah, 2),
        "_risk_state": risk_state,
        "_s_risk": s_risk, "_s_range": np.round(s_range, 1),
        "_s_struct": s_struct, "_s_regime": s_regime, "_s_trendq": s_trendq,
        # data-quality guards used by _finalize
        "_coverage": np.isfinite(c[-200:]).sum(axis=0) / 200.0,
        "_vol_floor_ok": vol20 >= VOL_FLOOR_D * sc,
    })
    return out


# ── daily path: the nightly dump panel (fast, whole market) ─────────
def available_sectors() -> list:
    """Sectors present in the nightly dump, sorted, for the UI picker."""
    import cascade_engine as ce
    panel, tickers, sectors, mdv, dates = ce.load_dump_panel()
    return sorted({str(s) for s in sectors if s and str(s).strip()})


def scan_daily(min_price: float = 5.0, min_dollar_vol: float = 5e6,
               rs_band: float = 3.0, require_calm: bool = True,
               min_score: float = 0.0, top: int = 50,
               apply_rs: bool = True,
               only_sectors: list | None = None) -> pd.DataFrame:
    """Score the entire dump universe on daily bars.

    `only_sectors` restricts the universe to those sectors. It is applied
    BEFORE the top-N cut, so "top 40" means the best 40 within the sectors
    you picked — not the best 40 overall filtered down afterwards. Pass None
    (or every sector) to scan the whole market.
    """
    import cascade_engine as ce
    panel, tickers, sectors, mdv, dates = ce.load_dump_panel()
    o, h, l, c, v = (panel[k].astype(np.float64) for k in ("o", "h", "l", "c", "v"))
    if c.shape[0] < MIN_BARS:
        raise RuntimeError(f"dump has only {c.shape[0]} bars; need {MIN_BARS}")

    # benchmark return over the RS window
    bench_ret = None
    bix = np.where(tickers == BENCHMARK)[0]
    if len(bix):
        bc = c[:, bix[0]]
        if np.isfinite(bc[-21]) and bc[-21] > 0:
            bench_ret = float((bc[-1] / bc[-21] - 1) * 100)
    if bench_ret is None:
        bench_ret = float(np.nanmedian((c[-1] / c[-21] - 1) * 100))

    df = score_panel(o, h, l, c, v, bars_per_day=1.0,
                     bench_ret=bench_ret, rs_band=rs_band)
    df.insert(0, "Ticker", tickers)
    df.insert(1, "Sector", sectors)
    df["DollarVol"] = mdv

    valid = np.isfinite(c[-1]) & np.isfinite(c[-200])
    df = df[valid]
    df = df[(df["Price"] >= min_price) & (df["DollarVol"] >= min_dollar_vol)]
    if only_sectors:
        df = df[df["Sector"].isin(list(only_sectors))]
    return _finalize(df, require_calm, min_score, top, apply_rs)


def _finalize(df: pd.DataFrame, require_calm: bool, min_score: float,
              top: int, apply_rs: bool = True,
              guard_illiquid: bool = True) -> pd.DataFrame:
    df = df.copy()
    if guard_illiquid:
        # drop names with gappy history or a volatility reading so low it means
        # "doesn't trade" rather than "calm"
        df = df[(df["_coverage"] >= MIN_COVERAGE) & df["_vol_floor_ok"]]
    df["Gate"] = (df["Score"] >= 80) & (df["_risk_state"] == 2)
    if apply_rs:
        df["Gate"] &= df["RS_OK"]
        df = df[df["RS_OK"]]
    if require_calm:
        df = df[df["_risk_state"] == 2]
    df = df[df["Score"] >= min_score]
    df = df.sort_values("Score", ascending=False).head(top).reset_index(drop=True)
    df.insert(0, "#", np.arange(1, len(df) + 1))
    return df


# ── intraday path: Alpaca bars for a shortlisted universe ───────────
def _alpaca_bars(symbols: list, tf: str, start: str, feed: str = "iex") -> dict:
    """Multi-symbol bars from Alpaca. Returns {symbol: DataFrame}."""
    import cascade_engine as ce
    kid, sec = ce._alpaca_keys_simple()
    if not kid:
        return {}
    hdr = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    out: dict[str, list] = {}
    CH = 100
    for i in range(0, len(symbols), CH):
        chunk = symbols[i:i + CH]
        token = None
        while True:
            p = {"symbols": ",".join(chunk), "timeframe": tf, "start": start,
                 "limit": 10000, "adjustment": "split", "feed": feed}
            if token:
                p["page_token"] = token
            try:
                r = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                                 params=p, headers=hdr, timeout=45)
                if r.status_code != 200:
                    break
                j = r.json()
            except Exception:
                break
            for sym, bars in (j.get("bars") or {}).items():
                out.setdefault(sym, []).extend(bars)
            token = j.get("next_page_token")
            if not token:
                break
    frames = {}
    for sym, bars in out.items():
        if len(bars) < MIN_BARS:
            continue
        frames[sym] = pd.DataFrame([{
            "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"]}
            for b in bars])
    return frames


def scan_intraday(timeframe: str, universe: list, rs_band: float = 3.0,
                  require_calm: bool = True, min_score: float = 0.0,
                  top: int = 50, sectors: dict | None = None,
                  apply_rs: bool = True) -> pd.DataFrame:
    """Score a shortlisted universe on an intraday timeframe via Alpaca."""
    meta = TIMEFRAMES[timeframe]
    start = (date.today() - timedelta(days=meta["lookback_days"])).isoformat()
    syms = list(dict.fromkeys(universe + [BENCHMARK]))
    frames = _alpaca_bars(syms, meta["alpaca"], start)
    if not frames:
        return pd.DataFrame()

    bench_ret = None
    if BENCHMARK in frames:
        bc = frames[BENCHMARK]["c"].to_numpy()
        if len(bc) > 21 and bc[-21] > 0:
            bench_ret = float((bc[-1] / bc[-21] - 1) * 100)

    names = [s for s in frames if s != BENCHMARK]
    if not names:
        return pd.DataFrame()
    T = min(len(frames[s]) for s in names)
    T = max(T, MIN_BARS)
    def stack(key):
        return np.column_stack([frames[s][key].to_numpy()[-T:] for s in names
                                if len(frames[s]) >= T]).astype(np.float64)
    names = [s for s in names if len(frames[s]) >= T]
    if not names:
        return pd.DataFrame()
    o, h, l, c, v = (stack(k) for k in ("o", "h", "l", "c", "v"))

    df = score_panel(o, h, l, c, v, bars_per_day=meta["bars_per_day"],
                     bench_ret=bench_ret, rs_band=rs_band)
    df.insert(0, "Ticker", names)
    df.insert(1, "Sector", [(sectors or {}).get(s, "—") for s in names])
    df["DollarVol"] = np.nan
    return _finalize(df, require_calm, min_score, top, apply_rs)


def liquid_universe(n: int = 300, min_price: float = 5.0,
                    only_sectors: list | None = None) -> tuple:
    """Top-N most liquid dump names — the shortlist intraday scans run on.

    `only_sectors` narrows the pool first, so an intraday sector scan fetches
    the N most liquid names IN those sectors (fewer Alpaca calls, and the
    shortlist isn't wasted on sectors you excluded).
    """
    import cascade_engine as ce
    panel, tickers, sectors, mdv, dates = ce.load_dump_panel()
    c = panel["c"]
    px = c[-1]
    ok = np.isfinite(px) & (px >= min_price) & np.isfinite(mdv)
    if only_sectors:
        want = set(str(s) for s in only_sectors)
        ok = ok & np.array([str(s) in want for s in sectors])
    idx = np.argsort(np.where(ok, mdv, -1))[::-1][:n]
    return ([str(tickers[i]) for i in idx],
            {str(tickers[i]): str(sectors[i]) for i in idx})
