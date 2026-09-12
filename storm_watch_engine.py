"""
Storm Watch engine — identify names coiling *right before* they lift.

Point-in-time safe on the nightly dump (_hist OHLCV). Snapshot fundamentals
are used only as a live overlay (they are not dated, so they cannot be
backtested without look-ahead).

Usage:
    python storm_watch_engine.py data/stock_data.json.gz
    python storm_watch_engine.py data/stock_data.json.gz --csv data/storm_watch_picks.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MIN_PRICE = 5.0
MIN_MED_DOLLAR_VOL = 2e6
MIN_BARS = 70
STOP_PCT = -0.08
HORIZONS = (5, 10, 21)
HW_THRESH = 0.03  # +3% high-water within 10 sessions
TOP_N = 25

# Live composite — OOS-validated on 2026-04-24 → 2026-08-11 (see backtest).
# H1 was a momentum regime; H2 flipped to mean-reversion. These five kept a
# positive OOS IC and a positive top-decile (or event) excess. Tightness
# ranked well but the *tightest* tail underperformed, so it is not a long.
LIVE_FACTORS = (
    "rev5",             # 5d reversal: just sold off
    "rangepos_low",     # sitting in the lower part of the 63d range
    "peer_catchup",     # lagged 5d vs sector, 21d residual still constructive
    "lower_bb",         # stretched below the lower Bollinger band
    "rsi_oversold",     # RSI < 30 event
)


# ────────────────────────────────────────────────────────────────────
# IO
# ────────────────────────────────────────────────────────────────────
def load_dump(path: str) -> list:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def build_panels(data: list):
    rows = [r for r in data if len(r.get("_hist", {}).get("dates", [])) >= MIN_BARS]
    all_dates = sorted({d for r in rows for d in r["_hist"]["dates"]})
    dix = {d: i for i, d in enumerate(all_dates)}
    T, N = len(all_dates), len(rows)
    O = np.full((T, N), np.nan, dtype=np.float32)
    H = np.full((T, N), np.nan, dtype=np.float32)
    L = np.full((T, N), np.nan, dtype=np.float32)
    C = np.full((T, N), np.nan, dtype=np.float32)
    V = np.zeros((T, N), dtype=np.float32)
    for j, r in enumerate(rows):
        h = r["_hist"]
        ix = [dix[d] for d in h["dates"]]
        O[ix, j] = h.get("open") or h["close"]
        H[ix, j] = h["high"]
        L[ix, j] = h["low"]
        C[ix, j] = h["close"]
        V[ix, j] = h["volume"]
    C = pd.DataFrame(C).ffill(limit=5).values.astype(np.float32)
    H = pd.DataFrame(H).ffill(limit=5).values.astype(np.float32)
    L = pd.DataFrame(L).ffill(limit=5).values.astype(np.float32)
    O = pd.DataFrame(O).ffill(limit=5).values.astype(np.float32)
    sectors = np.array([r.get("Sector") or "Unknown" for r in rows])
    tickers = np.array([r["Ticker"] for r in rows])
    fund = pd.DataFrame(
        [{
            "Ticker": r["Ticker"],
            "Piotroski": r.get("Piotroski"),
            "ROIC": r.get("ROIC"),
            "OE_Yield": r.get("OE_Yield"),
            "ShortPctFloat": r.get("ShortPctFloat"),
            "DaysToCover": r.get("DaysToCover"),
            "MarketCap": r.get("MarketCap"),
            "P/E": r.get("P/E"),
            "GrossMargin": r.get("GrossMargin"),
        } for r in rows]
    ).set_index("Ticker")
    return dict(O=O, H=H, L=L, C=C, V=V, sectors=sectors, tickers=tickers,
                dates=np.array(all_dates), fund=fund)


def _rmean(a, w):
    return pd.DataFrame(a).rolling(w, min_periods=w).mean().values.astype(np.float32)


def _rmax(a, w):
    return pd.DataFrame(a).rolling(w, min_periods=w).max().values.astype(np.float32)


def _rmin(a, w):
    return pd.DataFrame(a).rolling(w, min_periods=w).min().values.astype(np.float32)


def _rstd(a, w):
    return pd.DataFrame(a).rolling(w, min_periods=w).std().values.astype(np.float32)


def _rsum(a, w):
    return pd.DataFrame(a).rolling(w, min_periods=w).sum().values.astype(np.float32)


def _rsi(C, n=14):
    d = np.diff(C, axis=0, prepend=C[:1])
    gain = np.clip(d, 0, None)
    loss = np.clip(-d, 0, None)
    ag, al = _rmean(gain, n), _rmean(loss, n)
    rs = ag / np.where(al == 0, np.nan, al)
    return (100.0 - 100.0 / (1.0 + rs)).astype(np.float32)


def _ema(a, span):
    return pd.DataFrame(a).ewm(span=span, adjust=False, min_periods=span).mean().values.astype(np.float32)


def _sector_mean(val, sectors, ok):
    """Equal-weight sector mean of `val` among ok names, broadcast to (T, N)."""
    out = np.full_like(val, np.nan)
    for sc in np.unique(sectors):
        if sc == "Unknown":
            continue
        m = sectors == sc
        # mean across names in sector, ignoring non-ok
        masked = np.where(ok & m, val, np.nan)
        with np.errstate(all="ignore"):
            mu = np.nanmean(masked, axis=1, keepdims=True)
        out[:, m] = mu
    return out.astype(np.float32)


def universe_ok(P) -> np.ndarray:
    C, V = P["C"], P["V"]
    dvol = C * V
    med = pd.DataFrame(dvol).rolling(21, min_periods=15).median().values
    return np.isfinite(C) & (C >= MIN_PRICE) & (med >= MIN_MED_DOLLAR_VOL)


def compute_features(P, ok) -> Dict[str, np.ndarray]:
    O, H, L, C, V = P["O"], P["H"], P["L"], P["C"], P["V"]
    T, N = C.shape
    ret1 = np.full_like(C, np.nan)
    ret1[1:] = C[1:] / C[:-1] - 1.0

    def lagret(k):
        out = np.full_like(C, np.nan)
        out[k:] = C[k:] / C[:-k] - 1.0
        return out

    sma10, sma20, sma50 = _rmean(C, 10), _rmean(C, 20), _rmean(C, 50)
    sma200 = _rmean(C, 200) if T >= 200 else np.full_like(C, np.nan)
    atr = _rmean(np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1, 0)),
                                              np.abs(L - np.roll(C, 1, 0)))), 14)
    rsi = _rsi(C, 14)
    ema12, ema26 = _ema(C, 12), _ema(C, 26)
    macd = ema12 - ema26
    macds = _ema(macd, 9)
    macdh = macd - macds
    bb_mid = sma20
    bb_sd = _rstd(C, 20)
    bb_up, bb_lo = bb_mid + 2 * bb_sd, bb_mid - 2 * bb_sd
    bb_w = (bb_up - bb_lo) / np.where(bb_mid == 0, np.nan, bb_mid)
    # Keltner for TTM squeeze
    kel_up, kel_lo = sma20 + 1.5 * atr, sma20 - 1.5 * atr

    hi20, lo20 = _rmax(H, 20), _rmin(L, 20)
    hi21, lo21 = _rmax(H, 21), _rmin(L, 21)
    hi63, lo63 = _rmax(H, 63), _rmin(L, 63)
    rng21 = (hi21 - lo21) / np.where(lo21 <= 0, np.nan, lo21)
    rngpos63 = (C - lo63) / np.where(hi63 - lo63 == 0, np.nan, hi63 - lo63)
    rngpos20 = (C - lo20) / np.where(hi20 - lo20 == 0, np.nan, hi20 - lo20)
    bar_rng = H - L
    close_loc = (C - L) / np.where(bar_rng == 0, np.nan, bar_rng)
    vol_mean63 = _rmean(V, 63)
    rvol5 = _rmean(V, 5) / np.where(vol_mean63 == 0, np.nan, vol_mean63)
    rvol1 = V / np.where(vol_mean63 == 0, np.nan, vol_mean63)
    rv10 = _rstd(ret1, 10)
    rv63 = _rstd(ret1, 63)
    vol_ratio = rv10 / np.where(rv63 == 0, np.nan, rv63)

    # OBV slope (20d)
    signed_v = np.sign(np.nan_to_num(ret1)) * V
    obv = np.cumsum(np.nan_to_num(signed_v), axis=0)
    obv_slope = (obv - np.roll(obv, 20, 0)) / 20.0
    obv_slope[:20] = np.nan

    # NR7: today's range is the narrowest of the last 7
    nr7 = bar_rng <= _rmin(bar_rng, 7)
    # inside day
    prev_H, prev_L = np.roll(H, 1, 0), np.roll(L, 1, 0)
    inside = (H <= prev_H) & (L >= prev_L)
    inside[0] = False

    # consecutive down days
    down = ret1 < 0
    down3 = down & np.roll(down, 1, 0) & np.roll(down, 2, 0)
    down3[:3] = False

    ret5, ret10, ret21, ret63 = lagret(5), lagret(10), lagret(21), lagret(63)
    mom63_skip5 = np.full_like(C, np.nan)
    mom63_skip5[63:] = C[58:-5] / C[:-63] - 1.0 if T > 63 else mom63_skip5
    # correct indexing: at t, C[t-5]/C[t-63]-1
    mom63_skip5 = np.full_like(C, np.nan)
    if T > 63:
        mom63_skip5[63:] = C[63 - 5:T - 5] / C[:T - 63] - 1.0

    resid21 = ret21 - _sector_mean(ret21, P["sectors"], ok)
    resid63 = ret63 - _sector_mean(ret63, P["sectors"], ok)

    # prior 20d high (exclude today) — proximity, not the breakout itself
    hi20_prior = np.roll(hi20, 1, 0)
    hi20_prior[0] = np.nan
    lo20_prior = np.roll(lo20, 1, 0)
    dist_hi20 = (hi20_prior - C) / np.where(C == 0, np.nan, C)

    # failed breakdown / spring: today's low breaks prior 20d low, close reclaims
    spring = (L < lo20_prior) & (C > lo20_prior) & (C > O)
    spring[0] = False

    # volume declining over 21d (linear slope via last vs first third)
    v7, v21 = _rmean(V, 7), _rmean(V, 21)
    vol_dry = v7 / np.where(v21 == 0, np.nan, v21)

    # tight vs own history: 21d range percentile proxy = range / 63d range
    rng63 = (hi63 - lo63) / np.where(lo63 <= 0, np.nan, lo63)
    tight_rel = rng21 / np.where(rng63 == 0, np.nan, rng63)

    F = dict(
        ret1=ret1, ret5=ret5, ret10=ret10, ret21=ret21, ret63=ret63,
        mom63_skip5=mom63_skip5, resid21=resid21, resid63=resid63,
        rsi=rsi, macdh=macdh, macd_prev=np.roll(macdh, 1, 0),
        sma10=sma10, sma20=sma20, sma50=sma50, sma200=sma200,
        atr=atr, bb_w=bb_w, bb_lo=bb_lo, bb_up=bb_up,
        rvol5=rvol5, rvol1=rvol1, rv10=rv10, vol_ratio=vol_ratio,
        rng21=rng21, rngpos63=rngpos63, rngpos20=rngpos20,
        close_loc=close_loc, obv_slope=obv_slope, nr7=nr7.astype(np.float32),
        inside=inside.astype(np.float32), down3=down3.astype(np.float32),
        dist_hi20=dist_hi20, spring=spring.astype(np.float32),
        vol_dry=vol_dry, tight_rel=tight_rel,
        kel_up=kel_up, kel_lo=kel_lo, hi20=hi20, lo20=lo20,
        C=C, O=O, H=H, L=L, V=V,
    )
    return F


def factor_scores(F) -> Dict[str, np.ndarray]:
    C = F["C"]
    rsi, macdh = F["rsi"], F["macdh"]
    # TTM squeeze: BB inside Keltner
    squeeze = (F["bb_up"] < F["kel_up"]) & (F["bb_lo"] > F["kel_lo"])

    scores = {}
    # --- classic ---
    scores["mom5"] = F["ret5"]
    scores["mom10"] = F["ret10"]
    scores["mom21"] = F["ret21"]
    scores["mom63"] = F["ret63"]
    scores["mom63_skip5"] = F["mom63_skip5"]
    scores["rev5"] = -F["ret5"]                          # short-term reversal
    scores["rsi_raw"] = rsi
    scores["rsi_oversold"] = (rsi < 30).astype(np.float32)
    scores["rsi_zone"] = ((rsi >= 55) & (rsi <= 70)).astype(np.float32)
    scores["macd_hist"] = macdh
    scores["macd_cross"] = ((macdh > 0) & (F["macd_prev"] <= 0)).astype(np.float32)
    scores["above_sma50"] = (C > F["sma50"]).astype(np.float32)
    dist50 = (C - F["sma50"]) / np.where(F["sma50"] == 0, np.nan, F["sma50"])
    # MA50 proximity: near but above (0-8%)
    scores["ma50_prox"] = np.where((dist50 >= 0) & (dist50 <= 0.08), 1.0 - dist50 / 0.08, 0.0)
    scores["golden"] = (F["sma50"] > F["sma200"]).astype(np.float32)
    scores["rvol"] = F["rvol5"]
    scores["donchian20"] = (C >= F["hi20"]).astype(np.float32)   # already breaking out
    scores["tight_base"] = -F["rng21"]                          # tighter better
    scores["rangepos_high"] = F["rngpos63"]                     # continuation
    scores["rangepos_low"] = 1.0 - F["rngpos63"]                # mean-reversion coil
    scores["bb_squeeze"] = -F["bb_w"]
    scores["lower_bb"] = (F["bb_lo"] - C) / np.where(F["atr"] == 0, np.nan, F["atr"])
    scores["close_loc"] = F["close_loc"]
    scores["obv_slope"] = F["obv_slope"]
    scores["resid21"] = F["resid21"]
    scores["resid63"] = F["resid63"]

    # --- outside the box ---
    # Storm coil: vol crushed vs own 63d, tight 21d range vs 63d, residual RS,
    # close in top of bar, still above SMA20 (direction filter).
    coil = (
        (F["vol_ratio"] < 0.65).astype(np.float32)
        + (F["tight_rel"] < 0.45).astype(np.float32)
        + (F["resid21"] > 0).astype(np.float32)
        + (F["close_loc"] > 0.7).astype(np.float32)
        + (C > F["sma20"]).astype(np.float32)
    )
    scores["storm_coil"] = coil

    # Silent accumulation: volume drying, price holding upper half of 63d range,
    # tight base, OBV still rising (stealth bid).
    silent = (
        (F["vol_dry"] < 0.85).astype(np.float32)
        + (F["rngpos63"] > 0.55).astype(np.float32)
        + (F["rng21"] < 0.12).astype(np.float32)
        + (F["obv_slope"] > 0).astype(np.float32)
        + (C > F["sma50"]).astype(np.float32)
    )
    scores["silent_accum"] = silent

    # Breakout proximity: within 3% of prior 20d high, not yet through it.
    scores["breakout_prox"] = (
        (F["dist_hi20"] >= 0) & (F["dist_hi20"] <= 0.03)
        & (C < F["hi20"])
        & (C > F["sma20"])
    ).astype(np.float32)

    # NR7 + close in upper quartile of the bar (energy coiled, buyers won the day)
    scores["nr7_thrust"] = (F["nr7"] > 0) & (F["close_loc"] >= 0.75)
    scores["nr7_thrust"] = scores["nr7_thrust"].astype(np.float32)

    # Wyckoff spring / failed breakdown
    scores["spring"] = F["spring"]

    # Climactic reversal: 3 down days then heavy-volume up-close
    scores["climax_rev"] = (
        (F["down3"] > 0) & (F["rvol1"] > 1.8) & (F["close_loc"] > 0.6) & (C > F["O"])
    ).astype(np.float32)

    # TTM squeeze on, histogram turning up
    scores["ttm_squeeze"] = (squeeze & (macdh > F["macd_prev"]) & (C > F["sma20"])).astype(np.float32)

    # VCP-lite: range contracting, near 63d highs, rvol muted
    scores["vcp"] = (
        (F["tight_rel"] < 0.5) & (F["rngpos63"] > 0.7) & (F["rvol5"] < 0.9) & (C > F["sma50"])
    ).astype(np.float32)

    # Peer catch-up: lagged 5d vs sector but 21d residual still positive
    scores["peer_catchup"] = np.where(
        (F["resid21"] > 0) & (F["ret5"] < 0), F["resid21"] - F["ret5"], 0.0
    ).astype(np.float32)

    # OBV/price divergence: OBV rising, 10d price flat-to-down (stealth bid)
    px_flat = np.abs(F["ret10"]) < 0.04
    scores["obv_div"] = ((F["obv_slope"] > 0) & px_flat & (C > F["sma20"])).astype(np.float32)

    # High-close after inside day (setup, not the breakout)
    scores["inside_coil"] = ((F["inside"] > 0) & (F["close_loc"] > 0.6) & (C > F["sma20"])).astype(np.float32)

    # Vol crush rank (continuous)
    scores["vol_crush"] = -F["vol_ratio"]

    # Coil × residual (continuous interaction)
    tight_score = -F["rng21"]
    scores["coil_rs"] = _cs_rank(tight_score) * _cs_rank(F["resid21"])

    # Shakeout coil — the "right before it lifts" setup that survived OOS:
    # sold off in the last week, sitting in the lower 40% of the 63d range,
    # RSI not overbought, sector hasn't abandoned it. Count 0–5.
    scores["shakeout_coil"] = (
        (F["ret5"] < 0).astype(np.float32)
        + (F["rngpos63"] < 0.40).astype(np.float32)
        + (rsi < 40).astype(np.float32)
        + (F["resid21"] > -0.03).astype(np.float32)
        + ((F["vol_ratio"] < 0.85) | (F["rng21"] < 0.15)).astype(np.float32)
    )

    return scores


def regime_spread(mom21: np.ndarray, ok: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Point-in-time 10d-smoothed spread of high- vs low-momentum 5d realized
    returns. Positive => momentum regime; negative => reversal regime.
    Uses only information through t-1 (signal taken at t-6, 5d fwd realized
    by t-1).
    """
    T = mom21.shape[0]
    spread = np.full(T, np.nan, dtype=np.float32)
    for t in range(30, T):
        t0 = t - 6
        m = ok[t0] & np.isfinite(mom21[t0]) & np.isfinite(C[t0]) & np.isfinite(C[t - 1])
        if m.sum() < 50:
            continue
        s = mom21[t0, m]
        r5 = C[t - 1, m] / C[t0, m] - 1.0
        q_hi, q_lo = np.nanquantile(s, 0.8), np.nanquantile(s, 0.2)
        hi, lo = s >= q_hi, s <= q_lo
        if hi.sum() >= 5 and lo.sum() >= 5:
            spread[t] = float(np.nanmean(r5[hi]) - np.nanmean(r5[lo]))
    return pd.Series(spread).rolling(10, min_periods=5).mean().values.astype(np.float32)


def add_regime_score(scores: Dict[str, np.ndarray], ok: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Momentum combo when the trailing spread is > 0, reversal combo otherwise."""
    sm = regime_spread(scores["mom21"], ok, C)
    mom_combo = (
        _cs_rank(scores["mom63_skip5"])
        + _cs_rank(scores["resid21"])
        + _cs_rank(scores["rangepos_high"])
    )
    rev_combo = (
        _cs_rank(scores["rev5"])
        + _cs_rank(scores["rangepos_low"])
        + _cs_rank(scores["peer_catchup"])
    )
    scores["regime_adapt"] = np.where(sm[:, None] > 0, mom_combo, rev_combo).astype(np.float32)
    return sm


def _cs_rank(a: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank per row. NaNs stay NaN."""
    df = pd.DataFrame(a)
    return df.rank(axis=1, pct=True).values.astype(np.float32)


def forward_returns(C, H, L) -> Dict[str, np.ndarray]:
    T = C.shape[0]
    out = {}
    for h in HORIZONS:
        f = np.full_like(C, np.nan)
        f[: T - h] = C[h:] / C[: T - h] - 1.0
        out[f"fwd{h}"] = f
    hw = np.full_like(C, np.nan)
    dd = np.full_like(C, np.nan)
    with np.errstate(all="ignore"):
        for t in range(T - 11):
            hw[t] = np.nanmax(H[t + 1:t + 11], axis=0) / C[t] - 1.0
            dd[t] = np.nanmin(L[t + 1:t + 11], axis=0) / C[t] - 1.0
    out["hw10"] = hw
    out["dd10"] = dd
    return out


def _spearman_row(s, f):
    m = np.isfinite(s) & np.isfinite(f)
    if m.sum() < 40:
        return np.nan
    rs = pd.Series(s[m]).rank().values
    rf = pd.Series(f[m]).rank().values
    if rs.std() == 0 or rf.std() == 0:
        return np.nan
    return float(np.corrcoef(rs, rf)[0, 1])


def evaluate(scores, fwd, ok, dates, warmup=80):
    T = ok.shape[0]
    split = warmup + (T - warmup - 22) // 2
    rows = []
    days = list(range(warmup, T - 22))
    is_days = [t for t in days if t < split]
    oos_days = [t for t in days if t >= split]
    univ_fwd10 = []
    univ_hw = []
    for t in days:
        m = ok[t]
        univ_fwd10.append(np.nanmean(fwd["fwd10"][t, m]))
        univ_hw.append(np.nanmean((fwd["hw10"][t, m] >= HW_THRESH).astype(np.float32)))
    univ10 = float(np.nanmean(univ_fwd10))
    univ_hw_p = float(np.nanmean(univ_hw))

    for name, S in scores.items():
        rec = {"factor": name}
        for label, dlist in (("is", is_days), ("oos", oos_days), ("all", days)):
            ics5, ics10, ics21 = [], [], []
            top_ex, top_hit, top_hw, n_top = [], [], [], []
            bin_ex, bin_hit, bin_n, bin_hw = [], [], [], []
            is_bin = True
            for t in dlist:
                m = ok[t] & np.isfinite(S[t]) & np.isfinite(fwd["fwd10"][t])
                if m.sum() < 40:
                    continue
                st, f10, f5, f21 = S[t, m], fwd["fwd10"][t, m], fwd["fwd5"][t, m], fwd["fwd21"][t, m]
                hw = fwd["hw10"][t, m]
                ics5.append(_spearman_row(st, f5))
                ics10.append(_spearman_row(st, f10))
                ics21.append(_spearman_row(st, f21))
                uniq = np.unique(st[np.isfinite(st)])
                if uniq.size > 3:
                    is_bin = False
                    q = np.nanquantile(st, 0.9)
                    top = st >= q
                    if top.sum() >= 5:
                        top_ex.append(float(np.nanmean(f10[top]) - np.nanmean(f10)))
                        top_hit.append(float(np.mean(f10[top] > 0)))
                        top_hw.append(float(np.mean(hw[top] >= HW_THRESH)))
                        n_top.append(int(top.sum()))
                fired = st > 0.5 if uniq.size <= 8 else None
                if fired is not None and fired.sum() >= 3:
                    bin_ex.append(float(np.nanmean(f10[fired]) - np.nanmean(f10)))
                    bin_hit.append(float(np.mean(f10[fired] > 0)))
                    bin_hw.append(float(np.mean(hw[fired] >= HW_THRESH)))
                    bin_n.append(int(fired.sum()))
            rec[f"ic5_{label}"] = _nanmean(ics5)
            rec[f"ic10_{label}"] = _nanmean(ics10)
            rec[f"ic21_{label}"] = _nanmean(ics21)
            rec[f"top_ex10_{label}"] = _nanmean(top_ex)
            rec[f"top_hit_{label}"] = _nanmean(top_hit)
            rec[f"top_hw_{label}"] = _nanmean(top_hw)
            rec[f"bin_ex10_{label}"] = _nanmean(bin_ex)
            rec[f"bin_hit_{label}"] = _nanmean(bin_hit)
            rec[f"bin_hw_{label}"] = _nanmean(bin_hw)
            rec[f"bin_n_{label}"] = _nanmean(bin_n)
        rec["kind"] = "binary" if rec.get("top_ex10_oos") != rec.get("top_ex10_oos") and rec.get("bin_ex10_oos") == rec.get("bin_ex10_oos") else (
            "binary" if not np.isfinite(rec.get("top_ex10_oos", np.nan)) and np.isfinite(rec.get("bin_ex10_oos", np.nan)) else "ranked"
        )
        if not np.isfinite(rec.get("top_ex10_oos", np.nan)) and np.isfinite(rec.get("bin_ex10_oos", np.nan)):
            rec["kind"] = "binary"
        elif np.isfinite(rec.get("top_ex10_oos", np.nan)):
            rec["kind"] = "ranked"
        else:
            rec["kind"] = "binary" if np.isfinite(rec.get("bin_ex10_all", np.nan)) else "ranked"
        rec["oos_edge"] = rec["top_ex10_oos"] if rec["kind"] == "ranked" else rec["bin_ex10_oos"]
        rec["oos_hw"] = rec["top_hw_oos"] if rec["kind"] == "ranked" else rec["bin_hw_oos"]
        rec["oos_ic10"] = rec["ic10_oos"]
        rows.append(rec)
    df = pd.DataFrame(rows)
    meta = dict(
        n_days=len(days),
        is_days=len(is_days),
        oos_days=len(oos_days),
        split_date=str(dates[split]) if split < len(dates) else None,
        start=str(dates[warmup]),
        end=str(dates[days[-1]]) if days else None,
        univ_fwd10=univ10,
        univ_hw10_p3=univ_hw_p,
        warmup=warmup,
        n_names=int(ok[-1].sum()),
    )
    return df, meta


def _nanmean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def composite_score(scores: Dict[str, np.ndarray], names=LIVE_FACTORS) -> np.ndarray:
    parts = []
    for n in names:
        if n not in scores:
            continue
        parts.append(_cs_rank(scores[n]))
    if not parts:
        raise ValueError("no live factors available")
    stack = np.stack(parts, axis=0)
    return np.nanmean(stack, axis=0).astype(np.float32)


def evaluate_composite(comp, fwd, ok, dates, warmup=80):
    """Same metrics as a ranked factor, plus a weekly top-N book."""
    dummy = {"composite": comp}
    df, meta = evaluate(dummy, fwd, ok, dates, warmup=warmup)
    T = ok.shape[0]
    split = warmup + (T - warmup - 22) // 2
    rets_is, rets_oos = [], []
    for t in range(warmup, T - 6, 5):  # weekly-ish
        m = ok[t] & np.isfinite(comp[t]) & np.isfinite(fwd["fwd5"][t])
        if m.sum() < 50:
            continue
        idx = np.where(m)[0]
        order = np.argsort(-comp[t, idx])[:TOP_N]
        r = float(np.nanmean(fwd["fwd5"][t, idx[order]]))
        u = float(np.nanmean(fwd["fwd5"][t, m]))
        (rets_oos if t >= split else rets_is).append((r, u))
    def pack(pairs):
        if not pairs:
            return {}
        r = np.array([p[0] for p in pairs])
        u = np.array([p[1] for p in pairs])
        excess = r - u
        return dict(
            n=len(pairs),
            avg_5d=float(r.mean()),
            univ_5d=float(u.mean()),
            excess_5d=float(excess.mean()),
            hit=float((r > 0).mean()),
            excess_hit=float((excess > 0).mean()),
        )
    meta["weekly_topn_is"] = pack(rets_is)
    meta["weekly_topn_oos"] = pack(rets_oos)
    return df.iloc[0].to_dict(), meta


def live_picks(P, F, scores, ok, n=TOP_N, regime: np.ndarray | None = None) -> pd.DataFrame:
    t = P["C"].shape[0] - 1
    comp = composite_score(scores)
    c = P["C"][t]
    m = ok[t] & np.isfinite(comp[t])
    idx = np.where(m)[0]
    order = idx[np.argsort(-comp[t, idx])]
    fund = P["fund"]
    rsi = F["rsi"][t]
    regime_now = "momentum" if (regime is not None and np.isfinite(regime[t]) and regime[t] > 0) else "reversal"
    MAX_CRASH = -0.22  # 5d dump beyond this is distress, not a shakeout
    rows = []
    for j in order:
        tk = P["tickers"][j]
        ret5 = float(F["ret5"][t, j]) if np.isfinite(F["ret5"][t, j]) else None
        if ret5 is not None and ret5 < MAX_CRASH:
            continue
        resid21 = float(F["resid21"][t, j]) if np.isfinite(F["resid21"][t, j]) else None
        if resid21 is not None and resid21 < -0.25:
            continue
        f = fund.loc[tk] if tk in fund.index else {}
        pio = f.get("Piotroski") if hasattr(f, "get") else None
        try:
            pio = float(pio) if pio is not None and np.isfinite(pio) else np.nan
        except Exception:
            pio = np.nan
        quality = "strong" if pio >= 6 else ("ok" if pio >= 4 else "weak")
        rec = dict(
            Ticker=tk,
            Sector=P["sectors"][j],
            Price=round(float(c[j]), 2),
            StormScore=round(float(comp[t, j]), 3),
            Shakeout=float(scores["shakeout_coil"][t, j]),
            Ret5=ret5,
            RangePos=float(F["rngpos63"][t, j]) if np.isfinite(F["rngpos63"][t, j]) else None,
            RSI=float(rsi[j]) if np.isfinite(rsi[j]) else None,
            Resid21=resid21,
            PeerCatchup=float(scores["peer_catchup"][t, j]) if np.isfinite(scores["peer_catchup"][t, j]) else None,
            LowerBB=float(scores["lower_bb"][t, j]) if np.isfinite(scores["lower_bb"][t, j]) else None,
            Range21=float(F["rng21"][t, j]) if np.isfinite(F["rng21"][t, j]) else None,
            VolRatio=float(F["vol_ratio"][t, j]) if np.isfinite(F["vol_ratio"][t, j]) else None,
            AboveMA50=bool(c[j] > F["sma50"][t, j]) if np.isfinite(F["sma50"][t, j]) else False,
            Piotroski=None if not np.isfinite(pio) else int(pio),
            Quality=quality,
            ShortPctFloat=f.get("ShortPctFloat") if hasattr(f, "get") else None,
            SuggestedStop=round(float(c[j]) * (1 + STOP_PCT), 2),
            Regime=regime_now,
        )
        rows.append(rec)
        if len(rows) >= n * 2:
            break
    df = pd.DataFrame(rows)
    qmap = {"strong": 0, "ok": 1, "weak": 2}
    df["qrank"] = df["Quality"].map(qmap)
    df = df.sort_values(["StormScore", "qrank"], ascending=[False, True]).head(n)
    return df.drop(columns="qrank")


def run_backtest(path: str, out_json: str | None = None, out_csv: str | None = None, n_picks: int = TOP_N):
    print(f"Loading {path} …")
    data = load_dump(path)
    P = build_panels(data)
    ok = universe_ok(P)
    print(f"Panel {P['C'].shape[0]} days × {P['C'].shape[1]} names, "
          f"tradeable on last bar: {int(ok[-1].sum())}")
    print("Computing features …")
    F = compute_features(P, ok)
    print("Scoring factors …")
    scores = factor_scores(F)
    regime = add_regime_score(scores, ok, P["C"])
    fwd = forward_returns(P["C"], P["H"], P["L"])
    print(f"Evaluating {len(scores)} methods (IS / OOS) …")
    table, meta = evaluate(scores, fwd, ok, P["dates"])
    table = table.sort_values("oos_ic10", ascending=False, na_position="last")
    print("\n=== FACTOR LEADERBOARD (ranked by OOS 10d Spearman IC) ===")
    show = table[["factor", "kind", "ic10_is", "ic10_oos", "ic5_oos", "ic21_oos",
                  "top_ex10_oos", "top_hw_oos", "bin_ex10_oos", "bin_hw_oos", "oos_edge"]].copy()
    pd.set_option("display.max_rows", 80)
    pd.set_option("display.width", 160)
    print(show.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    print("\nEvaluating live composite", LIVE_FACTORS, "…")
    comp = composite_score(scores)
    comp_row, meta2 = evaluate_composite(comp, fwd, ok, P["dates"])
    meta.update({k: v for k, v in meta2.items() if k.startswith("weekly")})
    print("Composite:", {k: comp_row.get(k) for k in
                         ("ic10_is", "ic10_oos", "top_ex10_is", "top_ex10_oos",
                          "top_hw_is", "top_hw_oos", "top_hit_oos")})
    print("Weekly top-N IS ", meta.get("weekly_topn_is"))
    print("Weekly top-N OOS", meta.get("weekly_topn_oos"))

    print("Evaluating regime-adaptive book …")
    adapt_row, adapt_meta = evaluate_composite(scores["regime_adapt"], fwd, ok, P["dates"])
    print("Regime-adapt:", {k: adapt_row.get(k) for k in
                            ("ic10_is", "ic10_oos", "top_ex10_oos", "top_hw_oos", "top_hit_oos")})
    print("Weekly top-N OOS", adapt_meta.get("weekly_topn_oos"))

    last_reg = float(regime[-1]) if np.isfinite(regime[-1]) else 0.0
    meta["regime_spread"] = last_reg
    meta["regime"] = "momentum" if last_reg > 0 else "reversal"

    picks = live_picks(P, F, scores, ok, n=n_picks, regime=regime)
    print(f"\n=== LIVE TOP {len(picks)} (as of {P['dates'][-1]}, regime={meta['regime']}) ===")
    print(picks.to_string(index=False))

    def _clean(d):
        return {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in d.items()}

    payload = dict(
        meta=meta,
        factors=table.replace({np.nan: None}).to_dict(orient="records"),
        composite=_clean(comp_row),
        regime_adapt=_clean(adapt_row),
        live_factors=list(LIVE_FACTORS),
        as_of=str(P["dates"][-1]),
    )
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nSaved {out_json}")
    if out_csv:
        picks.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
    return payload, picks


def score_from_panel(P: dict, n: int = TOP_N) -> Tuple[pd.DataFrame, dict]:
    ok = universe_ok(P)
    F = compute_features(P, ok)
    scores = factor_scores(F)
    regime = add_regime_score(scores, ok, P["C"])
    picks = live_picks(P, F, scores, ok, n=n, regime=regime)
    last_reg = float(regime[-1]) if np.isfinite(regime[-1]) else 0.0
    info = dict(
        as_of=str(P["dates"][-1])[:10],
        n_tradeable=int(ok[-1].sum()),
        n_names=int(P["C"].shape[1]),
        n_days=int(P["C"].shape[0]),
        regime="momentum" if last_reg > 0 else "reversal",
        regime_spread=last_reg,
    )
    return picks, info


def score_from_data(data: list, n: int = TOP_N) -> Tuple[pd.DataFrame, dict]:
    return score_from_panel(build_panels(data), n=n)


def score_live(path: str, n: int = TOP_N) -> Tuple[pd.DataFrame, dict]:
    return score_from_data(load_dump(path), n=n)


def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", default="data/storm_watch_backtest.json")
    ap.add_argument("--csv", default="data/storm_watch_picks.csv")
    ap.add_argument("--picks", type=int, default=TOP_N)
    a = ap.parse_args(argv)
    run_backtest(a.path, out_json=a.json, out_csv=a.csv, n_picks=a.picks)
    return 0


if __name__ == "__main__":
    _in_streamlit = False
    try:
        from streamlit.runtime import exists as _st_exists
        _in_streamlit = _st_exists()
    except Exception:
        pass
    if _in_streamlit:
        import streamlit as st
        st.error("storm_watch_engine.py is the batch/backtest script — "
                 "set Streamlit's main file to app.py.")
    else:
        sys.exit(main())
