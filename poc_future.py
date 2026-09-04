"""
POC FUTURE — AMD (Accumulation / Manipulation / Distribution) scanner.

A screener port of the "AMD + Volume Profile" Pine indicator. The pattern, as
the script detects it:

  1. ACCUMULATION  price coils — the whole high-low range over `accum_len` bars
                   is no wider than `max_range_atr` ATRs. That window is locked.
  2. VOLUME PROFILE built from ONLY the accumulation bars, giving POC/VAH/VAL.
  3. MANIPULATION  price sweeps beyond the range (a wick through the low),
                   grabbing the stops resting there.
  4. DISTRIBUTION  price closes back inside and reclaims the POC. That is the
                   trigger; the trade runs opposite the sweep.

WHAT THE PINE AUTHOR'S OWN BACKTEST FOUND (carried over verbatim, because it
decides the defaults here):

    LONG, enter on POC reclaim, target far side of range, accum_len 15
        1,175 trades  77.8% hit  +0.178R  PF 1.83
        positive in 10 of 10 months, both ticker halves, both date halves,
        and in 21 of 21 parameter combinations swept.

    SHORTS LOST MONEY IN EVERY CONFIGURATION TESTED (-0.08R to -0.26R).
    So this scanner is LONG-ONLY, exactly like the indicator's defaults.

    Fills were modelled at the NEXT bar's open with a 10bp round trip. Filling
    at the signal bar's close turned -0.057R into +0.147R, which is the
    difference between a real edge and reading the future.

Sample was ~11 months of one regime, and the September-2026 replication covers
the same window shifted a day — a replication, not an out-of-sample test.
Daily bars only; the method is usually traded intraday and that is untested.

The scanner reports each stock's stage, so you can watch a coil BEFORE it
sweeps rather than only catching entries that already fired.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# defaults mirror the Pine script's tuned inputs
POC_VERSION    = "1.2"   # shown in the tab — confirms which build is deployed

ACCUM_LEN      = 15      # retuned from 20; 4.6x more setups at equal expectancy
MAX_RANGE_ATR  = 2.5     # the coil must be no wider than this many ATRs
ATR_LEN        = 14
MAX_WAIT       = 30      # bars to wait for a sweep before abandoning the range
MIN_SWEEP_ATR  = 0.10    # sweep must reach this far past the range
MAX_ENTRY_WAIT = 10      # bars from sweep to reclaim before the setup expires
STOP_BUF_ATR   = 0.25    # stop sits this far beyond the sweep extreme
VP_BINS        = 24
VA_PCT         = 70.0
MIN_RR         = 0.30    # POC entries win on hit rate, not payoff — median RR ~0.39
MAX_RR         = 4.00


def _atr(h, l, c, n=ATR_LEN):
    """Wilder-style ATR on (T,) arrays, tolerant of gaps.

    The dump panel only forward-fills CLOSES — highs, lows and volume keep
    their NaNs wherever a symbol had no bar. A plain rolling mean turns a
    single NaN into a NaN ATR for the next 14 bars, which made scan_symbol
    bail out on 2,485 of 2,486 names. min_periods lets the average form from
    whatever real bars are in the window.
    """
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=max(3, n // 3)).mean().values


def _profile(h, l, c, v, lo, hi, bins=VP_BINS, va_pct=VA_PCT):
    """POC / VAH / VAL from typical-price-weighted volume, as the Pine does.

    Each bar's volume is dropped into the bin holding its typical price
    (H+L+C)/3, then the value area grows outward from the POC bin, always
    taking the heavier neighbour, until it holds `va_pct` of the volume.
    """
    if not (hi > lo):
        return np.nan, np.nan, np.nan
    step = (hi - lo) / bins
    arr = np.zeros(bins)
    tp = (h + l + c) / 3.0
    ix = np.clip(((tp - lo) / step).astype(int), 0, bins - 1)
    for k, vol in zip(ix, v):
        arr[k] += vol
    tot = arr.sum()
    if tot <= 0:
        return np.nan, np.nan, np.nan
    pi = int(np.argmax(arr))
    acc = arr[pi]; lo_i = hi_i = pi
    target = tot * va_pct / 100.0
    for _ in range(bins):
        if acc >= target:
            break
        dn = arr[lo_i - 1] if lo_i > 0 else -1.0
        up = arr[hi_i + 1] if hi_i < bins - 1 else -1.0
        if dn < 0 and up < 0:
            break
        if up >= dn:
            hi_i += 1; acc += arr[hi_i]
        else:
            lo_i -= 1; acc += arr[lo_i]
    poc = lo + (pi + 0.5) * step
    val = lo + lo_i * step
    vah = lo + (hi_i + 1) * step
    return poc, vah, val


def scan_symbol(h, l, c, v, accum_len=ACCUM_LEN, max_range_atr=MAX_RANGE_ATR,
                min_sweep_atr=MIN_SWEEP_ATR, max_wait=MAX_WAIT,
                max_entry_wait=MAX_ENTRY_WAIT, stop_buf_atr=STOP_BUF_ATR,
                search_back=45):
    """Evaluate the stock's CURRENT structure and return its live stage.

    Design note — this is deliberately not a forward replay. An earlier version
    walked history greedily, locked the first coil it met and reported whatever
    state it ended in; on a year of bars that meant 1,003 of 2,479 names showed
    "TRIGGERED" with a MEDIAN age of 99 sessions. Technically true, useless as a
    scan: it answered "did this ever happen?" rather than "what is this stock
    doing now?"

    So instead we search BACKWARD for the most recent accumulation window, then
    walk forward from it to classify what has happened since:

        COILING    the range is locked and no sweep yet (still within max_wait)
        SWEPT      the lows were taken and the reclaim window is still open
        TRIGGERED  price closed back above the POC — entry is live

    Anything older than that has expired and returns None, so every row on the
    board is current by construction.
    """
    # Compact to the symbol's REAL bars. The panel is a dense date grid, so a
    # stock that listed late or missed prints carries NaNs; highs/lows are not
    # forward-filled at all. Working on the valid rows keeps every window
    # (coil, sweep, reclaim) counted in actual sessions.
    c = np.asarray(c, dtype=float)
    good = np.isfinite(c)
    if good.sum() < accum_len + ATR_LEN + 5:
        return None
    c = c[good]
    h = np.asarray(h, dtype=float)[good]
    l = np.asarray(l, dtype=float)[good]
    v = np.nan_to_num(np.asarray(v, dtype=float)[good])
    # a missing high/low on a real bar falls back to the close
    h = np.where(np.isfinite(h), h, c)
    l = np.where(np.isfinite(l), l, c)

    T = len(c)
    if T < accum_len + ATR_LEN + 5:
        return None
    atr = _atr(h, l, c)
    if not np.isfinite(atr[-1]) or atr[-1] <= 0:
        return None
    last = T - 1

    # most recent bar at which a compressed window could have closed
    lo_k = max(accum_len, ATR_LEN) + 1
    for k in range(last, max(lo_k, last - search_back) - 1, -1):
        a = atr[k]
        if not np.isfinite(a) or a <= 0:
            continue
        s0 = k - accum_len + 1
        if s0 < 0:
            continue
        w_hi = np.nanmax(h[s0:k + 1])
        w_lo = np.nanmin(l[s0:k + 1])
        if not (np.isfinite(w_hi) and np.isfinite(w_lo)) or w_hi <= w_lo:
            continue
        if (w_hi - w_lo) > a * max_range_atr:
            continue                      # not a coil

        poc, vah, val = _profile(h[s0:k + 1], l[s0:k + 1], c[s0:k + 1],
                                 v[s0:k + 1], w_lo, w_hi)
        if not np.isfinite(poc):
            continue

        depth = a * min_sweep_atr
        # did the lows get swept after the coil closed?
        sweep_bar = -1
        sweep_px = np.nan
        for i in range(k + 1, min(k + max_wait, last) + 1):
            if l[i] < w_lo - depth:
                sweep_bar = i
                sweep_px = float(np.nanmin(l[k + 1:i + 1]))
                break

        px = float(c[last])
        base = dict(poc=float(poc), vah=float(vah), val=float(val),
                    acc_hi=float(w_hi), acc_lo=float(w_lo), atr=float(a),
                    price=px, range_atr=float((w_hi - w_lo) / a),
                    dist_to_poc=float((px - poc) / px * 100))

        if sweep_bar < 0:
            if (last - k) <= max_wait:    # still coiling, sweep may yet come
                base.update(stage="COILING", bars_in_stage=int(last - k),
                            entry=np.nan, stop=np.nan, target=np.nan, rr=np.nan)
                return base
            continue                      # coil went stale — look further back

        # deepest point of the sweep, and the reclaim search window
        w_end = min(sweep_bar + max_entry_wait, last)
        sweep_px = float(np.nanmin(l[k + 1:w_end + 1]))
        fired = -1
        for i in range(sweep_bar, w_end + 1):
            if c[i] > poc:
                fired = i
                break

        stop = float(sweep_px - a * stop_buf_atr)
        entry = float(poc)
        target = float(w_hi)
        risk = entry - stop
        rr = (target - entry) / risk if risk > 0 else np.nan
        base.update(sweep_px=sweep_px, stop=stop, entry=entry, target=target,
                    rr=float(rr) if np.isfinite(rr) else np.nan)

        if fired >= 0:
            base.update(stage="TRIGGERED", bars_in_stage=int(last - fired))
            return base
        if (last - sweep_bar) <= max_entry_wait:
            base.update(stage="SWEPT", bars_in_stage=int(last - sweep_bar))
            return base
        continue                          # reclaim window closed — keep looking

    return None


def scan(min_price: float = 5.0, min_dollar_vol: float = 5e6,
         stages: tuple = ("TRIGGERED", "SWEPT", "COILING"),
         min_rr: float = MIN_RR, max_rr: float = MAX_RR,
         only_sectors: list | None = None, top: int = 50,
         max_bars_ago: int = 5,
         accum_len: int = ACCUM_LEN,
         max_range_atr: float = MAX_RANGE_ATR) -> pd.DataFrame:
    """Scan the nightly dump for AMD setups. Long-only, per the backtest."""
    import cascade_engine as ce
    panel, tickers, sectors, mdv, dts = ce.load_dump_panel()
    o, h, l, c, v = (panel["o"], panel["h"], panel["l"], panel["c"], panel["v"])
    px = c[-1]
    ok = (np.isfinite(px) & (px >= min_price) & (mdv >= min_dollar_vol)
          & ce._recent_ok_mask(panel))
    if only_sectors:
        want = {str(s) for s in only_sectors}
        ok = ok & np.array([str(s) in want for s in sectors])

    rows = []
    for j in np.where(ok)[0]:
        try:
            r = scan_symbol(h[:, j], l[:, j], c[:, j], np.nan_to_num(v[:, j]),
                            accum_len=accum_len, max_range_atr=max_range_atr)
        except Exception:
            continue
        if not r or r["stage"] not in stages:
            continue
        # a trigger from three weeks ago is history, not a setup
        if r["stage"] == "TRIGGERED" and r["bars_in_stage"] > max_bars_ago:
            continue
        if r["stage"] in ("SWEPT", "TRIGGERED"):
            rr = r.get("rr")
            if not np.isfinite(rr) or rr < min_rr or rr > max_rr:
                continue
        r["Ticker"] = str(tickers[j])
        r["Sector"] = str(sectors[j])
        rows.append(r)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # freshest, tightest coils first — a setup that just triggered outranks one
    # that fired a week ago, and a tighter range is a cleaner base
    order = {"TRIGGERED": 0, "SWEPT": 1, "COILING": 2}
    df["_o"] = df.stage.map(order)
    df = df.sort_values(["_o", "bars_in_stage", "range_atr"],
                        ascending=[True, True, True]).drop(columns="_o")
    df = df.rename(columns={"stage": "Stage", "price": "Price", "poc": "POC",
                            "vah": "VAH", "val": "VAL", "entry": "Entry",
                            "stop": "Stop", "target": "Target", "rr": "R:R",
                            "range_atr": "RangeATR", "bars_in_stage": "BarsAgo",
                            "dist_to_poc": "ToPOC%", "acc_hi": "RangeHigh",
                            "acc_lo": "RangeLow"})
    cols = ["Ticker", "Sector", "Stage", "BarsAgo", "Price", "POC", "ToPOC%",
            "Entry", "Stop", "Target", "R:R", "RangeATR", "RangeLow",
            "RangeHigh", "VAH", "VAL"]
    return df[[c_ for c_ in cols if c_ in df.columns]].head(top).reset_index(drop=True)


def diagnose(min_price: float = 5.0, min_dollar_vol: float = 5e6,
             accum_len: int = ACCUM_LEN,
             max_range_atr: float = MAX_RANGE_ATR) -> dict:
    """Report the funnel on THIS deployment's data.

    Written because the scanner returned ~750 setups locally and 2 on the live
    app off the same nightly dump. Rather than guess across machines, the app
    now reports its own counts at every stage.
    """
    import cascade_engine as ce
    panel, tickers, sectors, mdv, dts = ce.load_dump_panel()
    h, l, c, v = panel["h"], panel["l"], panel["c"], panel["v"]
    px = c[-1]
    n_total = len(px)
    f_price = np.isfinite(px) & (px >= min_price)
    f_liq = f_price & (mdv >= min_dollar_vol)
    recent = ce._recent_ok_mask(panel)
    f_all = f_liq & recent

    out = dict(version=POC_VERSION, bars=int(c.shape[0]),
               last_date=str(pd.to_datetime(dts[-1]).date()),
               universe=n_total, after_price=int(f_price.sum()),
               after_liquidity=int(f_liq.sum()),
               recent_ok=int(recent.sum()), scannable=int(f_all.sum()))

    # how many even have enough bars / a valid ATR?
    need = accum_len + ATR_LEN + 5
    out["enough_bars"] = int(c.shape[0] >= need)
    stages = {"COILING": 0, "SWEPT": 0, "TRIGGERED": 0, "NONE": 0}
    coils_seen = 0
    for j in np.where(f_all)[0]:
        r = scan_symbol(h[:, j], l[:, j], c[:, j], np.nan_to_num(v[:, j]),
                        accum_len=accum_len, max_range_atr=max_range_atr)
        if r is None:
            stages["NONE"] += 1
        else:
            stages[r["stage"]] += 1
            coils_seen += 1
    out["stages_raw"] = stages
    out["setups_before_filters"] = coils_seen
    return out
