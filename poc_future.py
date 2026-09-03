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
    """Wilder-style ATR on (T,) arrays."""
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n).mean().values


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
                max_entry_wait=MAX_ENTRY_WAIT, stop_buf_atr=STOP_BUF_ATR):
    """Run the AMD state machine over one symbol's daily bars.

    Returns the CURRENT state at the last bar, or None if the symbol is not in
    a setup. Stages, in the order they occur:

        COILING   an accumulation range is locked, waiting for the sweep
        SWEPT     the lows have been swept, waiting for the POC reclaim
        TRIGGERED the reclaim happened on the final bar — the entry is live
    """
    T = len(c)
    if T < accum_len + ATR_LEN + 5:
        return None
    atr = _atr(h, l, c)
    if not np.isfinite(atr[-1]) or atr[-1] <= 0:
        return None

    state = 0                     # 0 hunting | 1 range locked | 2 swept
    acc_hi = acc_lo = np.nan
    acc_start = acc_end = -1
    acc_atr = np.nan              # ATR AT LOCK TIME — range width is judged
    poc = vah = val = np.nan      # against the volatility that formed it
    sweep_px = np.nan
    sweep_bar = -1
    fired_bar = -1
    # the machine keeps running to the last bar so we report the CURRENT state.
    # Breaking on the first trigger would surface a setup from months ago as if
    # it were live; every symbol eventually triggers once over a year of bars.
    last = None

    start = max(accum_len, ATR_LEN) + 1
    for i in range(start, T):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue

        if state == 0:
            w_hi = np.nanmax(h[i - accum_len + 1:i + 1])
            w_lo = np.nanmin(l[i - accum_len + 1:i + 1])
            if np.isfinite(w_hi) and np.isfinite(w_lo) and w_hi > w_lo \
                    and (w_hi - w_lo) <= a * max_range_atr:
                acc_hi, acc_lo = w_hi, w_lo
                acc_start, acc_end = i - accum_len + 1, i
                acc_atr = a
                sl = slice(acc_start, acc_end + 1)
                poc, vah, val = _profile(h[sl], l[sl], c[sl], v[sl], acc_lo, acc_hi)
                if np.isfinite(poc):
                    state = 1
            continue

        if state == 1:
            if (i - acc_end) > max_wait:          # coil went stale
                state = 0
                continue
            depth = a * min_sweep_atr
            if l[i] < acc_lo - depth:             # LONG side: lows swept
                state = 2
                sweep_px = l[i]
                sweep_bar = i
            continue

        if state == 2:
            if (i - sweep_bar) > max_entry_wait:  # never reclaimed
                state = 0
                continue
            sweep_px = min(sweep_px, l[i])        # track a deeper sweep
            if np.isfinite(poc) and c[i] > poc:   # DISTRIBUTION: POC reclaimed
                fired_bar = i
                last = dict(kind="TRIGGERED", bar=i, poc=poc, vah=vah, val=val,
                            acc_hi=acc_hi, acc_lo=acc_lo, acc_atr=acc_atr,
                            sweep_px=sweep_px)
                state = 0                        # hunt the next coil
            continue

    # A live coil or sweep outranks a past trigger; otherwise report the most
    # recent trigger and let BarsAgo say how stale it is.
    if state == 1:
        stage, bar = "COILING", acc_end
    elif state == 2:
        stage, bar = "SWEPT", sweep_bar
    elif last is not None:
        stage, bar = "TRIGGERED", last["bar"]
        poc, vah, val = last["poc"], last["vah"], last["val"]
        acc_hi, acc_lo, acc_atr = last["acc_hi"], last["acc_lo"], last["acc_atr"]
        sweep_px = last["sweep_px"]
    else:
        return None

    px = float(c[-1])
    a = float(acc_atr) if np.isfinite(acc_atr) and acc_atr > 0 else float(atr[-1])
    out = dict(stage=stage, bars_in_stage=int(T - 1 - bar),
               poc=float(poc) if np.isfinite(poc) else np.nan,
               vah=float(vah) if np.isfinite(vah) else np.nan,
               val=float(val) if np.isfinite(val) else np.nan,
               acc_hi=float(acc_hi), acc_lo=float(acc_lo), atr=a, price=px,
               range_atr=float((acc_hi - acc_lo) / a) if a > 0 else np.nan,
               dist_to_poc=float((px - poc) / px * 100) if np.isfinite(poc) else np.nan)

    if stage == "COILING":
        out.update(entry=np.nan, stop=np.nan, target=np.nan, rr=np.nan)
        return out

    stop = float(sweep_px - a * stop_buf_atr)
    target = float(acc_hi)                        # far side of the range
    entry = float(poc)
    risk = entry - stop
    rr = (target - entry) / risk if risk > 0 else np.nan
    out.update(sweep_px=float(sweep_px), stop=stop, target=target, entry=entry,
               rr=float(rr) if np.isfinite(rr) else np.nan)
    return out


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
