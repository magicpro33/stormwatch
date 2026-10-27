"""Elections scanner — names that outperform around US votes.

The nightly dump is ~14 months on current tape (Aug 2025 → Oct 2026), so the
only live US cycle in the file is the 2026 midterm (election day 2026-11-03).
Presidential years are still encoded: a longer dump lights them up automatically.

Method, point-in-time, no lookahead:

  1. Every even year overlapping the panel maps to an election day (first
     Tuesday after November 1). Presidential = year % 4 == 0; else midterm.
  2. Each cycle is cut into the windows traders actually use: 63 / 21 / 10
     session pre-vote stretches, calendar months (Aug, Sep, Oct), and weekly
     excess vs the liquid universe.
  3. Each election-year autumn window is paired with the same calendar
     stretch in the prior odd year when that control is on tape — so a name
     that *always* rips in the fall does not look like an election winner.
  4. Scan rank = excess in the current pre-vote window + off-year alpha +
     monthly/weekly hit rate + last-21-session overlay (still working now).

Catalysts / 3 Magic Rules / SAFE FROM AI are applied by the tab, same as
the other scanners.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EL_VERSION = "1.0"

MIN_PRICE = 5.0
MIN_DOLLAR_VOL = 5e6
MIN_WINDOW_BARS = 8
PRE_CAL_DAYS = (63, 90)   # label, calendar days before election day
MONTHS_PRE = (8, 9, 10)   # Aug / Sep / Oct of the election year


def us_election_day(year: int) -> pd.Timestamp:
    """First Tuesday after November 1."""
    d = pd.Timestamp(year=int(year), month=11, day=1)
    offset = (1 - int(d.weekday())) % 7
    return (d + pd.Timedelta(days=int(offset))).normalize()


def cycle_kind(year: int) -> str:
    y = int(year)
    if y % 2:
        return "off"
    return "presidential" if y % 4 == 0 else "midterm"


def _ix_ge(dts: pd.DatetimeIndex, ts) -> int:
    ts = pd.Timestamp(ts).normalize()
    i = int(dts.searchsorted(ts, side="left"))
    return i if 0 <= i < len(dts) else -1


def _ix_le(dts: pd.DatetimeIndex, ts) -> int:
    ts = pd.Timestamp(ts).normalize()
    i = int(dts.searchsorted(ts, side="right")) - 1
    return i if 0 <= i < len(dts) else -1


def _window_ret(C: np.ndarray, i0: int, i1: int, min_bars: int = 3) -> np.ndarray:
    """Close-to-close return using the first and last real print in the span.

    Requiring C[i0] to be finite dropped the 2025 control for almost every
    name — the dump's union calendar starts a few sessions before most
    tickers' first bar.
    """
    n = C.shape[1]
    out = np.full(n, np.nan, dtype=np.float64)
    if i0 < 0 or i1 < 0 or i1 <= i0:
        return out
    sl = np.asarray(C[i0:i1 + 1], dtype=np.float64)
    finite = np.isfinite(sl) & (sl > 0)
    has = finite.any(axis=0)
    if not has.any():
        return out
    first = finite.argmax(axis=0)
    last = finite.shape[0] - 1 - finite[::-1].argmax(axis=0)
    cols = np.arange(n)
    a = sl[first, cols]
    b = sl[last, cols]
    ok = has & ((last - first) >= int(min_bars)) & (a > 0)
    out[ok] = b[ok] / a[ok] - 1.0
    return out


def _xs(ret: np.ndarray, uni: np.ndarray) -> tuple[np.ndarray, float]:
    """Excess vs equal-weight liquid universe. Universe mean ignores NaNs."""
    mkt = np.nan
    if uni is not None and uni.any():
        sl = ret[uni & np.isfinite(ret)]
        if sl.size:
            mkt = float(np.mean(sl))
    xs = np.full(ret.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(ret) & np.isfinite(mkt)
    xs[ok] = ret[ok] - mkt
    return xs, mkt


def _z(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):
        m = np.nanmean(y)
        s = np.nanstd(y)
    out = np.zeros_like(y)
    if np.isfinite(s) and s > 1e-12 and np.isfinite(m):
        ok = np.isfinite(y)
        out[ok] = (y[ok] - m) / s
    return out


def _rank_pct(x: np.ndarray) -> np.ndarray:
    """0–100 percentile of finite values; NaN stays NaN."""
    y = np.asarray(x, dtype=np.float64)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(y)
    n = int(ok.sum())
    if n <= 1:
        out[ok] = 50.0
        return out
    order = np.argsort(y[ok], kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    out[ok] = ranks / (n - 1) * 100.0
    return out


def discover_windows(dts) -> list[dict]:
    """Election / control windows that actually overlap this panel."""
    dts = pd.DatetimeIndex(pd.to_datetime(dts)).normalize()
    if dts.empty:
        return []
    first, last = dts[0], dts[-1]
    y0 = int(first.year) - 1
    y1 = int(last.year) + 1
    out = []
    for year in range(y0, y1 + 1):
        if year % 2:
            continue
        eday = us_election_day(year)
        kind = cycle_kind(year)
        spans = []
        for bars, cal in (("pre63", 90), ("pre21", 35), ("pre10", 16)):
            spans.append((bars, eday - pd.Timedelta(days=cal), min(eday, last)))
        for m in MONTHS_PRE:
            m0 = pd.Timestamp(year=year, month=m, day=1)
            m1 = (m0 + pd.offsets.MonthEnd(0)).normalize()
            spans.append((f"{year}-{m:02d}", m0, min(m1, last)))
        for name, start, end in spans:
            if end < first or start > last:
                continue
            i0 = _ix_ge(dts, max(start, first))
            i1 = _ix_le(dts, min(end, last))
            if i0 < 0 or i1 < 0 or (i1 - i0) < MIN_WINDOW_BARS:
                continue
            rec = dict(
                name=name, year=year, kind=kind, role="election",
                start=pd.Timestamp(dts[i0]), end=pd.Timestamp(dts[i1]),
                i0=i0, i1=i1, n=int(i1 - i0),
            )
            out.append(rec)
            c_start = start - pd.DateOffset(years=1)
            c_end = end - pd.DateOffset(years=1)
            ci0 = _ix_ge(dts, c_start)
            ci1 = _ix_le(dts, min(c_end, last))
            if ci0 >= 0 and ci1 >= 0 and (ci1 - ci0) >= MIN_WINDOW_BARS:
                out.append(dict(
                    name=f"{name}|ctrl", year=year - 1, kind="off",
                    role="control", pair=name, election_year=year,
                    start=pd.Timestamp(dts[ci0]), end=pd.Timestamp(dts[ci1]),
                    i0=ci0, i1=ci1, n=int(ci1 - ci0),
                ))
    return out


def _week_ends(dts: pd.DatetimeIndex, start, end) -> list[int]:
    """Session index of each week's last bar in [start, end]."""
    sl = dts[(dts >= pd.Timestamp(start)) & (dts <= pd.Timestamp(end))]
    if sl.empty:
        return []
    weeks = sl.to_series().groupby(sl.to_period("W-FRI")).last()
    ix = []
    for ts in weeks:
        i = _ix_le(dts, ts)
        if i >= 0:
            ix.append(i)
    return ix


def score_panel(panel, tickers, sectors, mdv, dts,
                min_price: float = MIN_PRICE,
                min_dollar_vol: float = MIN_DOLLAR_VOL):
    """Return (stock_frame, sector_frame, meta) scored on this panel."""
    C = np.asarray(panel["c"], dtype=np.float32)
    dts = pd.DatetimeIndex(pd.to_datetime(dts)).normalize()
    T, N = C.shape
    px = C[-1].astype(np.float64)
    mdv = np.asarray(mdv, dtype=np.float64)
    tickers = np.asarray(tickers)
    sectors = np.asarray(sectors)
    try:
        import cascade_engine as ce
        recent = ce._recent_ok_mask(panel)
    except Exception:
        recent = np.isfinite(C[-3:]).any(axis=0)

    liquid = (np.isfinite(px) & (px >= float(min_price))
              & (mdv >= float(min_dollar_vol)) & recent)
    windows = discover_windows(dts)
    elec = [w for w in windows if w["role"] == "election"]
    ctrl = {w.get("pair"): w for w in windows if w["role"] == "control"}

    years = sorted({w["year"] for w in elec})
    latest = years[-1] if years else None
    kind = cycle_kind(latest) if latest else "—"
    eday = us_election_day(latest) if latest else None

    def _pick(name_suffix):
        hits = [w for w in elec if w["name"] == name_suffix
                or w["name"].endswith(name_suffix)]
        if latest is not None:
            pref = [w for w in hits if w["year"] == latest]
            if pref:
                return pref[0]
        return hits[-1] if hits else None

    pre = _pick("pre63") or _pick("pre21") or _pick("pre10")
    pre21 = _pick("pre21")
    pre10 = _pick("pre10")
    months = []
    if latest is not None:
        for m in MONTHS_PRE:
            hit = _pick(f"{latest}-{m:02d}")
            if hit:
                months.append(hit)

    uni = liquid
    rows_meta = []
    pre_ret = pre_xs = ctrl_ret = ctrl_xs = None
    mkt_pre = np.nan
    if pre is not None:
        pre_ret = _window_ret(C, pre["i0"], pre["i1"])
        pre_xs, mkt_pre = _xs(pre_ret, uni)
        cwin = ctrl.get(pre["name"])
        if cwin is not None:
            ctrl_ret = _window_ret(C, cwin["i0"], cwin["i1"])
            ctrl_xs, _ = _xs(ctrl_ret, uni)
        rows_meta.append(pre)

    month_xs = []
    month_hit = np.zeros(N, dtype=np.float64)
    month_n = np.zeros(N, dtype=np.float64)
    for mw in months:
        r = _window_ret(C, mw["i0"], mw["i1"])
        xs, mkt = _xs(r, uni)
        month_xs.append(xs)
        got = np.isfinite(xs)
        month_n += got.astype(np.float64)
        month_hit += (got & (xs > 0)).astype(np.float64)
        rows_meta.append(mw)
    month_mean = (np.nanmean(np.vstack(month_xs), axis=0)
                  if month_xs else np.full(N, np.nan))

    # Weekly excess in the latest pre-vote stretch — many slices, one cycle.
    week_hit = np.zeros(N, dtype=np.float64)
    week_n = np.zeros(N, dtype=np.float64)
    if pre is not None:
        ends = _week_ends(dts, pre["start"], pre["end"])
        for i1 in ends:
            i0 = i1 - 5
            if i0 < 0:
                continue
            r = _window_ret(C, i0, i1)
            xs, _ = _xs(r, uni)
            got = np.isfinite(xs)
            week_n += got.astype(np.float64)
            week_hit += (got & (xs > 0)).astype(np.float64)
    week_rate = np.divide(week_hit, week_n, out=np.full(N, np.nan),
                          where=week_n >= 4)

    live = _window_ret(C, max(0, T - 22), T - 1)
    live_xs, mkt_live = _xs(live, uni)

    alpha = np.full(N, np.nan)
    if pre_xs is not None and ctrl_xs is not None:
        both = np.isfinite(pre_xs) & np.isfinite(ctrl_xs)
        alpha[both] = pre_xs[both] - ctrl_xs[both]

    # Blend: current pre-vote excess, off-year alpha, monthly mean, weekly
    # hit rate, live 21d. Missing alpha redistributes onto pre_xs.
    z_pre = _z(pre_xs) if pre_xs is not None else np.zeros(N)
    z_alpha = _z(alpha)
    z_month = _z(month_mean)
    z_week = _z(week_rate)
    z_live = _z(live_xs)
    has_alpha = np.isfinite(alpha)
    # Missing off-year print (IPO / short tape) is neutral on alpha, not a
    # bonus for raw momentum.
    raw = (0.30 * z_pre + 0.20 * np.where(has_alpha, z_alpha, 0.0)
           + 0.15 * z_month + 0.15 * z_week + 0.20 * z_live)

    if pre_xs is None:
        ok = np.zeros(N, dtype=bool)
    else:
        ok = liquid & np.isfinite(pre_xs)
    score = np.full(N, np.nan)
    if ok.any():
        score[ok] = _rank_pct(np.where(ok, raw, np.nan))[ok]

    hits_lbl = []
    weeks_lbl = []
    for j in range(N):
        if month_n[j] > 0:
            hits_lbl.append(f"{int(month_hit[j])}/{int(month_n[j])}")
        else:
            hits_lbl.append("—")
        if week_n[j] > 0:
            weeks_lbl.append(f"{int(week_hit[j])}/{int(week_n[j])}")
        else:
            weeks_lbl.append("—")

    df = pd.DataFrame({
        "Ticker": tickers,
        "Sector": sectors,
        "Score": score,
        "Elec%": pre_ret if pre_ret is not None else np.nan,
        "XS%": pre_xs if pre_xs is not None else np.nan,
        "Ctrl%": ctrl_xs if ctrl_xs is not None else np.nan,
        "Alpha": alpha,
        "Hits": hits_lbl,
        "Weeks": weeks_lbl,
        "Live21": live_xs,
        "Price": px,
        "_ok": ok,
    })

    sec_rows = pd.DataFrame()
    if pre_xs is not None:
        tmp = pd.DataFrame({
            "Sector": [str(s) for s in sectors],
            "XS%": pre_xs, "Elec%": pre_ret, "Alpha": alpha, "ok": ok,
        })
        tmp = tmp[tmp["ok"] & tmp["Sector"].ne("") & tmp["Sector"].ne("nan")
                   & tmp["Sector"].str.lower().ne("unknown")]
        if not tmp.empty:
            sec_rows = (tmp.groupby("Sector", sort=False)
                        .agg(Names=("XS%", "size"),
                             XS=("XS%", "mean"),
                             Elec=("Elec%", "mean"),
                             Alpha=("Alpha", "mean"),
                             Beat=("XS%", lambda s: float(np.mean(np.asarray(s) > 0))))
                        .reset_index()
                        .sort_values("XS", ascending=False)
                        .reset_index(drop=True))
            sec_rows.insert(0, "#", np.arange(1, len(sec_rows) + 1))
            sec_rows = sec_rows.rename(columns={
                "XS": "XS%", "Elec": "Elec%", "Beat": "Beat%",
            })

    used = []
    for w in rows_meta:
        used.append(dict(
            name=w["name"], kind=w["kind"], year=w["year"],
            start=str(pd.Timestamp(w["start"]).date()),
            end=str(pd.Timestamp(w["end"]).date()),
            n=w["n"],
            market=None,
        ))
    if pre is not None:
        used[0]["market"] = mkt_pre

    meta = dict(
        version=EL_VERSION,
        dump_start=str(dts[0].date()),
        dump_end=str(dts[-1].date()),
        election_year=latest,
        kind=kind,
        election_day=str(eday.date()) if eday is not None else None,
        n_tickers=int(N),
        n_liquid=int(liquid.sum()),
        n_windows=len(elec),
        n_controls=len(ctrl),
        market_pre=mkt_pre,
        market_live=mkt_live,
        pre_name=pre["name"] if pre else None,
        pre_start=str(pd.Timestamp(pre["start"]).date()) if pre else None,
        pre_end=str(pd.Timestamp(pre["end"]).date()) if pre else None,
        windows=used,
        years=years,
        days_to_vote=(int((eday - dts[-1]).days) if eday is not None else None),
    )
    return df, (sec_rows if isinstance(sec_rows, pd.DataFrame) else pd.DataFrame()), meta


def scan(min_price: float = MIN_PRICE, min_dollar_vol: float = MIN_DOLLAR_VOL,
         top: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rank the nightly dump. Returns (stocks, sectors, meta)."""
    import cascade_engine as ce
    panel, tickers, sectors, mdv, dts = ce.load_dump_panel()
    df, sec, meta = score_panel(
        panel, tickers, sectors, mdv, dts,
        min_price=min_price, min_dollar_vol=min_dollar_vol)
    out = df.loc[df["_ok"]].drop(columns="_ok")
    out = out.sort_values("Score", ascending=False).head(int(top)).reset_index(drop=True)
    out.insert(0, "#", np.arange(1, len(out) + 1))
    return out, sec, meta


def panel_from_gz(path: str):
    """Load a stock_data.json.gz the same way the dump panel is built."""
    import gzip
    import json
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    rows = [x for x in data if len((x.get("_hist") or {}).get("dates") or []) >= 120]
    all_d = sorted({d for x in rows for d in x["_hist"]["dates"]})
    dix = {d: i for i, d in enumerate(all_d)}
    T, N = len(all_d), len(rows)
    C = np.full((T, N), np.nan, dtype=np.float32)
    V = np.zeros((T, N), dtype=np.float32)
    tickers, sectors = [], []
    for j, x in enumerate(rows):
        h = x["_hist"]
        ix = [dix[d] for d in h["dates"]]
        C[ix, j] = h["close"]
        V[ix, j] = h.get("volume") or 0
        tickers.append(x["Ticker"])
        sectors.append(x.get("Sector") or "Unknown")
    C = pd.DataFrame(C).ffill(limit=5).values.astype(np.float32)
    with np.errstate(all="ignore"):
        mdv = np.nanmedian((C * np.nan_to_num(V))[-21:], axis=0)
    panel = dict(c=C, v=V)
    return panel, np.array(tickers), np.array(sectors), mdv, pd.to_datetime(all_d)


if __name__ == "__main__":
    import os
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.expanduser("~"), "Documents", "GitHub", "stock",
        "data", "stock_data.json.gz")
    panel, tickers, sectors, mdv, dts = panel_from_gz(path)
    df, sec, meta = score_panel(panel, tickers, sectors, mdv, dts)
    out = (df.loc[df["_ok"]].drop(columns="_ok")
           .sort_values("Score", ascending=False).head(20))
    print("coverage", {k: meta[k] for k in (
        "dump_start", "dump_end", "election_year", "kind", "election_day",
        "n_windows", "n_controls", "market_pre", "pre_start", "pre_end",
        "days_to_vote")})
    print(out.to_string(index=False,
                        formatters={"Elec%": "{:+.1%}".format,
                                    "XS%": "{:+.1%}".format,
                                    "Ctrl%": "{:+.1%}".format,
                                    "Alpha": "{:+.1%}".format,
                                    "Live21": "{:+.1%}".format,
                                    "Score": "{:.0f}".format,
                                    "Price": "{:.2f}".format}))
    if sec is not None and not sec.empty:
        print("\nsectors")
        print(sec.head(10).to_string(index=False,
                                     formatters={"Elec%": "{:+.1%}".format,
                                                 "XS%": "{:+.1%}".format,
                                                 "Alpha": "{:+.1%}".format,
                                                 "Beat%": "{:.0%}".format}))
