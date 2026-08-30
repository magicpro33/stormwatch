"""
cascade_engine.py — the Global Flow Cascade engine behind Money Weather.

Concept: money propagates through the world's assets in repeatable paths —
fast, frictionless nodes react first (crypto, FX, semis), slow heavy ones
last. This engine estimates the directed lead-lag graph empirically, detects
flow waves entering upstream nodes, and forecasts the downstream nodes the
wave historically reaches — with the lag and hit rate attached.

Layers:
  0. Forced Flow Calendar  — mechanical, scheduled flows (rebalances, OpEx…)
  1. Pressure System       — global net liquidity nowcast (FRED + stablecoins)
  2. Sentinels             — 24/7 early-warning assets
  3. Cascade Graph         — the storm tracks themselves

All estimation is walk-forward-safe: edges at time t use only data <= t.
"""
import os
import io
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

# ── universe of nodes ────────────────────────────────────────────────
# symbol: (label, group). Groups: sector, factor, country, commodity,
# rates, fx, crypto, theme, vol
NODES = {
    "SPY": ("S&P 500", "core"), "QQQ": ("Nasdaq 100", "core"),
    "IWM": ("Small Caps", "core"), "DIA": ("Dow", "core"),
    "XLK": ("Technology", "sector"), "XLF": ("Financials", "sector"),
    "XLE": ("Energy", "sector"), "XLV": ("Healthcare", "sector"),
    "XLI": ("Industrials", "sector"), "XLY": ("Cons. Cyclical", "sector"),
    "XLP": ("Cons. Staples", "sector"), "XLB": ("Materials", "sector"),
    "XLU": ("Utilities", "sector"), "XLRE": ("Real Estate", "sector"),
    "XLC": ("Communication", "sector"),
    "SMH": ("Semiconductors", "theme"), "XBI": ("Biotech", "theme"),
    "ITA": ("Defense", "theme"), "KRE": ("Regional Banks", "theme"),
    "IYT": ("Transports", "theme"), "TAN": ("Solar", "theme"),
    "URA": ("Uranium", "theme"), "COPX": ("Copper Miners", "theme"),
    "GDX": ("Gold Miners", "theme"), "ARKK": ("High Beta Innov.", "theme"),
    "MTUM": ("Momentum", "factor"), "VLUE": ("Value", "factor"),
    "QUAL": ("Quality", "factor"), "USMV": ("Low Vol", "factor"),
    "EEM": ("Emerging Mkts", "country"), "FXI": ("China", "country"),
    "EWJ": ("Japan", "country"), "EWG": ("Germany", "country"),
    "INDA": ("India", "country"), "EWZ": ("Brazil", "country"),
    "GLD": ("Gold", "commodity"), "SLV": ("Silver", "commodity"),
    "CPER": ("Copper", "commodity"), "USO": ("Oil", "commodity"),
    "UNG": ("Nat Gas", "commodity"), "DBA": ("Agriculture", "commodity"),
    "TLT": ("20y Treasuries", "rates"), "IEF": ("10y Treasuries", "rates"),
    "HYG": ("High Yield", "rates"), "LQD": ("IG Credit", "rates"),
    "TIP": ("TIPS", "rates"),
    "UUP": ("US Dollar", "fx"), "FXY": ("Yen", "fx"), "FXE": ("Euro", "fx"),
    "BTC-USD": ("Bitcoin", "crypto"), "ETH-USD": ("Ethereum", "crypto"),
    "SOL-USD": ("Solana", "crypto"),
    "^VIX": ("VIX", "vol"),
    # ── wide-net expansion ──
    "XOP": ("Oil & Gas E&P", "theme"), "OIH": ("Oil Services", "theme"),
    "XME": ("Metals & Mining", "theme"), "KWEB": ("China Internet", "theme"),
    "IGV": ("Software", "theme"), "CIBR": ("Cybersecurity", "theme"),
    "XHB": ("Homebuilders", "theme"), "JETS": ("Airlines", "theme"),
    "XRT": ("Retail", "theme"), "PBW": ("Clean Energy", "theme"),
    "LIT": ("Lithium/Battery", "theme"), "IBB": ("Biotech LC", "theme"),
    "RSP": ("S&P Equal Weight", "breadth"),
    "EWY": ("South Korea", "country"), "EWT": ("Taiwan", "country"),
    "EWU": ("UK", "country"), "EWC": ("Canada", "country"),
    "EWA": ("Australia", "country"), "EWW": ("Mexico", "country"),
    "SHY": ("2y Treasuries", "rates"), "EMB": ("EM Debt", "rates"),
    "MBB": ("Mortgages", "rates"), "BKLN": ("Bank Loans", "rates"),
    "PPLT": ("Platinum", "commodity"), "CORN": ("Corn", "commodity"),
    "WEAT": ("Wheat", "commodity"),
    "FXB": ("British Pound", "fx"),
    "^N225": ("Nikkei 225", "country"),
}

# canonical upstream sentinels (fast, frictionless)
# ── sector-flow constants (defined early: mega_scan uses them as defaults) ──
SECTOR_FLOW_LOOKBACK = 1
SECTOR_FLOW_WEIGHT = 8.0        # points added to the ~100-point cascade score
SECTOR_FLOW_MAX_BACK = 15       # how far back the day/range pickers may go

ENGINE_VERSION = "2.24"   # app.py checks this — push both files together

SENTINELS = ["BTC-USD", "ETH-USD", "FXY", "CPER", "GLD", "SMH", "HYG", "^VIX",
             "KRE", "EMB", "UUP", "TLT", "^N225"]

# ratio sentinels — relationships that lead, computed from node closes
RATIO_SENTINELS = {
    "SMH/SPY":  ("Semis vs Market", "the AI-cycle leader — semis roll over before the index does"),
    "XLY/XLP":  ("Discretionary vs Staples", "consumer risk appetite — falling = defensive rotation"),
    "HYG/IEF":  ("Junk vs Treasuries", "credit risk appetite — the bond market's fear gauge"),
    "CPER/GLD": ("Copper vs Gold", "growth vs fear — Dr. Copper against the bunker asset"),
    "RSP/SPY":  ("Equal vs Cap Weight", "breadth — narrow rallies (falling ratio) are fragile"),
    "IWM/SPY":  ("Small vs Large", "risk breadth — small caps lead risk-on and risk-off"),
    "EEM/SPY":  ("EM vs US", "global liquidity reach — EM outperforms when dollars flow out"),
    "FXY/UUP":  ("Yen vs Dollar", "the carry trade's engine — yen surging against the dollar = unwind risk for every yen-funded position on earth"),
}


def ratio_sentinel_impulses(closes: pd.DataFrame) -> pd.DataFrame:
    """Impulse z + 63d trend for each ratio sentinel available in the data."""
    rows = []
    for pair, (name, meaning) in RATIO_SENTINELS.items():
        a, b = pair.split("/")
        if a not in closes.columns or b not in closes.columns:
            continue
        r = (closes[a] / closes[b]).dropna()
        if len(r) < IMPULSE_Z_WIN:
            continue
        imp5 = r.pct_change(IMPULSE_W)
        mu = imp5.rolling(IMPULSE_Z_WIN, min_periods=60).mean()
        sd = imp5.rolling(IMPULSE_Z_WIN, min_periods=60).std()
        z = float(((imp5 - mu) / sd).iloc[-1])
        t63 = float(r.iloc[-1] / r.iloc[-64] - 1) if len(r) > 64 else np.nan
        rows.append(dict(pair=pair, name=name, meaning=meaning,
                         z=round(z, 2) if np.isfinite(z) else np.nan,
                         trend63=round(t63, 4) if np.isfinite(t63) else np.nan))
    return pd.DataFrame(rows)

LOCAL_HISTORY = os.path.join(os.path.dirname(__file__), "data", "history.parquet")
HISTORY_YEARS = 3

IMPULSE_W = 5          # days for the impulse return
IMPULSE_Z_WIN = 126    # z-score window
EDGE_TRAIN = 252       # days used to estimate edges (walk-forward)
EDGE_HORIZON = 10      # forward days an edge predicts
EDGE_MIN_ABS_IC = 0.13
WAVE_Z = 1.25          # |impulse z| to call a node "active"


# ── data ─────────────────────────────────────────────────────────────
def fetch_history(years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Daily closes for all nodes. Local parquet first (offline/test seam),
    then yfinance batch download. Returns DataFrame[date x symbol]."""
    if os.path.exists(LOCAL_HISTORY):
        df = pd.read_parquet(LOCAL_HISTORY)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        # refetch when stale (>4 calendar days) — but keep the stale copy
        # unless the fresh download actually succeeds (stale beats empty)
        if (pd.Timestamp.today() - df.index[-1]).days <= 4:
            # align on LOAD as well as after a fetch — a parquet written before
            # the calendar fix still contains weekend rows
            return _align_to_equity_calendar(df)
        stale_df = df
    else:
        stale_df = None
    global LAST_HISTORY_SOURCE
    closes = alpaca_history(list(NODES), years)          # ── PRIMARY: Alpaca
    missing = [s for s in NODES if s not in closes.columns
               or closes[s].dropna().empty] if not closes.empty else list(NODES)
    if missing:                                           # ── fallback: yfinance
        os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")
        import yfinance as yf
        start = (date.today() - timedelta(days=int(years * 365.25 + 30))).isoformat()
        raw = yf.download(missing, start=start, auto_adjust=True,
                          progress=False, group_by="column")
        yfc = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        if isinstance(yfc, pd.Series):
            yfc = yfc.to_frame(missing[0])
        yfc = yfc.dropna(how="all")
        yfc.index = pd.to_datetime(yfc.index).tz_localize(None)
        closes = yfc if closes.empty else closes.join(yfc, how="outer")
    LAST_HISTORY_SOURCE = ("Alpaca (primary)" + (f" + yfinance ({len(missing)} symbols)" if missing else "")
                           ) if len(missing) < len(NODES) else "yfinance (Alpaca keys not set)"
    # weekends out BEFORE the ffill, so Friday's equity close is never carried
    # onto a Saturday/Sunday row created by the crypto columns
    closes = _align_to_equity_calendar(closes.dropna(how="all")).ffill(limit=5)
    if len(closes) < 30 or closes.shape[1] < 3:      # download failed
        if stale_df is not None:
            LAST_HISTORY_SOURCE = "stale cache (feeds unreachable — will retry)"
            return _align_to_equity_calendar(stale_df)
        return closes
    try:
        closes.to_parquet(LOCAL_HISTORY)
    except Exception:
        pass
    return closes


def _align_to_equity_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Drop weekend rows when any non-crypto column is present.

    Crypto trades every calendar day; equities trade weekdays. Joining them and
    forward-filling copied Friday's SPY onto Sat/Sun, so pct_change(5) treated
    those as real sessions with ~0% equity return — corrupting impulse
    z-scores, Spearman ICs, waves, sentinels and the backtest's "252 days".
    Monday's crypto bar still carries the weekend move. Holidays stay (the
    caller ffills them). A crypto-only frame keeps its weekends.
    """
    if df is None or df.empty:
        return df
    crypto = {c for c in df.columns if str(c).upper().endswith("-USD")}
    if not (set(df.columns) - crypto):
        return df                      # crypto only — weekends are real
    idx = pd.to_datetime(df.index)
    return df[idx.dayofweek < 5]


def refresh_history():
    """Force a fresh download, but restore the previous cache if the feeds
    fail — a manual refresh should never leave you with less than you had."""
    backup = None
    try:
        backup = pd.read_parquet(LOCAL_HISTORY)
        backup.index = pd.to_datetime(backup.index)
    except Exception:
        pass
    try:
        os.remove(LOCAL_HISTORY)
    except FileNotFoundError:
        pass
    fresh = fetch_history()
    if (fresh is None or len(fresh) < 30) and backup is not None:
        global LAST_HISTORY_SOURCE
        try:
            backup.to_parquet(LOCAL_HISTORY)
        except Exception:
            pass
        LAST_HISTORY_SOURCE = "previous cache (refresh failed — feeds unreachable)"
        return backup
    return fresh


# ── impulse + graph ──────────────────────────────────────────────────
def impulses(closes: pd.DataFrame) -> pd.DataFrame:
    """Cross-time z-score of the 5d return for every node — 'flow impulse'."""
    r = closes.pct_change(IMPULSE_W)
    mu = r.rolling(IMPULSE_Z_WIN, min_periods=60).mean()
    sd = r.rolling(IMPULSE_Z_WIN, min_periods=60).std()
    return (r - mu) / sd


def estimate_edges(closes: pd.DataFrame, asof: int | None = None,
                   train: int = EDGE_TRAIN, horizon: int = EDGE_HORIZON,
                   min_ic: float = EDGE_MIN_ABS_IC,
                   imp: pd.DataFrame | None = None,
                   fwd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Directed lead-lag edges i -> j estimated on data up to `asof` (iloc).
    Edge = Spearman IC between impulse_i(t) and fwd-return_j(t+1..t+horizon).
    Walk-forward safe. Vectorized: all pairs in one rank-correlation pass
    (was a Python double loop — ~50x faster, identical semantics)."""
    if imp is None:
        imp = impulses(closes)
    if fwd is None:
        fwd = closes.shift(-horizon) / closes - 1.0
    T = len(closes)
    end = T - 1 if asof is None else asof
    lo = max(0, end - train)
    ii = imp.iloc[lo:end - horizon]
    ff = fwd.iloc[lo:end - horizon]
    cols = [c for c in closes.columns if ii[c].notna().sum() > 60]
    if not cols:
        return pd.DataFrame(columns=["source", "target", "ic", "hit_rate",
                                     "source_name", "target_name"])
    ii, ff = ii[cols], ff[cols]

    # rank-transform per column (Spearman = Pearson on ranks), then one
    # masked matmul gives every pairwise IC with pairwise-complete counts
    def _std_ranks(df):
        r = df.rank()
        m = df.notna().values
        v = r.values.copy()
        mu = np.nanmean(np.where(m, v, np.nan), axis=0)
        sd = np.nanstd(np.where(m, v, np.nan), axis=0)
        sd[sd == 0] = np.nan
        z = (v - mu) / sd
        z[~m] = 0.0
        return z, m

    Xz, Xm = _std_ranks(ii)
    Yz, Ym = _std_ranks(ff)
    n_pair = Xm.astype(np.float64).T @ Ym.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic_mat = (Xz.T @ Yz) / n_pair
    np.fill_diagonal(ic_mat, np.nan)
    ic_mat[n_pair < 60] = np.nan

    # pass 1 (fast screen with safety margin) -> pass 2 (EXACT joint-mask
    # Spearman for the few candidates) — vector speed, loop-identical output
    si, ti = np.where(np.abs(ic_mat) >= max(min_ic - 0.03, 0.05))
    if not len(si):
        return pd.DataFrame(columns=["source", "target", "ic", "hit_rate",
                                     "source_name", "target_name"])
    iiv, ffv = ii.values, ff.values
    keep, exact_ics = [], []
    for i, j in zip(si, ti):
        x, y = iiv[:, i], ffv[:, j]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 60:
            continue
        rx = pd.Series(x[m]).rank().values
        ry = pd.Series(y[m]).rank().values
        ic = np.corrcoef(rx, ry)[0, 1]
        if np.isfinite(ic) and abs(ic) >= min_ic:
            keep.append((i, j))
            exact_ics.append(round(float(ic), 3))
    if not keep:
        return pd.DataFrame(columns=["source", "target", "ic", "hit_rate",
                                     "source_name", "target_name"])
    df = pd.DataFrame({"source": [cols[i] for i, _ in keep],
                       "target": [cols[j] for _, j in keep],
                       "ic": exact_ics})

    # hit rate of the directional call on the same window
    col_ix = {c: k for k, c in enumerate(cols)}
    hits = []
    for s, t, ic in zip(df.source, df.target, df.ic):
        x = iiv[:, col_ix[s]]; y = ffv[:, col_ix[t]]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        strong = np.abs(x) >= WAVE_Z
        if strong.sum() >= 8:
            pred = np.sign(x[strong]) * np.sign(ic)
            hits.append(float((pred == np.sign(y[strong])).mean()))
        else:
            hits.append(np.nan)
    df["hit_rate"] = hits
    df["source_name"] = df.source.map(lambda s: NODES.get(s, (s,))[0])
    df["target_name"] = df.target.map(lambda s: NODES.get(s, (s,))[0])
    return df.sort_values("ic", key=abs, ascending=False, ignore_index=True)


def active_waves(closes: pd.DataFrame, edges: pd.DataFrame,
                 z_th: float = WAVE_Z) -> pd.DataFrame:
    """Nodes whose impulse fired now, mapped to their downstream forecasts."""
    imp = impulses(closes)
    now = imp.iloc[-1]
    live = now[now.abs() >= z_th].dropna()
    rows = []
    for src, z in live.items():
        for _, e in edges[edges.source == src].iterrows():
            direction = np.sign(z) * np.sign(e.ic)
            rows.append(dict(
                source=src, source_name=NODES.get(src, (src,))[0],
                source_z=round(float(z), 2),
                target=e.target, target_name=NODES.get(e.target, (e.target,))[0],
                call="📈 UP" if direction > 0 else "📉 DOWN",
                horizon_days=EDGE_HORIZON, edge_ic=e.ic,
                hit_rate=e.hit_rate,
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["hit_rate", "edge_ic"], key=lambda s: s.abs() if s.name == "edge_ic" else s,
                        ascending=False, ignore_index=True)
    return df


# ── walk-forward validation ──────────────────────────────────────────
def backtest(closes: pd.DataFrame, step: int = 5, top_k: int = 5,
             horizon: int = EDGE_HORIZON, train: int | None = None):
    """Weekly walk-forward: re-estimate edges on trailing window, follow the
    top-k active-wave forecasts, measure realised forward returns vs the
    equal-weight universe. Returns (per-period df, summary dict)."""
    imp = impulses(closes)
    fwd = closes.shift(-horizon) / closes - 1.0
    T = len(closes)
    if train is None:                       # adapt to available history
        train = int(min(EDGE_TRAIN, max(100, T * 0.45)))
    start = max(train + IMPULSE_Z_WIN // 2, 160)
    recs = []
    for t in range(start, T - horizon, step):
        edges = estimate_edges(closes, asof=t, train=train, horizon=horizon,
                               imp=imp, fwd=fwd)
        if edges.empty:
            continue
        now = imp.iloc[t]
        live = now[now.abs() >= WAVE_Z].dropna()
        picks = []
        for src, z in live.items():
            for _, e in edges[edges.source == src].iterrows():
                picks.append((e.target, np.sign(z) * np.sign(e.ic),
                              abs(e.ic) * (e.hit_rate if np.isfinite(e.hit_rate) else 0.5)))
        if not picks:
            continue
        pk = (pd.DataFrame(picks, columns=["target", "dir", "conv"])
              .groupby("target").agg(dir=("dir", "mean"), conv=("conv", "sum"))
              .query("dir != 0").nlargest(top_k, "conv"))
        rets = []
        for tgt, row in pk.iterrows():
            r = fwd[tgt].iloc[t]
            if np.isfinite(r):
                rets.append(np.sign(row.dir) * r)
        if not rets:
            continue
        bench = fwd.iloc[t].mean()
        recs.append(dict(date=closes.index[t], n=len(rets),
                         strat=float(np.mean(rets)), bench=float(bench)))
    df = pd.DataFrame(recs)
    if df.empty:
        return df, {}
    df["excess"] = df.strat - df.bench
    half = len(df) // 2
    def sh(x):
        return float(np.mean(x) / np.std(x) * np.sqrt(252 / horizon)) if np.std(x) > 0 else np.nan
    summary = dict(
        periods=len(df),
        mean_fwd=float(df.strat.mean()), bench_fwd=float(df.bench.mean()),
        mean_excess=float(df.excess.mean()),
        hit_vs_bench=float((df.excess > 0).mean()),
        sharpe=sh(df.strat), bench_sharpe=sh(df.bench),
        h1_excess=float(df.excess.iloc[:half].mean()),
        h2_excess=float(df.excess.iloc[half:].mean()),
    )
    return df, summary


# ── layer 1: pressure system ─────────────────────────────────────────
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

def _fred(sid: str) -> pd.Series:
    r = requests.get(FRED.format(sid=sid), timeout=15)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "v"]
    s = pd.to_numeric(df.v, errors="coerce")
    s.index = pd.to_datetime(df.date)
    return s.dropna()


def pressure_system() -> dict:
    """Net US liquidity = Fed balance sheet - TGA - reverse repo, plus
    stablecoin supply (crypto dry powder) and HY credit spread."""
    out = {"components": {}, "errors": []}
    try:
        walcl = _fred("WALCL") / 1e6          # $tn
        tga = _fred("WTREGEN") / 1e6
        rrp = _fred("RRPONTSYD") / 1e6
        idx = walcl.index.union(tga.index).union(rrp.index)
        netliq = (walcl.reindex(idx).ffill() - tga.reindex(idx).ffill()
                  - rrp.reindex(idx).ffill()).dropna()
        out["netliq"] = netliq
        out["components"]["Net US Liquidity ($tn)"] = float(netliq.iloc[-1])
        out["components"]["NetLiq Δ 21d ($bn)"] = float((netliq.iloc[-1] - netliq.iloc[-22]) * 1000)
    except Exception as e:
        out["errors"].append(f"FRED liquidity: {e}")
    try:
        hy = _fred("BAMLH0A0HYM2")
        out["hy_oas"] = hy
        out["components"]["HY Spread (%)"] = float(hy.iloc[-1])
        out["components"]["HY Δ 21d (bp)"] = float((hy.iloc[-1] - hy.iloc[-22]) * 100)
    except Exception as e:
        out["errors"].append(f"FRED HY OAS: {e}")
    try:
        r = requests.get("https://stablecoins.llama.fi/stablecoincharts/all",
                         timeout=20).json()
        s = pd.Series({pd.to_datetime(int(x["date"]), unit="s"):
                       x["totalCirculatingUSD"]["peggedUSD"] for x in r}) / 1e9
        out["stables"] = s
        out["components"]["Stablecoin Supply ($bn)"] = float(s.iloc[-1])
        out["components"]["Stables Δ 21d ($bn)"] = float(s.iloc[-1] - s.iloc[-22])
    except Exception as e:
        out["errors"].append(f"DefiLlama stablecoins: {e}")
    # gauge: only components that ACTUALLY ARRIVED get a vote. Previously a
    # missing key defaulted to 0, which failed the ">0" test and voted -1 — so
    # three dead feeds silently produced gauge -3 "Draining", and the regime,
    # mega_scan and the advisor all keyed off that phantom reading.
    score, n_feeds = 0, 0
    c = out["components"]
    for key, supportive in (("NetLiq Δ 21d ($bn)", lambda v: v > 0),
                            ("Stables Δ 21d ($bn)", lambda v: v > 0),
                            ("HY Δ 21d (bp)", lambda v: v < 0)):
        v = c.get(key)
        if v is None or not np.isfinite(v):
            continue
        n_feeds += 1
        score += 1 if supportive(v) else -1
    out["n_feeds"] = n_feeds
    if n_feeds == 0:
        out["gauge"] = None
        out["gauge_label"] = "⚪ Unavailable — pressure feeds did not load"
    else:
        out["gauge"] = score
        out["gauge_label"] = {3: "🟢 High pressure — liquidity building",
                              2: "🟢 Supportive",
                              1: "🟢 Mildly supportive",
                              0: "🟡 Mixed",
                              -1: "🟡 Mixed / draining",
                              -2: "🔴 Draining",
                              -3: "🔴 Draining — waves unlikely to travel far"}.get(score, "🟡 Mixed")
        if n_feeds < 3:
            out["gauge_label"] += f" (only {n_feeds} of 3 feeds reporting)"
    return out


# ── layer 2: sentinels ───────────────────────────────────────────────
def sentinel_board(closes: pd.DataFrame) -> pd.DataFrame:
    imp = impulses(closes)
    rows = []
    for s in SENTINELS:
        if s not in closes.columns or closes[s].dropna().empty:
            continue
        px = closes[s].dropna()
        z = imp[s].dropna()
        rows.append(dict(
            Sentinel=NODES[s][0], Symbol=s,
            Last=float(px.iloc[-1]),
            **{"5d %": float(px.iloc[-1] / px.iloc[-6] - 1) * 100 if len(px) > 6 else np.nan},
            **{"21d %": float(px.iloc[-1] / px.iloc[-22] - 1) * 100 if len(px) > 22 else np.nan},
            ImpulseZ=float(z.iloc[-1]) if len(z) else np.nan,
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Signal"] = df.ImpulseZ.map(
            lambda z: "🔥 surging" if z >= WAVE_Z else
                      ("🧊 dumping" if z <= -WAVE_Z else "— quiet"))
    return df


# ── layer 0: forced flow calendar ────────────────────────────────────
def _third_friday(y, m):
    d = date(y, m, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d



def _buyback_buy_line(earnings: dict | None) -> str:
    """Exact per-ticker dip-buy dates derived from the real earnings calendar."""
    base = ("🛒 Buy: post-earnings DIPS in the heaviest repurchasers. "
            "Broad: QQQ/SPY. Pure-play: PKW (Buyback Achievers ETF). ")
    if not earnings:
        return (base + "Per-ticker dates: earnings calendar unavailable right "
                "now — the dip window is 1-2 sessions after each of "
                "AAPL, GOOGL, MSFT, META, NVDA, JPM, XOM reports.")
    parts = []
    for tk, ed in sorted(earnings.items(), key=lambda kv: kv[1]):
        dip0 = ed + timedelta(days=1)
        dip1 = ed + timedelta(days=2)
        parts.append(f"{tk}: reports {ed:%a %b %d} → dip window {dip0:%b %d}–{dip1:%b %d}")
    return base + "Exact dates — " + "; ".join(parts) + "."


def forced_flows(today: date | None = None, days_ahead: int = 45,
                 closes: pd.DataFrame | None = None,
                 earnings: dict | None = None) -> pd.DataFrame:
    """Mechanical, scheduled flows in the next `days_ahead` days — with who
    is forced to trade, what they trade, and what to watch."""
    today = today or date.today()
    horizon = today + timedelta(days=days_ahead)
    events = []

    def ev(d, name, why, who, what, watch, buy):
        events.append(dict(Date=d, Event=name, Why=why, Who=who,
                           What=what, Watch=watch, Buy=buy))

    # dynamic month-end call: which side do pensions rebalance INTO?
    _pension_buy = ("Direction unknown without SPY/TLT history — pensions buy "
                    "whichever of stocks/bonds LAGGED this month.")
    if closes is not None and "SPY" in closes and "TLT" in closes:
        try:
            mtd = closes.loc[(closes.index.month == closes.index[-1].month)
                             & (closes.index.year == closes.index[-1].year),
                             ["SPY", "TLT"]]
            gap = float((mtd.SPY.iloc[-1] / mtd.SPY.iloc[0] - 1)
                        - (mtd.TLT.iloc[-1] / mtd.TLT.iloc[0] - 1))
            if gap > 0.005:
                _pension_buy = (f"Stocks beat bonds by {gap:+.1%} this month → "
                                "pensions SELL equities / BUY bonds into month-end. "
                                "Play: long TLT for the final 1-3 sessions; expect "
                                "a mild SPY headwind, relief on day 1 of the new month.")
            elif gap < -0.005:
                _pension_buy = (f"Bonds beat stocks by {-gap:+.1%} this month → "
                                "pensions SELL bonds / BUY equities into month-end. "
                                "Play: long SPY for the final 1-3 sessions.")
            else:
                _pension_buy = ("Stocks and bonds are roughly tied this month → "
                                "rebalance flow is small. No trade.")
        except Exception:
            pass

    for k in range(3):
        m = (today.month - 1 + k) % 12 + 1
        y = today.year + (today.month - 1 + k) // 12
        opex = _third_friday(y, m)
        ev(opex, "Options expiration (OpEx)",
           "Dealer hedging pins prices near big strikes into expiry; when the "
           "options expire the pin releases and volatility often expands the "
           "following week.",
           "Market-maker desks (Citadel Securities, Susquehanna, Wolverine) "
           "mechanically hedging their options books.",
           "Index & mega-cap options: SPY, QQQ, SPX, and the highest open-"
           "interest single names (NVDA, TSLA, AAPL).",
           "Expect drift INTO OpEx week, bigger moves the week AFTER. "
           "Fade the pin, don't fight it.",
           f"🛒 Buy date: {opex + timedelta(days=3):%a %b %d} (first session "
           "after expiry). Hold off NEW entries during OpEx week; place "
           "planned buys that Monday when the pin releases — dips then are "
           "mechanical, not fundamental.")
        if m in (3, 6, 9, 12):
            ev(opex, "S&P quarterly rebalance (effective at the close)",
               "Every S&P index fund must own the new weights at that close — "
               "trillions tracking, zero price sensitivity.",
               "Vanguard, BlackRock, State Street index funds (~$12tn+ "
               "tracking S&P indices).",
               "The announced adds get bought, deletes get sold. Adds are "
               "published ~5-10 days early on spglobal.com press releases.",
               "The classic play — buy the add at announcement — has decayed "
               "as it got crowded; the reliable part is the huge closing "
               "auction volume, good for exiting positions with zero impact.",
               f"🛒 Buy: the announced ADD tickers on announcement day — watch "
               f"spglobal.com press releases from {opex - timedelta(days=12):%a %b %d}. "
               f"Sell into the {opex:%b %d} rebalance close. Small size — decayed edge.")
        last = (date(y, m, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        ev(last, "Month-end pension rebalance window (final 1-3 sessions)",
           "Pensions restore their stock/bond targets: whatever RALLIED this "
           "month gets trimmed, whatever lagged gets topped up.",
           "Corporate & state pensions (CalPERS-scale), target-date funds, "
           "sovereign wealth funds — roughly $1tn rebalancing monthly.",
           "If stocks beat bonds this month: they SELL equities (SPY) and "
           "BUY bonds (TLT). Reverse if bonds won.",
           "Estimate the direction from the month's stock-vs-bond gap; the "
           "flow hits the last 1-3 closes, then pressure vanishes on day 1 "
           "of the new month.",
           "🛒 " + _pension_buy
           + f" Exact window: {', '.join(d.strftime('%a %b %d') for d in pd.bdate_range(end=last, periods=3))}."
           )
        if m in (1, 4, 7, 10):
            ev(date(y, m, 15), "Buyback blackout lifts (approx.)",
               "Companies can't repurchase shares in the ~5 weeks before "
               "earnings; as each company reports, its buyback desk switches "
               "back on. Corporates are the single largest net buyer of US "
               "equities (~$1tn/yr authorized).",
               "The companies themselves via broker algos: Apple (~$100bn/yr "
               "program), Alphabet, Microsoft, Meta, NVIDIA, JPMorgan, "
               "Exxon — the mega-cap cash machines.",
               "Their OWN stock — which concentrates the bid in mega-cap "
               "indices. Broad exposure: SPY/QQQ; pure-play: PKW (Buyback "
               "Achievers ETF) holds the heaviest repurchasers.",
               "Support returns to mega-caps 1-2 days after each one "
               "reports. Post-earnings dips in heavy-buyback names get "
               "bought by the company itself.",
               _buyback_buy_line(earnings))
    if today.month <= 6:
        rr = _third_friday(today.year, 6) + timedelta(days=7)
        ev(rr, "Russell reconstitution (late June)",
           "FTSE Russell rebuilds the Russell 1000/2000 once a year — the "
           "single largest forced-flow day: ~$100bn+ trades in one closing "
           "auction.",
           "Every small-cap index fund and closet indexer tracking the "
           "Russell 2000 (~$10tn benchmarked).",
           "Adds to the Russell 2000 (fast-growing small caps, recent IPOs) "
           "get bought; graduates and deletes get sold. Preliminary lists "
           "publish in May on ftserussell.com.",
           "Adds tend to run up AFTER the preliminary list, into recon day; "
           "the effect fades fast after the auction. IWM sees enormous "
           "closing volume.",
           "🛒 Buy: preliminary-list ADDS (ftserussell.com, published May) in "
           "early June, exit AT the reconstitution close — do not hold "
           "through it. Lazy version: IWM into recon week.")
    if today.month >= 11 or today.month == 12:
        ev(date(today.year, 12, 15), "Tax-loss selling peak window",
           "Investors dump the year's losers before Dec 31 to harvest "
           "capital losses — selling that has nothing to do with the "
           "companies' prospects.",
           "Retail investors and taxable funds; advisors run harvesting "
           "programs Nov-Dec.",
           "The year's WORST performers, hardest in small caps where retail "
           "owns more. Screen: down 30%+ YTD, still profitable businesses.",
           "Don't catch the falling knives in early Dec; build the January-"
           "reversal shopping list instead.",
           "🛒 Buy: nothing yet — this window is for LIST-BUILDING. Screen: "
           "down 30%+ YTD, still profitable, small/mid cap. Your buy date is "
           "the January-reversal card below.")
        ev(date(today.year + (1 if today.month == 12 else 0), 1, 5),
           "January reversal window",
           "The tax-selling pressure disappears on Jan 1 and the beaten-down "
           "names bounce — the 'January effect', strongest in the first two "
           "weeks.",
           "The same sellers stop selling; bargain hunters and small-cap "
           "funds step in.",
           "Last year's oversold losers, small-cap value especially. Broad "
           "proxy: IWM vs SPY spread in early January.",
           "Enter the final week of Dec, exit mid-Jan. It's a decayed but "
           "still-positive seasonal — size it small.",
           f"🛒 Buy dates: {date(today.year, 12, 24):%b %d}–{date(today.year, 12, 31):%b %d} "
           f"(final Dec week). Exit by {date(today.year + 1, 1, 15):%b %d, %Y}. "
           "Basket of 10+ December losers equal-weight (never one name), or "
           "simply IWM. Small size — decayed seasonal.")

    df = pd.DataFrame([e for e in events if today <= e["Date"] <= horizon])
    return df.sort_values("Date", ignore_index=True)


# ── plain-language forecast board: aggregate waves by target ─────────
def forecast_board(waves: pd.DataFrame, min_sources: int = 1) -> pd.DataFrame:
    """Group active-wave forecasts by TARGET into simple net calls.
    Multiple waves agreeing on one target = conviction."""
    if waves is None or waves.empty:
        return pd.DataFrame()
    w = waves.copy()
    w["dir"] = np.where(w.call.str.contains("UP"), 1, -1)
    w["weight"] = w.edge_ic.abs() * w.hit_rate.fillna(0.5)
    rows = []
    for tgt, g in w.groupby("target"):
        net = float((g.dir * g.weight).sum())
        if net == 0:
            continue
        agree = g[g.dir == np.sign(net)]
        rows.append(dict(
            target=tgt, target_name=NODES.get(tgt, (tgt,))[0],
            call="UP" if net > 0 else "DOWN",
            n_sources=int(len(agree)),
            sources=", ".join(agree.sort_values("weight", ascending=False)
                              .source_name.head(3)),
            avg_hit=float(agree.hit_rate.fillna(0.5).mean()),
            conviction=abs(net),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["conviction"] = df.conviction / df.conviction.max()
    df = df[df.n_sources >= min_sources]
    return df.sort_values(["conviction"], ascending=False, ignore_index=True)


def investment_plan(b, closes: pd.DataFrame) -> dict:
    """Rule-based trade plan for one forecast-board row `b`, sized from the
    target's own volatility and the trigger's historical hit rate.
    Research output, not personal advice."""
    tgt = b["target"]
    px = float(closes[tgt].dropna().iloc[-1])
    dly = closes[tgt].pct_change().dropna()
    sigma10 = float(dly.tail(63).std() * np.sqrt(EDGE_HORIZON))  # 10-session vol
    up = b["call"] == "UP"
    stop_pct = 1.25 * sigma10
    tgt_pct = 1.50 * sigma10
    hit = float(b["avg_hit"])
    strong = hit >= 0.62 and b["n_sources"] >= 2
    unit = "1 full unit (≈1% account risk)" if strong else "½ unit (≈0.5% account risk)"
    if up:
        action = f"BUY {tgt} ({b['target_name']})"
        entry = f"Enter within 1-2 sessions at ≈ ${px:,.2f} (signal decays over the horizon)"
        stop = f"${px * (1 - stop_pct):,.2f}  ({-stop_pct:.1%} — 1.25× its own 10-session volatility)"
        target = f"${px * (1 + tgt_pct):,.2f}  ({tgt_pct:+.1%}) or time exit, whichever first"
    else:
        action = f"AVOID / TRIM {tgt} ({b['target_name']}) — take profits, skip new longs"
        entry = (f"If expressing short: puts or inverse exposure near ≈ ${px:,.2f}; "
                 "simplest edge capture is just NOT buying this for 10 sessions")
        stop = f"${px * (1 + stop_pct):,.2f}  (+{stop_pct:.1%} against a short)"
        target = f"${px * (1 - tgt_pct):,.2f}  ({-tgt_pct:.1%}) or time exit"
    return dict(
        action=action,
        trigger=(f"{b['n_sources']} independent wave{'s' if b['n_sources']>1 else ''} "
                 f"({b['sources']}) firing into edges that hit {hit:.0%} "
                 f"historically over the next {EDGE_HORIZON} sessions"),
        entry=entry, stop=stop, target=target,
        time_exit=(f"Close after {EDGE_HORIZON} sessions regardless — the edge is "
                   "only measured to there; holding past it is a different, "
                   "untested trade"),
        size=unit,
        invalidation=("Stand down if the source wave's impulse flips sign "
                      "before you enter, or if the Pressure gauge drops to 🔴 "
                      "— waves don't travel in draining liquidity"),
    )


# ═════════════════════════════════════════════════════════════════════
# Stock-level layer: nightly dump + Alpaca + earnings dates
# ═════════════════════════════════════════════════════════════════════
DUMP_URL = "https://raw.githubusercontent.com/magicpro33/stock/main/data/stock_data.json.gz"
LOCAL_DUMP = os.path.join(os.path.dirname(__file__), "data", "dump_panel_v4.npz")

FUND_FIELDS = ["ShortPctFloat", "DaysToCover", "P/E", "RevenueGrowth",
               "EarningsGrowth", "MarketCap", "Piotroski", "GoldenCross",
               "ROIC", "DividendYieldPct", "DividendRate", "ShortSqueeze",
               "CleanSetupScore", "MFI", "OE_Yield", "PCV", "ROIC_Trend"]

BUYBACK_TITANS = ["AAPL", "GOOGL", "MSFT", "META", "NVDA", "JPM", "XOM"]


_PANEL_CACHE = {}          # in-process: avoid re-reading the ~27MB npz per call


def load_dump_panel():
    """Full OHLCV panel for ~5,700 stocks from the nightly magicpro33/stock
    dump. Cached to disk AND in-process (mtime-keyed); refetched when >4 days
    stale. Returns (panel dict, tickers, sectors, mdv, dates)."""
    import gzip as _gz
    if os.path.exists(LOCAL_DUMP):
        mt = os.path.getmtime(LOCAL_DUMP)
        hit = _PANEL_CACHE.get("panel")
        if hit and hit[0] == mt:
            return hit[1]
        z = np.load(LOCAL_DUMP, allow_pickle=True)
        dts = pd.to_datetime(z["dates"])
        if (pd.Timestamp.today() - dts[-1]).days <= 4:
            panel = {f: z[f] for f in ("o", "h", "l", "c", "v")}
            out = (panel, z["tickers"], z["sectors"], z["mdv"], dts)
            _PANEL_CACHE["panel"] = (mt, out)
            _PANEL_CACHE["tick_ix"] = (mt, {t: i for i, t in enumerate(z["tickers"])})
            return out
    try:
        r = requests.get(DUMP_URL, timeout=120)
        r.raise_for_status()
    except Exception as _de:
        # GitHub unreachable and the cache is >4 days old. A stale dump beats
        # a dead Top 20 / Lookup / APEX — fetch_history already works this way.
        if os.path.exists(LOCAL_DUMP):
            z = _np_load_dump()
            if z is not None:
                return z
        raise RuntimeError(f"nightly dump unavailable and no local copy: {_de}")
    data = json.loads(_gz.decompress(r.content).decode())
    rows = [x for x in data if len(x.get("_hist", {}).get("dates", [])) >= 120]
    all_d = sorted({d for x in rows for d in x["_hist"]["dates"]})
    dix = {d: i for i, d in enumerate(all_d)}
    T, N = len(all_d), len(rows)
    panel = {f: np.full((T, N), np.nan, dtype=np.float32)
             for f in ("o", "h", "l", "c", "v")}
    key = dict(o="open", h="high", l="low", c="close", v="volume")
    tickers, sectors = [], []
    funds = {f: np.full(N, np.nan, dtype=np.float64) for f in FUND_FIELDS}
    for j, x in enumerate(rows):
        ix = [dix[d] for d in x["_hist"]["dates"]]
        for f, kk in key.items():
            panel[f][ix, j] = x["_hist"][kk]
        tickers.append(x["Ticker"])
        sectors.append(x.get("Sector") or "Unknown")
        for f in FUND_FIELDS:
            v = x.get(f)
            if v is not None:
                try:
                    funds[f][j] = float(v)
                except (TypeError, ValueError):
                    pass
    # The tradeable guard needs to know whether a name actually PRINTED
    # recently. Snapshot that from the raw closes BEFORE the ffill — reading
    # it afterwards is a no-op, because ffill(limit=5) makes a halted or
    # delisted name look like it has a fresh price for another five sessions.
    recent_ok = np.isfinite(panel["c"][-3:]).any(axis=0)
    panel["c"] = pd.DataFrame(panel["c"]).ffill(limit=5).values.astype(np.float32)
    mdv = np.nanmedian((panel["c"] * np.nan_to_num(panel["v"]))[-21:], axis=0)
    tickers, sectors = np.array(tickers), np.array(sectors)
    np.savez_compressed(LOCAL_DUMP, tickers=tickers, sectors=sectors,
                        mdv=mdv, dates=np.array(all_d), recent_ok=recent_ok, **panel,
                        **{f"fund_{i}": funds[f] for i, f in enumerate(FUND_FIELDS)})
    out = (panel, tickers, sectors, mdv, pd.to_datetime(all_d))
    mt = os.path.getmtime(LOCAL_DUMP)
    _PANEL_CACHE.clear()
    _PANEL_CACHE["panel"] = (mt, out)
    _PANEL_CACHE["tick_ix"] = (mt, {t: i for i, t in enumerate(tickers)})
    return out


def _np_load_dump():
    """Load whatever dump npz is on disk, ignoring its age. Used as the
    stale fallback when the GitHub download fails."""
    try:
        z = np.load(LOCAL_DUMP, allow_pickle=True)
        dts = pd.to_datetime(z["dates"])
        panel = {f: z[f] for f in ("o", "h", "l", "c", "v")}
        out = (panel, z["tickers"], z["sectors"], z["mdv"], dts)
        mt = os.path.getmtime(LOCAL_DUMP)
        _PANEL_CACHE["panel"] = (mt, out)
        _PANEL_CACHE["tick_ix"] = (mt, {t: i for i, t in enumerate(z["tickers"])})
        return out
    except Exception:
        return None


def _recent_ok_mask(panel) -> np.ndarray:
    """True where the stock had a REAL close in the last 3 sessions.

    Reads the mask persisted by load_dump_panel (captured pre-ffill). Older
    npz files predate it, so fall back to the post-ffill check rather than
    crash — that fallback is permissive, not wrong-permissive-forever: the
    next nightly rebuild writes the real mask.
    """
    try:
        z = np.load(LOCAL_DUMP, allow_pickle=True)
        if "recent_ok" in z.files:
            return z["recent_ok"].astype(bool)
    except Exception:
        pass
    return np.isfinite(panel["c"][-3:]).any(axis=0)


def _ticker_index(ticker: str):
    """Row index of a ticker in the dump panel. Case/whitespace tolerant so
    every caller path (search box, row-select, URL, future callers) resolves
    the same symbol."""
    load_dump_panel()
    hit = _PANEL_CACHE.get("tick_ix")
    if not hit:
        return None
    return hit[1].get(str(ticker).strip().upper())


def dump_fundamentals_all():
    """All dump fundamentals as {field: array} aligned to load_dump_panel tickers."""
    load_dump_panel()   # ensure the npz exists / is fresh
    mt = os.path.getmtime(LOCAL_DUMP)
    hit = _PANEL_CACHE.get("funds")
    if hit and hit[0] == mt:
        return hit[1]
    z = np.load(LOCAL_DUMP, allow_pickle=True)
    out = {f: z[f"fund_{i}"] for i, f in enumerate(FUND_FIELDS)}
    _PANEL_CACHE["funds"] = (mt, out)
    return out


def dump_fundamentals(ticker: str) -> dict:
    """One stock's nightly-dump fundamentals (NaNs dropped)."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    ix = np.where(tickers == ticker)[0]
    if not len(ix):
        return {}
    fa = dump_fundamentals_all()
    out = {f: float(fa[f][ix[0]]) for f in FUND_FIELDS if np.isfinite(fa[f][ix[0]])}
    out["Sector"] = str(sectors[ix[0]])
    return out


def dump_ohlcv(ticker: str) -> pd.DataFrame:
    """Full OHLCV history for one stock from the nightly dump."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    j = _ticker_index(ticker)
    if j is None:
        return pd.DataFrame()
    df = pd.DataFrame({"Open": panel["o"][:, j], "High": panel["h"][:, j],
                       "Low": panel["l"][:, j], "Close": panel["c"][:, j],
                       "Volume": panel["v"][:, j]}, index=dts)
    return df.dropna(subset=["Close"]).astype(float)


def ticker_stats(df: pd.DataFrame) -> dict:
    """IGNITION-style indicator pack from an OHLCV (or Close-only) frame."""
    c = df["Close"].dropna()
    out = {"price": float(c.iloc[-1])}
    for label, n in (("r5", 5), ("r21", 21), ("r63", 63)):
        out[label] = float(c.iloc[-1] / c.iloc[-n - 1] - 1) if len(c) > n else np.nan
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    out["rsi"] = float(rsi.iloc[-1]) if rsi.notna().any() else np.nan
    out["sma20"] = float(c.tail(20).mean())
    out["sma50"] = float(c.tail(50).mean()) if len(c) >= 50 else np.nan
    out["vol21"] = float(c.pct_change().tail(21).std() * np.sqrt(252))
    lo, hi = float(c.tail(63).min()), float(c.tail(63).max())
    out["rangepos"] = (out["price"] - lo) / (hi - lo) if hi > lo else np.nan
    if "Volume" in df and df["Volume"].notna().any():
        v = df["Volume"].fillna(0)
        out["rvol"] = float(v.tail(5).mean() / max(v.tail(63).mean(), 1))
    else:
        out["rvol"] = np.nan
    return out


def fastest_followers(node_symbol: str, node_closes: pd.DataFrame,
                      top: int = 5) -> pd.DataFrame:
    """Which individual stocks (from the nightly dump) historically follow
    this node's moves the fastest? Score = corr(node 5d move at t,
    stock 5d move at t+5) — a lagged response, not just same-day beta."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    C = panel["c"]
    node = node_closes[node_symbol].dropna()
    node.index = pd.to_datetime(node.index).tz_localize(None)
    common = dts.intersection(node.index)
    if len(common) < 120:
        return pd.DataFrame()
    n_ix = {d: i for i, d in enumerate(dts)}
    rows_ix = np.array([n_ix[d] for d in common])
    Cc = C[rows_ix]
    nd = node.reindex(common).values
    node_r5 = nd[5:] / nd[:-5] - 1.0                    # node 5d move at t
    stk_r5 = Cc[5:] / Cc[:-5] - 1.0                     # stock 5d move
    x = node_r5[:-5]                                    # node move at t
    y = stk_r5[5:]                                      # stock move at t+5
    ok = (mdv >= 2e6) & np.isfinite(Cc[-1]) & (Cc[-1] >= 3.0)
    xm = x - np.nanmean(x)
    ym = y - np.nanmean(y, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nansum(xm[:, None] * ym, axis=0) / (
            np.sqrt(np.nansum(xm ** 2) * np.nansum(ym ** 2, axis=0)))
        beta = np.nansum(xm[:, None] * ym, axis=0) / np.nansum(xm ** 2)
    corr = np.where(ok, corr, np.nan)
    idx = np.argsort(-np.nan_to_num(corr, nan=-9))[:top]
    return pd.DataFrame({
        "Ticker": tickers[idx], "Sector": sectors[idx],
        "FollowCorr": corr[idx].round(2), "Beta": beta[idx].round(2),
        "Price": Cc[-1][idx].round(2),
    })


def alpaca_prices(tickers: list, chunk: int = 80) -> dict:
    """Fresh prices via Alpaca batch snapshots. {} without keys/network.

    Requests are chunked: the whole list used to go into one query string, so a
    long watchlist or a 50-row scan could exceed the URL/API limit and return
    nothing — which surfaced as silently stale prices.
    """
    tickers = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
    if len(tickers) > chunk:
        out = {}
        for i in range(0, len(tickers), chunk):
            try:
                out.update(alpaca_prices(tickers[i:i + chunk], chunk=chunk))
            except Exception:
                continue
        return out
    pairs = [("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
             ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
             ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")]
    kid = sec = None
    getters = [lambda k: os.environ.get(k, "")]
    try:
        import streamlit as st
        getters.insert(0, lambda k: st.secrets.get(k, ""))
    except Exception:
        pass
    for a, b in pairs:
        for g in getters:
            try:
                if g(a) and g(b):
                    kid, sec = g(a), g(b)
                    break
            except Exception:
                continue
        if kid:
            break
    if not kid:
        return {}
    try:
        r = requests.get("https://data.alpaca.markets/v2/stocks/snapshots",
                         params={"symbols": ",".join(tickers), "feed": "iex"},
                         headers={"APCA-API-KEY-ID": kid,
                                  "APCA-API-SECRET-KEY": sec}, timeout=8)
        if r.status_code != 200:
            return {}
        out = {}
        for tk, snap in r.json().items():
            p = (snap.get("latestTrade") or {}).get("p") or                 (snap.get("dailyBar") or {}).get("c")
            if p:
                out[tk] = float(p)
        return out
    except Exception:
        return {}


def upcoming_earnings(tickers: list) -> dict:
    """{ticker: next earnings date} via yfinance. {} on any failure."""
    os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")
    out = {}
    try:
        import yfinance as yf
        for tk in tickers:
            try:
                ed = yf.Ticker(tk).earnings_dates
                if ed is None or ed.empty:
                    continue
                fut = ed.index.tz_localize(None)
                fut = fut[fut >= pd.Timestamp.today().normalize()]
                if len(fut):
                    out[tk] = fut.min().date()
            except Exception:
                continue
    except Exception:
        pass
    return out


CRYPTO_MAP = {"BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD", "SOL-USD": "SOL/USD"}
LAST_HISTORY_SOURCE = "unknown"


def _alpaca_keys_simple():
    pairs = [("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
             ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
             ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")]
    getters = [lambda k: os.environ.get(k, "")]
    try:
        import streamlit as st
        getters.insert(0, lambda k: st.secrets.get(k, ""))
    except Exception:
        pass
    for a, b in pairs:
        for g in getters:
            try:
                if g(a) and g(b):
                    return g(a), g(b)
            except Exception:
                continue
    return None, None


def alpaca_history(symbols: list, years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Daily closes for many symbols straight from Alpaca (IEX stocks feed +
    crypto endpoint). Empty frame when keys are missing or requests fail —
    caller falls back to yfinance."""
    kid, sec = _alpaca_keys_simple()
    if not kid:
        return pd.DataFrame()
    hdr = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    start = (date.today() - timedelta(days=int(years * 365.25 + 30))).isoformat()
    out = {}

    def _paged(url, params, unmap=None):
        token = None
        while True:
            p = dict(params, **({"page_token": token} if token else {}))
            try:
                r = requests.get(url, params=p, headers=hdr, timeout=60)
                if r.status_code != 200:
                    return
                j = r.json()
            except Exception:
                return
            for sym, bars in (j.get("bars") or {}).items():
                key = unmap.get(sym, sym) if unmap else sym
                d = out.setdefault(key, {})
                for b in bars:
                    d[b["t"][:10]] = b["c"]
            token = j.get("next_page_token")
            if not token:
                return

    stocks = [s for s in symbols if s not in CRYPTO_MAP and not s.startswith("^")]
    for i in range(0, len(stocks), 50):
        _paged("https://data.alpaca.markets/v2/stocks/bars",
               {"symbols": ",".join(stocks[i:i + 50]), "timeframe": "1Day",
                "start": start, "limit": 10000, "adjustment": "all",
                "feed": "iex"})
    cryptos = [s for s in symbols if s in CRYPTO_MAP]
    if cryptos:
        unmap = {v: k for k, v in CRYPTO_MAP.items()}
        _paged("https://data.alpaca.markets/v1beta3/crypto/us/bars",
               {"symbols": ",".join(CRYPTO_MAP[s] for s in cryptos),
                "timeframe": "1Day", "start": start, "limit": 10000},
               unmap=unmap)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame({k: pd.Series(v) for k, v in out.items()})
    df.index = pd.to_datetime(df.index)
    return df.sort_index().ffill(limit=5)


# ═════════════════════════════════════════════════════════════════════
# Stock Lookup: analog-outcome forecast, upstream drivers, watchlist
# ═════════════════════════════════════════════════════════════════════
WATCHLIST_PATH = os.path.join(os.path.dirname(__file__), "data", "watchlist.json")


def _feature_panels():
    """Point-in-time features + forward returns for every (day, stock) in the
    dump — the analog library. Sampled every 3 sessions after warmup.
    Cached in-process (mtime-keyed) — it's ~150k rows of pure numpy."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    mt = os.path.getmtime(LOCAL_DUMP)
    hit = _PANEL_CACHE.get("featpan")
    if hit and hit[0] == mt:
        return hit[1]
    C, V = panel["c"], np.nan_to_num(panel["v"])
    T, N = C.shape
    days = list(range(70, T - 22, 3))
    feats, fwds = [], []
    dvol = C * V
    for t in days:
        mom = C[t - 5] / C[t - 63] - 1.0
        mom_pct = pd.Series(mom).rank(pct=True).values
        lo = np.nanmin(panel["l"][t - 62:t + 1], 0)
        hi = np.nanmax(panel["h"][t - 62:t + 1], 0)
        rangepos = (C[t] - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
        rvol = V[t - 4:t + 1].mean(0) / np.where(V[t - 62:t + 1].mean(0) == 0,
                                                 np.nan, V[t - 62:t + 1].mean(0))
        above = (C[t] > np.nanmean(C[t - 49:t + 1], 0)).astype(np.float32)
        ok = np.isfinite(C[t]) & (C[t] >= 3) &              (np.nanmedian(dvol[max(t - 20, 0):t + 1], 0) >= 2e6)
        f10 = C[t + 10] / C[t] - 1.0
        f21 = C[t + 21] / C[t] - 1.0
        m = ok & np.isfinite(mom_pct) & np.isfinite(rangepos) &             np.isfinite(rvol) & np.isfinite(f21)
        feats.append(np.column_stack([mom_pct[m], rangepos[m], rvol[m], above[m]]))
        fwds.append(np.column_stack([f10[m], f21[m]]))
    F = np.vstack(feats).astype(np.float32)
    R = np.vstack(fwds).astype(np.float32)
    _PANEL_CACHE["featpan"] = (mt, (F, R))
    return F, R


def _now_features(ticker: str):
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    ix = np.where(tickers == ticker)[0]
    if not len(ix):
        return None, None
    j = ix[0]
    C, V = panel["c"], np.nan_to_num(panel["v"])
    t = C.shape[0] - 1
    mom = C[t - 5] / C[t - 63] - 1.0
    mom_pct = float(pd.Series(mom).rank(pct=True).iloc[j])
    lo = np.nanmin(panel["l"][t - 62:t + 1, j])
    hi = np.nanmax(panel["h"][t - 62:t + 1, j])
    rangepos = float((C[t, j] - lo) / (hi - lo)) if hi > lo else np.nan
    rv_d = V[t - 62:t + 1, j].mean()
    rvol = float(V[t - 4:t + 1, j].mean() / rv_d) if rv_d > 0 else np.nan
    above = float(C[t, j] > np.nanmean(C[t - 49:t + 1, j]))
    return np.array([mom_pct, rangepos, rvol, above], dtype=np.float32), sectors[j]


def _features_from_hist(hist: pd.DataFrame):
    """Today's analog features for ANY ticker from a fetched OHLCV frame,
    with the momentum percentile ranked against the dump cross-section."""
    if hist is None or hist.empty or "Close" not in hist:
        return None
    c = hist["Close"].dropna()
    if len(c) < 70:
        return None
    n_mom = min(63, len(c) - 6)
    mom = float(c.iloc[-6] / c.iloc[-6 - n_mom] - 1)
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    C = panel["c"]
    dm = C[-6] / C[-64] - 1.0
    dm = dm[np.isfinite(dm)]
    mom_pct = float((dm <= mom).mean()) if len(dm) else np.nan
    w = min(63, len(c))
    if {"High", "Low"}.issubset(hist.columns) and hist["High"].notna().any():
        hi = float(hist["High"].tail(w).max()); lo = float(hist["Low"].tail(w).min())
    else:
        hi = float(c.tail(w).max()); lo = float(c.tail(w).min())
    rangepos = float((c.iloc[-1] - lo) / (hi - lo)) if hi > lo else np.nan
    rvol = np.nan
    if "Volume" in hist and hist["Volume"].notna().sum() > 63:
        v = hist["Volume"].fillna(0)
        base = v.tail(63).mean()
        if base > 0:
            rvol = float(v.tail(5).mean() / base)
    above = float(c.iloc[-1] > c.tail(50).mean())
    return np.array([mom_pct, rangepos, rvol, above], dtype=np.float32)


def outcome_forecast(ticker: str, F=None, R=None, hist: pd.DataFrame | None = None) -> dict:
    """Analog forecast: what happened to every look-alike (day, stock) in the
    dump. Works for ANY ticker — dump members use point-in-time dump features;
    anything else derives features from its fetched history (`hist`)."""
    now, sector = _now_features(ticker)
    if now is None and hist is not None:
        now, sector = _features_from_hist(hist), "—"
    if now is None or not np.isfinite(now[0]) or not np.isfinite(now[1]):
        return {}
    if F is None or R is None:
        F, R = _feature_panels()
    tol = np.array([0.10, 0.15, 0.50, 0.0])
    has_rvol = np.isfinite(now[2])
    for widen in (1.0, 1.6, 2.4):
        m = ((np.abs(F[:, 0] - now[0]) <= tol[0] * widen)
             & (np.abs(F[:, 1] - now[1]) <= tol[1] * widen)
             & (F[:, 3] == now[3]))
        if has_rvol:
            m &= np.abs(np.minimum(F[:, 2], 3) - min(now[2], 3)) <= tol[2] * widen
        if m.sum() >= 250:
            break
    sel = R[m]
    if len(sel) < 60:
        return {"n": int(len(sel))}
    base21 = R[:, 1]
    return dict(
        n=int(len(sel)), sector=str(sector),
        widen=float(widen),
        med10=float(np.median(sel[:, 0])), med21=float(np.median(sel[:, 1])),
        mean21=float(sel[:, 1].mean()),
        p_up=float((sel[:, 1] > 0).mean()),
        p_pop=float((sel[:, 1] >= 0.15).mean()),
        p_pop_base=float((base21 >= 0.15).mean()),
        p_drop=float((sel[:, 1] <= -0.15).mean()),
        p_drop_base=float((base21 <= -0.15).mean()),
        q10=float(np.quantile(sel[:, 1], 0.10)),
        q90=float(np.quantile(sel[:, 1], 0.90)),
        dist=sel[:, 1],
        feats=dict(mom_pct=float(now[0]), rangepos=float(now[1]),
                   rvol=float(now[2]) if np.isfinite(now[2]) else float("nan"),
                   above_ma50=bool(now[3])),
    )


def upstream_drivers(ticker: str, node_closes: pd.DataFrame, top: int = 5,
                     hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Which cascade NODES lead this stock? corr(node 5d move at t,
    stock 5d move at t+5) — plus each node's CURRENT impulse = tailwind.
    Vectorized: all nodes in one aligned matrix pass."""
    if node_closes is None or node_closes.empty:
        return pd.DataFrame()
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    j = _ticker_index(ticker)
    if j is not None:
        s = pd.Series(panel["c"][:, j], index=dts).dropna()
    elif hist is not None and not hist.empty and "Close" in hist:
        s = hist["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    else:
        return pd.DataFrame()
    nc = node_closes.copy()
    nc.index = pd.to_datetime(nc.index).tz_localize(None)
    common = s.index.intersection(nc.index)
    if len(common) < 120:
        return pd.DataFrame()
    sv = s.reindex(common).values
    NV = nc.reindex(common).values                      # (T, n_nodes)
    s5 = sv[5:] / sv[:-5] - 1.0
    N5 = NV[5:] / NV[:-5] - 1.0
    x = N5[:-5]                                          # node move at t
    y = s5[5:]                                           # stock move at t+5
    ym = np.isfinite(y)
    corr = np.full(NV.shape[1], np.nan)
    for k in range(NV.shape[1]):                         # cheap: ~50 cols, pure numpy
        m = np.isfinite(x[:, k]) & ym
        if m.sum() < 100:
            continue
        xa, ya = x[m, k], y[m]
        xa = xa - xa.mean(); ya = ya - ya.mean()
        d = np.sqrt((xa @ xa) * (ya @ ya))
        if d > 0:
            corr[k] = float(xa @ ya / d)
    imp_now = impulses(node_closes).iloc[-1]
    rows = [dict(node=node, node_name=NODES.get(node, (node,))[0],
                 follow_corr=round(float(corr[k]), 2),
                 node_z=round(float(imp_now.get(node, np.nan)), 2))
            for k, node in enumerate(nc.columns) if np.isfinite(corr[k])]
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return df
    df = df.reindex(df.follow_corr.abs().sort_values(ascending=False).index).head(top)
    df["push"] = (df.follow_corr * df.node_z).round(2)
    return df.reset_index(drop=True)


# ── watchlist persistence ────────────────────────────────────────────
def watchlist_load() -> list:
    try:
        with open(WATCHLIST_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def watchlist_save(items: list):
    try:
        with open(WATCHLIST_PATH, "w") as f:
            json.dump(items, f, indent=1)
    except Exception:
        pass


def watchlist_add(ticker: str, price: float, note: str = ""):
    items = [w for w in watchlist_load() if w["ticker"] != ticker]
    items.append(dict(ticker=ticker, added=str(date.today()),
                      price_at_add=round(float(price), 2), note=note))
    watchlist_save(items)


def watchlist_remove(ticker: str):
    watchlist_save([w for w in watchlist_load() if w["ticker"] != ticker])


# ═════════════════════════════════════════════════════════════════════
# IGNITION analyzer data chain: Alpaca → yfinance → nightly dump
# ═════════════════════════════════════════════════════════════════════
def _alpaca_ohlcv(ticker: str, days: int = 400) -> pd.DataFrame:
    """Single-symbol daily OHLCV bars from Alpaca (IEX feed)."""
    kid, sec = _alpaca_keys_simple()
    if not kid:
        return pd.DataFrame()
    hdr = {"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec}
    start = (date.today() - timedelta(days=days)).isoformat()
    rows, token = [], None
    while True:
        p = {"symbols": ticker, "timeframe": "1Day", "start": start,
             "limit": 10000, "adjustment": "all", "feed": "iex"}
        if token:
            p["page_token"] = token
        try:
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                             params=p, headers=hdr, timeout=30)
            if r.status_code != 200:
                return pd.DataFrame()
            j = r.json()
        except Exception:
            return pd.DataFrame()
        rows += (j.get("bars") or {}).get(ticker, [])
        token = j.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{ "Date": b["t"][:10], "Open": b["o"], "High": b["h"],
                         "Low": b["l"], "Close": b["c"], "Volume": b["v"]} for b in rows])
    df.index = pd.to_datetime(df.Date)
    return df.drop(columns=["Date"]).astype(float).sort_index()


# dump field → yfinance-style info key (IGNITION SCAN_FIELD_MAP equivalent)
DUMP_INFO_MAP = {
    "ShortPctFloat":   "shortPercentOfFloat",
    "DaysToCover":     "shortRatio",
    "P/E":             "trailingPE",
    "RevenueGrowth":   "revenueGrowth",
    "EarningsGrowth":  "earningsGrowth",
    "MarketCap":       "marketCap",
    "DividendYieldPct": "_divYieldPct",
    "DividendRate":    "dividendRate",
    "Piotroski":       "_scan_piotroski",
    "GoldenCross":     "_scan_golden_cross",
    "ROIC":            "_scan_roic",
    "ShortSqueeze":    "_scan_squeeze",
    "CleanSetupScore": "_scan_clean_setup",
    "MFI":             "_scan_mfi",
    "OE_Yield":        "_scan_oe_yield",
    "PCV":             "_scan_pcv",
}


def fetch_analyzer(ticker: str):
    """IGNITION Stock Analyzer data chain, ported: Alpaca history first,
    yfinance for history fallback + fundamentals + EPS, nightly dump for
    anything still missing. Returns (info, hist, eps_history, eps_forward)."""
    ticker = str(ticker).strip().upper()
    os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")
    _issues, info = [], {}

    # ── Step 1: price history — Alpaca → yfinance → dump ────────────
    hist = _alpaca_ohlcv(ticker)
    hist_src = "alpaca" if len(hist) >= 50 else None
    if hist_src is None:
        hist = pd.DataFrame()
    tk = None
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
    except Exception:
        pass
    if hist.empty and tk is not None:
        try:
            h = tk.history(period="1y", interval="1d")
            if h is not None and not h.empty:
                if isinstance(h.columns, pd.MultiIndex):
                    h.columns = h.columns.get_level_values(0)
                hist, hist_src = h, "yahoo"
        except Exception:
            pass
    if hist.empty:
        d = dump_ohlcv(ticker)
        if not d.empty:
            hist, hist_src = d, "dump"
    if hist.empty:
        _issues.append("price history: no bars from Alpaca, Yahoo, or the nightly dump")

    # ── Step 2: fundamentals from yfinance ───────────────────────────
    if tk is not None:
        try:
            info = tk.info or {}
            if not info or len(info) < 3:
                info = {}
                _issues.append("fundamentals: yfinance returned empty (rate-limited or no profile)")
        except Exception as _ie:
            m = str(_ie)[:80]
            if "404" in m or "Not Found" in m:
                _issues.append("fundamentals: not published for this symbol (ETFs/funds have none)")
            elif "429" in m or "rate" in m.lower():
                _issues.append("fundamentals: yfinance rate limit — retry shortly")
            else:
                _issues.append(f"fundamentals: {m}")

    # ── Step 2b: extended profitability from the statements ──────────
    # ROCE + the margin suite. Statements beat info-fields; info-fields
    # fill the gaps (grossMargins / operatingMargins).
    def _srow(df_, *names):
        try:
            if df_ is None or not hasattr(df_, "empty") or df_.empty:
                return None
            ri = {str(i).strip().lower(): i for i in df_.index}
            for nm in names:
                k = ri.get(nm.lower())
                if k is not None:
                    col = df_.loc[k].dropna()
                    if len(col):
                        return float(col.iloc[0])   # most recent period
        except Exception:
            pass
        return None

    if tk is not None:
        try:
            fin = getattr(tk, "financials", None)
            bs = getattr(tk, "balance_sheet", None)
            cf = getattr(tk, "cashflow", None)
            rev = _srow(fin, "Total Revenue", "Operating Revenue")
            opinc = _srow(fin, "Operating Income", "EBIT")
            gp = _srow(fin, "Gross Profit")
            ta = _srow(bs, "Total Assets")
            cliab = _srow(bs, "Current Liabilities",
                          "Total Current Liabilities")
            ocf = _srow(cf, "Operating Cash Flow",
                        "Total Cash From Operating Activities",
                        "Cash Flow From Continuing Operating Activities")
            fcf = _srow(cf, "Free Cash Flow")
            if fcf is None and ocf is not None:
                capex = _srow(cf, "Capital Expenditure")
                if capex is not None:
                    fcf = ocf + capex if capex < 0 else ocf - capex
            if opinc and ta and cliab and (ta - cliab) > 0:
                info["_roce"] = opinc / (ta - cliab)
            if rev and rev > 0:
                if gp is not None:
                    info["_gross_margin"] = gp / rev
                if opinc is not None:
                    info["_op_margin"] = opinc / rev
                if ocf is not None:
                    info["_cf_margin"] = ocf / rev
                if fcf is not None:
                    info["_fcf_margin"] = fcf / rev
        except Exception:
            pass
    # info-field fallbacks (TTM ratios from the profile)
    if info.get("_gross_margin") is None and info.get("grossMargins"):
        info["_gross_margin"] = info["grossMargins"]
    if info.get("_op_margin") is None and info.get("operatingMargins"):
        info["_op_margin"] = info["operatingMargins"]

    # ── Step 3: nightly dump fills whatever is still missing ─────────
    df_funds = dump_fundamentals(ticker)
    filled = []
    for dk, ik in DUMP_INFO_MAP.items():
        v = df_funds.get(dk)
        if v is None:
            continue
        if ik.startswith("_scan_") or ik.startswith("_div") or not info.get(ik):
            info[ik] = v
            filled.append(dk)
    if df_funds.get("Sector") and not info.get("sector"):
        info["sector"] = df_funds["Sector"]
    if filled:
        info["_from_scan_dump"] = True
        info["_dump_fields"] = filled

    # ── Step 4: EPS history — earnings_history → income stmt fallback ─
    # (NEVER tk.quarterly_earnings: deprecated + crash-prone upstream)
    eps_history = []
    if tk is not None:
        try:
            eh = getattr(tk, "earnings_history", None)
            if eh is not None and hasattr(eh, "empty") and not eh.empty:
                cols = {c.lower(): c for c in eh.columns}
                ac = cols.get("epsactual"); ec = cols.get("epsestimate")
                sc = cols.get("surprisepercent")
                for idx, row in eh.iterrows():
                    a = row.get(ac) if ac else None
                    e = row.get(ec) if ec else None
                    s = row.get(sc) if sc else None
                    if a is None and e is None:
                        continue
                    if s is None and a is not None and e not in (None, 0):
                        try: s = (float(a) - float(e)) / abs(float(e)) * 100
                        except Exception: s = None
                    try:
                        qd = pd.to_datetime(idx, errors="coerce")
                        ql = qd.strftime("%b %Y") if pd.notna(qd) else str(idx)
                    except Exception:
                        ql = str(idx)
                    eps_history.append(dict(quarter=ql,
                                            actual=float(a) if a is not None and pd.notna(a) else None,
                                            estimate=float(e) if e is not None and pd.notna(e) else None,
                                            surprise=float(s) if s is not None and pd.notna(s) else None))
                eps_history = eps_history[-8:]
        except Exception:
            eps_history = []
    if not eps_history and tk is not None:
        try:
            qis = getattr(tk, "quarterly_income_stmt", None)
            if qis is not None and hasattr(qis, "empty") and not qis.empty:
                ri = {str(i).strip().lower(): i for i in qis.index}
                er = ri.get("diluted eps") or ri.get("basic eps")
                if er is not None:
                    for col in qis.columns:
                        v = qis.loc[er, col]
                        if pd.isna(v):
                            continue
                        try:
                            qd = pd.to_datetime(col, errors="coerce")
                            ql = qd.strftime("%b %Y") if pd.notna(qd) else str(col)
                        except Exception:
                            ql = str(col)
                        eps_history.append(dict(quarter=ql, actual=float(v),
                                                estimate=None, surprise=None))
                    eps_history = list(reversed(eps_history))[-8:]
        except Exception:
            pass
    if not eps_history:
        _issues.append("EPS history: earnings records unavailable (thin coverage, ETF, or feed blocked)")

    # ── Forward EPS estimates ─────────────────────────────────────────
    eps_forward = []
    if tk is not None:
        try:
            ee = getattr(tk, "earnings_estimate", None)
            if ee is not None and hasattr(ee, "empty") and not ee.empty:
                labels = {"0q": "Next Qtr", "+1q": "Qtr After",
                          "0y": "This Year", "+1y": "Next Year"}
                cols = {c.lower(): c for c in ee.columns}
                av = cols.get("avg") or cols.get("average")
                nn = cols.get("numberofanalysts")
                for pk in ["0q", "+1q", "0y", "+1y"]:
                    if pk in ee.index:
                        row = ee.loc[pk]
                        est = row.get(av) if av else None
                        na = row.get(nn) if nn else None
                        if est is not None and pd.notna(est):
                            eps_forward.append(dict(period=labels[pk], estimate=float(est),
                                                    n_analysts=int(na) if na is not None and pd.notna(na) else None,
                                                    is_forward=True))
        except Exception:
            pass
    if not eps_forward and info:
        try:
            cy, ny = info.get("epsCurrentYear"), info.get("epsNextYear") or info.get("forwardEps")
            if cy is not None:
                eps_forward.append(dict(period="This Year (est)", estimate=float(cy),
                                        n_analysts=info.get("numberOfAnalystOpinions"), is_forward=True))
            if ny is not None and ny != cy:
                eps_forward.append(dict(period="Next Year (est)", estimate=float(ny),
                                        n_analysts=info.get("numberOfAnalystOpinions"), is_forward=True))
        except Exception:
            pass

    info["_hist_source"] = hist_src
    if _issues:
        info["_data_issues"] = _issues
    return info, hist, eps_history, eps_forward


# ═════════════════════════════════════════════════════════════════════
# Macro regime (ported from the macro simulator's gkey logic) + mega scan
# ═════════════════════════════════════════════════════════════════════
# The simulator keys six regimes off oil, CPI, dollar, QE/SLR and the curve.
# Live translation: oil = USO trend, dollar = UUP trend, QE-ness = the
# pressure gauge (net liquidity + stablecoins + credit), shock = VIX impulse.
REGIME_LABELS = {
    "qe":     "💧 Easy Money — the Fed's adding liquidity, risk assets float up",
    "stag":   "🔥 Hot Inflation — oil & hard assets win, growth gets repriced",
    "bull":   "☀️ Risk-On Calm — steady climb, cyclicals & tech lead",
    "bear":   "⛈️ Fear / Risk-Off — money flees to safety & gold",
    "strong": "💵 Rising Dollar — US quality holds, gold & foreign lag",
    "repress": "💸 Debasement — money printed to cap yields; cash & bonds bleed, hard assets hold",
    "base":   "⛅ No Clear Driver — no dominant force, quality quietly wins",
}

# short names for the dropdown (emoji + plain name only)
REGIME_NAMES = {
    "qe": "💧 Easy Money", "stag": "🔥 Hot Inflation", "bull": "☀️ Risk-On Calm",
    "bear": "⛈️ Fear / Risk-Off", "strong": "💵 Rising Dollar",
    "repress": "💸 Debasement", "base": "⛅ No Clear Driver",
}

# structured explainer cards (Option 3): name, aka, story, leads, lags, trigger
REGIME_CARDS = {
    "qe": dict(
        emoji="💧", name="Easy Money", aka="QE / liquidity flood",
        story="The Fed is pumping liquidity into the system. With cheap money "
              "everywhere, investors reach for risk — the more speculative, the "
              "better — and hard assets like gold rise as the dollar is diluted.",
        leads="Tech · Growth · Small caps · Gold & miners · Crypto-adjacent",
        lags="Cash · Defensive staples · Utilities",
        trigger="the pressure gauge is firmly positive and liquidity is expanding"),
    "stag": dict(
        emoji="🔥", name="Hot Inflation", aka="stagflation",
        story="Prices are rising faster than the economy is growing. Oil, metals, "
              "and the companies that dig them up hold their value; anything priced "
              "on future growth gets marked down as rates stay high.",
        leads="Energy · Materials & miners · Consumer staples · Utilities",
        lags="Tech · Consumer discretionary · Real estate · High-growth",
        trigger="oil is climbing, the Fed is holding rates high, and bonds are weak"),
    "bull": dict(
        emoji="☀️", name="Risk-On Calm", aka="calm melt-up",
        story="The economy is growing steadily and fear is low, so money moves out "
              "the risk curve into the things that do best when the expansion runs — "
              "economically-sensitive and higher-beta names.",
        leads="Consumer discretionary · Tech · Industrials · Financials · Small caps",
        lags="Defensive staples · Utilities · Energy",
        trigger="oil is soft, volatility is calm, and no shock is on the tape"),
    "bear": dict(
        emoji="⛈️", name="Fear / Risk-Off", aka="shock / crash",
        story="Something broke and volatility is spiking. Money doesn't ask "
              "questions — it flees anything risky and crowds into safety, "
              "defensives, and gold until the dust settles.",
        leads="Consumer staples · Utilities · Healthcare · Gold · Defense",
        lags="High-beta · Tech · Consumer discretionary · Small caps · Crypto-adjacent",
        trigger="the VIX is spiking, or a yen-carry unwind signature is firing"),
    "strong": dict(
        emoji="💵", name="Rising Dollar", aka="strong-dollar squeeze",
        story="A strengthening dollar squeezes anything that earns money abroad or "
              "is priced in dollars. Domestic, US-focused quality holds up while "
              "commodities and foreign markets lag.",
        leads="US-focused financials · Domestic quality · Defensive staples",
        lags="Gold · Emerging markets · Materials · Multinational exporters",
        trigger="the dollar is trending strongly higher over the past quarter"),
    "repress": dict(
        emoji="💸", name="Debasement", aka="financial repression / stealth QE",
        story="The government is buying its own debt with newly created money to "
              "hold borrowing costs down. Liquidity expands like QE, but yields are "
              "capped by policy rather than by demand — so the currency quietly "
              "erodes instead. Cash and long bonds lose purchasing power while "
              "things that can't be printed, and businesses that pass inflation "
              "straight through, hold their value.",
        leads="Gold & silver miners · Materials · Energy · Payment & insurance "
              "tolls · Real assets · Cash-generative compounders",
        lags="Cash · Long-duration bonds · Unprofitable growth · High-multiple tech",
        trigger="Treasury buybacks expand and the Fed's balance sheet grows while "
                "long yields stay pinned, gold makes new highs, and the dollar erodes"),
    "base": dict(
        emoji="⛅", name="No Clear Driver", aka="base case",
        story="No single macro force is in charge. Without a dominant tailwind or "
              "headwind, the market rewards fundamentals — durable, well-run, "
              "reasonably-priced businesses quietly win.",
        leads="Quality across sectors · Steady compounders",
        lags="Story stocks without earnings · Deep cyclicals",
        trigger="none of the other five conditions is clearly present"),
}

# regime → sector multiplier (distilled from the simulator's per-stock
# base/bull/bear/qe/stag/strong expected-return DB)
SECTOR_TILTS = {
    "qe":     {"Technology": 1.20, "Communication Services": 1.12, "Consumer Cyclical": 1.10,
               "Basic Materials": 1.18, "Financial Services": 1.10, "Real Estate": 1.08,
               "Industrials": 1.02, "Healthcare": 0.98, "Energy": 0.95,
               "Consumer Defensive": 0.88, "Utilities": 0.88},
    "stag":   {"Energy": 1.25, "Basic Materials": 1.18, "Consumer Defensive": 1.10,
               "Utilities": 1.06, "Healthcare": 1.02, "Industrials": 0.95,
               "Financial Services": 0.92, "Technology": 0.85,
               "Communication Services": 0.88, "Consumer Cyclical": 0.80, "Real Estate": 0.82},
    "bull":   {"Consumer Cyclical": 1.18, "Technology": 1.15, "Industrials": 1.10,
               "Financial Services": 1.10, "Communication Services": 1.08,
               "Basic Materials": 1.00, "Real Estate": 1.02, "Healthcare": 0.96,
               "Energy": 0.90, "Consumer Defensive": 0.88, "Utilities": 0.88},
    "bear":   {"Energy": 1.15, "Consumer Defensive": 1.15, "Utilities": 1.12,
               "Healthcare": 1.08, "Basic Materials": 1.05, "Industrials": 1.00,
               "Financial Services": 0.90, "Communication Services": 0.90,
               "Technology": 0.85, "Real Estate": 0.85, "Consumer Cyclical": 0.80},
    "strong": {"Financial Services": 1.10, "Utilities": 1.06, "Consumer Defensive": 1.05,
               "Healthcare": 1.04, "Technology": 1.00, "Communication Services": 1.00,
               "Industrials": 0.95, "Consumer Cyclical": 0.95, "Real Estate": 0.92,
               "Energy": 0.92, "Basic Materials": 0.85},
    # Debasement: currency erodes while yields are policy-capped. Money goes to
    # things that can't be printed (metals, real assets) and to businesses that
    # pass inflation straight through (payments, waste, insurance tolls).
    # Long-duration/high-multiple growth is the funding source, not the winner.
    "repress": {"Basic Materials": 1.25, "Energy": 1.15, "Financial Services": 1.12,
                "Real Estate": 1.10, "Industrials": 1.06, "Consumer Defensive": 1.05,
                "Utilities": 1.00, "Consumer Cyclical": 0.95, "Healthcare": 0.95,
                "Communication Services": 0.85, "Technology": 0.82},
    "base":   {},
}


def macro_regime(closes: pd.DataFrame, pressure_gauge=None) -> dict:
    """Classify the live macro state — same decision order as the simulator's
    gkey(): qe → stag → bull → bear → strong → base."""
    def r63(sym):
        if sym not in closes.columns:
            return np.nan
        s = closes[sym].dropna()
        return float(s.iloc[-1] / s.iloc[-64] - 1) if len(s) > 64 else np.nan
    imp = impulses(closes).iloc[-1]
    oil, uup, tlt = r63("USO"), r63("UUP"), r63("TLT")
    vix_z = float(imp.get("^VIX", np.nan))
    gold = r63("GLD")
    drivers = []
    if (pressure_gauge is not None and pressure_gauge >= 1
            and np.isfinite(gold) and gold >= 0.08
            and (not np.isfinite(uup) or uup <= 0.02)):
        # liquidity expanding AND gold leading AND the dollar not strengthening:
        # that combination is debasement, not a risk-on liquidity melt-up
        reg = "repress"
        drivers.append(f"gauge +{pressure_gauge} with gold {gold:+.0%}/63d and a soft dollar")
    elif pressure_gauge is not None and pressure_gauge >= 2:
        reg = "qe"; drivers.append(f"pressure gauge +{pressure_gauge} (liquidity building)")
    elif np.isfinite(oil) and oil > 0.15 and (not np.isfinite(tlt) or tlt < 0):
        reg = "stag"; drivers.append(f"oil +{oil:.0%}/63d with bonds soft")
    elif np.isfinite(oil) and oil < -0.05 and (not np.isfinite(vix_z) or vix_z < 0.5):
        reg = "bull"; drivers.append(f"oil {oil:+.0%}/63d, vol calm")
    elif np.isfinite(vix_z) and vix_z >= 1.25:
        reg = "bear"; drivers.append(f"VIX impulse z {vix_z:+.1f} (shock)")
    elif (np.isfinite(imp.get("FXY", np.nan)) and imp.get("FXY") >= 1.75
          and np.isfinite(imp.get("QQQ", np.nan)) and imp.get("QQQ") <= -0.75):
        reg = "bear"; drivers.append(
            f"yen carry unwind signature (FXY z {imp.get('FXY'):+.1f}, QQQ draining)")
    elif np.isfinite(uup) and uup > 0.04:
        reg = "strong"; drivers.append(f"dollar +{uup:.0%}/63d")
    else:
        reg = "base"; drivers.append("no dominant macro force")
    if np.isfinite(gold):
        drivers.append(f"gold {gold:+.0%}/63d")
    return dict(regime=reg, label=REGIME_LABELS[reg], drivers=drivers)


def _node_follow_corr(node: str, node_closes: pd.DataFrame):
    """Vectorized lagged corr of one node vs EVERY dump stock (t → t+5)."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    C = panel["c"]
    n = node_closes[node].dropna()
    n.index = pd.to_datetime(n.index).tz_localize(None)
    common = dts.intersection(n.index)
    if len(common) < 120:
        return None
    n_ix = {d: i for i, d in enumerate(dts)}
    rows_ix = np.array([n_ix[d] for d in common])
    Cc = C[rows_ix]
    nd = n.reindex(common).values
    node_r5 = nd[5:] / nd[:-5] - 1.0
    stk_r5 = Cc[5:] / Cc[:-5] - 1.0
    x = node_r5[:-5]; y = stk_r5[5:]
    xm = x - np.nanmean(x)
    ym = y - np.nanmean(y, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nansum(xm[:, None] * ym, axis=0) / (
            np.sqrt(np.nansum(xm ** 2) * np.nansum(ym ** 2, axis=0)))
    return corr


def mega_scan(node_closes: pd.DataFrame, pressure_gauge=None, top: int = 20,
              regime_override: str | None = None,
              apply_macro: bool = True,
              hot_only: int = 0, use_sector_flow: bool = True,
              flow_live: bool = False, flow_lookback: int = SECTOR_FLOW_LOOKBACK,
              flow_offset: int = 0) -> tuple:
    """THE combined screener: IGNITION technicals + macro-simulator quality
    DNA + cascade tailwind + macro-regime sector fit, over the whole dump
    (all markets). Returns (top-N DataFrame, regime dict)."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    funds = dump_fundamentals_all()
    C, V = panel["c"], np.nan_to_num(panel["v"])
    T, N = C.shape
    px = C[-1]
    # Stale-price guard: the panel forward-fills gaps, so a stock whose last
    # real print was days ago can still show a "current" price. Require an
    # actual (non-filled) close in the last 3 sessions of the raw panel.
    _raw_recent = _recent_ok_mask(panel)
    tradeable = np.isfinite(px) & (px >= 3.0) & (mdv >= 2e6) & _raw_recent

    def _pct(a):
        s = pd.Series(np.where(tradeable, a, np.nan))
        return s.rank(pct=True).values

    # ── pillar 1: IGNITION technicals (vectorized from OHLCV) ────────
    mom63 = C[-6] / C[-64] - 1.0
    lo63 = np.nanmin(panel["l"][-63:], 0); hi63 = np.nanmax(panel["h"][-63:], 0)
    rangepos = (px - lo63) / np.where(hi63 - lo63 == 0, np.nan, hi63 - lo63)
    rvol = V[-5:].mean(0) / np.where(V[-63:].mean(0) == 0, np.nan, V[-63:].mean(0))
    ma50 = np.nanmean(C[-50:], 0)
    above50 = (px > ma50).astype(float)
    cl = pd.DataFrame(C)
    d = cl.diff()
    up = d.clip(lower=0).rolling(14).mean(); dn = (-d.clip(upper=0)).rolling(14).mean()
    rsi = (100 - 100 / (1 + up / dn.replace(0, np.nan))).iloc[-1].values
    rsi_sweet = ((rsi > 45) & (rsi < 65)).astype(float)
    e12 = cl.ewm(span=12, adjust=False).mean(); e26 = cl.ewm(span=26, adjust=False).mean()
    macd = (e12 - e26)
    sig = macd.ewm(span=9, adjust=False).mean()
    mb = macd.values > sig.values
    macd_bull = mb[-1].astype(float)
    tech = (0.30 * _pct(mom63) + 0.22 * _pct(rangepos) + 0.18 * _pct(np.minimum(rvol, 5))
            + 0.10 * above50 + 0.10 * rsi_sweet + 0.10 * macd_bull)

    # ── catalyst pillar: data fingerprints of IGNITION's catalyst types ──
    # (walk-forward validated on the nightly dump: adding this at 0.15x the
    #  tech weight lifted top-20 excess from +2.78% to +3.58%/21d, 69% hit,
    #  positive in both honesty halves)
    brk = px >= np.nanmax(panel["h"][-63:-1], 0) * 0.999            # breakout
    ret1d = C[-1] / C[-2] - 1.0
    vshock = (rvol >= 2.5) & (np.abs(ret1d) >= 0.04)                # volume shock
    gaps = np.abs(panel["o"][-5:] / C[-6:-1] - 1.0)
    gp = np.nanmax(gaps, 0) >= 0.03                                  # recent gap
    fresh = mb[-1] & ~mb[-4]                                         # fresh MACD cross
    squeeze_setup = (np.nan_to_num(funds["ShortPctFloat"]) >= 0.15) & (mom63 > 0)
    cat = (0.35 * np.nan_to_num(brk) + 0.25 * np.nan_to_num(vshock)
           + 0.20 * np.nan_to_num(gp) + 0.20 * np.nan_to_num(fresh))
    cat_tags = []
    for k in range(N):
        tg = []
        if brk[k]: tg.append("🚀 breakout")
        if vshock[k]: tg.append("⚡ vol shock")
        if gp[k]: tg.append("🕳 gap")
        if fresh[k]: tg.append("📈 MACD cross")
        if squeeze_setup[k]: tg.append("🩳 squeeze setup")
        cat_tags.append(" · ".join(tg))
    cat_tags = np.array(cat_tags, dtype=object)

    # ── pillar 2: quality DNA (macro-simulator scoring philosophy) ───
    # UNKNOWN IS NOT AVERAGE. Previously a missing ROIC became 0.0 and was
    # percentile-ranked — but 41% of real stocks have NEGATIVE ROIC, so a
    # fake zero landed near the 43rd percentile and OUTRANKED them. Missing
    # Piotroski defaulted to 4/9, above the market median of 3. Net effect:
    # companies with no data scored like average companies. Now every missing
    # fundamental ranks at the BOTTOM of its component, and the row's overall
    # data completeness is reported so the user can see what's actually known.
    piotr = funds["Piotroski"]; gc = funds["GoldenCross"]
    roic = funds["ROIC"]; rg = funds["RevenueGrowth"]; eg = funds["EarningsGrowth"]

    def _pct_missing_last(a):
        """Percentile rank among tradeable stocks; NaN -> 0.0 (worst)."""
        s = pd.Series(np.where(tradeable & np.isfinite(a), a, np.nan))
        r = s.rank(pct=True).values
        return np.nan_to_num(r, nan=0.0)

    quality = (0.35 * np.nan_to_num(np.clip(piotr / 9.0, 0, 1), nan=0.0)
               + 0.15 * np.nan_to_num(np.clip(gc, 0, 1), nan=0.0)
               + 0.20 * _pct_missing_last(np.clip(roic, -1, 2))
               + 0.15 * _pct_missing_last(np.clip(rg, -1, 3))
               + 0.15 * _pct_missing_last(np.clip(eg, -2, 5)))

    # data completeness: how many of the 5 quality inputs this stock actually has
    _have = (np.isfinite(piotr).astype(int) + np.isfinite(gc).astype(int)
             + np.isfinite(roic).astype(int) + np.isfinite(rg).astype(int)
             + np.isfinite(eg).astype(int))
    _missing_names = []
    for k in range(N):
        miss = []
        if not np.isfinite(piotr[k]): miss.append("Piotroski")
        if not np.isfinite(roic[k]): miss.append("ROIC")
        if not np.isfinite(rg[k]): miss.append("RevGrowth")
        if not np.isfinite(eg[k]): miss.append("EarnGrowth")
        if not np.isfinite(gc[k]): miss.append("GoldenCross")
        _missing_names.append(", ".join(miss))
    _missing_names = np.array(_missing_names, dtype=object)

    # ── pillar 3: cascade tailwind (waves already moving toward it) ──
    imp = impulses(node_closes).iloc[-1]
    tail = np.zeros(N)
    hot = [(nsym, float(z)) for nsym, z in imp.items()
           if np.isfinite(z) and abs(z) >= 1.0 and nsym in node_closes.columns]
    used_nodes = []
    for nsym, z in hot:
        corr = _node_follow_corr(nsym, node_closes)
        if corr is None:
            continue
        c = np.where(np.abs(corr) >= 0.12, corr, 0.0)
        tail += np.nan_to_num(c) * np.clip(z, -3, 3)
        used_nodes.append(NODES.get(nsym, (nsym,))[0])
    tail_pct = _pct(tail) if len(used_nodes) else np.full(N, 0.5)

    # ── pillar 4: macro-regime sector fit ────────────────────────────
    if not apply_macro:
        # macro lens OFF — pure flow/quality ranking, no sector tilt at all
        regime = dict(regime="off", label="🚫 Macro lens off — no sector tilt applied",
                      drivers=["macro lens disabled — ranking on the raw score"])
        tilts = {}
    elif regime_override and regime_override in REGIME_LABELS:
        regime = dict(regime=regime_override, label=REGIME_LABELS[regime_override],
                      drivers=["manual scenario override — matched to your "
                               "Macro Sim sliders, not live detection"])
        tilts = SECTOR_TILTS.get(regime["regime"], {})
    else:
        regime = macro_regime(node_closes, pressure_gauge)
        tilts = SECTOR_TILTS.get(regime["regime"], {})
    macro_mult = np.array([tilts.get(s, 1.0) for s in sectors])
    sec_arr = np.array(sectors)

    # ── sector flow: where money went in the last session ────────────
    flow_pct = np.full(N, 0.5)
    flow_df = pd.DataFrame()
    if use_sector_flow or hot_only:
        try:
            flow_df = sector_flow(lookback=flow_lookback, offset=flow_offset,
                                  use_live=flow_live)
        except Exception:
            flow_df = pd.DataFrame()
    if not flow_df.empty:
        fmap = dict(zip(flow_df.Sector, flow_df.FlowPct))
        flow_pct = np.array([fmap.get(str(s), 0.5) for s in sec_arr])
        if hot_only:
            hot = set(flow_df.Sector.head(int(hot_only)))
            tradeable = tradeable & np.array([str(s) in hot for s in sec_arr])

    # ── base cascade score (technicals + quality + tailwind + catalyst) ──
    # NOTE: the macro multiplier is applied to RANKING WITHIN sectors, not as
    # a global scale — otherwise the single most-favored sector sweeps the
    # whole board. See the diversified allocator below.
    core = (42 * tech + 23 * quality + 29 * tail_pct + 6 * cat)
    if use_sector_flow and not flow_df.empty:
        # +8 points of sector-flow percentile == the 0.20 tilt validated in the
        # walk-forward (tech spans 42 points there, 0.20 x 42 ~ 8)
        core = core + SECTOR_FLOW_WEIGHT * flow_pct
    core = np.where(tradeable, core, -np.inf)

    if not tilts:
        # base / no-regime: plain global top-N by cascade score
        score = core.copy()
        order = np.argsort(-score)[:top]
    else:
        # ── DIVERSIFIED SCENARIO ALLOCATION ──────────────────────────
        # 1. the scenario's LEADS = sectors it favors (tilt >= 1.0).
        # 2. give each lead a SLOT QUOTA proportional to its tilt, but
        #    capped so no sector can dominate — forces a diverse list.
        # 3. within each sector, pick the best names by cascade score.
        # 4. backfill any remaining slots with the best leftover names
        #    (favored sectors first) so we always return `top` picks.
        leads = {s: t for s, t in tilts.items() if t >= 1.0}
        if not leads:                       # all-defensive regime: take top few
            leads = dict(sorted(tilts.items(), key=lambda kv: -kv[1])[:4])
        weight_sum = sum(leads.values())
        # cap: at most ~35% of the book in any one sector (min 2 slots)
        cap = max(2, int(np.ceil(top * 0.35)))
        quota = {}
        for s, t in leads.items():
            q = int(round(top * (t / weight_sum)))
            quota[s] = min(max(q, 1), cap)

        chosen = []
        picked_mask = np.zeros(len(core), dtype=bool)
        for s in sorted(leads, key=lambda x: -leads[x]):
            sec_ix = np.where((sec_arr == s) & np.isfinite(core))[0]
            if not len(sec_ix):
                continue
            sec_ix = sec_ix[np.argsort(-core[sec_ix])][:quota[s]]
            chosen.extend(sec_ix.tolist())
            picked_mask[sec_ix] = True
            if len(chosen) >= top:
                break

        # backfill remaining slots — best leftover names, favored sectors
        # weighted up so the thesis still tilts the fill, capped per sector
        if len(chosen) < top:
            from collections import Counter
            sec_count = Counter(sec_arr[chosen].tolist())
            fill_score = np.where(picked_mask, -np.inf, core) * macro_mult
            for gi in np.argsort(-fill_score):
                if len(chosen) >= top:
                    break
                if picked_mask[gi] or not np.isfinite(core[gi]):
                    continue
                s = sec_arr[gi]
                if sec_count[s] >= cap:       # respect the diversity cap
                    continue
                chosen.append(int(gi)); picked_mask[gi] = True
                sec_count[s] += 1

        chosen = chosen[:top]
        # order the final book by cascade score (best first)
        score = core.copy()
        order = np.array(chosen)[np.argsort(-core[chosen])] if chosen else np.array([], dtype=int)

    df = pd.DataFrame({
        "Ticker": tickers[order], "Sector": sec_arr[order],
        "Price": px[order].round(2), "Score": core[order].round(1),
        "Tech": (tech[order] * 100).round(0), "Quality": (quality[order] * 100).round(0),
        "Tailwind": (tail_pct[order] * 100).round(0),
        "MacroFit": macro_mult[order].round(2),
        "Piotroski": piotr[order], "RevGrowth": rg[order],
        "RVOL": np.round(rvol[order], 2), "RangePos": np.round(rangepos[order], 2),
        "SecFlow": np.round(flow_pct[order] * 100, 0),
        "Data": [f"{h}/5" for h in _have[order]],
        "Missing": _missing_names[order],
        "Catalysts": cat_tags[order],
    })
    regime["hot_nodes"] = used_nodes
    regime["n_sectors"] = int(df.Sector.nunique()) if not df.empty else 0
    regime["flow"] = flow_df
    regime["hot_only"] = int(hot_only)
    return df.reset_index(drop=True), regime


# ═════════ IGNITION news catalysts (keyword buckets, ported) ═════════
CATALYST_KEYWORDS = {
    "earnings": ["earnings", "eps", "revenue beat", "quarterly results", "q1", "q2",
                 "q3", "q4", "fiscal", "guidance", "outlook", "profit", "loss", "surprise"],
    "fda": ["fda", "food and drug", "pdufa", "nda", "bla", "inda", "clinical trial",
            "phase 1", "phase 2", "phase 3", "approval", "approved", "clearance",
            "510k", "drug", "biologics", "clinical hold"],
    "legal": ["lawsuit", "settlement", "verdict", "litigation", "court", "ruling",
              "judgment", "class action", "sued", "damages", "injunction", "doj",
              "sec investigation", "subpoena", "antitrust"],
    "buyout": ["acquisition", "acquire", "merger", "takeover", "buyout", "going private",
               "lbo", "strategic review", "sale process", "offer to acquire", "bid for",
               "deal with", "m&a"],
    "partnership": ["partnership", "collaboration", "joint venture", "alliance",
                    "agreement", "contract", "mou", "supply agreement", "licensing deal",
                    "strategic agreement", "selected by"],
    "squeeze": ["short squeeze", "short interest", "most shorted", "short seller",
                "short covering", "days to cover"],
    "breakout": ["52-week high", "all-time high", "breakout", "new high",
                 "technical breakout", "resistance broken", "record high"],
    "geopolitical": ["tariff", "sanction", "trade war", "geopolitical", "supply chain",
                     "export ban", "china", "russia", "ukraine", "energy crisis", "oil",
                     "opec", "nato", "war", "conflict", "defense contract", "pentagon"],
    "rate": ["fed", "federal reserve", "interest rate", "rate hike", "rate cut", "fomc",
             "powell", "inflation", "cpi", "ppi", "hawkish", "dovish", "treasury yield"],
    "earn_growth": ["record earnings", "earnings growth", "eps growth", "profit surge",
                    "earnings beat", "record profit", "blowout quarter", "record quarter",
                    "beat estimates", "exceeded expectations", "top-line beat"],
}
CATALYST_MIN_HITS = {"earnings": 1, "fda": 2, "legal": 2, "buyout": 2, "partnership": 2,
                     "squeeze": 1, "breakout": 1, "geopolitical": 2, "rate": 2, "earn_growth": 1}
CATALYST_SECTOR_WHITELIST = {
    "fda": ["health", "pharma", "biotech", "drug", "life science", "medical",
            "clinical", "therapeut", "diagnostic", "biolog", "genomic"],
    "geopolitical": ["energy", "material", "defense", "industrial", "semiconductor",
                     "technology", "mining", "oil", "chemical", "aerospace", "transport"],
    "rate": ["financial", "bank", "real estate", "reit", "utility", "insurance",
             "mortgage", "savings", "trust"],
}
CATALYST_EMOJI = {"earnings": "📊", "fda": "💊", "legal": "⚖️", "buyout": "🤝",
                  "partnership": "🔗", "squeeze": "🩳", "breakout": "🚀",
                  "geopolitical": "🌍", "rate": "🏦", "earn_growth": "📈"}


def news_catalysts(tickers: list, sectors: dict | None = None) -> dict:
    "IGNITION's news-keyword catalyst tags for a SHORTLIST (min-hit + sector rules)."
    os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")
    sectors = sectors or {}
    out = {}
    try:
        import yfinance as yf
    except Exception:
        return out
    for tk in tickers:
        try:
            arts = yf.Ticker(tk).news or []
        except Exception:
            continue
        blob = " ".join(
            f"{(a.get('content') or a).get('title','')} {(a.get('content') or a).get('summary','')}"
            for a in arts[:12]).lower()
        if not blob.strip():
            continue
        sec = (sectors.get(tk) or "").lower()
        tags = []
        for ctype, words in CATALYST_KEYWORDS.items():
            wl = CATALYST_SECTOR_WHITELIST.get(ctype)
            if wl is not None and sec and not any(w in sec for w in wl):
                continue
            hits = sum(blob.count(w) > 0 for w in words)
            if hits >= CATALYST_MIN_HITS[ctype]:
                tags.append(f"{CATALYST_EMOJI[ctype]} {ctype}")
        if tags:
            out[tk] = " · ".join(tags[:4])
    return out


def felix_scan(top: int = 20) -> pd.DataFrame:
    """🎩 Felix — the investment-banker quality checklist from the hybrid
    screener, run across the entire nightly dump. Five tests: return on
    capital (ROIC), moat (proxied by ROIC + its trend — the dump carries no
    gross margin), cash (owner-earnings yield), stability (Piotroski), sane
    price (hard P/E gate 0-50). Regime-agnostic by design — quality doesn't
    rotate with the weather. Weights mirror the screener preset exactly:
    ROIC x5, OE_Yield x4, Piotroski x4, ROIC_Trend x2, growth x1 each."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    funds = dump_fundamentals_all()
    px = panel["c"][-1]
    pe = funds["P/E"]
    roic = funds["ROIC"]; oe = funds["OE_Yield"]; pio = funds["Piotroski"]
    rt = funds["ROIC_Trend"]; rg = funds["RevenueGrowth"]; eg = funds["EarningsGrowth"]
    ok = (np.isfinite(px) & (px >= 3.0) & (mdv >= 2e6)
          & np.isfinite(pe) & (pe > 0) & (pe <= 50)          # Test 5: sane price
          & np.isfinite(roic) & np.isfinite(pio))

    # UNKNOWN IS NOT AVERAGE (same fix as mega_scan): a missing input ranks
    # at the BOTTOM of its component instead of being coerced to 0.0 and
    # percentile-ranked into the middle of the pack. ROIC and Piotroski are
    # already hard-required by `ok`; OE yield, ROIC trend and the growth
    # figures previously slipped through as fake zeros.
    def _pct(a):
        """Percentile among eligible names; missing -> 0.0 (worst)."""
        s = pd.Series(np.where(ok & np.isfinite(a), a, np.nan))
        return np.nan_to_num(s.rank(pct=True).values, nan=0.0)

    score = (5 * _pct(roic)
             + 4 * _pct(oe)
             + 4 * np.nan_to_num(np.clip(pio / 9.0, 0, 1), nan=0.0)
             + 2 * _pct(rt)
             + 1 * _pct(np.clip(rg, -1, 3))
             + 1 * _pct(np.clip(eg, -2, 5))) / 17 * 100
    score = np.where(ok, score, -np.inf)
    order = np.argsort(-score)[:top]
    # tests: a missing input can never PASS a test (isfinite guard)
    tests = ((np.isfinite(roic) & (roic >= 0.15)).astype(int)
             + (np.isfinite(pio) & (pio >= 7)).astype(int)
             + (np.isfinite(oe) & (oe >= 0.04)).astype(int)
             + (np.isfinite(rt) & (rt > 0)).astype(int))
    have = (np.isfinite(roic).astype(int) + np.isfinite(oe).astype(int)
            + np.isfinite(pio).astype(int) + np.isfinite(rt).astype(int)
            + np.isfinite(rg).astype(int) + np.isfinite(eg).astype(int))
    return pd.DataFrame({
        "Ticker": tickers[order], "Sector": np.array(sectors)[order],
        "Price": px[order].round(2), "Felix": np.round(score[order], 1),
        "Tests": [f"{t}/4" for t in tests[order]],
        "Data": [f"{h}/6" for h in have[order]],
        "ROIC": roic[order], "OE Yield": oe[order],
        "Piotroski": pio[order], "ROIC Trend": rt[order],
        "P/E": pe[order], "RevGrowth": rg[order],
    }).reset_index(drop=True)


# ═════════ 🇯🇵 yen carry trade monitor ═════════
def yen_carry_monitor(closes: pd.DataFrame) -> dict:
    """The world's biggest funding trade, watched live. Borrowing at Japan's
    near-zero rates to fund global risk positions works until the yen surges —
    then every yen-funded long gets margin-called at once (Aug 5, 2024).
    The unwind SIGNATURE is coincidence: yen impulse UP while risk impulses
    point DOWN. Yen up alone is a currency move; yen up + QQQ/BTC down +
    VIX up is forced deleveraging."""
    imp = impulses(closes).iloc[-1]
    def z(sym):
        v = imp.get(sym, np.nan)
        return float(v) if np.isfinite(v) else np.nan
    def t63(sym):
        if sym not in closes.columns:
            return np.nan
        s = closes[sym].dropna()
        return float(s.iloc[-1] / s.iloc[-64] - 1) if len(s) > 64 else np.nan

    yen_z, yen_t = z("FXY"), t63("FXY")
    nik_z, nik_t = z("^N225"), t63("^N225")
    qqq_z, btc_z, vix_z = z("QQQ"), z("BTC-USD"), z("^VIX")

    confirms = []
    if np.isfinite(qqq_z) and qqq_z <= -0.75:
        confirms.append("QQQ draining")
    if np.isfinite(btc_z) and btc_z <= -0.75:
        confirms.append("BTC draining (24/7 canary)")
    if np.isfinite(vix_z) and vix_z >= 0.75:
        confirms.append("VIX waking")
    if np.isfinite(nik_z) and nik_z <= -1.0:
        confirms.append("Nikkei cracking")

    stress = 0.0
    if np.isfinite(yen_z) and yen_z > 0:
        stress = yen_z * (1 + 0.5 * len(confirms))
    if not np.isfinite(yen_z):
        level, label = "na", "⚪ No yen data — hit refresh"
    elif yen_z >= 1.75 and len(confirms) >= 2:
        level, label = "unwind", "🔴 UNWIND SIGNATURE — yen surging with risk draining in sync"
    elif yen_z >= 1.0:
        level, label = "stirring", "🟡 Yen stirring — funding leg tightening, watch for risk confirmation"
    elif yen_z <= -1.0:
        level, label = "carry_on", "🟢 Yen weakening — carry trade being ADDED, a tailwind for risk"
    else:
        level, label = "calm", "🟢 Carry calm — yen quiet, leveraged longs comfortable"
    return dict(level=level, label=label, stress=round(stress, 2),
                yen_z=round(yen_z, 2) if np.isfinite(yen_z) else None,
                yen_t63=yen_t, nikkei_z=round(nik_z, 2) if np.isfinite(nik_z) else None,
                nikkei_t63=nik_t, confirms=confirms,
                qqq_z=round(qqq_z, 2) if np.isfinite(qqq_z) else None,
                btc_z=round(btc_z, 2) if np.isfinite(btc_z) else None,
                vix_z=round(vix_z, 2) if np.isfinite(vix_z) else None)


def forecast_scan(node_closes: pd.DataFrame, F, R, pressure_gauge=None,
                  regime_override: str | None = None,
                  shortlist: int = 120, top: int = 20,
                  min_n: int = 300) -> tuple:
    """🔮 Best-odds preset over a cascade SHORTLIST: take the top `shortlist`
    cascade names, forecast each, and rerank by ODDS OF GAIN.

    Uses the same vectorized analog matcher as forecast_all (restricted to the
    shortlist), so a stock's odds are identical whether you scan the whole
    universe or just the shortlist — no second code path to drift.
    """
    base, regime = mega_scan(node_closes, pressure_gauge=pressure_gauge,
                             top=shortlist, regime_override=regime_override)
    if base.empty:
        return base, regime
    odds = forecast_all(min_n=min_n, only_tickers=list(base.Ticker))
    if odds.empty:
        return odds, regime
    cat = dict(zip(base.Ticker, base.get("Catalysts", pd.Series(dtype=object))))
    odds = odds.copy()
    odds["Catalysts"] = [cat.get(t, "") for t in odds.Ticker]
    odds = odds.sort_values(["OddsUp", "Typical"], ascending=False).head(top)
    return odds.reset_index(drop=True), regime


def _now_features_all():
    """Today's analog features for EVERY dump stock at once — the vectorized
    twin of _now_features. Returns (feat matrix Nx4, tradeable mask)."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    C, V = panel["c"], np.nan_to_num(panel["v"])
    t = C.shape[0] - 1
    mom = C[t - 5] / C[t - 63] - 1.0
    mom_pct = pd.Series(mom).rank(pct=True).values
    lo = np.nanmin(panel["l"][t - 62:t + 1], 0)
    hi = np.nanmax(panel["h"][t - 62:t + 1], 0)
    rangepos = (C[t] - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
    rvbase = V[t - 62:t + 1].mean(0)
    rvol = V[t - 4:t + 1].mean(0) / np.where(rvbase == 0, np.nan, rvbase)
    above = (C[t] > np.nanmean(C[t - 49:t + 1], 0)).astype(np.float32)
    # NOTE: rvol is stored UNCAPPED here. forecast_all caps both sides at 3 when
    # matching (same as outcome_forecast); pre-capping only one side made the
    # universe scan select different analogs than the per-ticker Lookup.
    feats = np.column_stack([mom_pct, rangepos, rvol, above]).astype(np.float32)
    # same stale-price guard as mega_scan: the panel forward-fills gaps, so a
    # name whose last real print was days ago must not be treated as current
    _raw_recent = _recent_ok_mask(panel)
    tradeable = (np.isfinite(C[t]) & (C[t] >= 3)
                 & np.isfinite(mom_pct) & np.isfinite(rangepos) & np.isfinite(rvol)
                 & (mdv >= 2e6) & _raw_recent)
    return feats, tradeable


def forecast_all(min_n: int = 300, price_floor: float = 3.0,
                 mdv_floor: float = 2e6, regime: str | None = None,
                 only_tickers: list | None = None):
    """Odds-of-gain forecast for the ENTIRE tradeable universe in one
    vectorized sweep — no per-ticker Python calls, no shortlist. Mirrors
    outcome_forecast's analog math exactly (same tolerances, same widen
    ladder, same 300-case floor) but matches every stock against the
    library `F` in a tight numpy loop over pre-sorted momentum bins, so it
    scales to all ~5,700 names in a few seconds.

    When `regime` is set, the SAME sector playbook the Cascade scan uses is
    applied: the odds-of-gain ranking is nudged by that regime's sector tilt
    so a scenario's favored sectors surface and its out-of-favor sectors sink
    — keeping the scenario card's promise consistent with the results.
    Returns a DataFrame with OddsUp / Typical / PopOdds / Worst10 / Best10 /
    Cases per stock (plus a hidden RankScore when a regime is applied)."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    F, R = _feature_panels()
    feats, tradeable = _now_features_all()
    if price_floor != 3.0 or mdv_floor != 2e6:
        C = panel["c"]
        _raw_recent = _recent_ok_mask(panel)
        tradeable = (np.isfinite(C[-1]) & (C[-1] >= price_floor)
                     & np.isfinite(feats[:, 0]) & np.isfinite(feats[:, 1])
                     & (mdv >= mdv_floor) & _raw_recent)
    if only_tickers:
        want = {str(t).strip().upper() for t in only_tickers}
        tradeable = tradeable & np.array([str(t) in want for t in tickers])
    idx = np.where(tradeable)[0]

    # sort the library by momentum percentile so each stock scans only the
    # slice within the widest momentum tolerance (0.10 * 2.4 = 0.24) instead
    # of all ~150k rows — turns an O(N*L) sweep into O(N*window)
    order = np.argsort(F[:, 0])
    Fs = F[order]; Rs = R[order]
    fmom = Fs[:, 0]
    px = panel["c"][-1]

    tol = np.array([0.10, 0.15, 0.50, 0.0])
    rows = []
    for j in idx:
        now = feats[j]
        has_rvol = np.isfinite(now[2])
        # mirror outcome_forecast EXACTLY: widen until >=250 matches; if none
        # of the three levels reaches 250, keep the LAST (widest) sample — the
        # binning only restricts WHICH library rows are candidates, never how
        # the match itself is computed. The momentum window at widen 2.4 spans
        # +/-0.24, which always contains the full |mom_pct diff|<=0.24 set the
        # original scans, so the two select identical rows.
        sel = None
        for widen in (1.0, 1.6, 2.4):
            mtol = tol[0] * widen
            a = np.searchsorted(fmom, now[0] - mtol, "left")
            b = np.searchsorted(fmom, now[0] + mtol, "right")
            win_R = Rs[a:b]; win_F = Fs[a:b]
            m = ((np.abs(win_F[:, 0] - now[0]) <= tol[0] * widen)
                 & (np.abs(win_F[:, 1] - now[1]) <= tol[1] * widen)
                 & (win_F[:, 3] == now[3]))
            if has_rvol:
                m &= (np.abs(np.minimum(win_F[:, 2], 3) - min(now[2], 3))
                  <= tol[2] * widen)
            sel = win_R[m]                       # always keep the current sample
            if m.sum() >= 250:                   # ...but stop once we have enough
                break
        if sel is None or len(sel) < min_n:
            continue
        f21 = sel[:, 1]
        rows.append((tickers[j], sectors[j], round(float(px[j]), 2),
                     round(float((f21 > 0).mean()) * 100, 0),
                     round(float(np.median(f21)) * 100, 1),
                     round(float((f21 >= 0.15).mean()) * 100, 0),
                     round(float(np.quantile(f21, 0.10)) * 100, 0),
                     round(float(np.quantile(f21, 0.90)) * 100, 0),
                     int(len(sel))))
    cols = ["Ticker", "Sector", "Price", "OddsUp", "Typical",
            "PopOdds", "Worst10", "Best10", "Cases"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df
    if regime and regime in SECTOR_TILTS and SECTOR_TILTS[regime]:
        tilts = SECTOR_TILTS[regime]
        # nudge each stock's odds by its sector's regime multiplier, centered
        # on 1.0 so favored sectors rise and out-of-favor ones fall. A ±25%
        # tilt maps to roughly ±5 points of effective odds — enough to steer
        # the leaderboard toward the regime WITHOUT inventing odds the analog
        # history doesn't support. The displayed OddsUp stays the true number.
        mult = df.Sector.map(lambda s: tilts.get(s, 1.0)).astype(float)
        df["RankScore"] = (df.OddsUp * mult).round(2)
        df = df.sort_values(["RankScore", "Typical"], ascending=False)

        # Spread the head across sectors. Sorting on RankScore alone hands the
        # entire top of the book to whichever sector has the biggest multiplier
        # (Debasement -> 20/20 Basic Materials), which is one concentrated bet
        # wearing a diversified label. Interleave instead: rotate through the
        # sectors best-first, giving favoured sectors extra picks per rotation
        # in proportion to their tilt, so ANY prefix of the list is diversified.
        by_sec, order_secs = {}, []
        for s, grp in df.groupby("Sector", sort=False):
            by_sec[s] = grp.index.tolist()
        order_secs = sorted(by_sec, key=lambda s: -df.loc[by_sec[s][0], "RankScore"])
        picks, guard = [], 0
        while any(by_sec.values()) and guard < 10000:
            guard += 1
            for s in order_secs:
                if not by_sec[s]:
                    continue
                # favoured sectors take 2 slots per rotation, others 1
                n = 2 if tilts.get(s, 1.0) >= 1.15 else 1
                for _ in range(n):
                    if by_sec[s]:
                        picks.append(by_sec[s].pop(0))
        df = df.loc[picks]
        return df.reset_index(drop=True)
    return df.sort_values(["OddsUp", "Typical"], ascending=False).reset_index(drop=True)


# ═════════ LIVE UPDATE — refresh a result list against the market now ═════════
def yahoo_prices(tickers: list) -> dict:
    """Batch last-price fetch from Yahoo — the gap-filler for whatever
    Alpaca doesn't cover (indices, ADRs, thin names, crypto)."""
    os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")
    out = {}
    if not tickers:
        return out
    try:
        import yfinance as yf
    except Exception:
        return out
    try:
        data = yf.download(list(tickers), period="5d", interval="1d",
                           progress=False, group_by="ticker", threads=True,
                           auto_adjust=False)
    except Exception:
        return out
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if t not in data.columns.get_level_values(0):
                    continue
                s = data[t]["Close"].dropna()
            else:
                s = data["Close"].dropna()
            if len(s):
                out[t] = float(s.iloc[-1])
        except Exception:
            continue
    return out


def live_update(df: pd.DataFrame, price_col: str = "Price") -> tuple:
    """Refresh ANY Top-20 result frame against live market data.

    Chain: Alpaca batch snapshots first, then Yahoo for whatever's missing —
    the same priority the Stock Lookup analyzer uses. Recomputes the columns
    that actually move intraday (price, and the % change since the scan's
    close) and reports where each price came from.

    Returns (updated_df, meta) where meta has counts per source.
    """
    if df is None or df.empty or "Ticker" not in df.columns:
        return df, dict(alpaca=0, yahoo=0, missing=0, total=0)
    tickers = [str(t) for t in df["Ticker"].tolist()]

    # 1. Alpaca first (live snapshots), tracking the source explicitly
    live, source = {}, {}
    try:
        for k, v in alpaca_prices(tickers).items():
            if v and np.isfinite(v):
                live[k] = float(v); source[k] = "Alpaca"
    except Exception:
        pass
    n_alpaca = len(live)

    # 2. Yahoo fills whatever Alpaca didn't return
    missing = [t for t in tickers if t not in live]
    if missing:
        try:
            for k, v in yahoo_prices(missing).items():
                if v and np.isfinite(v):
                    live[k] = float(v); source[k] = "Yahoo"
        except Exception:
            pass
    n_yahoo = len(live) - n_alpaca

    out = df.copy()
    prev = pd.to_numeric(out[price_col], errors="coerce") if price_col in out else None
    new_px = np.array([live.get(t, np.nan) for t in tickers], dtype=float)
    out["Live"] = np.round(new_px, 2)
    if prev is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["Chg%"] = np.round((new_px / prev.values - 1.0) * 100, 2)
    out["Src"] = [source.get(t, "—") for t in tickers]
    meta = dict(alpaca=n_alpaca, yahoo=n_yahoo,
                missing=len(tickers) - len(live), total=len(tickers),
                stamp=pd.Timestamp.now(tz="US/Eastern").strftime("%b %d %I:%M %p ET"))
    return out, meta


# ═════════ 🧭 MACRO ADVISOR — evidence-weighted scenario recommendation ═════════
MACRO_NEWS_PROXIES = ["SPY", "USO", "GLD", "UUP", "TLT", "XLE", "XLK", "HYG"]

# headline buckets → which regime they argue for
MACRO_NEWS_BUCKETS = {
    "stag": (["inflation", "cpi", "ppi", "price pressure", "oil surge", "opec",
              "supply shock", "wage growth", "sticky inflation", "energy crisis",
              "tariff", "commodity rally"], "inflation/supply pressure in the news"),
    "bear": (["selloff", "crash", "plunge", "recession", "credit event",
              "default", "bank failure", "contagion", "war", "escalation",
              "risk-off", "slump", "layoffs", "bankruptcy"],
             "risk-off / stress language in the news"),
    "qe":   (["rate cut", "easing", "dovish", "stimulus", "quantitative easing",
              "liquidity injection", "balance sheet expansion", "pivot"],
             "easing / liquidity language in the news"),
    "bull": (["rally", "record high", "soft landing", "beat estimates",
              "strong earnings", "melt-up", "risk-on", "optimism", "goldilocks"],
             "risk-on / growth language in the news"),
    "strong": (["dollar strength", "strong dollar", "dxy", "hawkish",
                "rate hike", "tightening", "yields surge"],
               "strong-dollar / hawkish language in the news"),
    "repress": (["money printing", "debt monetization", "debasement",
                 "treasury buyback", "debt buyback", "liquidity support",
                 "yield curve control", "financial repression", "balance sheet",
                 "gold record", "gold all-time high", "devaluation",
                 "printing press", "monetize the debt"],
                "debt-monetisation / debasement language in the news"),
}


def macro_news_scan(max_articles: int = 10) -> dict:
    """Keyword-scan headlines on macro proxies. Returns {regime: (hits, note)}."""
    os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")
    out, blob = {}, ""
    try:
        import yfinance as yf
    except Exception:
        return out
    seen = 0
    for tk in MACRO_NEWS_PROXIES:
        try:
            arts = yf.Ticker(tk).news or []
        except Exception:
            continue
        for a in arts[:max_articles]:
            c = a.get("content") or a
            blob += " " + str(c.get("title", "")) + " " + str(c.get("summary", ""))
            seen += 1
    if not seen:
        return out
    blob = blob.lower()
    for reg, (words, note) in MACRO_NEWS_BUCKETS.items():
        hits = sum(blob.count(w) for w in words)
        if hits:
            out[reg] = (hits, note)
    out["_articles"] = seen
    return out


def macro_advisor(node_closes: pd.DataFrame, pressure: dict | None = None,
                  include_news: bool = True) -> dict:
    """Weigh EVERY macro signal the app produces and recommend a scenario.

    Sources: the pressure gauge (Fed liquidity, credit spreads, stablecoins),
    node impulses (oil/dollar/gold/VIX/rates), the ratio sentinels
    (risk appetite, credit, breadth, growth-vs-fear), the yen-carry monitor,
    and a keyword scan of macro headlines.

    Every piece of evidence votes for one regime with a weight; the winner is
    recommended and the full ballot is returned so the call is auditable rather
    than a black box.
    """
    scores = {k: 0.0 for k in REGIME_LABELS}
    evidence = []

    def vote(regime, weight, signal, reading):
        if regime in scores:
            scores[regime] += weight
        evidence.append(dict(signal=signal, reading=reading,
                             regime=regime, weight=weight))

    imp = impulses(node_closes).iloc[-1] if node_closes is not None else pd.Series(dtype=float)

    def z(sym):
        v = imp.get(sym, np.nan)
        return float(v) if np.isfinite(v) else np.nan

    def r63(sym):
        if node_closes is None or sym not in node_closes.columns:
            return np.nan
        s = node_closes[sym].dropna()
        return float(s.iloc[-1] / s.iloc[-64] - 1) if len(s) > 64 else np.nan

    # ── 1. liquidity / pressure gauge ────────────────────────────────
    gauge = None
    if pressure and pressure.get("gauge") is not None:
        gauge = pressure["gauge"]
        if gauge >= 2:
            vote("qe", 3.0, "Pressure gauge", f"+{gauge} — liquidity expanding")
        elif gauge <= -2:
            vote("bear", 2.5, "Pressure gauge", f"{gauge} — liquidity draining")
        elif gauge >= 1:
            vote("qe", 1.0, "Pressure gauge", f"+{gauge} — mildly supportive")
        elif gauge <= -1:
            vote("bear", 1.0, "Pressure gauge", f"{gauge} — mildly tight")
        comp = (pressure or {}).get("components", {})
        hy = comp.get("HY Δ 21d (bp)")
        if hy is not None:
            if hy > 25:
                vote("bear", 2.0, "HY credit spread", f"+{hy:.0f}bp/21d — credit stress building")
            elif hy < -25:
                vote("bull", 1.5, "HY credit spread", f"{hy:.0f}bp/21d — credit healing")

    # ── 2. commodities / inflation ───────────────────────────────────
    oil = r63("USO")
    if np.isfinite(oil):
        if oil > 0.15:
            vote("stag", 3.0, "Oil (USO)", f"{oil:+.0%} over 63d — cost-push pressure")
        elif oil < -0.05:
            vote("bull", 1.5, "Oil (USO)", f"{oil:+.0%} over 63d — input costs easing")
    gold = r63("GLD")
    if np.isfinite(gold) and gold > 0.10:
        vote("repress", 2.5, "Gold (GLD)", f"{gold:+.0%} over 63d — debasement bid")
        vote("stag", 1.0, "Gold (GLD)", f"{gold:+.0%} over 63d — haven/inflation bid")
    if (gauge is not None and gauge >= 1 and np.isfinite(gold) and gold >= 0.08
            and np.isfinite(uup := r63("UUP")) and uup <= 0.02):
        vote("repress", 3.0, "Liquidity + gold + soft dollar",
             "expanding liquidity with gold leading — printing without a strong dollar")

    # ── 3. dollar ────────────────────────────────────────────────────
    uup = r63("UUP")
    if np.isfinite(uup):
        if uup > 0.04:
            vote("strong", 3.0, "US Dollar (UUP)", f"{uup:+.0%} over 63d — squeezing global earners")
        elif uup < -0.03:
            vote("qe", 1.5, "US Dollar (UUP)", f"{uup:+.0%} over 63d — weak dollar, liquidity friendly")

    # ── 4. volatility / shock ────────────────────────────────────────
    vix = z("^VIX")
    if np.isfinite(vix):
        if vix >= 1.25:
            vote("bear", 3.5, "VIX impulse", f"z {vix:+.1f} — volatility shock firing")
        elif vix <= -0.5:
            vote("bull", 1.5, "VIX impulse", f"z {vix:+.1f} — fear draining")

    # ── 5. ratio sentinels: the relationships that lead ──────────────
    try:
        rs = ratio_sentinel_impulses(node_closes)
    except Exception:
        rs = pd.DataFrame()
    if not rs.empty:
        m = {r["pair"]: r for _, r in rs.iterrows()}
        def trend(pair):
            r = m.get(pair)
            return float(r["trend63"]) if r is not None and np.isfinite(r.get("trend63", np.nan)) else np.nan
        t = trend("XLY/XLP")
        if np.isfinite(t):
            if t > 0.03:
                vote("bull", 2.0, "Discretionary vs Staples", f"{t:+.1%}/63d — consumer risk appetite on")
            elif t < -0.03:
                vote("bear", 2.0, "Discretionary vs Staples", f"{t:+.1%}/63d — defensive rotation")
        t = trend("HYG/IEF")
        if np.isfinite(t):
            if t > 0.02:
                vote("bull", 1.5, "Junk vs Treasuries", f"{t:+.1%}/63d — credit appetite healthy")
            elif t < -0.02:
                vote("bear", 2.5, "Junk vs Treasuries", f"{t:+.1%}/63d — credit turning away")
        t = trend("CPER/GLD")
        if np.isfinite(t):
            if t > 0.03:
                vote("bull", 1.5, "Copper vs Gold", f"{t:+.1%}/63d — growth over fear")
            elif t < -0.03:
                vote("stag", 1.5, "Copper vs Gold", f"{t:+.1%}/63d — fear over growth")
        t = trend("RSP/SPY")
        if np.isfinite(t) and t < -0.03:
            vote("bear", 1.0, "Equal vs Cap Weight", f"{t:+.1%}/63d — narrow, fragile rally")
        t = trend("EEM/SPY")
        if np.isfinite(t) and t < -0.03:
            vote("strong", 1.5, "EM vs US", f"{t:+.1%}/63d — dollar pulling money home")
        t = trend("SMH/SPY")
        if np.isfinite(t):
            if t > 0.03:
                vote("bull", 1.5, "Semis vs Market", f"{t:+.1%}/63d — cycle leader leading")
            elif t < -0.05:
                vote("bear", 1.5, "Semis vs Market", f"{t:+.1%}/63d — cycle leader rolling over")

    # ── 6. yen carry ─────────────────────────────────────────────────
    try:
        cm = yen_carry_monitor(node_closes)
    except Exception:
        cm = {}
    if cm.get("level") == "unwind":
        vote("bear", 4.0, "Yen carry", cm["label"])
    elif cm.get("level") == "stirring":
        vote("bear", 1.0, "Yen carry", "yen stirring — funding leg tightening")
    elif cm.get("level") == "carry_on":
        vote("qe", 1.0, "Yen carry", "yen weakening — fresh carry funding risk assets")

    # ── 7. rates ─────────────────────────────────────────────────────
    tlt = r63("TLT")
    if np.isfinite(tlt):
        if tlt < -0.05:
            vote("stag", 1.5, "Long bonds (TLT)", f"{tlt:+.0%}/63d — yields up, duration punished")
            vote("strong", 0.5, "Long bonds (TLT)", f"{tlt:+.0%}/63d")
        elif tlt > 0.05:
            vote("qe", 1.0, "Long bonds (TLT)", f"{tlt:+.0%}/63d — yields falling")

    # ── 8. news ──────────────────────────────────────────────────────
    news = macro_news_scan() if include_news else {}
    n_articles = news.pop("_articles", 0) if news else 0
    for reg, (hits, note) in (news or {}).items():
        w = min(2.0, 0.4 * hits)          # capped: headlines confirm, never decide
        vote(reg, w, "Headlines", f"{hits} hits — {note}")

    # ── verdict ──────────────────────────────────────────────────────
    total = sum(scores.values())
    if total <= 0:
        best, margin, conf = "base", 0.0, "low"
    else:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best = ranked[0][0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = ranked[0][1] - runner
        share = ranked[0][1] / total
        if margin >= 4 and share >= 0.40:
            conf = "high"
        elif margin >= 2:
            conf = "medium"
        else:
            conf = "low"
        if ranked[0][1] < 3:              # nothing is really firing
            best, conf = "base", "low"
    evidence.sort(key=lambda e: -e["weight"])
    return dict(recommended=best, label=REGIME_LABELS[best], confidence=conf,
                scores=scores, evidence=evidence, margin=round(margin, 1),
                n_articles=n_articles, gauge=gauge,
                detected=macro_regime(node_closes, gauge).get("regime"),
                stamp=pd.Timestamp.now(tz="US/Eastern").strftime("%b %d %I:%M %p ET"))


def macro_only_scan(regime: str, top: int = 20, strict: bool = True) -> tuple:
    """🎯 Macro-only: rank purely on SCENARIO FIT and business quality — no
    cascade flow, no technicals, no analog odds.

    Answers "if this macro world is the one we're in, which quality companies
    are positioned for it?" Score = the regime's sector multiplier x a quality
    composite, with the same unknown-is-not-average rule used everywhere else
    (a missing fundamental ranks at the bottom, never in the middle).
    Diversified across the regime's LEADING sectors so one favoured sector
    can't take the whole book.
    """
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    funds = dump_fundamentals_all()
    C = panel["c"]
    N = C.shape[1]
    px = C[-1]
    _raw_recent = _recent_ok_mask(panel)
    tradeable = np.isfinite(px) & (px >= 3.0) & (mdv >= 2e6) & _raw_recent

    roic = funds["ROIC"]; pio = funds["Piotroski"]; oe = funds["OE_Yield"]
    pe = funds["P/E"]; rg = funds["RevenueGrowth"]; eg = funds["EarningsGrowth"]
    rt = funds["ROIC_Trend"]
    if strict:
        qual_ok = (np.isfinite(roic) & (roic > 0) & np.isfinite(pio) & (pio >= 4)
                   & np.isfinite(pe) & (pe > 0) & (pe <= 60))
    else:
        qual_ok = np.isfinite(pio) & (pio >= 3)
    eligible = tradeable & qual_ok

    def _pct(a):
        s = pd.Series(np.where(eligible & np.isfinite(a), a, np.nan))
        return np.nan_to_num(s.rank(pct=True).values, nan=0.0)

    quality = (0.35 * _pct(roic) + 0.25 * _pct(oe)
               + 0.20 * np.nan_to_num(np.clip(pio / 9.0, 0, 1), nan=0.0)
               + 0.10 * _pct(rt) + 0.05 * _pct(rg) + 0.05 * _pct(eg))

    tilts = SECTOR_TILTS.get(regime, {})
    sec_arr = np.array(sectors)
    mult = np.array([tilts.get(s, 1.0) for s in sec_arr])
    score = np.where(eligible, quality * 100 * mult, -np.inf)

    # diversify across the regime's leading sectors, capped like mega_scan
    leads = {s: t for s, t in tilts.items() if t >= 1.0}
    cap = max(2, int(np.ceil(top * 0.35)))
    chosen, picked = [], np.zeros(N, dtype=bool)
    if leads:
        wsum = sum(leads.values())
        for s in sorted(leads, key=lambda x: -leads[x]):
            q = min(max(int(round(top * (leads[s] / wsum))), 1), cap)
            ix = np.where((sec_arr == s) & np.isfinite(score))[0]
            if not len(ix):
                continue
            ix = ix[np.argsort(-score[ix])][:q]
            chosen.extend(ix.tolist()); picked[ix] = True
            if len(chosen) >= top:
                break
    if len(chosen) < top:
        from collections import Counter
        cnt = Counter(sec_arr[chosen].tolist())
        for gi in np.argsort(-np.where(picked, -np.inf, score)):
            if len(chosen) >= top:
                break
            if picked[gi] or not np.isfinite(score[gi]):
                continue
            if cnt[sec_arr[gi]] >= cap:
                continue
            chosen.append(int(gi)); picked[gi] = True
            cnt[sec_arr[gi]] += 1
    chosen = chosen[:top]
    order = (np.array(chosen)[np.argsort(-score[chosen])] if chosen
             else np.array([], dtype=int))

    have = (np.isfinite(roic).astype(int) + np.isfinite(oe).astype(int)
            + np.isfinite(pio).astype(int) + np.isfinite(rt).astype(int)
            + np.isfinite(rg).astype(int) + np.isfinite(eg).astype(int))
    df = pd.DataFrame({
        "Ticker": tickers[order], "Sector": sec_arr[order],
        "Price": np.round(px[order], 2),
        "Fit": np.round(score[order], 1),
        "MacroFit": np.round(mult[order], 2),
        "Quality": np.round(quality[order] * 100, 0),
        "ROIC": roic[order], "OE Yield": oe[order],
        "Piotroski": pio[order], "P/E": np.round(pe[order], 1),
        "RevGrowth": rg[order],
        "Data": [f"{h}/6" for h in have[order]],
    }).reset_index(drop=True)
    meta = dict(regime=regime, label=REGIME_LABELS.get(regime, regime),
                eligible=int(eligible.sum()), n_sectors=int(df.Sector.nunique())
                if not df.empty else 0)
    return df, meta


# ═════════ 🔥 SECTOR FLOW — where the money went in the last session ═════════
# Walk-forward validated on the nightly dump (32 windows, both honesty halves):
#   baseline tech-only            -0.02% excess / 21d, 47% hit
#   hot top-5 filter + 0.20 tilt  +3.41% excess / 21d, 60% hit
#                                 H1 +3.42% / H2 +3.41%  (no regime-fit)
# Falsification: the COLDEST sectors underperform (-0.22%), so the signal is
# directional rather than an artifact of filtering to fewer names.
# 1-day beat 3/5/10-day lookbacks and was the only window positive in BOTH
# halves — money rotation shows up fast and decays.




def sector_flow(lookback: int = SECTOR_FLOW_LOOKBACK,
                offset: int = 0,
                use_live: bool = False) -> pd.DataFrame:
    """Which sectors received money over the last `lookback` session(s).

    Price alone is a poor proxy for flow — one mega-cap can carry a sector.
    The score blends four things:
      • dollar-weighted return   (money-weighted, not name-weighted)
      • relative strength        (sector return minus the market's)
      • breadth                  (share of names in the sector that rose)
      • dollar-volume surge      (is turnover actually elevated?)

    `lookback` is the width of the window in sessions (1 = a single day,
    5 = the combined move over five sessions). `offset` slides the window
    back in time: offset=0 ends on the most recent session, offset=3 ends
    three sessions ago. Together they let you read one specific past day or
    any combined range up to SECTOR_FLOW_MAX_BACK sessions.

    With use_live=True the last close is replaced by a live Alpaca snapshot,
    so an intraday session counts as "the last 24 hours". Live is ignored
    when offset>0, because a historical window has no live price.
    """
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    C, V = panel["c"], np.nan_to_num(panel["v"])
    T = C.shape[0]
    lb = max(1, int(lookback))
    off = max(0, min(int(offset), SECTOR_FLOW_MAX_BACK))
    end = T - 1 - off                       # index of the window's last bar
    start = end - lb                        # bar the window is measured from
    if start < 0:
        off = 0; end = T - 1; start = max(0, end - lb)
    px = C[end].astype(float).copy()
    live_n = 0
    if use_live and off == 0:
        try:
            live = alpaca_prices(list(tickers))
            if live:
                for i, t in enumerate(tickers):
                    v = live.get(str(t))
                    if v and np.isfinite(v):
                        px[i] = float(v); live_n += 1
        except Exception:
            pass

    prev = C[start]
    r = px / np.where(prev == 0, np.nan, prev) - 1.0
    dvol = C * V
    dv_now = np.nanmean(dvol[start + 1:end + 1], axis=0)
    dv_base = np.nanmean(dvol[max(0, end - 62):end + 1], axis=0)
    surge = dv_now / np.where(dv_base == 0, np.nan, dv_base)
    ok = (np.isfinite(r) & np.isfinite(px) & (px >= 3.0) & (mdv >= 2e6)
          & _recent_ok_mask(panel))
    if not ok.any():
        return pd.DataFrame()
    mkt = float(np.nanmean(np.where(ok, r, np.nan)))

    sec_arr = np.array([str(s) for s in sectors])
    rows = []
    for s in sorted(set(sec_arr)):
        # "Unknown" is a grab-bag of unclassified tickers, not a sector. Left in,
        # it topped the table on a 31% move from 5 illiquid names and stole a
        # slot in the hot list. Real sectors need enough names to average out.
        if s.strip().lower() in ("unknown", "", "n/a", "none"):
            continue
        m = ok & (sec_arr == s)
        n = int(m.sum())
        if n < 15:                      # too thin to call a "sector flow"
            continue
        w = np.nan_to_num(dv_now[m])
        ret = float(np.nansum(r[m] * (w / w.sum()))) if w.sum() > 0 else float(np.nanmean(r[m]))
        breadth = float(np.nanmean(r[m] > 0))
        sg = float(np.nanmedian(surge[m])) if np.isfinite(surge[m]).any() else 1.0
        flow = (0.40 * (ret - mkt) + 0.25 * (breadth - 0.5) * 0.10
                + 0.20 * np.tanh(sg - 1.0) * 0.02 + 0.15 * ret)
        rows.append(dict(Sector=s, Flow=flow, Ret=ret, RS=ret - mkt,
                         Breadth=breadth, VolSurge=sg, Names=n))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Flow", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    df["FlowPct"] = df.Flow.rank(pct=True)
    df.attrs["market_return"] = mkt
    df.attrs["live_prices"] = live_n
    df.attrs["asof"] = str(pd.to_datetime(dts[end]).date())
    df.attrs["window_start"] = str(pd.to_datetime(dts[start + 1]).date())
    df.attrs["window_end"] = str(pd.to_datetime(dts[end]).date())
    df.attrs["lookback"] = lb
    df.attrs["offset"] = off
    return df


def flow_sessions(n: int = SECTOR_FLOW_MAX_BACK) -> list:
    """The last n trading dates in the dump, newest first — for the day picker."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    d = [str(pd.to_datetime(x).date()) for x in dts[-int(n):]]
    return list(reversed(d))


def hot_sectors(k: int = 5, lookback: int = SECTOR_FLOW_LOOKBACK,
                offset: int = 0, use_live: bool = False) -> list:
    """The k sectors receiving the most money over the chosen window."""
    df = sector_flow(lookback, offset, use_live)
    return [] if df.empty else [str(s) for s in df.Sector.head(int(k))]
