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
from datetime import date, datetime, timedelta

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
}

# canonical upstream sentinels (fast, frictionless)
SENTINELS = ["BTC-USD", "ETH-USD", "FXY", "CPER", "GLD", "SMH", "HYG", "^VIX"]

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
        # refetch automatically when the cache goes stale (>4 calendar days)
        if (pd.Timestamp.today() - df.index[-1]).days <= 4:
            return df
        try:
            os.remove(LOCAL_HISTORY)
        except OSError:
            return df
    os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")
    import yfinance as yf
    start = (date.today() - timedelta(days=int(years * 365.25 + 30))).isoformat()
    raw = yf.download(list(NODES), start=start, auto_adjust=True,
                      progress=False, group_by="column")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    closes = closes.dropna(how="all").ffill(limit=5)
    try:
        closes.to_parquet(LOCAL_HISTORY)
    except Exception:
        pass
    return closes


def refresh_history():
    try:
        os.remove(LOCAL_HISTORY)
    except FileNotFoundError:
        pass
    return fetch_history()


# ── impulse + graph ──────────────────────────────────────────────────
def impulses(closes: pd.DataFrame) -> pd.DataFrame:
    """Cross-time z-score of the 5d return for every node — 'flow impulse'."""
    r = closes.pct_change(IMPULSE_W)
    mu = r.rolling(IMPULSE_Z_WIN, min_periods=60).mean()
    sd = r.rolling(IMPULSE_Z_WIN, min_periods=60).std()
    return (r - mu) / sd


def _rank(a):
    return pd.Series(a).rank().values


def estimate_edges(closes: pd.DataFrame, asof: int | None = None,
                   train: int = EDGE_TRAIN, horizon: int = EDGE_HORIZON,
                   min_ic: float = EDGE_MIN_ABS_IC) -> pd.DataFrame:
    """Directed lead-lag edges i -> j estimated on data up to `asof` (iloc).
    Edge = Spearman IC between impulse_i(t) and fwd-return_j(t+1..t+horizon).
    Walk-forward safe: only rows <= asof are used."""
    imp = impulses(closes)
    fwd = closes.shift(-horizon) / closes - 1.0
    T = len(closes)
    end = T - 1 if asof is None else asof
    lo = max(0, end - train)
    # forward returns must be fully realised inside the training window
    ii = imp.iloc[lo:end - horizon]
    ff = fwd.iloc[lo:end - horizon]
    cols = [c for c in closes.columns if ii[c].notna().sum() > 60]
    edges = []
    ranks_i = {c: None for c in cols}
    for i in cols:
        xi = ii[i]
        for j in cols:
            if i == j:
                continue
            yj = ff[j]
            m = xi.notna() & yj.notna()
            if m.sum() < 60:
                continue
            x, y = xi[m].values, yj[m].values
            ic = np.corrcoef(_rank(x), _rank(y))[0, 1]
            if np.isfinite(ic) and abs(ic) >= min_ic:
                edges.append((i, j, round(float(ic), 3)))
    df = pd.DataFrame(edges, columns=["source", "target", "ic"])
    if df.empty:
        return df
    # hit rate of the directional call on the same window
    hits = []
    for _, e in df.iterrows():
        xi, yj = ii[e.source], ff[e.target]
        m = xi.notna() & yj.notna()
        x, y = xi[m].values, yj[m].values
        strong = np.abs(x) >= WAVE_Z
        if strong.sum() >= 8:
            pred = np.sign(x[strong]) * np.sign(e.ic)
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
        edges = estimate_edges(closes, asof=t, train=train, horizon=horizon)
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
    # gauge: rising liquidity + rising stables + tightening spreads = risk-on
    score = 0
    c = out["components"]
    score += 1 if c.get("NetLiq Δ 21d ($bn)", 0) > 0 else -1
    score += 1 if c.get("Stables Δ 21d ($bn)", 0) > 0 else -1
    score += 1 if c.get("HY Δ 21d (bp)", 0) < 0 else -1
    out["gauge"] = score          # -3 .. +3
    out["gauge_label"] = {3: "🟢 High pressure — liquidity building",
                          1: "🟢 Mildly supportive",
                          -1: "🟡 Mixed / draining",
                          -3: "🔴 Draining — waves unlikely to travel far"}.get(score, "🟡 Mixed")
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


def forced_flows(today: date | None = None, days_ahead: int = 45) -> pd.DataFrame:
    """Mechanical, scheduled flows in the next `days_ahead` days."""
    today = today or date.today()
    horizon = today + timedelta(days=days_ahead)
    events = []
    for k in range(3):
        m = (today.month - 1 + k) % 12 + 1
        y = today.year + (today.month - 1 + k) // 12
        opex = _third_friday(y, m)
        events.append((opex, "Options expiration (OpEx)",
                       "Dealer hedging unwinds pin prices into OpEx; volatility often expands the week after."))
        if m in (3, 6, 9, 12):
            events.append((opex, "S&P quarterly rebalance (effective)",
                           "Index funds must trade adds/deletes at the close — forced, price-insensitive flow."))
        last = (date(y, m, 28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        events.append((last, "Month-end pension rebalance window",
                       "Pensions rebalance equity/bond mix in the final 1-3 sessions; flows oppose the month's move."))
        if m in (1, 4, 7, 10):
            events.append((date(y, m, 15), "Buyback blackout lifts (approx.)",
                           "As earnings pass, corporate buybacks — the largest single equity buyer — switch back on."))
    if today.month <= 6:
        events.append((_third_friday(today.year, 6) + timedelta(days=7),
                       "Russell reconstitution (late June)",
                       "Largest forced-flow day of the year; adds/deletes see massive closing auctions."))
    if today.month >= 11 or today.month == 12:
        events.append((date(today.year, 12, 15), "Tax-loss selling peak window",
                       "Losers get sold for tax harvesting into mid-December…"))
        events.append((date(today.year + (1 if today.month == 12 else 0), 1, 5),
                       "January reversal window",
                       "…and the beaten-down names historically bounce in early January."))
    df = pd.DataFrame([(d, n, note) for d, n, note in events
                       if today <= d <= horizon],
                      columns=["Date", "Event", "Why it moves money"])
    return df.sort_values("Date", ignore_index=True)
