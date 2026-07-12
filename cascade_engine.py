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


def forced_flows(today: date | None = None, days_ahead: int = 45,
                 closes: pd.DataFrame | None = None) -> pd.DataFrame:
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
            mtd = closes.loc[closes.index.month == closes.index[-1].month,
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
           "🛒 Buy: patience. Hold off NEW entries during OpEx week (pinned, "
           "choppy); place planned buys the Monday-Tuesday AFTER expiry when "
           "the pin releases — dips then are mechanical, not fundamental.")
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
               "🛒 Buy: the announced ADD tickers at the announcement (small "
               "size — decayed edge), sell into the rebalance close. Check "
               "spglobal.com press releases ~5-10 days before this date.")
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
           "🛒 " + _pension_buy)
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
               "🛒 Buy: post-earnings DIPS in the heaviest repurchasers — "
               "AAPL, GOOGL, META, NVDA, MSFT, JPM, XOM — 1-2 days after each "
               "reports. Broad: QQQ/SPY. Pure-play: PKW (Buyback Achievers ETF).")
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
           "🛒 Buy: your December loser list (equal-weight basket of 10+, "
           "never one name), or simply IWM, in the final week of Dec. Exit "
           "mid-January. Small size — decayed seasonal.")

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
