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
        # refetch when stale (>4 calendar days) — but keep the stale copy
        # unless the fresh download actually succeeds (stale beats empty)
        if (pd.Timestamp.today() - df.index[-1]).days <= 4:
            return df
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
    closes = closes.dropna(how="all").ffill(limit=5)
    if len(closes) < 30 or closes.shape[1] < 3:      # download failed
        if stale_df is not None:
            LAST_HISTORY_SOURCE = "stale cache (feeds unreachable — will retry)"
            return stale_df
        return closes
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
LOCAL_DUMP = os.path.join(os.path.dirname(__file__), "data", "dump_panel_v2.npz")

BUYBACK_TITANS = ["AAPL", "GOOGL", "MSFT", "META", "NVDA", "JPM", "XOM"]


def load_dump_panel():
    """Full OHLCV panel for ~5,700 stocks from the nightly magicpro33/stock
    dump. Cached to disk; refetched when >4 days stale. Returns
    (panel dict[o/h/l/c/v -> (T,N) float32], tickers, sectors, mdv, dates)."""
    import gzip as _gz
    if os.path.exists(LOCAL_DUMP):
        z = np.load(LOCAL_DUMP, allow_pickle=True)
        dts = pd.to_datetime(z["dates"])
        if (pd.Timestamp.today() - dts[-1]).days <= 4:
            panel = {f: z[f] for f in ("o", "h", "l", "c", "v")}
            return panel, z["tickers"], z["sectors"], z["mdv"], dts
    r = requests.get(DUMP_URL, timeout=120)
    r.raise_for_status()
    data = json.loads(_gz.decompress(r.content).decode())
    rows = [x for x in data if len(x.get("_hist", {}).get("dates", [])) >= 120]
    all_d = sorted({d for x in rows for d in x["_hist"]["dates"]})
    dix = {d: i for i, d in enumerate(all_d)}
    T, N = len(all_d), len(rows)
    panel = {f: np.full((T, N), np.nan, dtype=np.float32)
             for f in ("o", "h", "l", "c", "v")}
    key = dict(o="open", h="high", l="low", c="close", v="volume")
    tickers, sectors = [], []
    for j, x in enumerate(rows):
        ix = [dix[d] for d in x["_hist"]["dates"]]
        for f, kk in key.items():
            panel[f][ix, j] = x["_hist"][kk]
        tickers.append(x["Ticker"])
        sectors.append(x.get("Sector") or "Unknown")
    panel["c"] = pd.DataFrame(panel["c"]).ffill(limit=5).values.astype(np.float32)
    mdv = np.nanmedian((panel["c"] * np.nan_to_num(panel["v"]))[-21:], axis=0)
    tickers, sectors = np.array(tickers), np.array(sectors)
    np.savez_compressed(LOCAL_DUMP, tickers=tickers, sectors=sectors,
                        mdv=mdv, dates=np.array(all_d), **panel)
    return panel, tickers, sectors, mdv, pd.to_datetime(all_d)


def dump_ohlcv(ticker: str) -> pd.DataFrame:
    """Full OHLCV history for one stock from the nightly dump."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    ix = np.where(tickers == ticker)[0]
    if not len(ix):
        return pd.DataFrame()
    j = ix[0]
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


def alpaca_prices(tickers: list) -> dict:
    """Fresh prices via Alpaca batch snapshots. {} without keys/network."""
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
                "start": start, "limit": 10000, "adjustment": "split",
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
    dump — the analog library. Sampled every 3 sessions after warmup."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
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


def outcome_forecast(ticker: str, F=None, R=None) -> dict:
    """Analog forecast: what happened to every look-alike (day, stock) in the
    dump. Returns forward-return distribution stats with base-rate lifts."""
    now, sector = _now_features(ticker)
    if now is None:
        return {}
    if F is None or R is None:
        F, R = _feature_panels()
    tol = np.array([0.10, 0.15, 0.50, 0.0])
    for widen in (1.0, 1.6, 2.4):
        m = (np.abs(F[:, 0] - now[0]) <= tol[0] * widen) &             (np.abs(F[:, 1] - now[1]) <= tol[1] * widen) &             (np.abs(np.minimum(F[:, 2], 3) - min(now[2], 3)) <= tol[2] * widen) &             (F[:, 3] == now[3])
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
                   rvol=float(now[2]), above_ma50=bool(now[3])),
    )


def upstream_drivers(ticker: str, node_closes: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    """Which cascade NODES lead this stock? corr(node 5d move at t,
    stock 5d move at t+5) — plus each node's CURRENT impulse = tailwind."""
    panel, tickers, sectors, mdv, dts = load_dump_panel()
    ix = np.where(tickers == ticker)[0]
    if not len(ix):
        return pd.DataFrame()
    if node_closes is None or node_closes.empty:
        return pd.DataFrame()
    s = pd.Series(panel["c"][:, ix[0]], index=dts).dropna()
    imp_now = impulses(node_closes).iloc[-1]
    rows = []
    for node in node_closes.columns:
        n = node_closes[node].dropna()
        n.index = pd.to_datetime(n.index).tz_localize(None)
        common = s.index.intersection(n.index)
        if len(common) < 120:
            continue
        sv, nv = s.reindex(common).values, n.reindex(common).values
        s5 = sv[5:] / sv[:-5] - 1.0
        n5 = nv[5:] / nv[:-5] - 1.0
        x, y = n5[:-5], s5[5:]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 100:
            continue
        c = float(np.corrcoef(x[ok], y[ok])[0, 1])
        rows.append(dict(node=node, node_name=NODES.get(node, (node,))[0],
                         follow_corr=round(c, 2),
                         node_z=round(float(imp_now.get(node, np.nan)), 2)))
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
