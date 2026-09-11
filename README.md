# 🌩 Money Weather — the Global Flow Cascade map

Money doesn't teleport — it propagates through the world's assets in
repeatable paths. Fast frictionless nodes (crypto, FX, semis) react first;
slow heavy ones follow. Money Weather estimates that directed lead-lag graph
empirically, detects flow waves entering upstream nodes, and forecasts the
downstream nodes each wave historically reaches — with lag and hit rate.

**Layers**
- 📅 Forced Flows — scheduled, price-insensitive money movement (rebalances,
  OpEx, month-end pensions, buyback windows). The knowable flows.
- 🌡 Pressure — global net liquidity nowcast (Fed BS − TGA − RRP, stablecoin
  supply, HY spreads). Rising pressure = waves travel far.
- 🛰 Sentinels — 24/7 early-warning assets (BTC, yen, copper, semis, HY).
- 🌊 Cascade Map — the storm tracks: ~50 global nodes, edges re-estimated
  walk-forward weekly, live wave → downstream forecasts.
- 🔬 Validation Lab — one-click walk-forward backtest with an honesty split.

**Deploy**
1. Push this folder (or connect the repo) to [share.streamlit.io](https://share.streamlit.io). Main file: `app.py`. Python 3.10+.
2. Optional Cloud secrets (top-level, not nested): `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. See `.streamlit/secrets.toml.example`. Without keys the app uses Yahoo; APEX stays daily-only.
3. First visit downloads the nightly dump (~5,700 stocks, up to ~2 min) and ~3 years of node history, then caches them. Do not ship `data/history.parquet` or `data/*.npz`.
4. Ship together: `app.py`, `cascade_engine.py`, `apex_flow.py`, `poc_future.py`, `macro_simulator.html`, `assets/aiupscale_logo.png`, `requirements.txt`, `.streamlit/config.toml`. Engine and app versions must match (`2.36`).

Research tool. Probability tilts, not prophecy. Not investment advice.
