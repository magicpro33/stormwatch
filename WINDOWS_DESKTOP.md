# Money Maker — Windows desktop

Build a double-click folder with `build_windows.ps1` (PowerShell, Python 3.10+ on the *build* machine). The finished folder does not need Python.

## Build

1. Put `assets\aiupscale_logo.png` next to `app.py` (the spec bundles it).
2. From the repo root: `.\build_windows.ps1`
3. The portable app is copied to `..\MoneyMaker\` (next to this repo). Copy that whole folder. Do not move `MoneyMaker.exe` out of it.

Engine and app versions must match (**2.43**).

## Run

Double-click `MoneyMaker.exe`. Keep that window open. The browser opens to a local `127.0.0.1` page. Close the window to quit.

First run downloads the nightly dump and node history (1–2 minutes). Caches live in `%LOCALAPPDATA%\MoneyWeather\`. Support log: `%LOCALAPPDATA%\MoneyWeather\moneyweather.log`.

## Optional live prices

Put a `secrets.toml` in `%LOCALAPPDATA%\MoneyWeather\`:

```
ALPACA_API_KEY = "..."
ALPACA_SECRET_KEY = "..."
```

Without keys the app still runs: daily scans and lookup use the nightly dump, then Yahoo.

Not investment advice.
