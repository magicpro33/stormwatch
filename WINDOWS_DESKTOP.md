# Money Weather on a clean Windows PC

The person who **builds** the zip needs Python 3.10+ once. The person who
**runs** it does not — they unzip and double-click `MoneyWeather.exe`.

## Build (developer machine)

```powershell
cd "…\moneyweather cursor fixes"
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

Zip **the folder** `dist\MoneyWeather\` (the whole folder, not just the exe).
The script also writes `dist\MoneyWeather-2.39-win64.zip`.

Do **not** ship the small source archive `moneyweather.zip` in this folder
if it exists — that is not the desktop build (~300 MB).

## Run (any Windows 10/11 PC)

1. Unzip.
2. Double-click `MoneyWeather.exe`.
3. Leave that console window open. A browser tab opens on `127.0.0.1`.
4. Close the console window to quit.

First launch downloads market data (about 1–2 minutes) and stores it under
`%LOCALAPPDATA%\MoneyWeather\`. Later launches reuse that cache.

Crash / support log: `%LOCALAPPDATA%\MoneyWeather\moneyweather.log`

Optional Alpaca keys (live prints / APEX / Ignition): create
`%LOCALAPPDATA%\MoneyWeather\secrets.toml`

```toml
ALPACA_API_KEY = "your-key"
ALPACA_SECRET_KEY = "your-secret"
```

The launcher copies that file into `.streamlit\secrets.toml` so both the
engine and Ignition see the same keys. Without keys the app uses Yahoo.
APEX stays daily-only.

## What this package is

A local Streamlit server plus the same engine as Cloud (`app.py` +
`cascade_engine.py` v2.39). It binds to localhost only.

## SmartScreen / code signing

Unsigned exes from a zip will often show a Windows SmartScreen warning.
For a customer release, Authenticode-sign `MoneyWeather.exe` with a
code-signing certificate (outside this repo) before you zip the folder.

Research tool. Not investment advice.
