# Build a double-click Windows folder.
# Output: dist\MoneyMaker\MoneyMaker.exe
# Also copies that folder to ..\MoneyMaker so you can zip/move it.
# Run from this directory in PowerShell. Takes several minutes. No UPX.
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$logo = Join-Path $PSScriptRoot "assets\aiupscale_logo.png"
if (-not (Test-Path -LiteralPath $logo)) {
    Write-Error "assets\aiupscale_logo.png is missing. Do not ship without the logo."
}

$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) {
    Write-Error "Python is required on THIS machine to build. The finished folder does not need Python."
}

$venv = Join-Path $PSScriptRoot ".venv-build"
$venvPy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPy)) {
    Write-Host "Creating build venv..."
    python -m venv $venv
}

Write-Host "Installing build dependencies..."
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r requirements.txt
& $venvPy -m pip install -r requirements-build.txt

Write-Host "Compile-check..."
& $venvPy -m compileall -q `
    app.py cascade_engine.py desktop_main.py mw_paths.py mw_log.py mw_secrets.py mw_yf.py `
    apex_flow.py poc_future.py hybrid_screener.py ignition_scanner.py `
    storm_watch_tab.py storm_watch_engine.py crypto_copycat.py `
    charts_tab.py dividend_calendar.py elections_scan.py mw_fortune_table.py
if ($LASTEXITCODE -ne 0) { Write-Error "compileall failed" }

if (Test-Path ".\dist\MoneyMaker") { Remove-Item -Recurse -Force ".\dist\MoneyMaker" }
if (Test-Path ".\dist\MoneyWeather") { Remove-Item -Recurse -Force ".\dist\MoneyWeather" }
if (Test-Path ".\build") { Remove-Item -Recurse -Force ".\build" }

Write-Host "PyInstaller (onedir)..."
& $venvPy -m PyInstaller --noconfirm --clean MoneyWeather.spec

$exe = ".\dist\MoneyMaker\MoneyMaker.exe"
$app = ".\dist\MoneyMaker\_internal\app.py"
$bundledLogo = ".\dist\MoneyMaker\_internal\assets\aiupscale_logo.png"
if (-not (Test-Path $exe)) { Write-Error "Build missing MoneyMaker.exe" }
if (-not (Test-Path $app)) { Write-Error "Build missing bundled app.py" }
if (-not (Test-Path $bundledLogo)) { Write-Error "Build missing bundled logo" }

foreach ($f in @("LICENSE", "THIRD_PARTY_NOTICES.txt", "WINDOWS_DESKTOP.md")) {
    if (Test-Path $f) { Copy-Item $f ".\dist\MoneyMaker\" -Force }
}

$pyVer = & $venvPy -c "import sys; print(sys.version.replace(chr(10),' '))"
$freeze = & $venvPy -m pip freeze
$buildInfo = @"
Money Maker 2.43
Python: $pyVer
$freeze
"@
Set-Content -Path ".\dist\MoneyMaker\BUILD_INFO.txt" -Value $buildInfo -Encoding UTF8

$readme = @"
Money Maker 2.43 — Windows
============================
Double-click MoneyMaker.exe. Keep that window open. Your browser opens
to a local page (127.0.0.1). Close the window to quit.

Copy this WHOLE folder to any Windows PC. Do not move MoneyMaker.exe
out of the folder — it needs the _internal folder next to it.

First run downloads the nightly dump and node history (1-2 minutes).
Caches live in %LOCALAPPDATA%\MoneyWeather\
Support log: %LOCALAPPDATA%\MoneyWeather\moneyweather.log

Optional live prices / APEX / Ignition: put a secrets.toml in that cache
folder:

    ALPACA_API_KEY = "..."
    ALPACA_SECRET_KEY = "..."

Not investment advice.
"@
Set-Content -Path ".\dist\MoneyMaker\README.txt" -Value $readme -Encoding UTF8

$ship = Join-Path (Split-Path $PSScriptRoot -Parent) "MoneyMaker"
Write-Host "Copying to $ship ..."
if (Test-Path $ship) { Remove-Item -Recurse -Force $ship }
Copy-Item -Path ".\dist\MoneyMaker" -Destination $ship -Recurse

Write-Host ""
Write-Host "Done. Folder:  $ship"
Write-Host "Run:          $ship\MoneyMaker.exe"
Write-Host "Copy that whole folder to any Windows PC and double-click MoneyMaker.exe."
