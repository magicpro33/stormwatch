"""Windows desktop entry — no Python install required after PyInstaller.

Double-click MoneyWeather.exe. This window must stay open while you use
the app. Closing it shuts the local server down.
"""
from __future__ import annotations
import os
import socket
import sys
import threading
import time
import webbrowser


def _bundle() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def _flat_toml(data: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (data or {}).items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat_toml(v, ""))
        elif v is not None:
            out[str(k)] = v
    return out


def _read_secrets_file(path: str) -> dict:
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(path, "rb") as f:
            return _flat_toml(tomllib.load(f))
    except Exception:
        out = {}
        try:
            with open(path, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and v:
                        out[k] = v
        except OSError:
            return {}
        return out


def _load_user_secrets(user_dir: str) -> None:
    """Alpaca keys from %LOCALAPPDATA%\\MoneyWeather\\secrets.toml (optional)."""
    import shutil
    st_dir = os.path.join(user_dir, ".streamlit")
    os.makedirs(st_dir, exist_ok=True)
    src = os.path.join(user_dir, "secrets.toml")
    dst = os.path.join(st_dir, "secrets.toml")
    path = src if os.path.isfile(src) else (dst if os.path.isfile(dst) else "")
    if not path:
        return
    try:
        if path != dst:
            shutil.copy2(path, dst)
    except OSError:
        pass
    data = _read_secrets_file(path)
    for k, v in data.items():
        s = str(v).strip()
        if k and s and s not in ("your-key", "your-secret"):
            os.environ.setdefault(str(k), s)


def _prepare_user_dir(bundle: str) -> str:
    from mw_paths import user_data_dir
    user = user_data_dir()
    st_dir = os.path.join(user, ".streamlit")
    data_dir = os.path.join(user, "data")
    os.makedirs(st_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    cfg_src = os.path.join(bundle, ".streamlit", "config.toml")
    cfg_dst = os.path.join(st_dir, "config.toml")
    if os.path.isfile(cfg_src):
        try:
            with open(cfg_src, "rb") as a, open(cfg_dst, "wb") as b:
                b.write(a.read())
        except OSError:
            pass
    _load_user_secrets(user)
    return user


def _say(msg: str) -> None:
    """Console on Windows is often cp1252 — never print box-drawing or dashes."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def main() -> int:
    os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")
    os.environ["MW_DESKTOP"] = "1"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    bundle = _bundle()
    if bundle not in sys.path:
        sys.path.insert(0, bundle)
    try:
        from mw_log import setup_logging, get_logger
        setup_logging()
        log = get_logger()
    except Exception:
        log = None

    try:
        user = _prepare_user_dir(bundle)
    except Exception as e:
        if log is not None:
            log.exception("prepare user dir")
        _say(f"Money Weather failed to start: {e}")
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
        return 1
    os.chdir(user)

    app = os.path.join(bundle, "app.py")
    if not os.path.isfile(app):
        _say("ERROR: app.py is missing from the package.")
        _say("Re-run build_windows.ps1 and use the whole dist\\MoneyWeather folder.")
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
        return 1

    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    _say("")
    _say("  Money Weather")
    _say("  ---------------------------------")
    _say(f"  Opening {url}")
    _say("  Keep this window open. Close it to quit.")
    _say("  First run downloads market data (1-2 minutes).")
    _say(f"  Cache folder: {user}")
    _say("  Optional Alpaca keys: secrets.toml in that folder.")
    _say("")

    def _open_browser():
        for _ in range(40):
            time.sleep(0.4)
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=0.4)
                s.close()
                break
            except OSError:
                continue
        try:
            webbrowser.open(url)
        except Exception:
            _say(f"  Open this address yourself: {url}")

    threading.Thread(target=_open_browser, daemon=True).start()

    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit", "run", app,
        f"--server.port={port}",
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
        "--client.toolbarMode=minimal",
    ]
    try:
        return int(stcli.main() or 0)
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        if log is not None:
            log.exception("Money Weather failed to start")
        else:
            import traceback
            traceback.print_exc()
        _say(f"Money Weather failed to start: {e}")
        _say("Log: %LOCALAPPDATA%\\MoneyWeather\\moneyweather.log")
        try:
            input("Press Enter to close...")
        except EOFError:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
