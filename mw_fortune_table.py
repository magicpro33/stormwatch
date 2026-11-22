"""Fortune scan lists as HTML with clickable headers and rows.

The glide grid sorts and selects by header and row clicks. Fortune cannot use
that grid, so the same actions are ordinary links. A header click sorts the
list. A row click opens that ticker.
"""
from __future__ import annotations

import html as _html
import re
from urllib.parse import urlencode

import pandas as pd

_EMPTY = {"", "nan", "none", "—", "–", "-", "n/a", "na"}
_TH = re.compile(r"<th([^>]*)>(.*?)</th>", re.I | re.S)
_TR = re.compile(r"<tr[^>]*>.*?</tr>", re.I | re.S)
_TD = re.compile(r"<td([^>]*)>(.*?)</td>", re.I | re.S)


def _ticker_col(frame):
    for name in ("Ticker", "ticker", "Symbol", "symbol"):
        if name in frame.columns:
            return name
    return None


def _as_styler(data):
    if hasattr(data, "to_html") and hasattr(data, "data"):
        return data
    return data.style


def _sort_frame(frame, col, ascending):
    if col not in frame.columns:
        return frame
    series = frame[col]
    if pd.api.types.is_numeric_dtype(series):
        return frame.sort_values(col, ascending=ascending, na_position="last", kind="mergesort")
    text = series.astype(str).str.strip()
    num = pd.to_numeric(
        text.str.replace(r"[$,%\s,]", "", regex=True).str.replace(r"^\+", "", regex=True).str.replace(r"x$", "", regex=True, case=False),
        errors="coerce",
    )
    usable = int(num.notna().sum())
    if usable and usable >= max(1, int(0.6 * len(frame))):
        tmp = frame.copy()
        tmp["_mw_sort_key"] = num
        out = tmp.sort_values("_mw_sort_key", ascending=ascending, na_position="last", kind="mergesort")
        return out.drop(columns="_mw_sort_key")
    tmp = frame.copy()
    key = text.str.lower()
    key = key.mask(key.isin(_EMPTY), other=pd.NA)
    tmp["_mw_sort_key"] = key
    out = tmp.sort_values("_mw_sort_key", ascending=ascending, na_position="last", kind="mergesort")
    return out.drop(columns="_mw_sort_key")


def _query_map(params):
    out = {}
    try:
        keys = list(params.keys())
    except Exception:
        return out
    for key in keys:
        try:
            val = params[key]
        except Exception:
            continue
        if isinstance(val, (list, tuple)):
            val = val[0] if val else ""
        out[str(key)] = "" if val is None else str(val)
    return out


def _href(params, list_key, **updates):
    query = _query_map(params)
    for key in ("mw_list", "mw_sort", "mw_dir", "mw_pick"):
        query.pop(key, None)
    query["mw_list"] = str(list_key)
    for key, val in updates.items():
        query[key] = "" if val is None else str(val)
    return "?" + urlencode(query)


def _consume(params, list_key):
    """Read this table's click, then drop it so the next run stays clean."""
    current = _query_map(params)
    if current.get("mw_list") != str(list_key):
        return None, None
    sort_col = current.get("mw_sort") or None
    sort_dir = current.get("mw_dir") if current.get("mw_dir") in ("asc", "desc") else "asc"
    pick = (current.get("mw_pick") or "").strip()
    for key in ("mw_list", "mw_sort", "mw_dir", "mw_pick"):
        try:
            del params[key]
        except Exception:
            pass
    return (sort_col, sort_dir) if sort_col else None, pick or None


def _header_label(raw):
    text = re.sub(r"<[^>]+>", "", raw or "")
    return _html.unescape(text).replace("\xa0", " ").strip()


def decorate_table(table_html, list_key, href_for, sort_col, sort_dir, tickers):
    """Turn column names into sort links and, when tickers are given, rows into pick links."""
    head, tag, body = table_html.partition("<tbody>")
    if not tag:
        return table_html

    def _th(match):
        label = _header_label(match.group(2))
        if not label:
            return match.group(0)
        direction = "desc" if sort_col == label and sort_dir == "asc" else "asc"
        href = _html.escape(href_for(mw_sort=label, mw_dir=direction), quote=True)
        arrow = ""
        if sort_col == label:
            arrow = " ▲" if sort_dir == "asc" else " ▼"
        shown = _html.escape(label) + arrow
        return f'<th{match.group(1)}><a class="mw-fq-sort" href="{href}">{shown}</a></th>'

    head = _TH.sub(_th, head)
    if not tickers:
        return head + tag + body

    rows = _TR.findall(body)
    if len(rows) != len(tickers):
        return head + tag + body
    linked = []
    for row, ticker in zip(rows, tickers):
        tk = str(ticker or "").strip().upper()
        if not tk or tk.lower() in _EMPTY:
            linked.append(row)
            continue
        href = _html.escape(href_for(mw_pick=tk), quote=True)

        def _td(match, href=href):
            return f'<td{match.group(1)}><a class="mw-fq-hit" href="{href}">{match.group(2)}</a></td>'

        linked.append(_TD.sub(_td, row))
    body_out = body
    for old, new in zip(rows, linked):
        body_out = body_out.replace(old, new, 1)
    return head + tag + body_out


def render_fortune_table(data, key, selectable=False, column_order=None):
    """Draw one Fortune list. Return original row positions when a row was clicked."""
    import streamlit as st

    list_key = str(key or "fq_list")
    styler = _as_styler(data)
    try:
        styler = styler.set_properties(**{
            "background-color": "#FFFCF6",
            "color": "#2C2416",
        })
        todo = getattr(styler, "_todo", None)
        if todo and len(todo) > 1:
            todo.insert(0, todo.pop())
    except Exception:
        pass
    try:
        styler = styler.hide(axis="index")
    except Exception:
        pass
    base = pd.DataFrame(styler.data).copy()
    if column_order:
        cols = [c for c in column_order if c in base.columns]
        rest = [c for c in base.columns if c not in cols]
        ordered = base.reindex(columns=cols + rest)
        try:
            styler.data = ordered
            if rest:
                styler = styler.hide(subset=rest, axis="columns")
            base = pd.DataFrame(styler.data).copy()
        except Exception:
            base = ordered
    sort_req, pick = _consume(st.query_params, list_key)
    state_key = f"fq_sort_{list_key}"
    saved = st.session_state.get(state_key) or {}
    if sort_req and sort_req[0] in base.columns:
        saved = {"col": sort_req[0], "dir": sort_req[1]}
        st.session_state[state_key] = saved
    sort_col = saved.get("col") if saved.get("col") in base.columns else None
    sort_dir = saved.get("dir") if saved.get("dir") in ("asc", "desc") else "asc"
    view = _sort_frame(base, sort_col, sort_dir == "asc") if sort_col else base
    try:
        styler.data = view.reindex(columns=list(styler.data.columns))
        table_html = styler.to_html()
    except Exception:
        table_html = view.to_html(index=False)
    tcol = _ticker_col(base)
    tickers = []
    if selectable and tcol and tcol in view.columns:
        tickers = [str(v).strip().upper() for v in view[tcol].tolist()]
    href_for = lambda **updates: _href(st.query_params, list_key, **updates)
    table_html = decorate_table(table_html, list_key, href_for, sort_col, sort_dir, tickers)
    wrapped = "<div class='mw-fq-table'>" + table_html + "</div>"
    if hasattr(st, "html"):
        st.html(wrapped)
    else:
        st.markdown(wrapped, unsafe_allow_html=True)
    if selectable:
        st.caption("Click a row to open it and add it to the watchlist. Click a column name to sort, then click it again to reverse.")
    picked = []
    if pick and tcol and tcol in base.columns:
        want = pick.strip().upper()
        for i, raw in enumerate(base[tcol].tolist()):
            if str(raw).strip().upper() == want:
                picked = [int(i)]
                break
    return picked
