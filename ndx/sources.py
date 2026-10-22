"""Price and index data from Yahoo Finance (via yfinance)."""
from __future__ import annotations

import time

import pandas as pd

INDEX_SYMBOL = "^NDX"


def fetch(tickers: list[str], start: str, retries: int = 4):
    """Daily closes and opens for `tickers` plus the Nasdaq-100 level since `start`.

    Closes are split-adjusted but not dividend-adjusted (price return, like the index).
    Returns (close, opens, level) with a tz-naive daily DatetimeIndex.
    """
    import yfinance as yf

    syms = sorted(set(tickers)) + [INDEX_SYMBOL]
    last_err = None
    df = None
    for attempt in range(retries):
        try:
            df = yf.download(syms, start=start, auto_adjust=False, actions=False, progress=False,
                             threads=True, group_by="column")
            if df is not None and not df.empty:
                break
        except Exception as e:  # network / rate limit
            last_err = e
        time.sleep(20 * (attempt + 1))
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo download failed: {last_err}")

    close, opens = df["Close"].copy(), df["Open"].copy()
    for frame in (close, opens):
        idx = pd.to_datetime(frame.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        frame.index = idx.normalize()
    level = close.pop(INDEX_SYMBOL)
    opens = opens.drop(columns=[INDEX_SYMBOL])
    return close, opens, level
