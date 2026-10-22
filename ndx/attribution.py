"""Nasdaq-100 contribution attribution. Pure functions: no network, no files.

Notation
--------
I_t      index level at close t
w_{i,t}  index weight of member i at close t (sums to 1 over members)
V_{i,t}  index points held in member i at close t  (= I_t * w_{i,t})
r_{i,t}  member i's price return on day t (spin-off adjusted)

Daily contribution:   c_{i,t} = I_{t-1} * w_{i,t-1} * r_{i,t}
Window contribution:  sum of daily contributions over the window.

Days on or before the log origin L (the first weights seed) have no stored weights, so
they are estimated by rolling the seed weights back with each member's own price:
    V_{i,s} = V_{i,L} / (1 + R_i(s->L))   =>   contribution(s->L) = V_{i,L} * R / (1 + R)
That is exact when nothing was rebalanced inside (s, L]. Days after L are logged daily with
the weights in force at the time, so they chain through rebalances and membership changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = [("1D", "1 day"), ("1W", "1 week"), ("1M", "1 month"), ("YTD", "Year to date"), ("1Y", "1 year")]


def anchor_dates(trading_days: pd.DatetimeIndex, T: pd.Timestamp) -> dict[str, pd.Timestamp]:
    """Start date (a prior close) for each period ending at close T."""
    prev = trading_days[trading_days < T]

    def on_or_before(d):
        return prev[prev <= d].max()

    return {
        "1D": prev.max(),
        "1W": on_or_before(T - pd.Timedelta(days=7)),
        "1M": on_or_before(T - pd.DateOffset(months=1)),
        "YTD": prev[prev.year < T.year].max(),
        "1Y": on_or_before(T - pd.DateOffset(years=1)),
    }


def adjust_for_spins(close: pd.DataFrame, opens: pd.DataFrame | None, spins: list[dict], min_gap: float = 0.02):
    """Back-adjust a parent's closes for a spin-off the price feed left unadjusted.

    The distribution is read off the ex-date opening gap: factor = open_ex / close_cum.
    Applied only when the gap is a drop of more than `min_gap`; a smaller gap means the
    feed already adjusted the history, so nothing is changed.
    """
    adj = close.copy()
    applied = []
    for s in spins:
        t, ex = s["ticker"], pd.Timestamp(s["ex_date"])
        if t not in adj.columns or opens is None or t not in opens.columns or ex not in opens.index:
            continue
        o = opens.at[ex, t]
        prior = adj[t].loc[: ex - pd.Timedelta(days=1)].dropna()
        if prior.empty or not np.isfinite(o):
            continue
        f = float(o / prior.iloc[-1])
        if f < 1 - min_gap:
            adj.loc[adj.index < ex, t] *= f
            applied.append({"ticker": t, "ex_date": s["ex_date"], "factor": round(f, 6)})
    return adj, applied


def base_price(px: pd.Series, start: pd.Timestamp, entered: pd.Timestamp | None):
    """Price a window measures from: the window's start close, or for a member that joined
    inside the window, the last close before it joined (first close if it listed on entry)."""
    s = px.dropna()
    if entered is not None and entered > start:
        before = s.loc[: entered - pd.Timedelta(days=1)]
        if not before.empty:
            return before.index[-1], float(before.iloc[-1]), entered
        after = s.loc[entered:]
        return (after.index[0], float(after.iloc[0]), entered) if not after.empty else (None, np.nan, entered)
    before = s.loc[:start]
    return (before.index[-1], float(before.iloc[-1]), None) if not before.empty else (None, np.nan, None)


def weights_at(d: pd.Timestamp, seeds: dict[pd.Timestamp, pd.Series], adj: pd.DataFrame) -> pd.Series:
    """Weights at close d: the latest seed on or before d, drifted by price to d."""
    sd = max(k for k in seeds if k <= d)
    w = seeds[sd]
    p_d = adj.loc[:d, w.index].ffill().iloc[-1]
    p_s = adj.loc[:sd, w.index].ffill().iloc[-1]
    w2 = (w * p_d / p_s).fillna(0.0)
    return w2 / w2.sum()


def log_days(adj: pd.DataFrame, level: pd.Series, seeds: dict, days: list[pd.Timestamp]):
    """Daily contributions for each day in `days` (each must have a prior index close).
    Stops at the first day with a missing member price so it is retried on the next run."""
    rows, idx_rows, problems = [], [], []
    tds = level.dropna().index
    for d in days:
        p = tds[tds < d].max()
        w = weights_at(p, seeds, adj)
        r = adj.loc[d, w.index] / adj.loc[:p, w.index].ffill().iloc[-1] - 1
        missing = r[r.isna()].index.tolist()
        if missing:
            problems.append(f"{d.date()}: no price for {', '.join(missing)}; day not logged")
            break
        pts = level[p] * w * r
        rows += [{"date": d.date().isoformat(), "ticker": t, "points": float(v)} for t, v in pts.items()]
        idx_rows.append({"date": d.date().isoformat(), "prev_close": float(level[p]), "close": float(level[d]),
                         "attributed": float(pts.sum())})
    return pd.DataFrame(rows), pd.DataFrame(idx_rows), problems


def attribute(adj: pd.DataFrame, level: pd.Series, seeds: dict, log: pd.DataFrame,
              members: dict, T: pd.Timestamp) -> dict:
    """Per-period index move and member contributions ending at close T."""
    L = min(seeds)                      # log origin: first seed date
    wL = seeds[L]
    VL = level[L] * wL                  # index points per member at L
    wT = weights_at(T, seeds, adj)
    tds = level.dropna().index
    anchors = anchor_dates(tds, T)
    log = log.copy()
    if not log.empty:
        log["date"] = pd.to_datetime(log["date"])
    out = {}
    for key, label in PERIODS:
        s = anchors[key]
        I0, I1 = float(level[s]), float(level[T])
        pts = pd.Series(0.0, index=sorted(set(wL.index) | set(wT.index) | set(log["ticker"] if not log.empty else [])))
        info = {}
        # estimated part: (s, L]
        if s < L:
            for t in wL.index:
                ent = members.get(t, {}).get("entered")
                ent = pd.Timestamp(ent) if ent else None
                bd, bp, _ = base_price(adj[t], s, ent)
                if ent is not None and ent > L:
                    continue
                pL = adj[t].loc[:L].dropna()
                if not np.isfinite(bp) or pL.empty:
                    continue
                R = pL.iloc[-1] / bp - 1
                pts[t] += VL[t] * R / (1 + R)
        # logged part: (max(s, L), T]
        if not log.empty:
            lo = max(s, L)
            seg = log[(log["date"] > lo) & (log["date"] <= T)]
            pts = pts.add(seg.groupby("ticker")["points"].sum(), fill_value=0.0)
        items = []
        for t in pts.index:
            ent = members.get(t, {}).get("entered")
            ent = pd.Timestamp(ent) if ent else None
            bd, bp, joined = base_price(adj[t], s, ent) if t in adj.columns else (None, np.nan, None)
            last = adj[t].loc[:T].dropna() if t in adj.columns else pd.Series(dtype=float)
            ret = (last.iloc[-1] / bp - 1) * 100 if (np.isfinite(bp) and not last.empty) else np.nan
            items.append({
                "ticker": t, "name": members.get(t, {}).get("name", t),
                "points": float(pts[t]), "indexPctPts": float(pts[t] / I0 * 100),
                "shareOfMove": float(pts[t] / (I1 - I0) * 100) if I1 != I0 else 0.0,
                "stockReturnPct": float(ret) if np.isfinite(ret) else None,
                "weightPct": float(wT.get(t, 0.0) * 100),
                "startDate": bd.date().isoformat() if bd is not None else None,
                "startPrice": bp if np.isfinite(bp) else None,
                "endPrice": float(last.iloc[-1]) if not last.empty else None,
                "joined": joined.date().isoformat() if joined is not None else "",
            })
        items = [i for i in items if i["points"] != 0.0 or i["ticker"] in wT.index]
        items.sort(key=lambda i: abs(i["points"]), reverse=True)
        tot = sum(i["points"] for i in items)
        out[key] = {"label": label, "start": s, "I0": I0, "I1": I1, "dI": I1 - I0, "ret": (I1 / I0 - 1) * 100,
                    "sum": tot, "residual": (I1 - I0) - tot, "items": items}
    return out
