"""Fetch data, update the daily contribution log, and render the dashboard.

    python -m ndx.build --out public              # live run (Yahoo Finance)
    python -m ndx.build --out public --fixture tests/fixture --no-log-write
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import attribution as A

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")
RECON_BP = 5.0          # max |index move - sum of contributions| on a day, in bp of the index
STALE_DAYS = 4
MIN_COVERAGE = 0.95   # share of current members that must have a close for a day to be used


def load_seeds(folder: Path) -> dict[pd.Timestamp, pd.Series]:
    seeds = {}
    for f in sorted(folder.glob("*.csv")):
        s = pd.read_csv(f).set_index("ticker")["weight"].astype(float)
        seeds[pd.Timestamp(f.stem)] = s / s.sum()
    if not seeds:
        raise SystemExit(f"No weight seeds in {folder}")
    return seeds


def load_log(folder: Path):
    p, q = folder / "daily_contrib.csv", folder / "daily_index.csv"
    log = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["date", "ticker", "points"])
    idx = pd.read_csv(q) if q.exists() else pd.DataFrame(columns=["date", "prev_close", "close", "attributed"])
    return log, idx


def trim_log(log: pd.DataFrame, idx_log: pd.DataFrame, T: pd.Timestamp):
    """Drop logged days on or after close T so they are recomputed from current prices."""
    keep = lambda df: df[pd.to_datetime(df["date"]) < T].reset_index(drop=True) if not df.empty else df
    return keep(log), keep(idx_log)


INDEX_COL = "^NDX"
CACHE_DAYS = 10         # recent closes kept in data/log/closes.csv
CACHE_TOL = 0.005       # cached and fresh closes must agree within 0.5% to trust the cache for a stock


def load_cache(folder: Path) -> pd.DataFrame:
    p = folder / "closes.csv"
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else pd.DataFrame()


def fill_from_cache(close: pd.DataFrame, level: pd.Series, cache: pd.DataFrame):
    """Fill closes Yahoo left out with ones saved by earlier runs.

    Only recent days are cached, and a stock is filled only if its cached and fresh closes
    agree on every day both have (so a split or other restatement is never mixed in).
    """
    if cache.empty:
        return close, level, []
    fresh = close.copy()
    fresh[INDEX_COL] = level
    fresh = fresh.reindex(fresh.index.union(cache.index))
    filled = []
    for t in cache.columns:
        if t not in fresh.columns:
            continue
        c, f = cache[t].dropna(), fresh[t]
        both = c.index.intersection(f.dropna().index)
        if len(both) and ((f[both] / c[both] - 1).abs() > CACHE_TOL).any():
            continue
        gaps = c.index[f.reindex(c.index).isna()]
        if len(gaps):
            fresh.loc[gaps, t] = c[gaps]
            filled += [f"{t} {d.date()}" for d in gaps]
    level = fresh.pop(INDEX_COL)
    return fresh, level, filled


def save_cache(folder: Path, close: pd.DataFrame, level: pd.Series):
    days = level.dropna().index[-CACHE_DAYS:]
    out = close.loc[days].copy()
    out[INDEX_COL] = level.loc[days]
    folder.mkdir(parents=True, exist_ok=True)
    out.index.name = "date"
    out.to_csv(folder / "closes.csv", float_format="%.4f", date_format="%Y-%m-%d")


def load_fixture(folder: Path):
    close = pd.read_csv(folder / "close.csv", index_col=0, parse_dates=True)
    opens = pd.read_csv(folder / "open.csv", index_col=0, parse_dates=True)
    level = pd.read_csv(folder / "index.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return close, opens, level


def usable_close(close: pd.DataFrame, level: pd.Series, members: list[str]) -> pd.Timestamp:
    """Latest date with an index close and closes for at least MIN_COVERAGE of members."""
    coverage = close.reindex(index=level.index, columns=members).notna().mean(axis=1)
    usable = coverage[coverage >= MIN_COVERAGE].index
    if usable.empty:
        raise SystemExit("No day with prices for the index and its members.")
    return usable[-1]


class NotPublished(Exception):
    """The fresh data is older than what is already published; keep the current page."""


def run(out: Path, fixture: Path | None = None, write_log: bool = True, now: dt.datetime | None = None,
        log_dir: Path | None = None) -> dict:
    members = json.loads((ROOT / "config/members.json").read_text())
    spins = json.loads((ROOT / "config/spins.json").read_text())
    seeds = load_seeds(ROOT / "data/seeds")
    log_dir = log_dir or ROOT / "data/log"
    log, idx_log = load_log(log_dir)
    tickers = sorted(set().union(*[set(s.index) for s in seeds.values()]) | set(log["ticker"].unique()))
    now = now or dt.datetime.now(ET)
    messages = []

    if fixture:
        close, opens, level = load_fixture(fixture)
    else:
        from .sources import fetch
        start = (now.date() - dt.timedelta(days=440)).isoformat()
        close, opens, level = fetch(tickers, start)

    close, level, filled = fill_from_cache(close, level, load_cache(log_dir))
    if filled:
        print("note: filled from saved closes: " + ", ".join(filled))
    level = level.dropna()
    # never treat a session that is still trading as a close
    if level.index[-1].date() == now.date() and now.time() < dt.time(16, 30):
        level = level.iloc[:-1]
    grid = close.index.union(level.index)
    close = close.reindex(index=grid, columns=sorted(set(close.columns) | set(tickers)))
    opens = opens.reindex(index=grid)

    # The latest usable close is the last day the index AND nearly all members have prices.
    # (Late in the evening Yahoo can return the index's bar but not the members'.)
    T = usable_close(close, level, [t for t in seeds[max(seeds)].index if t in close.columns])
    if T < level.index[-1]:
        print(f"note: member prices for {level.index[-1].date()} not available yet; using {T.date()}")
    level = level.loc[:T]
    last_logged = pd.Timestamp(log["date"].max()) if not log.empty else None
    if last_logged is not None and T < last_logged:
        raise NotPublished(f"fresh data ends {T.date()}, already published through {last_logged.date()}")
    adj, spin_applied = A.adjust_for_spins(close, opens, spins)

    # Log newly closed days. The latest close is always recomputed, so a later run the same
    # evening replaces the first run's numbers with any corrected closing prices; a day is
    # only replaced once its recomputation has succeeded.
    L = min(seeds)
    base, _ = trim_log(log, idx_log, T)
    last = pd.Timestamp(base["date"].max()) if not base.empty else L
    todo = [d for d in level.index if last < d <= T]
    new_idx = pd.DataFrame()
    if todo:
        new_rows, new_idx, problems = A.log_days(adj, level, seeds, todo)
        messages += problems
        if not new_rows.empty:
            done = set(new_idx["date"])
            log = pd.concat([log[~log["date"].astype(str).isin(done)], new_rows], ignore_index=True)
            idx_log = pd.concat([idx_log[~idx_log["date"].astype(str).isin(done)], new_idx], ignore_index=True)
            log = log.sort_values("date", kind="stable", ignore_index=True)
            idx_log = idx_log.sort_values("date", kind="stable", ignore_index=True)
            if write_log:
                log_dir.mkdir(parents=True, exist_ok=True)
                log.to_csv(log_dir / "daily_contrib.csv", index=False, float_format="%.6f")
                idx_log.to_csv(log_dir / "daily_index.csv", index=False, float_format="%.4f")
    for _, r in new_idx.iterrows():
        gap = (r["close"] - r["prev_close"]) - r["attributed"]
        if abs(gap) / r["prev_close"] * 1e4 > RECON_BP:
            messages.append(f"{r['date']}: contributions miss the index move by {gap:+.1f} pts. "
                            "Likely an index change (rebalance, addition or removal): refresh the weights seed.")

    res = A.attribute(adj, level, seeds, log, members, T)

    # quality checks
    d1 = res["1D"]
    if abs(d1["residual"]) / d1["I0"] * 1e4 > RECON_BP and not any("contributions miss" in m for m in messages):
        messages.append(f"1-day contributions miss the index move by {d1['residual']:+.1f} pts.")
    latest = adj.loc[T, [t for t in seeds[max(seeds)].index if t in adj.columns]]
    if latest.isna().any():
        messages.append("No closing price for " + ", ".join(latest[latest.isna()].index) + f" on {T.date()}.")
    if (now.date() - T.date()).days > STALE_DAYS:
        messages.append(f"Latest index close is {T.date()}; data may be stale.")

    order = [k for k, _ in A.PERIODS]
    moves, contrib = [], []
    for i, k in enumerate(order):
        p = res[k]
        items = p["items"]
        moves.append({"period": k, "order": i + 1, "label": p["label"], "startDate": p["start"].date().isoformat(),
                      "endDate": T.date().isoformat(), "startLevel": round(p["I0"], 2), "endLevel": round(p["I1"], 2),
                      "change": round(p["dI"], 2), "changePct": round(p["ret"], 2),
                      "constituentSum": round(p["sum"], 2), "residual": round(p["residual"], 2),
                      "top50Sum": round(sum(x["points"] for x in items[:50]), 2),
                      "advancers": sum(1 for x in items if x["points"] > 0),
                      "decliners": sum(1 for x in items if x["points"] < 0)})
        for r, x in enumerate(items, 1):
            contrib.append({"period": k, "rank": r, "ticker": x["ticker"], "name": x["name"],
                            "points": round(x["points"], 2), "indexPctPts": round(x["indexPctPts"], 3),
                            "shareOfMove": round(x["shareOfMove"], 1),
                            "stockReturnPct": None if x["stockReturnPct"] is None else round(x["stockReturnPct"], 2),
                            "weightPct": round(x["weightPct"], 2), "startDate": x["startDate"],
                            "startPrice": None if x["startPrice"] is None else round(x["startPrice"], 4),
                            "endPrice": x["endPrice"], "joined": x["joined"]})
    meta = [{"asOf": T.date().isoformat(), "generatedAt": now.strftime("%Y-%m-%d %H:%M %Z"),
             "logStart": L.date().isoformat(), "seedDate": max(seeds).date().isoformat(),
             "status": "warn" if messages else "ok", "messages": messages,
             "spinAdjustments": spin_applied, "filledFromCache": filled, "source": "fixture" if fixture else "Yahoo Finance"}]
    data = {"meta": meta, "index_moves": moves, "contributions": contrib}
    render(data, out)
    if write_log:
        top = {k: [{"ticker": c["ticker"], "points": c["points"]} for c in contrib if c["period"] == k][:5] for k in order}
        save_cache(log_dir, close, level)
        (log_dir.parent / "latest.json").write_text(json.dumps({"meta": meta[0], "index_moves": moves, "top5": top}, indent=1))
    return data


def render(data: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = (ROOT / "site/template.html").read_text().replace("__NX_DATA__", blob)
    (out / "index.html").write_text(html)
    (out / "data.json").write_text(json.dumps(data, indent=1, allow_nan=False))
    shutil.copy(ROOT / "site/d3.min.js", out / "d3.min.js")
    (out / ".nojekyll").write_text("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public")
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--no-log-write", action="store_true")
    a = ap.parse_args()
    gh = os.environ.get("GITHUB_OUTPUT")
    try:
        data = run(Path(a.out), Path(a.fixture) if a.fixture else None, write_log=not a.no_log_write)
    except NotPublished as e:
        print(f"not publishing: {e}")
        if gh:
            with open(gh, "a") as f:
                f.write("publish=false\nstatus=ok\nmessages<<EOF\n\nEOF\n")
        return
    m = data["meta"][0]
    print(f"as of {m['asOf']}: status {m['status']}")
    for x in m["messages"]:
        print("  -", x)
    for r in data["index_moves"]:
        print(f"  {r['period']:>3}  {r['change']:+10.2f}  sum {r['constituentSum']:+10.2f}  other {r['residual']:+8.2f}")
    if gh:
        with open(gh, "a") as f:
            f.write("publish=true\n")
            f.write(f"status={m['status']}\n")
            f.write("messages<<EOF\n" + "\n".join(m["messages"]) + "\nEOF\n")


if __name__ == "__main__":
    main()
