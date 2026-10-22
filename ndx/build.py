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


def load_fixture(folder: Path):
    close = pd.read_csv(folder / "close.csv", index_col=0, parse_dates=True)
    opens = pd.read_csv(folder / "open.csv", index_col=0, parse_dates=True)
    level = pd.read_csv(folder / "index.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return close, opens, level


def run(out: Path, fixture: Path | None = None, write_log: bool = True, now: dt.datetime | None = None) -> dict:
    members = json.loads((ROOT / "config/members.json").read_text())
    spins = json.loads((ROOT / "config/spins.json").read_text())
    seeds = load_seeds(ROOT / "data/seeds")
    log, idx_log = load_log(ROOT / "data/log")
    tickers = sorted(set().union(*[set(s.index) for s in seeds.values()]) | set(log["ticker"].unique()))
    now = now or dt.datetime.now(ET)
    messages = []

    if fixture:
        close, opens, level = load_fixture(fixture)
    else:
        from .sources import fetch
        start = (now.date() - dt.timedelta(days=440)).isoformat()
        close, opens, level = fetch(tickers, start)

    level = level.dropna()
    # never treat a session that is still trading as a close
    if level.index[-1].date() == now.date() and now.time() < dt.time(16, 30):
        level = level.iloc[:-1]
    T = level.index[-1]
    grid = close.index.union(level.index)
    close = close.reindex(index=grid, columns=sorted(set(close.columns) | set(tickers)))
    opens = opens.reindex(index=grid)
    adj, spin_applied = A.adjust_for_spins(close, opens, spins)

    # append newly closed days to the log
    L = min(seeds)
    last = pd.Timestamp(log["date"].max()) if not log.empty else L
    todo = [d for d in level.index if last < d <= T]
    new_idx = pd.DataFrame()
    if todo:
        new_rows, new_idx, problems = A.log_days(adj, level, seeds, todo)
        messages += problems
        if not new_rows.empty:
            log = pd.concat([log, new_rows], ignore_index=True)
            idx_log = pd.concat([idx_log, new_idx], ignore_index=True)
            if write_log:
                (ROOT / "data/log").mkdir(parents=True, exist_ok=True)
                log.to_csv(ROOT / "data/log/daily_contrib.csv", index=False, float_format="%.6f")
                idx_log.to_csv(ROOT / "data/log/daily_index.csv", index=False, float_format="%.4f")
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
             "spinAdjustments": spin_applied, "source": "fixture" if fixture else "Yahoo Finance"}]
    data = {"meta": meta, "index_moves": moves, "contributions": contrib}
    render(data, out)
    if write_log:
        top = {k: [{"ticker": c["ticker"], "points": c["points"]} for c in contrib if c["period"] == k][:5] for k in order}
        (ROOT / "data/latest.json").write_text(json.dumps({"meta": meta[0], "index_moves": moves, "top5": top}, indent=1))
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
    data = run(Path(a.out), Path(a.fixture) if a.fixture else None, write_log=not a.no_log_write)
    m = data["meta"][0]
    print(f"as of {m['asOf']}: status {m['status']}")
    for x in m["messages"]:
        print("  -", x)
    for r in data["index_moves"]:
        print(f"  {r['period']:>3}  {r['change']:+10.2f}  sum {r['constituentSum']:+10.2f}  other {r['residual']:+8.2f}")
    gh = os.environ.get("GITHUB_OUTPUT")
    if gh:
        with open(gh, "a") as f:
            f.write(f"status={m['status']}\n")
            f.write("messages<<EOF\n" + "\n".join(m["messages"]) + "\nEOF\n")


if __name__ == "__main__":
    main()
