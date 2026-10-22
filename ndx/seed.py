"""Write a new weights seed from a holdings file (e.g. the Invesco QQQ holdings download).

    python -m ndx.seed path/to/holdings.csv --date 2026-12-21

Use the holdings as of a close (the date given). The file needs a ticker column and a
weight column; weights may be in percent. New members also need an entry in
config/members.json (name, and the date they joined the index).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("holdings")
    ap.add_argument("--date", required=True, help="close the weights are as of, YYYY-MM-DD")
    a = ap.parse_args()
    df = pd.read_csv(a.holdings)
    tcol = next(c for c in df.columns if "ticker" in c.lower() or c.lower() in ("symbol", "holding"))
    wcol = next(c for c in df.columns if "weight" in c.lower())
    w = (df[[tcol, wcol]].dropna().rename(columns={tcol: "ticker", wcol: "weight"}))
    w["ticker"] = w["ticker"].astype(str).str.strip().str.upper()
    w["weight"] = pd.to_numeric(w["weight"].astype(str).str.rstrip("%"), errors="coerce")
    w = w[w["weight"] > 0]
    w = w[~w["ticker"].str.contains("CASH|USD|-")]
    w["weight"] = w["weight"] / w["weight"].sum()
    members = json.loads((ROOT / "config/members.json").read_text())
    new = sorted(set(w["ticker"]) - set(members))
    gone = sorted(set(members) - set(w["ticker"]))
    out = ROOT / "data/seeds" / f"{a.date}.csv"
    w.sort_values("weight", ascending=False).to_csv(out, index=False, float_format="%.8f")
    print(f"wrote {out} ({len(w)} members)")
    if new:
        print("add to config/members.json (name, entered):", ", ".join(new))
    if gone:
        print("no longer held (left the index):", ", ".join(gone))


if __name__ == "__main__":
    main()
