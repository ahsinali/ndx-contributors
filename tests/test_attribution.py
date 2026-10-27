"""Offline tests on a fixture of real closes around Oct 21, 2026.

    python -m pytest -q
"""
import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ndx import attribution as A
from ndx import build

FIX = Path(__file__).parent / "fixture"
NOW = dt.datetime(2026, 10, 21, 20, 0, tzinfo=build.ET)


def _inputs():
    close, opens, level = build.load_fixture(FIX)
    spins = json.loads((build.ROOT / "config/spins.json").read_text())
    members = json.loads((build.ROOT / "config/members.json").read_text())
    adj, applied = A.adjust_for_spins(close, opens, spins)
    seeds = build.load_seeds(build.ROOT / "data/seeds")
    return adj, level, seeds, members, applied


def test_anchor_dates():
    _, level, *_ = _inputs()
    a = A.anchor_dates(level.index, pd.Timestamp("2026-10-21"))
    assert {k: v.date().isoformat() for k, v in a.items()} == {
        "1D": "2026-10-20", "1W": "2026-10-14", "1M": "2026-09-21", "YTD": "2025-12-31", "1Y": "2025-10-21"}


def test_spin_adjustment_only_where_feed_is_unadjusted():
    *_, applied = _inputs()
    assert [a["ticker"] for a in applied] == ["CMCSA"]
    assert applied[0]["factor"] == pytest.approx(27.77 / 29.54, abs=1e-6)


def test_reproduces_published_numbers(tmp_path):
    data = build.run(tmp_path, fixture=FIX, write_log=False, now=NOW, log_dir=tmp_path / "log")
    exp = json.loads((FIX / "expected.json").read_text())
    got = {(r["period"], r["ticker"]): r for r in data["contributions"]}
    for r in exp["contributions"]:
        g = got[(r["period"], r["ticker"])]
        assert g["points"] == pytest.approx(r["points"], abs=0.02), r
        assert g["stockReturnPct"] == pytest.approx(r["stockReturnPct"], abs=0.01), r
    gm = {r["period"]: r for r in data["index_moves"]}
    for r in exp["index_moves"]:
        assert gm[r["period"]]["change"] == pytest.approx(r["change"], abs=0.01)
        assert gm[r["period"]]["residual"] == pytest.approx(r["residual"], abs=0.05)
    assert (tmp_path / "index.html").read_text().count("__NX_DATA__") == 0


def test_daily_log_chains_to_same_answer():
    """Seeding a week earlier and logging the days since must give (almost) the same
    contributions as rolling back today's weights: the only gap is the index's own
    daily rounding against implied moves."""
    adj, level, seeds, members, _ = _inputs()
    T, S = pd.Timestamp("2026-10-21"), pd.Timestamp("2026-10-14")
    w21 = seeds[T]
    w14 = w21 * adj.loc[S, w21.index] / adj.loc[T, w21.index]
    seeds14 = {S: w14 / w14.sum()}
    days = [d for d in level.index if S < d <= T]
    log, idx, problems = A.log_days(adj, level, seeds14, days)
    assert not problems and len(idx) == len(days)
    a = A.attribute(adj, level, seeds14, log, members, T)
    b = A.attribute(adj, level, {T: w21}, pd.DataFrame(columns=["date", "ticker", "points"]), members, T)
    for k in ["1D", "1W", "1M", "YTD", "1Y"]:
        pa = {i["ticker"]: i["points"] for i in a[k]["items"]}
        for i in b[k]["items"]:
            assert pa[i["ticker"]] == pytest.approx(i["points"], abs=max(1.0, 0.01 * abs(i["points"]))), (k, i["ticker"])


def test_rerun_same_evening_replaces_latest_day():
    """A second run after the close recomputes the latest day with corrected prices
    instead of skipping it, and leaves no duplicate rows."""
    adj, level, seeds, _, _ = _inputs()
    T, S = pd.Timestamp("2026-10-21"), pd.Timestamp("2026-10-14")
    w = seeds[T] * adj.loc[S, seeds[T].index] / adj.loc[T, seeds[T].index]
    seeds14 = {S: w / w.sum()}
    days = [d for d in level.index if S < d <= T]
    log, idx, _ = A.log_days(adj, level, seeds14, days)

    adj2 = adj.copy()
    adj2.loc[T, "NVDA"] *= 1.01                      # a corrected close for the latest day
    kept, kept_idx = build.trim_log(log, idx, T)
    assert pd.to_datetime(kept["date"]).max() < T and len(kept_idx) == len(idx) - 1
    redo, redo_idx, problems = A.log_days(adj2, level, seeds14, [T])
    merged = pd.concat([kept, redo], ignore_index=True)
    assert not problems and len(merged) == len(log)
    pick = lambda df: df[(df["date"] == "2026-10-21") & (df["ticker"] == "NVDA")]["points"]
    assert len(pick(merged)) == 1 and pick(merged).iloc[0] > pick(log).iloc[0]


def test_latest_day_needs_member_prices():
    """Late in the evening Yahoo can return the index's bar without the members': that day
    is not used."""
    close, _, level = build.load_fixture(FIX)
    members = list(close.columns)
    assert build.usable_close(close, level, members) == pd.Timestamp("2026-10-21")
    close.loc[pd.Timestamp("2026-10-21")] = float("nan")
    assert build.usable_close(close, level, members) == pd.Timestamp("2026-10-20")


def test_never_publishes_older_than_the_log(tmp_path):
    fx = tmp_path / "fx"
    shutil.copytree(FIX, fx)
    close = pd.read_csv(fx / "close.csv", index_col=0, parse_dates=True)
    close.loc[pd.Timestamp("2026-10-21")] = float("nan")
    close.to_csv(fx / "close.csv")
    logd = tmp_path / "log"
    logd.mkdir()
    pd.DataFrame([{"date": "2026-10-21", "ticker": "NVDA", "points": 1.0}]).to_csv(logd / "daily_contrib.csv", index=False)
    with pytest.raises(build.NotPublished):
        build.run(tmp_path / "out", fixture=fx, write_log=False, now=NOW, log_dir=logd)
    assert not (tmp_path / "out" / "index.html").exists()


def test_fill_from_cache_skips_restated_history():
    idx = pd.to_datetime(["2026-10-20", "2026-10-21"])
    cache = pd.DataFrame({"AAA": [100.0, 101.0], "BBB": [50.0, 51.0], "^NDX": [1000.0, 1010.0]}, index=idx)
    fresh = pd.DataFrame({"AAA": [100.0, float("nan")], "BBB": [25.0, float("nan")]}, index=idx)  # BBB split 2:1
    level = pd.Series([1000.0, 1010.0], index=idx)
    close, lvl, filled = build.fill_from_cache(fresh, level, cache)
    assert close.loc[idx[1], "AAA"] == 101.0 and filled == ["AAA 2026-10-21"]
    assert pd.isna(close.loc[idx[1], "BBB"])


def test_missing_price_is_filled_from_earlier_run(tmp_path):
    """A price Yahoo drops on a later run is taken from the closes the earlier run saved,
    so the page doesn't raise a false alarm or show a wrong return."""
    logd = tmp_path / "log"
    build.run(tmp_path / "o1", fixture=FIX, write_log=True, now=NOW, log_dir=logd)
    assert (logd / "closes.csv").exists() and (tmp_path / "latest.json").exists()

    fx = tmp_path / "fx"
    shutil.copytree(FIX, fx)
    close = pd.read_csv(fx / "close.csv", index_col=0, parse_dates=True)
    close.loc[pd.Timestamp("2026-10-21"), "META"] = float("nan")
    close.to_csv(fx / "close.csv")
    d = build.run(tmp_path / "o2", fixture=fx, write_log=True, now=NOW, log_dir=logd)
    m = d["meta"][0]
    assert m["status"] == "ok" and m["filledFromCache"] == ["META 2026-10-21"]
    exp = {(r["period"], r["ticker"]): r for r in json.loads((FIX / "expected.json").read_text())["contributions"]}
    got = {(r["period"], r["ticker"]): r for r in d["contributions"]}
    for k in ["1D", "1W", "YTD"]:
        assert got[(k, "META")]["stockReturnPct"] == pytest.approx(exp[(k, "META")]["stockReturnPct"], abs=0.01)
