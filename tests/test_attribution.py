"""Offline tests on a fixture of real closes around Oct 21, 2026.

    python -m pytest -q
"""
import datetime as dt
import json
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
    data = build.run(tmp_path, fixture=FIX, write_log=False, now=NOW)
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
