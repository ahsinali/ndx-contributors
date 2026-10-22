# Nasdaq-100 contributors

Which stocks drove the Nasdaq-100 over 1 day, 1 week, 1 month, year to date and 1 year:
index points, share of the index return, share of the move, the stock's own return and weight,
for the top 50 contributors.

**Live page:** https://ahsinali.github.io/ndx-contributors/ (`#p=YTD` etc. opens a period)
**Data:** https://ahsinali.github.io/ndx-contributors/data.json

## How it updates

`.github/workflows/update.yml` runs on weekdays at 20:35, 21:35 and 22:35 UTC, and on every push to
`main`. GitHub's schedule is in UTC with no daylight saving, so the first post-close run is 4:35pm ET
year-round (20:35 UTC in summer, 21:35 UTC in winter); later runs recompute the latest close to pick up
any price corrections. Each run:

1. Runs the engine tests (`tests/`, offline fixture of real Oct 2026 closes).
2. Pulls ~15 months of closes for every member plus `^NDX` from Yahoo Finance.
3. Recomputes the latest close and appends any newly closed days to `data/log/`, then commits it.
4. Rebuilds the page and publishes it to GitHub Pages.
5. Fails the `reconcile` job (so GitHub notifies you) if the contributions don't add up to the
   index move within 5 bp, or if prices are missing or stale. The page still publishes, with a
   "Check needed" banner.

Run it by hand from the Actions tab: **Update dashboard → Run workflow**.

## Method

- Daily contribution of member *i*: `I(t-1) × w_i(t-1) × r_i(t)`; a period is the sum over its days.
- From the log start (first seed, 2026-10-21) every day is logged with the weights in force that
  day, so periods chain exactly through rebalances and membership changes.
- Before that, contributions are estimated by rolling the seed weights back with each stock's own
  price: `points = V_i × R / (1 + R)` with `V_i` the stock's index points at the seed date. Exact
  absent rebalances; the "Other" line shows what it can't see.
- Members that joined inside a window count from the close before they joined.
- Spin-offs (`config/spins.json`): if the price feed left the parent unadjusted, its history is
  scaled by the ex-date opening gap.

## When the index changes

The daily check flags an addition, removal or rebalance (contributions stop adding up). Then:

1. Download current QQQ holdings (Invesco's QQQ page → Holdings → download) for the latest close.
2. `python -m ndx.seed holdings.csv --date YYYY-MM-DD`
3. Add any new member to `config/members.json` (`name`, `entered` = effective date).
4. Commit and push; the workflow rebuilds.

## Run locally (WSL / Linux)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m ndx.build --out public          # live data; writes data/log
python -m http.server -d public 8000      # open http://localhost:8000
```

## Layout

```
ndx/attribution.py   pure attribution engine (no I/O)
ndx/sources.py       Yahoo Finance fetch
ndx/build.py         run: fetch → log → attribute → render
ndx/seed.py          new weights seed from a holdings file
config/              members (names, join dates), spin-offs
data/seeds/          weights snapshots, one CSV per date
data/log/            daily contributions (written by the workflow)
site/template.html   the page; data is injected at build
```
