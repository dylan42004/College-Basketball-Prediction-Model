# College Basketball Spread Prediction Model

An end-to-end system that predicts the point spread of NCAA Division I men's basketball
games from tempo-free team ratings, validates the model walk-forward by season, and
backtests the predictions as a real betting strategy against DraftKings lines.

The model is trained on a strictly **pre-game** feature set — every input for a game is
built only from data available *before* tip-off — so the validation reflects how the
model would actually perform going forward, not a backward-looking fit.

## Results

Tracked live over the **2026 season (Jan 12 – Apr 6)**: 2,965 games evaluated, 420 bets
placed, ~482 units wagered. A *unit* is 1% of bankroll (see [Staking](#staking)).

| Metric | Value |
| --- | --- |
| Profit | **+23.25 units** |
| ROI | **+4.8%** |
| Model MAE | 8.90 pts |
| Market (closing line) MAE | 8.81 pts |
| Win rate — all games | 52.6% |
| Win rate — bets placed | 54.4% |
| Cumulative CLV — all games | +573 pts |
| Cumulative CLV — bets placed | +180.5 pts |
| % of bets beating closing line | 49.8% |

A few honest takeaways from these numbers:

- The model predicts game margins **about as accurately as the market's closing line**
  (8.90 vs 8.81 MAE). Matching the closing line from a from-scratch model is the bar that
  matters — the edge comes from finding spots where the model and the book disagree, not
  from being globally more accurate.
- Betting only the disagreements (the 420 placed bets) lifts the win rate from 52.6% to
  54.4% and produces a positive ROI, which is the result the whole pipeline is built to
  generate.
- CLV is positive in aggregate, but the **% of bets that beat the closing line is right at
  ~50%**. Over a single season this is a promising-but-not-yet-proven edge; see
  [Limitations](#limitations).

## How it works

The pipeline is four stages, one file each:

| Stage | File | What it does |
| --- | --- | --- |
| 1. Data | `historical_data.py` | Scrapes Barttorvik game logs, rebuilds shooting splits, and engineers leak-free pre-game features into one row per game. |
| 2. Model | `model.py` | Walk-forward validation by season across several model families, then fits and saves the final model. |
| 3. Predict | `predict.py` | Loads the saved model and predicts spreads for live games (a day's full slate or a single matchup). |
| 4. Backtest | `backtest.py` | Computes the edge vs the DK line for each historical bet and grid-searches staking strategies. |

### Methodology highlights

- **No look-ahead.** Pre-game features are season-to-date averages built with
  `cumsum().shift(1)`, so a game's features never include that game's own result.
- **Matchup differentials.** The ~30 raw team/opponent stats are collapsed into a handful
  of differential features (each team's offense measured against the *other's* defense),
  on the premise that the matchup matters more than either team's stats in a vacuum.
- **Walk-forward validation.** Models are trained on past seasons and tested on the next —
  an expanding window — rather than a random split that would leak future games into
  training.
- **Robust loss.** The final model is a `HuberRegressor`, chosen because scoring margins
  have fat tails (blowouts) that a squared-error loss would over-weight.

The final model lands at an out-of-sample MAE of roughly **8.9 points**, essentially even
with the market.

## Staking

Bets are sized with fractional-Kelly-style bankroll management:

- **1 unit = 1% of bankroll.**
- Stake scales with edge size. As of 2026-02-20 the tiers were: edge ≥ 2.5 → 1u,
  edge ≥ 4.0 → 2u.

"Edge" is the gap between the model's predicted spread and the DraftKings line at the time
of the bet — the bigger the disagreement, the larger the position.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`predict.py` drives headless Chrome via Selenium, so it also needs **Google Chrome**
installed locally. Selenium 4.6+ auto-manages the matching chromedriver, so there's
nothing else to install.

### Usage

Run all commands from the **repository root** — the scripts load `betting_tracker.csv`
and the saved `.pkl` artifacts by relative path, so the working directory needs to be the
root for those to resolve.

```bash
# Backtest staking strategies on the tracked bet history
python src/backtest.py betting_tracker.csv

# Predict a day's slate or a single matchup (interactive)
python src/predict.py

# Retrain the model from scratch and re-run model comparison
python src/model.py
```

## Repository structure

```
.
├── src/
│   ├── historical_data.py    # scrape + clean + feature engineering
│   ├── model.py              # walk-forward validation + final model
│   ├── predict.py            # live spread predictions
│   └── backtest.py           # betting-strategy backtest + optimization
├── betting_tracker.csv       # tracked bets (the data behind the results above)
├── huber_margin_model.pkl    # saved trained model
├── feature_cols_huber.pkl    # saved feature order for inference
├── requirements.txt
└── README.md
```

## Limitations

This is a personal research project, and the results above should be read with the usual
caveats:

- **The staking backtest optimizes in-sample.** `backtest.py` grid-searches edge
  thresholds, bet caps, and unit tiers and reports the most profitable combination *on the
  same history it searched over*. That winner is partly fit to noise; the honest reading is
  to treat the best config as a hypothesis to confirm on out-of-sample dates, not a
  guaranteed edge. The reported ROI is best understood as an upper bound.
- **Single-season sample.** ~420 placed bets over one season is a modest sample for a thin
  edge, and the ~50% closing-line-beating rate means the positive ROI isn't yet backed by
  consistent CLV. More seasons are needed to separate skill from variance.
- **The model roughly matches, rather than beats, the market.** The edge is in selective
  disagreement, which is inherently fragile and sensitive to line shopping and timing.
- **Injuries.** The model cannot accurately account for or predict injuries, so as a user
  you must manually check injury reports and decide how to change your approach. There
  were games that were not bet on because of injuries even if the margin recommended
  a bet.

## Notes

- The `.pkl` files are Python pickles, which execute code on load — only load model
  artifacts from a source you trust.
- In `betting_tracker.csv`, the predicted spread is always quoted on the side the model
  favors, and the DK line is quoted on the side that would be bet.

## Disclaimer

For research and educational purposes only. Nothing here is financial or betting advice,
and past performance does not predict future results.

