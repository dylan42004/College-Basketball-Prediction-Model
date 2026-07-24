# College Basketball Spread Prediction Model

An end-to-end system that predicts the point spread of NCAA Division I men's basketball games from tempo-free team ratings, validates the model walk-forward by season, and stress-tests its predictions against live market prices (DraftKings closing lines).

The project is a study in a hard question: **can a from-scratch statistical model match — and selectively disagree with — a liquid, fast-resolving prediction market?** Every input for a game is built only from data available before tip-off, so the validation reflects genuine forward performance rather than a backward-looking fit.

---

## Methodology

The core of the project is the modeling discipline, not the profit-and-loss backtest that follows from it.

- **No look-ahead.** Pre-game features are season-to-date averages built with `cumsum().shift(1)`, so a game's features never include that game's own result. This is enforced at the feature-engineering stage, not patched afterward.
- **Matchup differentials.** The ~30 raw team/opponent stats are collapsed into a handful of differential features — each team's offense measured against the other's defense — on the premise that the matchup matters more than either team's stats in a vacuum.
- **Walk-forward validation.** Models are trained on past seasons and tested on the next, in an expanding window, rather than a random split that would leak future games into training. This is the validation design that determines whether the reported accuracy is real.
- **Robust loss.** The final model is a `HuberRegressor`, chosen because scoring margins have fat tails (blowouts) that a squared-error loss would over-weight.

Several model families were compared under this framework (see `model.py`); the Huber model was selected on out-of-sample performance.

---

## Forecasting accuracy

The headline result is how closely the model forecasts game margins relative to the market's closing line — the benchmark that actually matters.

| Metric | Value |
| --- | --- |
| **Model MAE** | **8.90 pts** |
| **Market (closing line) MAE** | **8.81 pts** |
| Out-of-sample validation | Walk-forward, by season |

A from-scratch model landing within ~0.1 points of the closing line is the bar that matters. The market aggregates thousands of participants; matching it is the hard part, and it means the model is a credible estimate of the "true" margin rather than noise. Beating the market globally is not the goal — the objective is to find the specific games where the model and the book disagree enough to be worth acting on.

---

## Strategy backtest

Treating model-vs-market disagreements as positions, and sizing them by the size of the disagreement, produces the following over the tracked 2026 season (Jan 12 – Apr 6): 2,965 games evaluated, 420 positions taken.

> **Read this as exploratory, not proven.** The staking parameters are optimized in-sample and the sample is a single season (n = 420). Treat the ROI below as an *upper bound* and a hypothesis to confirm out-of-sample — not a demonstrated edge. See [Limitations](#limitations).

| Metric | Value |
| --- | --- |
| Return on capital | +4.8% (upper bound — see note above) |
| Hit rate — disagreement positions | 54.4% |
| Hit rate — all games | 52.6% |
| Closing-line value (aggregate) | Positive |
| % of positions beating the closing line | 49.8% |

Selecting only the disagreements lifts the hit rate from 52.6% to 54.4%, which is the mechanism the whole pipeline is built around. But the ~50% closing-line-beating rate means the positive return is **not yet backed by consistent line-beating** over this sample — the honest reading is a promising signal that needs more seasons to separate from variance.

---

## How it works

Four stages, one file each:

| Stage | File | What it does |
| --- | --- | --- |
| 1. Data | `historical_data.py` | Scrapes Barttorvik game logs, rebuilds shooting splits, and engineers leak-free pre-game features into one row per game. |
| 2. Model | `model.py` | Walk-forward validation by season across several model families, then fits and saves the final model. |
| 3. Predict | `predict.py` | Loads the saved model and predicts spreads for live games (a full slate or a single matchup). |
| 4. Backtest | `backtest.py` | Computes the edge vs the market line for each historical game and grid-searches position-sizing strategies. |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`predict.py` drives headless Chrome via Selenium, so it also needs Google Chrome installed locally. Selenium 4.6+ auto-manages the matching chromedriver, so there's nothing else to install.

## Usage

Run all commands from the repository root — the scripts load the tracked history and the saved `.pkl` artifacts by relative path.

```bash
# Backtest position-sizing strategies on the tracked history
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
│   └── backtest.py           # strategy backtest + optimization
├── betting_tracker.csv       # tracked positions (the data behind the results above)
├── huber_margin_model.pkl    # saved trained model
├── feature_cols_huber.pkl    # saved feature order for inference
├── requirements.txt
└── README.md
```

---

## Limitations

This is a personal research project, and the results should be read with the usual caveats:

- **The strategy backtest optimizes in-sample.** `backtest.py` grid-searches thresholds, caps, and sizing tiers and reports the most profitable combination on the same history it searched over. That winner is partly fit to noise; the best config is a hypothesis to confirm on out-of-sample dates, not a guaranteed edge. The reported return is best understood as an upper bound.
- **Single-season sample.** ~420 positions over one season is modest for a thin edge, and the ~50% closing-line-beating rate means the positive return isn't yet backed by consistent closing-line value. More seasons are needed to separate skill from variance.
- **The model roughly matches, rather than beats, the market.** The value is in selective disagreement, which is inherently fragile and sensitive to timing.
- **Injuries.** The model does not account for injuries; late roster news must be checked manually, and some model-favored games are correctly skipped for this reason.

## Notes

- The `.pkl` files are Python pickles, which execute code on load — only load model artifacts from a source you trust.
- In `betting_tracker.csv`, the predicted spread is quoted on the side the model favors, and the market line is quoted on the side that would be taken.

## Disclaimer

For research and educational purposes only. Nothing here is financial or betting advice, and past performance does not predict future results.
