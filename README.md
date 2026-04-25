# NBA Fantasy Points Predictor

Predicts DraftKings fantasy points per player per game using rolling game history and derived features.

**Final model:** `HistGradientBoostingRegressor` with multi-output decomposition on 107 features. Test RMSE **9.533** (2.83% improvement over linear baseline) on the 2023-24 season.

For the full story of how we got here — what we tried, what worked, what didn't — see [`JOURNEY.md`](JOURNEY.md). For charts and visual explanations see [`feature_engineering_review.ipynb`](feature_engineering_review.ipynb).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Collecting data

The notebook reads CSV files from `data/`. These are not in the repo (too large) — you must collect them. All scripts are resume-safe (Ctrl-C and re-run is fine).

```bash
# Orchestrator - runs all collection scripts in order (~40 min total)
python collect_all.py

# Or run individually:
python collect_data.py          # ~1 min  - team game logs
python collect_players.py       # ~2 min  - player game logs
python collect_player_info.py   # ~35 min - player position, height, draft year
```

For incremental refresh (only the current season), pass `--update` to `collect_data.py` / `collect_players.py`.

## Running the model

```bash
jupyter notebook nba_fantasy_model.ipynb
```

The notebook loads whatever data files are present and gracefully skips feature groups whose source data isn't available yet, so you can run it mid-collection.

## Project structure

```
NBA_Fantasy/
├── nba_fantasy_model.ipynb           # Current model
├── feature_engineering_review.ipynb  # Charts + narrative for presentation
├── JOURNEY.md                        # Full chronological journey
├── README.md                         # This file
├── snapshots/                        # Timeline trace (v1-v5)
├── collect_all.py                    # Data collection orchestrator
├── collect_data.py                   # Team game logs
├── collect_players.py                # Player game logs
├── collect_player_info.py            # Player info (position, height, draft)
├── config.py                         # Paths + rate-limit settings
├── requirements.txt
├── project_description.pdf           # Assignment spec
└── data/                             # (gitignored) Output of collection scripts
```

## Snapshots

Each major modeling milestone is preserved in `snapshots/` as a runnable notebook:

| Snapshot | Test RMSE | What it shows |
|---|---|---|
| `v1_linear_baseline` | 9.811 | OLS on 240 raw lag features |
| `v2_feature_iterations` | 9.846 | Tested 5 feature sets on LinReg — no significant improvement |
| `v3_per36_rates` | 10.147 | Per-36 normalization on LinReg — worse, diagnosed why |
| `v4_random_forest` | 9.942 | RandomForest on rolling features — model class wasn't the bottleneck |
| `v5_external_features` | 9.527 | External data + missing teammates + HistGB + multi-output + ablation |

Current notebook is simplified from v5 (Vegas removed — only -0.009 RMSE marginal value didn't justify added complexity).
