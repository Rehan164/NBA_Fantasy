# NBA Fantasy Points Predictor

Predicts DraftKings fantasy points per player per game using rolling game history and derived features.

Final model: `HistGradientBoostingRegressor` with multi-output decomposition on ~95 features. Test RMSE ~9.55 (2.7% improvement over linear baseline) on 2023-24 season.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Collecting data

The notebook reads CSV files from `data/`. These are not in the repo (too large) - you must collect them. All scripts are resume-safe (Ctrl-C and re-run is fine).

```bash
# Orchestrator - runs all collection scripts in order (~8 hours total)
python collect_all.py

# Or run individually:
python collect_data.py          # ~20 min - team game logs
python collect_players.py       # ~20 min - player game logs
python collect_player_info.py   # ~35 min - player position, height, draft year
python collect_advanced_box.py  # ~7 hours - pace, off/def rating, TS%, USG%
```

If you already have everything collected, use the `--update` flag on `collect_data.py` / `collect_players.py` for fast incremental refresh of just the current season.

### Rate limits

The NBA API rate-limits hard after a few thousand requests. If `collect_advanced_box.py` starts returning empty responses (`'resultSet'` errors), stop, wait 24 hours, and resume. The script skips already-fetched game IDs automatically.

## Running the model

```bash
jupyter notebook nba_fantasy_model.ipynb
```

The notebook loads whatever data files are present and gracefully skips feature groups whose source data isn't available yet. So you can run it mid-collection.

## Project structure

```
NBA_Fantasy/
├── nba_fantasy_model.ipynb    # Main modeling notebook
├── collect_all.py             # Orchestrator for all data collection
├── collect_data.py            # Team game logs
├── collect_players.py         # Player game logs
├── collect_player_info.py     # Position, height, draft year
├── collect_advanced_box.py    # Pace, off/def rating, TS%, USG%
├── config.py                  # Paths and rate-limit settings
├── data_collection.md         # Pipeline docs
├── features.md                # Feature descriptions for the team
├── requirements.txt
├── snapshots/                 # Versioned model snapshots
└── data/                      # (gitignored) Output of collection scripts
```

## Feature groups

See `features.md` for the full list. Most impactful (by ablation):

1. **Missing teammates** (-0.249 RMSE) - derived from player_logs, no external data needed
2. **Player position + DvP** (-0.022 RMSE combined) - from CommonPlayerInfo
3. **Efficiency rolling** (-0.018 RMSE) - FGA/FG_PCT/PLUS_MINUS L3/L5/L10

## Modeling approach

1. **Temporal train/test split:** games before 2023-10-01 are train, rest is test
2. **Time-series cross-validation** for model selection (no leakage from future folds)
3. **HistGradientBoostingRegressor** as the main model - natively handles NaN in optional feature groups
4. **Multi-output decomposition** - predict PTS/REB/AST/STL/BLK/TOV/FG3M separately and combine via DK formula

## Snapshots

Major milestones in `snapshots/`:

- `v1_linear_baseline` - Lag 1-10 features, OLS. Test RMSE 9.811.
- `v2_feature_iterations` - Tested 5 feature sets. No significant improvement.
- `v3_per36_rates` - Per-36 rate normalization. Worse than v1.
- `v4_random_forest` - RandomForest. Marginal improvement.
- `v5_external_features` - Added Vegas/position/advanced box hooks + multi-output + ablation. Best test RMSE 9.527 with Vegas data present.
- current notebook - simplified from v5: dropped Vegas (not enough marginal value for added complexity).
