# NBA Fantasy Points Predictor

Predicts DraftKings fantasy points per player per game using rolling game history and derived features.

**Final model:** `HistGradientBoostingRegressor` with multi-output decomposition on 107 features. Test RMSE ~9.55 on the 2023-24 season (compared to ~9.62 for a linear regression baseline).

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Pipeline

The project is organized in five phases:

1. **Collection** -- `collect_data.py`, `collect_players.py`, `collect_player_info.py` pull from `nba_api` into `data/*.csv`.
2. **Processing** -- `process_data.py` (imported as a module) loads the raw CSVs, parses dates, normalizes types, and computes the DK fantasy points target.
3. **Feature engineering** -- `build_features.py` produces `data/nba_features.csv` (rolling player/team/opponent stats, missing teammates, schedule density, position, DvP) plus a `data/nba_features_manifest.json` manifest of feature groups.
4. **Modeling** -- `nba_fantasy_model.ipynb` trains the final HistGradientBoosting multi-output model on the feature CSV.
5. **Tuning** -- `optuna_nba_fantasy_study.py` runs the Optuna hyperparameter study; `optuna_nba_fantasy_study_v2.py` is an alternate methodology using TimeSeriesSplit cross-validation.

## Running it end-to-end

```bash
python collect_all.py            # phases 1-3 (~40 min if data is empty)
jupyter notebook nba_fantasy_model.ipynb
python optuna_nba_fantasy_study.py
```

For incremental refresh of just the current season, pass `--update` to `collect_data.py` and `collect_players.py` individually.

## Project structure

```
NBA_Fantasy/
├── nba_fantasy_model.ipynb          # final model (load -> train -> evaluate)
├── collect_data.py                  # team game logs
├── collect_players.py               # player game logs
├── collect_player_info.py           # player position, height, draft
├── process_data.py                  # processing module
├── build_features.py                # feature engineering script
├── collect_all.py                   # orchestrator (phases 1-3)
├── optuna_nba_fantasy_study.py      # Optuna study (single fold, 100 trials)
├── optuna_nba_fantasy_study_v2.py   # Optuna study (TimeSeriesSplit, 120 trials)
├── config.py                        # paths + season range + API delay
├── requirements.txt
├── project_description.pdf
└── data/                            # gitignored
    ├── nba_historical_games.csv
    ├── nba_player_game_logs.csv
    ├── nba_player_info.csv
    ├── nba_features.csv             # produced by build_features.py
    └── nba_features_manifest.json   # produced by build_features.py
```
