# Data Collection Pipeline

Everything the notebook needs, how long it takes, and how to rerun.

## Files written to `data/`

| File | Source | Script | Runtime | Notes |
|---|---|---|---|---|
| `nba_player_game_logs.csv` | nba_api `LeagueGameLog` (player mode) | `collect_players.py` | ~20 min full, <1 min update | already collected |
| `nba_historical_games.csv` | nba_api `LeagueGameLog` (team mode) | `collect_data.py` | ~20 min full, <1 min update | already collected |
| `nba_vegas_odds.csv` | sportsbookreviewsonline.com + optional Kaggle | `collect_vegas.py` | ~1 min | covers 2007-08 through mid-2022-23 |
| `nba_player_info.csv` | nba_api `CommonPlayerInfo` | `collect_player_info.py` | ~35 min | one row per unique player |
| `nba_advanced_team_stats.csv` | nba_api `BoxScoreAdvancedV2` | `collect_advanced_box.py` | ~7 hours | per team-game (pace, off/def rating, TS%, etc.) |
| `nba_advanced_player_stats.csv` | nba_api `BoxScoreAdvancedV2` | `collect_advanced_box.py` | same run | per player-game (usage %, PIE, etc.) |

All collection scripts are checkpoint-safe - they append as they run and skip IDs that have already been fetched. You can Ctrl-C and resume.

## Running all collections

```bash
# fast — already done
python collect_data.py --update
python collect_players.py --update

# ~1 min
python collect_vegas.py

# ~35 min (run once, rerun only when new players enter the league)
python collect_player_info.py

# ~7 hours (run once, resume-safe, append --season 2015 to limit range)
python collect_advanced_box.py
```

## Vegas odds gap (important)

`sportsbookreviewsonline.com` stopped updating after ~mid-January 2023. For Vegas data from Jan 2023 onward (including our entire test set: 2023-24, 2024-25, 2025-26) the best free source is a Kaggle dataset:

**https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024**

To add it:
1. Sign in to Kaggle (free account)
2. Download the CSV
3. Put it at `data/raw/kaggle_nba_odds.csv`
4. Run `python collect_vegas.py --kaggle` (will merge Kaggle rows into `nba_vegas_odds.csv`)

Without the Kaggle data, Vegas features are NaN for the test period. The model (`HistGradientBoostingRegressor`) handles NaN natively, but the Vegas signal contributes only to training.

## How the notebook uses these files

Each file feeds one feature group in the "External Data & Advanced Features" section:

- `nba_vegas_odds.csv` -> `vegas_spread`, `vegas_total`, `vegas_implied_total`
- `nba_player_info.csv` -> `pos_G`, `pos_F`, `pos_C`, `height_in`, `years_experience`, then `dvp_L20` (derived)
- `nba_advanced_team_stats.csv` -> `adv_PACE_L10`, `adv_OFF_RATING_L10`, etc. + opponent versions

Each cell skips gracefully if its file isn't present, so the notebook runs even mid-collection.
