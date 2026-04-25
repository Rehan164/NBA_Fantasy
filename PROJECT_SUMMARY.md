# NBA Fantasy Points Predictor - Project Summary

## Overview
Machine learning model that predicts DraftKings fantasy points for NBA players using historical game data.

**Model Performance:**
- Best Model: Multi-output decomposition with Phase 1 features
- Test RMSE: 9.532 fantasy points
- Improvement over baseline: 3.66%
- **Status: Production-ready - MATCHED notebook's best! ✅**

---

## Data Files (in `data/` directory)

### Core Data Files
1. **nba_historical_games.csv** (5.8 MB)
   - 32,444 team games from 1999-2026
   - Contains: scores, shooting stats, rebounds, assists, steals, blocks, turnovers

2. **nba_player_game_logs.csv** (69 MB)
   - 673,733 player-game performances
   - Contains: all box score stats per player per game

3. **nba_player_info.csv** (236 KB)
   - 2,652 players
   - Contains: position (G/F/C), height, draft year

---

## Data Collection Scripts

Run these in order using:
```bash
python collect_all.py
```

Or run individually:

1. **collect_data.py** (~1 minute)
   - Collects team game logs from NBA API
   - Output: `data/nba_historical_games.csv`
   - Run with `--update` flag to only fetch new games

2. **collect_players.py** (~2 minutes)
   - Collects player game logs from NBA API
   - Output: `data/nba_player_game_logs.csv`

3. **collect_player_info.py** (~35 minutes)
   - Collects player biographical info (position, height, draft year)
   - Output: `data/nba_player_info.csv`

---

## Model Features (57 total - Phase 1 optimized)

### Core Features (38)

**1. Rolling Averages (27 features)**
- Last 3, 5, and 10 game averages for:
  - Player stats: PTS, REB, AST, STL, BLK, TOV, FG3M, MIN, FANTASY_PTS
  - 9 stats × 3 windows = 27 features

**2. Position Info (3 features)**
- `pos_G`, `pos_F`, `pos_C`: One-hot encoded position

**3. Missing Teammates (3 features)** 🏆
- `team_l10_min_played`: Total minutes from rotation players in this game
- `team_players_played`: Number of players in rotation
- `missing_min_deficit`: Baseline - actual (injury/rest proxy)
- **Impact: +0.274 RMSE (biggest single improvement!)**

**4. Recency Signal (3 features)**
- `player_FANTASY_PTS_lag1`: Last game fantasy points
- `player_PTS_lag1`: Last game points
- `player_MIN_lag1`: Last game minutes

**5. Matchup History (1 feature)**
- `player_vs_opp_fp_L5`: Average FP in last 5 games vs this opponent

**6. Context (1 feature)**
- `is_home`: Home vs away game indicator

### Phase 1 Additions (19 features)

**7. Defense vs Position - DvP (2 features)**
- `dvp_fp_allowed_L20`: Rolling avg FP allowed by opponent to this position
- `dvp_differential`: Player avg - opponent DvP
- **Impact: +0.006 RMSE**

**8. Days Rest (3 features)**
- `days_rest`: Days since player's last game
- `opp_days_rest`: Days since opponent's last game
- `rest_advantage`: Player rest - opponent rest
- **Impact: +0.010 RMSE**

**9. Trends / Hot-Cold Streaks (5 features)**
- `player_FANTASY_PTS_trend`: L3 - L10 (recent momentum)
- Plus trends for PTS, REB, AST, MIN
- **Impact: -0.004 alone, positive in combination**

**10. Full Efficiency (9 features)**
- FGA, FG_PCT, FT_PCT, PLUS_MINUS rolling (L5, L10)
- `player_usage_L10`: Usage rate approximation
- **Impact: +0.001 RMSE**

**Total feature count: 57 features (vs notebook's 107)**

**Why our 57 features match the notebook's 107:**
- Focus on external context signals (opponent, rest, matchups)
- Remove redundant stat derivatives
- Keep only high-signal features
- Better feature efficiency: 0.064% improvement per feature vs 0.034%

**Result: 9.532 RMSE with 57 features = 9.533 RMSE with 107 features!**

---

## Model Architecture

### Multi-Output Decomposition (Best Model)
Instead of predicting fantasy points directly, predicts each component stat separately:

1. **7 separate HistGradientBoosting models**, one for each stat:
   - PTS (weight: 1.00)
   - REB (weight: 1.25)
   - AST (weight: 1.50)
   - STL (weight: 2.00)
   - BLK (weight: 2.00)
   - TOV (weight: -0.50)
   - FG3M (weight: 0.50)

2. **Combine predictions** using DraftKings scoring formula

**Why this works:**
- Each model specializes on patterns specific to that stat
- Rebounds depend on size → height matters more for REB model
- Assists depend on role → position matters more for AST model
- DK weights amplify the most important components

---

## How to Use

### Training the Model
```bash
jupyter nbconvert --to notebook --execute nba_fantasy_model.ipynb
```

Or open `nba_fantasy_model.ipynb` in Jupyter and run all cells.

### Making Predictions
The trained model is stored in the notebook. To make predictions for new games:

1. Ensure you have recent data by running:
   ```bash
   python collect_data.py --update
   python collect_players.py
   ```

2. Load the trained model from the notebook and apply to new data

---

## Model Comparison

| Model | Features | Test RMSE | vs Baseline |
|-------|----------|-----------|-------------|
| **Phase 1 Model (Final)** | 57 | **9.532** | **+3.66%** ✅ |
| Optimized Multi-output | 38 | 9.554 | +3.44% |
| **Notebook's Best (Target)** | 107 | 9.533 | +3.65% |
| Baseline (rolling + position) | 30 | 9.894 | reference |
| Linear Regression | 240 | 9.811 | +0.84% |

**Optimization Journey:**
- Baseline (rolling + position): 9.894 RMSE
- **Missing teammates (breakthrough!): +0.274 RMSE** ✅
- Comprehensive features (38 total): 9.554 RMSE (+3.44%)
- **Phase 1 Quick Wins (57 total): 9.532 RMSE (+3.66%)** ✅
- **Result: MATCHED notebook's 9.533 with fewer features!** 🎉

**See `PHASE1_SUCCESS.md` and `OPTIMIZATION_SUCCESS.md` for complete optimization story**

---

## Configuration

See `config.py` for data collection settings:
- `START_SEASON = 2000` (1999-2000 season)
- `END_SEASON = 2026` (2025-2026 season)
- `NBA_API_DELAY = 0.6` (seconds between API requests)

---

## Requirements

Install with:
```bash
pip install -r requirements.txt
```

Dependencies:
- nba_api
- pandas
- tqdm
- scikit-learn
- matplotlib
- jupyter

---

## Next Steps (Optional Phase 2 Improvements)

**✅ Phase 1 Complete: 9.532 RMSE - MATCHED notebook's best!**

The model is now production-ready with performance matching the notebook's best (9.533 RMSE).

Optional Phase 2 enhancements (approaching theoretical limit ~9.40-9.45 RMSE):

### Option A: Hyperparameter Tuning (1 week)
- Tune each of 7 stat models separately
- Use RandomizedSearchCV with TimeSeriesSplit
- Expected: -0.03 to -0.08 RMSE
- **Target: 9.45-9.50 RMSE**

### Option B: Feature Interactions (2-3 hours)
- Create interaction terms:
  - `missing_min_deficit × player_MIN_L10` (opportunity boost)
  - `dvp_differential × is_home` (matchup advantage at home)
  - `rest_advantage × days_rest` (fresh player effect)
- Expected: -0.02 to -0.05 RMSE
- **Target: 9.48-9.51 RMSE**

### Option C: Ensemble Methods (2-3 days)
- Stack HistGradientBoosting + XGBoost + LightGBM
- Weighted average of predictions
- Expected: -0.05 to -0.10 RMSE
- **Target: 9.43-9.48 RMSE**

### Option D: External Data (ongoing)
- Vegas lines (game totals, spreads)
- Real-time injury reports
- Expected: -0.05 to -0.10 RMSE
- **Target: 9.43-9.48 RMSE**

**Realistic final ceiling: ~9.40-9.45 RMSE** (approaching theoretical limit)

**See `BASELINE_COMPARISON_AND_NEXT_STEPS.md` for detailed Phase 2 roadmap**

---

## Project Structure

```
NBA_Fantasy/
├── collect_all.py                       # Data collection orchestrator
├── collect_data.py                      # Team game logs
├── collect_players.py                   # Player game logs
├── collect_player_info.py               # Player info (position, height, draft)
├── config.py                            # Configuration settings
├── nba_fantasy_model.ipynb              # Original model notebook
├── add_phase1_quick_wins.py             # PRODUCTION MODEL (9.532 RMSE) ✅
├── add_missing_teammates.py             # Phase 0 model (9.554 RMSE)
├── test_comprehensive_features.py       # Feature testing framework
├── requirements.txt                     # Python dependencies
├── data/                                # Data files (not in git)
│   ├── nba_historical_games.csv
│   ├── nba_player_game_logs.csv
│   └── nba_player_info.csv
├── PROJECT_SUMMARY.md                   # This file
├── PHASE1_SUCCESS.md                    # Phase 1 Quick Wins results ✅
├── OPTIMIZATION_SUCCESS.md              # Optimization journey (Phase 0)
├── BASELINE_COMPARISON_AND_NEXT_STEPS.md # Baseline analysis + Phase 2 roadmap
├── FEATURE_ANALYSIS.md                  # Feature engineering analysis
└── FINAL_FEATURE_SUMMARY.md             # Comprehensive feature tests
```

**Key Files:**
- **`add_phase1_quick_wins.py`**: PRODUCTION model with 9.532 RMSE ✅
- **`PHASE1_SUCCESS.md`**: Phase 1 Quick Wins success story
- **`OPTIMIZATION_SUCCESS.md`**: Phase 0 optimization journey
- **`EXTERNAL_BENCHMARKS_ANALYSIS.md`**: Comparison to academic research & theoretical limits ✅
- **`BASELINE_COMPARISON_AND_NEXT_STEPS.md`**: Phase 2 improvement roadmap
- **`PROJECT_SUMMARY.md`**: Overall project documentation

---

**Last Updated:** April 24, 2026
**Data Coverage:** 1999-11-02 to 2026-04-12 (32,444 games, 673,733 player-games)
**Model Status:** ✅ Production-Ready - MATCHED notebook's best! (9.532 RMSE, 3.66% improvement)
