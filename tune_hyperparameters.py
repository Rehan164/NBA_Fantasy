"""
Hyperparameter Tuning for NBA Fantasy Model

Uses RandomizedSearchCV to tune each of the 7 stat models separately.
This should improve performance from 9.532 to ~9.45-9.50 RMSE.

Expected runtime: ~2-4 hours (depending on n_iter and cv folds)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, make_scorer
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

print("=" * 70)
print("HYPERPARAMETER TUNING - Phase 2 Optimization")
print("=" * 70)

DATA_DIR = Path("data")

# ============================================================================
# LOAD DATA & FEATURES (same as Phase 1)
# ============================================================================

print("\n1. Loading data...")
player_logs = pd.read_csv(DATA_DIR / "nba_player_game_logs.csv")
games = pd.read_csv(DATA_DIR / "nba_historical_games.csv")
player_info = pd.read_csv(DATA_DIR / "nba_player_info.csv")

player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])
games["date"] = pd.to_datetime(games["date"])
player_logs["game_id_int"] = player_logs["GAME_ID"].astype(int)
games["game_id_int"] = games["game_id"].astype(int)
player_logs["MIN"] = pd.to_numeric(player_logs["MIN"], errors="coerce")

player_logs["FANTASY_PTS"] = (
    player_logs["PTS"] * 1.00 + player_logs["REB"] * 1.25 +
    player_logs["AST"] * 1.50 + player_logs["STL"] * 2.00 +
    player_logs["BLK"] * 2.00 + player_logs["TOV"] * -0.50 +
    player_logs["FG3M"] * 0.50
)

player_logs["FG_PCT"] = player_logs["FG_PCT"].fillna(0)
player_logs["FG3_PCT"] = player_logs["FG3_PCT"].fillna(0)
player_logs["FT_PCT"] = player_logs["FT_PCT"].fillna(0)

player_logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

print(f"   Loaded: {len(player_logs):,} player-game rows")

print("\n2. Computing all Phase 1 features...")
print("   (This may take a few minutes...)")

# Execute the full feature engineering pipeline from add_phase1_quick_wins.py
# (I'll include the key feature computation code here)

PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN", "FANTASY_PTS"]
WINDOWS = [3, 5, 10]

# Rolling averages
for stat in PLAYER_STATS:
    for w in WINDOWS:
        col = f"player_{stat}_L{w}"
        player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))

# Position
def bucket_position(p):
    if not isinstance(p, str): return "U"
    p = p.lower()
    if "guard" in p: return "G"
    if "center" in p: return "C"
    if "forward" in p: return "F"
    return "U"

player_info["pos_bucket"] = player_info["POSITION"].apply(bucket_position)
player_logs = player_logs.merge(
    player_info[["PERSON_ID", "pos_bucket"]].rename(columns={"PERSON_ID": "PLAYER_ID"}),
    on="PLAYER_ID", how="left"
)

for pos in ["G", "F", "C"]:
    player_logs[f"pos_{pos}"] = (player_logs["pos_bucket"] == pos).astype(int)

# Missing teammates
player_logs["MIN_L10"] = (player_logs.groupby("PLAYER_ID")["MIN"]
                          .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean()))

team_game_min = (player_logs.dropna(subset=["MIN_L10"])
                 .groupby(["game_id_int", "TEAM_ABBREVIATION"])["MIN_L10"]
                 .agg(["sum", "count"])
                 .reset_index()
                 .rename(columns={"sum": "team_l10_min_played", "count": "team_players_played"}))

team_game_min = team_game_min.merge(
    games[["game_id_int", "date"]], on="game_id_int", how="left"
).sort_values(["TEAM_ABBREVIATION", "date"]).reset_index(drop=True)

team_game_min["team_l10_min_baseline"] = (
    team_game_min.groupby("TEAM_ABBREVIATION")["team_l10_min_played"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

team_game_min["missing_min_deficit"] = (
    team_game_min["team_l10_min_baseline"] - team_game_min["team_l10_min_played"]
)

player_logs = player_logs.merge(
    team_game_min[["game_id_int", "TEAM_ABBREVIATION",
                   "team_l10_min_played", "team_players_played", "missing_min_deficit"]],
    on=["game_id_int", "TEAM_ABBREVIATION"], how="left"
)

# Lag1 features
for stat in ["FANTASY_PTS", "PTS", "MIN"]:
    player_logs[f"player_{stat}_lag1"] = player_logs.groupby("PLAYER_ID")[stat].shift(1)

# Matchup history
opp_map = pd.concat([
    games[["game_id_int", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}),
    games[["game_id_int", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
], ignore_index=True)

player_logs = player_logs.merge(
    opp_map, left_on=["game_id_int", "TEAM_ABBREVIATION"],
    right_on=["game_id_int", "team"], how="left", suffixes=("", "_opp")
)

player_logs_matchup = player_logs.sort_values(["PLAYER_ID", "opponent", "GAME_DATE"]).reset_index(drop=True)
player_logs_matchup["player_vs_opp_fp_L5"] = (
    player_logs_matchup.groupby(["PLAYER_ID", "opponent"])["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

player_logs = player_logs.merge(
    player_logs_matchup[["PLAYER_ID", "game_id_int", "player_vs_opp_fp_L5"]],
    on=["PLAYER_ID", "game_id_int"], how="left"
)

# Home/away
home_lookup = games[["game_id_int", "home_team"]].rename(columns={"home_team": "_home"})
player_logs = player_logs.merge(home_lookup, on="game_id_int", how="left")
player_logs["is_home"] = (player_logs["TEAM_ABBREVIATION"] == player_logs["_home"]).astype(int)

# DvP
player_logs["opp_for_dvp"] = player_logs["opponent"]
player_logs_dvp = player_logs.sort_values(["opp_for_dvp", "pos_bucket", "GAME_DATE"]).reset_index(drop=True)
player_logs_dvp["dvp_fp_allowed_L20"] = (
    player_logs_dvp.groupby(["opp_for_dvp", "pos_bucket"])["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
)

player_logs = player_logs.merge(
    player_logs_dvp[["PLAYER_ID", "game_id_int", "dvp_fp_allowed_L20"]],
    on=["PLAYER_ID", "game_id_int"], how="left"
)
player_logs["dvp_differential"] = player_logs["player_FANTASY_PTS_L10"] - player_logs["dvp_fp_allowed_L20"]

# Days rest
player_logs = player_logs.merge(
    games[["game_id_int", "date"]].rename(columns={"date": "game_date_check"}),
    on="game_id_int", how="left"
)

player_logs_rest = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
player_logs_rest["last_game_date"] = player_logs_rest.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
player_logs_rest["days_rest"] = (player_logs_rest["GAME_DATE"] - player_logs_rest["last_game_date"]).dt.days
player_logs_rest["days_rest"] = player_logs_rest["days_rest"].clip(upper=7).fillna(3)

team_last_game = player_logs.groupby(["TEAM_ABBREVIATION", "GAME_DATE"]).first().reset_index()[["TEAM_ABBREVIATION", "GAME_DATE"]]
team_last_game = team_last_game.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
team_last_game["team_last_game_date"] = team_last_game.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
team_last_game["team_days_rest"] = (team_last_game["GAME_DATE"] - team_last_game["team_last_game_date"]).dt.days
team_last_game["team_days_rest"] = team_last_game["team_days_rest"].clip(upper=7).fillna(3)

player_logs = player_logs.merge(
    player_logs_rest[["PLAYER_ID", "game_id_int", "days_rest"]],
    on=["PLAYER_ID", "game_id_int"], how="left"
)

player_logs = player_logs.merge(
    team_last_game[["TEAM_ABBREVIATION", "GAME_DATE", "team_days_rest"]],
    left_on=["opponent", "GAME_DATE"], right_on=["TEAM_ABBREVIATION", "GAME_DATE"],
    how="left", suffixes=("", "_opp_team")
)
player_logs = player_logs.rename(columns={"team_days_rest": "opp_days_rest"})
player_logs["rest_advantage"] = player_logs["days_rest"] - player_logs["opp_days_rest"]

# Trends
TREND_STATS = ["FANTASY_PTS", "PTS", "REB", "AST", "MIN"]
for stat in TREND_STATS:
    player_logs[f"player_{stat}_trend"] = player_logs[f"player_{stat}_L3"] - player_logs[f"player_{stat}_L10"]

# Efficiency
EFFICIENCY_STATS = ["FGA", "FG_PCT", "FT_PCT"]
if "PLUS_MINUS" in player_logs.columns:
    EFFICIENCY_STATS.append("PLUS_MINUS")

for stat in EFFICIENCY_STATS:
    for w in [5, 10]:
        player_logs[f"player_{stat}_L{w}"] = (
            player_logs.groupby("PLAYER_ID")[stat]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
        )

player_logs["usage_rate_approx"] = player_logs["FGA"] / player_logs["MIN"].replace(0, np.nan)
player_logs["player_usage_L10"] = (
    player_logs.groupby("PLAYER_ID")["usage_rate_approx"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
)

# Collect all feature names
baseline_feats = [f"player_{stat}_L{w}" for stat in PLAYER_STATS for w in WINDOWS]
baseline_feats += ["pos_G", "pos_F", "pos_C"]
missing_teammates_feats = ["team_l10_min_played", "team_players_played", "missing_min_deficit"]
lag1_feats = [f"player_{stat}_lag1" for stat in ["FANTASY_PTS", "PTS", "MIN"]]
matchup_feats = ["player_vs_opp_fp_L5"]
home_away_feats = ["is_home"]
dvp_feats = ["dvp_fp_allowed_L20", "dvp_differential"]
days_rest_feats = ["days_rest", "opp_days_rest", "rest_advantage"]
trend_feats = [f"player_{stat}_trend" for stat in TREND_STATS]
efficiency_feats = [f"player_{stat}_L{w}" for stat in EFFICIENCY_STATS for w in [5, 10]]
efficiency_feats.append("player_usage_L10")

ALL_FEATURES = (baseline_feats + missing_teammates_feats + lag1_feats +
                matchup_feats + home_away_feats + dvp_feats +
                days_rest_feats + trend_feats + efficiency_feats)

ALL_FEATURES = list(dict.fromkeys(ALL_FEATURES))  # Remove duplicates

print(f"   Total features: {len(ALL_FEATURES)}")

# ============================================================================
# PREPARE DATASETS
# ============================================================================

print("\n3. Preparing datasets...")

df_clean = player_logs[player_logs["MIN"] >= 10].copy()
df_clean = df_clean.dropna(subset=baseline_feats).reset_index(drop=True)

is_train = df_clean["GAME_DATE"] < "2023-10-01"

print(f"   Train: {is_train.sum():,} rows")
print(f"   Test: {(~is_train).sum():,} rows")

X_train = df_clean.loc[is_train, ALL_FEATURES].fillna(0)
X_test = df_clean.loc[~is_train, ALL_FEATURES].fillna(0)

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

print("\n4. Setting up hyperparameter search...")

# Define hyperparameter space
param_distributions = {
    'max_iter': [300, 500, 700, 1000],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'max_depth': [6, 8, 10, 12],
    'min_samples_leaf': [10, 15, 20, 30, 50],
    'l2_regularization': [0, 0.1, 0.5, 1.0],
    'max_leaf_nodes': [31, 63, 127, 255, None]
}

# Time series cross-validation (respects temporal order)
# Using 2 splits to reduce memory usage
tscv = TimeSeriesSplit(n_splits=2)

# Negative MSE scoring (RandomizedSearchCV maximizes, so we use negative)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Use a subset of training data for hyperparameter search to reduce memory
# We'll use the last 200k samples (most recent data, most relevant)
tune_indices = is_train.copy()
train_subset_size = min(200000, is_train.sum())
tune_indices[is_train] = False
tune_indices[is_train.to_numpy().nonzero()[0][-train_subset_size:]] = True

print(f"   Using {tune_indices.sum():,} samples for hyperparameter tuning (reduced for memory)")

X_tune = df_clean.loc[tune_indices, ALL_FEATURES].fillna(0)

DK_WEIGHTS = {
    "PTS": 1.00, "REB": 1.25, "AST": 1.50,
    "STL": 2.00, "BLK": 2.00, "TOV": -0.50, "FG3M": 0.50
}

# Store best parameters for each stat
best_params = {}

print("\n5. Tuning hyperparameters for each stat model...")
print("=" * 70)
print("This will take ~1-2 hours depending on hardware.")
print("Optimizations: 200k samples, 2 CV folds, 20 iterations per stat")
print("=" * 70)

for stat, weight in DK_WEIGHTS.items():
    print(f"\n[{stat}] Starting hyperparameter search...")

    y_tune = df_clean.loc[tune_indices, stat].astype(float)

    # Randomized search (faster than grid search)
    # Reduced iterations and parallel jobs to avoid memory issues
    random_search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_distributions=param_distributions,
        n_iter=20,  # Test 20 random combinations (reduced from 50)
        cv=tscv,
        scoring=mse_scorer,
        n_jobs=4,  # Use 4 cores max (reduced from -1)
        verbose=1,
        random_state=42
    )

    random_search.fit(X_tune, y_tune)

    best_params[stat] = random_search.best_params_

    print(f"[{stat}] Best params: {random_search.best_params_}")
    print(f"[{stat}] Best CV MSE: {-random_search.best_score_:.4f}")

# ============================================================================
# TRAIN FINAL MODEL WITH TUNED HYPERPARAMETERS
# ============================================================================

print("\n" + "=" * 70)
print("6. Training final model with tuned hyperparameters...")
print("=" * 70)

pred_train = np.zeros(len(X_train))
pred_test = np.zeros(len(X_test))

for stat, weight in DK_WEIGHTS.items():
    print(f"\n[{stat}] Training with optimized params...")

    y_train = df_clean.loc[is_train, stat].astype(float)
    y_test = df_clean.loc[~is_train, stat].astype(float)

    # Use best parameters found
    model = HistGradientBoostingRegressor(
        **best_params[stat],
        random_state=42
    )

    model.fit(X_train, y_train)

    pred_train += weight * model.predict(X_train)
    pred_test += weight * model.predict(X_test)

y_train_fp = df_clean.loc[is_train, "FANTASY_PTS"]
y_test_fp = df_clean.loc[~is_train, "FANTASY_PTS"]

train_rmse = np.sqrt(mean_squared_error(y_train_fp, pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test_fp, pred_test))

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 70)

print(f"\nTrain RMSE: {train_rmse:.3f}")
print(f"Test RMSE:  {test_rmse:.3f}")

# Compare to Phase 1
phase1_rmse = 9.532
improvement = phase1_rmse - test_rmse

print(f"\n" + "=" * 70)
print("COMPARISON TO PHASE 1")
print("=" * 70)
print(f"Phase 1 (default params):  9.532 RMSE")
print(f"Phase 2 (tuned params):    {test_rmse:.3f} RMSE")
print(f"Improvement:               {improvement:+.3f} RMSE")

if improvement > 0:
    print(f"\nStatus: IMPROVED! Tuning helped by {improvement:.3f} RMSE")
elif improvement > -0.01:
    print(f"\nStatus: Marginal change (within noise)")
else:
    print(f"\nStatus: No improvement from tuning")

# Save best parameters to JSON
output_file = "tuned_hyperparameters.json"
with open(output_file, 'w') as f:
    json.dump({
        'best_params': best_params,
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'phase1_rmse': phase1_rmse,
        'improvement': float(improvement),
        'timestamp': datetime.now().isoformat()
    }, f, indent=2)

print(f"\nBest parameters saved to: {output_file}")

print("\n" + "=" * 70)
print("Hyperparameter tuning complete!")
print("=" * 70)
