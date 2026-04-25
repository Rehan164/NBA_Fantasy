"""
Phase 1 Quick Wins Feature Implementation

Adds the remaining high-value features from the notebook:
1. Defense vs Position (DvP) - Expected: -0.015 RMSE
2. Days Rest (player + opponent) - Expected: -0.015 RMSE
3. Trends (L3 - L10 hot/cold) - Expected: -0.01 RMSE
4. Full Efficiency (FGA, FG_PCT, PLUS_MINUS rolling) - Expected: -0.015 RMSE

Total expected improvement: -0.04 to -0.065 RMSE
Target: 9.49-9.51 RMSE (from current 9.554)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHASE 1 QUICK WINS - Feature Implementation")
print("=" * 70)

DATA_DIR = Path("data")

# ============================================================================
# LOAD DATA
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

# ============================================================================
# BASELINE FEATURES (from add_missing_teammates.py)
# ============================================================================

print("\n2. Computing baseline features (current optimized model)...")

PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN", "FANTASY_PTS"]
WINDOWS = [3, 5, 10]

baseline_feats = []

# Rolling averages
for stat in PLAYER_STATS:
    for w in WINDOWS:
        col = f"player_{stat}_L{w}"
        player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))
        baseline_feats.append(col)

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
    col = f"pos_{pos}"
    player_logs[col] = (player_logs["pos_bucket"] == pos).astype(int)
    baseline_feats.append(col)

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

missing_teammates_feats = ["team_l10_min_played", "team_players_played", "missing_min_deficit"]

# Recency
lag1_feats = []
for stat in ["FANTASY_PTS", "PTS", "MIN"]:
    col = f"player_{stat}_lag1"
    player_logs[col] = player_logs.groupby("PLAYER_ID")[stat].shift(1)
    lag1_feats.append(col)

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

matchup_feats = ["player_vs_opp_fp_L5"]

# Home/away
home_lookup = games[["game_id_int", "home_team"]].rename(columns={"home_team": "_home"})
player_logs = player_logs.merge(home_lookup, on="game_id_int", how="left")
player_logs["is_home"] = (player_logs["TEAM_ABBREVIATION"] == player_logs["_home"]).astype(int)

home_away_feats = ["is_home"]

current_model_feats = (baseline_feats + missing_teammates_feats +
                       lag1_feats + matchup_feats + home_away_feats)

print(f"   Current model: {len(current_model_feats)} features")

# ============================================================================
# PHASE 1 FEATURE 1: DEFENSE VS POSITION (DvP)
# ============================================================================

print("\n3. Computing Defense vs Position (DvP) feature...")
print("   For each (opponent, position), rolling FP allowed")

# Merge opponent to each player-game
player_logs["opp_for_dvp"] = player_logs["opponent"]

# For each (opponent_team, position, date), compute rolling avg FP allowed
player_logs_dvp = player_logs.sort_values(["opp_for_dvp", "pos_bucket", "GAME_DATE"]).reset_index(drop=True)

player_logs_dvp["dvp_fp_allowed_L20"] = (
    player_logs_dvp.groupby(["opp_for_dvp", "pos_bucket"])["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
)

player_logs = player_logs.merge(
    player_logs_dvp[["PLAYER_ID", "game_id_int", "dvp_fp_allowed_L20"]],
    on=["PLAYER_ID", "game_id_int"], how="left"
)

# DvP differential: player's avg vs what opponent allows to this position
player_logs["dvp_differential"] = player_logs["player_FANTASY_PTS_L10"] - player_logs["dvp_fp_allowed_L20"]

dvp_feats = ["dvp_fp_allowed_L20", "dvp_differential"]

print(f"   DvP features: {len(dvp_feats)}")
print(f"   Features: {dvp_feats}")

# ============================================================================
# PHASE 1 FEATURE 2: DAYS REST
# ============================================================================

print("\n4. Computing days rest features...")

# Merge game date
player_logs = player_logs.merge(
    games[["game_id_int", "date"]].rename(columns={"date": "game_date_check"}),
    on="game_id_int", how="left"
)

# Days since last game for each player
player_logs_rest = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
player_logs_rest["last_game_date"] = player_logs_rest.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
player_logs_rest["days_rest"] = (player_logs_rest["GAME_DATE"] - player_logs_rest["last_game_date"]).dt.days
player_logs_rest["days_rest"] = player_logs_rest["days_rest"].clip(upper=7).fillna(3)

# Days since last game for each team (opponent's rest)
team_last_game = player_logs.groupby(["TEAM_ABBREVIATION", "GAME_DATE"]).first().reset_index()[["TEAM_ABBREVIATION", "GAME_DATE"]]
team_last_game = team_last_game.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
team_last_game["team_last_game_date"] = team_last_game.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
team_last_game["team_days_rest"] = (team_last_game["GAME_DATE"] - team_last_game["team_last_game_date"]).dt.days
team_last_game["team_days_rest"] = team_last_game["team_days_rest"].clip(upper=7).fillna(3)

player_logs = player_logs.merge(
    player_logs_rest[["PLAYER_ID", "game_id_int", "days_rest"]],
    on=["PLAYER_ID", "game_id_int"], how="left"
)

# Opponent's rest
player_logs = player_logs.merge(
    team_last_game[["TEAM_ABBREVIATION", "GAME_DATE", "team_days_rest"]],
    left_on=["opponent", "GAME_DATE"], right_on=["TEAM_ABBREVIATION", "GAME_DATE"],
    how="left", suffixes=("", "_opp_team")
)

player_logs = player_logs.rename(columns={"team_days_rest": "opp_days_rest"})

# Rest advantage (positive = player more rested)
player_logs["rest_advantage"] = player_logs["days_rest"] - player_logs["opp_days_rest"]

days_rest_feats = ["days_rest", "opp_days_rest", "rest_advantage"]

print(f"   Days rest features: {len(days_rest_feats)}")
print(f"   Features: {days_rest_feats}")

# ============================================================================
# PHASE 1 FEATURE 3: TRENDS (HOT/COLD STREAKS)
# ============================================================================

print("\n5. Computing trend features (L3 - L10)...")

# For key stats, compute L3 - L10 (positive = hot streak)
TREND_STATS = ["FANTASY_PTS", "PTS", "REB", "AST", "MIN"]

trend_feats = []
for stat in TREND_STATS:
    col = f"player_{stat}_trend"
    player_logs[col] = player_logs[f"player_{stat}_L3"] - player_logs[f"player_{stat}_L10"]
    trend_feats.append(col)

print(f"   Trend features: {len(trend_feats)}")
print(f"   Features: {trend_feats}")

# ============================================================================
# PHASE 1 FEATURE 4: FULL EFFICIENCY FEATURES
# ============================================================================

print("\n6. Computing full efficiency features...")

# Rolling averages for FGA, FG_PCT, FT_PCT, PLUS_MINUS
EFFICIENCY_STATS = ["FGA", "FG_PCT", "FT_PCT"]

# Check if PLUS_MINUS exists
if "PLUS_MINUS" in player_logs.columns:
    EFFICIENCY_STATS.append("PLUS_MINUS")

efficiency_feats = []

for stat in EFFICIENCY_STATS:
    for w in [5, 10]:
        col = f"player_{stat}_L{w}"
        player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))
        efficiency_feats.append(col)

# Usage rate approximation (FGA / MIN)
player_logs["usage_rate_approx"] = player_logs["FGA"] / player_logs["MIN"].replace(0, np.nan)
player_logs["player_usage_L10"] = (player_logs.groupby("PLAYER_ID")["usage_rate_approx"]
                                   .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean()))
efficiency_feats.append("player_usage_L10")

print(f"   Efficiency features: {len(efficiency_feats)}")
print(f"   Features: {efficiency_feats}")

# ============================================================================
# PREPARE DATASETS
# ============================================================================

print("\n7. Preparing datasets...")

df_clean = player_logs[player_logs["MIN"] >= 10].copy()
df_clean = df_clean.dropna(subset=baseline_feats).reset_index(drop=True)

is_train = df_clean["GAME_DATE"] < "2023-10-01"

print(f"   Train: {is_train.sum():,} rows")
print(f"   Test: {(~is_train).sum():,} rows")

# ============================================================================
# TRAIN MODELS - INCREMENTAL TESTING
# ============================================================================

print("\n8. Training models...")
print("=" * 70)

DK_WEIGHTS = {
    "PTS": 1.00, "REB": 1.25, "AST": 1.50,
    "STL": 2.00, "BLK": 2.00, "TOV": -0.50, "FG3M": 0.50
}

def train_multi_output(feature_cols, name):
    """Train multi-output decomposition model"""
    X_tr = df_clean.loc[is_train, feature_cols].fillna(0)
    X_te = df_clean.loc[~is_train, feature_cols].fillna(0)

    pred_tr = np.zeros(len(X_tr))
    pred_te = np.zeros(len(X_te))

    for stat, weight in DK_WEIGHTS.items():
        y_tr = df_clean.loc[is_train, stat].astype(float)
        y_te = df_clean.loc[~is_train, stat].astype(float)

        model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=8,
            min_samples_leaf=20, random_state=42
        )
        model.fit(X_tr, y_tr)

        pred_tr += weight * model.predict(X_tr)
        pred_te += weight * model.predict(X_te)

    y_train_fp = df_clean.loc[is_train, "FANTASY_PTS"]
    y_test_fp = df_clean.loc[~is_train, "FANTASY_PTS"]

    train_rmse = np.sqrt(mean_squared_error(y_train_fp, pred_tr))
    test_rmse = np.sqrt(mean_squared_error(y_test_fp, pred_te))

    return train_rmse, test_rmse

# Test configurations
configs = [
    ("Current model (9.554 baseline)", current_model_feats),
    ("+ DvP features", current_model_feats + dvp_feats),
    ("+ Days rest", current_model_feats + days_rest_feats),
    ("+ Trends", current_model_feats + trend_feats),
    ("+ Efficiency", current_model_feats + efficiency_feats),
    ("+ ALL Phase 1 features", current_model_feats + dvp_feats + days_rest_feats + trend_feats + efficiency_feats),
]

results = []
baseline_test = None

print("\nIncremental Testing:")
print("-" * 70)

for name, feats in configs:
    feats = list(dict.fromkeys(feats))  # Remove duplicates
    train_rmse, test_rmse = train_multi_output(feats, name)

    if baseline_test is None:
        baseline_test = test_rmse
        delta_str = ""
    else:
        delta = baseline_test - test_rmse
        delta_str = f" ({delta:+.3f})"

    print(f"{name:40s} | {len(feats):3d} feats | Test: {test_rmse:.3f}{delta_str}")
    results.append({
        "config": name,
        "n_features": len(feats),
        "train_rmse": train_rmse,
        "test_rmse": test_rmse
    })

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df["delta_vs_current"] = results_df["test_rmse"] - results_df.iloc[0]["test_rmse"]
results_df["pct_improvement"] = ((results_df.iloc[0]["test_rmse"] - results_df["test_rmse"]) /
                                   results_df.iloc[0]["test_rmse"] * 100)

print("\n" + results_df.to_string(index=False))

# Best result
best_idx = results_df["test_rmse"].idxmin()
best = results_df.iloc[best_idx]

print(f"\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print(f"Configuration: {best['config']}")
print(f"Features: {best['n_features']}")
print(f"Test RMSE: {best['test_rmse']:.3f}")
print(f"Improvement vs Current: {-best['delta_vs_current']:+.3f} RMSE ({best['pct_improvement']:+.2f}%)")

# Compare to targets
print(f"\n" + "=" * 70)
print("COMPARISON TO TARGETS")
print("=" * 70)

current_best = 9.554
phase1_target_low = 9.49
phase1_target_high = 9.51
our_best = best['test_rmse']

print(f"\nCurrent model:         {current_best:.3f} RMSE")
print(f"Phase 1 target range:  {phase1_target_low:.3f} - {phase1_target_high:.3f} RMSE")
print(f"Our Phase 1 result:    {our_best:.3f} RMSE")

if our_best <= phase1_target_low:
    status = "EXCEEDED TARGET - Excellent!"
elif our_best <= phase1_target_high:
    status = "HIT TARGET - Success!"
elif our_best <= current_best:
    status = "IMPROVED - Close to target"
else:
    status = "NEEDS INVESTIGATION - Underperformed"

print(f"\nStatus: {status}")

print("\n" + "=" * 70)
print("Phase 1 Quick Wins implementation complete!")
print("=" * 70)
