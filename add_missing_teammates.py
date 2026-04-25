"""
Missing Teammates Feature Implementation

Implements the breakthrough feature from the notebook that provided -0.249 RMSE.

The feature detects when key rotation players are absent by comparing:
- Actual rotation minutes in this game's box score (sum of player L10 averages)
- Expected rotation minutes (rolling baseline from previous games)
- Deficit = baseline - actual (proxy for injuries/rest)

When the deficit is high, remaining players get increased opportunity.

Expected: -0.20 to -0.25 RMSE improvement
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MISSING TEAMMATES FEATURE - Implementation & Testing")
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

player_logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

print(f"   Loaded: {len(player_logs):,} player-game rows")

# ============================================================================
# BASELINE FEATURES
# ============================================================================

print("\n2. Computing baseline features...")

PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN", "FANTASY_PTS"]
WINDOWS = [3, 5, 10]

baseline_feats = []

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

print(f"   Baseline: {len(baseline_feats)} features")

# ============================================================================
# MISSING TEAMMATES FEATURE (from notebook)
# ============================================================================

print("\n3. Computing MISSING TEAMMATES feature...")
print("   This is the breakthrough feature from the notebook!")

# Step 1: Each player's rolling L10 average minutes
print("   [1/4] Computing player MIN_L10...")
player_logs["MIN_L10"] = (player_logs.groupby("PLAYER_ID")["MIN"]
                          .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean()))

# Step 2: For each (team, game), sum L10 minutes of players in box score
print("   [2/4] Aggregating team rotation minutes...")
team_game_min = (player_logs.dropna(subset=["MIN_L10"])
                 .groupby(["game_id_int", "TEAM_ABBREVIATION"])["MIN_L10"]
                 .agg(["sum", "count"])
                 .reset_index()
                 .rename(columns={"sum": "team_l10_min_played", "count": "team_players_played"}))

# Step 3: Rolling baseline of expected rotation minutes
print("   [3/4] Computing baseline (expected rotation minutes)...")
team_game_min = team_game_min.merge(
    games[["game_id_int", "date"]], on="game_id_int", how="left"
).sort_values(["TEAM_ABBREVIATION", "date"]).reset_index(drop=True)

team_game_min["team_l10_min_baseline"] = (
    team_game_min.groupby("TEAM_ABBREVIATION")["team_l10_min_played"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# Step 4: Missing minute deficit (baseline - actual)
print("   [4/4] Computing missing minute deficit...")
team_game_min["missing_min_deficit"] = (
    team_game_min["team_l10_min_baseline"] - team_game_min["team_l10_min_played"]
)

# Merge back to player logs
player_logs = player_logs.merge(
    team_game_min[["game_id_int", "TEAM_ABBREVIATION",
                   "team_l10_min_played", "team_players_played", "missing_min_deficit"]],
    on=["game_id_int", "TEAM_ABBREVIATION"], how="left"
)

missing_teammates_feats = ["team_l10_min_played", "team_players_played", "missing_min_deficit"]

print(f"\n   Missing teammates: {len(missing_teammates_feats)} features")
print(f"   Features: {missing_teammates_feats}")

# Stats on the deficit
print(f"\n   Missing minute deficit statistics:")
print(f"      Mean: {player_logs['missing_min_deficit'].mean():.2f} minutes")
print(f"      Std:  {player_logs['missing_min_deficit'].std():.2f} minutes")
print(f"      Min:  {player_logs['missing_min_deficit'].min():.2f} minutes (extra players)")
print(f"      Max:  {player_logs['missing_min_deficit'].max():.2f} minutes (major absences)")

# ============================================================================
# COMPREHENSIVE FEATURE SET (from previous test)
# ============================================================================

print("\n4. Computing comprehensive feature set...")

# We'll add the key features from our comprehensive test
# Phase 1: Lag1
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

print(f"   Comprehensive features added: {len(lag1_feats + matchup_feats + home_away_feats)}")

# ============================================================================
# PREPARE DATASETS
# ============================================================================

print("\n5. Preparing datasets...")

df_clean = player_logs[player_logs["MIN"] >= 10].copy()
df_clean = df_clean.dropna(subset=baseline_feats).reset_index(drop=True)

is_train = df_clean["GAME_DATE"] < "2023-10-01"

print(f"   Train: {is_train.sum():,} rows")
print(f"   Test: {(~is_train).sum():,} rows")

# ============================================================================
# TRAIN MODELS - INCREMENTAL TESTING
# ============================================================================

print("\n6. Training models...")
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
    ("Baseline", baseline_feats),
    ("+ Missing Teammates ONLY", baseline_feats + missing_teammates_feats),
    ("+ Comprehensive (no missing)", baseline_feats + lag1_feats + matchup_feats + home_away_feats),
    ("+ ALL (comprehensive + missing)", baseline_feats + missing_teammates_feats + lag1_feats + matchup_feats + home_away_feats),
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

    print(f"{name:35s} | {len(feats):3d} feats | Test: {test_rmse:.3f}{delta_str}")
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
results_df["delta_vs_baseline"] = results_df["test_rmse"] - results_df.iloc[0]["test_rmse"]
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
print(f"Improvement vs Baseline: {-best['delta_vs_baseline']:+.3f} RMSE ({best['pct_improvement']:+.2f}%)")

# Compare to targets
print(f"\n" + "=" * 70)
print("COMPARISON TO TARGETS")
print("=" * 70)

notebook_best = 9.533  # From notebook's multi-output decomposition
our_best = best['test_rmse']
gap = our_best - notebook_best

print(f"\nNotebook's best model: {notebook_best:.3f} RMSE")
print(f"Our best model:        {our_best:.3f} RMSE")
print(f"Gap:                   {gap:+.3f} RMSE")

if gap <= 0.05:
    status = "EXCELLENT - Matched notebook!"
elif gap <= 0.15:
    status = "VERY GOOD - Close to notebook"
elif gap <= 0.30:
    status = "GOOD - Within range"
else:
    status = "NEEDS MORE WORK"

print(f"\nStatus: {status}")

# Missing teammates impact
missing_only_rmse = results_df[results_df['config'] == "+ Missing Teammates ONLY"]["test_rmse"].values[0]
missing_impact = baseline_test - missing_only_rmse

print(f"\n" + "=" * 70)
print("MISSING TEAMMATES FEATURE IMPACT")
print("=" * 70)
print(f"Baseline:                   {baseline_test:.3f} RMSE")
print(f"+ Missing Teammates:        {missing_only_rmse:.3f} RMSE")
print(f"Impact:                     {missing_impact:+.3f} RMSE")
print(f"Expected from notebook:     -0.249 RMSE")
print(f"Match:                      {abs(missing_impact - 0.249) < 0.05}")

print("\n" + "=" * 70)
print("Missing teammates feature implementation complete!")
print("=" * 70)
