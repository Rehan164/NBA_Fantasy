"""
Incremental Phase 1 Testing

Tests adding Phase 1 features to the EXISTING best model (9.533 RMSE).
This script loads the notebook's feature set and adds Phase 1 improvements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Phase 1 Features - Incremental Testing")
print("Starting from best model: Multi-output decomposition (9.533 RMSE)")
print("=" * 70)

DATA_DIR = Path("data")

# Load data (same as notebook)
print("\nLoading data...")
player_logs = pd.read_csv(DATA_DIR / "nba_player_game_logs.csv")
games = pd.read_csv(DATA_DIR / "nba_historical_games.csv")
player_info = pd.read_csv(DATA_DIR / "nba_player_info.csv")

player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])
games["date"] = pd.to_datetime(games["date"])
player_logs["game_id_int"] = player_logs["GAME_ID"].astype(int)
games["game_id_int"] = games["game_id"].astype(int)
player_logs["MIN"] = pd.to_numeric(player_logs["MIN"], errors="coerce")

# Fantasy points
player_logs["FANTASY_PTS"] = (
    player_logs["PTS"] * 1.00 + player_logs["REB"] * 1.25 +
    player_logs["AST"] * 1.50 + player_logs["STL"] * 2.00 +
    player_logs["BLK"] * 2.00 + player_logs["TOV"] * -0.50 +
    player_logs["FG3M"] * 0.50
)

# Fill percentages
player_logs["FG_PCT"] = player_logs["FG_PCT"].fillna(0)
player_logs["FG3_PCT"] = player_logs["FG3_PCT"].fillna(0)
player_logs["FT_PCT"] = player_logs["FT_PCT"].fillna(0)

# Sort
player_logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

print(f"Player logs: {len(player_logs):,} rows")

# ============================================================================
# EXISTING BASELINE FEATURES (from notebook)
# ============================================================================

print("\n1. Computing existing baseline features...")

PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN", "FANTASY_PTS"]
WINDOWS = [3, 5, 10]

# Player rolling averages
existing_feats = []
for stat in PLAYER_STATS:
    for w in WINDOWS:
        col = f"player_{stat}_L{w}"
        player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))
        existing_feats.append(col)

print(f"   Player rolling: {len(existing_feats)} features")

# ============================================================================
# PHASE 1: NEW FEATURES
# ============================================================================

print("\n2. Adding Phase 1 features...")

phase1_new = []

# --- LAG1 Features ---
print("   [1/4] Lag1 (last game)")
for stat in ["FANTASY_PTS", "PTS", "MIN"]:  # Focus on most important
    col = f"player_{stat}_lag1"
    player_logs[col] = player_logs.groupby("PLAYER_ID")[stat].shift(1)
    phase1_new.append(col)

# --- Consistency Features ---
print("   [2/4] Consistency")
for stat in ["FANTASY_PTS", "PTS"]:
    # Std dev
    std_col = f"player_{stat}_std_L10"
    player_logs[std_col] = (player_logs.groupby("PLAYER_ID")[stat]
                            .transform(lambda x: x.shift(1).rolling(10, min_periods=5).std()))
    phase1_new.append(std_col)

    # Ceiling/floor
    ceiling = f"player_{stat}_ceiling_L10"
    floor = f"player_{stat}_floor_L10"
    player_logs[ceiling] = (player_logs.groupby("PLAYER_ID")[stat]
                            .transform(lambda x: x.shift(1).rolling(10, min_periods=5).max()))
    player_logs[floor] = (player_logs.groupby("PLAYER_ID")[stat]
                          .transform(lambda x: x.shift(1).rolling(10, min_periods=5).min()))
    phase1_new.append(ceiling)
    phase1_new.append(floor)

# --- Advanced Efficiency ---
print("   [3/4] Advanced efficiency")

# eFG%
player_logs["eFG_PCT"] = ((player_logs["FGM"] + 0.5 * player_logs["FG3M"]) /
                          (player_logs["FGA"] + 0.1)).fillna(0).clip(0, 1)

# TS%
player_logs["TS_PCT"] = (player_logs["PTS"] /
                         (2 * (player_logs["FGA"] + 0.44 * player_logs["FTA"]) + 0.1)).fillna(0).clip(0, 1)

# AST/TO
player_logs["AST_TO_RATIO"] = (player_logs["AST"] / (player_logs["TOV"] + 0.1)).fillna(0).clip(0, 20)

for stat in ["eFG_PCT", "TS_PCT", "AST_TO_RATIO"]:
    col = f"player_{stat}_L10"
    player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                        .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean()))
    phase1_new.append(col)

# --- Usage Shares ---
print("   [4/4] Usage shares")

# Team totals per game
team_totals = player_logs.groupby(["game_id_int", "TEAM_ABBREVIATION"]).agg({
    "FGA": "sum",
    "PTS": "sum"
}).reset_index().rename(columns={"FGA": "team_FGA", "PTS": "team_PTS"})

player_logs = player_logs.merge(team_totals, on=["game_id_int", "TEAM_ABBREVIATION"], how="left")

# Shares
player_logs["shot_share"] = player_logs["FGA"] / (player_logs["team_FGA"] + 0.1)
player_logs["scoring_share"] = player_logs["PTS"] / (player_logs["team_PTS"] + 0.1)

for stat in ["shot_share", "scoring_share"]:
    col = f"player_{stat}_L10"
    player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                        .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean()))
    phase1_new.append(col)

print(f"\n   New Phase 1 features: {len(phase1_new)}")

# ============================================================================
# POSITION FEATURES (from notebook)
# ============================================================================

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

position_feats = ["pos_G", "pos_F", "pos_C"]

# ============================================================================
# BUILD DATASETS
# ============================================================================

print("\n3. Building datasets...")

# Filter
df_clean = player_logs[player_logs["MIN"] >= 10].copy()
df_clean = df_clean.dropna(subset=existing_feats).reset_index(drop=True)

# Train/test split
is_train = df_clean["GAME_DATE"] < "2023-10-01"

# Baseline feature set (matches notebook's rolling baseline)
baseline_feats = existing_feats + position_feats

# Full Phase 1 feature set
all_feats = baseline_feats + phase1_new

print(f"   Baseline features: {len(baseline_feats)}")
print(f"   Phase 1 total features: {len(all_feats)}")
print(f"   Train rows: {is_train.sum():,}")
print(f"   Test rows: {(~is_train).sum():,}")

#=============================================================================
# MULTI-OUTPUT DECOMPOSITION (Best model from notebook)
# ============================================================================

print("\n4. Training multi-output decomposition models...")

DK_WEIGHTS = {
    "PTS": 1.00, "REB": 1.25, "AST": 1.50,
    "STL": 2.00, "BLK": 2.00, "TOV": -0.50, "FG3M": 0.50
}

def train_multi_output(feature_cols, name):
    """Train multi-output decomposition model"""
    X_tr = df_clean.loc[is_train, feature_cols]
    X_te = df_clean.loc[~is_train, feature_cols]

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

    print(f"\n{name}:")
    print(f"  Train RMSE: {train_rmse:.3f}")
    print(f"  Test RMSE:  {test_rmse:.3f}")

    return test_rmse

# Baseline
baseline_rmse = train_multi_output(baseline_feats, "Baseline (rolling + position)")

# Phase 1
phase1_rmse = train_multi_output(all_feats, "Phase 1 (+ lag1 + consistency + efficiency + usage)")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

improvement = baseline_rmse - phase1_rmse
pct = (improvement / baseline_rmse) * 100

print(f"\nBaseline Test RMSE:  {baseline_rmse:.3f}")
print(f"Phase 1 Test RMSE:   {phase1_rmse:.3f}")
print(f"\nImprovement:  {improvement:+.3f} RMSE ({pct:+.2f}%)")

if improvement >= 0.30:
    status = "[SUCCESS]"
elif improvement >= 0.10:
    status = "[GOOD]"
else:
    status = "[MARGINAL]"

print(f"\nStatus: {status}")
print(f"Target: -0.30 to -0.50 RMSE improvement")

print("\n" + "=" * 70)
