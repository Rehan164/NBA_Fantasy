"""
Matchup History Features - Priority 1

Adds player-specific performance against each opponent.
Expected improvement: -0.15 to -0.25 RMSE

Key insight: Some players consistently perform better/worse against specific teams
due to coaching schemes, psychological factors, and playstyle compatibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Matchup History Features - Testing")
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

# Fantasy points
player_logs["FANTASY_PTS"] = (
    player_logs["PTS"] * 1.00 + player_logs["REB"] * 1.25 +
    player_logs["AST"] * 1.50 + player_logs["STL"] * 2.00 +
    player_logs["BLK"] * 2.00 + player_logs["TOV"] * -0.50 +
    player_logs["FG3M"] * 0.50
)

player_logs["FG_PCT"] = player_logs["FG_PCT"].fillna(0)
player_logs["FG3_PCT"] = player_logs["FG3_PCT"].fillna(0)

player_logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

print(f"   Player logs: {len(player_logs):,} rows")

# ============================================================================
# IDENTIFY OPPONENT FOR EACH GAME
# ============================================================================

print("\n2. Mapping opponents...")

# Create opponent lookup: for each (game_id, team), who is the opponent?
opp_map = pd.concat([
    games[["game_id_int", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}),
    games[["game_id_int", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
], ignore_index=True)

# Merge opponent into player logs
player_logs = player_logs.merge(
    opp_map,
    left_on=["game_id_int", "TEAM_ABBREVIATION"],
    right_on=["game_id_int", "team"],
    how="left"
)

print(f"   Mapped {player_logs['opponent'].notna().sum():,} games to opponents")
print(f"   Coverage: {player_logs['opponent'].notna().mean()*100:.1f}%")

# ============================================================================
# BASELINE FEATURES (must be computed FIRST on correct time series order)
# ============================================================================

print("\n3. Computing baseline features...")

PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN", "FANTASY_PTS"]
WINDOWS = [3, 5, 10]

baseline_feats = []

# Player rolling averages
for stat in PLAYER_STATS:
    for w in WINDOWS:
        col = f"player_{stat}_L{w}"
        player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))
        baseline_feats.append(col)

# Player overall average (for matchup differential later)
player_logs["player_overall_fp_L10"] = (
    player_logs.groupby("PLAYER_ID")["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
)

# Position features
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

# ============================================================================
# MATCHUP HISTORY FEATURES (computed AFTER baseline on re-sorted data)
# ============================================================================

print("\n4. Computing matchup history features...")

# For each player-opponent pair, compute rolling stats
# Sort by (PLAYER_ID, opponent, GAME_DATE) for matchup-specific rolling features
player_logs_matchup = player_logs.sort_values(["PLAYER_ID", "opponent", "GAME_DATE"]).reset_index(drop=True)

print("   [1/3] Player vs opponent FP history...")
# Rolling average of fantasy points vs this opponent (last 5 games)
player_logs_matchup["player_vs_opp_fp_L5"] = (
    player_logs_matchup.groupby(["PLAYER_ID", "opponent"])["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

print("   [2/3] Games played vs opponent...")
# Count of games vs this opponent (before current game)
player_logs_matchup["player_vs_opp_count"] = (
    player_logs_matchup.groupby(["PLAYER_ID", "opponent"]).cumcount()
)

print("   [3/3] Matchup differential...")
# Matchup differential = performance vs this opponent - overall average
player_logs_matchup["player_vs_opp_fp_diff"] = (
    player_logs_matchup["player_vs_opp_fp_L5"] - player_logs_matchup["player_overall_fp_L10"]
)

# Merge matchup features back to main dataframe (which is still sorted by PLAYER_ID, GAME_DATE)
matchup_cols = ["PLAYER_ID", "game_id_int", "player_vs_opp_fp_L5", "player_vs_opp_count", "player_vs_opp_fp_diff"]
player_logs = player_logs.merge(
    player_logs_matchup[matchup_cols],
    on=["PLAYER_ID", "game_id_int"],
    how="left"
)

# ============================================================================
# PREPARE DATASETS
# ============================================================================

print("\n5. Preparing datasets...")

# Filter valid rows
df_clean = player_logs[player_logs["MIN"] >= 10].copy()
df_clean = df_clean.dropna(subset=baseline_feats).reset_index(drop=True)

print(f"   Clean dataset: {len(df_clean):,} rows")

# Train/test split
is_train = df_clean["GAME_DATE"] < "2023-10-01"

print(f"   Train: {is_train.sum():,} rows")
print(f"   Test: {(~is_train).sum():,} rows")

# Feature sets
matchup_feats = ["player_vs_opp_fp_L5", "player_vs_opp_count", "player_vs_opp_fp_diff"]
all_feats = baseline_feats + matchup_feats

print(f"\n   Baseline features: {len(baseline_feats)}")
print(f"   Matchup features: {len(matchup_feats)}")
print(f"   Total features: {len(all_feats)}")

# Check coverage of matchup features
print(f"\n   Matchup feature coverage:")
for feat in matchup_feats:
    coverage = df_clean[feat].notna().mean() * 100
    print(f"      {feat}: {coverage:.1f}%")

# ============================================================================
# TRAIN MULTI-OUTPUT MODELS
# ============================================================================

print("\n6. Training models...")

DK_WEIGHTS = {
    "PTS": 1.00, "REB": 1.25, "AST": 1.50,
    "STL": 2.00, "BLK": 2.00, "TOV": -0.50, "FG3M": 0.50
}

def train_multi_output(feature_cols, name):
    """Train multi-output decomposition model"""
    # Fill NaN in matchup features with 0 (no matchup history available)
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

    print(f"\n{name}:")
    print(f"  Train RMSE: {train_rmse:.3f}")
    print(f"  Test RMSE:  {test_rmse:.3f}")

    return test_rmse

# Baseline
print("\n[Model 1/2] Baseline (player rolling + position)...")
baseline_rmse = train_multi_output(baseline_feats, "Baseline")

# With matchup history
print("\n[Model 2/2] With matchup history...")
matchup_rmse = train_multi_output(all_feats, "Baseline + Matchup History")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

improvement = baseline_rmse - matchup_rmse
pct = (improvement / baseline_rmse) * 100

print(f"\nBaseline Test RMSE:          {baseline_rmse:.3f}")
print(f"With Matchup History RMSE:   {matchup_rmse:.3f}")
print(f"\nImprovement:  {improvement:+.3f} RMSE ({pct:+.2f}%)")
print(f"Target:       -0.15 to -0.25 RMSE")

if improvement >= 0.15:
    status = "[SUCCESS]"
    msg = "Target achieved!"
elif improvement >= 0.10:
    status = "[GOOD]"
    msg = "Good progress, close to target"
elif improvement >= 0.05:
    status = "[MODERATE]"
    msg = "Moderate improvement"
else:
    status = "[MARGINAL]"
    msg = "Below expectations"

print(f"\nStatus: {status} {msg}")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

# Sample some interesting matchups
print("\nSample matchup histories (players with strong matchup effects):")

# Calculate matchup differential for recent test set
test_data = df_clean[~is_train].copy()
test_data["abs_diff"] = test_data["player_vs_opp_fp_diff"].abs()

# Top players with biggest matchup advantages/disadvantages
top_matchups = (test_data[test_data["player_vs_opp_count"] >= 3]
                .nlargest(10, "abs_diff")
                [["PLAYER_NAME", "opponent", "player_vs_opp_fp_L5",
                  "player_overall_fp_L10", "player_vs_opp_fp_diff",
                  "player_vs_opp_count"]])

print(top_matchups.to_string(index=False))

print("\n" + "=" * 70)
print("Matchup history feature implementation complete!")
print("=" * 70)
