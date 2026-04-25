"""
Opportunity Context Features

Following the "missing teammates" pattern - features that capture situational
opportunity changes that the model can't infer from box scores alone.

Feature Groups:
1. Game Competitiveness (blowout vs competitive)
2. Team Strength Context (playing strong vs weak opponents)
3. Role Change Detection (recent usage shifts)

Expected: -0.10 to -0.20 RMSE improvement
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Opportunity Context Features - Testing")
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
print(f"   Games: {len(games):,} rows")

# ============================================================================
# OPPORTUNITY CONTEXT FEATURE GROUP 1: GAME COMPETITIVENESS
# ============================================================================

print("\n2. Computing game competitiveness features...")

# Get final score differential for each game
games["score_diff"] = abs(games["home_score"] - games["away_score"])
games["is_blowout"] = (games["score_diff"] >= 15).astype(int)  # 15+ point margin
games["is_close"] = (games["score_diff"] <= 5).astype(int)     # Within 5 points

print("   [1/3] Score differential history...")

# Create team-game view with score differentials
home_games = games[["game_id_int", "date", "home_team", "home_score", "away_score"]].copy()
home_games["team"] = home_games["home_team"]
home_games["team_score"] = home_games["home_score"]
home_games["opp_score"] = home_games["away_score"]
home_games["score_diff"] = home_games["home_score"] - home_games["away_score"]

away_games = games[["game_id_int", "date", "away_team", "away_score", "home_score"]].copy()
away_games["team"] = away_games["away_team"]
away_games["team_score"] = away_games["away_score"]
away_games["opp_score"] = away_games["home_score"]
away_games["score_diff"] = away_games["away_score"] - away_games["home_score"]

team_games = pd.concat([
    home_games[["game_id_int", "date", "team", "team_score", "opp_score", "score_diff"]],
    away_games[["game_id_int", "date", "team", "team_score", "opp_score", "score_diff"]]
], ignore_index=True)

team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)

# Rolling team performance metrics
team_games["team_score_diff_L10"] = (
    team_games.groupby("team")["score_diff"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

team_games["team_win_pct_L10"] = (
    team_games.groupby("team")["score_diff"]
    .transform(lambda x: (x.shift(1).rolling(10, min_periods=3).apply(lambda y: (y > 0).mean())))
)

# Merge into player logs
player_logs = player_logs.merge(
    team_games[["game_id_int", "team", "team_score_diff_L10", "team_win_pct_L10"]],
    left_on=["game_id_int", "TEAM_ABBREVIATION"],
    right_on=["game_id_int", "team"],
    how="left"
)

# Merge game-level competitiveness
player_logs = player_logs.merge(
    games[["game_id_int", "score_diff", "is_blowout", "is_close"]],
    on="game_id_int",
    how="left"
)

comp_feats = ["team_score_diff_L10", "team_win_pct_L10"]

print(f"      Added: {comp_feats}")

# ============================================================================
# OPPORTUNITY CONTEXT FEATURE GROUP 2: OPPONENT STRENGTH
# ============================================================================

print("   [2/3] Opponent strength context...")

# Get opponent for each game
opp_map = pd.concat([
    games[["game_id_int", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}),
    games[["game_id_int", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
], ignore_index=True)

player_logs = player_logs.merge(
    opp_map,
    left_on=["game_id_int", "TEAM_ABBREVIATION"],
    right_on=["game_id_int", "team"],
    how="left",
    suffixes=("", "_opp")
)

# Opponent strength (their rolling performance)
opp_strength = team_games[["game_id_int", "team", "team_score_diff_L10", "team_win_pct_L10"]].rename(
    columns={
        "team": "opponent",
        "team_score_diff_L10": "opp_strength_L10",
        "team_win_pct_L10": "opp_win_pct_L10"
    }
)

player_logs = player_logs.merge(
    opp_strength,
    on=["game_id_int", "opponent"],
    how="left"
)

# Strength differential (our team - opponent)
player_logs["strength_differential"] = (
    player_logs["team_score_diff_L10"] - player_logs["opp_strength_L10"]
)

opp_strength_feats = ["opp_strength_L10", "opp_win_pct_L10", "strength_differential"]

print(f"      Added: {opp_strength_feats}")

# ============================================================================
# OPPORTUNITY CONTEXT FEATURE GROUP 3: ROLE CHANGES
# ============================================================================

print("   [3/3] Role change detection...")

# Minutes-based role indicators
player_logs["player_MIN_L5"] = (
    player_logs.groupby("PLAYER_ID")["MIN"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
)

player_logs["player_MIN_L10"] = (
    player_logs.groupby("PLAYER_ID")["MIN"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
)

# Role indicators
player_logs["is_starter"] = (player_logs["player_MIN_L5"] >= 24).astype(int)

# Minutes trend (increasing vs decreasing role)
player_logs["min_trend"] = player_logs["player_MIN_L5"] - player_logs["player_MIN_L10"]

# Usage rate proxy (FGA share)
team_fga = player_logs.groupby(["game_id_int", "TEAM_ABBREVIATION"])["FGA"].transform("sum")
player_logs["usage_rate"] = player_logs["FGA"] / (team_fga + 0.1)

player_logs["usage_rate_L10"] = (
    player_logs.groupby("PLAYER_ID")["usage_rate"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
)

# Usage change (recent spike or drop)
player_logs["usage_rate_L3"] = (
    player_logs.groupby("PLAYER_ID")["usage_rate"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
)

player_logs["usage_change"] = player_logs["usage_rate_L3"] - player_logs["usage_rate_L10"]

role_feats = ["is_starter", "min_trend", "usage_rate_L10", "usage_change"]

print(f"      Added: {role_feats}")

# ============================================================================
# BASELINE FEATURES
# ============================================================================

print("\n3. Computing baseline features...")

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

# ============================================================================
# PREPARE DATASETS
# ============================================================================

print("\n4. Preparing datasets...")

df_clean = player_logs[player_logs["MIN"] >= 10].copy()
df_clean = df_clean.dropna(subset=baseline_feats).reset_index(drop=True)

print(f"   Clean dataset: {len(df_clean):,} rows")

is_train = df_clean["GAME_DATE"] < "2023-10-01"

print(f"   Train: {is_train.sum():,} rows")
print(f"   Test: {(~is_train).sum():,} rows")

# Feature sets
opportunity_feats = comp_feats + opp_strength_feats + role_feats
all_feats = baseline_feats + opportunity_feats

print(f"\n   Baseline features: {len(baseline_feats)}")
print(f"   Opportunity context features: {len(opportunity_feats)}")
print(f"   Total features: {len(all_feats)}")

# Coverage check
print(f"\n   Feature coverage:")
for feat in opportunity_feats:
    coverage = df_clean[feat].notna().mean() * 100
    print(f"      {feat}: {coverage:.1f}%")

# ============================================================================
# TRAIN MODELS
# ============================================================================

print("\n5. Training models...")

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

    print(f"\n{name}:")
    print(f"  Train RMSE: {train_rmse:.3f}")
    print(f"  Test RMSE:  {test_rmse:.3f}")

    return test_rmse

# Baseline
print("\n[Model 1/2] Baseline...")
baseline_rmse = train_multi_output(baseline_feats, "Baseline")

# With opportunity context
print("\n[Model 2/2] With opportunity context...")
opportunity_rmse = train_multi_output(all_feats, "Baseline + Opportunity Context")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

improvement = baseline_rmse - opportunity_rmse
pct = (improvement / baseline_rmse) * 100

print(f"\nBaseline Test RMSE:               {baseline_rmse:.3f}")
print(f"With Opportunity Context RMSE:    {opportunity_rmse:.3f}")
print(f"\nImprovement:  {improvement:+.3f} RMSE ({pct:+.2f}%)")
print(f"Target:       -0.10 to -0.20 RMSE")

if improvement >= 0.15:
    status = "[SUCCESS]"
    msg = "Strong improvement - target exceeded!"
elif improvement >= 0.10:
    status = "[GOOD]"
    msg = "Target achieved!"
elif improvement >= 0.05:
    status = "[MODERATE]"
    msg = "Moderate improvement"
else:
    status = "[MARGINAL]"
    msg = "Below expectations"

print(f"\nStatus: {status} {msg}")

# ============================================================================
# FEATURE BREAKDOWN
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE GROUP BREAKDOWN")
print("=" * 70)

# Test each group individually
print("\nTesting individual feature groups against baseline...\n")

groups = [
    ("Competitiveness", comp_feats),
    ("Opponent Strength", opp_strength_feats),
    ("Role Changes", role_feats),
]

for group_name, group_feats in groups:
    test_feats = baseline_feats + group_feats
    rmse = train_multi_output(test_feats, f"Baseline + {group_name}")
    delta = baseline_rmse - rmse
    print(f"   {group_name}: {delta:+.3f} RMSE")

print("\n" + "=" * 70)
print("Opportunity context features complete!")
print("=" * 70)
