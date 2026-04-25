"""
Comprehensive Feature Test

Combines ALL features tested so far + new priority features:
- Baseline (rolling averages + position)
- Phase 1 (lag1, consistency, advanced efficiency, usage)
- Matchup history
- Opportunity context
- Home/Away splits (NEW)
- Pace features (NEW)

Tests individual groups and combinations to find optimal feature set.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("COMPREHENSIVE FEATURE TEST")
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
# PHASE 1 FEATURES
# ============================================================================

print("\n3. Computing Phase 1 features...")

phase1_feats = []

# Lag1
for stat in ["FANTASY_PTS", "PTS", "MIN"]:
    col = f"player_{stat}_lag1"
    player_logs[col] = player_logs.groupby("PLAYER_ID")[stat].shift(1)
    phase1_feats.append(col)

# Consistency
for stat in ["FANTASY_PTS", "PTS"]:
    std_col = f"player_{stat}_std_L10"
    player_logs[std_col] = (player_logs.groupby("PLAYER_ID")[stat]
                            .transform(lambda x: x.shift(1).rolling(10, min_periods=5).std()))
    phase1_feats.append(std_col)

# Advanced efficiency
player_logs["eFG_PCT"] = ((player_logs["FGM"] + 0.5 * player_logs["FG3M"]) /
                          (player_logs["FGA"] + 0.1)).fillna(0).clip(0, 1)
player_logs["TS_PCT"] = (player_logs["PTS"] /
                         (2 * (player_logs["FGA"] + 0.44 * player_logs["FTA"]) + 0.1)).fillna(0).clip(0, 1)

for stat in ["eFG_PCT", "TS_PCT"]:
    col = f"player_{stat}_L10"
    player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                        .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean()))
    phase1_feats.append(col)

# Usage shares
team_totals = player_logs.groupby(["game_id_int", "TEAM_ABBREVIATION"]).agg({
    "FGA": "sum", "PTS": "sum"
}).reset_index().rename(columns={"FGA": "team_FGA", "PTS": "team_PTS"})

player_logs = player_logs.merge(team_totals, on=["game_id_int", "TEAM_ABBREVIATION"], how="left")

player_logs["shot_share"] = player_logs["FGA"] / (player_logs["team_FGA"] + 0.1)
for stat in ["shot_share"]:
    col = f"player_{stat}_L10"
    player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                        .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean()))
    phase1_feats.append(col)

print(f"   Phase 1: {len(phase1_feats)} features")

# ============================================================================
# MATCHUP HISTORY FEATURES
# ============================================================================

print("\n4. Computing matchup history features...")

# Get opponent
opp_map = pd.concat([
    games[["game_id_int", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}),
    games[["game_id_int", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
], ignore_index=True)

player_logs = player_logs.merge(
    opp_map, left_on=["game_id_int", "TEAM_ABBREVIATION"],
    right_on=["game_id_int", "team"], how="left"
)

# Player overall average
player_logs["player_overall_fp_L10"] = (
    player_logs.groupby("PLAYER_ID")["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
)

# Matchup-specific rolling
player_logs_matchup = player_logs.sort_values(["PLAYER_ID", "opponent", "GAME_DATE"]).reset_index(drop=True)

player_logs_matchup["player_vs_opp_fp_L5"] = (
    player_logs_matchup.groupby(["PLAYER_ID", "opponent"])["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

player_logs_matchup["player_vs_opp_fp_diff"] = (
    player_logs_matchup["player_vs_opp_fp_L5"] - player_logs_matchup["player_overall_fp_L10"]
)

matchup_cols = ["PLAYER_ID", "game_id_int", "player_vs_opp_fp_L5", "player_vs_opp_fp_diff"]
player_logs = player_logs.merge(
    player_logs_matchup[matchup_cols],
    on=["PLAYER_ID", "game_id_int"], how="left"
)

matchup_feats = ["player_vs_opp_fp_L5", "player_vs_opp_fp_diff"]

print(f"   Matchup: {len(matchup_feats)} features")

# ============================================================================
# OPPORTUNITY CONTEXT FEATURES
# ============================================================================

print("\n5. Computing opportunity context features...")

# Team strength
home_games = games[["game_id_int", "date", "home_team", "home_score", "away_score"]].copy()
home_games["team"] = home_games["home_team"]
home_games["score_diff"] = home_games["home_score"] - home_games["away_score"]

away_games = games[["game_id_int", "date", "away_team", "away_score", "home_score"]].copy()
away_games["team"] = away_games["away_team"]
away_games["score_diff"] = away_games["away_score"] - away_games["home_score"]

team_games = pd.concat([
    home_games[["game_id_int", "date", "team", "score_diff"]],
    away_games[["game_id_int", "date", "team", "score_diff"]]
], ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)

team_games["team_score_diff_L10"] = (
    team_games.groupby("team")["score_diff"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

player_logs = player_logs.merge(
    team_games[["game_id_int", "team", "team_score_diff_L10"]],
    left_on=["game_id_int", "TEAM_ABBREVIATION"],
    right_on=["game_id_int", "team"], how="left", suffixes=("", "_y")
)

# Opponent strength
opp_strength = team_games[["game_id_int", "team", "team_score_diff_L10"]].rename(
    columns={"team": "opponent", "team_score_diff_L10": "opp_strength_L10"}
)

player_logs = player_logs.merge(
    opp_strength, on=["game_id_int", "opponent"], how="left"
)

player_logs["strength_differential"] = (
    player_logs["team_score_diff_L10"] - player_logs["opp_strength_L10"]
)

opportunity_feats = ["team_score_diff_L10", "opp_strength_L10", "strength_differential"]

print(f"   Opportunity: {len(opportunity_feats)} features")

# ============================================================================
# HOME/AWAY SPLITS (NEW)
# ============================================================================

print("\n6. Computing home/away splits...")

# Is home indicator
home_lookup = games[["game_id_int", "home_team"]].rename(columns={"home_team": "_home"})
player_logs = player_logs.merge(home_lookup, on="game_id_int", how="left")
player_logs["is_home"] = (player_logs["TEAM_ABBREVIATION"] == player_logs["_home"]).astype(int)

# Home/away performance splits
player_logs_home = player_logs[player_logs["is_home"] == 1].copy()
player_logs_away = player_logs[player_logs["is_home"] == 0].copy()

# Compute home averages
player_logs_home["player_home_fp_L10"] = (
    player_logs_home.groupby("PLAYER_ID")["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# Compute away averages
player_logs_away["player_away_fp_L10"] = (
    player_logs_away.groupby("PLAYER_ID")["FANTASY_PTS"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

# Merge back
player_logs = player_logs.merge(
    player_logs_home[["PLAYER_ID", "game_id_int", "player_home_fp_L10"]],
    on=["PLAYER_ID", "game_id_int"], how="left"
)
player_logs = player_logs.merge(
    player_logs_away[["PLAYER_ID", "game_id_int", "player_away_fp_L10"]],
    on=["PLAYER_ID", "game_id_int"], how="left"
)

# Fill with overall average when split not available
player_logs["player_home_fp_L10"] = player_logs["player_home_fp_L10"].fillna(player_logs["player_overall_fp_L10"])
player_logs["player_away_fp_L10"] = player_logs["player_away_fp_L10"].fillna(player_logs["player_overall_fp_L10"])

# Home/away differential
player_logs["player_home_away_diff"] = (
    player_logs["player_home_fp_L10"] - player_logs["player_away_fp_L10"]
)

home_away_feats = ["is_home", "player_home_fp_L10", "player_away_fp_L10", "player_home_away_diff"]

print(f"   Home/Away: {len(home_away_feats)} features")

# ============================================================================
# PACE FEATURES (NEW)
# ============================================================================

print("\n7. Computing pace features...")

# Approximate pace (possessions) from box score
home_pace = games[["game_id_int", "date", "home_team",
                    "home_fg_att", "home_ft_att", "home_oreb", "home_tov"]].copy()
home_pace["team"] = home_pace["home_team"]
home_pace["pace"] = home_pace["home_fg_att"] + 0.44 * home_pace["home_ft_att"] - home_pace["home_oreb"] + home_pace["home_tov"]

away_pace = games[["game_id_int", "date", "away_team",
                    "away_fg_att", "away_ft_att", "away_oreb", "away_tov"]].copy()
away_pace["team"] = away_pace["away_team"]
away_pace["pace"] = away_pace["away_fg_att"] + 0.44 * away_pace["away_ft_att"] - away_pace["away_oreb"] + away_pace["away_tov"]

team_pace = pd.concat([
    home_pace[["game_id_int", "date", "team", "pace"]],
    away_pace[["game_id_int", "date", "team", "pace"]]
], ignore_index=True).sort_values(["team", "date"]).reset_index(drop=True)

team_pace["team_pace_L10"] = (
    team_pace.groupby("team")["pace"]
    .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
)

player_logs = player_logs.merge(
    team_pace[["game_id_int", "team", "team_pace_L10"]],
    left_on=["game_id_int", "TEAM_ABBREVIATION"],
    right_on=["game_id_int", "team"], how="left", suffixes=("", "_pace")
)

# Opponent pace
opp_pace = team_pace[["game_id_int", "team", "team_pace_L10"]].rename(
    columns={"team": "opponent", "team_pace_L10": "opp_pace_L10"}
)

player_logs = player_logs.merge(
    opp_pace, on=["game_id_int", "opponent"], how="left"
)

player_logs["pace_differential"] = player_logs["team_pace_L10"] - player_logs["opp_pace_L10"]

pace_feats = ["team_pace_L10", "opp_pace_L10", "pace_differential"]

print(f"   Pace: {len(pace_feats)} features")

# ============================================================================
# PREPARE DATASETS
# ============================================================================

print("\n8. Preparing datasets...")

df_clean = player_logs[player_logs["MIN"] >= 10].copy()
df_clean = df_clean.dropna(subset=baseline_feats).reset_index(drop=True)

is_train = df_clean["GAME_DATE"] < "2023-10-01"

print(f"   Train: {is_train.sum():,} rows")
print(f"   Test: {(~is_train).sum():,} rows")

# ============================================================================
# TRAIN MODELS - INCREMENTAL TESTING
# ============================================================================

print("\n9. Training models (incremental testing)...")
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

# Test configurations incrementally
configs = [
    ("Baseline", baseline_feats),
    ("+ Phase 1", baseline_feats + phase1_feats),
    ("+ Matchup", baseline_feats + matchup_feats),
    ("+ Opportunity", baseline_feats + opportunity_feats),
    ("+ Home/Away", baseline_feats + home_away_feats),
    ("+ Pace", baseline_feats + pace_feats),
    ("All New Features", baseline_feats + phase1_feats + matchup_feats + opportunity_feats + home_away_feats + pace_feats),
]

results = []
baseline_test = None

for name, feats in configs:
    feats = list(dict.fromkeys(feats))  # Remove duplicates
    train_rmse, test_rmse = train_multi_output(feats, name)

    if baseline_test is None:
        baseline_test = test_rmse
        delta_str = ""
    else:
        delta = baseline_test - test_rmse
        delta_str = f" ({delta:+.3f})"

    print(f"{name:25s} | {len(feats):3d} feats | Train: {train_rmse:.3f} | Test: {test_rmse:.3f}{delta_str}")
    results.append({"config": name, "n_features": len(feats), "train_rmse": train_rmse, "test_rmse": test_rmse})

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE RESULTS SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df["delta_vs_baseline"] = results_df["test_rmse"] - results_df.iloc[0]["test_rmse"]
results_df["pct_improvement"] = ((results_df.iloc[0]["test_rmse"] - results_df["test_rmse"]) /
                                   results_df.iloc[0]["test_rmse"] * 100)

print("\n" + results_df.to_string(index=False))

best_idx = results_df["test_rmse"].idxmin()
best_config = results_df.iloc[best_idx]

print(f"\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)
print(f"Config: {best_config['config']}")
print(f"Features: {best_config['n_features']}")
print(f"Test RMSE: {best_config['test_rmse']:.3f}")
print(f"Improvement vs Baseline: {-best_config['delta_vs_baseline']:+.3f} RMSE ({best_config['pct_improvement']:+.2f}%)")

print("\n" + "=" * 70)
print("Comprehensive feature test complete!")
print("=" * 70)
