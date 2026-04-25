"""
Phase 1 Feature Engineering - Quick Wins

Adds high-impact features identified in FEATURE_ANALYSIS.md:
1. Lag1 features (direct last-game performance)
2. Usage proxy features (shot share, minute share, scoring share)
3. Consistency features (std dev, CV, ceiling, floor)
4. Advanced efficiency metrics (eFG%, TS%, AST/TO ratio)
5. Pace differential features

Expected improvement: -0.30 to -0.50 RMSE
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print("=" * 70)
print("Phase 1 Feature Engineering - Quick Wins")
print("=" * 70)

DATA_DIR = Path("data")

# Load data
print("\n1. Loading data...")
player_logs = pd.read_csv(DATA_DIR / "nba_player_game_logs.csv")
games = pd.read_csv(DATA_DIR / "nba_historical_games.csv")
player_info = pd.read_csv(DATA_DIR / "nba_player_info.csv")

player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])
games["date"] = pd.to_datetime(games["date"])
player_logs["game_id_int"] = player_logs["GAME_ID"].astype(int)
games["game_id_int"] = games["game_id"].astype(int)
player_logs["MIN"] = pd.to_numeric(player_logs["MIN"], errors="coerce")

# Calculate fantasy points
player_logs["FANTASY_PTS"] = (
    player_logs["PTS"] * 1.00 +
    player_logs["REB"] * 1.25 +
    player_logs["AST"] * 1.50 +
    player_logs["STL"] * 2.00 +
    player_logs["BLK"] * 2.00 +
    player_logs["TOV"] * -0.50 +
    player_logs["FG3M"] * 0.50
)

# Fill NaN percentages
player_logs["FG_PCT"] = player_logs["FG_PCT"].fillna(0)
player_logs["FG3_PCT"] = player_logs["FG3_PCT"].fillna(0)
player_logs["FT_PCT"] = player_logs["FT_PCT"].fillna(0)

# Sort for time series operations
player_logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

print(f"   Player logs: {len(player_logs):,} rows")
print(f"   Games: {len(games):,} rows")
print(f"   Players: {len(player_info):,} rows")

# ============================================================================
# PHASE 1 FEATURES
# ============================================================================

PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN", "FANTASY_PTS"]
WINDOWS = [3, 5, 10]

print("\n2. Computing Phase 1 features...")

# ----------------------------------------------------------------------------
# Feature Group 1: Lag1 (Last Game Performance)
# ----------------------------------------------------------------------------
print("   [1/6] Lag1 features (last game)")
lag1_cols = []
for stat in PLAYER_STATS:
    col = f"player_{stat}_lag1"
    player_logs[col] = player_logs.groupby("PLAYER_ID")[stat].shift(1)
    lag1_cols.append(col)

# ----------------------------------------------------------------------------
# Feature Group 2: Rolling Averages (existing baseline)
# ----------------------------------------------------------------------------
print("   [2/6] Rolling averages (L3, L5, L10)")
roll_cols = []
for stat in PLAYER_STATS:
    for w in WINDOWS:
        col = f"player_{stat}_L{w}"
        player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))
        roll_cols.append(col)

# ----------------------------------------------------------------------------
# Feature Group 3: Consistency Features (std, CV, ceiling, floor)
# ----------------------------------------------------------------------------
print("   [3/6] Consistency features (std, CV, ceiling, floor)")
consistency_cols = []

# Standard deviation over L10
for stat in ["FANTASY_PTS", "PTS", "MIN"]:
    col = f"player_{stat}_std_L10"
    player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                       .transform(lambda x: x.shift(1).rolling(10, min_periods=5).std()))
    consistency_cols.append(col)

# Coefficient of variation (std / mean) - measures relative consistency
for stat in ["FANTASY_PTS", "PTS"]:
    mean_col = f"player_{stat}_L10"
    std_col = f"player_{stat}_std_L10"
    cv_col = f"player_{stat}_cv_L10"
    player_logs[cv_col] = player_logs[std_col] / (player_logs[mean_col] + 0.1)  # +0.1 to avoid div by zero
    consistency_cols.append(cv_col)

# Ceiling (max) and floor (min) over L10
for stat in ["FANTASY_PTS", "PTS"]:
    ceiling_col = f"player_{stat}_ceiling_L10"
    floor_col = f"player_{stat}_floor_L10"
    player_logs[ceiling_col] = (player_logs.groupby("PLAYER_ID")[stat]
                               .transform(lambda x: x.shift(1).rolling(10, min_periods=5).max()))
    player_logs[floor_col] = (player_logs.groupby("PLAYER_ID")[stat]
                             .transform(lambda x: x.shift(1).rolling(10, min_periods=5).min()))
    consistency_cols.append(ceiling_col)
    consistency_cols.append(floor_col)

# ----------------------------------------------------------------------------
# Feature Group 4: Advanced Efficiency Metrics
# ----------------------------------------------------------------------------
print("   [4/6] Advanced efficiency (eFG%, TS%, AST/TO ratio)")
efficiency_cols = []

# Effective Field Goal % = (FGM + 0.5 * FG3M) / FGA
player_logs["eFG_PCT"] = (player_logs["FGM"] + 0.5 * player_logs["FG3M"]) / (player_logs["FGA"] + 0.1)
player_logs["eFG_PCT"] = player_logs["eFG_PCT"].fillna(0).clip(0, 1)

# True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
player_logs["TS_PCT"] = player_logs["PTS"] / (2 * (player_logs["FGA"] + 0.44 * player_logs["FTA"]) + 0.1)
player_logs["TS_PCT"] = player_logs["TS_PCT"].fillna(0).clip(0, 1)

# AST/TO ratio
player_logs["AST_TO_RATIO"] = player_logs["AST"] / (player_logs["TOV"] + 0.1)
player_logs["AST_TO_RATIO"] = player_logs["AST_TO_RATIO"].fillna(0).clip(0, 20)

# Rolling averages of efficiency metrics
for stat in ["eFG_PCT", "TS_PCT", "AST_TO_RATIO"]:
    for w in WINDOWS:
        col = f"player_{stat}_L{w}"
        player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))
        efficiency_cols.append(col)

# Existing efficiency features
for stat in ["FGA", "FG_PCT", "PLUS_MINUS"]:
    for w in WINDOWS:
        col = f"player_{stat}_L{w}"
        if col not in player_logs.columns:
            player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                               .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))
        efficiency_cols.append(col)

# ----------------------------------------------------------------------------
# Feature Group 5: Usage Proxy Features
# ----------------------------------------------------------------------------
print("   [5/6] Usage proxy features (shot/min/scoring share)")

# Build team-game aggregates
team_games_usage = player_logs.groupby(["game_id_int", "TEAM_ABBREVIATION"]).agg({
    "FGA": "sum",
    "MIN": "sum",
    "PTS": "sum",
}).reset_index()
team_games_usage.columns = ["game_id_int", "TEAM_ABBREVIATION", "team_FGA", "team_MIN", "team_PTS"]

# Merge team totals back to player logs
player_logs = player_logs.merge(
    team_games_usage,
    on=["game_id_int", "TEAM_ABBREVIATION"],
    how="left"
)

# Calculate usage shares
player_logs["shot_share"] = player_logs["FGA"] / (player_logs["team_FGA"] + 0.1)
player_logs["min_share"] = player_logs["MIN"] / (player_logs["team_MIN"] / 5 + 0.1)  # normalize by 5 players
player_logs["scoring_share"] = player_logs["PTS"] / (player_logs["team_PTS"] + 0.1)

# Rolling averages of usage shares
usage_cols = []
for stat in ["shot_share", "min_share", "scoring_share"]:
    for w in WINDOWS:
        col = f"player_{stat}_L{w}"
        player_logs[col] = (player_logs.groupby("PLAYER_ID")[stat]
                           .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean()))
        usage_cols.append(col)

# ----------------------------------------------------------------------------
# Feature Group 6: Pace Features
# ----------------------------------------------------------------------------
print("   [6/6] Pace features")

# Team pace = possessions per game (approximation using box score)
# Possessions ≈ FGA + 0.44*FTA - ORB + TOV
# Using team games table from before

home_games = games[["game_id_int", "date", "home_team",
                     "home_fg_att", "home_ft_att", "home_oreb", "home_tov"]].copy()
home_games.columns = ["game_id_int", "date", "team", "fga", "fta", "oreb", "tov"]

away_games = games[["game_id_int", "date", "away_team",
                     "away_fg_att", "away_ft_att", "away_oreb", "away_tov"]].copy()
away_games.columns = ["game_id_int", "date", "team", "fga", "fta", "oreb", "tov"]

team_games = pd.concat([home_games, away_games], ignore_index=True)
team_games["pace"] = team_games["fga"] + 0.44 * team_games["fta"] - team_games["oreb"] + team_games["tov"]
team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)

# Rolling pace
team_games["team_pace_L10"] = (team_games.groupby("team")["pace"]
                               .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean()))

# Get opponent info
opp_map = pd.concat([
    games[["game_id_int", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}),
    games[["game_id_int", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"})
], ignore_index=True)

team_games = team_games.merge(opp_map, on=["game_id_int", "team"], how="left")

# Opponent pace lookup
opp_pace = team_games[["game_id_int", "team", "team_pace_L10"]].rename(
    columns={"team": "opponent", "team_pace_L10": "opp_pace_L10"}
)
team_games = team_games.merge(opp_pace, on=["game_id_int", "opponent"], how="left")

# Pace differential
team_games["pace_differential"] = team_games["team_pace_L10"] - team_games["opp_pace_L10"]

pace_cols = ["team_pace_L10", "opp_pace_L10", "pace_differential"]

# Merge pace features into player logs
player_logs = player_logs.merge(
    team_games[["game_id_int", "team"] + pace_cols],
    left_on=["game_id_int", "TEAM_ABBREVIATION"],
    right_on=["game_id_int", "team"],
    how="left"
)

# ============================================================================
# BUILD FEATURE MATRIX
# ============================================================================

print("\n3. Building feature matrix...")

# Context features (existing)
home_lookup = games[["game_id_int", "home_team"]].copy()
home_lookup.columns = ["game_id_int", "_home"]
player_logs = player_logs.merge(home_lookup, on="game_id_int", how="left")
player_logs["is_home"] = (player_logs["TEAM_ABBREVIATION"] == player_logs["_home"]).astype(int)

# Days rest
team_sched = (player_logs[["TEAM_ABBREVIATION", "GAME_DATE", "game_id_int"]]
              .drop_duplicates(subset=["TEAM_ABBREVIATION", "game_id_int"])
              .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
              .reset_index(drop=True))
team_sched["days_rest"] = (team_sched.groupby("TEAM_ABBREVIATION")["GAME_DATE"]
                           .diff().dt.days.clip(upper=7).fillna(3).astype(int))
player_logs = player_logs.merge(
    team_sched[["game_id_int", "TEAM_ABBREVIATION", "days_rest"]],
    on=["game_id_int", "TEAM_ABBREVIATION"], how="left"
)

context_cols = ["is_home", "days_rest"]

# Position features
def bucket_position(p):
    if not isinstance(p, str):
        return "U"
    p = p.lower()
    if "guard" in p:   return "G"
    if "center" in p:  return "C"
    if "forward" in p: return "F"
    return "U"

player_info["pos_bucket"] = player_info["POSITION"].apply(bucket_position)

def height_to_inches(h):
    if not isinstance(h, str) or "-" not in h:
        return np.nan
    ft, inch = h.split("-")
    return int(ft) * 12 + int(inch)

player_info["height_in"] = player_info["HEIGHT"].apply(height_to_inches)
player_info["draft_year"] = pd.to_numeric(player_info["DRAFT_YEAR"], errors="coerce")

player_logs = player_logs.merge(
    player_info[["PERSON_ID", "pos_bucket", "height_in", "draft_year"]].rename(
        columns={"PERSON_ID": "PLAYER_ID"}),
    on="PLAYER_ID", how="left"
)

for pos in ["G", "F", "C"]:
    player_logs[f"pos_{pos}"] = (player_logs["pos_bucket"] == pos).astype(int)

player_logs["years_experience"] = player_logs["GAME_DATE"].dt.year - player_logs["draft_year"]

position_cols = ["pos_G", "pos_F", "pos_C", "height_in", "years_experience"]

# Assemble all Phase 1 features
phase1_cols = (
    lag1_cols +              # 9 features
    roll_cols +              # 27 features (9 stats × 3 windows)
    consistency_cols +       # 10 features
    efficiency_cols +        # 18 features
    usage_cols +             # 9 features
    pace_cols +              # 3 features
    context_cols +           # 2 features
    position_cols            # 5 features
)

print(f"\n   Total Phase 1 features: {len(phase1_cols)}")
print(f"   - Lag1: {len(lag1_cols)}")
print(f"   - Rolling avgs: {len(roll_cols)}")
print(f"   - Consistency: {len(consistency_cols)}")
print(f"   - Efficiency: {len(efficiency_cols)}")
print(f"   - Usage: {len(usage_cols)}")
print(f"   - Pace: {len(pace_cols)}")
print(f"   - Context: {len(context_cols)}")
print(f"   - Position: {len(position_cols)}")

# Filter valid rows
df_clean = player_logs[player_logs["MIN"] >= 10].copy()
required_features = lag1_cols + roll_cols
df_clean = df_clean.dropna(subset=required_features).reset_index(drop=True)

print(f"\n   Clean dataset: {len(df_clean):,} rows")

# Train/test split
is_train = df_clean["GAME_DATE"] < "2023-10-01"
X_train = df_clean.loc[is_train, phase1_cols]
X_test = df_clean.loc[~is_train, phase1_cols]
y_train = df_clean.loc[is_train, "FANTASY_PTS"]
y_test = df_clean.loc[~is_train, "FANTASY_PTS"]

print(f"   Train: {len(X_train):,} rows")
print(f"   Test: {len(X_test):,} rows")

# ============================================================================
# BASELINE MODEL (Rolling Only)
# ============================================================================

print("\n4. Training baseline model (rolling features only)...")
baseline_cols = roll_cols + context_cols + position_cols

X_train_base = df_clean.loc[is_train, baseline_cols]
X_test_base = df_clean.loc[~is_train, baseline_cols]

model_base = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)
model_base.fit(X_train_base, y_train)

train_rmse_base = np.sqrt(mean_squared_error(y_train, model_base.predict(X_train_base)))
test_rmse_base = np.sqrt(mean_squared_error(y_test, model_base.predict(X_test_base)))

print(f"   Baseline - Train RMSE: {train_rmse_base:.3f}")
print(f"   Baseline - Test RMSE:  {test_rmse_base:.3f}")

# ============================================================================
# PHASE 1 MODEL (All Features)
# ============================================================================

print("\n5. Training Phase 1 model (all features)...")

model_phase1 = HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)
model_phase1.fit(X_train, y_train)

train_rmse_phase1 = np.sqrt(mean_squared_error(y_train, model_phase1.predict(X_train)))
test_rmse_phase1 = np.sqrt(mean_squared_error(y_test, model_phase1.predict(X_test)))

print(f"   Phase 1 - Train RMSE: {train_rmse_phase1:.3f}")
print(f"   Phase 1 - Test RMSE:  {test_rmse_phase1:.3f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\nBaseline (rolling + context + position):")
print(f"  Features: {len(baseline_cols)}")
print(f"  Test RMSE: {test_rmse_base:.3f}")

print(f"\nPhase 1 (+ lag1 + consistency + efficiency + usage + pace):")
print(f"  Features: {len(phase1_cols)}")
print(f"  Test RMSE: {test_rmse_phase1:.3f}")

improvement = test_rmse_base - test_rmse_phase1
pct_improvement = (improvement / test_rmse_base) * 100

print(f"\nImprovement:")
print(f"  RMSE Delta: {improvement:+.3f}")
print(f"  % improvement: {pct_improvement:+.2f}%")

if improvement >= 0.30:
    print(f"\n[SUCCESS] Target achieved! (expected -0.30 to -0.50)")
elif improvement >= 0.15:
    print(f"\n[GOOD] Good progress, but below target (expected -0.30 to -0.50)")
else:
    print(f"\n[NEEDS WORK] Below expectations (expected -0.30 to -0.50)")

# Feature importance (HistGradientBoostingRegressor doesn't have feature_importances_)
# Could use permutation importance but it's computationally expensive
# Skipping for now - we have the key metrics

print("\n" + "=" * 70)
print("Phase 1 feature engineering complete!")
print("=" * 70)
