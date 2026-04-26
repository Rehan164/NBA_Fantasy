"""
Optuna NBA Fantasy Points Hyperparameter Study — v2

Two methodological extensions to optuna_nba_fantasy_study.py, motivated by the
gap between our manually-tuned model (test RMSE 9.533) and the v1 Optuna run
(test RMSE ~9.70):

  1. INLINE MISSING-TEAMMATES DERIVATION
     The v1 script reads pre-computed injury features from
     nba_historical_games.csv (produced by add_injury_impact.py). Those features
     apply hard thresholds (>=15 rolling MIN, >=50% recent appearances) that
     filter out most of the rotation-depth signal — they only flag named
     "expected" players who didn't show up.

     The manual model (nba_fantasy_model.ipynb cell 28) uses a richer continuous
     "rotation deficit" derivation: sum of L10 minutes for players actually in
     the box score, plus the team's rolling baseline of that sum, plus the
     deficit between them. No thresholds. v2 ports that derivation in-process
     and applies it symmetrically (team-side and opp-side, 6 features total).

  2. TIMESERIESSPLIT CROSS-VALIDATION
     The v1 script tunes against a single 2022-23 holdout. That season is the
     first fully-normalized post-COVID year — its specific patterns may not
     transfer to the 2023-24+ test set, so single-fold tuning can overfit.
     v2 uses sklearn's TimeSeriesSplit(n_splits=3) on all pre-2023-10-01 data;
     each Optuna trial is scored by mean RMSE across three expanding-window
     folds. Optuna's MedianPruner stops trials whose first-fold RMSE is worse
     than the median of completed trials, recovering most of the compute that
     K-fold would otherwise add.

Trial count drops 200 -> 120 to keep total runtime under 10 hours. With the
median pruner cutting ~50% of trials after fold 1, effective compute is
roughly 75-80% of v1.

Outputs are written to models_v2/ to keep the v1 partner runs untouched.

Usage:
    python optuna_nba_fantasy_study_v2.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
import time
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import optuna
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path("data")
MODEL_DIR = Path("models_v2")

DK_WEIGHTS = {"PTS": 1.0, "REB": 1.25, "AST": 1.5, "STL": 2.0, "BLK": 2.0, "TOV": -0.5, "FG3M": 0.5}
PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]
TEAM_STATS = ["score", "fg_made", "fg3_made", "reb", "ast", "stl", "blk", "tov"]
EFF_STATS = ["FGA", "FTA", "FG_PCT", "FG3_PCT", "PLUS_MINUS"]
WINDOWS = [3, 5, 10]

# v2 CHANGES vs v1 partner script
N_TRIALS = 120                  # v1: 200 — reduced to fit 10h budget with K-fold
N_SPLITS = 3                    # v1: 1 single fold — v2 uses TimeSeriesSplit(3)
PRUNER_STARTUP_TRIALS = 15      # baseline trials before pruning kicks in
PRUNER_WARMUP_STEPS = 0         # prune after fold 1 (step 0 reported)

# Reference RMSEs for the final comparison block
MANUAL_BEST = 9.533             # nba_fantasy_model.ipynb (HistGB + multi-output, hand-tuned)
V1_OPTUNA_BEST = 9.700          # partner's optuna_nba_fantasy_study.py (single-fold)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    print("=" * 60)
    print("1. Loading data")
    print("=" * 60)

    player_logs = pd.read_csv(DATA_DIR / "nba_player_game_logs.csv")
    games = pd.read_csv(DATA_DIR / "nba_historical_games.csv")

    player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])
    games["date"] = pd.to_datetime(games["date"])
    player_logs["game_id_int"] = player_logs["GAME_ID"].astype(int)
    games["game_id_int"] = games["game_id"].astype(int)
    player_logs["MIN"] = pd.to_numeric(player_logs["MIN"], errors="coerce")
    player_logs["FG_PCT"] = player_logs["FG_PCT"].fillna(0)
    player_logs["FG3_PCT"] = player_logs["FG3_PCT"].fillna(0)

    player_logs["FANTASY_PTS"] = (
        player_logs["PTS"] * 1.0 + player_logs["REB"] * 1.25 +
        player_logs["AST"] * 1.5 + player_logs["STL"] * 2.0 +
        player_logs["BLK"] * 2.0 + player_logs["TOV"] * -0.5 +
        player_logs["FG3M"] * 0.5
    )
    player_logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    print(f"  Player logs: {player_logs.shape}")
    print(f"  Games:       {games.shape}")
    return player_logs, games


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def build_features(player_logs, games):
    print("\n" + "=" * 60)
    print("2. Feature Engineering")
    print("=" * 60)

    # ── Player lag + rolling ──
    player_lag_cols = []
    for stat in PLAYER_STATS:
        for lag in range(1, 11):
            col = f"player_{stat}_lag{lag}"
            player_logs[col] = player_logs.groupby("PLAYER_ID")[stat].shift(lag)
            player_lag_cols.append(col)

    roll_cols = []
    for stat in PLAYER_STATS:
        for w in WINDOWS:
            col = f"player_{stat}_L{w}"
            player_logs[col] = player_logs[
                [f"player_{stat}_lag{l}" for l in range(1, w + 1)]
            ].mean(axis=1)
            roll_cols.append(col)

    for w in WINDOWS:
        col = f"player_FANTASY_PTS_L{w}"
        player_logs[col] = (
            player_logs.groupby("PLAYER_ID")["FANTASY_PTS"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
        )
        roll_cols.append(col)

    # ── Trends ──
    trend_cols = []
    for stat in PLAYER_STATS:
        col = f"player_{stat}_trend"
        player_logs[col] = player_logs[f"player_{stat}_L3"] - player_logs[f"player_{stat}_L10"]
        trend_cols.append(col)

    # ── Efficiency ──
    eff_cols = []
    for stat in EFF_STATS:
        for w in WINDOWS:
            col = f"player_{stat}_L{w}"
            player_logs[col] = (
                player_logs.groupby("PLAYER_ID")[stat]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
            )
            eff_cols.append(col)
    curated_eff_cols = [
        f"player_{s}_L{w}" for s in ["FGA", "FG_PCT", "PLUS_MINUS"] for w in WINDOWS
    ]

    print(f"  Player rolling: {len(roll_cols)}, Trends: {len(trend_cols)}, Efficiency: {len(curated_eff_cols)}")

    # ── Team lags + rolling ──
    home_games = games[["game_id_int", "date", "home_team",
                         "home_score", "home_fg_made", "home_fg3_made",
                         "home_reb", "home_ast", "home_stl",
                         "home_blk", "home_tov"]].copy()
    home_games.columns = ["game_id_int", "date", "team"] + TEAM_STATS

    away_games = games[["game_id_int", "date", "away_team",
                         "away_score", "away_fg_made", "away_fg3_made",
                         "away_reb", "away_ast", "away_stl",
                         "away_blk", "away_tov"]].copy()
    away_games.columns = ["game_id_int", "date", "team"] + TEAM_STATS

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)

    team_lag_cols = []
    for stat in TEAM_STATS:
        for lag in range(1, 11):
            col = f"team_{stat}_lag{lag}"
            team_games[col] = team_games.groupby("team")[stat].shift(lag)
            team_lag_cols.append(col)

    team_roll_cols = []
    for stat in TEAM_STATS:
        for w in WINDOWS:
            col = f"team_{stat}_L{w}"
            team_games[col] = team_games[
                [f"team_{stat}_lag{l}" for l in range(1, w + 1)]
            ].mean(axis=1)
            team_roll_cols.append(col)
    roll_cols += team_roll_cols

    # ── Opponent rolling ──
    opp_map = pd.concat([
        games[["game_id_int", "home_team", "away_team"]].rename(
            columns={"home_team": "team", "away_team": "opponent"}),
        games[["game_id_int", "away_team", "home_team"]].rename(
            columns={"away_team": "team", "home_team": "opponent"})
    ], ignore_index=True)
    team_games = team_games.merge(opp_map, on=["game_id_int", "team"], how="left")

    opp_roll_cols = [c.replace("team_", "opp_") for c in team_roll_cols]
    opp_lookup = team_games[["game_id_int", "team"] + team_roll_cols].rename(
        columns={"team": "opponent", **{c: c.replace("team_", "opp_") for c in team_roll_cols}}
    )
    team_games = team_games.merge(opp_lookup, on=["game_id_int", "opponent"], how="left")
    roll_cols += opp_roll_cols

    print(f"  Team rolling: {len(team_roll_cols)}, Opp rolling: {len(opp_roll_cols)}")

    # ── Merge player + team/opp ──
    team_merge_cols = ["game_id_int", "team"] + team_roll_cols + opp_roll_cols
    df = player_logs.merge(
        team_games[team_merge_cols].drop_duplicates(subset=["game_id_int", "team"]),
        left_on=["game_id_int", "TEAM_ABBREVIATION"],
        right_on=["game_id_int", "team"],
        how="inner",
    )

    # ── Context ──
    home_lookup = games[["game_id_int", "home_team"]].copy()
    df = df.merge(home_lookup, on="game_id_int", how="left")
    df["is_home"] = (df["TEAM_ABBREVIATION"] == df["home_team"]).astype(int)

    team_sched = (
        df[["TEAM_ABBREVIATION", "GAME_DATE", "game_id_int"]]
        .drop_duplicates(subset=["TEAM_ABBREVIATION", "game_id_int"])
        .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
        .reset_index(drop=True)
    )
    team_sched["days_rest"] = (
        team_sched.groupby("TEAM_ABBREVIATION")["GAME_DATE"]
        .diff().dt.days.clip(upper=7).fillna(3).astype(int)
    )
    df = df.merge(
        team_sched[["game_id_int", "TEAM_ABBREVIATION", "days_rest"]],
        on=["game_id_int", "TEAM_ABBREVIATION"], how="left",
    )

    opp_key = pd.concat([
        games[["game_id_int", "home_team", "away_team"]].rename(
            columns={"home_team": "TEAM_ABBREVIATION", "away_team": "_opp"}),
        games[["game_id_int", "away_team", "home_team"]].rename(
            columns={"away_team": "TEAM_ABBREVIATION", "home_team": "_opp"})
    ], ignore_index=True)
    opp_rest = team_sched[["game_id_int", "TEAM_ABBREVIATION", "days_rest"]].rename(
        columns={"TEAM_ABBREVIATION": "_opp", "days_rest": "opp_days_rest"}
    )
    opp_key = opp_key.merge(opp_rest, on=["game_id_int", "_opp"], how="left")
    df = df.merge(
        opp_key[["game_id_int", "TEAM_ABBREVIATION", "opp_days_rest"]],
        on=["game_id_int", "TEAM_ABBREVIATION"], how="left",
    )
    context_cols = ["is_home", "days_rest", "opp_days_rest"]

    # ── Inline missing-teammates (v2 CHANGE) ──
    # Replaces the v1 merge of pre-computed injury columns from
    # nba_historical_games.csv with the continuous-deficit derivation from
    # nba_fantasy_model.ipynb cell 28. Same recipe applied to opp-side too.
    pl_min = player_logs[["PLAYER_ID", "game_id_int", "TEAM_ABBREVIATION", "MIN"]].copy()
    pl_min["MIN_L10"] = (
        pl_min.groupby("PLAYER_ID")["MIN"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    # Per (game, team): rotation depth actually present
    team_min = (
        pl_min.dropna(subset=["MIN_L10"])
        .groupby(["game_id_int", "TEAM_ABBREVIATION"])["MIN_L10"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "team_l10_min_played", "count": "team_players_played"})
    )

    # Rolling baseline (last 10 team games) → deficit signal
    team_min = team_min.merge(
        games[["game_id_int", "date"]], on="game_id_int", how="left"
    ).sort_values(["TEAM_ABBREVIATION", "date"]).reset_index(drop=True)
    team_min["team_l10_min_baseline"] = (
        team_min.groupby("TEAM_ABBREVIATION")["team_l10_min_played"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
    team_min["missing_min_deficit"] = (
        team_min["team_l10_min_baseline"] - team_min["team_l10_min_played"]
    )

    # Merge team-side
    df = df.merge(
        team_min[["game_id_int", "TEAM_ABBREVIATION",
                  "team_l10_min_played", "team_players_played", "missing_min_deficit"]],
        on=["game_id_int", "TEAM_ABBREVIATION"], how="left",
    )

    # Merge opp-side: same numbers, looked up by opponent abbreviation
    opp_min = team_min[["game_id_int", "TEAM_ABBREVIATION",
                        "team_l10_min_played", "team_players_played", "missing_min_deficit"]].rename(
        columns={
            "TEAM_ABBREVIATION": "_opp",
            "team_l10_min_played": "opp_l10_min_played",
            "team_players_played": "opp_players_played",
            "missing_min_deficit": "opp_missing_min_deficit",
        }
    )
    opp_team_map = pd.concat([
        games[["game_id_int", "home_team", "away_team"]].rename(
            columns={"home_team": "TEAM_ABBREVIATION", "away_team": "_opp"}),
        games[["game_id_int", "away_team", "home_team"]].rename(
            columns={"away_team": "TEAM_ABBREVIATION", "home_team": "_opp"})
    ], ignore_index=True).drop_duplicates(subset=["game_id_int", "TEAM_ABBREVIATION"])
    df = df.merge(opp_team_map, on=["game_id_int", "TEAM_ABBREVIATION"], how="left")
    df = df.merge(opp_min, on=["game_id_int", "_opp"], how="left").drop(columns=["_opp"])

    injury_cols = [
        "team_l10_min_played", "team_players_played", "missing_min_deficit",
        "opp_l10_min_played", "opp_players_played", "opp_missing_min_deficit",
    ]
    print(f"  Injury impact (inline): {len(injury_cols)}  "
          f"(coverage: {df['missing_min_deficit'].notna().mean()*100:.1f}%)")

    # ── Schedule density ──
    team_sched_sorted = team_sched.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    team_sched_sorted["is_b2b"] = (team_sched_sorted["days_rest"] == 1).astype(int)

    def games_in_last_n_days(group, n):
        dates = group["GAME_DATE"].values.astype("datetime64[D]")
        out = np.zeros(len(dates), dtype=int)
        for i in range(len(dates)):
            out[i] = ((dates < dates[i]) & (dates >= dates[i] - np.timedelta64(n, "D"))).sum()
        return pd.Series(out, index=group.index)

    team_sched_sorted["games_last_4d"] = (
        team_sched_sorted.groupby("TEAM_ABBREVIATION", group_keys=False)
        .apply(lambda g: games_in_last_n_days(g, 4), include_groups=False)
    )
    team_sched_sorted["games_last_7d"] = (
        team_sched_sorted.groupby("TEAM_ABBREVIATION", group_keys=False)
        .apply(lambda g: games_in_last_n_days(g, 7), include_groups=False)
    )
    df = df.merge(
        team_sched_sorted[["game_id_int", "TEAM_ABBREVIATION", "is_b2b", "games_last_4d", "games_last_7d"]],
        on=["game_id_int", "TEAM_ABBREVIATION"], how="left",
    )
    schedule_cols = ["is_b2b", "games_last_4d", "games_last_7d"]

    # ── Position ──
    position_path = DATA_DIR / "nba_player_info.csv"
    position_cols = []
    if position_path.exists():
        pinfo = pd.read_csv(position_path)

        def bucket_position(p):
            if not isinstance(p, str):
                return "U"
            p = p.lower()
            if "guard" in p:   return "G"
            if "center" in p:  return "C"
            if "forward" in p: return "F"
            return "U"

        pinfo["pos_bucket"] = pinfo["POSITION"].apply(bucket_position)
        pinfo["height_in"] = pinfo["HEIGHT"].apply(
            lambda h: int(h.split("-")[0]) * 12 + int(h.split("-")[1])
            if isinstance(h, str) and "-" in h else np.nan
        )
        pinfo["draft_year"] = pd.to_numeric(pinfo["DRAFT_YEAR"], errors="coerce")
        df = df.merge(
            pinfo[["PERSON_ID", "pos_bucket", "height_in", "draft_year"]].rename(
                columns={"PERSON_ID": "PLAYER_ID"}
            ),
            on="PLAYER_ID", how="left",
        )
        for pos in ["G", "F", "C"]:
            df[f"pos_{pos}"] = (df["pos_bucket"] == pos).astype(int)
        df["years_experience"] = df["GAME_DATE"].dt.year - df["draft_year"]
        position_cols = ["pos_G", "pos_F", "pos_C", "height_in", "years_experience"]
        print(f"  Position: {len(position_cols)}")
    else:
        print("  Position: skipped (no nba_player_info.csv)")

    # ── DvP ──
    dvp_cols = []
    if "pos_bucket" in df.columns:
        g_opp_simple = pd.concat([
            games[["game_id_int", "home_team", "away_team"]].rename(
                columns={"home_team": "TEAM_ABBREVIATION", "away_team": "opp_team"}),
            games[["game_id_int", "away_team", "home_team"]].rename(
                columns={"away_team": "TEAM_ABBREVIATION", "home_team": "opp_team"})
        ], ignore_index=True).drop_duplicates(subset=["game_id_int", "TEAM_ABBREVIATION"])
        df = df.merge(g_opp_simple, on=["game_id_int", "TEAM_ABBREVIATION"], how="left")

        dvp_base = (
            df.dropna(subset=["pos_bucket"])
            .groupby(["game_id_int", "opp_team", "pos_bucket"])
            .agg(fp_sum=("FANTASY_PTS", "sum"), n=("FANTASY_PTS", "count"))
            .reset_index()
        )
        dvp_base["fp_per_player"] = dvp_base["fp_sum"] / dvp_base["n"]
        dvp_base = dvp_base.merge(games[["game_id_int", "date"]], on="game_id_int", how="left")
        dvp_base = dvp_base.sort_values(["opp_team", "pos_bucket", "date"]).reset_index(drop=True)
        dvp_base["dvp_L20"] = (
            dvp_base.groupby(["opp_team", "pos_bucket"])["fp_per_player"]
            .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
        )
        df = df.merge(
            dvp_base[["game_id_int", "opp_team", "pos_bucket", "dvp_L20"]],
            on=["game_id_int", "opp_team", "pos_bucket"], how="left",
        )
        dvp_cols = ["dvp_L20"]
        print(f"  DvP: {len(dvp_cols)}")

    # ── Assemble ──
    feature_groups = {
        "rolling":    roll_cols,
        "context":    context_cols,
        "trends":     trend_cols,
        "efficiency": curated_eff_cols,
        "injury":     injury_cols,
        "schedule":   schedule_cols,
        "position":   position_cols,
        "dvp":        dvp_cols,
    }

    all_features = []
    for gc in feature_groups.values():
        all_features += gc
    all_features = list(dict.fromkeys(all_features))

    required = roll_cols + trend_cols + curated_eff_cols
    df_clean = df.dropna(subset=required).reset_index(drop=True)
    df_clean = df_clean[df_clean["MIN"] >= 10].reset_index(drop=True)

    print(f"\n  Clean dataset: {len(df_clean):,} rows, {len(all_features)} features")
    print(f"  Feature groups: {[(k, len(v)) for k, v in feature_groups.items()]}")

    return df_clean, feature_groups, all_features, roll_cols, trend_cols, curated_eff_cols


# ═══════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE — TimeSeriesSplit (v2 CHANGE)
# ═══════════════════════════════════════════════════════════════════════

def make_objective(X_full, y_full, stat_targets_full, fold_indices,
                   feature_groups, roll_cols, feature_index):
    """
    Optuna objective: average RMSE across N_SPLITS time-ordered folds.

    X_full              : (N, F_max) numpy matrix — all candidate features for
                          the full pre-2023-10-01 train data, sorted by date.
    y_full              : (N,) FANTASY_PTS targets aligned to X_full.
    stat_targets_full   : dict[stat -> (N,) array] for multi-output decomposition.
    fold_indices        : list[(train_idx, val_idx)] from TimeSeriesSplit.
    feature_index       : dict[col_name -> column index in X_full] for fast slicing.

    Each fold's RMSE is reported via trial.report() so MedianPruner can stop
    weak trials before they pay for the larger later folds.
    """

    def objective(trial):
        # ── Model hyperparameters ──
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        max_iter         = trial.suggest_int("max_iter", 300, 1500)
        max_depth        = trial.suggest_int("max_depth", 4, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 20, 100)
        l2_reg           = trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)
        max_bins         = trial.suggest_int("max_bins", 128, 255)
        loss             = trial.suggest_categorical("loss", ["squared_error", "absolute_error"])

        use_leaf_nodes = trial.suggest_categorical("use_max_leaf_nodes", [True, False])
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 31, 127) if use_leaf_nodes else None

        early_stopping      = True
        n_iter_no_change    = trial.suggest_int("n_iter_no_change", 10, 30)
        validation_fraction = trial.suggest_float("validation_fraction", 0.10, 0.20)

        # ── Strategy ──
        use_multi_output = trial.suggest_categorical("use_multi_output", [True, False])

        # ── Feature selection ──
        features = list(roll_cols)
        for group_name in ["context", "trends", "efficiency", "injury", "schedule", "position", "dvp"]:
            use_it = trial.suggest_categorical(f"use_{group_name}", [True, False])
            if use_it and feature_groups[group_name]:
                features += feature_groups[group_name]
        features = list(dict.fromkeys(features))
        feat_idx = np.array([feature_index[f] for f in features])

        params = dict(
            loss=loss, learning_rate=learning_rate, max_iter=max_iter,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_reg, max_bins=max_bins,
            max_leaf_nodes=max_leaf_nodes, early_stopping=early_stopping,
            random_state=42,
        )
        if early_stopping:
            params["n_iter_no_change"] = n_iter_no_change
            params["validation_fraction"] = validation_fraction

        # ── K-fold evaluation ──
        fold_rmses = []
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_indices):
            X_tr = X_full[np.ix_(tr_idx, feat_idx)]
            X_va = X_full[np.ix_(va_idx, feat_idx)]
            y_tr = y_full[tr_idx]
            y_va = y_full[va_idx]

            try:
                if use_multi_output:
                    pred_va = np.zeros(len(X_va))
                    for stat, weight in DK_WEIGHTS.items():
                        m = HistGradientBoostingRegressor(**params)
                        m.fit(X_tr, stat_targets_full[stat][tr_idx])
                        pred_va += weight * m.predict(X_va)
                    rmse = np.sqrt(mean_squared_error(y_va, pred_va))
                else:
                    m = HistGradientBoostingRegressor(**params)
                    m.fit(X_tr, y_tr)
                    rmse = np.sqrt(mean_squared_error(y_va, m.predict(X_va)))
            except Exception:
                return float("inf")

            fold_rmses.append(rmse)

            # Report intermediate value and let the pruner decide
            trial.report(rmse, fold_idx)
            if trial.should_prune():
                trial.set_user_attr("fold_rmses", fold_rmses)
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_rmses", fold_rmses)
        trial.set_user_attr("fold_rmse_std", float(np.std(fold_rmses)))
        return float(np.mean(fold_rmses))

    return objective


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def save_visuals(study, visuals_dir):
    """Save Optuna study plots as PNGs."""
    print("\n  Saving visualizations...")
    trials_df = study.trials_dataframe()
    completed = trials_df[trials_df["state"] == "COMPLETE"].copy()
    completed = completed[completed["value"] < float("inf")].sort_values("number")

    # 1. Optimization history (completed trials only — pruned trials have no final value)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(completed["number"], completed["value"], "o-", alpha=0.4, markersize=3, label="Trial mean fold RMSE")
    best_so_far = completed["value"].cummin()
    ax.plot(completed["number"], best_so_far, "r-", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean validation RMSE across folds")
    ax.set_title("Optimization History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(visuals_dir / "optimization_history.png", dpi=150)
    plt.close(fig)

    # 2. Hyperparameter importance (correlation-based)
    numeric_params = []
    for col in completed.columns:
        if col.startswith("params_") and completed[col].dtype in [np.float64, np.int64, float, int]:
            numeric_params.append(col)

    if numeric_params:
        importances = {}
        for col in numeric_params:
            valid = completed[[col, "value"]].dropna()
            if len(valid) > 5:
                importances[col.replace("params_", "")] = abs(valid[col].corr(valid["value"]))

        if importances:
            imp_df = pd.Series(importances).sort_values(ascending=True).tail(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            imp_df.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("|Correlation with mean fold RMSE|")
            ax.set_title("Hyperparameter Importance (top 15)")
            ax.grid(True, alpha=0.3, axis="x")
            fig.tight_layout()
            fig.savefig(visuals_dir / "param_importance.png", dpi=150)
            plt.close(fig)

    # 3. RMSE distribution + pruning summary
    n_pruned = (trials_df["state"] == "PRUNED").sum()
    n_complete = len(completed)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(completed["value"], bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(study.best_value, color="red", linestyle="--", linewidth=2, label=f"Best: {study.best_value:.4f}")
    ax.set_xlabel("Mean validation RMSE")
    ax.set_ylabel("Count")
    ax.set_title(f"Trial RMSE Distribution  ({n_complete} completed, {n_pruned} pruned)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(visuals_dir / "rmse_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Learning rate vs RMSE
    if "params_learning_rate" in completed.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(completed["params_learning_rate"], completed["value"], alpha=0.5, s=15, c="steelblue")
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Mean validation RMSE")
        ax.set_title("Learning Rate vs RMSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(visuals_dir / "learning_rate_vs_rmse.png", dpi=150)
        plt.close(fig)

    # 5. max_depth vs RMSE
    if "params_max_depth" in completed.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(completed["params_max_depth"], completed["value"], alpha=0.5, s=15, c="steelblue")
        ax.set_xlabel("Max Depth")
        ax.set_ylabel("Mean validation RMSE")
        ax.set_title("Max Depth vs RMSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(visuals_dir / "max_depth_vs_rmse.png", dpi=150)
        plt.close(fig)

    # 6. Multi-output vs direct boxplot
    if "params_use_multi_output" in completed.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = completed.groupby("params_use_multi_output")["value"].apply(list)
        labels = [f"Direct\n(n={len(groups.get(False, []))})", f"Multi-output\n(n={len(groups.get(True, []))})"]
        data = [groups.get(False, []), groups.get(True, [])]
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("Mean validation RMSE")
        ax.set_title("Direct vs Multi-Output Decomposition")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(visuals_dir / "multi_output_comparison.png", dpi=150)
        plt.close(fig)

    # 7. Feature group usage in top 20 trials
    top_20 = completed.nsmallest(20, "value")
    feat_usage = {}
    for grp in ["context", "trends", "efficiency", "injury", "schedule", "position", "dvp"]:
        col = f"params_use_{grp}"
        if col in top_20.columns:
            feat_usage[grp] = top_20[col].mean() * 100

    if feat_usage:
        fig, ax = plt.subplots(figsize=(10, 5))
        grps = list(feat_usage.keys())
        vals = [feat_usage[g] for g in grps]
        ax.bar(grps, vals, color="steelblue", edgecolor="black")
        ax.set_ylabel("% of top-20 trials using this group")
        ax.set_title("Feature Group Usage in Best Trials")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(vals):
            ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(visuals_dir / "feature_group_usage.png", dpi=150)
        plt.close(fig)

    # 8. Per-fold RMSE for top-10 trials (v2-specific) ──────────────────
    fold_rows = []
    for t in study.trials:
        if t.state.name == "COMPLETE" and "fold_rmses" in t.user_attrs:
            fold_rows.append({"trial": t.number, "value": t.value,
                              **{f"fold_{i}": v for i, v in enumerate(t.user_attrs["fold_rmses"])}})
    if fold_rows:
        fold_df = pd.DataFrame(fold_rows).nsmallest(10, "value")
        fold_cols = [c for c in fold_df.columns if c.startswith("fold_")]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(fold_df))
        for i, c in enumerate(fold_cols):
            ax.scatter(x, fold_df[c], label=c, alpha=0.7, s=40)
        ax.plot(x, fold_df["value"], "k-", linewidth=2, label="Mean (objective)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{t}" for t in fold_df["trial"]], rotation=45, ha="right")
        ax.set_xlabel("Trial number")
        ax.set_ylabel("RMSE")
        ax.set_title("Per-fold RMSE for top-10 trials (TimeSeriesSplit folds)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(visuals_dir / "fold_rmse_top10.png", dpi=150)
        plt.close(fig)

    print(f"  Saved {len(list(visuals_dir.glob('*.png')))} plots to {visuals_dir}")


# ═══════════════════════════════════════════════════════════════════════
# RUN LOGGING
# ═══════════════════════════════════════════════════════════════════════

def log_run(run_id, n_trials, n_completed, n_pruned, best_val_rmse, best_fold_std,
            test_rmse, linear_baseline, best_params, elapsed_min):
    """Append run summary to persistent CSV log."""
    log_path = MODEL_DIR / "optuna_v2_run_log.csv"
    is_new = not log_path.exists()

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "n_completed": n_completed,
        "n_pruned": n_pruned,
        "n_splits": N_SPLITS,
        "best_val_rmse_mean": round(best_val_rmse, 4),
        "best_val_rmse_std": round(best_fold_std, 4),
        "test_rmse": round(test_rmse, 4),
        "linear_baseline": round(linear_baseline, 4),
        "manual_best": MANUAL_BEST,
        "v1_optuna_best": V1_OPTUNA_BEST,
        "delta_vs_manual": round(test_rmse - MANUAL_BEST, 4),
        "delta_vs_v1_optuna": round(test_rmse - V1_OPTUNA_BEST, 4),
        "pct_improvement_vs_linear": round((linear_baseline - test_rmse) / linear_baseline * 100, 2),
        "elapsed_min": round(elapsed_min, 1),
        "best_params": json.dumps(best_params, default=str),
    }

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Run logged to {log_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = run_dir / "visuals"
    visuals_dir.mkdir(exist_ok=True)

    print(f"v2 study — TimeSeriesSplit({N_SPLITS}) + inline injury features")
    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}\n")

    # ── Data + features ──
    player_logs, games = load_data()
    df_clean, feature_groups, all_features, roll_cols, trend_cols, curated_eff_cols = build_features(player_logs, games)

    # ── Splits ──
    # v1 used: inner_train < 2022-10-01, valid = 2022-23 season, test >= 2023-10-01
    # v2 uses: train < 2023-10-01 (passed to TimeSeriesSplit), test >= 2023-10-01
    is_train = df_clean["GAME_DATE"] < "2023-10-01"
    is_test  = df_clean["GAME_DATE"] >= "2023-10-01"

    # Sort the train portion by date so TimeSeriesSplit produces time-ordered folds
    df_train = df_clean[is_train].sort_values("GAME_DATE").reset_index(drop=True)
    df_test  = df_clean[is_test].reset_index(drop=True)

    print(f"\n  Train rows: {len(df_train):,}  (date range "
          f"{df_train['GAME_DATE'].min().date()} → {df_train['GAME_DATE'].max().date()})")
    print(f"  Test rows:  {len(df_test):,}  (date range "
          f"{df_test['GAME_DATE'].min().date()} → {df_test['GAME_DATE'].max().date()})")

    # Pre-build the full feature matrix once. Per-trial slicing is then a
    # cheap np.ix_ operation on column indices instead of repeated DataFrame
    # extraction.
    X_full = df_train[all_features].values.astype(np.float32, copy=False)
    y_full = df_train["FANTASY_PTS"].values.astype(np.float32, copy=False)
    feature_index = {f: i for i, f in enumerate(all_features)}
    stat_targets_full = {
        stat: df_train[stat].astype(np.float32).values for stat in DK_WEIGHTS
    }

    # TimeSeriesSplit produces time-ordered (train, val) index pairs
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_indices = [(tr_idx, va_idx) for tr_idx, va_idx in tscv.split(df_train)]

    print(f"\n  TimeSeriesSplit({N_SPLITS}) folds:")
    for i, (tr_idx, va_idx) in enumerate(fold_indices):
        tr_dates = df_train.iloc[tr_idx]["GAME_DATE"]
        va_dates = df_train.iloc[va_idx]["GAME_DATE"]
        print(f"    Fold {i+1}: train {tr_dates.min().date()} → {tr_dates.max().date()} "
              f"({len(tr_idx):,} rows)  |  val {va_dates.min().date()} → {va_dates.max().date()} "
              f"({len(va_idx):,} rows)")

    # ── Linear baseline (full train, evaluated on test) ──
    print("\n" + "=" * 60)
    print("3. Linear Regression Baseline")
    print("=" * 60)

    lr = make_pipeline(StandardScaler(), LinearRegression())
    lr.fit(df_train[all_features].fillna(0), df_train["FANTASY_PTS"])
    lr_test_rmse = np.sqrt(mean_squared_error(
        df_test["FANTASY_PTS"],
        lr.predict(df_test[all_features].fillna(0)),
    ))
    LINEAR_BASELINE = lr_test_rmse
    print(f"  Test RMSE: {lr_test_rmse:.4f}")

    # ── Optuna study ──
    print("\n" + "=" * 60)
    print(f"4. Optuna Study ({N_TRIALS} trials, TimeSeriesSplit({N_SPLITS}), MedianPruner)")
    print("=" * 60)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=PRUNER_STARTUP_TRIALS,
        n_warmup_steps=PRUNER_WARMUP_STEPS,
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="nba_fantasy_v2_tss",
    )

    objective = make_objective(
        X_full, y_full, stat_targets_full,
        fold_indices, feature_groups, roll_cols, feature_index,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    n_pruned   = sum(1 for t in study.trials if t.state.name == "PRUNED")

    print(f"\n  Trials: {n_complete} completed, {n_pruned} pruned, "
          f"{len(study.trials) - n_complete - n_pruned} other")
    print(f"  Best mean fold RMSE: {study.best_value:.4f} (trial #{best.number})")
    if "fold_rmses" in best.user_attrs:
        fr = best.user_attrs["fold_rmses"]
        print(f"  Best per-fold:       " + ", ".join(f"{r:.4f}" for r in fr) +
              f"  (std {np.std(fr):.4f})")
    print(f"  Best params:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:25s} = {v}")

    # ── Retrain best model on full training data ──
    print("\n" + "=" * 60)
    print("5. Retrain Best Model on Full Training Data")
    print("=" * 60)

    bp = best.params
    best_features = list(roll_cols)
    for grp in ["context", "trends", "efficiency", "injury", "schedule", "position", "dvp"]:
        if bp.get(f"use_{grp}", False) and feature_groups[grp]:
            best_features += feature_groups[grp]
    best_features = list(dict.fromkeys(best_features))

    best_model_params = dict(
        loss=bp["loss"], learning_rate=bp["learning_rate"],
        max_iter=bp["max_iter"], max_depth=bp["max_depth"],
        min_samples_leaf=bp["min_samples_leaf"],
        l2_regularization=bp["l2_regularization"],
        max_bins=bp["max_bins"],
        max_leaf_nodes=bp.get("max_leaf_nodes") if bp.get("use_max_leaf_nodes") else None,
        early_stopping=True, random_state=42,
        n_iter_no_change=bp["n_iter_no_change"],
        validation_fraction=bp["validation_fraction"],
    )

    X_train = df_train[best_features].values
    X_test  = df_test[best_features].values
    y_train = df_train["FANTASY_PTS"].values
    y_test  = df_test["FANTASY_PTS"].values

    use_multi = bp["use_multi_output"]

    if use_multi:
        models = {}
        pred_train = np.zeros(len(X_train))
        pred_test  = np.zeros(len(X_test))
        for stat, weight in DK_WEIGHTS.items():
            m = HistGradientBoostingRegressor(**best_model_params)
            m.fit(X_train, df_train[stat].astype(float).values)
            pred_train += weight * m.predict(X_train)
            pred_test  += weight * m.predict(X_test)
            models[stat] = m
            rmse_te = np.sqrt(mean_squared_error(df_test[stat].astype(float).values, m.predict(X_test)))
            print(f"  {stat:5s} (w={weight:+.2f})  test RMSE={rmse_te:.3f}")
    else:
        m = HistGradientBoostingRegressor(**best_model_params)
        m.fit(X_train, y_train)
        pred_train = m.predict(X_train)
        pred_test  = m.predict(X_test)
        models = {"direct": m}

    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    test_rmse  = np.sqrt(mean_squared_error(y_test, pred_test))

    print(f"\n  Strategy:  {'multi-output' if use_multi else 'direct'}")
    print(f"  Features:  {len(best_features)}")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")

    # ── Comparison ──
    print("\n" + "=" * 60)
    print("6. Final Comparison")
    print("=" * 60)

    summary = pd.DataFrame([
        {"model": "Linear Regression baseline",                "test_rmse": LINEAR_BASELINE},
        {"model": "v1 Optuna (single-fold, threshold injury)", "test_rmse": V1_OPTUNA_BEST},
        {"model": "Manual hand-tuned (HistGB + multi-output)", "test_rmse": MANUAL_BEST},
        {"model": f"v2 Optuna (TSS + inline injury, {N_TRIALS} trials)", "test_rmse": test_rmse},
    ])
    summary["delta_vs_linear"]  = (summary["test_rmse"] - LINEAR_BASELINE).round(4)
    summary["pct_improvement"]  = ((LINEAR_BASELINE - summary["test_rmse"]) / LINEAR_BASELINE * 100).round(2)
    summary = summary.sort_values("test_rmse").reset_index(drop=True)
    print(summary.to_string(index=False))

    delta_manual = MANUAL_BEST - test_rmse
    delta_v1     = V1_OPTUNA_BEST - test_rmse
    print(f"\n  vs manual best:    {delta_manual:+.4f} RMSE  ({delta_manual / MANUAL_BEST * 100:+.2f}%)")
    print(f"  vs v1 Optuna best: {delta_v1:+.4f} RMSE  ({delta_v1 / V1_OPTUNA_BEST * 100:+.2f}%)")

    # ── Save model ──
    print("\n" + "=" * 60)
    print("7. Saving Model & Visuals")
    print("=" * 60)

    fold_std = float(np.std(best.user_attrs.get("fold_rmses", [study.best_value])))
    model_artifact = {
        "models": models,
        "feature_cols": best_features,
        "is_multi_output": use_multi,
        "dk_weights": DK_WEIGHTS,
        "best_params": dict(best.params),
        "best_model_params": best_model_params,
        "n_splits": N_SPLITS,
        "fold_rmses": best.user_attrs.get("fold_rmses"),
        "metrics": {
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "linear_baseline_rmse": float(LINEAR_BASELINE),
            "best_val_rmse_mean": float(study.best_value),
            "best_val_rmse_std": fold_std,
            "delta_vs_linear": float(test_rmse - LINEAR_BASELINE),
            "delta_vs_manual": float(test_rmse - MANUAL_BEST),
            "delta_vs_v1_optuna": float(test_rmse - V1_OPTUNA_BEST),
            "pct_improvement_vs_linear": float((LINEAR_BASELINE - test_rmse) / LINEAR_BASELINE * 100),
        },
    }

    model_path = run_dir / "optuna_v2_best_model.pkl"
    joblib.dump(model_artifact, model_path)
    print(f"  Model: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    study_json = {
        "run_id": run_id,
        "best_value": study.best_value,
        "best_params": dict(best.params),
        "n_trials": len(study.trials),
        "n_completed": n_complete,
        "n_pruned": n_pruned,
        "n_splits": N_SPLITS,
        "fold_rmses": best.user_attrs.get("fold_rmses"),
        "metrics": model_artifact["metrics"],
    }
    json_path = run_dir / "optuna_v2_study_results.json"
    with open(json_path, "w") as f:
        json.dump(study_json, f, indent=2, default=str)
    print(f"  Results: {json_path}")

    save_visuals(study, visuals_dir)

    elapsed_min = (time.time() - t0) / 60
    log_run(run_id, N_TRIALS, n_complete, n_pruned, study.best_value, fold_std,
            test_rmse, LINEAR_BASELINE, dict(best.params), elapsed_min)

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed_min:.1f} minutes")
    print(f"All outputs in: {run_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
