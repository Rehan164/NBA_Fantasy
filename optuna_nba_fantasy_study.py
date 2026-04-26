"""
Optuna NBA Fantasy Points Hyperparameter Study

Builds on the existing model pipeline, replacing inline missing-teammates
with pre-computed injury impact data from the CSV. Runs a 200-trial Optuna
study with 20 hyperparameters to find the best HistGradientBoosting config.

Each run:
  - Logs results to models/optuna_run_log.csv
  - Saves best model to models/run_<timestamp>/optuna_best_model.pkl
  - Saves Optuna visuals to models/run_<timestamp>/visuals/

Usage:
    python optuna_nba_fantasy_study.py
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
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import optuna
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

DK_WEIGHTS = {"PTS": 1.0, "REB": 1.25, "AST": 1.5, "STL": 2.0, "BLK": 2.0, "TOV": -0.5, "FG3M": 0.5}
PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]
TEAM_STATS = ["score", "fg_made", "fg3_made", "reb", "ast", "stl", "blk", "tov"]
EFF_STATS = ["FGA", "FTA", "FG_PCT", "FG3_PCT", "PLUS_MINUS"]
WINDOWS = [3, 5, 10]
N_TRIALS = 200


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

    # ── Injury impact from CSV ──
    injury_game_cols = [
        "home_missing_players", "home_missing_ppg", "home_missing_min", "home_top_missing_ppg",
        "away_missing_players", "away_missing_ppg", "away_missing_min", "away_top_missing_ppg",
    ]
    df = df.merge(games[["game_id_int"] + injury_game_cols], on="game_id_int", how="left")
    for suffix in ["missing_players", "missing_ppg", "missing_min", "top_missing_ppg"]:
        df[f"team_{suffix}"] = np.where(df["is_home"] == 1, df[f"home_{suffix}"], df[f"away_{suffix}"])
        df[f"opp_{suffix}"] = np.where(df["is_home"] == 1, df[f"away_{suffix}"], df[f"home_{suffix}"])
    df = df.drop(columns=injury_game_cols)

    injury_cols = [
        "team_missing_players", "team_missing_ppg", "team_missing_min", "team_top_missing_ppg",
        "opp_missing_players", "opp_missing_ppg", "opp_missing_min", "opp_top_missing_ppg",
    ]
    print(f"  Injury impact: {len(injury_cols)}")

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
# OPTUNA OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════

def make_objective(df_clean, feature_groups, roll_cols, is_inner_train, is_valid, stat_targets, y_it, y_v):
    """Build the Optuna objective closure."""

    def objective(trial):
        # ── Model hyperparameters ──
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
        max_iter         = trial.suggest_int("max_iter", 100, 2000)
        max_depth        = trial.suggest_int("max_depth", 3, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 100)
        l2_reg           = trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True)
        max_bins         = trial.suggest_int("max_bins", 32, 255)
        loss             = trial.suggest_categorical("loss", ["squared_error", "absolute_error"])

        use_leaf_nodes = trial.suggest_categorical("use_max_leaf_nodes", [True, False])
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 500) if use_leaf_nodes else None

        early_stopping = trial.suggest_categorical("early_stopping", [True, False])
        if early_stopping:
            n_iter_no_change    = trial.suggest_int("n_iter_no_change", 5, 30)
            validation_fraction = trial.suggest_float("validation_fraction", 0.05, 0.2)
        else:
            n_iter_no_change = None
            validation_fraction = None

        # ── Strategy ──
        use_multi_output = trial.suggest_categorical("use_multi_output", [True, False])

        # ── Feature selection ──
        features = list(roll_cols)
        for group_name in ["context", "trends", "efficiency", "injury", "schedule", "position", "dvp"]:
            use_it = trial.suggest_categorical(f"use_{group_name}", [True, False])
            if use_it and feature_groups[group_name]:
                features += feature_groups[group_name]
        features = list(dict.fromkeys(features))

        X_it = df_clean.loc[is_inner_train, features].values
        X_v = df_clean.loc[is_valid, features].values

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

        try:
            if use_multi_output:
                pred_v = np.zeros(len(X_v))
                for stat, weight in DK_WEIGHTS.items():
                    m = HistGradientBoostingRegressor(**params)
                    m.fit(X_it, stat_targets[stat]["inner_train"])
                    pred_v += weight * m.predict(X_v)
                rmse = np.sqrt(mean_squared_error(y_v, pred_v))
            else:
                m = HistGradientBoostingRegressor(**params)
                m.fit(X_it, y_it)
                rmse = np.sqrt(mean_squared_error(y_v, m.predict(X_v)))
        except Exception:
            return float("inf")

        return rmse

    return objective


# ═══════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def save_visuals(study, visuals_dir):
    """Save Optuna study plots as PNGs."""
    print("\n  Saving visualizations...")
    trials_df = study.trials_dataframe()
    trials_df = trials_df[trials_df["value"] < float("inf")].sort_values("number")

    # 1. Optimization history
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trials_df["number"], trials_df["value"], "o-", alpha=0.4, markersize=3, label="Trial RMSE")
    best_so_far = trials_df["value"].cummin()
    ax.plot(trials_df["number"], best_so_far, "r-", linewidth=2, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Optimization History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(visuals_dir / "optimization_history.png", dpi=150)
    plt.close(fig)

    # 2. Hyperparameter importance (correlation-based)
    numeric_params = []
    for col in trials_df.columns:
        if col.startswith("params_") and trials_df[col].dtype in [np.float64, np.int64, float, int]:
            numeric_params.append(col)

    if numeric_params:
        importances = {}
        for col in numeric_params:
            valid = trials_df[[col, "value"]].dropna()
            if len(valid) > 5:
                importances[col.replace("params_", "")] = abs(valid[col].corr(valid["value"]))

        if importances:
            imp_df = pd.Series(importances).sort_values(ascending=True).tail(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            imp_df.plot(kind="barh", ax=ax, color="steelblue")
            ax.set_xlabel("|Correlation with RMSE|")
            ax.set_title("Hyperparameter Importance (top 15)")
            ax.grid(True, alpha=0.3, axis="x")
            fig.tight_layout()
            fig.savefig(visuals_dir / "param_importance.png", dpi=150)
            plt.close(fig)

    # 3. RMSE distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(trials_df["value"], bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(study.best_value, color="red", linestyle="--", linewidth=2, label=f"Best: {study.best_value:.4f}")
    ax.set_xlabel("Validation RMSE")
    ax.set_ylabel("Count")
    ax.set_title("Trial RMSE Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(visuals_dir / "rmse_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Top params: learning_rate vs RMSE
    if "params_learning_rate" in trials_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(trials_df["params_learning_rate"], trials_df["value"], alpha=0.5, s=15, c="steelblue")
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Validation RMSE")
        ax.set_title("Learning Rate vs RMSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(visuals_dir / "learning_rate_vs_rmse.png", dpi=150)
        plt.close(fig)

    # 5. max_depth vs RMSE
    if "params_max_depth" in trials_df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(trials_df["params_max_depth"], trials_df["value"], alpha=0.5, s=15, c="steelblue")
        ax.set_xlabel("Max Depth")
        ax.set_ylabel("Validation RMSE")
        ax.set_title("Max Depth vs RMSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(visuals_dir / "max_depth_vs_rmse.png", dpi=150)
        plt.close(fig)

    # 6. Multi-output vs direct boxplot
    if "params_use_multi_output" in trials_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = trials_df.groupby("params_use_multi_output")["value"].apply(list)
        labels = [f"Direct\n(n={len(groups.get(False, []))})", f"Multi-output\n(n={len(groups.get(True, []))})"]
        data = [groups.get(False, []), groups.get(True, [])]
        ax.boxplot(data, labels=labels)
        ax.set_ylabel("Validation RMSE")
        ax.set_title("Direct vs Multi-Output Decomposition")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(visuals_dir / "multi_output_comparison.png", dpi=150)
        plt.close(fig)

    # 7. Feature group usage in top 20 trials
    top_20 = trials_df.nsmallest(20, "value")
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

    print(f"  Saved {len(list(visuals_dir.glob('*.png')))} plots to {visuals_dir}")


# ═══════════════════════════════════════════════════════════════════════
# RUN LOGGING
# ═══════════════════════════════════════════════════════════════════════

def log_run(run_id, n_trials, best_val_rmse, test_rmse, linear_baseline, best_params, elapsed_min):
    """Append run summary to persistent CSV log."""
    log_path = MODEL_DIR / "optuna_run_log.csv"
    is_new = not log_path.exists()

    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "n_trials": n_trials,
        "best_val_rmse": round(best_val_rmse, 4),
        "test_rmse": round(test_rmse, 4),
        "linear_baseline": round(linear_baseline, 4),
        "delta_vs_linear": round(test_rmse - linear_baseline, 4),
        "pct_improvement": round((linear_baseline - test_rmse) / linear_baseline * 100, 2),
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

    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}\n")

    # ── Data + features ──
    player_logs, games = load_data()
    df_clean, feature_groups, all_features, roll_cols, trend_cols, curated_eff_cols = build_features(player_logs, games)

    # ── Splits ──
    is_inner_train = df_clean["GAME_DATE"] < "2022-10-01"
    is_valid       = (df_clean["GAME_DATE"] >= "2022-10-01") & (df_clean["GAME_DATE"] < "2023-10-01")
    is_train       = df_clean["GAME_DATE"] < "2023-10-01"
    is_test        = df_clean["GAME_DATE"] >= "2023-10-01"

    y_it = df_clean.loc[is_inner_train, "FANTASY_PTS"].values
    y_v  = df_clean.loc[is_valid, "FANTASY_PTS"].values

    stat_targets = {}
    for stat in DK_WEIGHTS:
        stat_targets[stat] = {
            "inner_train": df_clean.loc[is_inner_train, stat].astype(float).values,
            "valid":       df_clean.loc[is_valid, stat].astype(float).values,
            "train":       df_clean.loc[is_train, stat].astype(float).values,
            "test":        df_clean.loc[is_test, stat].astype(float).values,
        }

    # ── Linear baseline ──
    print("\n" + "=" * 60)
    print("3. Linear Regression Baseline")
    print("=" * 60)

    lr = make_pipeline(StandardScaler(), LinearRegression())
    lr.fit(df_clean.loc[is_train, all_features].fillna(0), df_clean.loc[is_train, "FANTASY_PTS"])
    lr_test_rmse = np.sqrt(mean_squared_error(
        df_clean.loc[is_test, "FANTASY_PTS"],
        lr.predict(df_clean.loc[is_test, all_features].fillna(0)),
    ))
    LINEAR_BASELINE = lr_test_rmse
    print(f"  Test RMSE: {lr_test_rmse:.4f}")

    # ── Optuna study ──
    print("\n" + "=" * 60)
    print(f"4. Optuna Study ({N_TRIALS} trials)")
    print("=" * 60)

    objective = make_objective(df_clean, feature_groups, roll_cols, is_inner_train, is_valid, stat_targets, y_it, y_v)
    study = optuna.create_study(direction="minimize", study_name="nba_fantasy_hp_tuning")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print(f"\n  Best validation RMSE: {study.best_value:.4f} (trial #{best.number})")
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
        early_stopping=bp["early_stopping"], random_state=42,
    )
    if bp["early_stopping"]:
        best_model_params["n_iter_no_change"] = bp["n_iter_no_change"]
        best_model_params["validation_fraction"] = bp["validation_fraction"]

    X_train = df_clean.loc[is_train, best_features].values
    X_test  = df_clean.loc[is_test, best_features].values
    y_train = df_clean.loc[is_train, "FANTASY_PTS"].values
    y_test  = df_clean.loc[is_test, "FANTASY_PTS"].values

    use_multi = bp["use_multi_output"]

    if use_multi:
        models = {}
        pred_train = np.zeros(len(X_train))
        pred_test  = np.zeros(len(X_test))
        for stat, weight in DK_WEIGHTS.items():
            m = HistGradientBoostingRegressor(**best_model_params)
            m.fit(X_train, stat_targets[stat]["train"])
            pred_train += weight * m.predict(X_train)
            pred_test  += weight * m.predict(X_test)
            models[stat] = m
            rmse_te = np.sqrt(mean_squared_error(stat_targets[stat]["test"], m.predict(X_test)))
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

    PREV_BEST = 9.533
    summary = pd.DataFrame([
        {"model": "Linear Regression baseline",    "test_rmse": LINEAR_BASELINE},
        {"model": "Previous best (HistGB default)", "test_rmse": PREV_BEST},
        {"model": f"Optuna best ({N_TRIALS} trials)", "test_rmse": test_rmse},
    ])
    summary["delta_vs_linear"] = (summary["test_rmse"] - LINEAR_BASELINE).round(4)
    summary["pct_improvement"] = ((LINEAR_BASELINE - summary["test_rmse"]) / LINEAR_BASELINE * 100).round(2)
    summary = summary.sort_values("test_rmse").reset_index(drop=True)
    print(summary.to_string(index=False))

    delta_prev = PREV_BEST - test_rmse
    print(f"\n  vs previous best: {delta_prev:+.4f} RMSE ({delta_prev / PREV_BEST * 100:+.2f}%)")

    # ── Save model ──
    print("\n" + "=" * 60)
    print("7. Saving Model & Visuals")
    print("=" * 60)

    model_artifact = {
        "models": models,
        "feature_cols": best_features,
        "is_multi_output": use_multi,
        "dk_weights": DK_WEIGHTS,
        "best_params": dict(best.params),
        "best_model_params": best_model_params,
        "metrics": {
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "linear_baseline_rmse": float(LINEAR_BASELINE),
            "delta_vs_linear": float(test_rmse - LINEAR_BASELINE),
            "pct_improvement": float((LINEAR_BASELINE - test_rmse) / LINEAR_BASELINE * 100),
        },
    }

    model_path = run_dir / "optuna_best_model.pkl"
    joblib.dump(model_artifact, model_path)
    print(f"  Model: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Study results JSON
    study_json = {
        "run_id": run_id,
        "best_value": study.best_value,
        "best_params": dict(best.params),
        "n_trials": len(study.trials),
        "metrics": model_artifact["metrics"],
    }
    json_path = run_dir / "optuna_study_results.json"
    with open(json_path, "w") as f:
        json.dump(study_json, f, indent=2, default=str)
    print(f"  Results: {json_path}")

    # Visuals
    save_visuals(study, visuals_dir)

    # Run log
    elapsed_min = (time.time() - t0) / 60
    log_run(run_id, N_TRIALS, study.best_value, test_rmse, LINEAR_BASELINE, dict(best.params), elapsed_min)

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed_min:.1f} minutes")
    print(f"All outputs in: {run_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
