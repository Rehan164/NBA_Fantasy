# Optuna study for HistGradientBoostingRegressor hyperparameters.
# Loads the prebuilt feature matrix from data/nba_features.csv (built by
# build_features.py). 100-trial single-fold study over the 2022-23 holdout.
# Logs to models/optuna_run_log.csv; saves best model and study results to
# models/run_<timestamp>/.

import csv
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

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
N_TRIALS = 100

FORCED_GROUPS = ["trends", "efficiency", "missing"]
OPTIONAL_GROUPS = ["context", "schedule", "position", "dvp"]


def load_features():
    df = pd.read_csv(DATA_DIR / "nba_features.csv", parse_dates=["GAME_DATE"])
    with open(DATA_DIR / "nba_features_manifest.json") as f:
        manifest = json.load(f)
    print(f"  loaded {len(df):,} rows, {sum(len(v) for v in manifest['groups'].values())} features")
    return df, manifest


def make_objective(df, manifest, is_inner_train, is_valid, stat_targets, y_v):
    def objective(trial):
        # model hyperparameters
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        max_iter         = trial.suggest_int("max_iter", 300, 1500)
        max_depth        = trial.suggest_int("max_depth", 4, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 20, 100)
        l2_reg           = trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True)
        max_bins         = trial.suggest_int("max_bins", 128, 255)

        use_leaf_nodes = trial.suggest_categorical("use_max_leaf_nodes", [True, False])
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 31, 127) if use_leaf_nodes else None

        # always early-stop (caps max_iter automatically)
        n_iter_no_change    = trial.suggest_int("n_iter_no_change", 10, 30)
        validation_fraction = trial.suggest_float("validation_fraction", 0.10, 0.20)

        # always include rolling + the groups that earned their keep in the ablation
        features = list(manifest["groups"]["rolling"])
        for grp in FORCED_GROUPS:
            features += manifest["groups"].get(grp, [])
        for grp in OPTIONAL_GROUPS:
            use_it = trial.suggest_categorical(f"use_{grp}", [True, False])
            if use_it:
                features += manifest["groups"].get(grp, [])
        features = list(dict.fromkeys(features))

        X_it = df.loc[is_inner_train, features].values
        X_v  = df.loc[is_valid, features].values

        params = dict(
            loss="squared_error",
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_reg,
            max_bins=max_bins,
            max_leaf_nodes=max_leaf_nodes,
            early_stopping=True,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            random_state=42,
        )

        try:
            pred_v = np.zeros(len(X_v))
            for stat, weight in DK_WEIGHTS.items():
                m = HistGradientBoostingRegressor(**params)
                m.fit(X_it, stat_targets[stat]["inner_train"])
                pred_v += weight * m.predict(X_v)
            rmse = np.sqrt(mean_squared_error(y_v, pred_v))
        except Exception:
            return float("inf")

        return rmse

    return objective


def save_visuals(study, visuals_dir):
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

    # 6. Feature group usage in top 20 trials (optional groups only)
    top_20 = trials_df.nsmallest(20, "value")
    feat_usage = {}
    for grp in OPTIONAL_GROUPS:
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


def log_run(run_id, n_trials, best_val_rmse, test_rmse, linear_baseline, best_params, elapsed_min):
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


def main():
    t0 = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = run_dir / "visuals"
    visuals_dir.mkdir(exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}\n")

    # data + features
    df_clean, manifest = load_features()
    all_features = []
    for cols in manifest["groups"].values():
        all_features += cols
    all_features = list(dict.fromkeys(all_features))

    # train/valid/test splits
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

    print("\nLinear regression baseline")

    lr = make_pipeline(StandardScaler(), LinearRegression())
    lr.fit(df_clean.loc[is_train, all_features].fillna(0), df_clean.loc[is_train, "FANTASY_PTS"])
    lr_test_rmse = np.sqrt(mean_squared_error(
        df_clean.loc[is_test, "FANTASY_PTS"],
        lr.predict(df_clean.loc[is_test, all_features].fillna(0)),
    ))
    LINEAR_BASELINE = lr_test_rmse
    print(f"  Test RMSE: {lr_test_rmse:.4f}")

    print(f"\nOptuna study ({N_TRIALS} trials)")

    objective = make_objective(df_clean, manifest, is_inner_train, is_valid, stat_targets, y_v)
    study = optuna.create_study(direction="minimize", study_name="nba_fantasy_hp_tuning")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print(f"\n  Best validation RMSE: {study.best_value:.4f} (trial #{best.number})")
    print(f"  Best params:")
    for k, v in sorted(best.params.items()):
        print(f"    {k:25s} = {v}")

    print("\nRetraining best model")

    bp = best.params

    best_features = list(manifest["groups"]["rolling"])
    for grp in FORCED_GROUPS:
        best_features += manifest["groups"].get(grp, [])
    for grp in OPTIONAL_GROUPS:
        if bp.get(f"use_{grp}", False):
            best_features += manifest["groups"].get(grp, [])
    best_features = list(dict.fromkeys(best_features))

    best_model_params = dict(
        loss="squared_error",
        learning_rate=bp["learning_rate"],
        max_iter=bp["max_iter"],
        max_depth=bp["max_depth"],
        min_samples_leaf=bp["min_samples_leaf"],
        l2_regularization=bp["l2_regularization"],
        max_bins=bp["max_bins"],
        max_leaf_nodes=bp.get("max_leaf_nodes") if bp.get("use_max_leaf_nodes") else None,
        early_stopping=True,
        n_iter_no_change=bp["n_iter_no_change"],
        validation_fraction=bp["validation_fraction"],
        random_state=42,
    )

    X_train = df_clean.loc[is_train, best_features].values
    X_test  = df_clean.loc[is_test, best_features].values
    y_train = df_clean.loc[is_train, "FANTASY_PTS"].values
    y_test  = df_clean.loc[is_test, "FANTASY_PTS"].values

    use_multi = True
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

    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    test_rmse  = np.sqrt(mean_squared_error(y_test, pred_test))

    print(f"\n  Strategy:  {'multi-output' if use_multi else 'direct'}")
    print(f"  Features:  {len(best_features)}")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")

    print("\nFinal comparison")

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

    print("\nSaving model and visuals")

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

    print(f"\nDone in {elapsed_min:.1f} minutes")
    print(f"All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
