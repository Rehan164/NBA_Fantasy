# Optuna study, alternate methodology.
# Same feature matrix as optuna_nba_fantasy_study.py (data/nba_features.csv),
# but uses TimeSeriesSplit(n_splits=3) for cross-validation instead of a single
# 2022-23 holdout, and Optuna's MedianPruner to skip trials that look poor
# after fold 1. 120 trials.

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
from sklearn.model_selection import TimeSeriesSplit
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
MODEL_DIR = Path("models_v2")

DK_WEIGHTS = {"PTS": 1.0, "REB": 1.25, "AST": 1.5, "STL": 2.0, "BLK": 2.0, "TOV": -0.5, "FG3M": 0.5}
N_TRIALS = 120
N_SPLITS = 3
PRUNER_STARTUP_TRIALS = 15
PRUNER_WARMUP_STEPS = 0

# reference RMSEs for the comparison block at the end
MANUAL_BEST = 9.533
V1_OPTUNA_BEST = 9.700

FORCED_GROUPS = ["trends", "efficiency", "missing"]
OPTIONAL_GROUPS = ["context", "schedule", "position", "dvp"]


def load_features():
    df = pd.read_csv(DATA_DIR / "nba_features.csv", parse_dates=["GAME_DATE"])
    with open(DATA_DIR / "nba_features_manifest.json") as f:
        manifest = json.load(f)
    print(f"  loaded {len(df):,} rows, {sum(len(v) for v in manifest['groups'].values())} features")
    return df, manifest


def make_objective(X_full, y_full, stat_targets_full, fold_indices,
                   manifest, feature_index):
    def objective(trial):
        # model hyperparameters
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

        # multi-output decomposition vs direct fantasy-points fit
        use_multi_output = trial.suggest_categorical("use_multi_output", [True, False])

        # always include rolling + the groups that earned their keep in the ablation
        features = list(manifest["groups"]["rolling"])
        for grp in FORCED_GROUPS:
            features += manifest["groups"].get(grp, [])
        for grp in OPTIONAL_GROUPS:
            use_it = trial.suggest_categorical(f"use_{grp}", [True, False])
            if use_it:
                features += manifest["groups"].get(grp, [])
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


def save_visuals(study, visuals_dir):
    print("\n  Saving visualizations...")
    trials_df = study.trials_dataframe()
    completed = trials_df[trials_df["state"] == "COMPLETE"].copy()
    completed = completed[completed["value"] < float("inf")].sort_values("number")

    # 1. Optimization history (completed trials only -- pruned trials have no final value)
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

    # 8. Per-fold RMSE for top-10 trials
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


def log_run(run_id, n_trials, n_completed, n_pruned, best_val_rmse, best_fold_std,
            test_rmse, linear_baseline, best_params, elapsed_min):
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


def main():
    t0 = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = MODEL_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = run_dir / "visuals"
    visuals_dir.mkdir(exist_ok=True)

    print(f"v2 study -- TimeSeriesSplit({N_SPLITS})")
    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}\n")

    # data + features
    df_clean, manifest = load_features()
    all_features = []
    for cols in manifest["groups"].values():
        all_features += cols
    all_features = list(dict.fromkeys(all_features))

    # train/test split -- train portion is fed to TimeSeriesSplit
    is_train = df_clean["GAME_DATE"] < "2023-10-01"
    is_test  = df_clean["GAME_DATE"] >= "2023-10-01"

    # Sort the train portion by date so TimeSeriesSplit produces time-ordered folds
    df_train = df_clean[is_train].sort_values("GAME_DATE").reset_index(drop=True)
    df_test  = df_clean[is_test].reset_index(drop=True)

    print(f"\n  Train rows: {len(df_train):,}  (date range "
          f"{df_train['GAME_DATE'].min().date()} -> {df_train['GAME_DATE'].max().date()})")
    print(f"  Test rows:  {len(df_test):,}  (date range "
          f"{df_test['GAME_DATE'].min().date()} -> {df_test['GAME_DATE'].max().date()})")

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
        print(f"    Fold {i+1}: train {tr_dates.min().date()} -> {tr_dates.max().date()} "
              f"({len(tr_idx):,} rows)  |  val {va_dates.min().date()} -> {va_dates.max().date()} "
              f"({len(va_idx):,} rows)")

    print("\nLinear regression baseline")

    lr = make_pipeline(StandardScaler(), LinearRegression())
    lr.fit(df_train[all_features].fillna(0), df_train["FANTASY_PTS"])
    lr_test_rmse = np.sqrt(mean_squared_error(
        df_test["FANTASY_PTS"],
        lr.predict(df_test[all_features].fillna(0)),
    ))
    LINEAR_BASELINE = lr_test_rmse
    print(f"  Test RMSE: {lr_test_rmse:.4f}")

    print(f"\nOptuna study ({N_TRIALS} trials, TimeSeriesSplit({N_SPLITS}), MedianPruner)")

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
        fold_indices, manifest, feature_index,
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

    print("\nFinal comparison")

    summary = pd.DataFrame([
        {"model": "Linear Regression baseline",                "test_rmse": LINEAR_BASELINE},
        {"model": "v1 Optuna (single fold)", "test_rmse": V1_OPTUNA_BEST},
        {"model": "Manual hand-tuned (HistGB + multi-output)", "test_rmse": MANUAL_BEST},
        {"model": f"v2 Optuna (TimeSeriesSplit, {N_TRIALS} trials)", "test_rmse": test_rmse},
    ])
    summary["delta_vs_linear"]  = (summary["test_rmse"] - LINEAR_BASELINE).round(4)
    summary["pct_improvement"]  = ((LINEAR_BASELINE - summary["test_rmse"]) / LINEAR_BASELINE * 100).round(2)
    summary = summary.sort_values("test_rmse").reset_index(drop=True)
    print(summary.to_string(index=False))

    delta_manual = MANUAL_BEST - test_rmse
    delta_v1     = V1_OPTUNA_BEST - test_rmse
    print(f"\n  vs manual best:    {delta_manual:+.4f} RMSE  ({delta_manual / MANUAL_BEST * 100:+.2f}%)")
    print(f"  vs v1 Optuna best: {delta_v1:+.4f} RMSE  ({delta_v1 / V1_OPTUNA_BEST * 100:+.2f}%)")

    print("\nSaving model and visuals")

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

    print(f"\nDone in {elapsed_min:.1f} minutes")
    print(f"All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
