import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")

DK_WEIGHTS = {
    "PTS":  1.00,
    "REB":  1.25,
    "AST":  1.50,
    "STL":  2.00,
    "BLK":  2.00,
    "TOV": -0.50,
    "FG3M": 0.50,
}

df = pd.read_csv(DATA_DIR / "nba_features.csv", parse_dates=["GAME_DATE"])
manifest = json.load(open(DATA_DIR / "nba_features_manifest.json"))

feature_cols = []
for cols in manifest["groups"].values():
    feature_cols += cols
feature_cols = list(dict.fromkeys(feature_cols))

target = manifest["target"]
components = manifest["target_components"]

print(f"rows:       {len(df):,}")
print(f"features:   {len(feature_cols)}")
print(f"target:     {target}")
print(f"components: {components}")

is_train = df["GAME_DATE"] < "2023-10-01"
is_test  = ~is_train

X_train = df.loc[is_train,  feature_cols]
X_test  = df.loc[is_test,   feature_cols]
y_train = df.loc[is_train,  target]
y_test  = df.loc[is_test,   target]

print(f"\ntrain: {len(X_train):,} rows")
print(f"test:  {len(X_test):,} rows")
print(f"train date range: {df.loc[is_train, 'GAME_DATE'].min().date()} -> {df.loc[is_train, 'GAME_DATE'].max().date()}")
print(f"test  date range: {df.loc[is_test, 'GAME_DATE'].min().date()} -> {df.loc[is_test, 'GAME_DATE'].max().date()}")

# Linear baseline (all features, fillna 0)
print("\n--- Linear regression baseline ---")
lr = make_pipeline(StandardScaler(), LinearRegression())
lr.fit(X_train.fillna(0), y_train)
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr.predict(X_test.fillna(0))))
lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr.predict(X_train.fillna(0))))
print(f"LinReg train RMSE: {lr_train_rmse:.3f}")
print(f"LinReg test  RMSE: {lr_test_rmse:.3f}")

# Multi-output HistGB
print("\n--- HistGradientBoosting multi-output (sklearn defaults+) ---")
y_train_components = {s: df.loc[is_train, s].astype(float).values for s in components}
y_test_components  = {s: df.loc[is_test,  s].astype(float).values for s in components}

default_params = dict(
    max_iter=500,
    learning_rate=0.05,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
)

pred_tr = np.zeros(len(X_train))
pred_te = np.zeros(len(X_test))

for stat, weight in DK_WEIGHTS.items():
    m = HistGradientBoostingRegressor(**default_params)
    m.fit(X_train, y_train_components[stat])
    ptr = m.predict(X_train)
    pte = m.predict(X_test)
    pred_tr += weight * ptr
    pred_te += weight * pte
    rmse_te = np.sqrt(mean_squared_error(y_test_components[stat], pte))
    print(f"  {stat:5s} (w={weight:+.2f})  test RMSE={rmse_te:.3f}")

default_train_rmse = np.sqrt(mean_squared_error(y_train, pred_tr))
default_test_rmse  = np.sqrt(mean_squared_error(y_test,  pred_te))
print()
print(f"FANTASY_PTS train RMSE: {default_train_rmse:.3f}")
print(f"FANTASY_PTS test  RMSE: {default_test_rmse:.3f}")

# Save results
results = {
    "n_rows": int(len(df)),
    "n_features": int(len(feature_cols)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "linreg_test_rmse": float(lr_test_rmse),
    "linreg_train_rmse": float(lr_train_rmse),
    "histgb_test_rmse": float(default_test_rmse),
    "histgb_train_rmse": float(default_train_rmse),
    "improvement_vs_linreg": float(lr_test_rmse - default_test_rmse),
    "pct_improvement": float((lr_test_rmse - default_test_rmse) / lr_test_rmse * 100),
}
print("\n--- Summary ---")
for k, v in results.items():
    print(f"  {k}: {v}")

with open("reproduce_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to reproduce_results.json")
