# Train each model class on the SAME final 107-feature matrix to get
# apples-to-apples test RMSE for the model-progression slide.

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("../data")
OUT_DIR = Path(".")

DK_WEIGHTS = {"PTS": 1.0, "REB": 1.25, "AST": 1.5, "STL": 2.0, "BLK": 2.0,
              "TOV": -0.5, "FG3M": 0.5}

df = pd.read_csv(DATA_DIR / "nba_features.csv", parse_dates=["GAME_DATE"])
manifest = json.load(open(DATA_DIR / "nba_features_manifest.json"))
feature_cols = []
for cols in manifest["groups"].values():
    feature_cols += cols
feature_cols = list(dict.fromkeys(feature_cols))

is_train = df["GAME_DATE"] < "2023-10-01"
is_test  = ~is_train
y_train = df.loc[is_train, "FANTASY_PTS"].values
y_test  = df.loc[is_test,  "FANTASY_PTS"].values

print(f"train={is_train.sum():,}  test={is_test.sum():,}  features={len(feature_cols)}\n")

results = []

# ---------- LinReg
print("LinReg...")
t0 = time.time()
lr = make_pipeline(StandardScaler(), LinearRegression())
lr.fit(df.loc[is_train, feature_cols].fillna(0), y_train)
pred = lr.predict(df.loc[is_test, feature_cols].fillna(0))
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
results.append({"model": "Linear Regression", "test_rmse": rmse, "elapsed_s": time.time() - t0})
print(f"  RMSE = {rmse:.4f}  ({time.time()-t0:.1f}s)")

# ---------- RandomForest (smaller config so it actually finishes)
print("RandomForest (n=100, max_depth=15)...")
t0 = time.time()
rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(df.loc[is_train, feature_cols].fillna(0), y_train)
pred = rf.predict(df.loc[is_test, feature_cols].fillna(0))
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
results.append({"model": "Random Forest", "test_rmse": rmse, "elapsed_s": time.time() - t0})
print(f"  RMSE = {rmse:.4f}  ({time.time()-t0:.1f}s)")

# ---------- HistGB single-output (predict FANTASY_PTS directly)
print("HistGB (single-output, direct FP)...")
t0 = time.time()
hgb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=8,
                                    min_samples_leaf=20, random_state=42)
hgb.fit(df.loc[is_train, feature_cols], y_train)
pred = hgb.predict(df.loc[is_test, feature_cols])
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
results.append({"model": "HistGB single-output", "test_rmse": rmse, "elapsed_s": time.time() - t0})
print(f"  RMSE = {rmse:.4f}  ({time.time()-t0:.1f}s)")

# ---------- HistGB multi-output (final model)
print("HistGB (multi-output, predict each component then combine)...")
t0 = time.time()
pred_te = np.zeros(len(y_test))
for stat, weight in DK_WEIGHTS.items():
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=8,
                                      min_samples_leaf=20, random_state=42)
    m.fit(df.loc[is_train, feature_cols], df.loc[is_train, stat].astype(float).values)
    pred_te += weight * m.predict(df.loc[is_test, feature_cols])
rmse = float(np.sqrt(mean_squared_error(y_test, pred_te)))
results.append({"model": "HistGB multi-output (final)", "test_rmse": rmse, "elapsed_s": time.time() - t0})
print(f"  RMSE = {rmse:.4f}  ({time.time()-t0:.1f}s)")

# Save
print("\n--- Summary ---")
for r in results:
    print(f"  {r['model']:<35} RMSE={r['test_rmse']:.4f}  ({r['elapsed_s']:.0f}s)")

with open(OUT_DIR / "model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote {OUT_DIR / 'model_comparison.json'}")

# Chart with naive baselines + model results
baselines = json.load(open(OUT_DIR / "baselines.json"))
naive_rows = [
    ("Predict league mean FP", baselines["mean_fp"]["test_rmse"], "naive"),
    ("Predict player career mean", baselines["player_career_mean"]["test_rmse"], "naive"),
    ("Predict player L10 rolling avg", baselines["player_L10_rolling"]["test_rmse"], "naive"),
]
ml_rows = [(r["model"], r["test_rmse"], "ml") for r in results]
all_rows = naive_rows + ml_rows

fig, ax = plt.subplots(figsize=(11, 5))
labels = [r[0] for r in all_rows]
values = [r[1] for r in all_rows]
colors = ["#7f7f7f" if r[2] == "naive" else "#1f77b4" for r in all_rows]
# highlight final model
colors[-1] = "#2ca02c"

bars = ax.barh(labels, values, color=colors, edgecolor="black")
ax.invert_yaxis()
ax.set_xlabel("Test RMSE  (lower = better)")
ax.set_title("Model progression: naive baselines vs. ML models\n(all evaluated on the same 2023-2026 holdout)")
ax.grid(True, alpha=0.3, axis="x")
for b, v in zip(bars, values):
    ax.annotate(f"{v:.3f}", xy=(v, b.get_y() + b.get_height()/2),
                xytext=(5, 0), textcoords="offset points", va="center", fontsize=10)
ax.set_xlim(0, max(values) * 1.1)
fig.tight_layout()
fig.savefig(OUT_DIR / "model_progression.png", dpi=150)
plt.close(fig)
print(f"Wrote {OUT_DIR / 'model_progression.png'}")
