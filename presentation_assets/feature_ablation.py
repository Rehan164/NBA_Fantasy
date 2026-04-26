# Forward ablation: start with rolling features, add one group at a time, record test RMSE.
# Single-output HistGB for speed. Final-model architecture (multi-output) is reported separately
# for the absolute number; here we just want the *relative* impact of each group.

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

DATA_DIR = Path("../data")
OUT_DIR = Path(".")

df = pd.read_csv(DATA_DIR / "nba_features.csv", parse_dates=["GAME_DATE"])
manifest = json.load(open(DATA_DIR / "nba_features_manifest.json"))

is_train = df["GAME_DATE"] < "2023-10-01"
is_test  = ~is_train
y_train = df.loc[is_train, "FANTASY_PTS"].values
y_test  = df.loc[is_test,  "FANTASY_PTS"].values

# order of group additions for the ablation
GROUP_ORDER = [
    "rolling",      # baseline
    "context",
    "trends",
    "efficiency",
    "missing",      # the killer feature
    "schedule",
    "position",
    "dvp",
]

PARAMS = dict(max_iter=500, learning_rate=0.05, max_depth=8,
              min_samples_leaf=20, random_state=42)

results = []
features_so_far = []
prev_rmse = None

print(f"Forward ablation, single-output HistGB, train={is_train.sum():,}  test={is_test.sum():,}\n")
print(f"{'Step':<30} {'#feat':>6} {'Test RMSE':>10} {'delta':>11}")
print("-" * 65)

t0 = time.time()
for grp in GROUP_ORDER:
    new_cols = [c for c in manifest["groups"][grp] if c not in features_so_far]
    features_so_far += new_cols
    features_so_far = list(dict.fromkeys(features_so_far))

    X_tr = df.loc[is_train, features_so_far].values
    X_te = df.loc[is_test,  features_so_far].values

    m = HistGradientBoostingRegressor(**PARAMS)
    m.fit(X_tr, y_train)
    rmse = float(np.sqrt(mean_squared_error(y_test, m.predict(X_te))))

    delta = "" if prev_rmse is None else f"{rmse - prev_rmse:+.4f}"
    label = f"+ {grp}" if results else f"{grp} only (baseline)"
    print(f"{label:<30} {len(features_so_far):>6}  {rmse:>9.4f}  {delta:>11}")

    results.append({
        "step": label,
        "group_added": grp,
        "n_features": len(features_so_far),
        "test_rmse": rmse,
        "delta_vs_prev": (rmse - prev_rmse) if prev_rmse is not None else 0.0,
    })
    prev_rmse = rmse

elapsed = (time.time() - t0) / 60
print(f"\nElapsed: {elapsed:.1f} min")

with open(OUT_DIR / "ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Wrote {OUT_DIR / 'ablation_results.json'}")

# Chart: incremental RMSE delta per group (the "what each group earned" view)
fig, ax = plt.subplots(figsize=(10, 5))
labels = [r["group_added"] for r in results[1:]]   # skip the baseline (no delta)
deltas = [r["delta_vs_prev"] for r in results[1:]]
colors = ["#2ca02c" if d < 0 else "#d62728" for d in deltas]

bars = ax.bar(labels, deltas, color=colors, edgecolor="black")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Δ Test RMSE  (negative = better)")
ax.set_title("Feature group ablation — incremental RMSE impact")
ax.grid(True, alpha=0.3, axis="y")
for b, d in zip(bars, deltas):
    ax.annotate(f"{d:+.3f}", xy=(b.get_x() + b.get_width()/2, d),
                xytext=(0, -12 if d < 0 else 4), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold")
plt.xticks(rotation=20)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_incremental.png", dpi=150)
plt.close(fig)
print(f"Wrote {OUT_DIR / 'ablation_incremental.png'}")

# Chart: cumulative RMSE as we add groups
fig, ax = plt.subplots(figsize=(10, 5))
xs = list(range(len(results)))
rmses = [r["test_rmse"] for r in results]
ax.plot(xs, rmses, marker="o", color="#1f77b4", linewidth=2, markersize=8)
for i, r in enumerate(results):
    ax.annotate(f"{r['test_rmse']:.3f}", xy=(i, r["test_rmse"]),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([r["step"] for r in results], rotation=20, ha="right")
ax.set_ylabel("Test RMSE")
ax.set_title("Cumulative test RMSE as feature groups are added")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "ablation_cumulative.png", dpi=150)
plt.close(fig)
print(f"Wrote {OUT_DIR / 'ablation_cumulative.png'}")
