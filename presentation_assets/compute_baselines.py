# Compute proper naive baselines for the model-progression slide.
# Also produces the per-season player-games chart for the data-source slide.

import json
from pathlib import Path

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

DATA_DIR = Path("../data")
OUT_DIR = Path(".")

DK_WEIGHTS = {"PTS": 1.0, "REB": 1.25, "AST": 1.5, "STL": 2.0, "BLK": 2.0, "TOV": -0.5, "FG3M": 0.5}

# ---------- load
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

results = {}

# ---------- baseline 1: predict the global mean FP from train
mean_fp_train = float(y_train.mean())
pred_mean = np.full_like(y_test, mean_fp_train, dtype=float)
rmse_mean = float(np.sqrt(mean_squared_error(y_test, pred_mean)))
results["mean_fp"] = {
    "label": "Predict league-mean FP for every player-game",
    "value_used": mean_fp_train,
    "test_rmse": rmse_mean,
}
print(f"[1] mean-FP baseline:           RMSE = {rmse_mean:.3f}  (predict {mean_fp_train:.2f} for every game)")

# ---------- baseline 2: predict the player's career-mean FP (computed from train only)
player_mean_train = (
    df.loc[is_train].groupby("PLAYER_ID")["FANTASY_PTS"].mean()
)
pred_player_mean = (
    df.loc[is_test, "PLAYER_ID"].map(player_mean_train).fillna(mean_fp_train).values
)
rmse_player_mean = float(np.sqrt(mean_squared_error(y_test, pred_player_mean)))
results["player_career_mean"] = {
    "label": "Predict the player's career mean FP (from train)",
    "test_rmse": rmse_player_mean,
}
print(f"[2] player-career-mean baseline: RMSE = {rmse_player_mean:.3f}")

# ---------- baseline 3: predict the player's L10 rolling FP (already a feature)
# Use existing player_FANTASY_PTS_L10 column
l10_col = "player_FANTASY_PTS_L10"
if l10_col in df.columns:
    pred_l10 = df.loc[is_test, l10_col].fillna(mean_fp_train).values
    rmse_l10 = float(np.sqrt(mean_squared_error(y_test, pred_l10)))
    results["player_L10_rolling"] = {
        "label": "Predict the player's last-10-game average FP",
        "test_rmse": rmse_l10,
    }
    print(f"[3] player-L10-rolling baseline: RMSE = {rmse_l10:.3f}")
else:
    print("[3] L10 col missing")

# Also compute L5 and L3 since they exist
for col, key, lbl in [
    ("player_FANTASY_PTS_L5", "player_L5_rolling", "Predict player's last-5-game avg FP"),
    ("player_FANTASY_PTS_L3", "player_L3_rolling", "Predict player's last-3-game avg FP"),
]:
    if col in df.columns:
        pred = df.loc[is_test, col].fillna(mean_fp_train).values
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results[key] = {"label": lbl, "test_rmse": rmse}
        print(f"    {lbl}: RMSE = {rmse:.3f}")

# ---------- ML baseline 4: linear regression on full engineered features
print("\nFitting LinReg on full engineered features...")
lr = make_pipeline(StandardScaler(), LinearRegression())
lr.fit(df.loc[is_train, feature_cols].fillna(0), y_train)
rmse_lr = float(np.sqrt(mean_squared_error(y_test, lr.predict(df.loc[is_test, feature_cols].fillna(0)))))
results["linreg_full_features"] = {
    "label": "Linear regression on 107 engineered features",
    "test_rmse": rmse_lr,
}
print(f"[4] LinReg (107 features):       RMSE = {rmse_lr:.3f}")

# ---------- final model: HistGB multi-output (default-ish params)
print("\nFitting HistGB multi-output...")
default_params = dict(max_iter=500, learning_rate=0.05, max_depth=8,
                     min_samples_leaf=20, random_state=42)
pred_te = np.zeros(len(y_test))
for stat, weight in DK_WEIGHTS.items():
    m = HistGradientBoostingRegressor(**default_params)
    m.fit(df.loc[is_train, feature_cols], df.loc[is_train, stat].astype(float).values)
    pred_te += weight * m.predict(df.loc[is_test, feature_cols])
rmse_hgb = float(np.sqrt(mean_squared_error(y_test, pred_te)))
results["histgb_multi_output"] = {
    "label": "HistGradientBoosting multi-output (final)",
    "test_rmse": rmse_hgb,
}
print(f"[5] HistGB multi-output:         RMSE = {rmse_hgb:.3f}")

# ---------- save
with open(OUT_DIR / "baselines.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nWrote {OUT_DIR / 'baselines.json'}")

# ---------- per-season player-games chart for slide 2
print("\nProducing per-season chart...")
plog = pd.read_csv(DATA_DIR / "nba_player_game_logs.csv")
plog["GAME_DATE"] = pd.to_datetime(plog["GAME_DATE"])
# season label: e.g. 2023-11 → 2023-24
plog["season_start"] = np.where(plog["GAME_DATE"].dt.month >= 7,
                                 plog["GAME_DATE"].dt.year,
                                 plog["GAME_DATE"].dt.year - 1)
season_counts = plog.groupby("season_start").size()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(season_counts.index, season_counts.values, marker="o", color="#1f77b4")
ax.fill_between(season_counts.index, season_counts.values, alpha=0.2, color="#1f77b4")
ax.set_xlabel("Season start year")
ax.set_ylabel("Player-games")
ax.set_title("Player-games collected per NBA season (1999-2026)")
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1999, 2027, 2))
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(OUT_DIR / "player_games_per_season.png", dpi=150)
plt.close(fig)
print(f"Wrote {OUT_DIR / 'player_games_per_season.png'}")
print(f"\nSeason counts:\n{season_counts.to_string()}")
