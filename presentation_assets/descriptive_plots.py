# Two descriptive plots for slide 3:
#   1. Histogram of FANTASY_PTS across all clean player-games
#   2. Scatter of FANTASY_PTS vs MIN (50K random sample for readability)

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("../data")
OUT_DIR = Path(".")

df = pd.read_csv(DATA_DIR / "nba_features.csv", parse_dates=["GAME_DATE"])
print(f"loaded {len(df):,} clean player-games")
print(f"FP stats: mean={df['FANTASY_PTS'].mean():.2f}  median={df['FANTASY_PTS'].median():.2f}  "
      f"std={df['FANTASY_PTS'].std():.2f}  max={df['FANTASY_PTS'].max():.2f}")

# 1) FP histogram
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["FANTASY_PTS"], bins=60, color="#1f77b4", edgecolor="white", alpha=0.85)
ax.axvline(df["FANTASY_PTS"].median(), color="red", linestyle="--", linewidth=2,
           label=f"Median = {df['FANTASY_PTS'].median():.1f}")
ax.axvline(df["FANTASY_PTS"].mean(), color="darkorange", linestyle="--", linewidth=2,
           label=f"Mean   = {df['FANTASY_PTS'].mean():.1f}")
ax.set_xlabel("DraftKings Fantasy Points (per player-game)")
ax.set_ylabel("Player-game count")
ax.set_title("Distribution of fantasy points\n(clean player-games, MIN >= 10, n=563,774)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "fp_distribution.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT_DIR / 'fp_distribution.png'}")

# 2) FP vs MIN scatter
sample = df.sample(50_000, random_state=42)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(sample["MIN"], sample["FANTASY_PTS"], s=2, alpha=0.15, color="#1f77b4")
# overlay simple OLS line
slope, intercept = np.polyfit(sample["MIN"], sample["FANTASY_PTS"], 1)
xs = np.linspace(sample["MIN"].min(), sample["MIN"].max(), 100)
ax.plot(xs, slope * xs + intercept, color="red", linewidth=2,
        label=f"OLS fit: FP = {slope:.2f}*MIN + {intercept:.2f}")
ax.set_xlabel("Minutes played")
ax.set_ylabel("DraftKings Fantasy Points")
ax.set_title("Fantasy points vs minutes played\n(50K random sample of player-games)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "fp_vs_min.png", dpi=150)
plt.close(fig)
print(f"wrote {OUT_DIR / 'fp_vs_min.png'}")

# correlation
corr = df["MIN"].corr(df["FANTASY_PTS"])
print(f"\nMIN-FP Pearson correlation: {corr:.3f}")
