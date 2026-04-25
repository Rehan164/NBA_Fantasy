# NBA Fantasy Predictor — Project Journey

The chronological record of how we built, tested, and iterated on this model. Each major iteration is preserved as a snapshot in `snapshots/` so this story is reproducible.

**Final result:** Test RMSE **9.533** (multi-output decomposition, 107 features, HistGradientBoosting).
**Improvement vs linear baseline:** -0.282 / 2.83% relative.
**Test set:** 2023-24 season onward (64,358 player-game rows).

---

## Timeline at a glance

| Version | What changed | Test RMSE | Notes |
|---|---|---|---|
| v1 | Linear regression on 240 raw lag features | **9.811** | Strong baseline |
| v2 | Engineered 5 feature sets (rolling, context, trend, efficiency, full) on LinReg | 9.846 | None significantly improved on v1 |
| v3 | Per-36 minute rate normalization on LinReg | 10.147 | **Worse** — diagnosed why |
| v4 | RandomForest on rolling features | 9.942 | Tree models ≠ silver bullet |
| v5 | External features + missing teammates + HistGB + multi-output + ablation | 9.527 | The breakthrough |
| current | Simplified v5 — dropped Vegas (low ROI) and per-position (regressed) | **9.533** | Ship state |

---

## v1 — Linear baseline

We started with the simplest approach that could possibly work. For every player-game, build features from the player's last 10 games, their team's last 10 games, and their opponent's last 10 games — eight stats × ten lags × three sources = **240 raw lag features**. Standard sklearn `LinearRegression` with `StandardScaler`, `TimeSeriesSplit` for cross-validation.

**Result:** Test RMSE 9.811. CV was tight (9.465 ± 0.104) confirming we weren't overfitting on the train set — the train/test gap (9.427 → 9.811) was real-world generalization loss, not noise.

**Why this baseline matters:** Every later iteration is measured against this number. Without it we'd have no idea whether new features were actually helping.

---

## v2 — Five feature iterations on linear regression

**Hypothesis:** Smarter features will beat raw lags. We built five progressively richer feature sets and trained a separate LinReg on each:

| Feature set | Features | Train | Test | CV |
|---|---|---|---|---|
| v2 rolling (L3/L5/L10) | 72 | 9.446 | 9.881 | 9.479 |
| v3 + context (is_home, days_rest) | 75 | 9.436 | 9.881 | 9.471 |
| v4 + trend (L3 − L10) | 80 | 9.446 | 9.881 | 9.479 |
| v5 + efficiency (FGA, FG%, +/−) | 87 | 9.412 | 9.846 | 9.447 |
| v6 full (all of above) | 98 | 9.403 | 9.846 | 9.439 |

**Result:** None of these significantly beat v1 (9.811). The "full" 98-feature set still came in slightly worse on test (9.846).

**Diagnosis:** Linear regression had hit a ceiling. Throwing more features at it didn't help because:
1. Many features were redundant (rolling averages of similar signals)
2. The relationships we cared about (rate × minutes → fantasy points) are *multiplicative*, but linear regression can only sum features

This was the moment we knew we needed either a different model class or fundamentally new information.

---

## v3 — The per-36 experiment (and why it failed)

**Hypothesis:** Normalize each production stat by minutes played, so the model could separate *rate* (how good per minute) from *opportunity* (how many minutes). Production stats become per-36-minute rates, with rolling MIN kept as its own feature.

**Result:** Test RMSE 10.147 — **worse than v1 by 0.336**.

**Why it failed:** Linear regression cannot reconstruct `points = rate × minutes` from separate `rate` and `minutes` columns. To predict raw points (the actual target's largest component), the model needs to multiply them. LinReg can only sum.

This was the **clinical evidence** that the model class — not the features — was the bottleneck. It directly motivated the v4 switch.

---

## v4 — Random Forest

**Hypothesis:** Tree models can capture non-linear interactions and the multiplicative relationships LinReg couldn't.

Same rolling feature set as v2's "rolling" config, just swap `LinearRegression` → `RandomForestRegressor(n_estimators=100)`.

**Result:** Test RMSE 9.942 — *worse* than the linear baseline. RF train RMSE was 3.541 (massive overfit) and the model couldn't find anything useful in test.

**Diagnosis:** Pure tree depth and ensembling weren't enough. The features themselves weren't capturing what mattered. This pushed us to invest heavily in feature engineering instead of more model tweaks.

---

## v5 — External features, the breakthrough, and the full toolkit

This is the version where we actually moved the needle. Several changes happened in parallel:

### New feature sources
- **`CommonPlayerInfo` (nba_api):** position, height, draft year for 2,648 players
- **Vegas odds:** scraped 19,636 games from sportsbookreviewsonline.com (later supplemented with Kaggle for 2023-25)
- **`BoxScoreAdvancedV2` script:** ready but never produced data (NBA API rate-limited us after 7 hours)

### Derived features (no external data)
- **Missing teammates** — for each (team, game), sum the rolling-L10 minutes of all players who actually appeared, compare to a rolling baseline. Deficit = rotation minutes missing = opportunity for the players who *did* play.
- **Schedule density** — back-to-back indicator, games-in-last-4-days, games-in-last-7-days
- **Defense vs Position (DvP)** — opponent's rolling FP allowed to this player's position bucket

### Modeling changes
- Switched to `HistGradientBoostingRegressor` (handles NaN natively — important since Vegas had partial coverage)
- Added a **feature-group ablation** to measure each bucket's marginal impact
- Added **multi-output decomposition** — train one HistGB per stat (PTS/REB/AST/STL/BLK/TOV/FG3M), combine via DK formula
- Tested **RandomizedSearchCV** hyperparameter tuning
- Tested **per-position separate models** (Guards/Forwards/Centers)

### The killer ablation result

| Added | Test RMSE | Δ |
|---|---|---|
| rolling only | 9.840 | baseline |
| + game context | 9.833 | -0.008 |
| + trend | 9.836 | +0.003 |
| + efficiency | 9.812 | -0.024 |
| **+ missing teammates** | **9.567** | **-0.249** ⭐ |
| + schedule density | 9.568 | -0.002 |
| + vegas | 9.560 | -0.009 |
| + position | 9.547 | -0.012 |
| + DvP | 9.542 | -0.014 |

**Missing teammates contributed -0.249 RMSE alone — 84% of the entire improvement from the ablation.** Every other feature group combined contributed about as much as the noise.

### Final v5 model lineup

| Model | Test RMSE |
|---|---|
| **Multi-output decomposition** | **9.527** |
| HistGB tuned (RandomizedSearchCV, 60 fits) | 9.538 |
| HistGB default | 9.542 |
| Per-position separate models | 9.553 |
| Random Forest (rolling only) | 9.942 |

---

## current — Simplification

After v5, we audited what was earning its keep:

**Dropped:**
- **Vegas data** — only contributed -0.009 RMSE in the ablation. Required us to maintain a scraper, manage Kaggle integration, handle test-period coverage (only 68%), and add three features. Not worth the complexity.
- **Per-position models** — actually *regressed* (+0.011 RMSE). Splitting the data destroys cross-position patterns the global model learns from.
- **Hyperparameter tuning step** — gave -0.004 RMSE. Negligible. Default HistGB params were already strong.

**Kept:**
- All derived features (missing teammates, schedule density, trend, efficiency)
- Player position + DvP from `CommonPlayerInfo`
- HistGradientBoosting + multi-output decomposition

**Final:** Test RMSE **9.533** — only 0.006 worse than the most complex v5 variant, with significantly less code and zero runtime dependency on external data sources.

---

## What worked, in priority order

1. **Missing teammates feature (-0.249 RMSE).** Pure derivation from existing data — no external API, no scraping. Detects rotation absences and the opportunity boost they create for the remaining players.
2. **HistGradientBoosting + multi-output decomposition (-0.013 RMSE).** Each stat gets its own specialized model; combined via the DK formula.
3. **Player position + DvP (-0.026 RMSE combined).** Position alone barely helped, but it unlocked DvP — opponent FP allowed to the player's position.
4. **Player rolling efficiency stats (-0.018 RMSE).** Rolling FGA / FG% / PLUS_MINUS captured a complementary "production quality" signal.

---

## What didn't work (and why we now understand it)

1. **Per-36 normalization (failed badly).** Linear regression cannot multiply features. Once we moved to tree models the issue went away, but by then the *ratio* signals weren't needed because trees discovered the same relationship from raw features.
2. **Trend features L3 − L10 (no signal).** The model already has L3, L5, and L10 separately — it can compute the difference internally if it matters.
3. **Schedule density (B2B, games-in-7d) (no signal).** Strong intuition that load management would matter, but coaches manage rest case-by-case in ways our features couldn't capture.
4. **Vegas spread/total/implied (-0.009, basically noise).** Team rolling stats already capture most of the team-scoring signal. Vegas is largely redundant once you have those.
5. **Per-position separate models (regressed).** Splitting the data forfeits cross-position patterns. Global model > position-specific models in our setting.
6. **Hyperparameter tuning (negligible).** With dominant features in place, default HistGB params were already near-optimal. The bottleneck was features, not hyperparameters.

---

## Lessons we'd take to the next project

1. **Build a strong, simple baseline first.** v1's 9.811 was the ruler everything else was measured against. Without a baseline, "improvement" claims are unfalsifiable.

2. **Diagnose failures before adding more features.** Per-36 didn't fail because the idea was bad — it failed because the model class couldn't represent the relationship. Knowing *why* something failed redirected the entire effort.

3. **Treat features and models as orthogonal experiments.** v4 swapped the model with no feature changes. Result was *worse*. That isolated the conclusion: "trees alone won't save us, we need better features."

4. **Look for derived features before paying for external data.** Our biggest single win (-0.249 RMSE) came from a clever derivation of data we already had. The most-hyped external data source (Vegas) contributed -0.009.

5. **Quantify each component's marginal value.** The ablation cell in v5 made it obvious which features earned their keep and which didn't. Without it we'd still be carrying around Vegas, per-position models, and the rest.

6. **Most "obvious" features have near-zero impact.** Game context, schedule density, trend, and Vegas were all obvious-seeming wins. Each contributed <0.02 RMSE. Intuition is a starting point; ablation is the verdict.

---

## Where we hit the ceiling

Our final 9.533 RMSE is close to the practical ceiling for this dataset. To break through to ~7-8 RMSE (the level of commercial DFS models) we'd need data sources we don't have free historical access to:

- **Real-time injury reports** — published 30 min before tip-off
- **Starting lineup news** — same window
- **Live Vegas line movement** — closing lines move on injury news
- **Player tracking data** — touches, distance traveled, defensive matchups

What remains in our error is largely irreducible game-to-game variance (a player goes 4-for-15 one night and 12-for-18 the next; foul trouble; blowout-induced bench rest). Without real-time information our model has fundamentally less signal than commercial systems.

---

## How to navigate this repo

- **Start here:** `README.md` for setup; `nba_fantasy_model.ipynb` for the current model
- **Visual story:** `feature_engineering_review.ipynb` — same content as this doc but with charts
- **Timeline trace:** `snapshots/v1_linear_baseline.ipynb` through `snapshots/v5_external_features.ipynb` — each major iteration preserved as a runnable notebook
- **Data pipeline:** `collect_all.py` orchestrates all collection scripts
