# NBA Fantasy Predictor — Final Presentation Outline

**Course:** BA576 — Machine Learning for Business Analytics
**Format:** 10-minute final presentation + Q&A
**Deliverable:** This outline is the spec for the slide deck. It tells the designer what content goes on each slide and what visuals to produce. It does NOT prescribe slide layout, fonts, or color palettes.

---

## Headline numbers anchor (single source of truth)

These are the canonical numbers for the deck. Re-derived from a fresh notebook run on 2026-04-26.

| Quantity | Value |
|---|---|
| Train set | **497,892** player-games (Nov 1999 → Sept 2023) |
| Test set | **65,882** player-games (Oct 2023 → April 2026) |
| Features | **107** across 8 conceptual groups |
| Naive baseline (predict league-mean FP) | **14.00** test RMSE |
| Final model (HistGB multi-output) | **9.55** test RMSE |
| Absolute improvement vs. baseline | **−4.45 RMSE** |
| Relative improvement vs. baseline | **~32% reduction in RMSE** |
| Train RMSE of final model | 9.06 (small generalization gap → no severe overfit) |

> **Note on the Optuna number:** an Optuna v2 study is currently running. If it finds a better parameter set, swap 9.55 → the Optuna result throughout the deck. If it finds nothing meaningful, keep 9.55 and use Slide 8 to honestly report "Optuna confirmed our hand-tuned defaults were near-optimal."

---

## Slide 1 — Title + The Problem

**Goal:** in 30 seconds, anchor the audience on what we're predicting and why anyone cares.

**Title:** Predicting NBA DraftKings Fantasy Points

**Beats:**
- DFS is a multi-billion-dollar prediction market. DraftKings runs daily contests where you build a 9-player NBA lineup with a salary cap, and your lineup wins money based on how many fantasy points your players actually score that night.
- Same model also wins your friends' season-long league (and might save you from whatever cursed punishment your group has for last place).
- **The prediction problem:** given everything we know about a player and matchup *before tip-off*, predict their fantasy point total for that game.

**Centerpiece visual:** the DK formula prominently displayed.

> **DK Fantasy Points = 1·PTS + 1.25·REB + 1.5·AST + 2·STL + 2·BLK − 0.5·TOV + 0.5·FG3M**

**Visuals to produce:** none. Optional small DraftKings or NBA logo.

---

## Slide 2 — Data Sources

**Goal:** in 45 seconds, communicate where the data came from, what's in it, and how much we have. Establish credibility through scale + recency.

**Header:** "The Data — NBA's full box-score record, direct from the source"

**Beats:**
- **Source:** NBA Stats API via the `nba_api` Python package (free, public, official — same API powering nba.com/stats). Rate-limited to ~1.6 req/sec.
- **Three endpoints:** `LeagueGameLog` (player + team modes) and `CommonPlayerInfo` for player metadata.
- **Scale (the wow numbers):**
  - 673,733 player-games
  - 32,443 unique games
  - 2,653 unique players
  - 27 seasons (1999-00 through 2025-26)
  - 24 raw box-score columns per player-game
- **Cleanliness:** official source = essentially zero missing data. Cleaning was almost entirely format normalization.

**Visuals to produce:**
- ✅ `player_games_per_season.png` — line chart, player-games per season (1999-2026). Confirms steady ~25K/season coverage with visible dips for the 2011 lockout (20,758) and 2019 COVID-shortened year (20,501).

**Explicit non-content:** Vegas / SBRO scraper and `BoxScoreAdvancedV2` are intentionally NOT here — they belong to the feature-engineering story on Slides 4-5.

---

## Slide 3 — Building the Prediction Problem

**Goal:** in 60 seconds, explain how raw box scores become a supervised-learning dataset, including the train/test split and one descriptive picture of the target.

**Header:** "Building the prediction problem"

**Beats:**
1. **Target construction:** computed row-by-row using the DK formula on the raw box-score columns. Worked example: 30-pt, 8-reb, 5-ast, 1-stl, 0-blk, 3-tov, 2-3PM line = **49 FP**.
2. **Time-based train/test split (most important beat):**
   - Train: pre-Oct 2023 → 497,892 player-games
   - Test: Oct 2023 → April 2026 → 65,882 player-games (2 full seasons + current)
   - **Why time-based, not random?** A random split would leak future information into training. Time-based mimics real deployment: train on the past, predict the future.
3. **Cleaning that mattered:**
   - Drop player-games with `MIN < 10` (DNPs and garbage-time = noise)
   - Drop the first ~10 games of each player's career so rolling features have history
   - Net usable rows after cleaning: **563,774** player-games

**Visuals to produce:**
- ✅ `fp_distribution.png` — histogram of FANTASY_PTS across all clean player-games (median 21.5, mean 23.6, std 13.2, max 107.75). Shows the right-skew of DFS scoring.
- ✅ `fp_vs_min.png` — scatter of FP vs MIN on a 50K random sample with the OLS fit overlaid. Pearson correlation 0.754. Foreshadows the missing-teammates story on Slide 5 (more minutes → more FP, but with significant scatter at every MIN level).

**Why these two plots:** together they answer "what does the target look like?" and "what's the dominant input signal?" in two glances.

---

## Slide 4 — Feature Engineering Overview

**Goal:** in 60 seconds, show the structure of our feature engineering work. Set up Slide 5 (the ablation) by naming the 8 groups upfront.

**Header:** "Feature engineering — 107 features across 8 conceptual groups"

**Centerpiece:** an 8-row grid/table of feature groups. The "missing" row should be visually emphasized (star, highlight) to foreshadow Slide 5.

| Group | # features | What it captures | Example feature |
|---|---|---|---|
| Rolling player/team/opp stats | 75 | Recent form across 3, 5, 10-game windows | `player_PTS_L10`, `opp_score_L5` |
| Game context | 3 | Home/away, days rest, opponent days rest | `is_home`, `days_rest` |
| Trends | 8 | L3 minus L10 — "are they hot/cold lately?" | `player_PTS_trend` |
| Rolling efficiency | 9 | FGA volume, shooting %, plus/minus | `player_FG_PCT_L10` |
| **Missing teammates** ⭐ | 3 | Rotation minutes absent vs team's baseline | `missing_min_deficit` |
| Schedule density | 3 | Back-to-back, games in last 4/7 days | `is_b2b` |
| Position + bio | 5 | G/F/C bucket, height, years experience | `pos_C`, `height_in` |
| Defense vs Position | 1 | Opp's L20 FP allowed to this player's pos | `dvp_L20` |

**Anti-leakage callout:**
> "Every rolling feature uses `.shift(1)` before averaging — the model never sees the current game's stats. Same for team and opponent rolling features."

**External-data shoutout (one bullet):**
> "We also collected Vegas spreads/totals (~19,600 games scraped from sportsbookreviewsonline.com + Kaggle) and `BoxScoreAdvancedV2` data. The Vegas story is on the next slide; advanced box scores got rate-limited out before producing usable coverage."

**Visuals to produce:** none beyond the table itself. The table IS the visual.

---

## Slide 5 — The Killer Feature: Missing Teammates + Ablation

**Goal:** in 90 seconds (the longest slide of the talk — this is the project's centerpiece), make the audience feel the disparity in feature impact and understand the missing-teammates derivation.

**Header:** "The killer feature: missing teammates"

**Centerpiece visual (LEFT):** ablation bar chart showing incremental ΔRMSE per feature group added (forward addition order: rolling → +context → +trends → +efficiency → +missing → +schedule → +position → +dvp).

The "+missing" bar should visually dwarf every other bar.

| Step | # features | Test RMSE | Δ vs. prev |
|---|---|---|---|
| rolling only (baseline) | 75 | 9.872 | — |
| + context | 78 | 9.869 | -0.003 |
| + trends | 86 | 9.868 | -0.002 |
| + efficiency | 95 | 9.846 | -0.022 |
| **+ missing teammates** ⭐ | 98 | **9.589** | **-0.257** |
| + schedule | 101 | 9.587 | -0.002 |
| + position | 106 | 9.574 | -0.013 |
| + dvp | 107 | 9.566 | -0.008 |

**Headline number:** "Of -0.306 total improvement from feature engineering, missing-teammates contributed -0.257 alone = **84% of the entire gain.**"

**Right side — explanation + worked example:**

> **Missing teammates feature:** for each (team, game), compute:
> 1. The total minutes the players in tonight's box averaged over their last 10 games
> 2. The team's rolling baseline of that same sum
> 3. **Deficit = baseline − actual = "rotation minutes that didn't show up tonight"**
>
> **Worked example:** Lakers play Memphis. The Lakers' normal rotation has averaged ~240 collective minutes per game over the last 10 games. Tonight LeBron rests + Davis is questionable. The players who actually take the floor have collectively averaged only ~205 minutes in their L10s. **Deficit = 35 missing rotation minutes** = three quarters' worth of opportunity that the model knows must be redistributed to the players who *did* play.
>
> The model learns: "35 missing minutes for this team → bump up Russell, Reaves, and Hachimura's predicted FP."
>
> **It's a derived feature** — no external data, no API call, no scraping. It came from staring at the box scores and asking *why are some games so much higher-scoring than the rolling average predicts?*

**Bottom strip — "what we tried and dropped" (brief, not the focus):**
- **Vegas spreads/totals/moneylines** — collected and tested, but only contributed -0.009 RMSE. Redundant with team rolling stats. **Removed from the final 107-feature model.**
- **Per-position separate models** (G/F/C trained independently) — tested, regressed by +0.011 RMSE. Splitting destroys cross-position learning. **Architecture rejected.**
- **Other low-impact feature groups still in the model** (trends, schedule density, game context) — each contributed essentially 0 RMSE individually. We kept them because they don't *hurt* and removing them would only save ~14 features out of 107. **In the model but not pulling weight.**

**Closing one-liner:**
> "**Always exhaust your derivable features before paying the complexity cost of external data.**"

**Visuals to produce:**
- ✅ `ablation_incremental.png` — bar chart of incremental ΔRMSE per group. Use a striking color for the missing-teammates bar.
- ✅ `ablation_cumulative.png` — line chart of cumulative test RMSE as groups are added. Backup slide / appendix only.

---

## Slide 6 — Models Tested: Failures Were the Most Diagnostic

**Goal:** in 60 seconds, walk through the model comparison apples-to-apples on the same 107-feature matrix and show that the failed experiments told us where the bottleneck actually was.

**Header:** "Models we tested — and what each one taught us"

**Centerpiece visual:** horizontal bar chart, sorted worst → best, all evaluated on the same 2023-26 holdout.

| Approach | Test RMSE | Notes |
|---|---|---|
| **Naive baseline** (predict league mean) | **14.00** | Comparison floor |
| Predict player career mean | 11.97 | Reference heuristic |
| Predict player L10 rolling avg | 10.30 | "Smart-naive" — already 83% of the journey |
| Linear Regression (107 features) | 9.62 | Engineered features → big jump |
| Random Forest (107 features) | 9.63 | **Worse than LinReg!** 73× the compute |
| HistGB single-output (107 features) | 9.57 | Gradient boosting found new signal |
| **HistGB multi-output (final)** | **9.55** | Per-stat specialization |

**Side callout — the failed experiments and what they taught us:**
- **Per-36 normalization on LinReg → 10.15 RMSE (worse).** Diagnostic: LinReg can't reconstruct `points = rate × minutes` from separate rate and minutes columns. Pointed us at non-linear models.
- **Random Forest barely beat LinReg.** Diagnostic: ensembling alone isn't the answer. The problem needed gradient boosting's sequential residual correction, not bagging.
- **Both failures pointed directly at HistGradientBoosting** as the right model class.

**Headline insight (bottom of slide):**
> "The L10-rolling-average heuristic alone (10.30) gets us most of the way from naive (14.00) to final (9.55). Engineered features carried the rest. **Switching from LinReg to HistGB only added the final 0.07 RMSE — about 2% of the total improvement.**"

**Visuals to produce:**
- ✅ `model_progression.png` — horizontal bar chart with naive baselines (gray), ML models (blue), final model (green-highlighted). Already produced.

---

## Slide 7 — Final Model Architecture

**Goal:** in 45 seconds, explain HistGradientBoosting + multi-output decomposition (the final architecture) and show per-stat error breakdown to demonstrate the model is well-calibrated.

**Header:** "The final model — HistGB + multi-output decomposition"

**Architecture diagram (LEFT, the visual centerpiece):**

```
[107 features] ──► 7 parallel HistGradientBoosting models
                     │
                     ├── PTS prediction  × 1.00
                     ├── REB prediction  × 1.25
                     ├── AST prediction  × 1.50
                     ├── STL prediction  × 2.00   ──► SUM ──► FP prediction
                     ├── BLK prediction  × 2.00
                     ├── TOV prediction  × −0.50
                     └── FG3M prediction × 0.50
```

**Why HistGB specifically (3 short bullets):**
- **Tree-based** → captures multiplicative interactions LinReg can't (e.g., `points = rate × minutes`)
- **Gradient boosted** → sequentially corrects residuals, so each tree fixes the previous tree's mistakes
- **Native NaN handling** → no need to impute missing rolling features for new/returning players

**Why multi-output decomposition:**
- Each component stat has different drivers: rebounds depend on size & opp pace; assists on role & teammates; blocks on position & matchup
- Training one model per stat lets each one specialize on its own signal
- Combining via the DK formula respects the actual scoring weights — STL/BLK are 2× weighted, so we want sharp predictions even though their absolute RMSE is small

**Per-stat RMSE breakdown (RIGHT, small table):**

| Stat | DK Weight | Test RMSE |
|---|---|---|
| PTS | ×1.00 | 6.10 |
| REB | ×1.25 | 2.54 |
| AST | ×1.50 | 1.91 |
| TOV | ×−0.50 | 1.21 |
| FG3M | ×0.50 | 1.31 |
| STL | ×2.00 | 0.97 |
| BLK | ×2.00 | 0.78 |
| **FANTASY_PTS combined** | — | **9.55** |

**One closing insight:**
> "Multi-output gave us +0.014 RMSE over single-output HistGB on the same features (9.566 → 9.551). Modest — but mathematically free, since per-stat predictions are valuable themselves for stat-specific lineup constraints in DFS."

**Visuals to produce:** the architecture diagram is most important — a hand-drawn or simple flowchart graphic. The per-stat table is just a table.

---

## Slide 8 — Hyperparameter Tuning with Optuna

**Goal:** in 60 seconds, explain what Optuna is, why we picked it over GridSearchCV, and what we found.

**Header:** "Hyperparameter tuning with Optuna"

**What Optuna is (one sentence):**
> Optuna is a Bayesian hyperparameter optimization library — it treats tuning as its own ML problem. After each trial it predicts which untried parameter combination is most likely to do well, then tries that next.

**Why Optuna over GridSearchCV (the rubric-relevant comparison):**
- HistGradientBoosting has 10+ tunable hyperparameters (learning_rate, max_iter, max_depth, min_samples_leaf, l2_regularization, max_bins, max_leaf_nodes, n_iter_no_change, validation_fraction, loss).
- A GridSearchCV with even 4 values per parameter = 4¹⁰ ≈ **1 million combinations**. Infeasible.
- Optuna's TPE sampler converges to a near-optimum in ~100 trials by intelligently exploring the most promising regions.

**Our methodology (key details):**
- **120 trials**, single Optuna study
- **TimeSeriesSplit(n_splits=3)** cross-validation on the train portion — each trial scored as the mean RMSE across 3 time-respecting folds
- **MedianPruner** — abandons trials that look worse than median after fold 1, saving compute
- Search space: 10 model hyperparameters + binary toggles for 4 optional feature groups + multi-output vs. direct switch
- Final model retrained on full train data with the best parameters, evaluated once on the 2023-26 holdout

**Result section — TO FILL IN once Optuna v2 study finishes:**
- Best parameters found: *[placeholder]*
- Test RMSE with Optuna best params: *[placeholder]*
- Δ vs. hand-tuned default model (9.551): *[placeholder]*

**Honest takeaway — pick the right framing once we see the result:**
- **If improvement is small:** "Optuna confirmed that our hand-tuned defaults were near-optimal. The bottleneck was features, not hyperparameters — itself a valuable finding."
- **If improvement is moderate (~0.05 RMSE):** "Optuna found a meaningfully better config — the chart shows learning rate and max_depth had the biggest impact. Our manual tuning had been too conservative."
- **If improvement is large (>0.1 RMSE):** "We had been substantially under-tuning. Optuna's main wins were [parameter X] — turns out we'd been [too conservative / aggressive] there."

**Visuals to produce (auto-generated by `optuna_nba_fantasy_study_v2.py`):**
- `optimization_history.png` — RMSE per trial + best-so-far line. **Slide centerpiece** — visually demonstrates the Bayesian optimization working over time.
- `param_importance.png` — bar chart of which hyperparameters mattered most. Sidebar.
- `learning_rate_vs_rmse.png`, `max_depth_vs_rmse.png` — scatter plots showing the search space. Backup slide only.
- `feature_group_usage.png` — % of top-20 trials that used each optional feature group. Confirms whether the ablation conclusions hold under tuning.

---

## Slide 9 — Final Results + Ceiling

**Goal:** in 60 seconds, deliver the headline result with the right comparison number, then honestly discuss where we hit the ceiling and what would be required to break through it. This earns Interpretation rubric points.

**Header:** "Final results — and where we hit the ceiling"

**Top half — the headline result table:**

| Metric | Value |
|---|---|
| Naive baseline (predict league-mean FP) | **14.00** test RMSE |
| **Final model** (HistGB + multi-output, 107 features) | **9.55** test RMSE |
| Absolute improvement | **−4.45 RMSE** |
| **Relative improvement** | **~32% reduction in RMSE** |
| Test set size | 65,882 player-games (Oct 2023 – April 2026) |
| Train RMSE of final model | 9.06 (small generalization gap → no severe overfit) |

**Bottom half — the ceiling discussion:**

**Why our 9.55 RMSE is near the practical ceiling for this dataset.** To get below ~7-8 RMSE (the level commercial DFS models hit), we'd need information we don't have public access to:
- **Real-time injury reports** (published 30 minutes before tip-off; we use last-game-played as a proxy)
- **Confirmed starting lineups** (also published ~30 min before tip-off)
- **Vegas line movement** (closing lines move on injury news; we only have opens)
- **Player tracking data** (touches, distance traveled, defensive matchups — Second Spectrum / Synergy data, paywalled)

**What's left in our error is largely irreducible game-to-game variance:**
- A player goes 4-for-15 one night and 12-for-18 the next — same shot quality, different outcomes
- Foul trouble (random matchup-dependent)
- Blowout-induced bench rest (not predictable from pre-game info)
- In-game coaching decisions (DNP-rest, garbage-time benchings)

**One-line summary:**
> "We're hitting the limit of what historical box scores alone can predict. Further improvement requires real-time information unavailable through public APIs."

**Visual idea:** Result table on the left. Right side: a "ceiling stack" diagram showing layers — *box-score signal we captured* (large block, ~9.5 RMSE worth) | *real-time info we don't have* (medium block, ~1.5 RMSE worth) | *irreducible game-to-game variance* (small block, residual floor).

---

## Slide 10 — What We Actually Learned (Lessons + Obstacles)

**Goal:** in 30 seconds, honestly tell the story of what tripped us up and what we'd do differently. This is the "lessons learned" + "biggest obstacle" combined slide. The framing is intentionally retrospective — these are mistakes we made and corrected, not pre-baked best practices we knew going in.

**Header:** "What we actually learned"

**Lesson 1 — Measure every change. Don't blind-update.**
- For weeks we changed code without a clear way to attribute improvement. We'd swap a feature, retrain, watch the test RMSE wiggle by ±0.01, and not know if the change actually helped or if we were chasing noise.
- The moment we built the **feature-group ablation** was the turning point. It killed off Vegas, per-position models, and trend features in a single analysis — and exposed missing-teammates as the only thing actually moving the number.
- **Takeaway:** never make a code change you can't quantify. If you can't say "this experiment moved RMSE by X," you don't know if it helped.

**Lesson 2 — Features beat model tuning. We learned this the hard way.**
- Our smart-naive baseline (predict the player's last-10-game average) was already at **10.30 RMSE**. We *thought* the way to beat it was to try fancier models — per-36 normalization, Random Forest, hyperparameter sweeps. Almost none of it worked. Random Forest came in at 9.63 (worse than LinReg). Per-36 made things worse, not better.
- The model-class change from LinReg to HistGB on the same features only contributed **0.07 RMSE** — about 2% of our total improvement.
- The actual unlock was the **missing-teammates feature alone (-0.257 RMSE)**, built from data we already had.
- **Takeaway:** when you're stuck, the answer is usually a better feature, not a fancier model. Invest in features before model architecture.

**Lesson 3 — Data collection has real costs we underestimated.**
- We assumed "get more data" would obviously help. It mostly didn't.
- **Vegas odds:** scraped 19,600+ games from sportsbookreviewsonline.com, integrated a Kaggle dataset for 2023+ coverage. Net contribution: **-0.009 RMSE** (basically noise).
- **Advanced box scores (`BoxScoreAdvancedV2`):** the NBA API rate-limited us out after **7 hours of collection** — we never got usable coverage and dropped it.
- Meanwhile, missing-teammates — derived from data we already had — delivered 84% of all feature-engineering gain.
- **Takeaway:** before paying the engineering cost of new data, exhaust what your existing data can tell you. External data is the most expensive feature.

**Biggest obstacle (one-line callout, satisfies the rubric "obstacles" line):**
> "Our biggest mistake was thinking we could break our ~10.5 baseline by tweaking models. Two months of model tuning got us almost nothing; one afternoon of feature derivation got us most of our improvement. We had to learn to invest where the leverage actually was."

**Closing line of the entire presentation:**
> "Final model: HistGradientBoosting + multi-output decomposition + 107 features. Test RMSE 9.55 — a ~32% improvement over guessing the average, hitting the ceiling of what public box-score data alone can predict."

**Visual idea:** Three numbered cards, one per lesson. Each card has: the lesson title, the specific project moment that taught it (with the number that proved it), and the one-line generalization. No chart needed. The "biggest obstacle" callout sits beneath the three cards as a prominent quote-block.

---

## Slide 11 (Backup) — For Q&A

**Goal:** material to point to if asked. Not part of the timed presentation.

Suggested content:
- Full feature ablation table (all 8 groups, train + test RMSE)
- Full model comparison table (all model classes with timing)
- Full Optuna best parameters
- Per-position counts and dataset-balance notes
- The cumulative ablation chart (`ablation_cumulative.png`)
- Per-fold RMSE for top-10 Optuna trials
- Failed experiments NOT shown earlier: hyperparameter tuning detail, BoxScoreAdvancedV2 collection attempt that got rate-limited

---

## Cross-cutting notes for the designer

**Color coding consistency throughout the deck:**
- Naive baselines / heuristics → gray
- ML model results → blue
- Final model / "winner" → green or distinguished color
- Failed experiments / things we dropped → red or muted

**The narrative thread to maintain across slides:**
> Features did most of the work. The model class change contributed the smallest piece. The most impactful feature was a derivation of data we already had, not external data we collected. We measured what worked AND what didn't, dropping the things that didn't earn their keep.

**Must-have callouts the rubric explicitly rewards:**
- ✅ External data sources mentioned (Slide 4: Vegas, BoxScoreAdvancedV2 — and on Slide 5 we explain why they didn't make the cut, which is itself rigorous)
- ✅ Quantification of each component's marginal value (Slide 5: ablation chart)
- ✅ All model classes from class tested (Slide 6: LinReg, RF, gradient boosting)
- ✅ Hyperparameter tuning (Slide 8: Optuna study, with TimeSeriesSplit CV)
- ✅ Train/test performance for all models (Slide 6 + 9)
- ✅ Generalization discussion (Slide 9: train vs test RMSE gap, ceiling discussion)
- ✅ Limitations / model interpretation (Slide 9 ceiling stack, Slide 10 lessons)
- ✅ Biggest obstacle named explicitly (Slide 10 — combined with lessons learned)
- ✅ Descriptive analysis (Slide 3 distribution + scatter)

**Image assets ready in `presentation_assets/`:**
- `player_games_per_season.png` (Slide 2)
- `fp_distribution.png`, `fp_vs_min.png` (Slide 3)
- `ablation_incremental.png`, `ablation_cumulative.png` (Slide 5 + backup)
- `model_progression.png` (Slide 6)
- Optuna visuals — auto-produced into `models_v2/run_<id>/visuals/` once the v2 study finishes

**Image assets still TO produce:**
- Architecture diagram for Slide 7 (simple flowchart; can be drawn directly in the slide tool)
- Ceiling-stack diagram for Slide 9 (simple stacked-block infographic; can be drawn in the slide tool)
- Three-card layout for Slide 10 (no chart, pure design)
