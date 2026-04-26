# Data Cleaning & Processing

How the three raw CSVs become the feature matrix that goes into the model. This document covers the *plumbing* — the structural transforms, joins, and filters — not the individual feature derivations (those come next).

The pipeline runs entirely inside `nba_fantasy_model.ipynb`. From raw load to model-ready matrix is roughly seven stages.

---

## The end-to-end flow at a glance

```
   3 raw CSVs                                    1 feature matrix
   ──────────                                    ────────────────
   nba_player_game_logs.csv  ┐
   nba_historical_games.csv  ├──►  [pipeline]  ──►  X_train / y_train
   nba_player_info.csv       ┘                      X_test  / y_test
```

| # | Stage | What it does |
|---|---|---|
| 1 | Load + type-normalize | Parse dates, coerce MIN to numeric, normalize game IDs |
| 2 | Compute target | Apply DK formula to derive `FANTASY_PTS` per row |
| 3 | Build lag/rolling structure | Sort temporally, then `shift(1).rolling(...)` per entity |
| 4 | Pivot team data to long form | One row per team per game (instead of one per game) |
| 5 | Cross-table join | Glue player → team → opponent into a single row |
| 6 | Filter | Drop rows with incomplete lag history; require MIN ≥ 10 |
| 7 | Temporal train/test split | Cut at 2023-10-01 |

---

## Stage 1 — Load and normalize types

The raw CSVs come straight from the NBA API, but they have a few quirks that need cleaning before anything else:

| Quirk | Fix |
|---|---|
| `GAME_DATE` and `date` come as strings (`"2023-11-08"`) | `pd.to_datetime(...)` — required for any temporal operation |
| `GAME_ID` has different formats across files. Player logs use a 10-char zero-padded string (`"0029900001"`); games use a numeric value | Cast both to int (`game_id_int`) for consistent join keys |
| `MIN` occasionally comes as a string like `"30:45"` (minutes:seconds) instead of a float | `pd.to_numeric(player_logs["MIN"], errors="coerce")` — anything unparseable becomes NaN |

This stage is short but skipping it breaks every downstream join silently.

---

## Stage 2 — Compute the target

Fantasy points aren't in the raw data. We compute them once per row using DraftKings scoring:

```
FANTASY_PTS = PTS×1.00 + REB×1.25 + AST×1.50 + STL×2.00 + BLK×2.00 + TOV×(−0.50) + FG3M×0.50
```

Distribution after computation:
- Mean: 20.54 fantasy points
- Std: 14.15
- Range: −2.00 to 107.75

We compute it *before* the lag structure is built so that fantasy points itself becomes available as a feature later (we use rolling FANTASY_PTS as one of the strongest predictors of next-game FANTASY_PTS).

---

## Stage 3 — Build the lag/rolling structure (the temporal-correctness story)

This is the most important stage in the entire pipeline. **It is also the easiest place to introduce data leakage.** Get this wrong and the model looks great on test but fails in production.

### The risk

For any player-game row in the dataset, we want features describing what that player has done *before* this game. If we compute a rolling average that includes the current game, we've leaked the answer into the features.

### Our pattern

Every per-entity rolling feature follows the same template:

```python
df = df.sort_values(["entity_id", "GAME_DATE"]).reset_index(drop=True)

df["stat_L10"] = (
    df.groupby("entity_id")["stat"]
      .transform(lambda x: x.shift(1).rolling(10, min_periods=10).mean())
)
```

Three things happen here, and the order matters:

1. **Sort by entity then date.** `groupby` doesn't guarantee row order; without an explicit sort, `shift(1)` might shift across players or backward in time.
2. **`shift(1)` first.** This drops the current row out of the window, so we're rolling over the prior 10 games — not the current and the prior 9.
3. **`rolling(10, min_periods=10).mean()` second.** Requires a full 10-game history to produce a value; otherwise NaN. We could relax this with `min_periods=3` for early-career rows, and we do for some derived features, but for the headline rolling stats we require the full window.

### Three entities get this treatment

The same pattern is applied independently to three entity types:

| Entity | Source data | Rolling features produced |
|---|---|---|
| **Player** | `player_logs` grouped by `PLAYER_ID` | Player rolling stats (PTS, REB, AST, STL, BLK, TOV, FG3M, MIN, FANTASY_PTS over L3/L5/L10) |
| **Team** | Pivoted `team_games` grouped by `TEAM_ABBREVIATION` | Team rolling stats (score, fg_made, fg3_made, reb, ast, stl, blk, tov over L3/L5/L10) |
| **Opponent** | Same `team_games`, looked up by who each team played | Opponent rolling stats (same eight stats × three windows) |

This separation is what lets the model see, for any given player-game, the player's recent form *plus* the team's recent form *plus* the opponent's recent form — three independent rolling histories.

---

## Stage 4 — Pivot team data to long form

The raw `nba_historical_games.csv` is in **wide format**: one row per game with `home_*` and `away_*` columns. That's good for game-level analysis but bad for computing per-team rolling stats — you'd have to do every operation twice (once for home columns, once for away).

We restructure it into **long format**: one row per team per game.

```
Wide (raw)                       Long (derived)
─────────                        ──────────────
game_id  home_team  away_team    game_id  team  score  ...
1234     LAL        BOS          1234     LAL   115    ...
                                 1234     BOS   108    ...
```

Each game becomes two rows. Then we can group by `team` and apply the same Stage-3 rolling template — same code as for players.

After computing team rolling features, we also build an opponent lookup: a small lookup table that maps each (game, team) pair to (game, opponent), then re-merges the team's own rolling features under a renamed `opp_*` prefix to give us the opponent perspective.

---

## Stage 5 — Cross-table join

The model needs one row per player-game with player history, team history, and opponent history all in the same row. The join chain:

```
   player_logs                        team_games (long form)
   (PLAYER_ID, GAME_ID, ...)          (game_id, team, lag features, opp_lag features)
              \                         /
               \                       /
                ▼                     ▼
              merge on (game_id, team_abbreviation)
                          │
                          ▼
                   feature matrix
                  (player history + team history + opp history)
```

The `inner` join here is intentional — we want only player-games where we successfully matched the team-side data, and vice versa. After merge we have a player-game-level dataframe with all three perspectives lined up.

This stage produces our first useful intermediate: `df_clean` — the dataframe everything else builds on.

---

## Stage 6 — Filter for usable training rows

Two filters get applied in sequence:

### Filter 6a: Drop rows with incomplete lag history

```python
df_clean = df.dropna(subset=all_feature_cols).reset_index(drop=True)
```

A player's first game in the dataset has no prior games to roll over — so its lag features are NaN. Same for any player who's missing 10 prior games. We drop those rows to give the model only training examples where every feature is defined.

This is more aggressive than necessary for `HistGradientBoostingRegressor` (which handles NaN natively), but we keep the filter because:
1. It's required for the linear-regression baseline (LR can't handle NaN)
2. It removes noisy rookie/early-career rows where the rolling history is dominated by garbage-time appearances
3. It makes the train and test sets directly comparable

### Filter 6b: Require MIN ≥ 10

```python
df_clean = df_clean[df_clean["MIN"] >= 10].reset_index(drop=True)
```

NBA box scores include garbage-time appearances — the last 30 seconds of a 30-point blowout, end-of-bench players, two-way contracts called up for one possession. These rows have a real `FANTASY_PTS` number, but they don't reflect the player's actual role and would just add noise to training.

Cutting at 10 minutes is a domain-driven threshold. It removes about 17% of rows and keeps only meaningful playing time.

After both filters: **561,108 rows** remain (down from 673,733), with **240 base features** ready for engineered additions.

---

## Stage 7 — Temporal train/test split

Standard random k-fold cross-validation would leak future games into training. Basketball is non-stationary (rule changes, pace evolution, three-point revolution, individual player aging), so we split temporally instead.

```python
train_mask = df_clean["GAME_DATE"] < "2023-10-01"
```

| Set | Date range | Rows |
|---|---|---|
| Train | 1999-11-02 to 2023-10-01 (~24 seasons) | 496,750 |
| Test | 2023-10-01 onward (2023-24 + 2024-25 + 2025-26 to date) | 64,358 |

For cross-validation during model selection, we use `TimeSeriesSplit` instead of `KFold`. Each fold respects the temporal order: fold 1 trains on year 1, validates on year 2; fold 2 trains on years 1-2, validates on year 3; and so on. Future folds never leak into training folds.

---

## What this stage looks like for an external data source

When we add player position from `nba_player_info.csv` (Stage 3 in `01_data_sources.md`), the integration is much simpler than the lag/rolling stages:

```python
df_clean = df_clean.merge(player_info, on="PLAYER_ID", how="left")
```

It's a left join on `PLAYER_ID` because position is a static per-player attribute, not a time-series. No shifting, no rolling, no temporal correctness concerns. The same pattern would apply to any other external biographical data we added later.

---

## Pipeline guarantees

Putting it all together, the pipeline guarantees:

1. **No data leakage from the future.** Every rolling feature uses `shift(1)` before `rolling(...)`, so the current game is never in its own feature window.
2. **No data leakage across entities.** Sort-then-groupby ensures rolling windows respect entity boundaries (one player's stats don't leak into another's).
3. **No data leakage across the train/test boundary.** Temporal split + `TimeSeriesSplit` for CV ensures evaluation always reflects "predicting the future from the past."
4. **Consistent join keys.** `game_id_int` is normalized once in Stage 1 and used everywhere.
5. **Comparable train and test distributions.** The MIN ≥ 10 filter and the lag-completeness filter are applied identically to both sets, so distribution shift between train and test is purely temporal — not artifactual.
6. **Reproducible.** No random sampling in the pipeline. Same input CSVs → same feature matrix every time.

---

## Bottom line

The pipeline is conceptually simple: load, compute target, build lag history per entity, join entities, filter, split. Most of the engineering effort went into the **temporal correctness** of Stage 3 (the `shift(1).rolling(...)` pattern, the entity-then-date sort) and the **structural transform** of Stage 4 (wide → long for team data). Get those two right and everything downstream is straightforward merges.

What this pipeline produces is `df_clean` — 561,108 rows × 240+ feature columns — which is the substrate that every individual feature engineering decision (covered in the next document) builds on top of.
