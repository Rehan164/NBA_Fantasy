# Phase 1 Feature Engineering Results

## Executive Summary

**Result: Phase 1 features provided only +0.021 RMSE improvement (0.21%)**
- Baseline: 9.894 RMSE
- With Phase 1: 9.874 RMSE
- Target was: -0.30 to -0.50 RMSE

**Status: Below expectations** - Need to pivot strategy.

---

## Phase 1 Features Tested

### 1. Lag1 Features (Last Game Performance)
**Added:** `player_FANTASY_PTS_lag1`, `player_PTS_lag1`, `player_MIN_lag1`

**Impact:** Minimal - These are highly redundant with L3 rolling averages, which already heavily weight recent games.

**Why it failed:**
- L3 average = (lag1 + lag2 + lag3) / 3
- Lag1 is already 33% of the L3 signal
- Gradient boosting can extract lag1 from L3, L5, L10 differences

### 2. Consistency Features (Volatility Metrics)
**Added:**
- Standard deviation: `player_FANTASY_PTS_std_L10`, `player_PTS_std_L10`
- Ceiling/Floor: `player_FANTASY_PTS_ceiling_L10`, `player_FP_floor_L10`

**Impact:** Minimal - Ceiling/floor are extreme values that gradient boosting may be overfitting to.

**Why it failed:**
- These features are orthogonal to the prediction task
- We're predicting expected value, not variance
- Ceiling/floor are too noisy (single games can be outliers)
- Model may already account for player variance implicitly through other features

### 3. Advanced Efficiency Metrics
**Added:**
- Effective FG%: `player_eFG_PCT_L10`
- True Shooting %: `player_TS_PCT_L10`
- AST/TO Ratio: `player_AST_TO_RATIO_L10`

**Impact:** Minimal - These are derived from existing stats (FGM, FGA, FG3M, FTA, AST, TOV).

**Why it failed:**
- These are linear combinations of existing features
- Gradient boosting can learn non-linear transformations
- FG_PCT was already in the model
- The model can infer efficiency from PTS/FGA patterns

### 4. Usage Share Features
**Added:**
- Shot share: `player_shot_share_L10` = player_FGA / team_FGA
- Scoring share: `player_scoring_share_L10` = player_PTS / team_PTS

**Impact:** Minimal - Player FGA and PTS patterns already capture usage implicitly.

**Why it failed:**
- player_PTS_L10 and player_FGA_L10 already exist
- Team totals are relatively constant (~240 FGA per team per game)
- The share is mostly driven by player's own stats, not team context
- Missing the KEY insight: **opportunity changes due to teammate absence**

---

## What Actually Worked in the Notebook

### ✅ Missing Teammates: -0.249 RMSE (BIGGEST WIN)

**What it does:**
- Detects when rotation players are missing from the box score
- Computes `missing_min_deficit` = expected rotation minutes - actual rotation minutes
- Captures opportunity spikes for remaining players

**Why it works:**
- This information is NOT inferrable from the player's own stats
- A player averaging 30 minutes might play 35 tonight because a teammate is out
- The model can't predict this without the "missing minutes" signal
- It's a direct proxy for usage opportunity

**Key lesson:** Features that provide NEW information (not derivable from existing stats) are valuable.

---

## Root Cause Analysis

### Why Phase 1 Failed

1. **Redundancy**: Most Phase 1 features are linear/non-linear transformations of existing features. Gradient boosting can already learn these relationships.

2. **Information already captured**: Player rolling averages (PTS_L10, FGA_L10, etc.) already encode usage, efficiency, and consistency patterns.

3. **Wrong abstraction level**: We added stat derivatives when we should have added context signals.

### What We Learned

**Good features provide information that's orthogonal to existing features:**
- ✅ Missing teammates (opportunity context)
- ✅ Defense vs Position (matchup quality)
- ✅ Position (physical attributes)

**Bad features are transformations of existing features:**
- ❌ Lag1 (already in rolling averages)
- ❌ eFG% (transformation of FGM, FG3M, FGA)
- ❌ Usage share (transformation of player stats / team stats)
- ❌ Ceiling/floor (extreme values, noisy)

---

## Recommended Next Steps

### Strategy Pivot: Focus on Context, Not Derivatives

Instead of deriving new stats from box scores, add features that capture:
1. **External context** the model can't infer from stats alone
2. **Matchup-specific patterns** that vary by opponent
3. **Temporal patterns** that aren't linear time trends
4. **Interaction effects** between existing features

---

### High-Impact Features to Try Next

#### 🔴 **PRIORITY 1: Matchup History** (Expected: -0.15 to -0.25 RMSE)

**Why this should work:**
- Some players consistently perform well/poorly against specific teams
- LeBron vs Celtics, Curry vs Cavs, etc.
- This is NOT captured by generic DvP (defense vs position)

**Features to add:**
```python
player_vs_opponent_fp_L5   # Last 5 games vs THIS opponent
player_vs_opponent_count   # Sample size
player_vs_opp_fp_diff      # vs_opp_L5 - overall_L10 (matchup differential)
```

**Why it's different from DvP:**
- DvP = how opponent defends ALL point guards
- Matchup history = how THIS player performs vs THIS opponent
- Player-specific matchups capture coaching schemes, psychological factors, playstyle compatibility

---

#### 🟡 **PRIORITY 2: Home/Away Splits** (Expected: -0.05 to -0.15 RMSE)

**Why this should work:**
- Some players are road warriors, others struggle away from home
- Home court advantage is real but player-specific
- Current model only has `is_home` (0/1), not player-specific home/away performance

**Features to add:**
```python
player_home_fp_L10         # Average FP in home games
player_away_fp_L10         # Average FP in away games
player_home_away_diff      # Home advantage magnitude
is_home × player_home_away_diff  # Interaction
```

---

#### 🟡 **PRIORITY 3: Pace & Possessions** (Expected: -0.05 to -0.10 RMSE)

**Why this should work:**
- More possessions = more opportunities for stats
- Fast-paced games benefit volume scorers
- Slow-paced games benefit efficient players
- Current model doesn't capture game speed

**Features to add:**
```python
team_pace_L10              # Team's avg possessions per game
opp_pace_L10               # Opponent's avg possessions
pace_differential          # Combined game pace (team + opp) / 2
projected_possessions      # Estimated total possessions for this game
```

**Interaction features:**
```python
player_pts_per_poss_L10    # Scoring efficiency per possession
usage × pace               # High-usage players benefit more from pace
```

---

#### 🟡 **PRIORITY 4: Exponentially Weighted Moving Averages** (Expected: -0.03 to -0.08 RMSE)

**Why this should work:**
- Current rolling averages weight all games equally: L10 = (g1+g2+...+g10)/10
- EWMA weights recent games more: recent games get 2-3x more weight
- Captures momentum, hot/cold streaks better than linear trends

**Implementation:**
```python
player_PTS_EWMA_span10  # More responsive to recent changes
player_MIN_EWMA_span10  # Catches role changes faster
```

**Why it's different from lag1:**
- Lag1 = only last game (too noisy)
- L3/L5/L10 = equal weight (too slow to adapt)
- EWMA = smooth but responsive (best of both)

---

#### 🟢 **PRIORITY 5: Role Stability Features** (Expected: -0.03 to -0.06 RMSE)

**Why this should work:**
- Players transitioning from bench to starter see usage spikes
- Injured players returning have minutes restrictions
- Rookie minutes increase as season progresses
- Current model doesn't detect role changes

**Features to add:**
```python
min_volatility_L10          # std(MIN) - detects role changes
is_starter                  # 1 if MIN_L5 > 24 minutes
min_trend_L5_vs_L10         # Minutes increasing or decreasing?
games_since_return          # For players returning from injury (if injury data available)
```

---

#### 🟢 **PRIORITY 6: Interaction Features** (Expected: -0.05 to -0.10 RMSE)

**Why this should work:**
- Features interact: high minutes in good matchup = big game
- Current model learns interactions but explicit features can help
- Especially useful for capturing edge cases

**Top interactions to try:**
```python
player_MIN_L10 × dvp_L20                  # Opportunity in favorable matchup
player_shot_share × pace_differential     # Volume scorer in fast game
missing_min_deficit × is_starter          # Starter benefits most from injuries
is_home × days_rest                       # Well-rested home performance
```

---

## Implementation Plan

### Recommended Approach: Incremental Ablation

1. **Test each feature group individually** against the current best model (9.533 RMSE)
2. **Measure marginal impact** of each group
3. **Keep only features that improve by ≥0.01 RMSE**
4. **Combine winners** and test for interactions/redundancy

### Timeline

- **Week 1:** Matchup history + Home/away splits
- **Week 2:** Pace features + EWMA
- **Week 3:** Role stability + Interactions
- **Week 4:** Ensemble methods + hyperparameter tuning

### Expected Outcome

**Realistic target: 9.2 to 9.4 RMSE** (vs current 9.533)
- Matchup history: -0.15 RMSE
- Home/away: -0.08 RMSE
- Pace: -0.06 RMSE
- EWMA: -0.04 RMSE
- Interactions: -0.05 RMSE
**Total: -0.38 RMSE → ~9.15 RMSE**

---

## Key Takeaways

### ✅ Do This
- Add features that capture information NOT in box scores (context, matchups, environment)
- Focus on high-impact features (1 feature worth -0.20 RMSE > 20 features worth -0.01 each)
- Test incrementally and measure each addition
- Look for interactions between existing features

### ❌ Don't Do This
- Add derivatives of existing features (eFG%, usage shares)
- Add redundant features (lag1 when you have L3)
- Add noisy features (ceiling/floor based on single games)
- Assume more features = better model

---

## Next Action

**Recommendation:** Implement Priority 1 (Matchup History) first.

This has the highest expected impact and is completely orthogonal to existing features. If a player averages 25 FP but scores 35 vs Team X, that's information the current model can't infer from rolling averages alone.

**Code to add:**
```python
# Group by player + opponent
matchup_history = player_logs.groupby(["PLAYER_ID", "opponent"]).apply(
    lambda g: g.sort_values("GAME_DATE").assign(
        player_vs_opp_fp_L5=g["FANTASY_PTS"].shift(1).rolling(5, min_periods=2).mean()
    )
)
```

Would you like me to implement matchup history features next?
