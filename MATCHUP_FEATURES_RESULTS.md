# Matchup History Features - Results

## Summary

**Result: Matchup history features provided +0.052 RMSE improvement (0.52%)**
- Baseline: 9.894 RMSE
- With matchup history: 9.843 RMSE
- **Target was: -0.15 to -0.25 RMSE**

**Status: Moderate improvement, below target**

---

## Features Implemented

1. **`player_vs_opp_fp_L5`**: Average fantasy points in last 5 games vs this opponent
   - Coverage: 94.6% of games
   - Uses min_periods=1, so starts working after 1st game vs opponent

2. **`player_vs_opp_count`**: Number of games played vs this opponent (before current game)
   - Coverage: 100%
   - Proxy for sample size reliability

3. **`player_vs_opp_fp_diff`**: Matchup differential (vs_opp_L5 - overall_L10)
   - Coverage: 94.6%
   - Direct measure of matchup advantage/disadvantage

---

## Why It Underperformed

### 1. **Small Sample Sizes**

Most matchups in the test set have limited history:
- Teams play each other 3-4 times per season
- With L5 rolling window, many matchups use 1-2 games
- High variance in small samples

### 2. **Noisy Signal**

Example from results: Jalen Johnson vs HOU
- Game 1-3 average: 5.5 FP (vs 39.8 overall) → -34.3 differential
- Game 4 added: 14.25 FP (vs 46.2 overall) → -31.9 differential
- This volatility makes it hard for the model to learn reliable patterns

### 3. **Partial Capture by Existing Features**

The model may already implicitly learn some matchup effects through:
- Player rolling averages adapt to recent performance (which includes opponent effects)
- Position × DvP interaction captures some matchup quality
- Overall variance in player performance includes opponent-driven variance

### 4. **Multi-Output Architecture Limitation**

- We're predicting PTS, REB, AST, etc. separately
- Matchup effects might be composite (good rebounding matchup + bad scoring matchup)
- Fantasy points differential doesn't decompose cleanly into stat-specific matchups

---

## Observable Matchup Effects

Despite the modest model improvement, strong matchup effects DO exist:

**Top Matchup Advantages (Test Set):**
| Player | Opponent | vs_Opp FP | Overall FP | Differential | Games |
|--------|----------|-----------|------------|--------------|-------|
| Ben Simmons | BKN | 47.4 | 15.5 | **+31.9** | 14 |
| Spencer Dinwiddie | IND | 42.1 | 10.9 | **+31.2** | 15 |

**Top Matchup Disadvantages:**
| Player | Opponent | vs_Opp FP | Overall FP | Differential | Games |
|--------|----------|-----------|------------|--------------|-------|
| Jalen Johnson | HOU | 5.5 | 39.8 | **-34.3** | 3 |
| Daniss Jenkins | ORL | 3.4 | 37.6 | **-34.1** | 3 |
| Coby White | CLE | 11.5 | 45.0 | **-33.6** | 14 |

The signal is real, but the model isn't extracting it effectively.

---

## Why Baseline Performed Well (9.894 RMSE)

The simple baseline (player rolling averages + position) is already strong because:

1. **Player_FANTASY_PTS_L10** is highly predictive
   - 10-game average smooths variance
   - Captures player's current form

2. **Recency weighting** through L3, L5, L10 combination
   - L3 responds quickly to hot/cold streaks
   - L10 provides stable baseline
   - Model can learn optimal weighting

3. **Multi-output decomposition** specializes on each stat
   - REB model focuses on rebounding factors
   - PTS model focuses on scoring factors
   - Composite FP prediction benefits from specialized models

---

## Comparison: All Feature Tests So Far

| Feature Group | RMSE | vs Baseline | Status |
|---------------|------|-------------|--------|
| **Baseline** (rolling + position) | 9.894 | - | Reference |
| + Phase 1 (lag1, consistency, efficiency, usage) | 9.874 | **+0.021** | Marginal |
| + Matchup history | 9.843 | **+0.052** | Moderate |
| **Notebook's best** (+ missing teammates + DvP + trends + efficiency) | **9.533** | **+0.361** | Strong |

**Key insight:** The notebook's feature set is 7x more effective than our new features.

**Missing teammates alone: -0.249 RMSE** (5x better than matchup history!)

---

## Next Steps: Pivot Strategy

### Option A: Combine All New Features

Test matchup + Phase 1 + home/away + pace together:
- Maybe features interact (e.g., high usage in good matchup)
- Cumulative effect might exceed individual contributions
- **Estimated combined impact: -0.10 to -0.15 RMSE**

### Option B: Return to Missing Teammates Pattern

Ask: What other features capture **opportunity context** like missing teammates?

Candidates:
1. **Blowout games** - Garbage time inflates bench player stats
2. **Overtime games** - Extra minutes for starters
3. **Team strength differential** - Playing vs weak teams increases opportunity
4. **Recent trades/injuries** - Role changes after roster moves

**Estimated impact: -0.10 to -0.20 RMSE per feature**

### Option C: Model Architecture Change

The multi-output decomposition might not be optimal for composite features:
- Try **direct FP regression** instead of stat decomposition
- Try **ensemble methods** (stack multiple models)
- Try **neural network** for non-linear interactions

**Estimated impact: -0.05 to -0.15 RMSE**

---

## Recommended Action

### Immediate: Test Feature Combinations

Create a comprehensive model with ALL new features:
```python
comprehensive_feats = (
    baseline_feats +
    matchup_feats +      # +0.052
    phase1_feats +       # +0.021
    home_away_feats +    # estimated +0.05
    pace_feats           # estimated +0.05
)
```

**Expected: -0.15 to -0.20 RMSE** (if features don't overlap too much)

### If that fails, pivot to Option B:

Implement **blowout/competitive game features**:
```python
score_differential_L10  # Team's avg margin of victory
is_competitive_game     # Within 10 points in 4th quarter
player_blowout_usage    # Usage rate in non-competitive games
```

This follows the "missing teammates" pattern: capturing **situational opportunity** rather than stat derivatives.

---

## Key Learnings

### ✅ What Works
- Features that capture **external context** (missing teammates: -0.249)
- Features that proxy **opportunity changes** (DvP: -0.015)
- **Multi-output decomposition** for specialization

### ❌ What Doesn't Work
- Stat derivatives (Phase 1: +0.021)
- Small-sample matchup averages (matchup history: +0.052)
- Features the model can infer from existing stats

### 🤔 Unclear
- Feature combinations (might have interaction effects)
- Home/away splits (not tested yet)
- Pace features (not tested yet)
- Alternative model architectures

---

## Revised Timeline

**Week 1 (Current):**
- ✅ Phase 1 features: +0.021 RMSE
- ✅ Matchup history: +0.052 RMSE
- ⏳ Test comprehensive feature combination

**Week 2:**
- Home/away splits
- Pace differential
- If combined < -0.15, pivot to Option B (opportunity features)

**Week 3:**
- Blowout/competitive game features
- Team strength features
- EWMA (if time permits)

**Week 4:**
- Ensemble methods
- Hyperparameter tuning
- Final model selection

**Revised realistic target: 9.4 to 9.5 RMSE** (vs original 9.533)
- Conservative estimate given underperformance so far
- Still represents ~3-4% improvement over baseline

---

## Conclusion

Matchup history features provide measurable but modest improvement (+0.052 RMSE). The signal exists in the data (±30 FP matchup differentials), but small sample sizes and noise limit the model's ability to learn from it.

**Recommendation:** Proceed with comprehensive feature combination test to see if features interact positively. If results remain modest, pivot to "opportunity context" features that follow the missing teammates pattern.
