# Feature Engineering Analysis & Optimization Plan

## Current State: 107 Features, 9.533 RMSE (2.83% improvement)

---

## Current Feature Groups - Impact Analysis

### 1. Rolling Averages (75 features) - **BASELINE**
```
Windows: L3, L5, L10
Player: PTS, REB, AST, STL, BLK, TOV, FG3M, MIN, FANTASY_PTS
Team: score, fg_made, fg3_made, reb, ast, stl, blk, tov
Opponent: same as team
Test RMSE: 9.839
```
✅ **Strong foundation** - most important feature group

### 2. Context (3 features) - **Δ -0.001**
```
is_home, days_rest, opp_days_rest
Test RMSE: 9.838
```
⚠️ **Minimal impact** - needs enrichment

### 3. Trends (8 features) - **Δ -0.005**
```
L3 avg - L10 avg for each player stat (hot/cold streaks)
Test RMSE: 9.833
```
⚠️ **Weak signal** - could be improved with better trend detection

### 4. Efficiency (9 features) - **Δ -0.018**
```
FGA, FG_PCT, PLUS_MINUS (L3, L5, L10)
Test RMSE: 9.816
```
⚠️ **Incomplete** - missing advanced metrics

### 5. Missing Teammates (3 features) - **Δ -0.249** 🏆
```
team_l10_min_played, team_players_played, missing_min_deficit
Test RMSE: 9.567
```
✅ **BIGGEST SINGLE IMPROVEMENT** - opportunity/usage proxy works!

### 6. Schedule Density (3 features) - **Δ +0.002**
```
is_b2b, games_last_4d, games_last_7d
Test RMSE: 9.568
```
❌ **Slightly hurts** - may need refinement

### 7. Position (5 features) - **Δ -0.011**
```
pos_G, pos_F, pos_C, height_in, years_experience
Test RMSE: 9.557
```
✅ **Small but consistent gain**

### 8. Defense vs Position (1 feature) - **Δ -0.015**
```
dvp_L20: FP allowed by opponent to this position
Test RMSE: 9.542
```
✅ **Matchup quality matters** - need more matchup features!

---

## Critical Gaps & Opportunities

### 🔴 **CRITICAL GAP #1: Usage & Role**
**Why it matters:** Missing teammates helped (-0.249 RMSE) because it captures opportunity. We need direct usage metrics.

**Missing features:**
- ✗ Usage rate (FGA share, MIN share, AST share)
- ✗ Role stability (starter vs bench, recent changes)
- ✗ Touch distribution proxy
- ✗ Shot attempts trend (increasing/decreasing)

**Expected impact:** -0.15 to -0.30 RMSE

---

### 🔴 **CRITICAL GAP #2: Recent Performance Signal**
**Why it matters:** We use L3/L5/L10 averages but NOT individual recent games. Last game is highly predictive!

**Missing features:**
- ✗ `lag1` features (last game performance)
- ✗ Exponentially weighted moving average (recent games weighted more)
- ✗ Last 2 games average (L2)
- ✗ Recency bias indicators

**Expected impact:** -0.10 to -0.20 RMSE

---

### 🟡 **MAJOR GAP #3: Player Consistency/Variance**
**Why it matters:** DFS players care about floor/ceiling. Consistent 20 FP ≠ volatile 10-30 FP player.

**Missing features:**
- ✗ Standard deviation of FP (L10)
- ✗ Coefficient of variation (consistency)
- ✗ Ceiling (max FP in L10)
- ✗ Floor (min FP in L10)
- ✗ Boom/bust indicator (high variance flagging)

**Expected impact:** -0.05 to -0.15 RMSE

---

### 🟡 **MAJOR GAP #4: Advanced Efficiency Metrics**
**Why it matters:** Our current efficiency features (FG%, PLUS_MINUS) are incomplete.

**Missing features:**
- ✗ **eFG%**: (FGM + 0.5×FG3M) / FGA (weights 3-pointers correctly)
- ✗ **TS%**: PTS / (2 × (FGA + 0.44×FTA)) (true shooting efficiency)
- ✗ **AST/TO ratio**: Assists per turnover (guard efficiency)
- ✗ **PER-36**: Stats normalized per 36 minutes (role-adjusted productivity)
- ✗ **Shot distribution**: FG2A vs FG3A ratio, FTA rate

**Expected impact:** -0.05 to -0.10 RMSE

---

### 🟡 **MAJOR GAP #5: Matchup History**
**Why it matters:** DvP helps (-0.015), but player-specific matchups likely matter more.

**Missing features:**
- ✗ Player vs this opponent history (last 5 games vs OPP)
- ✗ Player home/away splits (some players are road warriors)
- ✗ Player vs opponent team strength (vs top-10 def vs bottom-10 def)
- ✗ Recent performance vs similar opponents

**Expected impact:** -0.10 to -0.20 RMSE

---

### 🟢 **OPPORTUNITY #6: Pace & Possessions**
**Why it matters:** More possessions = more opportunities for fantasy points.

**Missing features:**
- ✗ Team pace (possessions per game) L10
- ✗ Opponent pace L10
- ✗ **Pace differential** (our pace - opp pace)
- ✗ Projected total possessions
- ✗ Player per-possession stats

**Expected impact:** -0.05 to -0.10 RMSE

---

### 🟢 **OPPORTUNITY #7: Team Context**
**Why it matters:** Winning teams distribute usage differently than losing teams.

**Missing features:**
- ✗ Team win % L10
- ✗ Team win/loss streak
- ✗ Team net rating L10 (point differential)
- ✗ Opponent win % L10
- ✗ Opponent net rating L10
- ✗ Game competitiveness proxy

**Expected impact:** -0.03 to -0.08 RMSE

---

### 🟢 **OPPORTUNITY #8: Interaction Features**
**Why it matters:** Features interact (e.g., high minutes in good matchup = big game).

**Missing features:**
- ✗ `minutes × DvP` (opportunity in favorable matchup)
- ✗ `home × rest` (well-rested home performance)
- ✗ `usage × pace` (high usage in fast game)
- ✗ `position × opponent_pace` (guards benefit more from pace)
- ✗ `missing_minutes × player_role` (starter benefits more from injuries)

**Expected impact:** -0.05 to -0.10 RMSE

---

### 🟢 **OPPORTUNITY #9: Temporal Patterns**
**Why it matters:** Player performance evolves over the season.

**Missing features:**
- ✗ Month indicator (November vs March performance differs)
- ✗ Season progression (game number / 82)
- ✗ Days since season start
- ✗ Playoff race indicator (post-trade deadline, playoff seeding locked)

**Expected impact:** -0.02 to -0.05 RMSE

---

### 🟢 **OPPORTUNITY #10: Feature Engineering Quality**
**Why it matters:** Better feature representation = better predictions.

**Issues:**
- High correlation between L3, L5, L10 (redundancy)
- Equal weighting of past games (recent should matter more)
- Fixed windows (L3, L5, L10) might not be optimal
- No feature selection (using all 107 features)

**Improvements:**
- ✓ Add lag1 (most recent game)
- ✓ Use exponentially weighted moving average (EWMA)
- ✓ Test other windows (L2, L7, L15, L20)
- ✓ Feature selection / PCA for dimensionality reduction
- ✓ Polynomial features for non-linearity

**Expected impact:** -0.05 to -0.15 RMSE

---

## Recommended Feature Additions - Prioritized

### **PHASE 1: Quick Wins (High Impact, Low Effort)**
**Target: -0.30 to -0.50 RMSE improvement**

1. **Lag1 Features (Recency)**
   ```python
   for stat in PLAYER_STATS:
       player_logs[f'player_{stat}_lag1']
   # Direct last-game performance
   ```

2. **Usage Proxy Features**
   ```python
   player_shot_share_L10 = player_FGA_L10 / team_FGA_L10
   player_min_share_L10 = player_MIN_L10 / (team_total_min / 5)
   player_scoring_share_L10 = player_PTS_L10 / team_score_L10
   ```

3. **Consistency Features**
   ```python
   player_fp_std_L10 = std_dev(last 10 FANTASY_PTS)
   player_fp_cv_L10 = std / mean (coefficient of variation)
   player_fp_ceiling = max(last 10 FANTASY_PTS)
   player_fp_floor = min(last 10 FANTASY_PTS)
   ```

4. **Advanced Efficiency**
   ```python
   player_efg_pct_L10 = (FGM + 0.5*FG3M) / FGA
   player_ts_pct_L10 = PTS / (2 * (FGA + 0.44*FTA))
   player_ast_to_ratio_L10 = AST / TOV
   ```

5. **Pace Differential**
   ```python
   team_pace_L10 = team possessions per game
   pace_differential = team_pace_L10 - opp_pace_L10
   ```

---

### **PHASE 2: Matchup & Context (Medium Impact, Medium Effort)**
**Target: -0.15 to -0.30 RMSE improvement**

6. **Player vs Opponent History**
   ```python
   player_vs_opp_fp_L5 = avg FP in last 5 vs this opponent
   player_vs_opp_count = games played vs this opponent
   ```

7. **Home/Away Splits**
   ```python
   player_home_fp_avg_L10
   player_away_fp_avg_L10
   player_home_away_diff
   ```

8. **Team Strength Indicators**
   ```python
   team_win_pct_L10
   team_net_rating_L10 = (team_score_L10 - opp_score_L10)
   opp_win_pct_L10
   opp_def_rating_proxy = points_allowed_L10
   ```

9. **Role Stability**
   ```python
   is_starter = 1 if MIN_L5 > 24 else 0
   min_volatility_L10 = std_dev(MIN last 10)
   role_change = change in starter status
   ```

---

### **PHASE 3: Interactions & Advanced (High Potential, High Effort)**
**Target: -0.10 to -0.20 RMSE improvement**

10. **Interaction Terms**
    ```python
    min_times_dvp = player_MIN_L10 * dvp_L20
    usage_times_pace = player_shot_share * pace_differential
    home_times_rest = is_home * days_rest
    missing_min_times_role = missing_min_deficit * is_starter
    ```

11. **Exponentially Weighted Moving Averages**
    ```python
    # Instead of simple average, weight recent games more
    player_PTS_EWMA = EWMA(PTS, span=10, alpha=0.2)
    # More responsive to recent trends
    ```

12. **Per-Possession Stats**
    ```python
    player_pts_per_poss_L10 = player_PTS_L10 / team_pace_L10
    player_reb_per_poss_L10 = player_REB_L10 / team_pace_L10
    ```

---

### **PHASE 4: Model Optimization**
**Target: -0.10 to -0.20 RMSE improvement**

13. **Hyperparameter Tuning** (currently SKIPPED!)
    ```python
    RandomizedSearchCV on HistGradientBoosting:
    - learning_rate: [0.01, 0.05, 0.1]
    - max_depth: [6, 8, 10, 12]
    - min_samples_leaf: [10, 20, 50]
    - max_leaf_nodes: [31, 63, 127, None]
    ```

14. **Ensemble Methods**
    ```python
    - Stack multiple models (HistGB + XGBoost + LightGBM)
    - Weighted average based on validation performance
    - Different models for different stat components
    ```

15. **Feature Selection**
    ```python
    - Remove highly correlated features (>0.95 correlation)
    - Use feature importance to drop low-value features
    - Test different window combinations
    ```

---

## Expected Total Improvement

| Phase | Features | Expected RMSE Gain | Cumulative RMSE |
|-------|----------|-------------------|-----------------|
| Current | 107 | - | 9.533 |
| Phase 1 | +20-30 | -0.30 to -0.50 | 9.03 to 9.23 |
| Phase 2 | +15-20 | -0.15 to -0.30 | 8.73 to 9.08 |
| Phase 3 | +10-15 | -0.10 to -0.20 | 8.53 to 8.98 |
| Phase 4 | Model tuning | -0.10 to -0.20 | **8.33 to 8.88** |

**Realistic target: 8.8 to 9.0 RMSE (7-10% total improvement from baseline)**

---

## Feature Redundancy Analysis

### High Correlation (likely redundant):
- L3, L5, L10 windows overlap significantly
- Team and opponent stats are mirrors
- FG_PCT already includes FGA information

### Recommendations:
1. **Keep:** L3 and L10 (drop L5) - captures short-term and medium-term
2. **Add:** L1 (last game), EWMA (adaptive)
3. **Test:** L15, L20 for longer trends
4. **Consider:** Feature selection to remove low-importance features

---

## Data Quality Checks Needed

Before adding features:
1. ✓ Check for data leakage (no future information)
2. ✓ Validate all features have no NaN in test set
3. ✓ Ensure temporal ordering (no look-ahead bias)
4. ✓ Check for outliers (cap extreme values?)
5. ✓ Validate opponent mapping (home/away correct)

---

## Implementation Strategy

### Week 1: Quick Wins (Phase 1)
- Implement lag1 features
- Add usage proxies
- Add consistency metrics
- Add advanced efficiency
- **Expected gain: -0.30 RMSE**

### Week 2: Context (Phase 2)
- Player vs opponent history
- Home/away splits
- Team strength
- **Expected gain: -0.20 RMSE**

### Week 3: Advanced (Phase 3)
- Interaction terms
- EWMA features
- Per-possession stats
- **Expected gain: -0.15 RMSE**

### Week 4: Optimization (Phase 4)
- Hyperparameter tuning
- Feature selection
- Ensemble methods
- **Expected gain: -0.15 RMSE**

---

## Next Steps

**IMMEDIATE ACTION:**
1. Start with Phase 1 - implement lag1 and usage features
2. Re-run ablation study to quantify each addition
3. Track RMSE improvement at each step
4. Document what works and what doesn't

**QUESTION FOR YOU:**
- Do you want to implement all Phase 1 features at once?
- Or should we add them incrementally with ablation studies between each?
- How much time do we have for this optimization?
