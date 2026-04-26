# Data Sources & Raw Data

Everything in our final model comes from a single source: the **official NBA stats API at `stats.nba.com`**, accessed through the open-source `nba_api` Python package. No web scraping, no purchased datasets, no third-party aggregators.

We hit three endpoints and produced three CSV files. Combined coverage: **26 NBA seasons (1999-2000 through 2025-26), 32,444 games, 673,733 player-game appearances, 2,648 unique players.**

---

## The source: NBA Stats API

**URL:** `https://stats.nba.com/stats/`
**Access:** Open, no authentication, no API key
**Library:** [`nba_api`](https://github.com/swar/nba_api) — the maintained Python wrapper used by virtually every public NBA analytics project

**Rate limit:** Empirically ~0.6s between requests (`NBA_API_DELAY = 0.6` in `config.py`). Heavier hammering gets throttled or blocked.

### Why we chose it

1. **Authoritative.** It's the league's own data — same source the broadcasters and team analytics departments use. No discrepancies between our numbers and the source of truth.
2. **Free and unmetered.** Other sports data sources (Sportradar, the-odds-api, BigDataBall) charge per call or have low free tiers. NBA Stats has no free-tier limits, only behavioral rate limits.
3. **Historical depth.** Reliable coverage back to the late 1990s. Many free alternatives only go back 5-10 seasons.
4. **Stable schema.** Column names and data types are consistent across decades, which is rare for free sports data.
5. **Ecosystem.** `nba_api` has 4k+ GitHub stars, active maintenance, and well-documented endpoint signatures. Lower integration risk than scraping.

---

## Dataset 1: Team game logs

**File:** `data/nba_historical_games.csv`
**Script:** `collect_data.py`
**Endpoint:** `LeagueGameLog` (team mode)
**API calls:** 26 (one per season)
**Runtime:** ~1 minute

### What the endpoint returns

`LeagueGameLog` returns one row per *team* per game for a specified season. A single game produces two rows (one for each team). Our script pivots these into one row per game with `home_*` and `away_*` prefixed columns, then concatenates all seasons.

### Final dimensions

| | |
|---|---|
| Rows | **32,444** (one per game) |
| Columns | 46 |
| Date range | 1999-11-02 → 2026-04-12 |
| Seasons | 26 |

### What's in it

- **Identifiers:** `game_id`, `date`, `season`, `home_team`, `away_team`, team IDs
- **Outcomes:** `home_score`, `away_score`, `home_win`, `home_margin`, `total_score`
- **Box score (per side):** `fg_made`/`fg_att`/`fg_pct`, `fg3_made`/`fg3_att`/`fg3_pct`, `ft_made`/`ft_att`/`ft_pct`, `oreb`, `dreb`, `reb`, `ast`, `stl`, `blk`, `tov`, `pf`

### What it powers in the model

- **24 team rolling features** — rolling L3/L5/L10 averages of score, field goals made, 3PT made, rebounds, assists, steals, blocks, turnovers
- **24 opponent rolling features** — same eight stats, but for the team the player is playing against
- **`is_home` indicator** for the player's team
- **`days_rest` and `opp_days_rest`** computed from each team's game-date sequence
- **Schedule-density features** (back-to-back, games-in-last-7-days)

---

## Dataset 2: Player game logs (the core dataset)

**File:** `data/nba_player_game_logs.csv`
**Script:** `collect_players.py`
**Endpoint:** `LeagueGameLog` with `player_or_team_abbreviation="P"`
**API calls:** 26 (one per season)
**Runtime:** ~2 minutes

### What the endpoint returns

The same `LeagueGameLog` endpoint, but in player mode it returns one row per *player* per game instead of per team per game. A single regular-season game with 20 active players produces 20 rows.

### Final dimensions

| | |
|---|---|
| Rows | **673,733** (one per player-appearance) |
| Columns | 24 |
| Date range | 1999-11-02 → 2026-04-12 |
| Unique players | 2,653 |

### What's in it

- **Identifiers:** `season_id`, `player_id`, `player_name`, `team_id`, `team_abbreviation`, `game_id`, `game_date`
- **Box score:** `MIN`, `PTS`, `REB`, `AST`, `STL`, `BLK`, `TOV`
- **Shooting:** `FGM`/`FGA`/`FG_PCT`, `FG3M`/`FG3A`/`FG3_PCT`, `FTM`/`FTA`/`FT_PCT`
- **Plus/minus:** `PLUS_MINUS`

### What it powers in the model

This is the **central dataset.** Most of the engineered features come from it directly or are computed against it:

- **The target itself.** DK fantasy points (`PTS×1 + REB×1.25 + AST×1.5 + STL×2 + BLK×2 + TOV×−0.5 + FG3M×0.5`) is computed row-by-row from these columns
- **27 player rolling features** — L3/L5/L10 averages of FANTASY_PTS, PTS, REB, AST, STL, BLK, TOV, FG3M, MIN
- **9 efficiency rolling features** — L3/L5/L10 of FGA, FG_PCT, PLUS_MINUS
- **8 trend features** — L3 minus L10 for each of the eight production stats
- **Missing-teammates feature** — derived entirely from this file by checking which players from a team's L10 rotation are absent from each game's box score. This single derivation produced 84% of the model's total improvement over the linear baseline.

We also filter this dataset to `MIN >= 10` before training to remove garbage-time appearances that don't reflect a player's actual role.

---

## Dataset 3: Player biographical info

**File:** `data/nba_player_info.csv`
**Script:** `collect_player_info.py`
**Endpoint:** `CommonPlayerInfo`
**API calls:** 2,648 (one per unique player ID found in Dataset 2)
**Runtime:** ~35 minutes

### What the endpoint returns

A different endpoint from the first two. `CommonPlayerInfo` returns a single row of biographical data for one player at a time. Slow because it must be called per-player rather than per-season.

### Final dimensions

| | |
|---|---|
| Rows | **2,648** (one per player) |
| Columns | 13 |

### What's in it

- **Identifiers:** `person_id`, `display_name`
- **Static traits:** `birthdate`, `height` (string like `"6-7"`), `weight`, `position` (e.g. `"Guard"`, `"Forward-Center"`), `country`
- **Career markers:** `season_exp`, `draft_year`, `draft_round`, `draft_number`, `from_year`, `to_year`

### What it powers in the model

- **Position one-hot encoding** — `pos_G`, `pos_F`, `pos_C` (we bucket the full-word positions like "Guard-Forward" into the dominant category)
- **`height_in`** — height converted to inches
- **`years_experience`** — game-date year minus draft year
- **Defense-vs-Position (DvP) feature** — the most important downstream use. Once we know each player's position, we can compute the rolling fantasy points an opponent has allowed to players of that position over their last 20 games. This single feature contributed -0.014 RMSE in the ablation.

---

## Two sources we considered and dropped

For completeness, the data-sourcing story includes two paths we explored and ultimately removed from the final model:

### Vegas odds (dropped after testing)

- **Sources:** Scraped 19,636 historical games from `sportsbookreviewsonline.com` (2007-08 through mid-2022-23) plus the Kaggle dataset `cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024` (extending coverage through 2024-25, 23,115 total games)
- **Features added:** `vegas_spread`, `vegas_total`, `vegas_implied_total` (market-implied team scoring)
- **Ablation result:** -0.009 RMSE — barely above noise
- **Why dropped:** Team-level rolling stats already capture most of the team-scoring signal; Vegas was largely redundant. Not worth the maintenance burden of two extra data-collection scripts and a Kaggle dependency for sub-1% improvement.

### Advanced box scores (dropped due to API rate limit)

- **Source:** NBA Stats API `BoxScoreAdvancedV2` endpoint
- **Features intended:** PACE, OFF_RATING, DEF_RATING, TS_PCT, EFG_PCT, USG_PCT — sophisticated efficiency metrics tracked since the 2007-08 season
- **Why dropped:** Requires one API call per game (32k+ calls, ~7 hours). The NBA Stats API rate-limited us partway through and started returning empty bodies; no usable data was ever written. Given the expected impact (-0.1 to -0.3 RMSE based on related DFS work) was modest relative to the operational cost, we cut it from scope.

---

## Why this minimal-source approach worked

Our final model uses 107 features, but they all derive from these three CSVs. The lesson from the ablation: **clever derivation of one good source beats integration of many mediocre sources.**

The single highest-impact feature in the entire model — missing teammates (-0.249 RMSE) — uses no external data at all. It's a derived signal computed from `nba_player_game_logs.csv` alone. Vegas data, which is widely cited as the most predictive feature in DFS literature, contributed less than 1/25th of that.

By staying within the NBA Stats API we:

- **Eliminated the brittlest part of any data pipeline** — third-party scrapes break when source sites change layouts
- **Kept collection runtime under 40 minutes end-to-end** (vs. 7+ hours with advanced box scores)
- **Made the project trivially reproducible** by teammates — one `pip install`, one `python collect_all.py`, no accounts to create
- **Avoided licensing and attribution complications** — `nba_api` is MIT-licensed and the underlying NBA Stats data is publicly accessible

---

## Quick reference: collection commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all collection (one-shot, ~40 min)
python collect_all.py

# Or run scripts individually
python collect_data.py          # ~1 min  — team game logs
python collect_players.py       # ~2 min  — player game logs
python collect_player_info.py   # ~35 min — player biographical info

# Incremental refresh (current season only)
python collect_data.py --update
python collect_players.py --update
```

All scripts are checkpoint-safe: they append to existing files and skip already-fetched IDs, so Ctrl-C and resume is fine.
