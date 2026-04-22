# Features

All features are computed from the player's and team's history *before* the current game (no leakage). We use rolling windows of L3, L5, and L10 (last 3, 5, or 10 games).

---

## Player Rolling Averages (27 features)

Average of the last 3, 5, and 10 games for each of these stats:

| Stat | Description |
|------|-------------|
| FANTASY_PTS | DraftKings fantasy points scored |
| PTS | Points |
| REB | Rebounds |
| AST | Assists |
| STL | Steals |
| BLK | Blocks |
| TOV | Turnovers |
| FG3M | 3-pointers made |
| MIN | Minutes played |

We include MIN as its own rolling feature alongside the production stats rather than dividing each stat by minutes. This means the model sees both "how many minutes this player has been getting" and "how many points/rebounds/etc. they've been producing" as separate inputs, and learns the relationship between them on its own. For example, 15 points in 28 minutes looks very different from 15 points in 15 minutes once the model also sees the MIN feature.

Games where the player logged fewer than 10 minutes are excluded entirely, since those are garbage-time appearances that don't reflect the player's actual role.

---

## Team Rolling Averages (24 features)

Average of the last 3, 5, and 10 games for the player's **team**:

`score`, `fg_made`, `fg3_made`, `reb`, `ast`, `stl`, `blk`, `tov`

---

## Opponent Rolling Averages (24 features)

Same stats, same windows, for the **opposing team**.

---

## Game Context (3 features)

| Feature | Description |
|---------|-------------|
| is_home | 1 if the player's team is at home, 0 if away |
| days_rest | Days since the team's last game (capped at 7) |
| opp_days_rest | Days since the opponent's last game (capped at 7) |

---

## Player Trend (8 features)

For each of the 8 player stats above (excluding FANTASY_PTS), trend = L3 avg - L10 avg. Positive means the player is trending up recently.

---

## Efficiency Rolling Averages (9 features)

Average of the last 3, 5, and 10 games for:

| Stat | Description |
|------|-------------|
| FGA | Field goal attempts (usage/volume) |
| FG_PCT | Field goal percentage (shooting efficiency) |
| PLUS_MINUS | Point differential while on the court |

---

**Total: ~95 features**
