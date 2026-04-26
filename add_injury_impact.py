"""
Add Missing Player Impact Features

Derives injury/absence impact from existing player game logs.
For each game, identifies key players missing from each team
by comparing against their recent active roster.

No additional API calls needed — uses nba_player_game_logs.csv.

New columns added to nba_historical_games.csv:
  home_missing_players   - count of significant players missing
  home_missing_ppg       - sum of their rolling avg PPG
  home_missing_min       - sum of their rolling avg minutes
  home_top_missing_ppg   - highest single missing player's PPG
  (same four for away_)

Usage:
    python add_injury_impact.py
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import HISTORICAL_GAMES_CSV, PLAYER_GAME_LOGS_CSV

# Columns this script adds (used for idempotent re-runs)
NEW_COLS = [
    "home_missing_players", "home_missing_ppg",
    "home_missing_min", "home_top_missing_ppg",
    "home_missing_names",
    "away_missing_players", "away_missing_ppg",
    "away_missing_min", "away_top_missing_ppg",
    "away_missing_names",
]

LOOKBACK = 10          # recent team games to consider
MIN_APPEAR_FRAC = 0.5  # must play in >= 50% of recent games
SIGNIFICANT_MIN = 15   # rolling avg minutes threshold


def compute_missing_impact(plogs: pd.DataFrame) -> pd.DataFrame:
    """
    For every (game_id, team), compute the impact of missing players.

    Returns a DataFrame with columns:
        game_id, team, missing_count, missing_ppg, missing_min, top_missing_ppg
    """
    results = []

    for team, team_df in tqdm(plogs.groupby("TEAM_ABBREVIATION"), desc="Teams"):
        # Ordered unique games for this team
        team_games = (
            team_df.groupby("GAME_ID")["GAME_DATE"]
            .first().sort_values().reset_index()
        )
        game_ids = team_games["GAME_ID"].values

        # Pre-build per-game lookups for speed
        game_players = {}      # game_id -> set of player_ids
        game_player_stats = {} # game_id -> {player_id: {roll_PTS, roll_MIN, name}}

        for gid, gdf in team_df.groupby("GAME_ID"):
            game_players[gid] = set(gdf["PLAYER_ID"].values)
            game_player_stats[gid] = {
                row.PLAYER_ID: {
                    "roll_PTS": row.roll_PTS,
                    "roll_MIN": row.roll_MIN,
                    "name": row.PLAYER_NAME,
                }
                for row in gdf.itertuples()
            }

        for i, gid in enumerate(game_ids):
            recent_ids = game_ids[max(0, i - LOOKBACK):i]
            if len(recent_ids) < 3:
                continue

            # Build expected roster from recent games
            appearances = {}
            latest_stats = {}

            for rgid in recent_ids:
                for pid, stats in game_player_stats.get(rgid, {}).items():
                    appearances[pid] = appearances.get(pid, 0) + 1
                    latest_stats[pid] = stats  # most recent overwrites

            min_app = max(1, int(len(recent_ids) * MIN_APPEAR_FRAC))

            expected = {
                pid for pid, cnt in appearances.items()
                if cnt >= min_app
                and latest_stats[pid]["roll_MIN"] >= SIGNIFICANT_MIN
            }

            actual = game_players.get(gid, set())
            missing = expected - actual

            m_ppg = sum(latest_stats[p]["roll_PTS"] for p in missing)
            m_min = sum(latest_stats[p]["roll_MIN"] for p in missing)
            top_ppg = max((latest_stats[p]["roll_PTS"] for p in missing), default=0.0)
            names = sorted(latest_stats[p]["name"] for p in missing)

            results.append({
                "game_id": gid,
                "team": team,
                "missing_count": len(missing),
                "missing_ppg": round(m_ppg, 1),
                "missing_min": round(m_min, 1),
                "top_missing_ppg": round(top_ppg, 1),
                "missing_names": "{" + ", ".join(names) + "}" if names else "",
            })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Adding Missing Player Impact Features")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────
    print("\n1. Loading data...")
    plogs = pd.read_csv(PLAYER_GAME_LOGS_CSV)
    games = pd.read_csv(HISTORICAL_GAMES_CSV)

    plogs["GAME_DATE"] = pd.to_datetime(plogs["GAME_DATE"])
    plogs["MIN"] = pd.to_numeric(plogs["MIN"], errors="coerce").fillna(0)
    games["date"] = pd.to_datetime(games["date"])

    print(f"   Player logs: {len(plogs):,}")
    print(f"   Games:       {len(games):,}")

    # Drop columns from prior runs
    games = games.drop(columns=[c for c in NEW_COLS if c in games.columns])

    # ── Rolling averages (shifted to prevent leakage) ───────────────
    print("\n2. Computing player rolling averages...")
    plogs = plogs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    for stat in ["PTS", "MIN"]:
        plogs[f"roll_{stat}"] = (
            plogs.groupby("PLAYER_ID")[stat]
            .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        )

    # Keep only rows with valid rolling stats
    plogs_valid = plogs[plogs["roll_MIN"].notna()].copy()
    print(f"   Rows with rolling stats: {len(plogs_valid):,}")

    # ── Compute missing impact ──────────────────────────────────────
    print("\n3. Computing missing player impact...")
    missing_df = compute_missing_impact(plogs_valid)
    print(f"   Computed for {len(missing_df):,} team-games")

    # ── Merge into games (home & away) ──────────────────────────────
    print("\n4. Merging into game-level features...")

    home = missing_df.rename(columns={
        "missing_count":    "home_missing_players",
        "missing_ppg":      "home_missing_ppg",
        "missing_min":      "home_missing_min",
        "top_missing_ppg":  "home_top_missing_ppg",
        "missing_names":    "home_missing_names",
    })
    home_cols = ["game_id", "team", "home_missing_players", "home_missing_ppg",
                 "home_missing_min", "home_top_missing_ppg", "home_missing_names"]
    games = games.merge(
        home[home_cols],
        left_on=["game_id", "home_team"],
        right_on=["game_id", "team"],
        how="left",
    ).drop(columns=["team"])

    away = missing_df.rename(columns={
        "missing_count":    "away_missing_players",
        "missing_ppg":      "away_missing_ppg",
        "missing_min":      "away_missing_min",
        "top_missing_ppg":  "away_top_missing_ppg",
        "missing_names":    "away_missing_names",
    })
    away_cols = ["game_id", "team", "away_missing_players", "away_missing_ppg",
                 "away_missing_min", "away_top_missing_ppg", "away_missing_names"]
    games = games.merge(
        away[away_cols],
        left_on=["game_id", "away_team"],
        right_on=["game_id", "team"],
        how="left",
    ).drop(columns=["team"])

    # Fill NaN — numeric with 0, names with empty string
    numeric_cols = [c for c in NEW_COLS if "names" not in c]
    games[numeric_cols] = games[numeric_cols].fillna(0)
    games["home_missing_names"] = games["home_missing_names"].fillna("")
    games["away_missing_names"] = games["away_missing_names"].fillna("")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    has_missing = (games["home_missing_players"] + games["away_missing_players"]) > 0
    print(f"Games with >= 1 missing player: {has_missing.sum():,} / {len(games):,} "
          f"({has_missing.mean()*100:.1f}%)")
    print(f"Avg missing players (home): {games['home_missing_players'].mean():.2f}")
    print(f"Avg missing PPG     (home): {games['home_missing_ppg'].mean():.1f}")
    print(f"Avg missing players (away): {games['away_missing_players'].mean():.2f}")
    print(f"Avg missing PPG     (away): {games['away_missing_ppg'].mean():.1f}")

    print(f"\nTop 10 games by combined missing PPG:")
    games["_tmp"] = games["home_missing_ppg"] + games["away_missing_ppg"]
    top = games.nlargest(10, "_tmp")
    print(top[["date", "home_team", "away_team", "home_score", "away_score",
               "home_missing_players", "home_missing_ppg",
               "away_missing_players", "away_missing_ppg"]].to_string(index=False))
    games = games.drop(columns=["_tmp"])

    # ── Save ────────────────────────────────────────────────────────
    print(f"\nColumns before save: {[c for c in games.columns if 'missing' in c]}")
    print(f"DataFrame shape: {games.shape}")

    out_path = HISTORICAL_GAMES_CSV
    try:
        games.to_csv(out_path, index=False)
    except PermissionError:
        print(f"\nERROR: Cannot write to {out_path} — is the file open in Excel?")
        print("Close Excel and re-run.")
        return

    # Verify the save actually worked
    verify = pd.read_csv(out_path, nrows=1)
    saved_cols = [c for c in verify.columns if "missing" in c]
    if saved_cols:
        print(f"\nSaved and verified: {out_path}")
        print(f"New columns confirmed: {saved_cols}")
    else:
        print(f"\nWARNING: Save appeared to succeed but columns not found in file!")
        print(f"File columns: {list(verify.columns)}")


if __name__ == "__main__":
    main()
