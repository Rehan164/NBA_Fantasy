"""
NBA Advanced Box Score Collection (per game)

For each unique GAME_ID in the existing data, fetch BoxScoreAdvancedV2 to get:
  - Team-level: PACE, OFF/DEF/NET_RATING, TS_PCT, EFG_PCT
  - Player-level: USG_PCT, TS_PCT, OFF/DEF_RATING, AST_PCT, REB_PCT

Outputs:
    data/nba_advanced_team_stats.csv
    data/nba_advanced_player_stats.csv

Both files use append+checkpoint so the script is resume-safe.
~32k games * 0.6s = ~5 hours.

Usage:
    python collect_advanced_box.py                # all games
    python collect_advanced_box.py --season 2020  # only games from season 2020 onward
"""

import sys
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from nba_api.stats.endpoints import boxscoreadvancedv2

from config import DATA_DIR, NBA_API_DELAY, HISTORICAL_GAMES_CSV, PLAYER_GAME_LOGS_CSV

ADV_TEAM_CSV   = DATA_DIR / "nba_advanced_team_stats.csv"
ADV_PLAYER_CSV = DATA_DIR / "nba_advanced_player_stats.csv"

TEAM_FIELDS = [
    "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION",
    "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "TS_PCT", "EFG_PCT", "AST_PCT", "REB_PCT", "TM_TOV_PCT",
]

PLAYER_FIELDS = [
    "GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "MIN",
    "USG_PCT", "TS_PCT", "OFF_RATING", "DEF_RATING",
    "AST_PCT", "REB_PCT", "TM_TOV_PCT", "PIE",
]


def fetch_one(game_id: str):
    try:
        box = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
        team_df = box.team_stats.get_data_frame()
        player_df = box.player_stats.get_data_frame()
        return (team_df[TEAM_FIELDS], player_df[PLAYER_FIELDS])
    except Exception as e:
        print(f"  Error for {game_id}: {e}")
        return None, None


def normalize_game_id(gid):
    """NBA GAME_IDs are 10-digit zero-padded strings."""
    return str(gid).zfill(10)


def main():
    print("=" * 60)
    print("NBA Advanced Box Score Collection")
    print("=" * 60)

    # min season filter
    min_season = None
    if "--season" in sys.argv:
        min_season = int(sys.argv[sys.argv.index("--season") + 1])
        print(f"Filtering to season >= {min_season}")

    # Load all unique GAME_IDs from games file (preferred — has season column)
    games = pd.read_csv(HISTORICAL_GAMES_CSV, usecols=["game_id", "season"])
    games["game_id"] = games["game_id"].apply(normalize_game_id)
    if min_season is not None:
        games = games[games["season"] >= min_season]
    all_ids = games["game_id"].unique().tolist()
    print(f"Game IDs to process: {len(all_ids):,}")

    # Resume support
    done_ids = set()
    if ADV_TEAM_CSV.exists():
        done = pd.read_csv(ADV_TEAM_CSV, usecols=["GAME_ID"], dtype={"GAME_ID": str})
        done_ids = set(done["GAME_ID"].apply(normalize_game_id).tolist())
        print(f"Already fetched: {len(done_ids):,}")

    todo = [gid for gid in all_ids if gid not in done_ids]
    print(f"To fetch: {len(todo):,}")
    if not todo:
        print("Nothing to do.")
        return

    write_header_t = not ADV_TEAM_CSV.exists()
    write_header_p = not ADV_PLAYER_CSV.exists()

    team_buf, player_buf = [], []
    for i, gid in enumerate(tqdm(todo, desc="Games")):
        team_df, player_df = fetch_one(gid)
        if team_df is not None and len(team_df) > 0:
            team_buf.append(team_df)
        if player_df is not None and len(player_df) > 0:
            player_buf.append(player_df)
        time.sleep(NBA_API_DELAY)

        # Flush every 100 games
        if (i + 1) % 100 == 0 or i == len(todo) - 1:
            if team_buf:
                pd.concat(team_buf, ignore_index=True).to_csv(
                    ADV_TEAM_CSV, mode="a", header=write_header_t, index=False
                )
                write_header_t = False
                team_buf = []
            if player_buf:
                pd.concat(player_buf, ignore_index=True).to_csv(
                    ADV_PLAYER_CSV, mode="a", header=write_header_p, index=False
                )
                write_header_p = False
                player_buf = []

    print(f"\nDone. Outputs: {ADV_TEAM_CSV} | {ADV_PLAYER_CSV}")


if __name__ == "__main__":
    main()
