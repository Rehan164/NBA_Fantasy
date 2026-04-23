"""
NBA Player Info Collection (one row per player)

For each unique PLAYER_ID in nba_player_game_logs.csv, fetch CommonPlayerInfo
to get static player traits: position, height, weight, draft year, country, birthdate.

Output: data/nba_player_info.csv

Checkpointing: appends as it goes; resume-safe (skips already-fetched IDs).
~5,000 players * 0.6s = ~50 minutes.

Usage:
    python collect_player_info.py
"""

import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from nba_api.stats.endpoints import commonplayerinfo

from config import DATA_DIR, NBA_API_DELAY, PLAYER_GAME_LOGS_CSV

PLAYER_INFO_CSV = DATA_DIR / "nba_player_info.csv"

KEEP_FIELDS = [
    "PERSON_ID",
    "DISPLAY_FIRST_LAST",
    "BIRTHDATE",
    "HEIGHT",
    "WEIGHT",
    "SEASON_EXP",
    "POSITION",
    "COUNTRY",
    "DRAFT_YEAR",
    "DRAFT_ROUND",
    "DRAFT_NUMBER",
    "FROM_YEAR",
    "TO_YEAR",
]


def fetch_one(player_id: int) -> dict | None:
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        df = info.common_player_info.get_data_frame()
        if df.empty:
            return None
        row = df.iloc[0]
        return {f: row.get(f) for f in KEEP_FIELDS}
    except Exception as e:
        print(f"  Error for player {player_id}: {e}")
        return None


def main():
    print("=" * 60)
    print("NBA Player Info Collection")
    print("=" * 60)

    # Load all unique player IDs from existing logs
    logs = pd.read_csv(PLAYER_GAME_LOGS_CSV, usecols=["PLAYER_ID"])
    all_ids = sorted(logs["PLAYER_ID"].unique())
    print(f"Unique player IDs in logs: {len(all_ids):,}")

    # Resume from existing file if present
    existing_ids = set()
    if PLAYER_INFO_CSV.exists():
        existing = pd.read_csv(PLAYER_INFO_CSV)
        existing_ids = set(existing["PERSON_ID"].astype(int).tolist())
        print(f"Already fetched: {len(existing_ids):,}")

    todo = [pid for pid in all_ids if pid not in existing_ids]
    print(f"To fetch: {len(todo):,}")
    if not todo:
        print("Nothing to do.")
        return

    # Open in append mode; write header only if new file
    write_header = not PLAYER_INFO_CSV.exists()

    rows_buffer = []
    for i, pid in enumerate(tqdm(todo, desc="Fetching")):
        row = fetch_one(int(pid))
        if row is not None:
            rows_buffer.append(row)
        time.sleep(NBA_API_DELAY)

        # Flush every 100 to be checkpoint-safe
        if len(rows_buffer) >= 100 or i == len(todo) - 1:
            pd.DataFrame(rows_buffer).to_csv(
                PLAYER_INFO_CSV, mode="a", header=write_header, index=False
            )
            write_header = False
            rows_buffer = []

    print(f"\nDone. Output: {PLAYER_INFO_CSV}")


if __name__ == "__main__":
    main()
