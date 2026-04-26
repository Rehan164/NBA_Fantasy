# Phase 2: Processing.
# Loads the raw collection CSVs, normalizes types, computes the DraftKings
# fantasy points target, and returns the cleaned DataFrames.
# Imported by build_features.py - does not write to disk.

import pandas as pd

from config import HISTORICAL_GAMES_CSV, PLAYER_GAME_LOGS_CSV


def load_processed():
    player_logs = pd.read_csv(PLAYER_GAME_LOGS_CSV)
    games = pd.read_csv(HISTORICAL_GAMES_CSV)

    player_logs["GAME_DATE"] = pd.to_datetime(player_logs["GAME_DATE"])
    games["date"] = pd.to_datetime(games["date"])

    # player logs use leading-zero string GAME_IDs ('0029900001'), games use ints
    player_logs["game_id_int"] = player_logs["GAME_ID"].astype(int)
    games["game_id_int"] = games["game_id"].astype(int)

    # MIN occasionally comes back as "30:45" string from the API
    player_logs["MIN"] = pd.to_numeric(player_logs["MIN"], errors="coerce")

    # 0 attempts -> NaN shooting %, treat as 0% so rolling averages don't drop rows
    player_logs["FG_PCT"] = player_logs["FG_PCT"].fillna(0)
    player_logs["FG3_PCT"] = player_logs["FG3_PCT"].fillna(0)

    # DraftKings scoring
    player_logs["FANTASY_PTS"] = (
        player_logs["PTS"] * 1.0
        + player_logs["REB"] * 1.25
        + player_logs["AST"] * 1.5
        + player_logs["STL"] * 2.0
        + player_logs["BLK"] * 2.0
        + player_logs["TOV"] * -0.5
        + player_logs["FG3M"] * 0.5
    )

    player_logs = player_logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    return player_logs, games
