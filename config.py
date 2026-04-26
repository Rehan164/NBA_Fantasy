"""
Configuration constants for NBA Fantasy data collection.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Output files
HISTORICAL_GAMES_CSV = DATA_DIR / "nba_historical_games.csv"
PLAYER_GAME_LOGS_CSV = DATA_DIR / "nba_player_game_logs.csv"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)

# Data Collection
START_SEASON = 2000  # First season to collect (2000 = 1999-2000 season)
END_SEASON = 2026    # Last season to collect (2026 = 2025-2026 season)

# Rate limiting for NBA API
NBA_API_DELAY = 0.6  # seconds between requests
