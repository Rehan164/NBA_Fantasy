import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

print("=" * 60)
print("RAW DATA SUMMARY")
print("=" * 60)

# Player game logs
plog = pd.read_csv(DATA_DIR / "nba_player_game_logs.csv")
plog["GAME_DATE"] = pd.to_datetime(plog["GAME_DATE"])
print(f"\nPLAYER GAME LOGS: {DATA_DIR / 'nba_player_game_logs.csv'}")
print(f"  rows:        {len(plog):,}")
print(f"  unique players: {plog['PLAYER_ID'].nunique():,}")
print(f"  unique games:   {plog['GAME_ID'].nunique():,}")
print(f"  date range:  {plog['GAME_DATE'].min().date()} to {plog['GAME_DATE'].max().date()}")
print(f"  seasons:     {plog['SEASON_ID'].nunique() if 'SEASON_ID' in plog else 'n/a'}")
print(f"  columns:     {len(plog.columns)}")
print(f"  cols:        {list(plog.columns)}")

# Team game logs
games = pd.read_csv(DATA_DIR / "nba_historical_games.csv")
games["date"] = pd.to_datetime(games["date"])
print(f"\nTEAM GAMES: {DATA_DIR / 'nba_historical_games.csv'}")
print(f"  rows:        {len(games):,}")
print(f"  date range:  {games['date'].min().date()} to {games['date'].max().date()}")
print(f"  seasons:     {games['date'].dt.year.nunique()}")
print(f"  columns:     {len(games.columns)}")

# Player info
pinfo = pd.read_csv(DATA_DIR / "nba_player_info.csv")
print(f"\nPLAYER INFO: {DATA_DIR / 'nba_player_info.csv'}")
print(f"  rows:        {len(pinfo):,}")
print(f"  columns:     {len(pinfo.columns)}")
print(f"  position counts:")
if "POSITION" in pinfo.columns:
    print(pinfo["POSITION"].value_counts().head(10).to_string())

# Features
import json
feat = pd.read_csv(DATA_DIR / "nba_features.csv", parse_dates=["GAME_DATE"])
manifest = json.load(open(DATA_DIR / "nba_features_manifest.json"))
print(f"\nFEATURES: {DATA_DIR / 'nba_features.csv'}")
print(f"  rows:        {len(feat):,}")
print(f"  date range:  {feat['GAME_DATE'].min().date()} to {feat['GAME_DATE'].max().date()}")
total_features = sum(len(v) for v in manifest['groups'].values())
print(f"  total feature cols: {total_features}")
print(f"  feature groups:")
for grp, cols in manifest["groups"].items():
    print(f"    {grp:12s}: {len(cols)}")

# Target distribution
print(f"\nFANTASY_PTS distribution (clean rows, MIN >= 10):")
print(feat["FANTASY_PTS"].describe().to_string())
