# Phase 3: Feature engineering.
# Builds the model-ready feature matrix from the cleaned game logs.
# Output: data/nba_features.csv (one row per player-game, ~561k rows, ~115 cols)
#         data/nba_features_manifest.json (group -> column-list mapping)

import json

import numpy as np
import pandas as pd

from config import DATA_DIR
from process_data import load_processed

PLAYER_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M", "MIN"]
TEAM_STATS = ["score", "fg_made", "fg3_made", "reb", "ast", "stl", "blk", "tov"]
EFF_STATS = ["FGA", "FG_PCT", "PLUS_MINUS"]
WINDOWS = [3, 5, 10]


def main():
    print("Loading processed data...")
    player_logs, games = load_processed()
    print(f"  player logs: {player_logs.shape}")
    print(f"  games:       {games.shape}")

    # Player rolling features
    # build lag-1..10 first, then average them into L3/L5/L10
    player_lag_cols = []
    for stat in PLAYER_STATS:
        for lag in range(1, 11):
            col = f"player_{stat}_lag{lag}"
            player_logs[col] = player_logs.groupby("PLAYER_ID")[stat].shift(lag)
            player_lag_cols.append(col)

    player_roll_cols = []
    for stat in PLAYER_STATS:
        for w in WINDOWS:
            col = f"player_{stat}_L{w}"
            player_logs[col] = player_logs[
                [f"player_{stat}_lag{l}" for l in range(1, w + 1)]
            ].mean(axis=1)
            player_roll_cols.append(col)
    print(f"  player rolling: {len(player_roll_cols)}")

    # Player FANTASY_PTS rolling
    fp_roll_cols = []
    for w in WINDOWS:
        col = f"player_FANTASY_PTS_L{w}"
        player_logs[col] = (
            player_logs.groupby("PLAYER_ID")["FANTASY_PTS"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
        )
        fp_roll_cols.append(col)
    print(f"  FP rolling:     {len(fp_roll_cols)}")

    # Team rolling features
    home = games[["game_id_int", "date", "home_team",
                  "home_score", "home_fg_made", "home_fg3_made",
                  "home_reb", "home_ast", "home_stl", "home_blk", "home_tov"]].copy()
    home.columns = ["game_id_int", "date", "team"] + TEAM_STATS

    away = games[["game_id_int", "date", "away_team",
                  "away_score", "away_fg_made", "away_fg3_made",
                  "away_reb", "away_ast", "away_stl", "away_blk", "away_tov"]].copy()
    away.columns = ["game_id_int", "date", "team"] + TEAM_STATS

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(["team", "date"]).reset_index(drop=True)

    team_lag_cols = []
    for stat in TEAM_STATS:
        for lag in range(1, 11):
            col = f"team_{stat}_lag{lag}"
            team_games[col] = team_games.groupby("team")[stat].shift(lag)
            team_lag_cols.append(col)

    team_roll_cols = []
    for stat in TEAM_STATS:
        for w in WINDOWS:
            col = f"team_{stat}_L{w}"
            team_games[col] = team_games[
                [f"team_{stat}_lag{l}" for l in range(1, w + 1)]
            ].mean(axis=1)
            team_roll_cols.append(col)
    print(f"  team rolling:   {len(team_roll_cols)}")

    # Opponent rolling (= the other team's team rolling for the same game)
    opp_map = pd.concat([
        games[["game_id_int", "home_team", "away_team"]].rename(
            columns={"home_team": "team", "away_team": "opponent"}),
        games[["game_id_int", "away_team", "home_team"]].rename(
            columns={"away_team": "team", "home_team": "opponent"}),
    ], ignore_index=True)
    team_games = team_games.merge(opp_map, on=["game_id_int", "team"], how="left")

    opp_roll_cols = [c.replace("team_", "opp_") for c in team_roll_cols]
    opp_lookup = team_games[["game_id_int", "team"] + team_roll_cols].rename(
        columns={"team": "opponent",
                 **{c: c.replace("team_", "opp_") for c in team_roll_cols}}
    )
    team_games = team_games.merge(opp_lookup, on=["game_id_int", "opponent"], how="left")
    print(f"  opp rolling:    {len(opp_roll_cols)}")

    # Merge player + team/opp
    df = player_logs.merge(
        team_games[["game_id_int", "team"] + team_roll_cols + opp_roll_cols].drop_duplicates(
            subset=["game_id_int", "team"]
        ),
        left_on=["game_id_int", "TEAM_ABBREVIATION"],
        right_on=["game_id_int", "team"],
        how="inner",
    )

    # Game context
    home_lookup = games[["game_id_int", "home_team"]].copy()
    df = df.merge(home_lookup, on="game_id_int", how="left")
    df["is_home"] = (df["TEAM_ABBREVIATION"] == df["home_team"]).astype(int)

    team_sched = (
        df[["TEAM_ABBREVIATION", "GAME_DATE", "game_id_int"]]
        .drop_duplicates(subset=["TEAM_ABBREVIATION", "game_id_int"])
        .sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
        .reset_index(drop=True)
    )
    team_sched["days_rest"] = (
        team_sched.groupby("TEAM_ABBREVIATION")["GAME_DATE"]
        .diff().dt.days.clip(upper=7).fillna(3).astype(int)
    )
    df = df.merge(
        team_sched[["game_id_int", "TEAM_ABBREVIATION", "days_rest"]],
        on=["game_id_int", "TEAM_ABBREVIATION"], how="left",
    )

    opp_key = pd.concat([
        games[["game_id_int", "home_team", "away_team"]].rename(
            columns={"home_team": "TEAM_ABBREVIATION", "away_team": "_opp"}),
        games[["game_id_int", "away_team", "home_team"]].rename(
            columns={"away_team": "TEAM_ABBREVIATION", "home_team": "_opp"}),
    ], ignore_index=True)
    opp_rest = team_sched[["game_id_int", "TEAM_ABBREVIATION", "days_rest"]].rename(
        columns={"TEAM_ABBREVIATION": "_opp", "days_rest": "opp_days_rest"}
    )
    opp_key = opp_key.merge(opp_rest, on=["game_id_int", "_opp"], how="left")
    df = df.merge(
        opp_key[["game_id_int", "TEAM_ABBREVIATION", "opp_days_rest"]],
        on=["game_id_int", "TEAM_ABBREVIATION"], how="left",
    )
    context_cols = ["is_home", "days_rest", "opp_days_rest"]
    print(f"  context:        {len(context_cols)}")

    # Trends (L3 - L10)
    trend_cols = []
    for stat in PLAYER_STATS:
        col = f"player_{stat}_trend"
        df[col] = df[f"player_{stat}_L3"] - df[f"player_{stat}_L10"]
        trend_cols.append(col)
    print(f"  trend:          {len(trend_cols)}")

    # Player rolling efficiency
    eff_cols = []
    for stat in EFF_STATS:
        for w in WINDOWS:
            col = f"player_{stat}_L{w}"
            player_logs[col] = (
                player_logs.groupby("PLAYER_ID")[stat]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=w).mean())
            )
            eff_cols.append(col)
    df = df.merge(
        player_logs[["PLAYER_ID", "game_id_int"] + eff_cols],
        on=["PLAYER_ID", "game_id_int"], how="left",
    )
    print(f"  efficiency:     {len(eff_cols)}")

    # Missing teammates (continuous deficit)
    # For each (team, game): sum the L10 minutes of players in the box,
    # compare to the team's rolling baseline of that sum.
    # Deficit = baseline - actual = absent rotation minutes.
    player_logs["MIN_L10"] = (
        player_logs.groupby("PLAYER_ID")["MIN"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    team_game_min = (
        player_logs.dropna(subset=["MIN_L10"])
        .groupby(["game_id_int", "TEAM_ABBREVIATION"])["MIN_L10"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "team_l10_min_played", "count": "team_players_played"})
    )
    team_game_min = team_game_min.merge(
        games[["game_id_int", "date"]], on="game_id_int", how="left"
    ).sort_values(["TEAM_ABBREVIATION", "date"]).reset_index(drop=True)

    team_game_min["team_l10_min_baseline"] = (
        team_game_min.groupby("TEAM_ABBREVIATION")["team_l10_min_played"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )
    team_game_min["missing_min_deficit"] = (
        team_game_min["team_l10_min_baseline"] - team_game_min["team_l10_min_played"]
    )

    missing_cols = ["team_l10_min_played", "team_players_played", "missing_min_deficit"]
    df = df.merge(
        team_game_min[["game_id_int", "TEAM_ABBREVIATION"] + missing_cols],
        on=["game_id_int", "TEAM_ABBREVIATION"], how="left",
    )
    print(f"  missing:        {len(missing_cols)}")

    # Schedule density
    team_sched_sd = team_sched.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    team_sched_sd["is_b2b"] = (team_sched_sd["days_rest"] == 1).astype(int)

    def games_in_last_n_days(group, n):
        dates = group["GAME_DATE"].values.astype("datetime64[D]")
        out = np.zeros(len(dates), dtype=int)
        for i in range(len(dates)):
            out[i] = ((dates < dates[i]) & (dates >= dates[i] - np.timedelta64(n, "D"))).sum()
        return pd.Series(out, index=group.index)

    team_sched_sd["games_last_4d"] = (
        team_sched_sd.groupby("TEAM_ABBREVIATION", group_keys=False)
        .apply(lambda g: games_in_last_n_days(g, 4), include_groups=False)
    )
    team_sched_sd["games_last_7d"] = (
        team_sched_sd.groupby("TEAM_ABBREVIATION", group_keys=False)
        .apply(lambda g: games_in_last_n_days(g, 7), include_groups=False)
    )

    df = df.merge(
        team_sched_sd[["game_id_int", "TEAM_ABBREVIATION", "is_b2b", "games_last_4d", "games_last_7d"]],
        on=["game_id_int", "TEAM_ABBREVIATION"], how="left",
    )
    schedule_cols = ["is_b2b", "games_last_4d", "games_last_7d"]
    print(f"  schedule:       {len(schedule_cols)}")

    # Position + biographical
    position_path = DATA_DIR / "nba_player_info.csv"
    if not position_path.exists():
        raise FileNotFoundError(
            f"{position_path} missing - run collect_player_info.py first"
        )
    pinfo = pd.read_csv(position_path)

    # NBA's full-word positions -> G/F/C buckets
    def bucket_position(p):
        if not isinstance(p, str):
            return "U"
        p = p.lower()
        if "guard" in p:   return "G"
        if "center" in p:  return "C"
        if "forward" in p: return "F"
        return "U"
    pinfo["pos_bucket"] = pinfo["POSITION"].apply(bucket_position)

    pinfo["height_in"] = pinfo["HEIGHT"].apply(
        lambda h: int(h.split("-")[0]) * 12 + int(h.split("-")[1])
        if isinstance(h, str) and "-" in h else np.nan
    )
    pinfo["draft_year"] = pd.to_numeric(pinfo["DRAFT_YEAR"], errors="coerce")

    df = df.merge(
        pinfo[["PERSON_ID", "pos_bucket", "height_in", "draft_year"]].rename(
            columns={"PERSON_ID": "PLAYER_ID"}
        ),
        on="PLAYER_ID", how="left",
    )
    for pos in ["G", "F", "C"]:
        df[f"pos_{pos}"] = (df["pos_bucket"] == pos).astype(int)
    df["years_experience"] = df["GAME_DATE"].dt.year - df["draft_year"]

    position_cols = ["pos_G", "pos_F", "pos_C", "height_in", "years_experience"]
    print(f"  position:       {len(position_cols)}")

    # Defense vs Position (DvP, L20)
    g_opp = pd.concat([
        games[["game_id_int", "home_team", "away_team"]].rename(
            columns={"home_team": "TEAM_ABBREVIATION", "away_team": "opp_team"}),
        games[["game_id_int", "away_team", "home_team"]].rename(
            columns={"away_team": "TEAM_ABBREVIATION", "home_team": "opp_team"}),
    ], ignore_index=True).drop_duplicates(subset=["game_id_int", "TEAM_ABBREVIATION"])
    df = df.merge(g_opp, on=["game_id_int", "TEAM_ABBREVIATION"], how="left")

    dvp_base = (
        df.dropna(subset=["pos_bucket"])
        .groupby(["game_id_int", "opp_team", "pos_bucket"])
        .agg(fp_sum=("FANTASY_PTS", "sum"), n=("FANTASY_PTS", "count"))
        .reset_index()
    )
    dvp_base["fp_per_player"] = dvp_base["fp_sum"] / dvp_base["n"]
    dvp_base = dvp_base.merge(games[["game_id_int", "date"]], on="game_id_int", how="left")
    dvp_base = dvp_base.sort_values(["opp_team", "pos_bucket", "date"]).reset_index(drop=True)
    dvp_base["dvp_L20"] = (
        dvp_base.groupby(["opp_team", "pos_bucket"])["fp_per_player"]
        .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    )
    df = df.merge(
        dvp_base[["game_id_int", "opp_team", "pos_bucket", "dvp_L20"]],
        on=["game_id_int", "opp_team", "pos_bucket"], how="left",
    )
    dvp_cols = ["dvp_L20"]
    print(f"  dvp:            {len(dvp_cols)}")

    # Filter to clean training rows
    rolling_cols = player_roll_cols + fp_roll_cols + team_roll_cols + opp_roll_cols
    required = rolling_cols + trend_cols + eff_cols
    df = df.dropna(subset=required).reset_index(drop=True)
    df = df[df["MIN"] >= 10].reset_index(drop=True)
    print(f"  clean rows:     {len(df):,}")

    # Assemble manifest
    id_cols = ["PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "game_id_int",
               "TEAM_ABBREVIATION", "MIN"]
    target = "FANTASY_PTS"
    target_components = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]

    groups = {
        "rolling":    rolling_cols,
        "context":    context_cols,
        "trends":     trend_cols,
        "efficiency": eff_cols,
        "missing":    missing_cols,
        "schedule":   schedule_cols,
        "position":   position_cols,
        "dvp":        dvp_cols,
    }
    manifest = {
        "id_cols": id_cols,
        "target": target,
        "target_components": target_components,
        "groups": groups,
    }

    all_features = []
    for cols in groups.values():
        all_features += cols
    all_features = list(dict.fromkeys(all_features))

    keep = id_cols + [target] + target_components + all_features
    keep = list(dict.fromkeys(keep))

    out_csv = DATA_DIR / "nba_features.csv"
    out_json = DATA_DIR / "nba_features_manifest.json"
    df[keep].to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  wrote {out_csv}  ({out_csv.stat().st_size / 1e6:.1f} MB)")
    print(f"  wrote {out_json}")
    print(f"  total features: {len(all_features)}")


if __name__ == "__main__":
    main()
