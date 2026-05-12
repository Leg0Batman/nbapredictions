"""
NBA Feature Engineering
Erstellt aussagekräftige Features für das ML-Modell aus Rohdaten.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")


def build_team_rolling_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Berechnet Rolling-Averages (letzte N Spiele) pro Team:
    - Offense/Defense Rating
    - Win Streak
    - Home/Away Performance
    - Rest Days
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])
    df["WL_NUM"] = (df["WL"] == "W").astype(int)
    
    rolling_cols = [
        "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
        "REB", "AST", "TOV", "STL", "BLK",
        "PLUS_MINUS", "WL_NUM"
    ]
    
    for col in rolling_cols:
        if col in df.columns:
            df[f"ROLL_{window}_{col}"] = (
                df.groupby("TEAM_ID")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
            )
    
    # Win Streak (positiv = Gewinnserie, negativ = Verlustserie)
    df["WIN_STREAK"] = df.groupby("TEAM_ID")["WL_NUM"].transform(
        lambda x: x.shift(1).groupby((x.shift(1) != x.shift(2)).cumsum()).cumcount() + 1
    )
    
    # Rest Days
    df["REST_DAYS"] = (
        df.groupby("TEAM_ID")["GAME_DATE"]
        .transform(lambda x: x.diff().dt.days.fillna(3))
    )
    df["REST_DAYS"] = df["REST_DAYS"].clip(1, 10)
    
    return df


def merge_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Führt Home- und Away-Team für jedes Spiel zusammen.
    Erstellt ein Sample pro Spiel (statt 2 Zeilen pro Team).
    """
    # Trenne Home und Away anhand von MATCHUP
    df["IS_HOME"] = df["MATCHUP"].str.contains(r" vs\. ", regex=True).astype(int)
    
    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()
    
    # Merge über GAME_ID
    roll_cols = [c for c in df.columns if c.startswith("ROLL_") or c in ["WIN_STREAK", "REST_DAYS"]]
    
    home_feats = home[["GAME_ID", "GAME_DATE", "TEAM_ID", "WL_NUM"] + roll_cols].copy()
    away_feats = away[["GAME_ID", "TEAM_ID"] + roll_cols].copy()
    
    home_feats.columns = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "HOME_WIN"] + [f"HOME_{c}" for c in roll_cols]
    away_feats.columns = ["GAME_ID", "AWAY_TEAM_ID"] + [f"AWAY_{c}" for c in roll_cols]
    
    merged = home_feats.merge(away_feats, on="GAME_ID", how="inner")
    
    # Differenz-Features (Home minus Away)
    for col in roll_cols:
        h_col = f"HOME_{col}"
        a_col = f"AWAY_{col}"
        if h_col in merged.columns and a_col in merged.columns:
            merged[f"DIFF_{col}"] = merged[h_col] - merged[a_col]
    
    return merged


def add_playoff_flag(df: pd.DataFrame, season_games_raw: pd.DataFrame) -> pd.DataFrame:
    """Fügt Flag hinzu ob Playoff-Spiel."""
    # SEASON_ID beginnt mit '4' für Playoffs
    playoff_ids = season_games_raw[season_games_raw["SEASON_ID"].str.startswith("4")]["GAME_ID"].unique()
    df["IS_PLAYOFF"] = df["GAME_ID"].isin(playoff_ids).astype(int)
    return df


def add_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    H2H Win-Rate der letzten 10 Begegnungen zwischen den Teams.
    """
    df = df.sort_values("GAME_DATE")
    h2h_wins = {}
    h2h_records = []
    
    for _, row in df.iterrows():
        key = tuple(sorted([row["HOME_TEAM_ID"], row["AWAY_TEAM_ID"]]))
        history = h2h_wins.get(key, [])
        
        # Wie oft hat Home gewonnen in letzten 10 H2H?
        recent = history[-10:]
        h2h_home_winrate = np.mean(recent) if recent else 0.5
        h2h_records.append(h2h_home_winrate)
        
        # Update
        history.append(row["HOME_WIN"])
        h2h_wins[key] = history
    
    df["H2H_HOME_WINRATE"] = h2h_records
    return df


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vollständige Feature-Pipeline.
    Input: raw games DataFrame von nba_api
    Output: Feature-Matrix für Training
    """
    print("[Features] Berechne Rolling Stats...")
    df = build_team_rolling_stats(raw_df, window=10)
    
    print("[Features] Merge Matchup-Features...")
    df = merge_matchup_features(df)
    
    print("[Features] Head-to-Head Features...")
    df = add_head_to_head(df)
    
    # Entferne Zeilen mit zu vielen NaNs (Saisonstart)
    feature_cols = [c for c in df.columns if c.startswith(("HOME_ROLL", "AWAY_ROLL", "DIFF_", "H2H"))]
    df = df.dropna(subset=feature_cols[:5])  # Nur wenn Kernfeatures fehlen
    
    print(f"[Features] {len(df)} Spiele mit {len(feature_cols)} Features.")
    
    df.to_csv(DATA_DIR / "features.csv", index=False)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Gibt alle Feature-Spalten zurück (ohne Target und IDs)."""
    exclude = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "HOME_WIN"]
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]