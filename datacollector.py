"""
NBA Data Collector
Zieht historische Spieldaten + Team-Stats via nba_api
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import (
    leaguegamefinder,
    leaguestandings,
    scoreboardv2,
    teamdashboardbygeneralsplits,
    teamgamelogs,
)
from nba_api.stats.static import teams

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# NBA.com Rate Limiting — nicht zu schnell!
SLEEP_BETWEEN_CALLS = 0.7  # Sekunden


def get_all_teams():
    """Gibt alle NBA-Teams zurück."""
    return teams.get_teams()


def fetch_season_games(season: str = "2024-25") -> pd.DataFrame:
    """
    Holt alle Spiele einer Saison.
    season z.B. '2023-24', '2024-25'
    """
    print(f"[DataCollector] Hole Spiele für Saison {season}...")

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",  # NBA
    )
    time.sleep(SLEEP_BETWEEN_CALLS)

    df = finder.get_data_frames()[0]

    # Nur Regular Season + Playoffs
    df = df[df["SEASON_ID"].str.startswith(("2", "4"))]

    return df


def fetch_team_gamelogs(team_id: int, season: str = "2024-25") -> pd.DataFrame:
    """Detailliertes Gamelog für ein Team."""
    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable="Regular Season",
    )
    time.sleep(SLEEP_BETWEEN_CALLS)
    return logs.get_data_frames()[0]


def fetch_upcoming_games() -> pd.DataFrame:
    """
    Holt die heutigen + morgigen Spiele via ScoreboardV2.
    Gibt Teams + Game-IDs zurück.
    """
    from datetime import date, timedelta

    today = date.today().strftime("%m/%d/%Y")
    tomorrow = (date.today() + timedelta(days=1)).strftime("%m/%d/%Y")

    upcoming = []
    for day in [today, tomorrow]:
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=day)
            time.sleep(SLEEP_BETWEEN_CALLS)

            games = sb.get_data_frames()[0]
            if not games.empty:
                games["GAME_DATE"] = day
                upcoming.append(games)
        except Exception as e:
            print(f"  [Warn] Kein Scoreboard für {day}: {e}")

    if not upcoming:
        return pd.DataFrame()
    return pd.concat(upcoming, ignore_index=True)


def build_historical_dataset(seasons: list = None) -> pd.DataFrame:
    """
    Baut den vollständigen Trainingsdatensatz.
    Speichert als CSV in data/.
    """
    if seasons is None:
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]

    all_games = []
    for season in seasons:
        df = fetch_season_games(season)
        all_games.append(df)
        print(f"  → {len(df)} Einträge für {season}")

    combined = pd.concat(all_games, ignore_index=True)
    combined.to_csv(DATA_DIR / "raw_games.csv", index=False)
    print(f"[DataCollector] {len(combined)} Spieldaten gespeichert.")
    return combined


def fetch_standings(season: str = "2024-25") -> pd.DataFrame:
    """Aktuelle Tabelle + Win%."""
    standings = leaguestandings.LeagueStandings(season=season)
    time.sleep(SLEEP_BETWEEN_CALLS)
    return standings.get_data_frames()[0]
