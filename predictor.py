"""
NBA Game Predictor — Hauptinterface
Autonomous: Daten holen → Features bauen → Vorhersagen ausgeben

Usage:
  python predictor.py train          # Modell trainieren
  python predictor.py predict        # Nächste Spiele vorhersagen
  python predictor.py update         # Neue Daten + Retrain
"""

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from datacollector import (
    build_historical_dataset,
    fetch_season_games,
    fetch_upcoming_games,
    get_all_teams,
)

from features import build_features, get_feature_columns
from model import NBAEnsemble

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Team-ID zu Name Mapping (Cache)
TEAM_MAP = {t["id"]: t["full_name"] for t in get_all_teams()}


def train_pipeline(seasons=None):
    """
    Vollständige Trainings-Pipeline:
    1. Daten sammeln
    2. Features bauen
    3. Modell trainieren
    4. Speichern
    """
    print("=" * 60)
    print("  NBA PREDICTOR — TRAINING PIPELINE")
    print("=" * 60)

    if seasons is None:
        seasons = ["2022-23", "2023-24", "2024-25"]

    # 1. Daten sammeln
    print(f"\n[1/3] Sammle Daten für Saisons: {seasons}")
    raw_df = build_historical_dataset(seasons=seasons)

    # 2. Features bauen
    print("\n[2/3] Feature Engineering...")
    features_df = build_features(raw_df)

    feature_cols = get_feature_columns(features_df)
    X = features_df
    y = features_df["HOME_WIN"]

    print(f"  → {len(X)} Trainingsspiele")
    print(f"  → Home-Win Rate: {y.mean():.1%} (Home-Vorteil)")

    # 3. Training
    print("\n[3/3] Modell Training...")
    model = NBAEnsemble()
    metrics = model.train(X, y, feature_cols)
    model.save()

    print("\n" + "=" * 60)
    print("  TRAINING ABGESCHLOSSEN")
    print(f"  CV Accuracy: {metrics['xgb_cv_accuracy']:.1%}")
    print(f"  ROC AUC:     {metrics['roc_auc']:.3f}")
    print("=" * 60)

    return model, features_df


def build_prediction_features(
    home_team_id: int, away_team_id: int, features_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Baut Feature-Vektor für ein kommendes Spiel aus historischen Daten.
    Nutzt die letzten verfügbaren Rolling-Stats der Teams.
    """
    feature_cols = get_feature_columns(features_df)

    # Letztes bekanntes Sample für Home Team
    home_recent = features_df[features_df["HOME_TEAM_ID"] == home_team_id].tail(1)
    away_recent = features_df[features_df["AWAY_TEAM_ID"] == away_team_id].tail(1)

    if home_recent.empty:
        home_recent = features_df[features_df["AWAY_TEAM_ID"] == home_team_id].tail(1)
    if away_recent.empty:
        away_recent = features_df[features_df["HOME_TEAM_ID"] == away_team_id].tail(1)

    if home_recent.empty or away_recent.empty:
        return None

    # Kombiniere zu einem Feature-Vektor
    pred_row = {}
    for col in feature_cols:
        if col.startswith("HOME_"):
            # Nimm Home-Stats aus home_recent
            if col in home_recent.columns:
                pred_row[col] = home_recent[col].values[0]
            else:
                away_col = "AWAY_" + col[5:]
                if away_col in home_recent.columns:
                    pred_row[col] = home_recent[away_col].values[0]
                else:
                    pred_row[col] = 0
        elif col.startswith("AWAY_"):
            if col in away_recent.columns:
                pred_row[col] = away_recent[col].values[0]
            else:
                home_col = "HOME_" + col[5:]
                if home_col in away_recent.columns:
                    pred_row[col] = away_recent[home_col].values[0]
                else:
                    pred_row[col] = 0
        elif col.startswith("DIFF_"):
            base = col[5:]
            h_col = f"HOME_{base}"
            a_col = f"AWAY_{base}"
            h_val = pred_row.get(h_col, 0)
            a_val = pred_row.get(a_col, 0)
            pred_row[col] = h_val - a_val
        else:
            pred_row[col] = 0

    return pd.DataFrame([pred_row])


def predict_upcoming(model: NBAEnsemble = None, features_df: pd.DataFrame = None):
    """
    Holt kommende Spiele und macht Vorhersagen.
    """
    print("\n" + "=" * 60)
    print("  NBA PREDICTOR — UPCOMING GAMES")
    print("=" * 60)

    # Modell laden falls nicht übergeben
    if model is None:
        model_path = MODEL_DIR / "nba_model.pkl"
        if not model_path.exists():
            print("[ERROR] Kein trainiertes Modell gefunden!")
            print("  → Starte erst: python predictor.py train")
            return
        model = NBAEnsemble()
        model.load(str(model_path))

    # Features laden falls nicht übergeben
    if features_df is None:
        feat_path = DATA_DIR / "features.csv"
        if feat_path.exists():
            features_df = pd.read_csv(feat_path)
        else:
            print("[ERROR] Keine Feature-Daten gefunden. Erst trainieren!")
            return

    # Kommende Spiele holen
    print("\n[NBA API] Hole heutige + morgige Spiele...")
    upcoming = fetch_upcoming_games()

    if upcoming is None or upcoming.empty:
        print("  → Keine Spiele gefunden (Off-Season oder API-Problem)")
        print("  → Zeige Demo-Vorhersage für bekannte Teams...")
        _demo_predictions(model, features_df)
        return

    print(f"  → {len(upcoming)} Spiele gefunden\n")

    results = []
    for _, game in upcoming.iterrows():
        home_id = game.get("HOME_TEAM_ID") or game.get("TEAM_ID")
        away_id = game.get("VISITOR_TEAM_ID") or game.get("TEAM_ID")
        game_date = game.get("GAME_DATE", "Unbekannt")

        if pd.isna(home_id) or pd.isna(away_id):
            continue

        home_name = TEAM_MAP.get(int(home_id), f"Team {home_id}")
        away_name = TEAM_MAP.get(int(away_id), f"Team {away_id}")

        # Features bauen
        X_pred = build_prediction_features(int(home_id), int(away_id), features_df)

        if X_pred is None:
            print(f"  ⚠ Keine Daten für: {away_name} @ {home_name}")
            continue

        # Vorhersage
        pred = model.predict_with_confidence(X_pred, home_name, away_name)
        pred["game_date"] = game_date
        results.append(pred)

    _print_predictions(results)
    return results


def _print_predictions(results: list):
    """Formatierter Output der Vorhersagen."""
    if not results:
        print("  Keine Vorhersagen möglich.")
        return

    print(f"\n{'─' * 60}")
    print(f"  {'SPIEL':<35} {'TIPP':<20} {'CONFIDENCE'}")
    print(f"{'─' * 60}")

    for r in results:
        matchup = f"{r['away_team']} @ {r['home_team']}"
        winner = r["predicted_winner"]
        conf = r["confidence"]
        prob = r["confidence_pct"]

        print(f"\n  📅 {r.get('game_date', '')}")
        print(f"  🏀 {matchup}")
        print(f"  🏆 Sieger: {winner} ({prob}%)")
        print(f"  📈 Konfidenz: {conf}")
        print(
            f"     {r['home_team']}: {r['prob_home_win']}%  |  {r['away_team']}: {r['prob_away_win']}%"
        )

    print(f"\n{'─' * 60}")

    # Speichere Vorhersagen
    pred_path = DATA_DIR / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✅ Vorhersagen gespeichert: {pred_path}")


def _demo_predictions(model: NBAEnsemble, features_df: pd.DataFrame):
    """Demo mit bekannten Playoff-Teams falls keine Live-Spiele."""
    # Boston Celtics vs Miami Heat als Beispiel
    celtics_id = 1610612738
    heat_id = 1610612748
    lakers_id = 1610612747
    warriors_id = 1610612744

    demos = [
        (celtics_id, heat_id, "Playoffs Demo"),
        (lakers_id, warriors_id, "Regular Season Demo"),
    ]

    results = []
    for home_id, away_id, label in demos:
        X_pred = build_prediction_features(home_id, away_id, features_df)
        if X_pred is not None:
            home_name = TEAM_MAP.get(home_id, "Home")
            away_name = TEAM_MAP.get(away_id, "Away")
            pred = model.predict_with_confidence(X_pred, home_name, away_name)
            pred["game_date"] = label
            results.append(pred)

    _print_predictions(results)


def update_and_retrain():
    """
    Update: Neueste Daten holen + Modell nachtrainieren.
    Sollte wöchentlich laufen.
    """
    print("[Update] Hole neueste Daten...")
    # Nur aktuelle Saison neu laden
    raw_df = fetch_season_games("2024-25")
    raw_df.to_csv(DATA_DIR / "raw_current_season.csv", index=False)

    # Kombiniere mit historischen Daten
    hist_path = DATA_DIR / "raw_games.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        combined = pd.concat([hist_df, raw_df]).drop_duplicates("GAME_ID")
    else:
        combined = raw_df

    combined.to_csv(DATA_DIR / "raw_games.csv", index=False)

    # Feature Engineering + Training
    features_df = build_features(combined)
    feature_cols = get_feature_columns(features_df)

    model = NBAEnsemble()
    model.train(features_df, features_df["HOME_WIN"], feature_cols)
    model.save()

    print("[Update] Fertig! Modell aktualisiert.")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"

    if cmd == "train":
        model, features_df = train_pipeline()
        # Nach Training direkt Vorhersagen
        predict_upcoming(model, features_df)

    elif cmd == "predict":
        predict_upcoming()

    elif cmd == "update":
        update_and_retrain()
        predict_upcoming()

    else:
        print("Usage: python predictor.py [train|predict|update]")
