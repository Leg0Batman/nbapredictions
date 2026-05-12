#!/usr/bin/env python3
"""
Custom matchup predictor — Timberwolves vs Spurs
"""
import pandas as pd
from pathlib import Path
from model import NBAEnsemble
from features import get_feature_columns
from predictor import build_prediction_features, TEAM_MAP

MODEL_DIR = Path("models")
DATA_DIR = Path("data")

# Team IDs
HOME_TEAM_ID = 1610612759  # San Antonio Spurs (home)
AWAY_TEAM_ID = 1610612750  # Minnesota Timberwolves (away)

HOME_NAME = "San Antonio Spurs"
AWAY_NAME = "Minnesota Timberwolves"

def predict_matchup(home_id: int, away_id: int, home_name: str, away_name: str):
    """Make prediction for a specific matchup."""
    
    # Load model
    model_path = MODEL_DIR / "nba_model.pkl"
    if not model_path.exists():
        print("[ERROR] Trainiertes Modell nicht gefunden!")
        print("  → Starte erst: python predictor.py train")
        return
    
    model = NBAEnsemble()
    model.load(str(model_path))
    
    # Load features
    feat_path = DATA_DIR / "features.csv"
    if not feat_path.exists():
        print("[ERROR] Feature-Daten nicht gefunden. Erst trainieren!")
        return
    
    features_df = pd.read_csv(feat_path)
    
    print("\n" + "=" * 60)
    print("  NBA PLAYOFF PREDICTION")
    print("=" * 60)
    
    # Build features for this matchup
    X_pred = build_prediction_features(home_id, away_id, features_df)
    
    if X_pred is None:
        print(f"\n  ⚠ Keine Daten für: {away_name} @ {home_name}")
        print("  (Teams spielen möglicherweise nicht genug zusammen)")
        return
    
    # Make prediction
    pred = model.predict_with_confidence(X_pred, home_name, away_name)
    
    print(f"\n  📅 Playoffs 2026")
    print(f"  🏀 {away_name} @ {home_name}")
    print(f"  🏆 Sieger: {pred['predicted_winner']} ({pred['confidence_pct']:.1f}%)")
    print(f"  📈 Konfidenz: {pred['confidence']}")
    print(f"     {home_name}: {pred['prob_home_win']:.1f}%  |  {away_name}: {pred['prob_away_win']:.1f}%")
    print("\n" + "=" * 60)
    
    # Save prediction
    import json
    predictions = {
        "game": f"{away_name} @ {home_name}",
        "winner": pred['predicted_winner'],
        "winner_probability": pred['confidence_pct'],
        "home_probability": pred['prob_home_win'],
        "away_probability": pred['prob_away_win'],
        "confidence": pred['confidence'],
    }
    
    pred_file = DATA_DIR / "last_prediction.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"  ✅ Vorhersage gespeichert: {pred_file}")

if __name__ == "__main__":
    predict_matchup(HOME_TEAM_ID, AWAY_TEAM_ID, HOME_NAME, AWAY_NAME)
