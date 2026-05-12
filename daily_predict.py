#!/usr/bin/env python3
"""
Daily Automated Predictions
Runs every day to predict upcoming NBA games.
"""
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from predictor import predict_upcoming
from model import NBAEnsemble

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  DAILY PREDICTIONS — {timestamp}")
    print(f"{'='*60}\n")
    
    # Load trained model
    model_path = Path("models/nba_model.pkl")
    if not model_path.exists():
        print("[ERROR] Trainiertes Modell nicht gefunden!")
        print("  → Starte erst: python predictor.py train")
        sys.exit(1)
    
    model = NBAEnsemble()
    model.load(str(model_path))
    
    # Get predictions for upcoming games
    predict_upcoming(model)
    
    print(f"\n✅ Tägliche Vorhersage abgeschlossen um {timestamp}\n")

if __name__ == "__main__":
    main()
