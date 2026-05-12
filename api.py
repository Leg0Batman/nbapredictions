#!/usr/bin/env python3
"""
NBA Predictions API Server
Serves predictions to iPhone app via REST API

Usage:
  python api.py --dev    # Development (localhost:5000)
  python api.py --prod   # Production (0.0.0.0:5000)
"""
import json
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for iPhone app

DATA_DIR = Path("data")
PREDICTIONS_FILE = DATA_DIR / "predictions.json"

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    """Get today's predictions."""
    if not PREDICTIONS_FILE.exists():
        return jsonify({
            "success": False,
            "message": "No predictions available yet",
            "predictions": []
        }), 404
    
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            predictions = json.load(f)
        
        return jsonify({
            "success": True,
            "message": "Predictions loaded successfully",
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error loading predictions: {str(e)}",
            "predictions": []
        }), 500

@app.route("/api/predictions/formatted", methods=["GET"])
def get_formatted_predictions():
    """Get predictions in formatted iPhone-friendly format."""
    if not PREDICTIONS_FILE.exists():
        return jsonify({
            "success": False,
            "games": [],
            "summary": "No predictions available"
        }), 404
    
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            data = json.load(f)
        
        # Format for iPhone display
        games = []
        if isinstance(data, list):
            raw_games = data
        elif isinstance(data, dict) and "predictions" in data:
            raw_games = data["predictions"]
        else:
            raw_games = [data] if data else []
        
        for game in raw_games:
            if isinstance(game, dict):
                formatted_game = {
                    "matchup": game.get("matchup", game.get("game", "Unknown")),
                    "winner": game.get("winner", "Unknown"),
                    "confidence": game.get("confidence", "UNKNOWN"),
                    "home_prob": game.get("home_prob", game.get("prob_home_win", 50)),
                    "away_prob": game.get("away_prob", game.get("prob_away_win", 50)),
                    "confidence_pct": game.get("confidence_pct", 0),
                }
                games.append(formatted_game)
        
        return jsonify({
            "success": True,
            "games": games,
            "count": len(games),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "games": [],
            "error": str(e)
        }), 500

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get model statistics."""
    model_path = Path("models/nba_model.pkl")
    
    if not model_path.exists():
        return jsonify({
            "success": False,
            "message": "Model not found"
        }), 404
    
    try:
        from model import NBAEnsemble
        model = NBAEnsemble()
        model.load(str(model_path))
        
        return jsonify({
            "success": True,
            "cv_accuracy": f"{model.metrics.get('xgb_cv_accuracy', 0):.1%}",
            "roc_auc": f"{model.metrics.get('roc_auc', 0):.3f}",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

if __name__ == "__main__":
    # Determine mode
    dev_mode = "--prod" not in sys.argv
    host = "127.0.0.1" if dev_mode else "0.0.0.0"
    port = 8000  # Changed from 5000 to avoid AirTunes conflict
    
    mode = "Development" if dev_mode else "Production"
    print(f"\n{'='*60}")
    print(f"  NBA PREDICTIONS API — {mode.upper()}")
    print(f"{'='*60}")
    print(f"🚀 Running on http://{host}:{port}")
    print(f"\n📱 iPhone app endpoints:")
    print(f"   GET /health — Health check")
    print(f"   GET /api/predictions — Raw predictions")
    print(f"   GET /api/predictions/formatted — iPhone-friendly format")
    print(f"   GET /api/stats — Model statistics")
    print(f"\n{'='*60}\n")
    
    app.run(host=host, port=port, debug=dev_mode, use_reloader=False)
