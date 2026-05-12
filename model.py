"""
NBA Prediction Model
XGBoost + LightGBM Ensemble für Game-Winner Vorhersage
Zielt auf ~67% Accuracy (menschliche Expert-Level)
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[Model] LightGBM nicht verfügbar, nutze nur XGBoost")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


class NBAEnsemble:
    """
    Ensemble aus XGBoost (+ optional LightGBM).
    Kalibrated Probabilities für Confidence-Score.
    """
    
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.calibrated = None
        self.metrics = {}
    
    def _build_xgb(self):
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    
    def _build_lgb(self):
        if not HAS_LGB:
            return None
        return lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series, feature_cols: list):
        """Training mit Time-Series Cross Validation."""
        self.feature_cols = feature_cols
        X_feat = X[feature_cols].fillna(0)
        
        print(f"[Model] Training mit {len(X_feat)} Samples, {len(feature_cols)} Features...")
        
        # TimeSeriesSplit: Wichtig! Kein Data Leakage
        tscv = TimeSeriesSplit(n_splits=5)
        
        # XGBoost
        self.xgb_model = self._build_xgb()
        xgb_scores = cross_val_score(
            self.xgb_model, X_feat, y,
            cv=tscv, scoring="accuracy", n_jobs=-1
        )
        print(f"  XGBoost CV Accuracy: {xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}")
        
        # Final Training auf allen Daten
        self.xgb_model.fit(X_feat, y)
        
        # LightGBM optional
        lgb_scores = None
        if HAS_LGB:
            self.lgb_model = self._build_lgb()
            lgb_scores = cross_val_score(
                self.lgb_model, X_feat, y,
                cv=tscv, scoring="accuracy", n_jobs=-1
            )
            print(f"  LightGBM CV Accuracy: {lgb_scores.mean():.3f} ± {lgb_scores.std():.3f}")
            self.lgb_model.fit(X_feat, y)
        
        # Kalibrierung der Wahrscheinlichkeiten (Platt Scaling)
        print("[Model] Kalibriere Probabilities...")
        self.calibrated = CalibratedClassifierCV(self._build_xgb(), cv=5, method="sigmoid")
        self.calibrated.fit(X_feat, y)
        
        # Finale Metriken
        y_pred = self.predict_winner(X)
        y_prob = self.predict_proba(X)
        
        self.metrics = {
            "xgb_cv_accuracy": float(xgb_scores.mean()),
            "xgb_cv_std": float(xgb_scores.std()),
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "brier_score": float(brier_score_loss(y, y_prob)),
            "roc_auc": float(roc_auc_score(y, y_prob)),
        }
        if lgb_scores is not None:
            self.metrics["lgb_cv_accuracy"] = float(lgb_scores.mean())
        
        print(f"[Model] Training abgeschlossen!")
        print(f"  → CV Accuracy: {self.metrics['xgb_cv_accuracy']:.1%}")
        print(f"  → ROC AUC: {self.metrics['roc_auc']:.3f}")
        
        self._print_top_features()
        return self.metrics
    
    def predict_winner(self, X: pd.DataFrame) -> np.ndarray:
        """0 = Away gewinnt, 1 = Home gewinnt."""
        X_feat = X[self.feature_cols].fillna(0)
        
        xgb_pred = self.xgb_model.predict_proba(X_feat)[:, 1]
        
        if self.lgb_model is not None:
            lgb_pred = self.lgb_model.predict_proba(X_feat)[:, 1]
            ensemble_prob = (xgb_pred * 0.55) + (lgb_pred * 0.45)
        else:
            ensemble_prob = xgb_pred
        
        return (ensemble_prob > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Kalibrierte Wahrscheinlichkeit dass Home gewinnt."""
        X_feat = X[self.feature_cols].fillna(0)
        return self.calibrated.predict_proba(X_feat)[:, 1]
    
    def predict_with_confidence(self, X: pd.DataFrame, home_name: str, away_name: str) -> dict:
        """
        Vorhersage + Confidence + Erklärung.
        Returns dict mit allen relevanten Infos.
        """
        prob_home = float(self.predict_proba(X)[0])
        prob_away = 1.0 - prob_home
        
        winner_name = home_name if prob_home > 0.5 else away_name
        winner_prob = max(prob_home, prob_away)
        
        # Confidence Level
        if winner_prob >= 0.75:
            confidence = "🔥 SEHR HOCH"
        elif winner_prob >= 0.65:
            confidence = "✅ HOCH"
        elif winner_prob >= 0.58:
            confidence = "📊 MITTEL"
        else:
            confidence = "🤷 NIEDRIG (Toss-Up)"
        
        return {
            "predicted_winner": winner_name,
            "home_team": home_name,
            "away_team": away_name,
            "prob_home_win": round(prob_home * 100, 1),
            "prob_away_win": round(prob_away * 100, 1),
            "confidence": confidence,
            "confidence_pct": round(winner_prob * 100, 1),
        }
    
    def _print_top_features(self, top_n: int = 15):
        """Top Feature Importances ausgeben."""
        importances = self.xgb_model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        print(f"\n[Model] Top {top_n} Features:")
        for _, row in feat_imp.head(top_n).iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"  {row['feature']:<40} {bar} {row['importance']:.4f}")
    
    def save(self, path: str = "models/nba_model.pkl"):
        """Speichert Modell + Metadaten."""
        joblib.dump({
            "xgb_model": self.xgb_model,
            "lgb_model": self.lgb_model,
            "calibrated": self.calibrated,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "metrics": self.metrics,
        }, path)
        
        with open("models/metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"[Model] Gespeichert: {path}")
    
    def load(self, path: str = "models/nba_model.pkl"):
        """Lädt gespeichertes Modell."""
        data = joblib.load(path)
        self.xgb_model = data["xgb_model"]
        self.lgb_model = data["lgb_model"]
        self.calibrated = data["calibrated"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.metrics = data["metrics"]
        print(f"[Model] Geladen: {path}")
        print(f"  → CV Accuracy: {self.metrics.get('xgb_cv_accuracy', 'N/A'):.1%}")
        return self