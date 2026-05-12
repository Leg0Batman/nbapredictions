# 🏀 NBA Game Predictor

Autonomes ML-System das NBA-Spielsieger vorhersagt — Regular Season & Playoffs.

## Architektur

```
nba_api  ──→  data_collector.py  ──→  features.py  ──→  model.py  ──→  predictor.py
              (Daten sammeln)        (Feature Eng.)     (XGBoost+LGB)    (Interface)
```

## Modell-Entscheidung: Warum XGBoost und kein Deep Learning?

| Modell         | Accuracy (NBA) | Trainingszeit | Braucht viele Daten | Empfehlung         |
|----------------|---------------|---------------|---------------------|--------------------|
| **XGBoost**    | **~66-68%**   | Sekunden      | Nein (500 Games ok) | ✅ **Beste Wahl**  |
| LightGBM       | ~65-67%       | Sekunden      | Nein                | ✅ Gut als Ensemble|
| LSTM/RNN       | ~62-65%       | Minuten       | Ja (5000+ Games)    | ⚠️ Nur mit viel Data|
| DNN            | ~60-64%       | Minuten       | Ja                  | ❌ Schlechter      |
| CNN            | Nicht geeignet| -             | -                   | ❌ Für Bilder      |
| Lineare Reg.   | ~55-58%       | Sofort        | Nein                | ❌ Zu simpel       |

**Fazit:** Bei tabellarischen Sportdaten gewinnt Gradient Boosting fast immer gegen DNNs.
LSTM lohnt sich als zusätzlicher Input (Formkurve der letzten 10 Spiele).

## Features die das Modell nutzt

- **Rolling Stats (letzte 10 Spiele):** PTS, FG%, 3P%, Rebounds, Assists, Turnovers, +/-
- **Win Streak:** Aktuelle Sieges-/Niederlagenserie  
- **Rest Days:** Tage seit letztem Spiel (wichtig für B2B Games)
- **Home/Away Split:** Home-Vorteil ist real (~60% Winrate)
- **Head-to-Head:** Historische Bilanz der letzten 10 Begegnungen
- **Differenz-Features:** Home minus Away (zeigt relative Stärke)

## Setup

```bash
# 1. Installation
pip install nba_api xgboost lightgbm scikit-learn pandas numpy joblib

# 2. Modell trainieren (dauert 2-5 Minuten durch NBA API Rate Limiting)
python predictor.py train

# 3. Vorhersagen für heute/morgen
python predictor.py predict

# 4. Wöchentliches Update (neue Daten + Retrain)
python predictor.py update
```

## Output-Beispiel

```
──────────────────────────────────────────────────────────
  SPIEL                               TIPP                 CONFIDENCE
──────────────────────────────────────────────────────────

  📅 05/11/2025
  🏀 Miami Heat @ Boston Celtics
  🏆 Sieger: Boston Celtics (71.3%)
  📈 Konfidenz: ✅ HOCH
     Boston Celtics: 71.3%  |  Miami Heat: 28.7%

  📅 05/11/2025
  🏀 Golden State Warriors @ LA Lakers
  🏆 Sieger: LA Lakers (54.1%)
  📈 Konfidenz: 🤷 NIEDRIG (Toss-Up)
     LA Lakers: 54.1%  |  Golden State Warriors: 45.9%
```

## Reale Accuracy-Erwartung

- **Zufällig:** 50%
- **Lineare Regression:** 55-57%
- **Dieses System:** 64-68%
- **Vegas Oddsmakers:** 67-70%
- **Menschliche Experten:** 65-68%

Du bist mit diesem System auf Expert-Level. Niemand erreicht zuverlässig >70%.

## Automatisierung (Cron Job)

```bash
# Jeden Tag um 10:00 Uhr Vorhersagen aktualisieren
0 10 * * * cd /pfad/zum/projekt && python predictor.py predict >> logs/daily.log

# Wöchentlich Sonntags retrain
0 8 * * 0 cd /pfad/zum/projekt && python predictor.py update >> logs/weekly.log
```

## Datei-Struktur

```
nba_predictor/
├── predictor.py        # Hauptinterface (hier starten)
├── datacollector.py   # NBA API Wrapper
├── features.py         # Feature Engineering
├── model.py            # XGBoost Ensemble
├── README.md
├── data/
│   ├── raw_games.csv   # Rohdaten
│   ├── features.csv    # Feature Matrix
│   └── predictions.json # Letzte Vorhersagen
└── models/
    ├── nba_model.pkl   # Trainiertes Modell
    └── metrics.json    # Model Performance
```