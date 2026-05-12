# 📱 iPhone App Setup Guide

## How It Works

1. **Backend** (`api.py`) — Runs on your Mac, serves predictions via HTTP
2. **iPhone App** (SwiftUI) — Connects to the API and displays predictions beautifully

## Step 1: Start the API Server

Run this on your Mac (in the project directory):

```bash
bash start_api.sh
```

Or manually:
```bash
source venv/bin/activate
python3 api.py
```

You should see:
```
🚀 Running on http://127.0.0.1:8000
📱 Your iPhone app should fetch from:
   http://YOUR_IP:8000
```

## Step 2: Find Your Computer's IP Address

```bash
ipconfig getifaddr en0
```

This returns something like: `192.168.1.100`

## Step 3: Create iOS App in Xcode

1. Open Xcode
2. Create new **App** project (SwiftUI)
3. Copy the entire content from `ios_app_example.swift`
4. Paste into `ContentView.swift` (replace everything)
5. **Update the IP address:**
   - Find this line: `private let apiURL = "http://192.168.1.100:8000/api/predictions/formatted"`
   - Replace `192.168.1.100` with YOUR IP from step 2
6. Run on your iPhone or simulator

## API Endpoints for Your App

**Get predictions (formatted for display):**
```
GET http://YOUR_IP:8000/api/predictions/formatted

Response:
{
  "success": true,
  "games": [
    {
      "matchup": "Minnesota Timberwolves @ San Antonio Spurs",
      "winner": "Minnesota Timberwolves",
      "confidence": "🤷 NIEDRIG (Toss-Up)",
      "home_prob": 47.7,
      "away_prob": 52.3,
      "confidence_pct": 52.3
    }
  ],
  "count": 1
}
```

**Health check:**
```
GET http://YOUR_IP:8000/health
```

## Troubleshooting

**"Cannot connect to server"**
- Check if API is running: `bash start_api.sh`
- Verify IP address with: `ipconfig getifaddr en0`
- Make sure iPhone/Mac are on same WiFi network

**"Port 8000 already in use"**
- Change port in `api.py` (line ~80)
- Change URL in iOS app to match

**Predictions not showing**
- Run: `python3 predictor.py train` first (trains the model)
- Check: `python3 predictor.py predict` (generates predictions)

## Making Predictions Automatic

Predictions are generated automatically every day at 10:00 AM (via cron job).

Your iPhone app will fetch the latest predictions when you tap refresh! ✅

## Customize the iOS App

Edit `ios_app_example.swift` to:
- Change colors/theme
- Add more game details
- Add notification when games end
- Add betting odds display
- Store favorites
