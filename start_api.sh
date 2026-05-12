#!/bin/bash
# Start NBA Predictions API Server
# Run this to serve predictions to your iPhone app

PROJECT_DIR="/Users/timsuskov/Desktop/nbapredictions"
VENV="$PROJECT_DIR/venv/bin/activate"

source "$VENV"
cd "$PROJECT_DIR"

echo ""
echo "=========================================="
echo "  NBA PREDICTIONS API SERVER"
echo "=========================================="
echo ""
echo "📱 Your iPhone app should fetch from:"
echo "   http://YOUR_IP:8000"
echo ""
echo "Available endpoints:"
echo "   GET /health — Health check"
echo "   GET /api/predictions/formatted — For iPhone app"
echo "   GET /api/stats — Model info"
echo ""
echo "Find your IP with: ipconfig getifaddr en0"
echo "=========================================="
echo ""

python3 "$PROJECT_DIR/api.py" --dev
