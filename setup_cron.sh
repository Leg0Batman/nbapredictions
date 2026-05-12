#!/bin/bash
# Setup daily cron job for NBA predictions
# Run this once: bash setup_cron.sh

PROJECT_DIR="/Users/timsuskov/Desktop/nbapredictions"
VENV="$PROJECT_DIR/venv/bin/activate"
SCRIPT="$PROJECT_DIR/daily_predict.py"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/daily_predictions.log"

# Create logs directory
mkdir -p "$LOG_DIR"

# Create cron job command
CRON_CMD="source $VENV && cd $PROJECT_DIR && python3 $SCRIPT >> $LOG_FILE 2>&1"

# Schedule for 10:00 AM every day
CRON_SCHEDULE="0 10 * * * $CRON_CMD"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "daily_predict.py"; then
    echo "✅ Cron job already exists!"
    echo ""
    echo "Current cron jobs:"
    crontab -l | grep -E "daily_predict|nba"
else
    # Add new cron job
    (crontab -l 2>/dev/null; echo "$CRON_SCHEDULE") | crontab -
    echo "✅ Cron job installed!"
    echo ""
    echo "📅 Scheduled: 10:00 AM every day"
    echo "📂 Project: $PROJECT_DIR"
    echo "📊 Logs: $LOG_FILE"
    echo ""
    echo "View logs with: tail -f $LOG_FILE"
fi
