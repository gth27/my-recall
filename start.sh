#!/bin/bash

# 1. Start the Docker Brain (Backend, Database, Frontend)
echo "Starting Recall Engine (Docker)..."
docker compose up -d

# 2. Start the Watcher (Host)
# We check if it's already running to prevent duplicates
if pgrep -f "watcher.py" >/dev/null; then
  echo "Watcher is already running."
else
  echo "Starting Screen Watcher..."
  # Run in background (&), ignore output (nohup), and use the venv python
  nohup ./watcher/venv/bin/python ./watcher/watcher.py >watcher/watcher.log 2>&1 &
fi

echo "✅ System Online!"
echo "   - UI: http://localhost:8501"
echo "   - Logs: tail -f watcher/watcher.log"
