#!/bin/bash

echo "Stopping Recall System..."

# 1. Kill the Python Watcher
pkill -f "watcher.py"
echo "   - Watcher stopped."

# 2. Stop Docker Containers
docker compose stop
echo "   - Docker engine stopped."

echo "Offline."
