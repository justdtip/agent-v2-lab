#!/bin/zsh
# After the first queue (pilot) exits and the lock is free: the runner-path comparison.
cd "/Users/daniel.tipton/Desktop/An app"
R="/Users/daniel.tipton/Desktop/An app/research/records/ARM-A-DIVERGENCE-2026-09-07"
while kill -0 53922 2>/dev/null || [ -f outputs/.model-run.lock ]; do sleep 15; done
echo "$(date '+%H:%M:%S') queue-2: model free; starting the runner-path comparison" >> "$R/queue.log"
.venv/bin/python scripts/cache_runner_paths.py --out "$R/cache" > "$R/cache/runner-paths.log" 2>&1
echo "$(date '+%H:%M:%S') queue-2: runner-path comparison exit $?" >> "$R/queue.log"
