#!/bin/zsh
# Runs after the fixed-history job releases the model: the cache-split isolation, then the live-lens pilot.
# Each step logs on its own and reports its own status; nothing is chained behind a failing step.
cd "/Users/daniel.tipton/Desktop/An app"
R="/Users/daniel.tipton/Desktop/An app/research/records/ARM-A-DIVERGENCE-2026-09-07"
L="/Users/daniel.tipton/Desktop/An app/research/records/LIVE-LENS-PILOT-2026-09-07"
while kill -0 46320 2>/dev/null || [ -f outputs/.model-run.lock ]; do sleep 15; done
echo "$(date '+%H:%M:%S') queue: model free; starting the cache-split isolation" >> "$R/queue.log"
.venv/bin/python scripts/cache_split_diagnostic.py --out "$R/cache" --limit 8 > "$R/cache/run.log" 2>&1
echo "$(date '+%H:%M:%S') queue: cache isolation exit $? " >> "$R/queue.log"
while [ -f outputs/.model-run.lock ]; do sleep 5; done
echo "$(date '+%H:%M:%S') queue: starting the live-lens pilot" >> "$R/queue.log"
.venv/bin/python scripts/live_lens_pilot.py --out "$L" > "$L/run.log" 2>&1
echo "$(date '+%H:%M:%S') queue: pilot exit $?" >> "$R/queue.log"
