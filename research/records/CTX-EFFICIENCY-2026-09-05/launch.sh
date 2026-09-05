#!/bin/sh
# RECORD, NOT A LAUNCHER (2026-09-05): see RECORD.md. Refuses unless --i-am-a-record.
case " $* " in *" --i-am-a-record "*) ;; *) echo "refusing to run: this is a record of the 2026-09-05 measurements, not a launcher (R47, issue 83)" >&2; exit 1;; esac
# Waits for the box to clear by the Deputy's library test (name-agnostic), then runs
# ctxmax2.py under guard.sh, waits on it, and records the reaped exit status (R46 vi).
PY="/Users/daniel.tipton/Desktop/An app/.venv/bin/python"
LOG=ctxmax2.log
mlx_procs() {
  for p in $(pgrep -x Python python3 python 2>/dev/null); do
    lsof -p "$p" 2>/dev/null | grep -qi libmlx && echo "$p"
  done
}
i=0
while [ $i -lt 240 ]; do
  i=$((i+1))
  others=$(mlx_procs)
  [ -z "$others" ] && break
  [ $((i % 20)) -eq 1 ] && echo "launch: waiting, model process(es) alive: $(echo $others | tr '\n' ' ')"
  sleep 15
done
if [ -n "$(mlx_procs)" ]; then echo "launch: gave up waiting, box never cleared"; exit 1; fi
echo "launch: box clear by libmlx test, starting run"
: > "$LOG"
"$PY" ctxmax2.py >> "$LOG" 2>&1 &
RUN=$!
echo "launch: run pid $RUN"
sh guard.sh "$RUN" 120 >> guard2.log 2>&1 &
wait $RUN
ST=$?
echo "{\"event\": \"exit\", \"pid\": $RUN, \"status\": $ST, \"signalled\": $([ $ST -gt 128 ] && echo true || echo false)}" | tee -a "$LOG"
echo "launch: done, exit status $ST"
