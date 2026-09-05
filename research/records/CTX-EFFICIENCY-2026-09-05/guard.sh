#!/bin/sh
# R46(iv,v): trigger is the kernel pressure level, critical on two consecutive samples.
# Swap free is logged as confirmation and never fires. Target identified by pid only.
# PRESSURE_CMD / INTERVAL exist so the guard can be tested against a dummy (R38).
PID="$1"; MAXMIN="${2:-90}"
INTERVAL="${INTERVAL:-30}"
PRESSURE_CMD="${PRESSURE_CMD:-sysctl -n kern.memorystatus_vm_pressure_level}"
i=0; pp=0
iters="${ITERS:-$(( MAXMIN * 60 / INTERVAL ))}"
while [ $i -lt $iters ]; do
  i=$((i+1))
  kill -0 "$PID" 2>/dev/null || { echo "guard: target pid $PID exited on its own"; exit 0; }
  pl=$($PRESSURE_CMD 2>/dev/null)
  sf=$(sysctl -n vm.swapusage | awk '{for(j=1;j<=NF;j++) if($j=="free"){gsub("M","",$(j+2)); print int($(j+2)); exit}}')
  crit=0; [ "$pl" = "4" ] && crit=1
  if [ "$crit" = "1" ] && [ "$pp" = "1" ]; then
    echo "guard: KILL pid $PID - pressure critical on two consecutive samples; swap free ${sf}M (confirmation only)"
    kill "$PID" 2>/dev/null; sleep 2
    if kill -0 "$PID" 2>/dev/null; then echo "guard: SIGTERM did not take, escalating to SIGKILL"; kill -9 "$PID" 2>/dev/null; sleep 2; fi
    if kill -0 "$PID" 2>/dev/null; then echo "guard: STILL ALIVE after SIGKILL - manual action needed"; else echo "guard: termination verified"; fi
    exit 0
  fi
  [ "$crit" = "1" ] && echo "guard: pressure critical on one sample, holding for a second; swap free ${sf}M"
  [ $((i % 20)) -eq 1 ] && echo "guard: alive t+$(( (i-1) * INTERVAL ))s pressure=${pl} swap_free=${sf}M"
  pp=$crit
  sleep "$INTERVAL"
done
echo "guard: lifetime expired with target still alive"
