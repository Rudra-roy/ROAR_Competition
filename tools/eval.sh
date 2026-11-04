#!/usr/bin/env bash
# Usage: tools/eval.sh <runs> <label>
# Appends to results.csv: git_sha,label,run,sim_time_s,collisions,status
# ROAR_SERVER_ARGS (e.g. -RenderOffScreen) is passed to run_monza.sh.
set -u
RUNS=${1:?usage: eval.sh <runs> <label>}; LABEL=${2:?usage: eval.sh <runs> <label>}
REPO=$(cd "$(dirname "$0")/.." && pwd); PARENT=$(dirname "$REPO")
# proton needs system python >=3.9: drop any active venv from PATH for the server
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.venv/' | paste -sd:)
mkdir -p "$REPO/logs"
CSV="$REPO/results.csv"
[ -f "$CSV" ] || echo "git_sha,label,run,sim_time_s,collisions,status" > "$CSV"
SHA=$(git -C "$REPO" rev-parse --short HEAD)
export WINEPREFIX=~/tools/pfx/pfx
killsrv() { ~/tools/GE-Proton9-27/files/bin/wineserver -k >/dev/null 2>&1; sleep 3; }
port_open() { (exec 3<>/dev/tcp/127.0.0.1/2000) 2>/dev/null; }

for i in $(seq 1 "$RUNS"); do
  killsrv
  T0=$(date +%s)
  LOG="$REPO/logs/${LABEL}_${i}"
  "$PARENT/run_monza.sh" ${ROAR_SERVER_ARGS:-} > "$LOG.server.log" 2>&1 &
  status=server_hang
  for _ in $(seq 300); do port_open && { status=up; break; }; sleep 1; done
  sim=""; coll=0
  if [ "$status" = up ]; then
    sleep 5
    # venv only for the client: its python3.8 breaks proton (needs >=3.9)
    (cd "$PARENT" && source .venv/bin/activate && timeout --foreground 1800 python -u demo.py) > "$LOG.client.log" 2>&1
    sim=$(grep -oP 'Solution finished in \K[0-9.]+' "$LOG.client.log" | tail -1 | xargs -r printf '%.2f')
    coll=$(grep -c 'major collision' "$LOG.client.log")
    if [ -n "$sim" ]; then status=finished
    elif grep -q 'Solution failed to finish' "$LOG.client.log"; then status=failed
    else status=crashed_client; fi
  fi
  killsrv
  echo "$SHA,$LABEL,$i,$sim,$coll,$status" >> "$CSV"
  echo "run $i: $status sim=${sim:-NA} collisions=$coll wall=$(( $(date +%s) - T0 ))s"
done
