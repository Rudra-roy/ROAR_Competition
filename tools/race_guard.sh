#!/usr/bin/env bash
# Usage: tools/race_guard.sh <label> <runs> [tune.py args...]   (env ROAR_SERVER_ARGS passes through)
# Runs tools/tune.py <label> <runs> ... and aborts the race at its FIRST collision ("major collision" in the client
# log): kills tune.py, eval.sh, the client and the server (one process group), keeps the run log as
# logs/<label>_abort.run.csv and appends "ABORTED ..." to logs/<label>.tune.log. No results.csv row is written for an
# aborted run. Exit 0 = finished, 3 = aborted.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
LABEL=${1:?usage: race_guard.sh <label> <runs> [args]}
CLOG="$REPO/logs/${LABEL}_1.client.log"  # tune.py runs eval.sh 1 <label> per race, so every race writes this file
TLOG="$REPO/logs/${LABEL}.tune.log"
rm -f "$CLOG"
cd "$REPO" || exit 1
setsid "$(dirname "$REPO")/.venv/bin/python" -u tools/tune.py "$@" > "$TLOG" 2>&1 &
PG=$!
while kill -0 "$PG" 2>/dev/null; do
  if grep -q "major collision" "$CLOG" 2>/dev/null; then
    cp "$REPO/logs/run.csv" "$REPO/logs/${LABEL}_abort.run.csv" 2>/dev/null
    kill -TERM -- "-$PG" 2>/dev/null
    sleep 2
    kill -KILL -- "-$PG" 2>/dev/null
    WINEPREFIX=~/tools/pfx/pfx ~/tools/GE-Proton9-27/files/bin/wineserver -k >/dev/null 2>&1
    echo "ABORTED at first collision: $(grep -m1 'major collision' "$CLOG")" >> "$TLOG"
    exit 3
  fi
  sleep 1
done
exit 0
