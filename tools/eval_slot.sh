#!/usr/bin/env bash
# One race in an isolated sim slot. Usage: eval_slot.sh <slot 1..4> <checkout dir> <label>
# Server: own CARLA instance on port 2000+3*slot, offscreen, own process group (never touches other slots' servers).
# Client: demo_slot.py on that port from <checkout>/competition_code (its own logs/run.csv). Aborts at the first
# collision (kills only this slot). Prints: <label>,<sim_time or empty>,<collisions>,<finished|aborted|server_hang|crashed>
# and keeps <checkout>/logs/<label>.run.csv (+ .client.log).
set -u
SLOT=$1; WT=$(cd "$2" && pwd); LABEL=$3
PARENT=/home/mt-labpc/Documents/roar_racing
PORT=$((2000 + 3 * SLOT))
LOG="$WT/logs/$LABEL"; mkdir -p "$WT/logs"
SRVPATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.venv/' | paste -sd:)
port_open() { (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; }
SRV=""; CPID=""
cleanup() {
  [ -n "$CPID" ] && { kill -TERM -- "-$CPID" 2>/dev/null; sleep 1; kill -KILL -- "-$CPID" 2>/dev/null; }
  [ -n "$SRV" ] && { kill -TERM -- "-$SRV" 2>/dev/null; sleep 2; kill -KILL -- "-$SRV" 2>/dev/null; }
}
trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
if port_open; then echo "$LABEL,,0,port_busy"; exit 2; fi
PATH=$SRVPATH setsid "$PARENT/run_monza.sh" -RenderOffScreen -carla-rpc-port=$PORT -carla-streaming-port=$((PORT + 1)) > "$LOG.server.log" 2>&1 &
SRV=$!
for _ in $(seq 300); do port_open && break; sleep 1; done
port_open || { echo "$LABEL,,0,server_hang"; exit 2; }
sleep 5
rm -f "$WT/logs/run.csv"
setsid bash -c "cd '$PARENT' && source .venv/bin/activate && ROAR_CODE='$WT/competition_code' ROAR_PORT=$PORT ROAR_LOG=1 exec timeout 1800 python -u demo_slot.py" > "$LOG.client.log" 2>&1 &
CPID=$!
status=aborted
while kill -0 "$CPID" 2>/dev/null; do
  if grep -q "major collision" "$LOG.client.log" 2>/dev/null; then kill -TERM -- "-$CPID" 2>/dev/null; break; fi
  sleep 1
done
wait "$CPID" 2>/dev/null
cp "$WT/logs/run.csv" "$LOG.run.csv" 2>/dev/null
sim=$(grep -oP 'Solution finished in \K[0-9.]+' "$LOG.client.log" | tail -1 | xargs -r printf '%.2f')
coll=$(grep -c 'major collision' "$LOG.client.log")
if [ -n "$sim" ]; then status=finished; elif [ "$coll" = 0 ]; then status=crashed; fi
echo "$LABEL,$sim,$coll,$status"
