#!/usr/bin/env bash
#
# stop.sh — tear down the vLLM server started by start.sh
#
#   * kills the PID recorded in vllm.pid (SIGTERM, then SIGKILL after 15s)
#   * sweeps any orphaned VLLM::EngineCore / API processes
#   * removes the stale vllm.pid file once the port is confirmed free
#   * is idempotent — safe to run when nothing is running
#
# NOTE: the recorded PID is stopped unconditionally; $PORT is only used to
# confirm the server is really down afterwards. Don't set PORT unless you
# also set it in start.sh (default: 8888).
#
# Usage: ./stop.sh
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8888}"
PID_FILE="$SCRIPT_DIR/vllm.pid"

log() { printf '\033[1;36m[stop.sh]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[stop.sh] ERROR:\033[0m %b\n' "$*" >&2; exit 1; }

# --- 0. what are we stopping? --------------------------------------------------
MAIN_PID=""
[ -f "$PID_FILE" ] && MAIN_PID="$(cat "$PID_FILE")"

if [ -z "$MAIN_PID" ] || ! kill -0 "$MAIN_PID" 2>/dev/null; then
  # no live recorded pid
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "a server answers on port $PORT but has no recorded PID — sweeping its processes"
  else
    log "nothing to stop (no live pid; port $PORT already free)."
    rm -f "$PID_FILE"
    exit 0
  fi
else
  log "stopping recorded server pid $MAIN_PID …"
  kill -TERM "$MAIN_PID" 2>/dev/null || true
  for _ in $(seq 1 15); do
    kill -0 "$MAIN_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$MAIN_PID" 2>/dev/null; then
    log "still alive after 15s — sending SIGKILL"
    kill -KILL "$MAIN_PID" 2>/dev/null || true
  fi
fi

# --- 1. sweep orphaned engine/API processes -------------------------------------
sleep 1
if pkill -f 'VLLM::EngineCore' 2>/dev/null; then
  log "killed orphaned VLLM::EngineCore process(es)"
else
  log "no VLLM::EngineCore processes found"
fi

# --- 2. confirm the port is actually free ---------------------------------------
if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  die "still answering on port $PORT after stop — investigate:\n  ps aux | grep -E 'vllm|VLLM::EngineCore'"
fi

# --- 3. cleanup ------------------------------------------------------------------
rm -f "$PID_FILE"
log "stopped. port $PORT is free, vllm.log kept for inspection."
exit 0
