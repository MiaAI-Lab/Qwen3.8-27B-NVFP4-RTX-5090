#!/usr/bin/env bash
#
# sglang-dflash2/stop.sh — tear down the SGLang server started by ./start.sh
#
#   * kills the PID recorded in sglang.pid (SIGTERM, then SIGKILL after 15s)
#   * sweeps orphaned sglang:: scheduler / detokenizer processes
#   * removes the stale sglang.pid file once the port is confirmed free
#   * is idempotent — safe to run when nothing is running
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8888}"
PID_FILE="$SCRIPT_DIR/sglang.pid"

log() { printf '\033[1;36m[sglang-dflash2 stop]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[sglang-dflash2 stop] ERROR:\033[0m %b\n' "$*" >&2; exit 1; }
up() { curl -s -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null; }   # any HTTP answer, auth or not

MAIN_PID=""
[ -f "$PID_FILE" ] && MAIN_PID="$(cat "$PID_FILE")"

if [ -n "$MAIN_PID" ] && kill -0 "$MAIN_PID" 2>/dev/null; then
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
elif ! up && ! pgrep -f 'sglang::' >/dev/null; then
  log "nothing to stop (no live pid; port $PORT already free)."
  rm -f "$PID_FILE"
  exit 0
fi

sleep 1
if pkill -KILL -f 'sglang::' 2>/dev/null; then
  log "killed orphaned sglang:: process(es)"
fi

up && die "still answering on port $PORT after stop — investigate:\n  ps aux | grep -E 'sglang'"
rm -f "$PID_FILE"
log "stopped. port $PORT is free, sglang.log kept for inspection."
exit 0
