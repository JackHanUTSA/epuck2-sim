#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: launch_webots_headless.sh WORLD [WEBOTS_ARGS...]" >&2
  exit 2
fi

WORLD_PATH="$1"
shift

WEBOTS_HOME="${WEBOTS_HOME:-/snap/webots/current/usr/share/webots}"
WEBOTS_LAUNCHER="$WEBOTS_HOME/webots"
if [[ ! -x "$WEBOTS_LAUNCHER" ]]; then
  echo "webots launcher not found under $WEBOTS_HOME" >&2
  exit 1
fi

find_free_display() {
  for n in $(seq 90 140); do
    if [[ ! -S "/tmp/.X11-unix/X$n" ]]; then
      echo ":$n"
      return 0
    fi
  done
  return 1
}

DISPLAY_NUM="$(find_free_display)"
XVFB_LOG="${WEBOTS_TMPDIR:-/tmp}/dreamer_xvfb_${DISPLAY_NUM#:}.log"
Xvfb "$DISPLAY_NUM" -screen 0 1280x720x24 -ac +extension GLX +render -noreset >"$XVFB_LOG" 2>&1 &
XVFB_PID=$!

cleanup() {
  if [[ -n "${WEBOTS_PID:-}" ]]; then
    kill -TERM "$WEBOTS_PID" 2>/dev/null || true
    wait "$WEBOTS_PID" 2>/dev/null || true
  fi
  kill -TERM "$XVFB_PID" 2>/dev/null || true
  wait "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

export DISPLAY="$DISPLAY_NUM"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-runner}"
mkdir -p "$XDG_RUNTIME_DIR"

"$WEBOTS_LAUNCHER" "$WORLD_PATH" "$@" &
WEBOTS_PID=$!
wait "$WEBOTS_PID"
STATUS=$?
WEBOTS_PID=""
exit "$STATUS"
