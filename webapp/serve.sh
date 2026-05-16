#!/bin/bash
# Start the CatFlow backend AND the cloudflared tunnel together — one command
# in one terminal. Ctrl-C stops both.
#
#   ./serve.sh                 # backend on $PORT + quick tunnel + push api_url.txt
#   ./serve.sh --no-push       # same, but do not commit/push the tunnel URL
#
# This just chains run.sh (backend) and tunnel.sh (tunnel); use those directly
# if you want the two processes in separate windows.
#
# Env overrides: PORT (default 8200), CATFLOW_DEVICE, CATFLOW_PY.
set -e
cd "$(dirname "$0")"

export PORT="${PORT:-8200}"
SERVER_LOG=/tmp/catflow_server.log

# Refuse to start if the port is already taken — a leftover server would keep
# serving stale data (see HANDOFF gotchas). Stop it first.
if curl -fsS "http://localhost:$PORT/api/adsorbates" >/dev/null 2>&1; then
    holder=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -oE 'pid=[0-9]+' | head -1)
    echo "[serve] port $PORT is already in use (${holder:-unknown})." >&2
    echo "[serve] stop that process first, then re-run ./serve.sh" >&2
    exit 1
fi

echo "[serve] starting backend on :$PORT  (logs -> $SERVER_LOG) ..."
./run.sh > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    echo
    echo "[serve] stopped backend (pid $SERVER_PID) and tunnel."
}
trap cleanup EXIT

# Wait for the backend to finish loading the models + UMA (~40 s).
printf '[serve] loading models + UMA '
for _ in $(seq 1 150); do
    if curl -fsS "http://localhost:$PORT/api/adsorbates" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo
        echo "[serve] ERROR: backend exited during startup — log tail:" >&2
        tail -20 "$SERVER_LOG" >&2
        exit 1
    fi
    printf '.'
    sleep 2
done
if ! curl -fsS "http://localhost:$PORT/api/adsorbates" >/dev/null 2>&1; then
    echo
    echo "[serve] ERROR: backend not ready after 300 s — log tail:" >&2
    tail -20 "$SERVER_LOG" >&2
    exit 1
fi
echo ' ready.'

# Foreground: the tunnel. Ctrl-C here stops the tunnel, then the EXIT trap
# stops the backend too.
./tunnel.sh "$@"
