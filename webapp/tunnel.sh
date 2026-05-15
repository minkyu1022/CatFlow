#!/bin/bash
# Expose the local CatFlow server through a cloudflared quick tunnel, write the
# tunnel URL into docs/app/api_url.txt, and push so GitHub Pages serves it.
# One command takes the public demo fully live.
#
#   ./tunnel.sh                  # tunnel localhost:$PORT, update api_url.txt, push
#   ./tunnel.sh --no-push        # update api_url.txt but do not commit/push
#
# Quick-tunnel URLs are ephemeral: they change whenever cloudflared restarts.
# The frontend reads api_url.txt at startup with a cache-buster, so the
# permanent QR/link target https://minkyu1022.github.io/CatFlow/app keeps
# working — only the backend URL behind it changes, and re-running this script
# is enough to re-point it.
#
# Env overrides: PORT, CLOUDFLARED (path to the cloudflared binary).
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
CF="${CLOUDFLARED:-$HOME/bin/cloudflared}"
API_FILE="../docs/app/api_url.txt"
LOG=/tmp/catflow_tunnel.log
PUSH=1
[ "$1" = "--no-push" ] && PUSH=0

[ -x "$CF" ] || { echo "[tunnel] cloudflared not found at $CF" >&2; exit 1; }
[ -d "$(dirname "$API_FILE")" ] || { echo "[tunnel] docs/app missing" >&2; exit 1; }

# fail fast if the local server is not up
if ! curl -fsS "http://localhost:$PORT/api/adsorbates" >/dev/null 2>&1; then
    echo "[tunnel] WARNING: http://localhost:$PORT is not responding." >&2
    echo "[tunnel]          start the backend first:  ./run.sh" >&2
fi

"$CF" tunnel --url "http://localhost:$PORT" > "$LOG" 2>&1 &
CF_PID=$!
trap 'kill $CF_PID 2>/dev/null || true' EXIT
echo "[tunnel] cloudflared started (pid $CF_PID), waiting for URL ..."

URL=""
for _ in $(seq 1 30); do
    URL=$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$LOG" | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
done
if [ -z "$URL" ]; then
    echo "[tunnel] ERROR: no tunnel URL after 30s — log follows:" >&2
    cat "$LOG" >&2
    exit 1
fi
echo "[tunnel] public backend URL: $URL"

printf '%s\n' "$URL" > "$API_FILE"
echo "[tunnel] wrote $API_FILE"

if [ "$PUSH" = 1 ]; then
    if git -C .. diff --quiet -- docs/app/api_url.txt; then
        echo "[tunnel] api_url.txt unchanged — nothing to push."
    else
        git -C .. add docs/app/api_url.txt
        git -C .. commit -q -m "webapp: point demo backend at new tunnel URL"
        git -C .. push -q
        echo "[tunnel] pushed — GitHub Pages will serve the new URL in ~1 min."
    fi
else
    echo "[tunnel] --no-push: commit docs/app/api_url.txt yourself when ready."
fi

echo
echo "  Demo (QR/link target, permanent):  https://minkyu1022.github.io/CatFlow/app"
echo "  Backend (this tunnel, ephemeral):  $URL"
echo
echo "[tunnel] live — leave this process running for the whole session."
echo "[tunnel] Ctrl-C stops the tunnel (the demo page then goes offline)."
wait $CF_PID
