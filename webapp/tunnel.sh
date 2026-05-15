#!/bin/bash
# Expose the local CatFlow server through a cloudflared quick tunnel, wire the
# GitHub Pages frontend (docs/app/app.js) to the tunnel URL, and push so Pages
# serves it. One command takes the public demo fully live.
#
#   ./tunnel.sh                  # tunnel localhost:$PORT, patch app.js, push
#   ./tunnel.sh --no-push        # patch app.js but do not commit/push
#
# Quick-tunnel URLs are ephemeral: they change whenever cloudflared restarts.
# This script re-patches and re-pushes docs/app/app.js every run, so the
# permanent QR/link target https://minkyu1022.github.io/CatFlow/app keeps
# working — only the backend URL behind it changes.
#
# Env overrides: PORT, CLOUDFLARED (path to the cloudflared binary).
set -e
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
CF="${CLOUDFLARED:-$HOME/bin/cloudflared}"
APP_JS="../docs/app/app.js"
LOG=/tmp/catflow_tunnel.log
PUSH=1
[ "$1" = "--no-push" ] && PUSH=0

[ -x "$CF" ] || { echo "[tunnel] cloudflared not found at $CF" >&2; exit 1; }
[ -f "$APP_JS" ] || { echo "[tunnel] $APP_JS missing" >&2; exit 1; }

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

sed -i -E "s|^const API = \".*\";|const API = \"$URL\";|" "$APP_JS"
echo "[tunnel] patched $APP_JS"

if [ "$PUSH" = 1 ]; then
    if git -C .. diff --quiet -- docs/app/app.js; then
        echo "[tunnel] app.js unchanged — nothing to push."
    else
        git -C .. add docs/app/app.js
        git -C .. commit -q -m "webapp: point demo backend at new tunnel URL"
        git -C .. push -q
        echo "[tunnel] pushed — GitHub Pages will serve the new URL in ~1 min."
    fi
else
    echo "[tunnel] --no-push: commit docs/app/app.js yourself when ready."
fi

echo
echo "  Demo (QR/link target, permanent):  https://minkyu1022.github.io/CatFlow/app"
echo "  Backend (this tunnel, ephemeral):  $URL"
echo
echo "[tunnel] live — leave this process running for the whole session."
echo "[tunnel] Ctrl-C stops the tunnel (the demo page then goes offline)."
wait $CF_PID
