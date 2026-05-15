#!/bin/bash
# Launch the CatFlow web demo.
#   ./run.sh            -> serves on 0.0.0.0:8000
# Env overrides: PORT, CATFLOW_DEVICE (default cuda:0)
set -e
cd "$(dirname "$0")"

export MAMBA_ROOT_PREFIX=/home/irteam/micromamba
PY=/home/irteam/micromamba/envs/adsorbgen/bin/python

# build the adsorbate / composition menus once
if [ ! -f data/adsorbates.json ] || [ ! -f data/compositions.json ]; then
    echo "[run] building menu data ..."
    "$PY" prepare_data.py
fi

export PORT="${PORT:-8000}"
export CATFLOW_DEVICE="${CATFLOW_DEVICE:-cuda:0}"
echo "[run] starting server on port $PORT (device $CATFLOW_DEVICE) ..."
exec "$PY" server.py
