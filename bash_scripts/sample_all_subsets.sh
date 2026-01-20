#!/bin/bash

# ================= CONFIGURATION =================
CHECKPOINT_PATH="/home/jovyan/mk-catgen-ckpts/gen_430M_final_L1_relpos/epoch=379_9728.ckpt"
DATA_ROOT="/home/jovyan/mk-catgen-data/dataset_per_adsorbate/val_id"
BASE_OUTPUT_DIR="unrelaxed_samples/dng_traj"

CUDA_DEVICES_STR="0,1,2,3,4,5,6,7"
# =================================================

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_DEVICES_STR"
NUM_AVAILABLE_GPUS=${#GPU_ARRAY[@]}

echo "Starting parallel generation..."
echo "Available GPUs: ${GPU_ARRAY[*]} (Total: $NUM_AVAILABLE_GPUS)"

LMDB_FILES=()
for f in "${DATA_ROOT}"/val_id_*_subset.lmdb; do
    [ -e "$f" ] || continue
    LMDB_FILES+=("$f")
done
TOTAL_FILES=${#LMDB_FILES[@]}

echo "Found $TOTAL_FILES lmdb files to process."
echo "========================================================"

pids=()

for ((i=0; i<NUM_AVAILABLE_GPUS; i++)); do
    GPU_ID=${GPU_ARRAY[i]}
    WORKER_IDX=$i

    (
        echo "[Worker $WORKER_IDX] Started on GPU $GPU_ID"

        for ((j=0; j<TOTAL_FILES; j++)); do

            if (( j % NUM_AVAILABLE_GPUS == WORKER_IDX )); then
                
                lmdb_path="${LMDB_FILES[j]}"
                filename=$(basename "$lmdb_path")

                temp=${filename#val_id_}
                composition_str=${temp%_subset.lmdb}

                formula=$(python3 -c "
from collections import Counter
import sys

s = '$composition_str'
elems = s.split('-')
counts = Counter(elems)

formula = ''

if 'C' in counts:
    formula += f\"C{counts['C']}\" if counts['C'] > 1 else \"C\"
    del counts['C']

if 'H' in counts:
    formula += f\"H{counts['H']}\" if counts['H'] > 1 else \"H\"
    del counts['H']

for k in sorted(counts.keys()):
    formula += f\"{k}{counts[k]}\" if counts[k] > 1 else k

print(formula)
")
                
                TARGET_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${formula}"
                
                mkdir -p "$TARGET_OUTPUT_DIR"
                LOG_FILE="$TARGET_OUTPUT_DIR/generation.log"

                echo "[GPU $GPU_ID] Processing: $composition_str -> $formula"

                CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/save_valid_samples.py \
                    --checkpoint "$CHECKPOINT_PATH" \
                    --val_lmdb_path "$lmdb_path" \
                    --output_dir "$TARGET_OUTPUT_DIR" \
                    --num_samples 1 \
                    --sampling_steps 50 \
                    --batch_size 128 \
                    --num_workers 128 \
                    --save_trajectory
                    --gpus 1 > "$LOG_FILE" 2>&1

                if [ $? -eq 0 ]; then
                    echo "[GPU $GPU_ID] Finished: $formula"
                else
                    echo "[GPU $GPU_ID] FAILED: $formula (See $LOG_FILE)"
                fi
            fi
        done
        
        echo "[Worker $WORKER_IDX] All assigned tasks completed."
    ) &

    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo ""
echo "========================================================"
echo "All subsets processed. Aggregating statistics..."
echo "========================================================"

python aggregate_stats.py --base_dir "$BASE_OUTPUT_DIR"

echo "Done!"