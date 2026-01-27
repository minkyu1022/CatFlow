#!/bin/bash

# ================= CONFIGURATION =================
CHECKPOINT_PATH="ckpts/pred_430M_final_L1_relpos/sp_model.ckpt"

DATA_ROOT="data/dataset_per_adsorbate"

BASE_OUTPUT_DIR="data/sp_results/initial_structures"

CUDA_DEVICES_STR="0,1,2,3,4,5,6,7"
# =================================================

export PYTHONPATH=$PYTHONPATH:.

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_DEVICES_STR"
NUM_AVAILABLE_GPUS=${#GPU_ARRAY[@]}

echo "Starting parallel generation..."
echo "Available GPUs: ${GPU_ARRAY[*]} (Total: $NUM_AVAILABLE_GPUS)"
echo "Target Data Directory: $DATA_ROOT"

LMDB_FILES=()
for f in "${DATA_ROOT}"/*.lmdb; do
    [ -e "$f" ] || continue
    LMDB_FILES+=("$f")
done
TOTAL_FILES=${#LMDB_FILES[@]}

echo "Found $TOTAL_FILES lmdb files to process."
echo "========================================================"

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "Error: No .lmdb files found! Please check DATA_ROOT path."
    exit 1
fi

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

                formula="${filename%.lmdb}"
                
                TARGET_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${formula}"
                
                mkdir -p "$TARGET_OUTPUT_DIR"
                LOG_FILE="$TARGET_OUTPUT_DIR/generation.log"

                echo "[GPU $GPU_ID] Processing: $filename -> Output: $TARGET_OUTPUT_DIR"

                CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/sampling/save_samples.py \
                    --checkpoint "$CHECKPOINT_PATH" \
                    --val_lmdb_path "$lmdb_path" \
                    --output_dir "$TARGET_OUTPUT_DIR" \
                    --num_samples 1 \
                    --sampling_steps 50 \
                    --batch_size 128 \
                    --num_workers 128 \
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

python scripts/sampling/aggregate_stats.py --base_dir "$BASE_OUTPUT_DIR"

echo "Done!"