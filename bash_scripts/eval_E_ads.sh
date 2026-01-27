#!/bin/bash

# ================= GLOBAL CONFIGURATION =================
PYTHON_SCRIPT="scripts/relax_energy/E_ads_eval_batch.py"
CUDA_DEVICES="0,1,2,3,4,5,6,7"
NUM_GPUS=8
NUM_WORKERS=128

LOG_FILE="execution_log.txt"
# ========================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$LOG_FILE" >&2
}

log "Starting Sequential Evaluation Pipeline..."
log "Using GPUs: $CUDA_DEVICES"
log "Log file: $LOG_FILE (full path: $(pwd)/$LOG_FILE)"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    log_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

BASE_DIR_1="sp_results/initial_structures"
OUTPUT_DIR_1="sp_results"

log ">>> STARTING: Processing Ref Samples"
log "Input: $BASE_DIR_1"
log "Output: $OUTPUT_DIR_1"

log "Comparing directories between BASE_DIR and OUTPUT_DIR..."

if ! BASE_FOLDERS=$(find "$BASE_DIR_1" -maxdepth 1 -type d ! -path "$BASE_DIR_1" 2>/dev/null | sed 's|.*/||' | sort); then
    log_error "Failed to read BASE_DIR: $BASE_DIR_1"
    exit 1
fi

OUTPUT_FOLDERS=$(find "$OUTPUT_DIR_1" -maxdepth 1 -type d ! -path "$OUTPUT_DIR_1" 2>/dev/null | sed 's|.*/||' | sort)

MISSING_FOLDERS=$(comm -23 <(echo "$BASE_FOLDERS") <(echo "$OUTPUT_FOLDERS"))

BASE_COUNT=$(echo "$BASE_FOLDERS" | wc -l)
OUTPUT_COUNT=$(echo "$OUTPUT_FOLDERS" | wc -l)
MISSING_COUNT=$(echo "$MISSING_FOLDERS" | grep -v '^$' | wc -l)

log "BASE_DIR folders: $BASE_COUNT"
log "OUTPUT_DIR folders: $OUTPUT_COUNT"
log "Missing folders to process: $MISSING_COUNT"

if [ -z "$MISSING_FOLDERS" ] || [ "$MISSING_COUNT" -eq 0 ]; then
    log "All folders already exist in OUTPUT_DIR. Nothing to process."
    log ">>> COMPLETED (skipped - all folders exist)."
    exit 0
fi

log "Missing folders:"
echo "$MISSING_FOLDERS" | while read -r folder; do
    [ -n "$folder" ] && log "  - $folder"
done

for subdir in "$BASE_DIR_1"/*/; do
    if [ ! -d "$subdir" ]; then continue; fi
    
    subdir=${subdir%/}
    dirname=$(basename "$subdir")
    
    if [ -d "$OUTPUT_DIR_1/$dirname" ]; then
        log "Skipping $dirname (already exists in OUTPUT_DIR)"
        continue
    fi
    
    log "Processing /$dirname..."
    
    TEMP_LOG=$(mktemp)
    
    if CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python "$PYTHON_SCRIPT" \
        --data_dir "$subdir" \
        --output_dir "$OUTPUT_DIR_1" \
        --num_gpus $NUM_GPUS \
        --num_workers $NUM_WORKERS >> "$TEMP_LOG" 2>&1; then
        
        OUTPUT_FOLDER="$OUTPUT_DIR_1/$dirname"
        INITIAL_JSON="$OUTPUT_FOLDER/initial_energy_results.json"
        RELAXED_JSON="$OUTPUT_FOLDER/relaxed_energy_results.json"
        
        HAS_DATA=false
        if [ -f "$INITIAL_JSON" ] && [ -s "$INITIAL_JSON" ]; then
            JSON_CONTENT=$(cat "$INITIAL_JSON" 2>/dev/null | tr -d ' \n\t')
            if [ "$JSON_CONTENT" != "[]" ] && [ ${#JSON_CONTENT} -gt 2 ]; then
                HAS_DATA=true
            fi
        fi
        
        if [ "$HAS_DATA" = false ] && [ -f "$RELAXED_JSON" ] && [ -s "$RELAXED_JSON" ]; then
            JSON_CONTENT=$(cat "$RELAXED_JSON" 2>/dev/null | tr -d ' \n\t')
            if [ "$JSON_CONTENT" != "[]" ] && [ ${#JSON_CONTENT} -gt 2 ]; then
                HAS_DATA=true
            fi
        fi
        
        cat "$TEMP_LOG" >> "$LOG_FILE"
        
        if [ "$HAS_DATA" = false ]; then

            log_error "Processing completed but no data was saved for $dirname"
            echo ""
            echo "=========================================="
            echo "WARNING: No data processed for: $dirname"
            echo "Output files are empty or missing."
            echo "------------------------------------------"
            echo "Python script output (last 50 lines):"
            cat "$TEMP_LOG" | tail -50
            echo "=========================================="
            echo ""
            
            echo "--- WARNING: NO DATA FOR $dirname ---" >> "$LOG_FILE"
            cat "$TEMP_LOG" >> "$LOG_FILE"
            echo "--- END WARNING ---" >> "$LOG_FILE"
            
            rm -f "$TEMP_LOG"
            log_error "Stopping execution due to empty results in $dirname"
            exit 1
        else
            log "Successfully processed $dirname (data verified)"
        fi
    else

        EXIT_CODE=$?
        log_error "Failed to process $dirname (exit code: $EXIT_CODE)"
        echo ""
        echo "=========================================="
        echo "ERROR occurred while processing: $dirname"
        echo "Exit code: $EXIT_CODE"
        echo "------------------------------------------"
        echo "Error output:"
        cat "$TEMP_LOG" | tail -50 
        echo "=========================================="
        echo ""
        
        echo "--- ERROR OUTPUT FOR $dirname ---" >> "$LOG_FILE"
        cat "$TEMP_LOG" >> "$LOG_FILE"
        echo "--- END ERROR OUTPUT ---" >> "$LOG_FILE"
        
        rm -f "$TEMP_LOG"
        
        log_error "Stopping execution due to error in $dirname"
        exit $EXIT_CODE
    fi
    
    rm -f "$TEMP_LOG"
done

python scripts/relax_energy/aggregate_relax_results.py --base_dir "$OUTPUT_DIR_1"

log ">>> COMPLETED."