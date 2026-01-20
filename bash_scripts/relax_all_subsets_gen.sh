#!/bin/bash

# ================= CONFIGURATION =================
BASE_DIR="unrelaxed_samples/de_novo_generation"

CUDA_DEVICES="0,1,2,3,4,5,6,7"
NUM_GPUS=8
# =================================================

echo "Starting relaxation pipeline..."
echo "Base Directory: $BASE_DIR"

for subdir in "$BASE_DIR"/*/; do

    if [ ! -d "$subdir" ]; then
        continue
    fi

    subdir=${subdir%/}
    dirname=$(basename "$subdir")
    
    echo ""
    echo "========================================================"
    echo "Processing Subset: $dirname"
    echo "Path: $subdir"
    echo "========================================================"
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python scripts/relax_energy/val_relax_gen.py \
        --data_dir "$subdir" \
        --num_gpus $NUM_GPUS \
        --num_workers 8  
        
    echo "Finished processing $dirname"
done

echo ""
echo "========================================================"
echo "All subsets processed. Aggregating results..."
echo "========================================================"

python scripts/relax_energy/aggregate_relax_results.py --base_dir "$BASE_DIR"

echo "Done!"