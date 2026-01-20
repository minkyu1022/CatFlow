#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python scripts/relax_energy/val_relax_pred.py \
    --data_dir /home/jovyan/MinCatFlow/unrelaxed_samples/sp_all \
    --num_gpus 7 \
    --num_workers 128 \
    --output_json relax_results.json