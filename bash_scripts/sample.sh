#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

torchrun --nproc_per_node=7 scripts/sampling/save_valid_samples_direct.py \
    --checkpoint /home/jovyan/mk-catgen-ckpts/gen_430M_final_L1_relpos/epoch=379.ckpt \
    --val_lmdb_path /home/jovyan/mk-catgen-data/dataset/val_id/dataset.lmdb \
    --output_dir unrelaxed_samples/dng_traj/ \
    --num_samples 1 \
    --sampling_steps 50 \
    --batch_size 128 \
    --num_workers 128 \
    --use_ddp
    --save_trajectory