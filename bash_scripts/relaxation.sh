#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/val_relax_gen.py \
    --data_dir unrelaxed_samples/de_novo_generation/C2H2O/ \
    --num_gpus 4
