#!/bin/bash

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python src/run.py \
    expname=pred_476M_E_cond \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp \
    model.flow_model_args.dng=false \
    model.training_args.flow_loss_type=x1_loss \
    train.pl_trainer.strategy=ddp_find_unused_parameters_true \
    data.batch_size.train=64 \
    data.batch_size.val=64 \
    data.batch_size.test=64 \
