#!/bin/bash

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python src/run.py \
    expname=gen_431M_E_cond_ads_aug_128_2500 \
    train.pl_trainer.devices=4 \
    train.pl_trainer.strategy=ddp \
    model.flow_model_args.dng=true \
    model.training_args.flow_loss_type=x1_loss \
    model.training_args.warmup_steps=2500 \
    model.atom_s=768 \
    model.token_s=768 \
    model.flow_model_args.atom_encoder_depth=8 \
    model.flow_model_args.atom_encoder_heads=12 \
    model.flow_model_args.token_transformer_depth=24 \
    model.flow_model_args.token_transformer_heads=12 \
    model.flow_model_args.atom_decoder_depth=8 \
    model.flow_model_args.atom_decoder_heads=12 \
    train.pl_trainer.strategy=ddp_find_unused_parameters_true \
    data.batch_size.train=128 \
    data.batch_size.val=128 \
    data.batch_size.test=128 \
