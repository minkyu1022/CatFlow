#!/bin/bash

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python src/run.py \
    expname=pred_430M_final_L1_relpos \
    train.pl_trainer.devices=7 \
    train.pl_trainer.strategy=ddp \
    train.pl_trainer.strategy=ddp_find_unused_parameters_true \
    model.flow_model_args.dng=false \
    model.training_args.flow_loss_type=x1_loss \
    model.training_args.loss_type=l1 \
    model.training_args.lr=2e-5 \
    model.training_args.warmup_steps=5000 \
    model.flow_model_args.use_energy_cond=false \
    model.validation_args.sample_every_n_epochs=5 \
    model.atom_s=768 \
    model.token_s=768 \
    model.flow_model_args.atom_encoder_depth=8 \
    model.flow_model_args.atom_encoder_heads=12 \
    model.flow_model_args.token_transformer_depth=24 \
    model.flow_model_args.token_transformer_heads=12 \
    model.flow_model_args.atom_decoder_depth=8 \
    model.flow_model_args.atom_decoder_heads=12 \
    data.batch_size.train=128 \
    data.batch_size.val=128 \
    data.batch_size.test=128 \
    data.num_workers.train=16 \
    data.num_workers.val=16 \
    data.num_workers.test=16 \
