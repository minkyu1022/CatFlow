"""
Utility functions for EfficientCatGen.
"""

import os
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf


STATS_KEY: str = "stats"


def log_hyperparameters(
    cfg: DictConfig,
    model: pl.LightningModule,
    trainer: pl.Trainer,
) -> None:
    """
    Controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - number of trainable model parameters

    Args:
        cfg: Hydra configuration
        model: Lightning module
        trainer: Lightning trainer
    """
    hparams = OmegaConf.to_container(cfg, resolve=True)

    # Save number of model parameters
    hparams[f"{STATS_KEY}/params_total"] = sum(p.numel() for p in model.parameters())
    hparams[f"{STATS_KEY}/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams[f"{STATS_KEY}/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # Send hparams to all loggers
    if trainer.logger:
        trainer.logger.log_hyperparams(hparams)
        # Disable logging any more hyperparameters
        trainer.logger.log_hyperparams = lambda params: None


# Project root directory
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()
