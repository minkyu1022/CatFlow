"""
Training script for EfficientCatGen with LMDB dataset.
"""

import os
import sys
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Callback,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.core.hydra_config import HydraConfig

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import log_hyperparameters

import wandb

torch.set_num_threads(8)

# Enable Tensor Core optimization for better performance on Tensor Core GPUs
# 'medium' trades off precision for performance, 'high' maintains higher precision
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')  # Use 'high' for higher precision if needed

# # Scaled Dot-Product Attention settings
# if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
#     print("[DEBUG] Configuring CUDA attention backends")
#     torch.backends.cuda.enable_flash_sdp(False)
#     torch.backends.cuda.enable_mem_efficient_sdp(False)
#     torch.backends.cuda.enable_math_sdp(True)


def set_deterministic_seed(seed: int) -> None:
    """Set deterministic random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    seed_everything(seed, workers=True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"[WARNING] Could not use deterministic algorithms: {e}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"[INFO] Random seed is completely fixed: {seed}")


# Patch torch.load for legacy checkpoint compatibility
# Explicitly set weights_only=False to avoid warnings in PyTorch 2.6+
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Patch torch.load to explicitly set weights_only=False (PyTorch 2.6+)."""
    # Always explicitly set weights_only=False to prevent warnings
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


class SaveEveryEpochCheckpoint(Callback):
    """Save a checkpoint at the end of every training epoch."""

    def __init__(
        self,
        dirpath: Path,
        filename_fmt: str = "epoch={epoch:03d}-train_epoch.ckpt",
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename_fmt = filename_fmt
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        filepath = self.dirpath / self.filename_fmt.format(epoch=epoch)
        trainer.save_checkpoint(str(filepath))


def build_callbacks(cfg: DictConfig, hydra_dir: Path) -> List[Callback]:
    """Build training callbacks from config."""
    callbacks: List[Callback] = []

    # Learning rate monitor
    if cfg.logging.get("lr_monitor"):
        print("[INFO] Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    # Early stopping
    if cfg.train.get("early_stopping"):
        print("[INFO] Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
            )
        )

    # Model checkpoints
    if cfg.train.get("model_checkpoints"):
        print("[INFO] Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=hydra_dir,
                filename="{epoch:02d}-{val_loss:.4f}",
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
            )
        )
        callbacks.append(SaveEveryEpochCheckpoint(dirpath=hydra_dir))

    return callbacks


def run(cfg: DictConfig) -> None:
    """
    Main training loop.

    Args:
        cfg: Configuration defined by Hydra
    """
    # Set seed for reproducibility
    if cfg.train.deterministic:
        set_deterministic_seed(cfg.train.random_seed)

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    print(f"[INFO] Instantiating DataModule: {cfg.data.datamodule._target_}")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    print(f"[INFO] Instantiating Model: {cfg.model._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    # Build callbacks
    callbacks: List[Callback] = build_callbacks(cfg, hydra_dir)

    # Logger setup
    wandb_logger = None
    if cfg.logging.get("wandb"):
        print("[INFO] Instantiating <WandbLogger>")
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=cfg.logging.wandb.name,
            mode=cfg.logging.wandb.mode,
            save_dir=str(hydra_dir),
            settings=wandb.Settings(start_method="fork"),
            tags=cfg.core.tags,
        )
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Save config
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    # Find existing checkpoints
    ckpts = list(hydra_dir.glob("*.ckpt"))
    ckpt = None
    if len(ckpts) > 0:
        last_ckpts = [c for c in ckpts if "last" in c.parts[-1]]
        if len(last_ckpts) > 0:
            ckpt = str(last_ckpts[-1])
        else:
            epoch_ckpts = [c for c in ckpts if "epoch=" in c.parts[-1]]
            if len(epoch_ckpts) > 0:
                ckpt_epochs = np.array(
                    [int(c.parts[-1].split("-")[0].split("=")[1]) for c in epoch_ckpts]
                )
                ckpt = str(epoch_ckpts[ckpt_epochs.argsort()[-1]])
    if ckpt is not None:
        print(f"[INFO] Found checkpoint: {ckpt}")

    # Instantiate trainer
    print("[INFO] Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        **cfg.train.pl_trainer,
    )

    if wandb_logger:
        log_hyperparameters(cfg=cfg, model=model, trainer=trainer)

    # Training
    print("[INFO] Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)

    # Testing
    if cfg.train.run_test:
        print("[INFO] Starting testing!")
        trainer.test(datamodule=datamodule)

    # Close logger
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"), config_name="default", version_base=None
)
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
