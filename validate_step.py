"""
Validation step만 실행하는 스크립트.

이 스크립트는 training 없이 validation step만 실행하여 에러를 빠르게 확인할 수 있습니다.

Usage:
    # 기본 사용 (1개 배치만)
    python validate_step.py
    
    # Checkpoint 로드하여 validation 실행
    python validate_step.py checkpoint=path/to/checkpoint.ckpt
    
    # 여러 배치 처리
    python validate_step.py limit_val_batches=5
    
    # Checkpoint와 여러 배치
    python validate_step.py checkpoint=path/to/checkpoint.ckpt limit_val_batches=3
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.core.hydra_config import HydraConfig

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import wandb

torch.set_num_threads(8)

# Enable Tensor Core optimization
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')


# Patch torch.load for legacy checkpoint compatibility
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Patch torch.load to explicitly set weights_only=False (PyTorch 2.6+)."""
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


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


def run_validation_only(cfg: DictConfig) -> None:
    """
    Validation만 실행하는 함수.

    Args:
        cfg: Configuration defined by Hydra
    """
    # Get checkpoint path and limit_val_batches from config overrides
    checkpoint_path = cfg.get("checkpoint", None)
    limit_val_batches = cfg.get("limit_val_batches", 1)
    
    print(f"[INFO] Validation configuration:")
    print(f"  - checkpoint: {checkpoint_path}")
    print(f"  - limit_val_batches: {limit_val_batches}")
    
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

    # Load checkpoint if provided
    if checkpoint_path is not None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("[INFO] Checkpoint loaded successfully")
    else:
        print("[INFO] No checkpoint provided, using randomly initialized model")

    # Logger setup (optional, for debugging)
    wandb_logger = None
    if cfg.logging.get("wandb") and cfg.logging.wandb.get("mode") != "disabled":
        print("[INFO] Instantiating <WandbLogger> (disabled mode for validation-only run)")
        # Disable wandb for validation-only runs to avoid cluttering
        wandb_logger = WandbLogger(
            project=cfg.logging.wandb.project,
            name=f"{cfg.logging.wandb.name}_val_only" if cfg.logging.wandb.get("name") else "val_only",
            mode="disabled",  # Disable wandb logging for quick validation checks
            save_dir=str(hydra_dir),
            settings=wandb.Settings(start_method="fork"),
        )

    # Instantiate trainer with validation-only settings
    print("[INFO] Instantiating the Trainer (validation-only mode)")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=[],  # No callbacks needed for validation-only
        deterministic=cfg.train.deterministic,
        limit_val_batches=limit_val_batches,  # Limit to specified number of batches
        num_sanity_val_steps=0,  # Skip sanity checks
        accelerator=cfg.train.pl_trainer.get("accelerator", "gpu"),
        devices=cfg.train.pl_trainer.get("devices", 1),
        strategy=cfg.train.pl_trainer.get("strategy", "auto"),
        precision=cfg.train.pl_trainer.get("precision", 32),
    )

    # Setup datamodule
    # Note: validation dataset is set up when stage="fit" (see LMDBDataModule.setup)
    print("[INFO] Setting up datamodule...")
    datamodule.setup(stage="fit")

    # Run validation
    print(f"[INFO] Starting validation (processing {limit_val_batches} batch(es))...")
    try:
        trainer.validate(model=model, datamodule=datamodule)
        print("[INFO] Validation completed successfully!")
    except Exception as e:
        print(f"[ERROR] Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Close logger
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(
    config_path=str(PROJECT_ROOT / "configs"), 
    config_name="default", 
    version_base=None
)
def main(cfg: DictConfig):
    run_validation_only(cfg=cfg)


if __name__ == "__main__":
    main()

