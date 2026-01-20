"""
Script to load checkpoint, generate samples via forward, and save only structurally valid samples.

Usage (single GPU or DataParallel):
    python scripts/save_valid_samples.py \
        --checkpoint path/to/checkpoint.ckpt \
        --val_lmdb_path path/to/val/dataset.lmdb \
        --num_samples 1 \
        --sampling_steps 100 \
        --batch_size 4 \
        --output_dir unrelaxed_samples/structure_prediction

Usage (DDP with torchrun):
    torchrun --nproc_per_node=4 scripts/save_valid_samples.py \
        --checkpoint path/to/checkpoint.ckpt \
        --val_lmdb_path path/to/val/dataset.lmdb \
        --num_samples 1 \
        --sampling_steps 100 \
        --batch_size 4 \
        --output_dir unrelaxed_samples/structure_prediction \
        --use_ddp
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm
from ase.io import write as ase_write

from src.module.effcat_module import EffCatModule
from src.data.datamodule import LMDBDataModule
from scripts.assemble import assemble
from src.models.loss.validation import compute_structural_validity_single


# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Patch torch.load to disable weights_only=True default (PyTorch 2.6+)."""
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    try:
        return _original_torch_load(*args, **kwargs, weights_only=False)
    except TypeError:
        return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def load_model(checkpoint_path: str, device: str = "cuda", use_ddp: bool = False, local_rank: int = 0) -> EffCatModule:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ("cuda" or "cpu")
        use_ddp: Whether to use DistributedDataParallel (requires torchrun)
        local_rank: Local rank for DDP (which GPU to use for this process)
    
    Returns:
        Loaded EffCatModule model
    """
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    
    model = EffCatModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu"
    )
    
    model.eval()
    
    if device == "cuda" and torch.cuda.is_available():
        if use_ddp:
            # For DDP, move model to the specific GPU for this rank
            model = model.cuda(local_rank)
            print(f"[INFO] Model moved to GPU {local_rank} for DDP")
        else:
            # Use DataParallel for simple multi-GPU
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"[INFO] Using DataParallel on {num_gpus} GPUs")
                model = model.cuda()
                model = torch.nn.DataParallel(model)
            else:
                model = model.cuda()
                print(f"[INFO] Moving model to GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Running on CPU.")
    
    return model


def load_val_dataloader(
    val_lmdb_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    preload_to_ram: bool = True,
):
    """
    Load validation dataloader.
    
    Args:
        val_lmdb_path: Path to validation LMDB file
        batch_size: Batch size
        num_workers: Number of data loading workers
        preload_to_ram: Whether to preload to RAM
    
    Returns:
        DataLoader object
    """
    from omegaconf import OmegaConf
    
    print(f"[INFO] Loading validation data: {val_lmdb_path}")
    
    # Create DictConfig
    batch_size_cfg = OmegaConf.create({
        "train": batch_size,
        "val": batch_size,
        "test": batch_size
    })
    num_workers_cfg = OmegaConf.create({
        "train": num_workers,
        "val": num_workers,
        "test": num_workers
    })
    
    datamodule = LMDBDataModule(
        train_lmdb_path=val_lmdb_path,  # Set val dataset as train (for setup)
        val_lmdb_path=val_lmdb_path,
        test_lmdb_path=None,
        batch_size=batch_size_cfg,
        num_workers=num_workers_cfg,
        preload_to_ram=preload_to_ram,
    )
    
    datamodule.setup(stage="fit")
    dataloader = datamodule.val_dataloader()
    
    print(f"[INFO] Loaded {len(datamodule.val_dataset)} samples in total")
    
    return dataloader


def generate_samples_and_save_valid(
    model: EffCatModule,
    dataloader,
    output_dir: str,
    num_samples: int = 1,
    sampling_steps: int = 100,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    sampler: Optional[Any] = None,
    save_trajectory: bool = False,
    save_invalid: bool = False,
) -> Dict[str, Any]:
    """
    Generate samples using model forward, check structural validity, and save valid ones.
    
    Args:
        model: EffCatModule model
        dataloader: DataLoader for validation data
        output_dir: Output directory to save traj files
        num_samples: Number of samples to generate per input
        sampling_steps: Number of sampling steps
        device: Device to use
        max_batches: Maximum number of batches to process (None for all)
        rank: Process rank for DDP (0 for single GPU or DataParallel)
        world_size: Number of processes for DDP (1 for single GPU or DataParallel)
        local_rank: Local rank (GPU index) for DDP (0 for single GPU or DataParallel)
        sampler: DistributedSampler instance (if using DDP) to track original indices
        save_trajectory: If True, save full sampling trajectory for each sample
        save_invalid: If True, save invalid samples to subdirectories by failure reason
    
    Returns:
        Statistics dictionary
    """
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for invalid samples (if requested)
    invalid_dirs = None
    sample_invalid_dirs = None
    if save_invalid:
        invalid_dirs = {
            'vol_invalid': output_path / 'vol_invalid',
            'dist_invalid': output_path / 'dist_invalid',
            'height_invalid': output_path / 'height_invalid',
            'assemble_failed': output_path / 'assemble_failed',
        }
        for invalid_dir in invalid_dirs.values():
            invalid_dir.mkdir(parents=True, exist_ok=True)
    
    # For num_samples > 1, create subdirectories for each sample index
    sample_dirs = None
    if num_samples > 1:
        sample_dirs = []
        for i in range(num_samples):
            sample_dir = output_path / str(i)
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_dirs.append(sample_dir)
        
        # Also create invalid subdirectories for each sample index (if requested)
        if save_invalid:
            sample_invalid_dirs = {key: [] for key in invalid_dirs.keys()}
            for i in range(num_samples):
                for key, base_dir in invalid_dirs.items():
                    sample_invalid_dir = base_dir / str(i)
                    sample_invalid_dir.mkdir(parents=True, exist_ok=True)
                    sample_invalid_dirs[key].append(sample_invalid_dir)
        
        if rank == 0:
            print(f"[INFO] Output directory: {output_path}")
            print(f"[INFO] Created {num_samples} sample subdirectories: 0/ to {num_samples-1}/")
            if save_invalid:
                print(f"[INFO] Created invalid sample directories: {', '.join(invalid_dirs.keys())}")
    else:
        if rank == 0:
            print(f"[INFO] Output directory: {output_path}")
            if save_invalid:
                print(f"[INFO] Created invalid sample directories: {', '.join(invalid_dirs.keys())}")
    
    # Statistics
    total_processed = 0
    total_valid = 0
    total_invalid = 0
    total_failed = 0
    
    # Validity check statistics (per-check breakdown)
    validity_stats = {
        'vol_failed': 0,
        'dist_failed': 0,
        'height_failed': 0,
        'assemble_failed': 0,
    }
    
    # Track original dataset indices
    # For DistributedSampler: indices are rank + i * num_replicas
    # We need to track which original dataset indices are being processed
    
    if rank == 0:
        print(f"\n[INFO] Starting sampling and validation")
        print(f"  - num_samples: {num_samples}")
        print(f"  - sampling_steps: {sampling_steps}")
        print(f"  - batch_size: {dataloader.batch_size}")
        if world_size > 1:
            print(f"  - Using {world_size} GPUs (rank {rank})")
    
    # Iterate over batches
    cumulative_samples_processed = 0  # Track cumulative count to handle variable batch sizes
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing batches (rank {rank})", disable=(rank != 0))):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        # Calculate original dataset indices for this batch
        current_batch_size = batch["ref_prim_slab_element"].shape[0]
        original_indices = []
        
        if sampler is not None and world_size > 1:
            for i in range(current_batch_size):
                sample_in_rank_sequence = cumulative_samples_processed + i
                original_idx = rank + sample_in_rank_sequence * world_size
                original_indices.append(original_idx)
        else:
            # For non-DDP or DataParallel, indices are sequential starting from 0
            for i in range(current_batch_size):
                original_indices.append(cumulative_samples_processed + i)
        
        # Update cumulative count AFTER calculating indices for this batch
        cumulative_samples_processed += current_batch_size
        
        # Move batch to device
        if device == "cuda" and torch.cuda.is_available():
            if isinstance(model, torch.nn.DataParallel):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif world_size > 1:
                batch = {k: v.cuda(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        try:
            # Generate samples using forward (like validation_step)
            with torch.no_grad():
                if isinstance(model, torch.nn.DataParallel):
                    out = model.module(
                        batch,
                        num_sampling_steps=sampling_steps,
                        center_during_sampling=False,
                        multiplicity_flow_sample=num_samples,
                        return_trajectory=save_trajectory,
                    )
                else:
                    out = model(
                        batch,
                        num_sampling_steps=sampling_steps,
                        center_during_sampling=False,
                        multiplicity_flow_sample=num_samples,
                        return_trajectory=save_trajectory,
                    )
            
            # Convert to numpy
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.cpu().numpy()
                return x
            
            sampled_prim_slab_coords = to_numpy(out["sampled_prim_slab_coords"])  # (B*M, N, 3)
            sampled_ads_coords = to_numpy(out["sampled_ads_coords"])  # (B*M, A, 3)
            sampled_lattices = to_numpy(out["sampled_lattice"])  # (B*M, 6)
            sampled_supercell_matrices = to_numpy(out["sampled_supercell_matrix"])  # (B*M, 3, 3)
            sampled_scaling_factors = to_numpy(out["sampled_scaling_factor"])  # (B*M,)
            
            if isinstance(model, torch.nn.DataParallel):
                model_dng = model.module.dng
            else:
                model_dng = model.dng
            
            if model_dng and "sampled_prim_slab_element" in out:
                prim_slab_atom_types = to_numpy(out["sampled_prim_slab_element"])  # (B*M, N)
            else:
                prim_slab_atom_types = to_numpy(batch["ref_prim_slab_element"])  # (B, N)
            
            ads_atom_types = to_numpy(batch["ref_ads_element"])  # (B, A)
            prim_slab_atom_mask = to_numpy(batch["prim_slab_atom_pad_mask"])  # (B, N)
            ads_atom_mask = to_numpy(batch["ads_atom_pad_mask"])  # (B, A)
            
            tags = batch.get("tags", None)
            if tags is not None:
                tags = to_numpy(tags)
            
            total_samples_in_batch = sampled_prim_slab_coords.shape[0]
            
            for sample_idx in range(total_samples_in_batch):
                batch_item_idx = sample_idx // num_samples
                original_dataset_idx = original_indices[batch_item_idx]
                sample_in_item = sample_idx % num_samples
                
                sample_prim_slab_coords = sampled_prim_slab_coords[sample_idx]
                sample_ads_coords = sampled_ads_coords[sample_idx]
                sample_lattice = sampled_lattices[sample_idx]
                sample_supercell_matrix = sampled_supercell_matrices[sample_idx]
                sample_scaling_factor = sampled_scaling_factors[sample_idx]
                
                if model_dng and "sampled_prim_slab_element" in out:
                    prim_slab_types = prim_slab_atom_types[sample_idx]
                else:
                    prim_slab_types = prim_slab_atom_types[batch_item_idx]
                
                ads_types = ads_atom_types[batch_item_idx]
                prim_slab_mask = prim_slab_atom_mask[batch_item_idx]
                ads_mask = ads_atom_mask[batch_item_idx]
                
                try:
                    assemble_failed = False
                    recon_system = None
                    recon_slab = None
                    
                    try:
                        recon_system, recon_slab = assemble(
                            generated_prim_slab_coords=sample_prim_slab_coords,
                            generated_ads_coords=sample_ads_coords,
                            generated_lattice=sample_lattice,
                            generated_supercell_matrix=sample_supercell_matrix.reshape(3, 3),
                            generated_scaling_factor=float(sample_scaling_factor),
                            prim_slab_atom_types=prim_slab_types,
                            ads_atom_types=ads_types,
                            prim_slab_atom_mask=prim_slab_mask,
                            ads_atom_mask=ads_mask,
                        )
                    except Exception as e:
                        assemble_failed = True
                        if rank == 0:
                            print(f"\n[WARNING] Assembly failed for sample {original_dataset_idx}: {e}")
                    
                    if not assemble_failed:
                        sc_matrix_3x3 = sample_supercell_matrix.reshape(3, 3)
                        validity_task = (
                            sample_prim_slab_coords[np.newaxis, :],
                            sample_ads_coords[np.newaxis, :],
                            sample_lattice[np.newaxis, :],
                            sc_matrix_3x3[np.newaxis, :, :],
                            np.array([sample_scaling_factor]),
                            prim_slab_types,
                            ads_types,
                            prim_slab_mask,
                            ads_mask,
                        )
                        
                        validity_results, validity_details = compute_structural_validity_single(
                            validity_task, return_details=True
                        )
                        is_valid = validity_results[0]
                        details = validity_details[0]
                    else:
                        is_valid = False
                        details = {
                            'vol_ok': False,
                            'dist_ok': False,
                            'height_ok': False,
                            'assemble_failed': True
                        }
                    
                    if not is_valid:
                        failed_checks = []
                        if details.get('assemble_failed', False):
                            validity_stats['assemble_failed'] += 1
                            failed_checks.append('assemble_failed')
                        else:
                            if not details.get('vol_ok', True):
                                validity_stats['vol_failed'] += 1
                                failed_checks.append('vol_invalid')
                            if not details.get('dist_ok', True):
                                validity_stats['dist_failed'] += 1
                                failed_checks.append('dist_invalid')
                            if not details.get('height_ok', True):
                                validity_stats['height_failed'] += 1
                                failed_checks.append('height_invalid')
                        
                        if save_invalid and failed_checks:
                            for failed_check in failed_checks:
                                if num_samples > 1:
                                    invalid_traj_path = sample_invalid_dirs[failed_check][sample_in_item] / f"{original_dataset_idx}.traj"
                                    invalid_pt_path = sample_invalid_dirs[failed_check][sample_in_item] / f"{original_dataset_idx}.pt"
                                else:
                                    invalid_traj_path = invalid_dirs[failed_check] / f"{original_dataset_idx}.traj"
                                    invalid_pt_path = invalid_dirs[failed_check] / f"{original_dataset_idx}.pt"
                                
                                try:
                                    if not details.get('assemble_failed', False) and recon_system is not None:
                                        ase_write(str(invalid_traj_path), recon_system, format="traj")
                                    
                                    validity_details_clean = {
                                        k: bool(v) if isinstance(v, (np.bool_, bool)) else v
                                        for k, v in details.items()
                                    }
                                    
                                    invalid_sample_data = {
                                        "prim_slab_coords": torch.from_numpy(sample_prim_slab_coords).cpu(),
                                        "ads_coords": torch.from_numpy(sample_ads_coords).cpu(),
                                        "lattice": torch.from_numpy(sample_lattice).cpu(),
                                        "supercell_matrix": torch.from_numpy(sample_supercell_matrix.reshape(3, 3)).cpu(),
                                        "scaling_factor": torch.tensor(float(sample_scaling_factor)).cpu(),
                                        "prim_slab_atom_types": torch.from_numpy(prim_slab_types).cpu() if isinstance(prim_slab_types, np.ndarray) else prim_slab_types.cpu() if isinstance(prim_slab_types, torch.Tensor) else torch.tensor(prim_slab_types).cpu(),
                                        "ads_atom_types": torch.from_numpy(ads_types).cpu() if isinstance(ads_types, np.ndarray) else ads_types.cpu() if isinstance(ads_types, torch.Tensor) else torch.tensor(ads_types).cpu(),
                                        "prim_slab_atom_mask": torch.from_numpy(prim_slab_mask).cpu() if isinstance(prim_slab_mask, np.ndarray) else prim_slab_mask.cpu() if isinstance(prim_slab_mask, torch.Tensor) else torch.tensor(prim_slab_mask).cpu(),
                                        "ads_atom_mask": torch.from_numpy(ads_mask).cpu() if isinstance(ads_mask, np.ndarray) else ads_mask.cpu() if isinstance(ads_mask, torch.Tensor) else torch.tensor(ads_mask).cpu(),
                                        "original_dataset_idx": int(original_dataset_idx),
                                        "sample_in_item": int(sample_in_item),
                                        "batch_idx": int(batch_idx),
                                        "validity_details": validity_details_clean,
                                        "failed_check": str(failed_check),
                                    }
                                    torch.save(invalid_sample_data, invalid_pt_path, _use_new_zipfile_serialization=True)
                                except Exception as e:
                                    if rank == 0:
                                        print(f"\n[WARNING] Failed to save invalid sample {original_dataset_idx} to {failed_check}: {e}")
                    
                    if is_valid:
                        if num_samples > 1:
                            traj_path = sample_dirs[sample_in_item] / f"{original_dataset_idx}.traj"
                        else:
                            traj_path = output_path / f"{original_dataset_idx}.traj"
                        
                        if save_trajectory and "prim_slab_coord_trajectory" in out:
                            traj_prim_slab_coords_tensor = out["prim_slab_coord_trajectory"]
                            traj_ads_coords_tensor = out["ads_coord_trajectory"]
                            traj_lattices_tensor = out["lattice_trajectory"]
                            traj_supercell_matrices_tensor = out["supercell_matrix_trajectory"]
                            traj_scaling_factors_tensor = out["scaling_factor_trajectory"]
                            
                            traj_prim_slab_element_tensor = None
                            traj_prim_slab_elements = None
                            if model_dng and "prim_slab_element_trajectory" in out:
                                traj_prim_slab_element_tensor = out["prim_slab_element_trajectory"]
                                traj_prim_slab_elements = to_numpy(traj_prim_slab_element_tensor)
                            
                            traj_prim_slab_coords = to_numpy(traj_prim_slab_coords_tensor)
                            traj_ads_coords = to_numpy(traj_ads_coords_tensor)
                            traj_lattices = to_numpy(traj_lattices_tensor)
                            traj_supercell_matrices = to_numpy(traj_supercell_matrices_tensor)
                            traj_scaling_factors = to_numpy(traj_scaling_factors_tensor)
                            
                            num_steps = traj_prim_slab_coords.shape[0]
                            trajectory_structures = []
                            
                            for step_idx in range(num_steps):
                                step_prim_slab_coords = traj_prim_slab_coords[step_idx, sample_idx]
                                step_ads_coords = traj_ads_coords[step_idx, sample_idx]
                                step_lattice = traj_lattices[step_idx, sample_idx]
                                step_supercell_matrix = traj_supercell_matrices[step_idx, sample_idx]
                                step_scaling_factor = traj_scaling_factors[step_idx, sample_idx]
                                
                                if traj_prim_slab_elements is not None:
                                    step_prim_slab_types = traj_prim_slab_elements[step_idx, sample_idx]
                                else:
                                    step_prim_slab_types = prim_slab_types
                                
                                try:
                                    step_system, _ = assemble(
                                        generated_prim_slab_coords=step_prim_slab_coords,
                                        generated_ads_coords=step_ads_coords,
                                        generated_lattice=step_lattice,
                                        generated_supercell_matrix=step_supercell_matrix,
                                        generated_scaling_factor=float(step_scaling_factor),
                                        prim_slab_atom_types=step_prim_slab_types,
                                        ads_atom_types=ads_types,
                                        prim_slab_atom_mask=prim_slab_mask,
                                        ads_atom_mask=ads_mask,
                                    )
                                    trajectory_structures.append(step_system)
                                except Exception:
                                    continue
                            
                            if trajectory_structures:
                                ase_write(str(traj_path), trajectory_structures, format="traj")
                            
                            sample_trajectory = {
                                "prim_slab_coord_trajectory": traj_prim_slab_coords_tensor[:, sample_idx, :, :].cpu(),
                                "ads_coord_trajectory": traj_ads_coords_tensor[:, sample_idx, :, :].cpu(),
                                "lattice_trajectory": traj_lattices_tensor[:, sample_idx, :].cpu(),
                                "supercell_matrix_trajectory": traj_supercell_matrices_tensor[:, sample_idx, :, :].cpu(),
                                "scaling_factor_trajectory": traj_scaling_factors_tensor[:, sample_idx].cpu(),
                                "prim_slab_atom_types": torch.from_numpy(prim_slab_types).cpu() if isinstance(prim_slab_types, np.ndarray) else prim_slab_types.cpu() if isinstance(prim_slab_types, torch.Tensor) else prim_slab_types,
                                "ads_atom_types": torch.from_numpy(ads_types).cpu() if isinstance(ads_types, np.ndarray) else ads_types.cpu() if isinstance(ads_types, torch.Tensor) else ads_types,
                                "prim_slab_atom_mask": torch.from_numpy(prim_slab_mask).cpu() if isinstance(prim_slab_mask, np.ndarray) else prim_slab_mask.cpu() if isinstance(prim_slab_mask, torch.Tensor) else prim_slab_mask,
                                "ads_atom_mask": torch.from_numpy(ads_mask).cpu() if isinstance(ads_mask, np.ndarray) else ads_mask.cpu() if isinstance(ads_mask, torch.Tensor) else ads_mask,
                            }
                            
                            if traj_prim_slab_element_tensor is not None:
                                sample_trajectory["prim_slab_element_trajectory"] = traj_prim_slab_element_tensor[:, sample_idx, :].cpu()
                            
                            pt_path = traj_path.with_suffix('.pt')
                            torch.save(sample_trajectory, pt_path)
                        else:
                            ase_write(str(traj_path), recon_system, format="traj")
                        
                        total_valid += 1
                    else:
                        total_invalid += 1
                    
                    total_processed += 1
                    
                except Exception as e:
                    if rank == 0:
                        print(f"\n[WARNING] Failed to process sample (batch {batch_idx}, sample {sample_idx}): {e}")
                    total_failed += 1
                    total_processed += 1
                    continue
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[WARNING] Out of memory at batch {batch_idx}. Skipping.")
                torch.cuda.empty_cache()
                total_failed += current_batch_size * num_samples
                continue
            else:
                raise e
    
    # Summary
    validity_rate = total_valid / total_processed * 100 if total_processed > 0 else 0.0
    saved_rate = total_valid / total_processed * 100 if total_processed > 0 else 0.0
    
    results = {
        "total_processed": total_processed,
        "total_valid": total_valid,
        "total_invalid": total_invalid,
        "total_failed": total_failed,
        "validity_rate": validity_rate,
        "saved_rate": saved_rate,
        "output_dir": str(output_path),
        "rank": rank,
        "validity_stats": validity_stats,
    }
    
    if rank == 0:
        print("\n" + "=" * 50)
        print("Results Summary")
        print("=" * 50)
        print(f"Total processed: {total_processed}")
        print(f"\n[FINAL RESULTS]")
        print(f"  - Saved samples (valid structures): {total_valid}")
        print(f"  - Saved sample rate: {saved_rate:.2f}% ({total_valid}/{total_processed})")
        print(f"\n[BREAKDOWN]")
        print(f"  - Valid structures saved: {total_valid}")
        print(f"  - Invalid structures (not saved): {total_invalid}")
        print(f"  - Failed (assemble/error): {total_failed}")
        print(f"  - Structural validity rate: {validity_rate:.2f}%")
        
        if total_invalid > 0:
            print(f"\n[VALIDITY CHECK BREAKDOWN]")
            print(f"  Total invalid samples: {total_invalid}")
            if save_invalid:
                print(f"  - Volume check failed: {validity_stats['vol_failed']} ({validity_stats['vol_failed']/total_invalid*100:.2f}%) → saved to vol_invalid/ (.traj + .pt)")
                print(f"  - Distance check failed: {validity_stats['dist_failed']} ({validity_stats['dist_failed']/total_invalid*100:.2f}%) → saved to dist_invalid/ (.traj + .pt)")
                print(f"  - Height check failed: {validity_stats['height_failed']} ({validity_stats['height_failed']/total_invalid*100:.2f}%) → saved to height_invalid/ (.traj + .pt)")
                print(f"  - Assembly failed: {validity_stats['assemble_failed']} ({validity_stats['assemble_failed']/total_invalid*100:.2f}%) → saved to assemble_failed/ (.pt only)")
            else:
                print(f"  - Volume check failed: {validity_stats['vol_failed']} ({validity_stats['vol_failed']/total_invalid*100:.2f}%)")
                print(f"  - Distance check failed: {validity_stats['dist_failed']} ({validity_stats['dist_failed']/total_invalid*100:.2f}%)")
                print(f"  - Height check failed: {validity_stats['height_failed']} ({validity_stats['height_failed']/total_invalid*100:.2f}%)")
                print(f"  - Assembly failed: {validity_stats['assemble_failed']} ({validity_stats['assemble_failed']/total_invalid*100:.2f}%)")
        
        print(f"\n[OUTPUT]")
        print(f"  - Output directory: {output_path}")
        if world_size > 1:
            print(f"  - Rank: {rank}/{world_size}")
        print("=" * 50)
        
        # [NEW] Save stats.json here, ensuring it runs for both single-GPU/DP and DDP rank 0
        stats_file_path = output_path / "stats.json"
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        try:
            with open(stats_file_path, 'w') as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            print(f"[INFO] Statistics saved to: {stats_file_path}")
        except Exception as e:
            print(f"[WARNING] Failed to save statistics json: {e}")

    else:
        print(f"\n[Rank {rank}] Saved {total_valid} valid samples out of {total_processed} processed ({saved_rate:.2f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Load checkpoint, generate samples, and save only structurally valid samples."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--val_lmdb_path",
        type=str,
        required=True,
        help="Path to validation LMDB file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="unrelaxed_samples/de_novo_generation",
        help="Output directory to save traj files",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate per input",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=100,
        help="Number of sampling steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (None for all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--use_ddp",
        action="store_true",
        help="Use DistributedDataParallel (requires torchrun). If False, uses DataParallel for multi-GPU.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (for DataParallel). If None, uses all available GPUs.",
    )
    parser.add_argument(
        "--save_trajectory",
        action="store_true",
        help="If set, save full sampling trajectory (all steps) for each sample",
    )
    parser.add_argument(
        "--save_invalid",
        action="store_true",
        help="If set, save invalid samples to subdirectories (vol_invalid/, dist_invalid/, etc.) as .traj and .pt files",
    )
    
    args = parser.parse_args()
    
    use_ddp = args.use_ddp
    rank = 0
    world_size = 1
    local_rank = 0
    
    if use_ddp:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(backend="nccl")
            print(f"[INFO] DDP initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        else:
            print("[WARNING] --use_ddp specified but not running with torchrun. Falling back to DataParallel.")
            use_ddp = False
    
    model = load_model(args.checkpoint, device=args.device, use_ddp=use_ddp, local_rank=local_rank)
    
    if use_ddp and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        print(f"[INFO] Model wrapped with DDP on rank {rank} (local_rank {local_rank})")
    
    dataloader = load_val_dataloader(
        val_lmdb_path=args.val_lmdb_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    sampler = None
    if use_ddp and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
        )
    
    results = generate_samples_and_save_valid(
        model=model,
        dataloader=dataloader,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        device=args.device,
        max_batches=args.max_batches,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        sampler=sampler,
        save_trajectory=args.save_trajectory,
        save_invalid=args.save_invalid,
    )
    
    if use_ddp and world_size > 1:
        import torch.distributed as dist
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, results)
        
        if rank == 0:
            total_processed_all = sum(r["total_processed"] for r in gathered_results)
            total_valid_all = sum(r["total_valid"] for r in gathered_results)
            total_invalid_all = sum(r["total_invalid"] for r in gathered_results)
            total_failed_all = sum(r["total_failed"] for r in gathered_results)
            
            validity_stats_all = {
                'vol_failed': sum(r["validity_stats"]['vol_failed'] for r in gathered_results),
                'dist_failed': sum(r["validity_stats"]['dist_failed'] for r in gathered_results),
                'height_failed': sum(r["validity_stats"]['height_failed'] for r in gathered_results),
                'assemble_failed': sum(r["validity_stats"]['assemble_failed'] for r in gathered_results),
            }
            
            saved_rate_all = total_valid_all / total_processed_all * 100 if total_processed_all > 0 else 0.0
            validity_rate_all = total_valid_all / total_processed_all * 100 if total_processed_all > 0 else 0.0
            
            print("\n" + "=" * 50)
            print("FINAL AGGREGATED RESULTS (All Ranks)")
            print("=" * 50)
            print(f"Total processed (all ranks): {total_processed_all}")
            print(f"  - Total saved: {total_valid_all}")
            print(f"  - Validity rate: {validity_rate_all:.2f}%")
            
            # Note: We don't save stats.json here for the aggregated results to avoid conflict with the per-rank file 
            # saved inside the function. If needed, save to 'stats_aggregated.json'.
            
            print(f"\n[OUTPUT]")
            print(f"  - Output directory: {Path(args.output_dir)}")
            print("=" * 50)
        
        torch.distributed.destroy_process_group()
    
    return results

if __name__ == "__main__":
    main()