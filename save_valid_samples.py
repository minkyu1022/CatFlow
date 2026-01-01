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
    
    Returns:
        Statistics dictionary
    """
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # For num_samples > 1, create subdirectories for each sample index
    sample_dirs = None
    if num_samples > 1:
        sample_dirs = []
        for i in range(num_samples):
            sample_dir = output_path / str(i)
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_dirs.append(sample_dir)
        if rank == 0:
            print(f"[INFO] Output directory: {output_path}")
            print(f"[INFO] Created {num_samples} sample subdirectories: 0/ to {num_samples-1}/")
    else:
        if rank == 0:
            print(f"[INFO] Output directory: {output_path}")
    
    # Statistics
    total_processed = 0
    total_valid = 0
    total_invalid = 0
    total_failed = 0
    
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
    # Note: For DDP, DistributedSampler already distributes batches across ranks
    # We need to track the original dataset indices for each batch
    cumulative_samples_processed = 0  # Track cumulative count to handle variable batch sizes
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing batches (rank {rank})", disable=(rank != 0))):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        # Calculate original dataset indices for this batch
        current_batch_size = batch["ref_prim_slab_element"].shape[0]
        original_indices = []
        
        if sampler is not None and world_size > 1:
            # For DistributedSampler, get the actual indices it would return
            # DistributedSampler distributes indices as: rank + i * num_replicas
            # where i = 0, 1, 2, ... (sequential within each rank)
            for i in range(current_batch_size):
                # Calculate which sample in the sampler's sequence this is (0-indexed within this rank)
                sample_in_rank_sequence = cumulative_samples_processed + i
                # DistributedSampler formula: rank + sample_in_rank_sequence * world_size
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
            # For DataParallel or DDP, use appropriate device
            if isinstance(model, torch.nn.DataParallel):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            elif world_size > 1:
                # DDP: each process uses its own GPU (use local_rank)
                batch = {k: v.cuda(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            else:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        try:
            # Generate samples using forward (like validation_step)
            with torch.no_grad():
                # For DataParallel, unwrap the model for forward call
                if isinstance(model, torch.nn.DataParallel):
                    out = model.module(
                        batch,
                        num_sampling_steps=sampling_steps,
                        center_during_sampling=False,
                        multiplicity_flow_sample=num_samples,
                    )
                else:
                    out = model(
                        batch,
                        num_sampling_steps=sampling_steps,
                        center_during_sampling=False,
                        multiplicity_flow_sample=num_samples,
                    )
            
            # Convert to numpy
            def to_numpy(x):
                if isinstance(x, torch.Tensor):
                    return x.cpu().numpy()
                return x
            
            sampled_prim_slab_coords = to_numpy(out["sampled_prim_slab_coords"])  # (B*M, N, 3)
            sampled_ads_coords = to_numpy(out["sampled_ads_coords"])  # (B*M, A, 3)
            sampled_lattices = to_numpy(out["sampled_lattice"])  # (B*M, 6)
            sampled_supercell_matrices = to_numpy(out["sampled_supercell_matrix"])  # (B*M, 3, 3) or (B*M, 9)
            sampled_scaling_factors = to_numpy(out["sampled_scaling_factor"])  # (B*M,)
            
            # Check if model is in dng mode and uses generated prim_slab_atom_types
            if isinstance(model, torch.nn.DataParallel):
                model_dng = model.module.dng
            else:
                model_dng = model.dng
            
            if model_dng and "sampled_prim_slab_element" in out:
                # dng=True: use model-generated prim_slab_atom_types
                prim_slab_atom_types = to_numpy(out["sampled_prim_slab_element"])  # (B*M, N)
            else:
                # dng=False: use reference from batch
                prim_slab_atom_types = to_numpy(batch["ref_prim_slab_element"])  # (B, N)
            
            ads_atom_types = to_numpy(batch["ref_ads_element"])  # (B, A)
            prim_slab_atom_mask = to_numpy(batch["prim_slab_atom_pad_mask"])  # (B, N)
            ads_atom_mask = to_numpy(batch["ads_atom_pad_mask"])  # (B, A)
            
            # Get tags if available
            tags = batch.get("tags", None)
            if tags is not None:
                tags = to_numpy(tags)  # (B,)
            
            # Process each sample
            total_samples_in_batch = sampled_prim_slab_coords.shape[0]
            
            for sample_idx in range(total_samples_in_batch):
                # Calculate original batch index
                batch_item_idx = sample_idx // num_samples
                
                # Get original dataset index for this batch item
                original_dataset_idx = original_indices[batch_item_idx]
                
                # Get sample index within this input (0 to num_samples-1)
                sample_in_item = sample_idx % num_samples
                
                # Prepare data for this sample
                sample_prim_slab_coords = sampled_prim_slab_coords[sample_idx]  # (N, 3)
                sample_ads_coords = sampled_ads_coords[sample_idx]  # (A, 3)
                sample_lattice = sampled_lattices[sample_idx]  # (6,)
                sample_supercell_matrix = sampled_supercell_matrices[sample_idx]  # (3, 3) or (9,)
                sample_scaling_factor = sampled_scaling_factors[sample_idx]  # scalar
                
                # prim_slab_atom_types shape depends on dng mode
                if model_dng and "sampled_prim_slab_element" in out:
                    # dng=True: (B*M, N) - use sample_idx
                    prim_slab_types = prim_slab_atom_types[sample_idx]  # (N,)
                else:
                    # dng=False: (B, N) - use batch_item_idx
                    prim_slab_types = prim_slab_atom_types[batch_item_idx]  # (N,)
                
                ads_types = ads_atom_types[batch_item_idx]  # (A,)
                prim_slab_mask = prim_slab_atom_mask[batch_item_idx]  # (N,)
                ads_mask = ads_atom_mask[batch_item_idx]  # (A,)
                
                try:
                    # Assemble the structure
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
                    
                    # Check structural validity
                    # Prepare task for compute_structural_validity_single
                    # Reshape supercell_matrix to (3, 3) first, then add sample dimension
                    sc_matrix_3x3 = sample_supercell_matrix.reshape(3, 3)
                    validity_task = (
                        sample_prim_slab_coords[np.newaxis, :],  # (1, N, 3) - add sample dimension
                        sample_ads_coords[np.newaxis, :],  # (1, A, 3)
                        sample_lattice[np.newaxis, :],  # (1, 6)
                        sc_matrix_3x3[np.newaxis, :, :],  # (1, 3, 3)
                        np.array([sample_scaling_factor]),  # (1,)
                        prim_slab_types,  # (N,)
                        ads_types,  # (A,)
                        prim_slab_mask,  # (N,)
                        ads_mask,  # (A,)
                    )
                    
                    # Check validity (returns list of bools, one per sample)
                    validity_results = compute_structural_validity_single(validity_task)
                    is_valid = validity_results[0]
                    
                    if is_valid:
                        # Save valid structure with tag information preserved
                        # For num_samples > 1, save to subdirectory based on sample_in_item
                        # For num_samples = 1, save directly to output_path
                        if num_samples > 1:
                            # Save to: output_dir/{sample_in_item}/{original_dataset_idx}.traj
                            traj_path = sample_dirs[sample_in_item] / f"{original_dataset_idx}.traj"
                        else:
                            # Save to: output_dir/{original_dataset_idx}.traj
                            traj_path = output_path / f"{original_dataset_idx}.traj"
                        ase_write(str(traj_path), recon_system, format="traj")
                        total_valid += 1
                    else:
                        total_invalid += 1
                    
                    total_processed += 1
                    
                except Exception as e:
                    print(f"\n[WARNING] Failed to process sample (batch {batch_idx}, sample {sample_idx}, original_idx {original_dataset_idx}, sample_in_item {sample_in_item}): {e}")
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
        print(f"\n[OUTPUT]")
        print(f"  - Output directory: {output_path}")
        if world_size > 1:
            print(f"  - Rank: {rank}/{world_size}")
        print("=" * 50)
    else:
        # For non-zero ranks, print minimal summary
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
    
    args = parser.parse_args()
    
    # Check for DDP environment variables
    use_ddp = args.use_ddp
    rank = 0
    world_size = 1
    local_rank = 0
    
    if use_ddp:
        # Check if running with torchrun
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
    
    # Load model (pass local_rank for DDP)
    model = load_model(args.checkpoint, device=args.device, use_ddp=use_ddp, local_rank=local_rank)
    
    # For DDP, wrap model
    if use_ddp and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        print(f"[INFO] Model wrapped with DDP on rank {rank} (local_rank {local_rank})")
    
    # Load dataloader
    dataloader = load_val_dataloader(
        val_lmdb_path=args.val_lmdb_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # For DDP, wrap dataloader with DistributedSampler
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
    
    # Generate samples and save valid ones
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
    )
    
    # For DDP, aggregate results from all ranks
    if use_ddp and world_size > 1:
        # Gather results from all ranks
        import torch.distributed as dist
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, results)
        
        if rank == 0:
            # Aggregate statistics
            total_processed_all = sum(r["total_processed"] for r in gathered_results)
            total_valid_all = sum(r["total_valid"] for r in gathered_results)
            total_invalid_all = sum(r["total_invalid"] for r in gathered_results)
            total_failed_all = sum(r["total_failed"] for r in gathered_results)
            
            saved_rate_all = total_valid_all / total_processed_all * 100 if total_processed_all > 0 else 0.0
            validity_rate_all = total_valid_all / total_processed_all * 100 if total_processed_all > 0 else 0.0
            
            print("\n" + "=" * 50)
            print("FINAL AGGREGATED RESULTS (All Ranks)")
            print("=" * 50)
            print(f"Total processed (all ranks): {total_processed_all}")
            print(f"\n[FINAL RESULTS]")
            print(f"  - Total saved samples (valid structures): {total_valid_all}")
            print(f"  - Total saved sample rate: {saved_rate_all:.2f}% ({total_valid_all}/{total_processed_all})")
            print(f"\n[BREAKDOWN]")
            print(f"  - Valid structures saved: {total_valid_all}")
            print(f"  - Invalid structures (not saved): {total_invalid_all}")
            print(f"  - Failed (assemble/error): {total_failed_all}")
            print(f"  - Structural validity rate: {validity_rate_all:.2f}%")
            print(f"\n[OUTPUT]")
            print(f"  - Output directory: {Path(args.output_dir)}")
            print(f"  - Number of GPUs used: {world_size}")
            print("=" * 50)
        
        torch.distributed.destroy_process_group()
    elif not use_ddp:
        # Single GPU or DataParallel - results already printed
        pass
    
    return results


if __name__ == "__main__":
    main()

