import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

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
    if "weights_only" in kwargs:
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    try:
        return _original_torch_load(*args, **kwargs, weights_only=False)
    except TypeError:
        return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

def load_model(checkpoint_path: str, device: str = "cuda", use_ddp: bool = False, local_rank: int = 0) -> EffCatModule:
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    model = EffCatModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        if use_ddp:
            model = model.cuda(local_rank)
            print(f"[INFO] Model moved to GPU {local_rank} for DDP")
        else:
            model = model.cuda()
            print(f"[INFO] Moving model to GPU: {torch.cuda.get_device_name(0)}")
    return model

def load_val_dataloader(val_lmdb_path: str, batch_size: int = 4, num_workers: int = 4, preload_to_ram: bool = True):
    from omegaconf import OmegaConf
    print(f"[INFO] Loading validation data: {val_lmdb_path}")
    batch_size_cfg = OmegaConf.create({"train": batch_size, "val": batch_size, "test": batch_size})
    num_workers_cfg = OmegaConf.create({"train": num_workers, "val": num_workers, "test": num_workers})
    datamodule = LMDBDataModule(
        train_lmdb_path=val_lmdb_path, val_lmdb_path=val_lmdb_path, test_lmdb_path=None,
        batch_size=batch_size_cfg, num_workers=num_workers_cfg, preload_to_ram=preload_to_ram,
    )
    datamodule.setup(stage="fit")
    return datamodule.val_dataloader()

def generate_samples_and_save_valid(
    model: EffCatModule, dataloader, output_dir: str, num_samples: int = 1,
    sampling_steps: int = 100, device: str = "cuda", rank: int = 0, 
    world_size: int = 1, local_rank: int = 0, sampler: Optional[Any] = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_processed, total_valid, total_failed = 0, 0, 0
    cumulative_samples_processed = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Rank {rank}", disable=(rank != 0))):
        current_batch_size = batch["ref_prim_slab_element"].shape[0]
        if sampler is not None and world_size > 1:
            original_indices = [rank + (cumulative_samples_processed + i) * world_size for i in range(current_batch_size)]
        else:
            original_indices = [cumulative_samples_processed + i for i in range(current_batch_size)]
        
        cumulative_samples_processed += current_batch_size
        if device == "cuda":
            batch = {k: v.cuda(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        try:
            with torch.no_grad():
                m = model.module if hasattr(model, "module") else model
                out = m(batch, num_sampling_steps=sampling_steps, multiplicity_flow_sample=num_samples)

            # --- 데이터 추출 및 넘파이 변환 ---
            sampled_prim_coords = out["sampled_prim_slab_coords"].cpu().numpy()
            sampled_ads_coords = out["sampled_ads_coords"].cpu().numpy()
            sampled_lattices = out["sampled_lattice"].cpu().numpy()
            sampled_supercell_matrices = out["sampled_supercell_matrix"].cpu().numpy()
            sampled_scaling_factors = out["sampled_scaling_factor"].cpu().numpy()
            
            if "sampled_prim_slab_element" in out:
                prim_slab_atom_types = out["sampled_prim_slab_element"].cpu().numpy()
            else:
                prim_slab_atom_types = batch["ref_prim_slab_element"].cpu().numpy()
                
            ads_atom_types = batch["ref_ads_element"].cpu().numpy()
            prim_slab_atom_mask = batch["prim_slab_atom_pad_mask"].cpu().numpy()
            ads_atom_mask = batch["ads_atom_pad_mask"].cpu().numpy()

            for s_idx in range(sampled_prim_coords.shape[0]):
                item_idx = s_idx // num_samples
                orig_idx = original_indices[item_idx]
                
                try:
                    recon_system, _ = assemble(
                        generated_prim_slab_coords=sampled_prim_coords[s_idx],
                        generated_ads_coords=sampled_ads_coords[s_idx],
                        generated_lattice=sampled_lattices[s_idx],
                        generated_supercell_matrix=sampled_supercell_matrices[s_idx].reshape(3,3),
                        generated_scaling_factor=float(sampled_scaling_factors[s_idx]),
                        prim_slab_atom_types=prim_slab_atom_types[s_idx] if prim_slab_atom_types.ndim > 1 else prim_slab_atom_types[item_idx],
                        ads_atom_types=ads_atom_types[item_idx],
                        prim_slab_atom_mask=prim_slab_atom_mask[item_idx],
                        ads_atom_mask=ads_atom_mask[item_idx],
                    )

                    sc_matrix_3x3 = sampled_supercell_matrices[s_idx].reshape(3, 3)
                    validity_task = (
                        sampled_prim_coords[s_idx:s_idx+1],
                        sampled_ads_coords[s_idx:s_idx+1],
                        sampled_lattices[s_idx:s_idx+1],
                        sc_matrix_3x3[np.newaxis, :, :],
                        sampled_scaling_factors[s_idx:s_idx+1],
                        prim_slab_atom_types[s_idx] if prim_slab_atom_types.ndim > 1 else prim_slab_atom_types[item_idx],
                        ads_atom_types[item_idx],
                        prim_slab_atom_mask[item_idx],
                        ads_atom_mask[item_idx],
                    )
                    
                    validity_results, _ = compute_structural_validity_single(validity_task, return_details=True)
                    is_valid = validity_results[0]

                    if is_valid:
                        recon_system.center(vacuum=10.0, axis=2)
                        traj_path = output_path / f"{orig_idx}.traj"
                        ase_write(str(traj_path), recon_system)
                        total_valid += 1
                    
                except Exception as e:
                    if rank == 0: print(f"[ERROR] Sample {orig_idx} assemble/validity failed: {e}")
                    total_failed += 1
                
                total_processed += 1
                
        except Exception as e:
            if rank == 0: print(f"[ERROR] Batch {batch_idx} failed: {e}")

    print(f"\n[Rank {rank}] Finished: Processed={total_processed}, Saved={total_valid}, Failed={total_failed}")
    return {"total_processed": total_processed, "total_valid": total_valid}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_lmdb_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--use_ddp", action="store_true")
    args = parser.parse_args()

    if args.use_ddp:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0

    model = load_model(args.checkpoint, use_ddp=args.use_ddp, local_rank=local_rank)
    if args.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    dataloader = load_val_dataloader(args.val_lmdb_path, args.batch_size, args.num_workers)
    sampler = None
    if args.use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=dataloader.collate_fn)

    generate_samples_and_save_valid(model, dataloader, args.output_dir, args.num_samples, args.sampling_steps, rank=rank, world_size=world_size, local_rank=local_rank, sampler=sampler)
    
    if args.use_ddp:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()