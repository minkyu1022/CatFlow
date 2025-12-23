"""
Script to load model checkpoint and sample structures from validation dataset, saving them as CIF files.

Usage:
    python scripts/eval.py \
        --checkpoint path/to/checkpoint.ckpt \
        --val_lmdb_path path/to/val/dataset.lmdb \
        --output_dir outputs/generated_structures \
        --num_samples 1 \
        --sampling_steps 100 \
        --batch_size 4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.io import write as ase_write

from src.module.effcat_module import EffCatModule
from src.data.datamodule import LMDBDataModule
from src.data.lmdb_dataset import collate_fn_with_dynamic_padding
from src.models.loss.validation import _structural_validity
from scripts.refine_sc_mat import refine_sc_mat_batch

from scripts.assemble import assemble, assemble_batch


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


def load_model(checkpoint_path: str, device: str = "cuda") -> EffCatModule:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on ("cuda" or "cpu")
    
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
        use_pyg=False,
    )
    
    datamodule.setup(stage="fit")
    dataloader = datamodule.val_dataloader()
    
    print(f"[INFO] Loaded {len(datamodule.val_dataset)} samples in total")
    
    return dataloader


def predict_and_assemble(
    model: EffCatModule,
    batch: Dict[str, torch.Tensor],
    num_samples: int = 1,
    sampling_steps: int = 100,
    center_during_sampling: bool = False,
    refine_final: bool = False,
    device: str = "cuda",
) -> Tuple[List[Optional[Atoms]], Dict[str, Any], List[int]]:
    """
    Predict structures using model and assemble into Atoms objects.
    
    Args:
        model: EffCatModule model
        batch: Batch data
        num_samples: Number of samples to generate per input
        sampling_steps: Number of sampling steps
        center_during_sampling: Whether to center coordinates during sampling
        refine_final: Whether to refine final results
        device: Device to use
    
    Returns:
        Tuple of (List of generated Atoms objects, prediction_output dictionary, error_indices)
    """
    # Move batch to device
    if device == "cuda" and torch.cuda.is_available():
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    # Set predict_args
    model.predict_args = {
        "num_samples": num_samples,
        "sampling_steps": sampling_steps,
        "sampling_center_coords": False,
        "refine_final": refine_final,
    }
    
    # Run prediction
    with torch.no_grad():
        prediction_output = model.predict_step(batch, batch_idx=0)
    
    if prediction_output.get("exception", False):
        print("[WARNING] Exception occurred during prediction")
        return [], {}, []
    
    # Refine supercell matrices to avoid singular matrices
    pred_matrices = prediction_output["generated_supercell_matrices"]
    refined_matrices = refine_sc_mat_batch(pred_matrices)
    prediction_output["generated_supercell_matrices"] = refined_matrices
    
    # Assemble structures and get error indices
    atoms_list, error_indices = assemble_batch(
        prediction_output, 
        num_samples=num_samples,
        return_error_indices=True
    )
    
    return atoms_list, prediction_output, error_indices


def save_atoms_to_cif(
    atoms: Atoms,
    output_path: str,
) -> None:
    """
    Save Atoms object to CIF file.
    
    Args:
        atoms: ASE Atoms object
        output_path: Path to save CIF file
    """
    ase_write(output_path, atoms, format="cif")


def save_atoms_to_traj(
    atoms: Atoms,
    output_path: str,
) -> None:
    """
    Save Atoms object to traj file (preserves tag information).
    
    Args:
        atoms: ASE Atoms object
        output_path: Path to save traj file
    """
    ase_write(output_path, atoms, format="traj")


def save_prediction_metadata_to_json(
    prediction_output: Dict[str, Any],
    sample_idx: int,
    num_samples: int,
    error_sample_indices: List[int],
    output_path: str,
) -> None:
    """
    Save prediction metadata (supercell matrices, ads coords, scaling factors) to JSON file.
    Only saves metadata for samples that had errors during assembly.
    
    Args:
        prediction_output: Prediction output dictionary from model
        sample_idx: Starting index for this batch
        num_samples: Number of samples per input
        error_sample_indices: List of local sample indices (within batch) that had errors
        output_path: Path to save JSON file
    """
    if not error_sample_indices:
        # No errors, don't save anything
        return
    
    # Extract batch size
    batch_size = prediction_output["generated_supercell_matrices"].shape[0] // num_samples
    
    metadata_list = []
    
    # Convert to list for JSON serialization
    def to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, (list, tuple)):
            return list(x)
        else:
            return x
    
    # Only process samples that had errors
    for local_sample_idx in error_sample_indices:
        # Calculate batch_item_idx and sample_in_item_idx from local_sample_idx
        batch_item_idx = local_sample_idx // num_samples
        sample_in_item_idx = local_sample_idx % num_samples
        global_sample_idx = sample_idx + local_sample_idx
        idx_in_tensor = local_sample_idx
        
        # Extract generated values
        gen_supercell_matrix = prediction_output["generated_supercell_matrices"][idx_in_tensor]
        gen_ads_coords = prediction_output["generated_ads_coords"][idx_in_tensor]
        gen_scaling_factor = prediction_output["generated_scaling_factors"][idx_in_tensor]
        
        # Convert to numpy and then to list for JSON serialization
        if isinstance(gen_supercell_matrix, torch.Tensor):
            gen_supercell_matrix = gen_supercell_matrix.cpu().numpy()
        if isinstance(gen_ads_coords, torch.Tensor):
            gen_ads_coords = gen_ads_coords.cpu().numpy()
        if isinstance(gen_scaling_factor, torch.Tensor):
            gen_scaling_factor = gen_scaling_factor.cpu().item()
        
        metadata = {
            "sample_idx": int(global_sample_idx),
            "batch_item_idx": int(batch_item_idx),
            "sample_in_item_idx": int(sample_in_item_idx),
            "local_sample_idx": int(local_sample_idx),
            "generated_supercell_matrix": to_list(gen_supercell_matrix),
            "generated_ads_coords": to_list(gen_ads_coords),
            "generated_scaling_factor": float(gen_scaling_factor),
        }
        
        # Extract true values if available
        if "true_supercells" in prediction_output:
            true_supercell_matrix = prediction_output["true_supercells"][batch_item_idx]
            if isinstance(true_supercell_matrix, torch.Tensor):
                true_supercell_matrix = true_supercell_matrix.cpu().numpy()
            metadata["true_supercell_matrix"] = to_list(true_supercell_matrix)
        
        if "true_ads_coords" in prediction_output:
            true_ads_coords = prediction_output["true_ads_coords"][batch_item_idx]
            if isinstance(true_ads_coords, torch.Tensor):
                true_ads_coords = true_ads_coords.cpu().numpy()
            metadata["true_ads_coords"] = to_list(true_ads_coords)
        
        if "true_scaling_factors" in prediction_output:
            true_scaling_factor = prediction_output["true_scaling_factors"][batch_item_idx]
            if isinstance(true_scaling_factor, torch.Tensor):
                true_scaling_factor = true_scaling_factor.cpu().item()
            metadata["true_scaling_factor"] = float(true_scaling_factor)
        
        metadata_list.append(metadata)
    
    # Save to JSON file only if there are errors
    if metadata_list:
        with open(output_path, "w") as f:
            json.dump(metadata_list, f, indent=2)


def run_evaluation(
    checkpoint_path: str,
    val_lmdb_path: str,
    output_dir: str,
    num_samples: int = 1,
    sampling_steps: int = 100,
    batch_size: int = 4,
    num_workers: int = 4,
    max_batches: Optional[int] = None,
    center_during_sampling: bool = False,
    refine_final: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline.
    
    Args:
        checkpoint_path: Path to checkpoint file
        val_lmdb_path: Path to validation LMDB file
        output_dir: Path to output directory
        num_samples: Number of samples to generate per input
        sampling_steps: Number of sampling steps
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_batches: Maximum number of batches to process (None for all)
        center_during_sampling: Whether to center coordinates during sampling
        refine_final: Whether to refine final results
        device: Device to use
    
    Returns:
        Evaluation results dictionary
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_path}")
    
    # Load model
    model = load_model(checkpoint_path, device=device)
    
    # Load dataloader
    dataloader = load_val_dataloader(
        val_lmdb_path=val_lmdb_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Statistics variables
    total_generated = 0
    total_failed = 0
    total_invalid = 0  # Structurally invalid samples
    sample_idx = 0
    
    print(f"\n[INFO] Starting sampling")
    print(f"  - num_samples: {num_samples}")
    print(f"  - sampling_steps: {sampling_steps}")
    print(f"  - batch_size: {batch_size}")
    
    # Iterate over batches
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        try:
            # Predict and assemble
            atoms_list, prediction_output, error_indices = predict_and_assemble(
                model=model,
                batch=batch,
                num_samples=num_samples,
                sampling_steps=sampling_steps,
                center_during_sampling=False,
                refine_final=refine_final,
                device=device,
            )
            
            # Save prediction metadata to JSON (only for samples with errors)
            if prediction_output and error_indices:
                json_path = output_path / f"metadata_batch_{batch_idx:06d}.json"
                save_prediction_metadata_to_json(
                    prediction_output=prediction_output,
                    sample_idx=sample_idx,
                    num_samples=num_samples,
                    error_sample_indices=error_indices,
                    output_path=str(json_path),
                )
            
            # Save to traj files (preserves tag information)
            # Only save structurally valid samples
            for atoms in atoms_list:
                if atoms is not None:
                    # Handle case where atoms is a tuple (e.g. (atoms, slab))
                    if isinstance(atoms, tuple):
                        atoms = atoms[0]

                    # Check structural validity before saving
                    try:
                        is_valid = _structural_validity(atoms)
                    except Exception as e:
                        print(f"\n[WARNING] Structural validity check failed for sample {sample_idx}: {e}")
                        is_valid = False
                    
                    if is_valid:
                        traj_path = output_path / f"structure_{sample_idx:06d}.traj"
                        save_atoms_to_traj(atoms, str(traj_path))
                        total_generated += 1
                    else:
                        total_invalid += 1
                else:
                    total_failed += 1
                sample_idx += 1
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n[WARNING] Out of memory at batch {batch_idx}. Skipping.")
                torch.cuda.empty_cache()
                total_failed += batch_size * num_samples
                sample_idx += batch_size * num_samples
                continue
            else:
                raise e
    
    # Summary of results
    total_processed = total_generated + total_failed + total_invalid
    validity_rate = total_generated / total_processed * 100 if total_processed > 0 else 0.0
    
    results = {
        "total_generated": total_generated,
        "total_failed": total_failed,
        "total_invalid": total_invalid,
        "total_processed": total_processed,
        "validity_rate": validity_rate,
        "output_dir": str(output_path),
    }
    
    print("\n" + "=" * 50)
    print("Evaluation Results Summary")
    print("=" * 50)
    print(f"Total processed: {total_processed}")
    print(f"Generated (valid) structures: {total_generated}")
    print(f"Structurally invalid: {total_invalid}")
    print(f"Failed (assemble error): {total_failed}")
    print(f"Structural validity rate: {validity_rate:.2f}%")
    print(f"Output directory: {output_path}")
    print("=" * 50)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Load model checkpoint and sample structures from validation dataset, saving them as CIF files."
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
        default="outputs/generated_structures",
        help="Output directory to save CIF files",
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
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (None for all)",
    )
    parser.add_argument(
        "--refine_final",
        action="store_true",
        help="Enable final result refinement",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_evaluation(
        checkpoint_path=args.checkpoint,
        val_lmdb_path=args.val_lmdb_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        center_during_sampling=False,
        refine_final=args.refine_final,
        device=args.device,
    )
    
    return results


if __name__ == "__main__":
    main()
