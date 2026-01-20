import os
import gc
import json
import time
import random
import multiprocessing as mp
from typing import Any, Optional

import torch
import numpy as np
from einops import rearrange
from lightning import Callback, LightningModule
from lightning.pytorch.utilities import rank_zero_info
from torch import Tensor, nn
from torchmetrics import MeanMetric
from pebble import ProcessPool
from concurrent.futures import TimeoutError as PebbleTimeoutError
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

from src.data.prior import CatPriorSampler
from src.models.loss.validation import (
    find_best_match_rmsd_prim,
    find_best_match_rmsd_slab,
    compute_structural_validity_single,
    compute_prim_structural_validity_single,
)
from src.models.loss.utils import stratify_loss_by_time
from src.module.flow import AtomFlowMatching


class EffCatModule(LightningModule):
    def __init__(
        self,
        atom_s: int,
        token_s: int,
        flow_model_args: dict[str, Any],
        training_args: dict[str, Any],
        validation_args: dict[str, Any],
        flow_process_args: dict[str, Any],
        prior_sampler_args: dict[str, Any],
        predict_args: Optional[dict[str, Any]] = None,
        use_kernels: bool = False,
        dng: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Args
        self.training_args = training_args
        self.validation_args = validation_args
        self.predict_args = predict_args
        
        # Use dng from flow_model_args if not explicitly provided at top level
        # This ensures consistency: flow_model_args.dng is the source of truth
        if "dng" in flow_model_args:
            dng = flow_model_args["dng"]
        
        self.dng = dng

        # Kernels
        self.use_kernels = use_kernels

        # Validation metrics
        self.matcher_kwargs = StructureMatcher(
            **self.validation_args["structure_matcher_args"]
        ).as_dict()
        self.val_prim_match_rate = MeanMetric(sync_on_compute=False)
        self.val_prim_rmsd = MeanMetric(sync_on_compute=False)
        self.val_slab_match_rate = MeanMetric(sync_on_compute=False)
        self.val_slab_rmsd = MeanMetric(sync_on_compute=False)
        self.val_structural_validity = MeanMetric(sync_on_compute=False)
        self.val_prim_structural_validity = MeanMetric(sync_on_compute=False)
        self.validation_step_outputs = []

        # Prior sampler for flow matching
        prior_sampler = CatPriorSampler(**prior_sampler_args)

        # Flow matching module
        self.structure_module = AtomFlowMatching(
            flow_model_args={
                "atom_s": atom_s,
                "token_s": token_s,
                **flow_model_args,
            },
            prior_sampler=prior_sampler,
            dng=dng,
            **flow_process_args,
        )
        
        # Load and cache histogram JSON file when dng=True
        if self.dng:
            histogram_path = self.validation_args.get("n_prim_slab_atoms_histogram_path", "primitive_atom_distribution.json")
            # Handle relative paths relative to project root
            if not os.path.isabs(histogram_path):
                # Find project root path (relative to current file)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                histogram_path = os.path.join(project_root, histogram_path)
            
            with open(histogram_path, 'r') as f:
                histogram_data = json.load(f)
            
            # JSON structure: {"prim_slab_num_atoms_hist": [0.0, 0.0115, ...]}
            self.n_prim_slab_atoms_hist = histogram_data["prim_slab_num_atoms_hist"]
            # Index = number of atoms, value = probability
            # max_n_prim_slab_atoms is histogram length - 1 (since indices start from 0)
            self.max_n_prim_slab_atoms = len(self.n_prim_slab_atoms_hist) - 1

    def setup(self, stage: str) -> None:
        """Set the model for training, validation and inference."""
        if stage == "predict" and not (
            torch.cuda.is_available()
            and torch.cuda.get_device_properties(torch.device("cuda")).major >= 8
        ):
            self.use_kernels = False
    
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Called when checkpoint is loaded. Use this to verify optimizer state."""
        if self.global_rank == 0:  # Only log on rank 0
            if "optimizer_states" in checkpoint:
                opt_states = checkpoint["optimizer_states"]
                if len(opt_states) > 0:
                    opt_state = opt_states[0]
                    if "state" in opt_state and len(opt_state["state"]) > 0:
                        rank_zero_info(
                            f"| [CHECKPOINT] Optimizer state loaded: {len(opt_state['state'])} parameters"
                        )
                        # Check first parameter's state
                        first_state = list(opt_state["state"].values())[0]
                        if "exp_avg" in first_state:
                            rank_zero_info(
                                f"| [CHECKPOINT] AdamW momentum buffers found (exp_avg, exp_avg_sq)"
                            )
                        if "param_groups" in opt_state and len(opt_state["param_groups"]) > 0:
                            lr = opt_state["param_groups"][0].get("lr", "N/A")
                            rank_zero_info(f"| [CHECKPOINT] Learning rate from checkpoint: {lr}")
                    else:
                        rank_zero_info("| [WARNING] Optimizer state exists but is empty")
                else:
                    rank_zero_info("| [WARNING] Optimizer states list is empty")
            else:
                rank_zero_info("| [WARNING] No optimizer_states found in checkpoint - starting fresh")
            
            if "global_step" in checkpoint:
                rank_zero_info(f"| [CHECKPOINT] Resuming from global_step: {checkpoint['global_step']}")
            if "epoch" in checkpoint:
                rank_zero_info(f"| [CHECKPOINT] Resuming from epoch: {checkpoint['epoch']}")

    def forward(
            self,
            feats: dict[str, Tensor],
            # Training arguments
            multiplicity_flow_train: int = 1,
            # Sampling arguments
            multiplicity_flow_sample: int = 1,
            num_sampling_steps: Optional[int] = None,
            center_during_sampling: bool = False,
            refine_final: bool = False,
            return_trajectory: bool = False,
        ) -> dict[str, Any]:

            dict_out = {}

            if self.training:
                # Returns denoised predictions for loss computation
                dict_out.update(
                    self.structure_module(feats=feats, multiplicity=multiplicity_flow_train)
                )
            else:
                # Returns sampled structure
                # When dng=True: sample n_prim_slab_atoms from histogram and dynamically create mask
                if self.dng:
                    batch_size = feats["ref_prim_slab_element"].shape[0]
                    original_n = feats["prim_slab_cart_coords"].shape[1]  # Original number of atoms in feats
                    
                    # Sample number of atoms from histogram according to probability
                    n_atoms_indices = np.arange(len(self.n_prim_slab_atoms_hist))
                    sampled_n_atoms = np.random.choice(
                        n_atoms_indices,
                        size=batch_size * multiplicity_flow_sample,
                        p=self.n_prim_slab_atoms_hist,
                        replace=True
                    ).tolist()
                    
                    # Determine max_n_prim_slab_atoms
                    max_n = max(sampled_n_atoms) if len(sampled_n_atoms) > 0 else self.max_n_prim_slab_atoms
                    
                    prim_slab_atom_mask = torch.zeros((batch_size * multiplicity_flow_sample, max_n), dtype=torch.bool, device=self.device)
                    for i, n in enumerate(sampled_n_atoms):
                        prim_slab_atom_mask[i, :n] = True
                    
                    feats["prim_slab_atom_pad_mask"] = prim_slab_atom_mask
                    
                    feats["prim_slab_token_pad_mask"] = prim_slab_atom_mask.clone()
                    
                    # Also dynamically regenerate prim_slab_atom_to_token
                    prim_slab_atom_to_token = torch.eye(max_n, device=self.device).unsqueeze(0).expand(batch_size * multiplicity_flow_sample, -1, -1)
                    feats["prim_slab_atom_to_token"] = prim_slab_atom_to_token
                    
                    # Adjust feats to match max_n: pad or slice prim_slab-related tensors
                    if max_n != original_n:
                        # Adjust prim_slab_cart_coords (Original Batch Size B -> Padded B)
                        if max_n > original_n:
                            padding = torch.zeros((batch_size, max_n - original_n, 3), device=feats["prim_slab_cart_coords"].device, dtype=feats["prim_slab_cart_coords"].dtype)
                            feats["prim_slab_cart_coords"] = torch.cat([feats["prim_slab_cart_coords"], padding], dim=1)
                        else:
                            feats["prim_slab_cart_coords"] = feats["prim_slab_cart_coords"][:, :max_n, :]
                        
                        # Adjust ref_prim_slab_element (Original Batch Size B -> Padded B)
                        if max_n > original_n:
                            padding = torch.zeros((batch_size, max_n - original_n), device=feats["ref_prim_slab_element"].device, dtype=feats["ref_prim_slab_element"].dtype)
                            feats["ref_prim_slab_element"] = torch.cat([feats["ref_prim_slab_element"], padding], dim=1)
                        else:
                            feats["ref_prim_slab_element"] = feats["ref_prim_slab_element"][:, :max_n]
                    
                    # Expand all ads-related tensors in feats to match prim_slab batch size
                    if "ads_atom_to_token" in feats:
                        feats["ads_atom_to_token"] = feats["ads_atom_to_token"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "ads_cart_coords" in feats:
                        feats["ads_cart_coords"] = feats["ads_cart_coords"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "ads_center" in feats:
                        feats["ads_center"] = feats["ads_center"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "ads_rel_pos" in feats:
                        feats["ads_rel_pos"] = feats["ads_rel_pos"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "ads_atom_pad_mask" in feats:
                        feats["ads_atom_pad_mask"] = feats["ads_atom_pad_mask"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "ads_token_pad_mask" in feats:
                        feats["ads_token_pad_mask"] = feats["ads_token_pad_mask"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "ref_ads_element" in feats:
                        feats["ref_ads_element"] = feats["ref_ads_element"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "ref_ads_pos" in feats:
                        feats["ref_ads_pos"] = feats["ref_ads_pos"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "bind_ads_atom" in feats:
                        feats["bind_ads_atom"] = feats["bind_ads_atom"].repeat_interleave(multiplicity_flow_sample, 0)
                    
                    
                    if "prim_slab_cart_coords" in feats:
                        feats["prim_slab_cart_coords"] = feats["prim_slab_cart_coords"].repeat_interleave(multiplicity_flow_sample, 0)
                    
                    if "ref_prim_slab_element" in feats:
                        feats["ref_prim_slab_element"] = feats["ref_prim_slab_element"].repeat_interleave(multiplicity_flow_sample, 0)
                    
                    # Expand other global tensors if they exist
                    if "lattice" in feats:
                        feats["lattice"] = feats["lattice"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "supercell_matrix" in feats:
                        feats["supercell_matrix"] = feats["supercell_matrix"].repeat_interleave(multiplicity_flow_sample, 0)
                    if "scaling_factor" in feats:
                        feats["scaling_factor"] = feats["scaling_factor"].repeat_interleave(multiplicity_flow_sample, 0)
                    
                    ads_atom_mask = feats["ads_atom_pad_mask"]
                else:
                    prim_slab_atom_mask = feats["prim_slab_atom_pad_mask"]
                    ads_atom_mask = feats["ads_atom_pad_mask"]

                network_condition_kwargs = dict(
                    feats=feats,
                )

                dict_out.update(
                    self.structure_module.sample(
                        prim_slab_atom_mask=prim_slab_atom_mask,
                        ads_atom_mask=ads_atom_mask,
                        num_sampling_steps=num_sampling_steps,
                        multiplicity=multiplicity_flow_sample,
                        center_coords=False,
                        refine_final=refine_final,
                        return_trajectory=return_trajectory,
                        **network_condition_kwargs,
                    )
                )

            return dict_out

    def _compute_and_log_loss(
        self, batch: dict[str, Tensor], prefix: str
    ) -> tuple[Tensor, dict, dict]:
        # Compute the forward pass
        out = self.structure_module(
            feats=batch,
            multiplicity=self.training_args["train_multiplicity"],
        )

        # Compute losses
        flow_loss_dict, check_dict = self.structure_module.compute_loss(
            batch,
            out,
            multiplicity=self.training_args["train_multiplicity"],
            loss_type=self.training_args["loss_type"],
            flow_loss_type=self.training_args.get("flow_loss_type", "v_loss"),
        )

        # Calculate total weighted loss
        ads_center_weight = self.training_args.get("ads_center_loss_weight", 1.0)
        ads_rel_pos_weight = self.training_args.get("ads_rel_pos_loss_weight", 1.0)
        total_loss = (
            self.training_args["prim_slab_coord_loss_weight"]
            * flow_loss_dict["prim_slab_coord_loss"].mean()
            + ads_center_weight * flow_loss_dict["ads_center_loss"].mean()
            + ads_rel_pos_weight * flow_loss_dict["ads_rel_pos_loss"].mean()
            + self.training_args["length_loss_weight"]
            * flow_loss_dict["length_loss"].mean()
            + self.training_args["angle_loss_weight"]
            * flow_loss_dict["angle_loss"].mean()
            + self.training_args["supercell_matrix_loss_weight"]
            * flow_loss_dict["supercell_matrix_loss"].mean()
            + self.training_args["scaling_factor_loss_weight"]
            * flow_loss_dict["scaling_factor_loss"].mean()
        )
        
        # Add prim_slab_element_loss when dng=True
        if self.dng and "prim_slab_element_loss" in flow_loss_dict:
            total_loss = total_loss + self.training_args.get("prim_slab_element_loss_weight", 1.0) * flow_loss_dict["prim_slab_element_loss"].mean()

        # Loggings
        batch_size = batch["ref_prim_slab_element"].shape[0]
        sync_dist = prefix == "val"

        # Loss per component
        for loss_name, batch_loss in flow_loss_dict.items():
            self.log(
                f"{prefix}/{loss_name}",
                batch_loss.mean(),
                batch_size=batch_size,
                sync_dist=sync_dist,
            )
        
        # Check metrics for supercell matrix (not trained)
        for metric_name, batch_metric in check_dict.items():
            self.log(
                f"{prefix}/{metric_name}",
                batch_metric.mean(),
                batch_size=batch_size,
                sync_dist=sync_dist,
            )

        # Total loss
        self.log(
            f"{prefix}/total_loss",
            total_loss,
            batch_size=batch_size,
            sync_dist=sync_dist,
        )

        return total_loss, flow_loss_dict, out

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        step_start_time = time.time()

        total_loss, flow_loss_dict, out = self._compute_and_log_loss(
            batch, prefix="train"
        )

        # Training-specific loggings
        batch_size = batch["ref_prim_slab_element"].shape[0]
        self.log("train/batch_size", batch_size)

        # Stratified loss logging
        for loss_name, batch_loss in flow_loss_dict.items():
            # Stratified loss
            stratified_losses = stratify_loss_by_time(
                batch_t=out["times"], batch_loss=batch_loss, loss_name=loss_name
            )
            for k, v in stratified_losses.items():
                self.log(f"train/{k}", v, batch_size=batch_size)

        # Time
        step_time = time.time() - step_start_time
        self.log("train/samples_per_second", batch_size / step_time)

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        # Compute and log validation loss
        self._compute_and_log_loss(batch, prefix="val")

        # Sampling and structure matching
        sample_every_n_epochs = self.validation_args.get("sample_every_n_epochs", 1)
        if (
            sample_every_n_epochs <= 0
            or (self.current_epoch + 1) % sample_every_n_epochs != 0
        ):
            return

        n_samples = self.validation_args["flow_samples"]

        try:
            out = self(
                batch,
                num_sampling_steps=self.validation_args["sampling_steps"],
                center_during_sampling=False,
                multiplicity_flow_sample=n_samples,
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                rank_zero_info("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

        sampled_ads_coords = out["sampled_ads_coords"]  # (B*n_samples, M, 3)
        true_ads_coords = batch["ads_cart_coords"]  # (B, M, 3)
        ads_mask = batch["ads_atom_pad_mask"]  # (B, M)
        
        B = true_ads_coords.shape[0]
        M = true_ads_coords.shape[1]
        sampled_ads_coords_reshaped = sampled_ads_coords.view(B, n_samples, M, 3)
        
        true_ads_coords_expanded = true_ads_coords.unsqueeze(1)  # (B, 1, M, 3)
        ads_mask_expanded = ads_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, M, 1)
        
        mse_per_sample = ((sampled_ads_coords_reshaped - true_ads_coords_expanded) ** 2) * ads_mask_expanded
        mse_per_sample = mse_per_sample.sum(dim=(2, 3)) / (ads_mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, n_samples)
        
        ads_coord_err = mse_per_sample.mean()
        
        batch_size = batch["ref_prim_slab_element"].shape[0]
        self.log(
            "val/ads_coord_err",
            ads_coord_err,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Aggregate outputs
        return_dict = {
            "sampled_prim_slab_coords": out["sampled_prim_slab_coords"],
            "sampled_ads_coords": out["sampled_ads_coords"],
            "sampled_lattices": out["sampled_lattice"],
            "sampled_supercell_matrices": out["sampled_supercell_matrix"],
            "sampled_scaling_factors": out["sampled_scaling_factor"],
            "true_prim_slab_coords": batch["prim_slab_cart_coords"],
            "true_ads_coords": batch["ads_cart_coords"],
            "true_lattices": batch["lattice"],
            "true_supercells": batch["supercell_matrix"],
            "true_scaling_factors": batch["scaling_factor"],
            "prim_slab_atom_mask": batch["prim_slab_atom_pad_mask"],
            "ads_atom_mask": batch["ads_atom_pad_mask"],
        }
        
        # Use sampled_prim_slab_element when dng=True, ref_prim_slab_element when dng=False
        if self.dng:
            return_dict["prim_slab_atom_types"] = out.get("sampled_prim_slab_element")
        else:
            return_dict["prim_slab_atom_types"] = batch["ref_prim_slab_element"]
        
        return_dict["ads_atom_types"] = batch["ref_ads_element"]
        self.validation_step_outputs.append(return_dict)

    def on_validation_epoch_end(self) -> None:
        # NOTE: This hook is called on ALL ranks in DDP, but we only execute code on rank 0.
        # PyTorch Lightning will synchronize all ranks after this hook completes.
        if self.global_rank == 0:
            # Filter out None
            valid_outputs = [
                out for out in self.validation_step_outputs if out is not None
            ]
            if not valid_outputs:
                return

            n_samples = self.validation_args["flow_samples"]
            prim_rmsd_tasks = []
            slab_rmsd_tasks = []
            prim_validity_tasks = []
            slab_validity_tasks = []

            # Collate results from all batches
            for batch_output in valid_outputs:
                # Reshape and move to cpu (prim_slab coords)
                sampled_prim_slab_coords = (
                    rearrange(
                        batch_output["sampled_prim_slab_coords"],
                        "(b m) n c -> b m n c",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )
                sampled_ads_coords = (
                    rearrange(
                        batch_output["sampled_ads_coords"],
                        "(b m) n c -> b m n c",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )
                sampled_lattices = (
                    rearrange(
                        batch_output["sampled_lattices"],
                        "(b m) c -> b m c",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )
                sampled_supercell_matrices = (
                    rearrange(
                        batch_output["sampled_supercell_matrices"],
                        "(b m) ... -> b m ...",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )
                sampled_scaling_factors = (
                    rearrange(
                        batch_output["sampled_scaling_factors"],
                        "(b m) -> b m",
                        m=n_samples,
                    )
                    .cpu()
                    .numpy()
                )

                true_prim_slab_coords = batch_output["true_prim_slab_coords"].cpu().numpy()
                true_ads_coords = batch_output["true_ads_coords"].cpu().numpy()
                true_lattices = batch_output["true_lattices"].cpu().numpy()
                true_supercells = batch_output["true_supercells"].cpu().numpy()
                true_scaling_factors = batch_output["true_scaling_factors"].cpu().numpy()
                prim_slab_atom_types = batch_output["prim_slab_atom_types"].cpu().numpy()
                ads_atom_types = batch_output["ads_atom_types"].cpu().numpy()
                prim_slab_atom_mask = batch_output["prim_slab_atom_mask"].cpu().numpy().astype(bool)
                ads_atom_mask = batch_output["ads_atom_mask"].cpu().numpy().astype(bool)

                # Prepare tasks for each item in the batch
                batch_size = sampled_prim_slab_coords.shape[0]
                for i in range(batch_size):
                    # Add RMSD tasks only when dng=False (excluded when dng=True)
                    if not self.dng:
                        # RMSD task (primitive slab only)
                        prim_rmsd_tasks.append(
                            (
                                sampled_prim_slab_coords[i],
                                sampled_lattices[i],
                                true_prim_slab_coords[i],
                                true_lattices[i],
                                prim_slab_atom_types[i],
                                prim_slab_atom_mask[i],
                                self.matcher_kwargs,
                            )
                        )
                        
                        # Slab RMSD task (full system after assemble)
                        slab_rmsd_tasks.append(
                            (
                                sampled_prim_slab_coords[i],
                                sampled_ads_coords[i],
                                sampled_lattices[i],
                                sampled_supercell_matrices[i],
                                sampled_scaling_factors[i],
                                true_prim_slab_coords[i],
                                true_ads_coords[i],
                                true_lattices[i],
                                true_supercells[i],
                                float(true_scaling_factors[i]),
                                prim_slab_atom_types[i],
                                ads_atom_types[i],
                                prim_slab_atom_mask[i],
                                ads_atom_mask[i],
                                self.matcher_kwargs,
                            )
                        )
                    
                    # Prim structural validity task (primitive slab only)
                    prim_validity_tasks.append(
                        (
                            sampled_prim_slab_coords[i],
                            sampled_lattices[i],
                            prim_slab_atom_types[i],
                            prim_slab_atom_mask[i],
                            self.matcher_kwargs,
                        )
                    )
                    
                    # Slab structural validity task (full system after assemble)
                    slab_validity_tasks.append(
                        (
                            sampled_prim_slab_coords[i],
                            sampled_ads_coords[i],
                            sampled_lattices[i],
                            sampled_supercell_matrices[i],
                            sampled_scaling_factors[i],
                            prim_slab_atom_types[i],
                            ads_atom_types[i],
                            prim_slab_atom_mask[i],
                            ads_atom_mask[i],
                        )
                    )

            # Settings for parallel processing
            timeout_seconds = self.validation_args["timeout"]
            num_workers = self.validation_args["num_workers"]

            # Execute RMSD tasks only when dng=False (excluded when dng=True)
            if not self.dng:
                # Execute prim RMSD tasks in parallel
                rank_zero_info(f"| Computing prim RMSD for {len(prim_rmsd_tasks)} structures (workers={num_workers})...")
                prim_rmsd_results_per_item = []
                with ProcessPool(
                    max_workers=num_workers, context=mp.get_context("spawn")
                ) as pool:
                    future_map = {
                        pool.schedule(find_best_match_rmsd_prim, args=(task,)): task
                        for task in prim_rmsd_tasks
                    }

                    for future in tqdm(
                        future_map, 
                        desc="Prim RMSD matching", 
                        unit="struct",
                        total=len(prim_rmsd_tasks),
                        leave=False,
                    ):
                        try:
                            result = future.result(timeout=timeout_seconds)
                            prim_rmsd_results_per_item.append(result)
                        except PebbleTimeoutError:
                            future.cancel()
                            rank_zero_info(
                                f"| WARNING: A prim RMSD matching task timed out. Skipping."
                            )
                            prim_rmsd_results_per_item.append([None] * n_samples)
                        except Exception as e:
                            future.cancel()
                            rank_zero_info(
                                f"| WARNING: A prim RMSD task failed with an exception: {e}"
                            )
                            prim_rmsd_results_per_item.append([None] * n_samples)
                
                rank_zero_info(f"| Prim RMSD computation completed.")

                # Compute prim match rate and RMSD (primitive slab)
                for result_list in prim_rmsd_results_per_item:
                    valid_rmsds = [r for r in result_list if r is not None]
                    if valid_rmsds:
                        self.val_prim_match_rate.update(1.0)
                        self.val_prim_rmsd.update(min(valid_rmsds))
                    else:
                        self.val_prim_match_rate.update(0.0)

                # Execute slab RMSD tasks in parallel
                rank_zero_info(f"| Computing slab RMSD for {len(slab_rmsd_tasks)} structures (workers={num_workers})...")
                slab_rmsd_results_per_item = []
                with ProcessPool(
                    max_workers=num_workers, context=mp.get_context("spawn")
                ) as pool:
                    future_map = {
                        pool.schedule(find_best_match_rmsd_slab, args=(task,)): task
                        for task in slab_rmsd_tasks
                    }

                    for future in tqdm(
                        future_map, 
                        desc="Slab RMSD matching", 
                        unit="struct",
                        total=len(slab_rmsd_tasks),
                        leave=False,
                    ):
                        try:
                            result = future.result(timeout=timeout_seconds)
                            slab_rmsd_results_per_item.append(result)
                        except PebbleTimeoutError:
                            future.cancel()
                            rank_zero_info(
                                f"| WARNING: A slab RMSD task timed out. Skipping."
                            )
                            slab_rmsd_results_per_item.append([None] * n_samples)
                        except Exception as e:
                            future.cancel()
                            rank_zero_info(
                                f"| WARNING: A slab RMSD task failed with an exception: {e}"
                            )
                            slab_rmsd_results_per_item.append([None] * n_samples)
                
                rank_zero_info(f"| Slab RMSD computation completed.")

                # Compute slab match rate and RMSD
                for result_list in slab_rmsd_results_per_item:
                    valid_rmsds = [r for r in result_list if r is not None]
                    if valid_rmsds:
                        self.val_slab_match_rate.update(1.0)
                        self.val_slab_rmsd.update(min(valid_rmsds))
                    else:
                        self.val_slab_match_rate.update(0.0)

            # Execute prim structural validity tasks in parallel (always computed)
            rank_zero_info(f"| Computing prim structural validity for {len(prim_validity_tasks)} structures (workers={num_workers})...")
            prim_validity_results_per_item = []
            with ProcessPool(
                max_workers=num_workers, context=mp.get_context("spawn")
            ) as pool:
                future_map = {
                    pool.schedule(compute_prim_structural_validity_single, args=(task,)): task
                    for task in prim_validity_tasks
                }

                for future in tqdm(
                    future_map, 
                    desc="Prim structural validity", 
                    unit="struct",
                    total=len(prim_validity_tasks),
                    leave=False,
                ):
                    try:
                        result = future.result(timeout=timeout_seconds)
                        prim_validity_results_per_item.append(result)
                    except PebbleTimeoutError:
                        future.cancel()
                        rank_zero_info(
                            f"| WARNING: A prim structural validity task timed out. Skipping."
                        )
                        prim_validity_results_per_item.append([False] * n_samples)
                    except Exception as e:
                        future.cancel()
                        rank_zero_info(
                            f"| WARNING: A prim validity task failed with an exception: {e}"
                        )
                        prim_validity_results_per_item.append([False] * n_samples)
            
            rank_zero_info(f"| Prim structural validity computation completed.")

            # Compute prim structural validity rate (always computed)
            # Count as valid if at least one sample is valid (consistent with prim/slab RMSD)
            for result_list in prim_validity_results_per_item:
                if any(result_list):
                    self.val_prim_structural_validity.update(1.0)
                else:
                    self.val_prim_structural_validity.update(0.0)

            # Execute slab structural validity tasks in parallel (always computed)
            rank_zero_info(f"| Computing slab structural validity for {len(slab_validity_tasks)} structures (workers={num_workers})...")
            slab_validity_results_per_item = []
            with ProcessPool(
                max_workers=num_workers, context=mp.get_context("spawn")
            ) as pool:
                future_map = {
                    pool.schedule(compute_structural_validity_single, args=(task,)): task
                    for task in slab_validity_tasks
                }

                for future in tqdm(
                    future_map, 
                    desc="Slab structural validity", 
                    unit="struct",
                    total=len(slab_validity_tasks),
                    leave=False,
                ):
                    try:
                        result = future.result(timeout=timeout_seconds)
                        slab_validity_results_per_item.append(result)
                    except PebbleTimeoutError:
                        future.cancel()
                        rank_zero_info(
                            f"| WARNING: A slab structural validity task timed out. Skipping."
                        )
                        slab_validity_results_per_item.append([False] * n_samples)
                    except Exception as e:
                        future.cancel()
                        rank_zero_info(
                            f"| WARNING: A slab validity task failed with an exception: {e}"
                        )
                        slab_validity_results_per_item.append([False] * n_samples)
            
            rank_zero_info(f"| Slab structural validity computation completed.")

            # Compute slab structural validity rate (always computed)
            # Count as valid if at least one sample is valid (consistent with prim/slab RMSD)
            for result_list in slab_validity_results_per_item:
                if any(result_list):
                    self.val_structural_validity.update(1.0)
                else:
                    self.val_structural_validity.update(0.0)

            # Log aggregated metrics
            # Primitive slab RMSD (only when dng=False, as RMSD is not computed when dng=True)
            if not self.dng:
                self.log(
                    "val/prim_match_rate", self.val_prim_match_rate.compute(), rank_zero_only=True
                )
                if self.val_prim_rmsd.update_count > 0:
                    self.log("val/avg_prim_rmsd", self.val_prim_rmsd.compute(), rank_zero_only=True)
                else:
                    self.log("val/avg_prim_rmsd", float("nan"), rank_zero_only=True)
                
                # Slab RMSD (whole slab after assemble) - only when dng=False
                self.log(
                    "val/slab_match_rate", self.val_slab_match_rate.compute(), rank_zero_only=True
                )
                if self.val_slab_rmsd.update_count > 0:
                    self.log("val/avg_slab_rmsd", self.val_slab_rmsd.compute(), rank_zero_only=True)
                else:
                    self.log("val/avg_slab_rmsd", float("nan"), rank_zero_only=True)
            
            # Log prim structural validity
            self.log(
                "val/prim_structural_validity_rate",
                self.val_prim_structural_validity.compute() if self.val_prim_structural_validity.update_count > 0 else float("nan"),
                rank_zero_only=True,
            )
            
            # Log slab structural validity
            self.log(
                "val/structural_validity_rate",
                self.val_structural_validity.compute() if self.val_structural_validity.update_count > 0 else float("nan"),
                rank_zero_only=True,
            )

            # Reset metrics for next epoch
            self.val_prim_match_rate.reset()
            self.val_prim_rmsd.reset()
            self.val_slab_match_rate.reset()
            self.val_slab_rmsd.reset()
            self.val_structural_validity.reset()
            self.val_prim_structural_validity.reset()
            self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step - same as validation step."""
        # Test step is handled the same as validation step
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """Test epoch end - same as validation epoch end."""
        # Test epoch end is handled the same as validation epoch end
        self.on_validation_epoch_end()

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        num_samples = self.predict_args.get("num_samples", 1)

        try:
            out = self(
                batch,
                num_sampling_steps=self.predict_args["sampling_steps"],
                center_during_sampling=False,
                refine_final=self.predict_args["refine_final"],
                multiplicity_flow_sample=num_samples,
            )

            prediction_output = {
                "exception": False,
                "generated_prim_slab_coords": out["sampled_prim_slab_coords"],
                "generated_ads_coords": out["sampled_ads_coords"],
                "generated_lattices": out["sampled_lattice"],
                "generated_supercell_matrices": out["sampled_supercell_matrix"],
                "generated_scaling_factors": out["sampled_scaling_factor"],
                "prim_slab_atom_types": batch["ref_prim_slab_element"],
                "ads_atom_types": batch["ref_ads_element"],
                "prim_slab_atom_mask": batch["prim_slab_atom_pad_mask"],
                "ads_atom_mask": batch["ads_atom_pad_mask"],
                "tags": batch.get("tags", None),  # TODO: May change feature name
            }

            if "prim_slab_cart_coords" in batch:
                prediction_output["true_prim_slab_coords"] = batch["prim_slab_cart_coords"]

            if "ads_cart_coords" in batch:
                prediction_output["true_ads_coords"] = batch["ads_cart_coords"]

            if "lattice" in batch:
                prediction_output["true_lattices"] = batch["lattice"]

            if "supercell_matrix" in batch:
                prediction_output["true_supercells"] = batch["supercell_matrix"]

            if "scaling_factor" in batch:
                prediction_output["true_scaling_factors"] = batch["scaling_factor"]

            return prediction_output

        except RuntimeError as e:
            if "out of memory" in str(e):
                rank_zero_info(
                    f"| WARNING: Ran out of memory during prediction on batch {batch_idx}. Skipping batch."
                )
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise e

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.training_args["lr"],
            weight_decay=self.training_args["weight_decay"],
        )
        
        # Learning rate scheduler with warm up only
        warmup_steps = self.training_args.get("warmup_steps", 10000)
        
        # Linear warm up scheduler (lr increases from lr*0.01 to lr, then stays constant)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01, 
            end_factor=1.0,  
            total_iters=warmup_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update lr at every step
                "frequency": 1,
            }
        }


    def configure_callbacks(self) -> list[Callback]:
        """Configure model callbacks.

        Returns
        -------
        List[Callback]
            List of callbacks to be used in the model.

        """
        return []

    def on_train_start(self) -> None:

        new_lr = self.training_args["lr"] 
        
        if self.global_rank == 0:
            rank_zero_info(f"| [OVERRIDE] Resetting Learning Rate to {new_lr} at step {self.global_step}")

        if self.trainer.optimizers:
            for opt in self.trainer.optimizers:
                for param_group in opt.param_groups:
                    param_group["lr"] = new_lr
                    if "initial_lr" in param_group:
                        param_group["initial_lr"] = new_lr

        if self.trainer.lr_scheduler_configs:
            for config in self.trainer.lr_scheduler_configs:
                scheduler = config.scheduler
                
                if hasattr(scheduler, "base_lrs"):
                    scheduler.base_lrs = [new_lr] * len(scheduler.base_lrs)