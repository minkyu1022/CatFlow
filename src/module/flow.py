from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from src.data.prior import PriorSampler, CatPriorSampler
from src.models.layers import (
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    TokenTransformer,
    NUM_ELEMENTS,
    MASK_TOKEN_INDEX,
    NUM_ELEMENTS_WITH_MASK,
)
from src.models.utils import LinearNoBias, center_random_augmentation, default, smooth_lddt_loss


class FlowModule(nn.Module):
    """
    Flow model that jointly processes primitive slab and adsorbate atoms.
    
    Architecture:
        1. Encoder: jointly encodes prim_slab and adsorbate atoms
        2. Token Transformer: processes token-level representations
        3. Decoder: outputs separate heads for prim_slab coords, ads coords, lattice, and supercell matrix
    """
    
    def __init__(
        self,
        atom_s,
        token_s,
        atom_encoder_depth,
        atom_encoder_heads,
        atom_encoder_positional_encoding,
        token_transformer_depth,
        token_transformer_heads,
        atom_decoder_depth,
        atom_decoder_heads,
        attention_impl,
        activation_checkpointing=False,
        dng: bool = False,
        use_energy_cond: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Encoder (jointly processes prim_slab + adsorbate)
        self.atom_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            positional_encoding=atom_encoder_positional_encoding,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
            dng=dng,
            use_energy_cond=use_energy_cond,
        )
        self.atom_to_token_trans = nn.Sequential(
            LinearNoBias(atom_s, token_s), nn.ReLU()
        )

        # Transformer
        self.token_transformer = TokenTransformer(
            token_s=token_s,
            token_transformer_depth=token_transformer_depth,
            token_transformer_heads=token_transformer_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
            use_energy_cond=use_energy_cond,
        )
        self.token_to_atom_trans = nn.Sequential(
            LinearNoBias(token_s, atom_s), nn.ReLU()
        )

        # Decoder (outputs prim_slab coords, ads coords, lattice, supercell matrix)
        self.atom_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
            dng=dng,
            use_energy_cond=use_energy_cond,
        )
        self.dng = dng

    def _aggregate_atoms_to_tokens(self, atom_feats, atom_to_token, multiplicity=1):
        atom_to_token = atom_to_token.float()
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        token_feats = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_feats)
        return token_feats

    def _broadcast_tokens_to_atoms(self, token_feats, atom_to_token, multiplicity=1):
        atom_to_token = atom_to_token.repeat_interleave(multiplicity, 0)
        atom_feats = torch.bmm(atom_to_token, token_feats)
        return atom_feats

    def forward(self, prim_slab_r_noisy, ads_center_noisy, ads_rel_pos_noisy, l_noisy, sm_noisy, sf_noisy, times, feats, multiplicity=1, prim_slab_element_noisy: Optional[torch.Tensor] = None):
        """
        Forward pass through the flow model.
        
        Args:
            prim_slab_r_noisy: (B*mult, N, 3) noisy prim_slab coordinates
            ads_center_noisy: (B*mult, 3) noisy adsorbate center
            ads_rel_pos_noisy: (B*mult, M, 3) noisy adsorbate relative positions
            l_noisy: (B*mult, 6) noisy lattice parameters
            sm_noisy: (B*mult, 3, 3) noisy supercell matrix (normalized)
            sf_noisy: (B*mult,) noisy scaling factor (normalized)
            times: (B*mult,) timesteps
            feats: feature dictionary
            multiplicity: number of samples per input
        
        Returns:
            dict with:
                - prim_slab_r_update: (B*mult, N, 3)
                - ads_center_update: (B*mult, 3)
                - ads_rel_pos_update: (B*mult, M, 3)
                - l_update: (B*mult, 6)
                - sm_update: (B*mult, 3, 3)
                - sf_update: (B*mult,)
        """
        # Encoder (atom-level attention, joint processing)
        h_atoms, n_prim_slab, n_ads = self.atom_encoder(
            prim_slab_x_t=prim_slab_r_noisy,
            ads_center_t=ads_center_noisy,
            ads_rel_pos_t=ads_rel_pos_noisy,
            l_t=l_noisy,
            sm_t=sm_noisy,
            sf_t=sf_noisy,
            t=times,
            feats=feats,
            multiplicity=multiplicity,
            prim_slab_element_t=prim_slab_element_noisy,
        )

        # Aggregate to token-level
        # Create joint atom_to_token matrix
        prim_slab_atom_to_token = feats["prim_slab_atom_to_token"]  # (B, N, N)
        ads_atom_to_token = feats["ads_atom_to_token"]  # (B, M, M)
        
        # Block diagonal concatenation for joint atom_to_token
        B = prim_slab_atom_to_token.shape[0]
        N = prim_slab_atom_to_token.shape[1]
        M = ads_atom_to_token.shape[1]
        
        # Create block diagonal matrix
        joint_atom_to_token = torch.zeros(B, N + M, N + M, device=h_atoms.device, dtype=h_atoms.dtype)
        joint_atom_to_token[:, :N, :N] = prim_slab_atom_to_token
        joint_atom_to_token[:, N:, N:] = ads_atom_to_token

        h_atoms_to_tokens = self.atom_to_token_trans(h_atoms)
        h_tokens = self._aggregate_atoms_to_tokens(
            atom_feats=h_atoms_to_tokens,
            atom_to_token=joint_atom_to_token,
            multiplicity=multiplicity,
        )

        # Trunk (token-level attention)
        h_tokens = self.token_transformer(
            x=h_tokens, t=times, feats=feats, multiplicity=multiplicity
        )

        # Broadcast to atom-level
        h_tokens_to_atoms = self.token_to_atom_trans(h_tokens)
        h_atoms = h_atoms + self._broadcast_tokens_to_atoms(
            token_feats=h_tokens_to_atoms,
            atom_to_token=joint_atom_to_token,
            multiplicity=multiplicity,
        )  # skip connection

        # Decoder (atom-level attention, separate output heads)
        if self.dng:
            prim_slab_r_update, ads_center_update, ads_rel_pos_update, l_update, sm_update, sf_update, prim_slab_element_update = self.atom_decoder(
                x=h_atoms, t=times, feats=feats, n_prim_slab=n_prim_slab, n_ads=n_ads, multiplicity=multiplicity
            )
            return {
                "prim_slab_r_update": prim_slab_r_update,
                "ads_center_update": ads_center_update,
                "ads_rel_pos_update": ads_rel_pos_update,
                "l_update": l_update,
                "sm_update": sm_update,
                "sf_update": sf_update,
                "prim_slab_element_update": prim_slab_element_update,
            }
        else:
            prim_slab_r_update, ads_center_update, ads_rel_pos_update, l_update, sm_update, sf_update = self.atom_decoder(
                x=h_atoms, t=times, feats=feats, n_prim_slab=n_prim_slab, n_ads=n_ads, multiplicity=multiplicity
            )
            return {
                "prim_slab_r_update": prim_slab_r_update,
                "ads_center_update": ads_center_update,
                "ads_rel_pos_update": ads_rel_pos_update,
                "l_update": l_update,
                "sm_update": sm_update,
                "sf_update": sf_update,
            }


class AtomFlowMatching(Module):
    """Atom flow matching module for joint prim_slab, adsorbate, lattice, and supercell matrix prediction."""

    # Lattice length conversion (Angstrom <-> nm)
    NM_TO_ANG_SCALE = 10.0
    ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

    def __init__(
        self,
        flow_model_args: dict,
        prior_sampler: CatPriorSampler,
        num_sampling_steps: int = 16,
        coordinate_augmentation: bool = True,
        synchronize_timesteps: bool = False,
        compile_model: bool = False,
        dng: bool = False,
        **kwargs: dict,
    ):
        super().__init__()

        # Model
        self.flow_model = FlowModule(**flow_model_args)
        if compile_model:
            self.flow_model = torch.compile(
                self.flow_model, dynamic=False, fullgraph=False
            )

        # Prior
        self.prior_sampler = prior_sampler

        # Parameters
        self.num_sampling_steps = num_sampling_steps
        self.coordinate_augmentation = coordinate_augmentation
        self.synchronize_timesteps = synchronize_timesteps
        self.dng = dng

    @property
    def device(self):
        return next(self.flow_model.parameters()).device

    def preconditioned_network_forward(
        self,
        noised_prim_slab_coords,
        noised_ads_center,
        noised_ads_rel_pos,
        noised_lattice,
        noised_supercell_matrix,  # normalized space
        noised_scaling_factor,  # normalized space
        time,
        network_condition_kwargs: dict,
        training: bool = True,
        noised_prim_slab_element: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with preconditioning (standardization).

        Args:
            noised_prim_slab_coords: (B, N, 3) noised prim_slab coordinates (raw space, Angstrom)
            noised_ads_center: (B, 3) noised adsorbate center (raw space, Angstrom)
            noised_ads_rel_pos: (B, M, 3) noised adsorbate relative positions (raw space, Angstrom)
            noised_lattice: (B, 6) noised lattice parameters (a, b, c in Angstrom, alpha, beta, gamma in degrees)
            noised_supercell_matrix: (B, 3, 3) noised supercell matrix (raw space)
            noised_scaling_factor: (B,) noised scaling factor (raw space)
            time: timestep(s)
            network_condition_kwargs: additional kwargs for network
            training: whether in training mode

        Returns:
            denoised_prim_slab_coords: (B, N, 3) denoised prim_slab coordinates (raw space, Angstrom)
            denoised_ads_center: (B, 3) denoised adsorbate center (raw space, Angstrom)
            denoised_ads_rel_pos: (B, M, 3) denoised adsorbate relative positions (raw space, Angstrom)
            denoised_lattice: (B, 6) denoised lattice parameters (Angstrom, degrees)
            denoised_supercell_matrix: (B, 3, 3) denoised supercell matrix (raw space)
            denoised_scaling_factor: (B,) denoised scaling factor (raw space)
        """
        batch, device = noised_prim_slab_coords.shape[0], noised_prim_slab_coords.device

        if isinstance(time, float):
            time = torch.full((batch,), time, device=device)

        # Standardize prim_slab coords, ads_center, and ads_rel_pos using mean/std
        prim_slab_r_noisy = self.prior_sampler.normalize_prim_slab_coords(noised_prim_slab_coords)
        ads_center_noisy = self.prior_sampler.normalize_ads_center(noised_ads_center)
        ads_rel_pos_noisy = self.prior_sampler.normalize_ads_rel_pos(noised_ads_rel_pos)
        
        # Lattice: lengths (Angstrom -> nm), angles (degrees -> radians)
        l_noisy = noised_lattice.clone()
        l_noisy[:, :3] = self.ANG_TO_NM_SCALE * noised_lattice[:, :3]
        l_noisy[:, 3:] = (torch.pi / 180.0) * noised_lattice[:, 3:]

        # Normalize supercell matrix (scaling factor uses raw values without normalization)
        # sm_noisy = self.prior_sampler.normalize_supercell(noised_supercell_matrix)
        sm_noisy = noised_supercell_matrix
        sf_noisy = noised_scaling_factor

        # Predict (network operates in standardized/normalized space)
        net_out = self.flow_model(
            prim_slab_r_noisy=prim_slab_r_noisy,
            ads_center_noisy=ads_center_noisy,
            ads_rel_pos_noisy=ads_rel_pos_noisy,
            l_noisy=l_noisy,
            sm_noisy=sm_noisy,
            sf_noisy=sf_noisy,
            times=time,
            prim_slab_element_noisy=noised_prim_slab_element,
            **network_condition_kwargs,
        )

        # Destandardize prim_slab coords, ads_center, and ads_rel_pos back to Angstrom
        denoised_prim_slab_coords = self.prior_sampler.denormalize_prim_slab_coords(net_out["prim_slab_r_update"])
        denoised_ads_center = self.prior_sampler.denormalize_ads_center(net_out["ads_center_update"])
        denoised_ads_rel_pos = self.prior_sampler.denormalize_ads_rel_pos(net_out["ads_rel_pos_update"])
        
        # Lattice: lengths (nm -> Angstrom), angles (radians -> degrees)
        denoised_lattice = net_out["l_update"].clone()
        denoised_lattice[:, :3] = self.NM_TO_ANG_SCALE * denoised_lattice[:, :3]
        denoised_lattice[:, 3:] = (180.0 / torch.pi) * denoised_lattice[:, 3:]

        # Denormalize supercell matrix back to raw space
        # denoised_supercell_matrix = self.prior_sampler.denormalize_supercell(net_out["sm_update"])
        denoised_supercell_matrix = net_out["sm_update"]
        denoised_scaling_factor = net_out["sf_update"]
        
        # Return element when dng=True (logits format)
        if self.dng:
            denoised_prim_slab_element = net_out.get("prim_slab_element_update")
            return denoised_prim_slab_coords, denoised_ads_center, denoised_ads_rel_pos, denoised_lattice, denoised_supercell_matrix, denoised_scaling_factor, denoised_prim_slab_element
        else:
            return denoised_prim_slab_coords, denoised_ads_center, denoised_ads_rel_pos, denoised_lattice, denoised_supercell_matrix, denoised_scaling_factor

    def forward(self, feats, multiplicity=1, **kwargs):
        """Flow matching training step.

        Args:
            feats: Dictionary containing:
                - prim_slab_cart_coords: (B, N, 3) target prim_slab coordinates
                - ads_center: (B, 3) target adsorbate center
                - ads_rel_pos: (B, M, 3) target adsorbate relative positions
                - lattice: (B, 6) target lattice parameters (a, b, c, alpha, beta, gamma)
                - supercell_matrix: (B, 3, 3) target supercell matrix
                - scaling_factor: (B,) target scaling factor
                - prim_slab_atom_pad_mask: (B, N) mask for valid prim_slab atoms
                - ads_atom_pad_mask: (B, M) mask for valid adsorbate atoms
            multiplicity: number of samples per input

        Returns:
            Dictionary with denoised predictions and ground truth
        """
        batch_size = feats["prim_slab_cart_coords"].shape[0]
        num_prim_slab_atoms = feats["prim_slab_cart_coords"].shape[1]
        num_ads_atoms = feats["ads_rel_pos"].shape[1]
        prim_slab_mask = feats["prim_slab_atom_pad_mask"]
        ads_mask = feats["ads_atom_pad_mask"]

        # Sample timesteps
        if self.synchronize_timesteps:
            times = torch.rand(batch_size, device=self.device).repeat_interleave(
                multiplicity, 0
            )
        else:
            times = torch.rand(batch_size * multiplicity, device=self.device)

        # Sample prior (prim_slab_coords_0, ads_center_0, ads_rel_pos_0, lattice_0, supercell_matrix_0, scaling_factor_0) from prior_sampler
        sampler_data = {
            "prim_slab_cart_coords": feats["prim_slab_cart_coords"].repeat_interleave(multiplicity, 0),
            "prim_slab_atom_mask": prim_slab_mask.repeat_interleave(multiplicity, 0),
            "ads_rel_pos": feats["ads_rel_pos"].repeat_interleave(multiplicity, 0),
            "ads_atom_mask": ads_mask.repeat_interleave(multiplicity, 0),
        }
        priors = self.prior_sampler.sample(sampler_data)
        prim_slab_coords_0 = priors["prim_slab_coords_0"]  # (B * multiplicity, N, 3) raw space
        ads_center_0 = priors["ads_center_0"]  # (B * multiplicity, 3) raw space
        ads_rel_pos_0 = priors["ads_rel_pos_0"]  # (B * multiplicity, M, 3) raw space
        lattice_0 = priors["lattice_0"]  # (B * multiplicity, 6) raw space
        supercell_matrix_0 = priors["supercell_matrix_0"]  # (B * multiplicity, 3, 3) raw space
        scaling_factor_0 = priors["scaling_factor_0"]  # (B * multiplicity,) raw space

        # Prepare target data
        prim_slab_coords = feats["prim_slab_cart_coords"].repeat_interleave(multiplicity, 0)
        ads_center = feats["ads_center"].repeat_interleave(multiplicity, 0)  # (B * multiplicity, 3)
        ads_rel_pos = feats["ads_rel_pos"].repeat_interleave(multiplicity, 0)  # (B * multiplicity, M, 3)
        lattice = feats["lattice"].repeat_interleave(multiplicity, 0)  # (B, 6)
        supercell_matrix = feats["supercell_matrix"].repeat_interleave(multiplicity, 0)  # (B, 3, 3)
        scaling_factor = feats["scaling_factor"].repeat_interleave(multiplicity, 0)  # (B,)

        # Add noise - interpolate between prior and target (in raw space for all variables)
        # All priors are now in raw space, so interpolation is straightforward
        noised_prim_slab_coords = (1 - times[:, None, None]) * prim_slab_coords_0 + times[:, None, None] * prim_slab_coords
        noised_ads_center = (1 - times[:, None]) * ads_center_0 + times[:, None] * ads_center  # (B, 3)
        noised_ads_rel_pos = (1 - times[:, None, None]) * ads_rel_pos_0 + times[:, None, None] * ads_rel_pos  # (B, M, 3)
        noised_lattice = (1 - times[:, None]) * lattice_0 + times[:, None] * lattice  # (B, 6)
        noised_supercell_matrix = (1 - times[:, None, None]) * supercell_matrix_0 + times[:, None, None] * supercell_matrix  # (B, 3, 3) raw space
        noised_scaling_factor = (1 - times) * scaling_factor_0 + times * scaling_factor  # (B,) raw values

        # Add element noise when dng=True (Discrete Flow Matching with Masking)
        noised_prim_slab_element = None
        aligned_true_prim_slab_element = None
        if self.dng:
            # === Discrete Flow Matching (OMatG-style Masking) ===
            # Target species: integer tensor (1-indexed: 1~100 = element atomic numbers)
            ref_prim_slab_element = feats["ref_prim_slab_element"]  # (B, N) integer tensor
            element_1 = ref_prim_slab_element.repeat_interleave(multiplicity, 0)  # (B*mult, N) integer tensor
            
            # Prior: all atoms are MASK (index 0)
            element_0 = torch.zeros_like(element_1)  # (B*mult, N) all zeros = MASK
            
            # Stochastic interpolation (Bernoulli sampling)
            # With probability t, reveal target; otherwise keep MASK
            unmask_prob = torch.rand_like(element_1.float())  # (B*mult, N)
            mask = unmask_prob < times[:, None]  # (B*mult, N) boolean
            noised_prim_slab_element = torch.where(mask, element_1, element_0)  # (B*mult, N) integer tensor
            
            # Ground truth for loss (integer tensor, 1-indexed)
            aligned_true_prim_slab_element = element_1  # (B*mult, N) integer tensor

        # Get model prediction
        if self.dng:
            denoised_prim_slab_coords, denoised_ads_center, denoised_ads_rel_pos, denoised_lattice, denoised_supercell_matrix, denoised_scaling_factor, denoised_prim_slab_element = self.preconditioned_network_forward(
                noised_prim_slab_coords,
                noised_ads_center,
                noised_ads_rel_pos,
                noised_lattice,
                noised_supercell_matrix,
                noised_scaling_factor,
                times,
                training=True,
                network_condition_kwargs=dict(
                    feats=feats, multiplicity=multiplicity, **kwargs
                ),
                noised_prim_slab_element=noised_prim_slab_element,
            )
        else:
            denoised_prim_slab_coords, denoised_ads_center, denoised_ads_rel_pos, denoised_lattice, denoised_supercell_matrix, denoised_scaling_factor = self.preconditioned_network_forward(
                noised_prim_slab_coords,
                noised_ads_center,
                noised_ads_rel_pos,
                noised_lattice,
                noised_supercell_matrix,
                noised_scaling_factor,
                times,
                training=True,
                network_condition_kwargs=dict(
                    feats=feats, multiplicity=multiplicity, **kwargs
                ),
            )

        out_dict = dict(
            denoised_prim_slab_coords=denoised_prim_slab_coords,  # raw space (Angstrom)
            denoised_ads_center=denoised_ads_center,  # raw space (Angstrom)
            denoised_ads_rel_pos=denoised_ads_rel_pos,  # raw space (Angstrom)
            denoised_lattice=denoised_lattice,  # raw space (Angstrom, degrees)
            denoised_supercell_matrix=denoised_supercell_matrix,  # raw space
            denoised_scaling_factor=denoised_scaling_factor,  # raw space
            times=times,
            aligned_true_prim_slab_coords=prim_slab_coords,  # raw space (Angstrom)
            aligned_true_ads_center=ads_center,  # raw space (Angstrom)
            aligned_true_ads_rel_pos=ads_rel_pos,  # raw space (Angstrom)
            aligned_true_lattice=lattice,  # raw space (Angstrom, degrees)
            aligned_true_supercell_matrix=supercell_matrix,  # raw space
            aligned_true_scaling_factor=scaling_factor,  # raw space
            noised_prim_slab_coords=noised_prim_slab_coords,  # raw space (for v_loss)
            noised_ads_center=noised_ads_center,  # raw space (for v_loss)
            noised_ads_rel_pos=noised_ads_rel_pos,  # raw space (for v_loss)
            noised_lattice=noised_lattice,  # raw space (for v_loss)
            noised_supercell_matrix=noised_supercell_matrix,  # raw space (for v_loss)
            noised_scaling_factor=noised_scaling_factor,  # raw space (for v_loss)
        )
        
        # Add element-related information when dng=True
        if self.dng:
            out_dict["denoised_prim_slab_element"] = denoised_prim_slab_element  # logits (B*mult, N, NUM_ELEMENTS)
            out_dict["aligned_true_prim_slab_element"] = aligned_true_prim_slab_element  # integer tensor (B*mult, N), 1-indexed
        
        return out_dict

    def compute_loss(self, feats, out_dict, multiplicity=1, loss_type: str = "l2", flow_loss_type: str = "v_loss") -> tuple[dict, dict]:
        """Compute losses for prim_slab coords, adsorbate center, adsorbate relative positions, lattice, supercell matrix, and scaling factor.

        Args:
            feats: Input features dictionary
            out_dict: Output dictionary from forward pass
            multiplicity: number of samples per input
            loss_type: "l1" or "l2" 
            flow_loss_type: "x1_loss" (predict final state) or "v_loss" (predict velocity)

        Returns:
            Dictionary with per-sample losses:
                - prim_slab_coord_loss: (B * multiplicity,) prim_slab coordinate loss
                - ads_center_loss: (B * multiplicity,) adsorbate center loss
                - ads_rel_pos_loss: (B * multiplicity,) adsorbate relative position loss
                - length_loss: (B * multiplicity,) lattice lengths loss (a, b, c)
                - angle_loss: (B * multiplicity,) lattice angles loss (alpha, beta, gamma)
                - supercell_matrix_loss: (B * multiplicity,) supercell matrix loss
                - supercell_matrix_cosine_reg: (B * multiplicity,) cosine regularization for integer supercell matrix
                - scaling_factor_loss: (B * multiplicity,) scaling factor loss
        """
        if loss_type == "l2":
            loss_fn = F.mse_loss
        elif loss_type == "l1":
            loss_fn = F.l1_loss
        else:
            raise ValueError(
                f"Unsupported loss_type: {loss_type}. Must be 'l1' or 'l2'."
            )

        # Get masks
        prim_slab_mask = feats["prim_slab_atom_pad_mask"].repeat_interleave(multiplicity, 0)
        ads_mask = feats["ads_atom_pad_mask"].repeat_interleave(multiplicity, 0)
        
        # Determine prediction and target based on flow_loss_type
        if flow_loss_type == "x1_loss":
            # Standard x_1 prediction loss
            true_prim_slab = out_dict["aligned_true_prim_slab_coords"]
            pred_prim_slab = out_dict["denoised_prim_slab_coords"]
            true_ads_center = out_dict["aligned_true_ads_center"]
            pred_ads_center = out_dict["denoised_ads_center"]
            true_ads_rel_pos = out_dict["aligned_true_ads_rel_pos"]
            pred_ads_rel_pos = out_dict["denoised_ads_rel_pos"]
            true_lattice = out_dict["aligned_true_lattice"]
            pred_lattice = out_dict["denoised_lattice"]
            true_supercell_matrix = out_dict["aligned_true_supercell_matrix"]
            pred_supercell_matrix = out_dict["denoised_supercell_matrix"]
            true_scaling_factor = out_dict["aligned_true_scaling_factor"]
            pred_scaling_factor = out_dict["denoised_scaling_factor"]
        elif flow_loss_type == "v_loss":
            # Velocity prediction loss: v_t = (x_1 - x_t) / (1 - t)
            times = out_dict["times"]
            eps = 1e-6
            
            # True velocity
            true_prim_slab = (out_dict["aligned_true_prim_slab_coords"] - out_dict["noised_prim_slab_coords"]) / (1 - times[:, None, None] + eps)
            true_ads_center = (out_dict["aligned_true_ads_center"] - out_dict["noised_ads_center"]) / (1 - times[:, None] + eps)
            true_ads_rel_pos = (out_dict["aligned_true_ads_rel_pos"] - out_dict["noised_ads_rel_pos"]) / (1 - times[:, None, None] + eps)
            true_lattice = (out_dict["aligned_true_lattice"] - out_dict["noised_lattice"]) / (1 - times[:, None] + eps)
            true_supercell_matrix = (out_dict["aligned_true_supercell_matrix"] - out_dict["noised_supercell_matrix"]) / (1 - times[:, None, None] + eps)
            true_scaling_factor = (out_dict["aligned_true_scaling_factor"] - out_dict["noised_scaling_factor"]) / (1 - times + eps)
            
            # Predicted velocity
            pred_prim_slab = (out_dict["denoised_prim_slab_coords"] - out_dict["noised_prim_slab_coords"]) / (1 - times[:, None, None] + eps)
            pred_ads_center = (out_dict["denoised_ads_center"] - out_dict["noised_ads_center"]) / (1 - times[:, None] + eps)
            pred_ads_rel_pos = (out_dict["denoised_ads_rel_pos"] - out_dict["noised_ads_rel_pos"]) / (1 - times[:, None, None] + eps)
            pred_lattice = (out_dict["denoised_lattice"] - out_dict["noised_lattice"]) / (1 - times[:, None] + eps)
            pred_supercell_matrix = (out_dict["denoised_supercell_matrix"] - out_dict["noised_supercell_matrix"]) / (1 - times[:, None, None] + eps)
            pred_scaling_factor = (out_dict["denoised_scaling_factor"] - out_dict["noised_scaling_factor"]) / (1 - times + eps)
        else:
            raise ValueError(f"Unsupported flow_loss_type: {flow_loss_type}. Must be 'x1_loss' or 'v_loss'.")

        # Loss for prim_slab Cartesian coordinates (raw space)
        prim_slab_coord_loss = loss_fn(pred_prim_slab, true_prim_slab, reduction="none")
        prim_slab_coord_loss = (prim_slab_coord_loss * prim_slab_mask[..., None]).sum(dim=(1, 2)) / (
            prim_slab_mask.sum(dim=1) + 1e-8
        )

        # Loss for adsorbate center (raw space)
        ads_center_loss = loss_fn(pred_ads_center, true_ads_center, reduction="none").mean(dim=1)
        
        # Loss for adsorbate relative positions (raw space)
        ads_rel_pos_loss = loss_fn(pred_ads_rel_pos, true_ads_rel_pos, reduction="none")
        ads_mask_sum = ads_mask.sum(dim=1)
        ads_rel_pos_loss = (ads_rel_pos_loss * ads_mask[..., None]).sum(dim=(1, 2)) / (ads_mask_sum + 1e-8)

        # Loss for lattice (6-dimensional: a, b, c, alpha, beta, gamma) in raw space
        # Separate loss for lengths (a, b, c) and angles (alpha, beta, gamma)
        length_loss = loss_fn(
            pred_lattice[:, :3], true_lattice[:, :3], reduction="none"
        ).mean(dim=1)
        angle_loss = loss_fn(
            pred_lattice[:, 3:], true_lattice[:, 3:], reduction="none"
        ).mean(dim=1)

        # Loss for supercell matrix (raw space)
        supercell_loss = loss_fn(
            pred_supercell_matrix, true_supercell_matrix, reduction="none"
        ).mean(dim=(1, 2))
        
        # Loss for scaling factor (raw space, no normalization)
        scaling_factor_loss = loss_fn(
            pred_scaling_factor, true_scaling_factor, reduction="none"
        )

        loss_dict = {
            # Per-sample losses of shape (batch_size * multiplicity,)
            "prim_slab_coord_loss": prim_slab_coord_loss,
            "ads_center_loss": ads_center_loss,
            "ads_rel_pos_loss": ads_rel_pos_loss,
            "length_loss": length_loss,
            "angle_loss": angle_loss,
            "supercell_matrix_loss": supercell_loss,
            "scaling_factor_loss": scaling_factor_loss,
        }
        
        check_dict = {}
        with torch.no_grad():
            current_det = torch.det(torch.round(out_dict["denoised_supercell_matrix"]))
            check_dict["supercell_matrix_det_min"] = current_det.min()
            
            invalid_ratio = (current_det < 0.0).float().mean()
            check_dict["supercell_det_invalid_ratio"] = invalid_ratio
            
            cosine_reg = (1 - torch.cos(2 * torch.pi * out_dict["denoised_supercell_matrix"])).mean(dim=(1, 2))
            check_dict["supercell_matrix_cosine_reg"] = cosine_reg

            true_ads_center_x1 = out_dict["aligned_true_ads_center"]
            true_ads_rel_pos_x1 = out_dict["aligned_true_ads_rel_pos"]
            true_ads_coords = true_ads_center_x1.unsqueeze(1) + true_ads_rel_pos_x1
            pred_ads_center_x1 = out_dict["denoised_ads_center"]
            pred_ads_rel_pos_x1 = out_dict["denoised_ads_rel_pos"]
            pred_ads_coords = pred_ads_center_x1.unsqueeze(1) + pred_ads_rel_pos_x1

            ads_internal_lddt = smooth_lddt_loss(
                pred_coords=pred_ads_coords,
                true_coords=true_ads_coords,
                coords_mask=ads_mask,
                t=out_dict["times"]
            )

            check_dict["ads_internal_lddt"] = ads_internal_lddt
        
        # Add prim_slab_element_loss when dng=True (Cross-Entropy for Discrete Flow Matching)
        if self.dng:
            pred_element_logits = out_dict["denoised_prim_slab_element"]  # (B*mult, N, NUM_ELEMENTS) - logits
            true_element = out_dict["aligned_true_prim_slab_element"]  # (B*mult, N) - integer tensor (1-indexed)
            
            # Cross-Entropy Loss (OMatG-style)
            # Convert 1-indexed target to 0-indexed for cross_entropy
            # Padding positions have true_element=0, which becomes -1 after conversion
            # Use ignore_index=-1 to ignore these padding positions in loss calculation
            target_indices = (true_element - 1).long()  # (B*mult, N) 0-indexed; padding becomes -1
            
            # Reshape for cross_entropy: (B*mult*N, NUM_ELEMENTS) vs (B*mult*N,)
            B_mult, N_atoms = pred_element_logits.shape[:2]
            
            element_loss_flat = F.cross_entropy(
                pred_element_logits.reshape(-1, NUM_ELEMENTS),  # (B*mult*N, NUM_ELEMENTS)
                target_indices.reshape(-1),  # (B*mult*N,)
                reduction='none',
                ignore_index=-1  # Ignore padding positions (where target=-1)
            )  # (B*mult*N,)
            
            # Reshape back and apply mask
            element_loss = element_loss_flat.reshape(B_mult, N_atoms)  # (B*mult, N)
            element_loss = (element_loss * prim_slab_mask).sum(dim=1) / (
                prim_slab_mask.sum(dim=1) + 1e-8
            )  # (B*mult,) shape to match other losses
            
            loss_dict["prim_slab_element_loss"] = element_loss
        
        return loss_dict, check_dict

    @torch.no_grad()
    def sample(
        self,
        prim_slab_atom_mask,
        ads_atom_mask,
        num_sampling_steps: int | None = None,
        multiplicity: int = 1,
        center_coords: bool = True,
        refine_final: bool = False,
        return_trajectory: bool = False,
        # The following arguments are passed directly to the model
        **network_condition_kwargs,
    ):
        """
        Performs sampling from the learned flow ODE for prim_slab coords, adsorbate center, adsorbate relative positions, lattice, supercell matrix, and scaling factor.

        Parameters
        ----------
        prim_slab_atom_mask : torch.Tensor
            A boolean tensor of shape (batch, num_prim_slab_atoms) indicating valid atoms.
        ads_atom_mask : torch.Tensor
            A boolean tensor of shape (batch, num_ads_atoms) indicating valid adsorbate atoms.
        num_sampling_steps : int, optional
            The number of discrete steps for the ODE solver.
        multiplicity : int, optional
            The number of samples to generate per input.
        center_coords : bool, optional
            If True, coordinates will be centered after each sampling step.
        refine_final : bool, optional
            If True, a final refinement step will be performed.
        return_trajectory : bool, optional
            If True, full trajectories will be returned.
        **network_condition_kwargs :
            Additional keyword arguments passed to the flow model.

        Returns
        -------
        dict
            Dictionary containing sampled prim_slab coords, adsorbate coords (center + rel_pos), lattice, supercell matrix, and scaling factor.
        """

        ### Setup and initialize
        num_steps = default(num_sampling_steps, self.num_sampling_steps)
        # When dng=True and multiplicity > 1, masks and feats are already expanded in effcat_module.py
        # Check if masks are already expanded by checking if both masks have the same batch size
        # (which would be batch_size * multiplicity if already expanded)
        feats_already_expanded = False
        if self.dng and multiplicity > 1 and prim_slab_atom_mask.shape[0] == ads_atom_mask.shape[0]:
            # Masks are already expanded in effcat_module.py, don't expand again
            # Also, feats are already expanded, so we should pass multiplicity=1 to flow_model.forward
            feats_already_expanded = True
        else:
            # Normal case: expand masks by multiplicity
            prim_slab_atom_mask = prim_slab_atom_mask.repeat_interleave(multiplicity, 0)
            ads_atom_mask = ads_atom_mask.repeat_interleave(multiplicity, 0)
        batch_size = prim_slab_atom_mask.shape[0]
        # Use multiplicity=1 for flow_model if feats are already expanded
        flow_model_multiplicity = 1 if feats_already_expanded else multiplicity
        num_prim_slab_atoms = prim_slab_atom_mask.shape[1]
        num_ads_atoms = ads_atom_mask.shape[1]
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
        
        # Update network_condition_kwargs to use flow_model_multiplicity if feats are already expanded
        # Create a copy to avoid modifying the original
        network_condition_kwargs = {**network_condition_kwargs, "multiplicity": flow_model_multiplicity}

        # Dummy data dict for correct sampling shape
        sampler_data = {
            "prim_slab_cart_coords": torch.zeros((batch_size, num_prim_slab_atoms, 3), device=self.device),
            "prim_slab_atom_mask": prim_slab_atom_mask,
            "ads_rel_pos": torch.zeros((batch_size, num_ads_atoms, 3), device=self.device),
            "ads_atom_mask": ads_atom_mask,
        }

        # Initialize from the prior sampler at t=0 (all priors are already in raw space)
        priors = self.prior_sampler.sample(sampler_data)
        prim_slab_coords_t = priors["prim_slab_coords_0"] * prim_slab_atom_mask.unsqueeze(-1)
        ads_center_t = priors["ads_center_0"]  # (B, 3)
        ads_rel_pos_t = priors["ads_rel_pos_0"] * ads_atom_mask.unsqueeze(-1)  # (B, M, 3)
        lattice_t = priors["lattice_0"]
        supercell_matrix_t = priors["supercell_matrix_0"]
        scaling_factor_t = priors["scaling_factor_0"]

        ads_mask_sum = ads_atom_mask.sum(dim=1, keepdim=True)
        current_center = (ads_rel_pos_t * ads_atom_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (ads_mask_sum.unsqueeze(-1) + 1e-8)
        ads_rel_pos_t = ads_rel_pos_t - current_center
        ads_rel_pos_t = ads_rel_pos_t * ads_atom_mask.unsqueeze(-1)
        
        # Initialize element when dng=True (Discrete Flow Matching with Masking)
        if self.dng:
            # Prior: all atoms are MASK (index 0)
            prim_slab_element_t = torch.zeros(
                (batch_size, num_prim_slab_atoms),
                dtype=torch.long,
                device=self.device
            )  # (B*mult, N) integer tensor, all MASK
        else:
            prim_slab_element_t = None

        # Initialize lists to store trajectories if requested
        # Note: For trajectory, we store ads_coords = ads_center + ads_rel_pos
        if return_trajectory:
            ads_coords_t_for_traj = ads_center_t.unsqueeze(1) + ads_rel_pos_t
            prim_slab_coord_trajectory = [prim_slab_coords_t]
            ads_coord_trajectory = [ads_coords_t_for_traj]
            lattice_trajectory = [lattice_t]
            supercell_matrix_trajectory = [supercell_matrix_t]
            scaling_factor_trajectory = [scaling_factor_t]
            prim_slab_element_trajectory = [prim_slab_element_t] if (return_trajectory and self.dng) else None
        else:
            prim_slab_coord_trajectory = None
            ads_coord_trajectory = None
            lattice_trajectory = None
            supercell_matrix_trajectory = None
            scaling_factor_trajectory = None
            prim_slab_element_trajectory = None

        ### ODE Solver Loop (Euler's Method)
        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Get the model's prediction for the final state (x_1)
            if self.dng:
                pred_prim_slab_coords_1, pred_ads_center_1, pred_ads_rel_pos_1, pred_lattice_1, pred_supercell_matrix_1, pred_scaling_factor_1, pred_element_1 = self.preconditioned_network_forward(
                    noised_prim_slab_coords=prim_slab_coords_t,
                    noised_ads_center=ads_center_t,
                    noised_ads_rel_pos=ads_rel_pos_t,
                    noised_lattice=lattice_t,
                    noised_supercell_matrix=supercell_matrix_t,
                    noised_scaling_factor=scaling_factor_t,
                    time=t.item(),
                    training=False,
                    network_condition_kwargs=network_condition_kwargs,
                    noised_prim_slab_element=prim_slab_element_t,
                )
            else:
                pred_prim_slab_coords_1, pred_ads_center_1, pred_ads_rel_pos_1, pred_lattice_1, pred_supercell_matrix_1, pred_scaling_factor_1 = self.preconditioned_network_forward(
                    noised_prim_slab_coords=prim_slab_coords_t,
                    noised_ads_center=ads_center_t,
                    noised_ads_rel_pos=ads_rel_pos_t,
                    noised_lattice=lattice_t,
                    noised_supercell_matrix=supercell_matrix_t,
                    noised_scaling_factor=scaling_factor_t,
                    time=t.item(),
                    training=False,
                    network_condition_kwargs=network_condition_kwargs,
                )

            # Calculate the flow (vector field)
            flow_prim_slab_coords = (pred_prim_slab_coords_1 - prim_slab_coords_t) / (1 - t + 1e-6)
            flow_ads_center = (pred_ads_center_1 - ads_center_t) / (1 - t + 1e-6)
            flow_ads_rel_pos = (pred_ads_rel_pos_1 - ads_rel_pos_t) / (1 - t + 1e-6)
            flow_lattice = (pred_lattice_1 - lattice_t) / (1 - t + 1e-6)
            flow_supercell_matrix = (pred_supercell_matrix_1 - supercell_matrix_t) / (1 - t + 1e-6)
            flow_scaling_factor = (pred_scaling_factor_1 - scaling_factor_t) / (1 - t + 1e-6)

            # Perform one step of Euler's method
            dt = t_next - t
            prim_slab_coords_t = prim_slab_coords_t + flow_prim_slab_coords * dt
            ads_center_t = ads_center_t + flow_ads_center * dt
            ads_rel_pos_t = ads_rel_pos_t + flow_ads_rel_pos * dt
            lattice_t = lattice_t + flow_lattice * dt
            supercell_matrix_t = supercell_matrix_t + flow_supercell_matrix * dt
            scaling_factor_t = scaling_factor_t + flow_scaling_factor * dt
            
            # Update element when dng=True (Rate-based unmask)
            if self.dng:
                # Model predicts logits → convert to probabilities → sample
                pred_element_logits = pred_element_1  # (B*mult, N, NUM_ELEMENTS)
                x_1_probs = F.softmax(pred_element_logits, dim=-1)  # (B*mult, N, NUM_ELEMENTS)
                x_1 = torch.distributions.Categorical(x_1_probs).sample() + 1  # (B*mult, N), 1-indexed
                
                # Unmask rate: dt / (1 - t)
                # Higher rate as t approaches 1 (more aggressive unmasking near the end)
                unmask_rate = dt / (1.0 - t + 1e-6)
                will_unmask = torch.rand_like(prim_slab_element_t.float()) < unmask_rate
                will_unmask = will_unmask & (prim_slab_element_t == MASK_TOKEN_INDEX)  # Only unmask MASK tokens
                
                # Apply unmask transition
                prim_slab_element_t = torch.where(will_unmask, x_1, prim_slab_element_t)

            # Ensure padding atoms remain at zero
            prim_slab_coords_t = prim_slab_coords_t * prim_slab_atom_mask.unsqueeze(-1)
            ads_rel_pos_t = ads_rel_pos_t * ads_atom_mask.unsqueeze(-1)

            ads_mask_sum = ads_atom_mask.sum(dim=1, keepdim=True)
            current_center = (ads_rel_pos_t * ads_atom_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (ads_mask_sum.unsqueeze(-1) + 1e-8)
            ads_rel_pos_t = ads_rel_pos_t - current_center
            ads_rel_pos_t = ads_rel_pos_t * ads_atom_mask.unsqueeze(-1)

            # Store the current state if collecting trajectories
            # Note: For trajectory, we store ads_coords = ads_center + ads_rel_pos
            if return_trajectory:
                ads_coords_t_for_traj = ads_center_t.unsqueeze(1) + ads_rel_pos_t
                prim_slab_coord_trajectory.append(prim_slab_coords_t)
                ads_coord_trajectory.append(ads_coords_t_for_traj)
                lattice_trajectory.append(lattice_t)
                supercell_matrix_trajectory.append(supercell_matrix_t)
                scaling_factor_trajectory.append(scaling_factor_t)
                if self.dng:
                    prim_slab_element_trajectory.append(prim_slab_element_t)

        ### Final Refinement Step
        if refine_final:
            final_prim_slab_coords, final_ads_center, final_ads_rel_pos, final_lattice, final_supercell_matrix, final_scaling_factor = self._refine_step(
                prim_slab_coords_t=prim_slab_coords_t,
                ads_center_t=ads_center_t,
                ads_rel_pos_t=ads_rel_pos_t,
                lattice_t=lattice_t,
                supercell_matrix_t=supercell_matrix_t,
                scaling_factor_t=scaling_factor_t,
                prim_slab_atom_mask=prim_slab_atom_mask,
                ads_atom_mask=ads_atom_mask,
                multiplicity=multiplicity,
                network_condition_kwargs=network_condition_kwargs,
            )
        else:
            final_prim_slab_coords = prim_slab_coords_t
            final_ads_center = ads_center_t
            final_ads_rel_pos = ads_rel_pos_t
            final_lattice = lattice_t
            final_supercell_matrix = supercell_matrix_t
            final_scaling_factor = scaling_factor_t

        # Post-processing: center ads_rel_pos (subtract mean so that mean becomes zero)
        # This ensures relative positions are truly centered around the center
        ads_mask_sum = ads_atom_mask.sum(dim=1, keepdim=True)  # (B, 1)
        ads_rel_pos_mean = (final_ads_rel_pos * ads_atom_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (ads_mask_sum.unsqueeze(-1) + 1e-8)  # (B, 1, 3)
        final_ads_rel_pos = final_ads_rel_pos - ads_rel_pos_mean  # Center the relative positions
        final_ads_rel_pos = final_ads_rel_pos * ads_atom_mask.unsqueeze(-1)  # Ensure padding remains zero

        # Combine ads_center + ads_rel_pos to get final ads_coords for compatibility
        final_ads_coords = final_ads_center.unsqueeze(1) + final_ads_rel_pos  # (B, M, 3)

        # All outputs are already in raw space after preconditioned_network_forward
        # Prepare the output dictionary (all in raw space)
        # Note: We store ads_coords = ads_center + ads_rel_pos for compatibility with downstream code
        output = {
            "sampled_prim_slab_coords": final_prim_slab_coords,  # (batch_size * multiplicity, N, 3) raw space
            "sampled_ads_coords": final_ads_coords,  # (batch_size * multiplicity, M, 3) raw space (ads_center + ads_rel_pos)
            "sampled_lattice": final_lattice,  # (batch_size * multiplicity, 6) raw space (Angstrom, degrees)
            "sampled_supercell_matrix": final_supercell_matrix,  # (batch_size * multiplicity, 3, 3) raw space
            "sampled_scaling_factor": final_scaling_factor,  # (batch_size * multiplicity,) raw value
        }
        
        # Final element handling when dng=True (Discrete Flow Matching)
        if self.dng:
            # Final step: unmask all remaining MASK tokens using last prediction from loop
            # pred_element_1 already contains the model's prediction from the last iteration
            x_1_probs = F.softmax(pred_element_1, dim=-1)
            x_1_final = torch.distributions.Categorical(x_1_probs).sample() + 1  # 1-indexed
            
            # Unmask all remaining MASK tokens
            remaining_masks = (prim_slab_element_t == MASK_TOKEN_INDEX)
            final_prim_slab_element = torch.where(remaining_masks, x_1_final, prim_slab_element_t)

            final_prim_slab_element = final_prim_slab_element * prim_slab_atom_mask.long()
            
            # prim_slab_element_t is already integer tensor (1-indexed)
            output["sampled_prim_slab_element"] = final_prim_slab_element  # (batch_size * multiplicity, N)

        if return_trajectory:
            # Only append final refined values if refine_final=True
            # (if refine_final=False, the last step is already in trajectory from the loop)
            if refine_final:
                final_ads_coords_for_traj = final_ads_center.unsqueeze(1) + final_ads_rel_pos
                prim_slab_coord_trajectory.append(final_prim_slab_coords)
                ads_coord_trajectory.append(final_ads_coords_for_traj)
                lattice_trajectory.append(final_lattice)
                supercell_matrix_trajectory.append(final_supercell_matrix)
                scaling_factor_trajectory.append(final_scaling_factor)
                if self.dng:
                    prim_slab_element_trajectory.append(prim_slab_element_t)

            output["prim_slab_coord_trajectory"] = torch.stack(prim_slab_coord_trajectory, dim=0)
            output["ads_coord_trajectory"] = torch.stack(ads_coord_trajectory, dim=0)
            output["lattice_trajectory"] = torch.stack(lattice_trajectory, dim=0)
            output["supercell_matrix_trajectory"] = torch.stack(supercell_matrix_trajectory, dim=0)
            output["scaling_factor_trajectory"] = torch.stack(scaling_factor_trajectory, dim=0)
            if self.dng:
                output["prim_slab_element_trajectory"] = torch.stack(prim_slab_element_trajectory, dim=0)

        return output

    @torch.no_grad()
    def _refine_step(
        self,
        prim_slab_coords_t,
        ads_center_t,
        ads_rel_pos_t,
        lattice_t,
        supercell_matrix_t,
        scaling_factor_t,
        prim_slab_atom_mask,
        ads_atom_mask,
        multiplicity,
        network_condition_kwargs,
    ):
        """
        Performs a final refinement step by feeding the t=1 state through the model once more.
        """
        pred_prim_slab_coords_1, pred_ads_center_1, pred_ads_rel_pos_1, pred_lattice_1, pred_supercell_matrix_1, pred_scaling_factor_1 = self.preconditioned_network_forward(
            noised_prim_slab_coords=prim_slab_coords_t,
            noised_ads_center=ads_center_t,
            noised_ads_rel_pos=ads_rel_pos_t,
            noised_lattice=lattice_t,
            noised_supercell_matrix=supercell_matrix_t,
            noised_scaling_factor=scaling_factor_t,
            time=1.0,
            training=False,
            network_condition_kwargs=network_condition_kwargs,
        )

        final_prim_slab_coords = pred_prim_slab_coords_1
        final_ads_center = pred_ads_center_1
        final_ads_rel_pos = pred_ads_rel_pos_1
        final_lattice = pred_lattice_1
        final_supercell_matrix = pred_supercell_matrix_1
        final_scaling_factor = pred_scaling_factor_1

        # Ensure padding atoms remain at zero
        final_prim_slab_coords = final_prim_slab_coords * prim_slab_atom_mask.unsqueeze(-1)
        final_ads_rel_pos = final_ads_rel_pos * ads_atom_mask.unsqueeze(-1)

        return final_prim_slab_coords, final_ads_center, final_ads_rel_pos, final_lattice, final_supercell_matrix, final_scaling_factor
