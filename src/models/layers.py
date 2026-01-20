from einops import rearrange
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from src.models.utils import LinearNoBias, center_random_augmentation
from src.models.transformers import DiT, PositionalEmbedder

# Number of elements for one-hot encoding (covers most elements in periodic table)
NUM_ELEMENTS = 100

# Discrete Flow Matching constants (for dng=True mode)
MASK_TOKEN_INDEX = 0  # Mask token index (used as prior in DFM)
NUM_ELEMENTS_WITH_MASK = NUM_ELEMENTS + 1  # 101 (0=MASK, 1-100=elements)


def lattice_params_to_matrix_torch(lengths, angles):
    """
    lengths: (B, 3) a, b, c
    angles: (B, 3) alpha, beta, gamma (in degrees)
    Returns: (B, 3, 3) Cell matrix (rows are lattice vectors)
    """
    a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
    alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]
    
    zeros = torch.zeros_like(a)
    
    # Deg to Rad
    alpha_rad = alpha * torch.pi / 180.0
    beta_rad = beta * torch.pi / 180.0
    gamma_rad = gamma * torch.pi / 180.0
    
    val = (torch.cos(alpha_rad) * torch.cos(beta_rad) - torch.cos(gamma_rad)) / (torch.sin(alpha_rad) * torch.sin(beta_rad))
    # Clamp val to avoid nan in sqrt
    val = torch.clamp(val, -1.0, 1.0)
    
    gamma_star = torch.acos(val)
    
    vector_a = torch.stack([a * torch.sin(beta_rad), zeros, a * torch.cos(beta_rad)], dim=1)
    vector_b = torch.stack([-b * torch.sin(alpha_rad) * torch.cos(gamma_star), b * torch.sin(alpha_rad) * torch.sin(gamma_star), b * torch.cos(alpha_rad)], dim=1)
    vector_c = torch.stack([zeros, zeros, c], dim=1)
    
    return torch.stack([vector_a, vector_b, vector_c], dim=1)


class GaussianRBF(nn.Module):
    def __init__(self, n_rbf=24, cutoff=4.0, start=0.0):
        super().__init__()
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(start, cutoff, n_rbf), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(0.5), requires_grad=False)

    def forward(self, dists, mask):
        # dists: (B, M, M)
        # mask: (B, M)
        
        dists_expanded = dists.unsqueeze(-1)
        centers = self.centers.view(1, 1, 1, -1)
        
        # (B, M, M, n_rbf)
        rbf = torch.exp(-self.gamma * (dists_expanded - centers)**2)
        
        # Masking
        pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1) 
        
        # Self-interaction exclusion
        B, M = mask.shape
        identity = torch.eye(M, device=mask.device).unsqueeze(0).expand(B, -1, -1)
        pair_mask = pair_mask * (1 - identity)
        
        rbf = rbf * pair_mask.unsqueeze(-1)
        
        # Mask-Normalized Mean Aggregation
        rbf_sum = rbf.sum(dim=2)
        neighbor_counts = pair_mask.sum(dim=2, keepdim=True)
        rbf_mean = rbf_sum / (neighbor_counts + 1e-8)
        
        return rbf_mean


class AtomAttentionEncoder(Module):
    """
    Encoder that jointly processes primitive slab and adsorbate atoms.
    
    Input:
        - prim_slab_x_t: (B, N, 3) noisy prim_slab coordinates
        - ads_center_t: (B, 3) noisy adsorbate center (center of mass)
        - ads_rel_pos_t: (B, M, 3) noisy adsorbate relative positions from center
        - l_t: (B, 6) noisy lattice parameters
        - sm_t: (B, 3, 3) noisy supercell matrix (normalized)
        - sf_t: (B,) noisy scaling factor
        - t: timesteps
        - feats: dictionary containing ref_prim_slab_element, ref_ads_element, masks, etc.
    
    Note:
        Internally reconstructs ads_x_t = ads_center_t + ads_rel_pos_t for position embedding.
    """
    def __init__(
        self,
        atom_s,
        atom_encoder_depth,
        atom_encoder_heads,
        attention_impl,
        positional_encoding=False,
        activation_checkpointing=False,
        dng: bool = False,
        use_energy_cond: bool = False,
    ):
        super().__init__()
        
        self.dng = dng
        self.use_energy_cond = use_energy_cond

        # 1. Base Feature Embedding
        if dng:
            prim_slab_feature_dim = 1 + NUM_ELEMENTS_WITH_MASK
        else:
            prim_slab_feature_dim = 1 + NUM_ELEMENTS
        self.embed_prim_slab_features = LinearNoBias(prim_slab_feature_dim, atom_s)
        
        ads_feature_dim = 3 + 1 + NUM_ELEMENTS
        self.embed_ads_features = LinearNoBias(ads_feature_dim, atom_s)

        # 2. Coordinate Embeddings
        self.x_to_q_trans = LinearNoBias(3, atom_s) 
        
        # RBF Density Embedding (Adsorbate Only)
        self.rbf_fn = GaussianRBF(n_rbf=24, cutoff=4.0)
        self.rbf_to_q_trans = LinearNoBias(24, atom_s)
        self.rbf_gate = nn.Parameter(torch.tensor(0.1))

        # 3. Global Condition Embeddings
        self.l_to_q_trans = LinearNoBias(6, atom_s)
        self.sm_to_q_trans = LinearNoBias(9, atom_s)
        self.sf_to_q_trans = LinearNoBias(1, atom_s)
        
        # Explicit Supercell Lattice Embedding
        self.super_l_to_q_trans = LinearNoBias(9, atom_s)

        self.atom_encoder = DiT(
            dim=atom_s,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
            use_energy_cond=use_energy_cond,
        )

    def forward(self, prim_slab_x_t, ads_center_t, ads_rel_pos_t, l_t, sm_t, sf_t, t, feats, multiplicity=1, prim_slab_element_t: Optional[torch.Tensor] = None):
        """
        Args:
            prim_slab_x_t: (B*mult, N, 3) noisy prim_slab coordinates
            ads_center_t: (B*mult, 3) noisy adsorbate center
            ads_rel_pos_t: (B*mult, M, 3) noisy adsorbate relative positions
            l_t: (B*mult, 6) noisy lattice parameters
            sm_t: (B*mult, 3, 3) noisy supercell matrix (normalized)
            sf_t: (B*mult,) noisy scaling factor (normalized)
            t: (B*mult,) timesteps
            feats: feature dictionary (not yet repeated for multiplicity)
            multiplicity: number of samples per input
        
        Returns:
            q: (B*mult, N+M, atom_s) joint atom representations
            n_prim_slab: int, number of prim_slab atoms (N)
            n_ads: int, number of adsorbate atoms (M)
        """        
        B, N, _ = feats["prim_slab_cart_coords"].shape
        M = feats["ads_cart_coords"].shape[1]

        # --- 1. Basic Node Featurization ---
        if prim_slab_element_t is not None and self.dng:
             # Handle DNG case (Discrete Flow Matching)
             B_mult = prim_slab_element_t.shape[0]
             N_actual = prim_slab_element_t.shape[1]
             prim_slab_element_onehot = F.one_hot(prim_slab_element_t.clamp(0, NUM_ELEMENTS_WITH_MASK - 1).long(), num_classes=NUM_ELEMENTS_WITH_MASK).float()
             
             # Expand and slice mask
             prim_slab_mask_for_feats = feats["prim_slab_atom_pad_mask"].repeat_interleave(multiplicity, 0)
             if prim_slab_mask_for_feats.shape[1] > N_actual:
                 prim_slab_mask_for_feats = prim_slab_mask_for_feats[:, :N_actual]
             
             N = N_actual
        else:
             prim_slab_element_onehot = F.one_hot(feats["ref_prim_slab_element"].clamp(0, NUM_ELEMENTS - 1), num_classes=NUM_ELEMENTS).float()
             prim_slab_mask_for_feats = feats["prim_slab_atom_pad_mask"]

        prim_slab_feats = torch.cat([prim_slab_mask_for_feats.unsqueeze(-1), prim_slab_element_onehot], dim=-1)
        c_prim_slab = self.embed_prim_slab_features(prim_slab_feats)
        
        if prim_slab_element_t is None or not self.dng:
            c_prim_slab = c_prim_slab.repeat_interleave(multiplicity, 0)

        ads_element_onehot = F.one_hot(feats["ref_ads_element"].clamp(0, NUM_ELEMENTS - 1), num_classes=NUM_ELEMENTS).float()
        
        # Center random augmentation for adsorbate positions during training
        ref_ads_pos_centered = center_random_augmentation(feats["ref_ads_pos"], feats["ads_atom_pad_mask"], s_trans=0.0, augmentation=True, centering=True)
        
        ads_feats = torch.cat([ref_ads_pos_centered, feats["ads_atom_pad_mask"].unsqueeze(-1), ads_element_onehot], dim=-1)
        c_ads = self.embed_ads_features(ads_feats).repeat_interleave(multiplicity, 0)

        c = torch.cat([c_prim_slab, c_ads], dim=1)
        
        # Create joint mask
        ads_mask_expanded = feats["ads_atom_pad_mask"].repeat_interleave(multiplicity, 0)
        joint_mask = torch.cat([prim_slab_mask_for_feats, ads_mask_expanded], dim=1)
        
        q = c

        # --- 2. Coordinate Handling ---
        # Reconstruct Full Noisy Coordinates: ads_x_t = center + rel_pos
        ads_x_t = ads_center_t.unsqueeze(1) + ads_rel_pos_t
        
        # Slice prim_slab_x_t if needed (for DNG)
        if prim_slab_x_t.shape[1] != N: 
            prim_slab_x_t = prim_slab_x_t[:, :N, :]
            
        x_t = torch.cat([prim_slab_x_t, ads_x_t], dim=1)  # (B*mult, N+M, 3)

        # Coordinate Embedding
        q = q + self.x_to_q_trans(x_t)

        ads_coords_only = x_t[:, N:, :]
        ads_dists = torch.cdist(ads_coords_only, ads_coords_only)
        
        rbf_feats_ads = self.rbf_fn(ads_dists, ads_mask_expanded)
        rbf_embed_ads = self.rbf_to_q_trans(rbf_feats_ads)
        
        q[:, N:, :] = q[:, N:, :] + self.rbf_gate * rbf_embed_ads

        # --- 3. Global Conditions ---
        l_embed = self.l_to_q_trans(l_t)
        q = q + rearrange(l_embed, "b d -> b 1 d")
        
        sm_flat = rearrange(sm_t, "b i j -> b (i j)")
        sm_embed = self.sm_to_q_trans(sm_flat)
        q = q + rearrange(sm_embed, "b d -> b 1 d")

        sf_embed = self.sf_to_q_trans(sf_t.unsqueeze(-1))
        q = q + rearrange(sf_embed, "b d -> b 1 d")

        # Explicit Supercell Lattice Embedding
        # Calculate Supercell Lattice: S * L
        prim_lat_matrix = lattice_params_to_matrix_torch(l_t[:, :3], l_t[:, 3:]) 
        super_lat_matrix = torch.bmm(sm_t, prim_lat_matrix)
        super_lat_flat = rearrange(super_lat_matrix, "b i j -> b (i j)")
        super_lat_embed = self.super_l_to_q_trans(super_lat_flat)
        q = q + rearrange(super_lat_embed, "b d -> b 1 d")

        # --- 4. Transformer Forward ---
        energy = None
        if self.use_energy_cond and "ref_energy" in feats:
            energy = feats["ref_energy"].repeat_interleave(multiplicity, 0)

        q = self.atom_encoder(q, t, joint_mask, energy=energy)

        return q, N, M


class AtomAttentionDecoder(Module):
    """
    Decoder that outputs predictions for prim_slab coords, adsorbate center, adsorbate relative positions, lattice, and supercell matrix.
    
    Input:
        - x: (B, N+M, atom_s) joint atom representations
        - n_prim_slab: int, number of prim_slab atoms (N)
    
    Output:
        - prim_slab_r_update: (B, N, 3) prim_slab coordinate updates
        - ads_center_update: (B, 3) adsorbate center updates
        - ads_rel_pos_update: (B, M, 3) adsorbate relative position updates
        - l_update: (B, 6) lattice updates
        - s_update: (B, 3, 3) supercell matrix updates (normalized)
    """
    
    def __init__(
        self,
        atom_s,
        atom_decoder_depth,
        atom_decoder_heads,
        attention_impl,
        activation_checkpointing=False,
        dng: bool = False,
        use_energy_cond: bool = False,
    ):
        super().__init__()
        self.use_energy_cond = use_energy_cond
        self.atom_decoder = DiT(
            dim=atom_s,
            depth=atom_decoder_depth,
            heads=atom_decoder_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
            use_energy_cond=use_energy_cond,
        )

        # Projection heads for prim_slab coords
        self.feats_to_prim_slab_coords = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )
        
        # Projection head for adsorbate center (global pooling -> 3D center)
        self.feats_to_ads_center = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )
        
        # Projection heads for adsorbate relative positions (per-atom)
        self.feats_to_ads_rel_pos = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 3)
        )

        # Lattice output is 6-dimensional (a, b, c, alpha, beta, gamma)
        self.feats_to_lattice = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 6)
        )

        # Supercell matrix output is 9-dimensional (flattened 3x3)
        self.feats_to_supercell_matrix = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 9)
        )

        # Scaling factor output is 1-dimensional (scalar)
        self.feats_to_scaling_factor = nn.Sequential(
            nn.LayerNorm(atom_s), LinearNoBias(atom_s, 1)
        )
        
        # Add element prediction head when dng=True
        if dng:
            # Output logits (softmax is applied during loss/vector field computation)
            self.feats_to_prim_slab_element = nn.Sequential(
                nn.LayerNorm(atom_s),
                LinearNoBias(atom_s, NUM_ELEMENTS)  # output logits
            )

    def forward(self, x, t, feats, n_prim_slab, n_ads=None, multiplicity=1):
        """
        Args:
            x: (B*mult, N+M, atom_s) joint atom representations
            t: (B*mult,) timesteps
            feats: feature dictionary (not yet repeated for multiplicity)
            n_prim_slab: int, number of prim_slab atoms (N)
            n_ads: int, optional, number of adsorbate atoms (M). If None, inferred from x.shape[1] - n_prim_slab
            multiplicity: number of samples per input
        
        Returns:
            prim_slab_r_update: (B*mult, N, 3) prim_slab coordinate updates
            ads_center_update: (B*mult, 3) adsorbate center updates
            ads_rel_pos_update: (B*mult, M, 3) adsorbate relative position updates
            l_update: (B*mult, 6) lattice updates
            sm_update: (B*mult, 3, 3) supercell matrix updates (normalized)
            sf_update: (B*mult,) scaling factor updates (normalized)
        """

        # print(
        #     "[DEBUG][Decoder:in]",
        #     "n_prim_slab:", n_prim_slab,
        #     "prim_slab_mask:", feats["prim_slab_atom_pad_mask"].shape,
        #     "ads_mask:", feats["ads_atom_pad_mask"].shape,
        # )

        prim_slab_mask = feats["prim_slab_atom_pad_mask"]
        ads_mask = feats["ads_atom_pad_mask"]
        
        prim_slab_mask = prim_slab_mask.repeat_interleave(multiplicity, 0)
        ads_mask = ads_mask.repeat_interleave(multiplicity, 0)
        
        joint_mask = torch.cat([prim_slab_mask, ads_mask], dim=1)

        # Get energy conditioning if enabled
        energy = None
        if self.use_energy_cond and "ref_energy" in feats:
            energy = feats["ref_energy"].repeat_interleave(multiplicity, 0)  # (B*mult,)

        x = self.atom_decoder(x, t, joint_mask, energy=energy)

        # print(
        #     "[DEBUG][Decoder:joint_mask]",
        #     "joint_mask:", joint_mask.shape,
        #     "x:", x.shape,
        # )

        # Split back to prim_slab and adsorbate
        x_prim_slab = x[:, :n_prim_slab, :]  # (B*mult, N, atom_s)
        x_ads = x[:, n_prim_slab:, :]  # (B*mult, M, atom_s)

        # Prim slab position prediction
        prim_slab_r_update = self.feats_to_prim_slab_coords(x_prim_slab)

        # Adsorbate center prediction (global pooling over adsorbate atoms)
        num_ads_atoms = ads_mask.sum(dim=1, keepdim=True)  # (B*mult, 1)
        x_ads_global = torch.sum(x_ads * ads_mask[..., None], dim=1) / (num_ads_atoms + 1e-8)  # (B*mult, atom_s)
        ads_center_update = self.feats_to_ads_center(x_ads_global)  # (B*mult, 3)
        
        # Adsorbate relative position prediction (per-atom)
        ads_rel_pos_update = self.feats_to_ads_rel_pos(x_ads)  # (B*mult, M, 3)

        # Global pooling (use prim_slab atoms only) for lattice/supercell/scaling_factor
        num_prim_slab_atoms = prim_slab_mask.sum(dim=1, keepdim=True)  # (B*mult, 1)
        x_global = torch.sum(x_prim_slab * prim_slab_mask[..., None], dim=1) / (num_prim_slab_atoms + 1e-8)

        # Lattice prediction
        l_update = self.feats_to_lattice(x_global)  # (B*mult, 6)

        # Supercell matrix prediction
        sm_flat = self.feats_to_supercell_matrix(x_global)  # (B*mult, 9)
        sm_update = rearrange(sm_flat, "b (i j) -> b i j", i=3, j=3)  # (B*mult, 3, 3)

        # Scaling factor prediction
        sf_update = self.feats_to_scaling_factor(x_global).squeeze(-1)  # (B*mult,)
        
        # Add element prediction when dng=True
        if hasattr(self, 'feats_to_prim_slab_element'):
            prim_slab_element_update = self.feats_to_prim_slab_element(x_prim_slab)
            # (B*mult, N, NUM_ELEMENTS) - logits format (softmax is applied during loss/vector field computation)
            return prim_slab_r_update, ads_center_update, ads_rel_pos_update, l_update, sm_update, sf_update, prim_slab_element_update
        else:
            return prim_slab_r_update, ads_center_update, ads_rel_pos_update, l_update, sm_update, sf_update


class TokenTransformer(Module):
    """
    Token-level transformer for joint prim_slab and adsorbate tokens.
    """
    
    def __init__(
        self,
        token_s,
        token_transformer_depth,
        token_transformer_heads,
        attention_impl,
        activation_checkpointing=False,
        use_energy_cond: bool = False,
    ):
        super().__init__()

        self.use_energy_cond = use_energy_cond
        self.s_init = nn.Linear(token_s, token_s, bias=False)

        # Normalization layers
        self.s_mlp = nn.Sequential(
            nn.LayerNorm(token_s), nn.Linear(token_s, token_s, bias=False)
        )

        self.token_transformer = DiT(
            dim=token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
            use_energy_cond=use_energy_cond,
        )

    def forward(self, x, t, feats, multiplicity=1):
        # Joint token mask (prim_slab + adsorbate)
        prim_slab_token_mask = feats["prim_slab_token_pad_mask"]
        ads_token_mask = feats["ads_token_pad_mask"]
        token_mask = torch.cat([prim_slab_token_mask, ads_token_mask], dim=1)
        token_mask = token_mask.repeat_interleave(multiplicity, 0)

        # Get energy conditioning if enabled
        energy = None
        if self.use_energy_cond and "ref_energy" in feats:
            energy = feats["ref_energy"].repeat_interleave(multiplicity, 0)  # (B*mult,)

        # Initialize single embeddings
        s_init = self.s_init(x)  # (batch_size, num_tokens, token_s)

        s = self.s_mlp(s_init)
        x = self.token_transformer(s, t, token_mask, energy=energy)

        return x
