from einops import rearrange
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from src.models.utils import LinearNoBias
from src.models.transformers import DiT, PositionalEmbedder

# Number of elements for one-hot encoding (covers most elements in periodic table)
NUM_ELEMENTS = 100

# Discrete Flow Matching constants (for dng=True mode)
MASK_TOKEN_INDEX = 0  # Mask token index (used as prior in DFM)
NUM_ELEMENTS_WITH_MASK = NUM_ELEMENTS + 1  # 101 (0=MASK, 1-100=elements)


class AtomAttentionEncoder(Module):
    """
    Encoder that jointly processes primitive slab and adsorbate atoms.
    
    Input:
        - prim_slab_x_t: (B, N, 3) noisy prim_slab coordinates
        - ads_x_t: (B, M, 3) noisy adsorbate coordinates
        - l_t: (B, 6) noisy lattice parameters
        - sm_t: (B, 3, 3) noisy supercell matrix (normalized)
        - t: timesteps
        - feats: dictionary containing ref_prim_slab_element, ref_ads_element, masks, etc.
    
    Output:
        - joint representation: (B, N+M, atom_s)
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

        # Prim slab features: atom_pad_mask (1) + element one-hot
        # dng=True: includes MASK token (NUM_ELEMENTS_WITH_MASK = 101)
        # dng=False: no MASK token (NUM_ELEMENTS = 100)
        if dng:
            prim_slab_feature_dim = 1 + NUM_ELEMENTS_WITH_MASK  # 102
        else:
            prim_slab_feature_dim = 1 + NUM_ELEMENTS  # 101
        self.embed_prim_slab_features = LinearNoBias(prim_slab_feature_dim, atom_s)
        
        # Adsorbate features: ref_pos (3) + atom_pad_mask (1) + bind_atomic_num (NUM_ELEMENTS) + element one-hot (NUM_ELEMENTS)
        ads_feature_dim = 3 + 1 + NUM_ELEMENTS + NUM_ELEMENTS
        self.embed_ads_features = LinearNoBias(ads_feature_dim, atom_s)

        self.positional_encoding = positional_encoding
        if self.positional_encoding:
            self.pos_emb = PositionalEmbedder(atom_s)

        # Position embedding for noisy coords
        self.x_to_q_trans = LinearNoBias(3, atom_s)

        # Lattice is 6-dimensional (a, b, c, alpha, beta, gamma)
        self.l_to_q_trans = LinearNoBias(6, atom_s)

        # Supercell matrix is 9-dimensional (flattened 3x3)
        self.sm_to_q_trans = LinearNoBias(9, atom_s)

        # Scaling factor is scalar (1-dimensional)
        self.sf_to_q_trans = LinearNoBias(1, atom_s)

        self.atom_encoder = DiT(
            dim=atom_s,
            depth=atom_encoder_depth,
            heads=atom_encoder_heads,
            attention_impl=attention_impl,
            activation_checkpointing=activation_checkpointing,
            use_energy_cond=use_energy_cond,
        )

    def forward(self, prim_slab_x_t, ads_x_t, l_t, sm_t, sf_t, t, feats, multiplicity=1, prim_slab_element_t: Optional[torch.Tensor] = None):
        """
        Args:
            prim_slab_x_t: (B*mult, N, 3) noisy prim_slab coordinates
            ads_x_t: (B*mult, M, 3) noisy adsorbate coordinates
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
        
        prim_slab_mask = feats["prim_slab_atom_pad_mask"].bool()
        ads_mask = feats["ads_atom_pad_mask"].bool()

        # === Prim slab features ===
        # Features: atom_pad_mask (1) + element one-hot
        # dng=True: prim_slab_element_t is integer tensor (0=MASK, 1-100=elements) → one-hot with MASK
        # dng=False: use ref_prim_slab_element directly → one-hot without MASK
        if prim_slab_element_t is not None and self.dng:
            # dng=True: prim_slab_element_t is integer tensor (B*mult, N_actual)
            # Values: 0=MASK, 1-100=element atomic numbers
            B_mult = prim_slab_element_t.shape[0]
            N_actual = prim_slab_element_t.shape[1]
            B_x, N_x, _ = prim_slab_x_t.shape
            
            # Ensure N_actual matches prim_slab_x_t
            if N_actual != N_x:
                N_actual = N_x
                prim_slab_element_t = prim_slab_element_t[:, :N_actual]
            
            # Convert integer tensor to one-hot with MASK token
            # prim_slab_element_t: (B*mult, N_actual) integer tensor (0=MASK, 1-100=elements)
            prim_slab_element_onehot = F.one_hot(
                prim_slab_element_t.clamp(0, NUM_ELEMENTS_WITH_MASK - 1).long(),
                num_classes=NUM_ELEMENTS_WITH_MASK
            ).float()  # (B*mult, N_actual, NUM_ELEMENTS_WITH_MASK)
            
            # mask needs multiplicity applied and sliced to match N_actual
            prim_slab_mask_for_feats = feats["prim_slab_atom_pad_mask"].repeat_interleave(multiplicity, 0)
            prim_slab_mask_for_feats = prim_slab_mask_for_feats[:, :N_actual]
            
            # Update N to N_actual for later use
            N = N_actual
        else:
            # dng=False: existing approach (no MASK token)
            prim_slab_element_onehot = F.one_hot(
                feats["ref_prim_slab_element"].clamp(0, NUM_ELEMENTS - 1), 
                num_classes=NUM_ELEMENTS
            ).float()  # (B, N, NUM_ELEMENTS)
            prim_slab_mask_for_feats = feats["prim_slab_atom_pad_mask"]  # (B, N)
        
        prim_slab_feats = torch.cat([
            prim_slab_mask_for_feats.unsqueeze(-1),  # (B*mult, N, 1) or (B, N, 1)
            prim_slab_element_onehot,  # (B*mult, N, NUM_ELEMENTS) or (B, N, NUM_ELEMENTS)
        ], dim=-1)
        
        c_prim_slab = self.embed_prim_slab_features(prim_slab_feats)  # (B*mult, N, atom_s) or (B, N, atom_s)
        
        # Apply multiplicity only when not in dng mode with prim_slab_element_t
        if prim_slab_element_t is None or not self.dng:
            # Existing approach: (B, N, atom_s) -> (B*mult, N, atom_s)
            c_prim_slab = c_prim_slab.repeat_interleave(multiplicity, 0)

        # === Adsorbate features ===
        # Features: atom_pad_mask (1) + element one-hot (NUM_ELEMENTS)
        ads_element_onehot = F.one_hot(
            feats["ref_ads_element"].clamp(0, NUM_ELEMENTS - 1), 
            num_classes=NUM_ELEMENTS
        ).float()  # (B, M, NUM_ELEMENTS)
        
        ref_ads_pos = feats["ref_ads_pos"] # (B, M, 3)
        
        bind_atom_idx = feats["bind_ads_atom"].clamp(0, NUM_ELEMENTS - 1) # (B)
        bind_atom_onehot = F.one_hot(bind_atom_idx, num_classes=NUM_ELEMENTS).float() # (B, NUM_ELEMENTS)
        bind_atom_expanded = bind_atom_onehot.unsqueeze(1).expand(-1, M, -1) # (B, M, NUM_ELEMENTS)
        
        ads_feats = torch.cat([
            ref_ads_pos, # (B, M, 3)
            feats["ads_atom_pad_mask"].unsqueeze(-1),  # (B, M, 1)
            bind_atom_expanded, # (B, M, NUM_ELEMENTS)
            ads_element_onehot,  # (B, M, NUM_ELEMENTS)
        ], dim=-1)
        
        c_ads = self.embed_ads_features(ads_feats)  # (B, M, atom_s)
        # ads always needs multiplicity applied
        c_ads = c_ads.repeat_interleave(multiplicity, 0)  # (B*mult, M, atom_s)

        # === Concat prim_slab and adsorbate ===
        # Now both c_prim_slab and c_ads are in (B*mult, ...) format
        # N may have been updated to N_actual in dng=True branch
        c = torch.cat([c_prim_slab, c_ads], dim=1)  # (B*mult, N+M, atom_s)
        joint_mask = torch.cat([prim_slab_mask_for_feats, ads_mask.repeat_interleave(multiplicity, 0)], dim=1)  # (B*mult, N+M)

        q = c

        # === Add noisy positions ===
        # Concat noisy coords: prim_slab_x_t (B*mult, N, 3) + ads_x_t (B*mult, M, 3)
        # Ensure prim_slab_x_t matches N (may need slicing if N was updated)
        if prim_slab_x_t.shape[1] != N:
            prim_slab_x_t = prim_slab_x_t[:, :N, :]
        x_t = torch.cat([prim_slab_x_t, ads_x_t], dim=1)  # (B*mult, N+M, 3)
        q = q + self.x_to_q_trans(x_t)

        # Add noisy lattice (l_t is (B*mult, 6): a, b, c, alpha, beta, gamma)
        l_embed = self.l_to_q_trans(l_t)
        q = q + rearrange(l_embed, "b d -> b 1 d")

        # Add noisy supercell matrix (sm_t is (B*mult, 3, 3) -> flatten to (B*mult, 9))
        sm_flat = rearrange(sm_t, "b i j -> b (i j)")  # (B*mult, 9)
        sm_embed = self.sm_to_q_trans(sm_flat)
        q = q + rearrange(sm_embed, "b d -> b 1 d")

        # Add noisy scaling factor (sf_t is (B*mult,) -> expand to (B*mult, 1))
        sf_embed = self.sf_to_q_trans(sf_t.unsqueeze(-1))  # (B*mult, atom_s)
        q = q + rearrange(sf_embed, "b d -> b 1 d")

        # Get energy conditioning if enabled
        energy = None
        if self.use_energy_cond and "ref_energy" in feats:
            energy = feats["ref_energy"].repeat_interleave(multiplicity, 0)  # (B*mult,)

        # Pass through transformer (joint attention over all atoms)
        q = self.atom_encoder(q, t, joint_mask, energy=energy)  # (B*mult, N+M, atom_s)

        return q, N, M


class AtomAttentionDecoder(Module):
    """
    Decoder that outputs predictions for prim_slab coords, adsorbate coords, lattice, and supercell matrix.
    
    Input:
        - x: (B, N+M, atom_s) joint atom representations
        - n_prim_slab: int, number of prim_slab atoms (N)
    
    Output:
        - prim_slab_r_update: (B, N, 3) prim_slab coordinate updates
        - ads_r_update: (B, M, 3) adsorbate coordinate updates
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
        
        # Projection heads for adsorbate coords
        self.feats_to_ads_coords = nn.Sequential(
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
            ads_r_update: (B*mult, M, 3) adsorbate coordinate updates
            l_update: (B*mult, 6) lattice updates
            sm_update: (B*mult, 3, 3) supercell matrix updates (normalized)
            sf_update: (B*mult,) scaling factor updates (normalized)
        """
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

        # Split back to prim_slab and adsorbate
        x_prim_slab = x[:, :n_prim_slab, :]  # (B*mult, N, atom_s)
        x_ads = x[:, n_prim_slab:, :]  # (B*mult, M, atom_s)

        # Prim slab position prediction
        prim_slab_r_update = self.feats_to_prim_slab_coords(x_prim_slab)

        # Adsorbate position prediction
        ads_r_update = self.feats_to_ads_coords(x_ads)

        # Global pooling (use prim_slab atoms only)
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
            return prim_slab_r_update, ads_r_update, l_update, sm_update, sf_update, prim_slab_element_update
        else:
            return prim_slab_r_update, ads_r_update, l_update, sm_update, sf_update


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
