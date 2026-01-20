"""Prior samplers for flow matching."""

from typing import Any, Dict, List, Protocol

import torch
from torch.distributions import LogNormal, Uniform


class PriorSampler(Protocol):
    """Protocol for prior samplers used in flow matching."""

    def sample(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Sample from the prior distribution.

        Args:
            data: Dictionary containing at least:
                - prim_slab_cart_coords: (B, N, 3) target primitive slab coordinates
                - prim_slab_atom_mask: (B, N) boolean mask for valid prim_slab atoms
                - ads_cart_coords: (B, M, 3) target adsorbate coordinates
                - ads_atom_mask: (B, M) boolean mask for valid adsorbate atoms

        Returns:
            Dictionary containing:
                - prim_slab_coords_0: (B, N, 3) sampled prior primitive slab coordinates
                - ads_coords_0: (B, M, 3) sampled prior adsorbate coordinates
                - lattice_0: (B, 6) sampled prior lattice parameters (a, b, c, alpha, beta, gamma)
                - supercell_matrix_0: (B, 3, 3) sampled prior supercell matrix (normalized)
        """
        ...


class CatPriorSampler:
    """Prior sampler for flow matching.
    
    Samples all priors needed for flow matching:
    - prim_slab_coords_0: N(0, coord_std^2) in normalized space, then denormalized to raw space (Angstrom)
    - ads_coords_0: N(0, coord_std^2) in normalized space, then denormalized to raw space (Angstrom)
    - lattice_0: LogNormal (lengths) + Uniform (angles) - raw space (Angstrom, degrees)
    - supercell_matrix_0: N(0, 1) in normalized space, then denormalized to raw space
    - scaling_factor_0: N(0, coord_std^2) - Gaussian (raw space, no normalization)
    
    Note: prim_slab_coords, ads_coords, and supercell_matrix are sampled in normalized space and then
    denormalized to raw space. Lattice uses non-Gaussian distributions in raw space. Scaling factor uses raw Gaussian prior.
    """

    def __init__(
        self,
        coord_std: float = 1.0,
        lognormal_loc: List[float] = None,
        lognormal_scale: List[float] = None,
        uniform_low: float = 60.0,
        uniform_high: float = 120.0,
        uniform_eps: float = 0.1,
        # Supercell matrix normalization parameters
        supercell_mean: List[List[float]] = None,
        supercell_std: List[List[float]] = None,
        # Coordinate standardization parameters (computed from train set)
        prim_slab_coord_mean: List[float] = None,
        prim_slab_coord_std: List[float] = None,
        # Adsorbate center (center of mass) standardization parameters
        ads_center_mean: List[float] = None,
        ads_center_std: List[float] = None,
        # Adsorbate relative position standardization parameters
        ads_rel_pos_mean: List[float] = None,
        ads_rel_pos_std: List[float] = None,
    ):
        self.coord_std = coord_std

        # LogNormal distribution for lattice lengths (a, b, c)
        self._lognormal = LogNormal(
            loc=torch.tensor(lognormal_loc),
            scale=torch.tensor(lognormal_scale),
        )

        # Uniform distribution for lattice angles (alpha, beta, gamma)
        self._uniform = Uniform(
            low=uniform_low - uniform_eps,
            high=uniform_high + uniform_eps,
        )
        
        self.supercell_mean = torch.tensor(supercell_mean, dtype=torch.float32)  # (3, 3)
        self.supercell_std = torch.tensor(supercell_std, dtype=torch.float32)  # (3, 3)
        
        self.prim_slab_coord_mean = torch.tensor(prim_slab_coord_mean, dtype=torch.float32)  # (3,)
        self.prim_slab_coord_std = torch.tensor(prim_slab_coord_std, dtype=torch.float32)  # (3,)
        
        # Adsorbate center (center of mass) normalization parameters
        self.ads_center_mean = torch.tensor(ads_center_mean, dtype=torch.float32)  # (3,)
        self.ads_center_std = torch.tensor(ads_center_std, dtype=torch.float32)  # (3,)
        
        # Adsorbate relative position normalization parameters
        self.ads_rel_pos_mean = torch.tensor(ads_rel_pos_mean, dtype=torch.float32)  # (3,)
        self.ads_rel_pos_std = torch.tensor(ads_rel_pos_std, dtype=torch.float32)  # (3,)

    def sample(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Sample from prior distributions for coords, lattice, and supercell matrix.
        
        Now samples ads_center and ads_rel_pos separately instead of ads_coords.
        """
        prim_slab_cart_coords = data["prim_slab_cart_coords"]
        prim_slab_atom_mask = data["prim_slab_atom_mask"]
        ads_rel_pos = data.get("ads_rel_pos")  # (B, M, 3)
        ads_atom_mask = data["ads_atom_mask"]

        batch_size, num_prim_slab_atoms, _ = prim_slab_cart_coords.shape
        num_ads_atoms = ads_rel_pos.shape[1] if ads_rel_pos is not None else ads_atom_mask.shape[1]
        device = prim_slab_cart_coords.device
        dtype = prim_slab_cart_coords.dtype

        # Sample primitive slab coordinates from Gaussian N(0, coord_std^2) in normalized space, then denormalize to raw space
        prim_slab_coords_0_normalized = (
            torch.randn(batch_size, num_prim_slab_atoms, 3, device=device, dtype=dtype)
            * self.coord_std
        )
        prim_slab_coords_0_normalized = prim_slab_coords_0_normalized * prim_slab_atom_mask.unsqueeze(-1)
        # Denormalize to raw space (Angstrom)
        prim_slab_coords_0 = self.denormalize_prim_slab_coords(prim_slab_coords_0_normalized)

        # Sample adsorbate center from Gaussian N(0, coord_std^2) in normalized space, then denormalize to raw space
        ads_center_0_normalized = (
            torch.randn(batch_size, 3, device=device, dtype=dtype)
            * self.coord_std
        )
        # Denormalize to raw space (Angstrom)
        ads_center_0 = self.denormalize_ads_center(ads_center_0_normalized)

        # # Sample adsorbate relative positions from Gaussian N(0, coord_std^2) in normalized space, then denormalize to raw space
        # ads_rel_pos_0_normalized = (
        #     torch.randn(batch_size, num_ads_atoms, 3, device=device, dtype=dtype)
        #     * self.coord_std
        # )
        # ads_rel_pos_0_normalized = ads_rel_pos_0_normalized * ads_atom_mask.unsqueeze(-1)
        # # Denormalize to raw space (Angstrom)
        # ads_rel_pos_0 = self.denormalize_ads_rel_pos(ads_rel_pos_0_normalized)

        noise = torch.randn(batch_size, num_ads_atoms, 3, device=device, dtype=dtype)
        noise_mask = ads_atom_mask.unsqueeze(-1)
        noise = noise * noise_mask

        noise_mean = noise.sum(dim=1, keepdim=True) / (ads_atom_mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        ads_rel_pos_0_normalized = (noise - noise_mean) * self.coord_std
        ads_rel_pos_0_normalized = ads_rel_pos_0_normalized * noise_mask

        ads_rel_pos_0 = self.denormalize_ads_rel_pos(ads_rel_pos_0_normalized)

        # Sample lattice lengths from LogNormal distribution
        lengths_0 = self._lognormal.sample((batch_size,))  # (B, 3)
        lengths_0 = lengths_0.to(device=device, dtype=dtype)

        # Sample lattice angles from Uniform distribution
        angles_0 = self._uniform.sample((batch_size, 3))  # (B, 3)
        angles_0 = angles_0.to(device=device, dtype=dtype)

        # Combine into lattice_0: (a, b, c, alpha, beta, gamma)
        lattice_0 = torch.cat([lengths_0, angles_0], dim=-1)  # (B, 6)

        # # Sample supercell matrix from standard normal N(0, 1) in normalized space, then denormalize to raw space
        # supercell_matrix_0_normalized = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)  # (B, 3, 3)
        # supercell_matrix_0 = self.denormalize_supercell(supercell_matrix_0_normalized)  # (B, 3, 3) raw space
        
        identity = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        noise = torch.randn(batch_size, 3, 3, device=device, dtype=dtype) * 0.1
        supercell_matrix_0 = identity + noise

        # Sample scaling factor from Gaussian N(0, coord_std^2) - no normalization
        scaling_factor_0 = torch.randn(batch_size, device=device, dtype=dtype) * self.coord_std  # (B,)

        return {
            "prim_slab_coords_0": prim_slab_coords_0,  # (B, N, 3) raw space (Angstrom)
            "ads_center_0": ads_center_0,  # (B, 3) raw space (Angstrom)
            "ads_rel_pos_0": ads_rel_pos_0,  # (B, M, 3) raw space (Angstrom)
            "lattice_0": lattice_0,  # (B, 6) raw space (Angstrom, degrees)
            "supercell_matrix_0": supercell_matrix_0,  # (B, 3, 3) raw space
            "scaling_factor_0": scaling_factor_0,  # (B,) raw value
        }

    def normalize_supercell(self, supercell_matrix: torch.Tensor) -> torch.Tensor:
        """Normalize supercell matrix using mean and std.
        
        Args:
            supercell_matrix: (B, 3, 3) raw supercell matrix
            
        Returns:
            (B, 3, 3) normalized supercell matrix
        """
        mean = self.supercell_mean.to(supercell_matrix.device, supercell_matrix.dtype)
        std = self.supercell_std.to(supercell_matrix.device, supercell_matrix.dtype)
        return (supercell_matrix - mean) / (std + 1e-8)

    def denormalize_supercell(self, supercell_matrix_normalized: torch.Tensor) -> torch.Tensor:
        """Denormalize supercell matrix back to original space.
        
        Args:
            supercell_matrix_normalized: (B, 3, 3) normalized supercell matrix
            
        Returns:
            (B, 3, 3) denormalized supercell matrix
        """
        mean = self.supercell_mean.to(supercell_matrix_normalized.device, supercell_matrix_normalized.dtype)
        std = self.supercell_std.to(supercell_matrix_normalized.device, supercell_matrix_normalized.dtype)
        return supercell_matrix_normalized * std + mean

    def normalize_prim_slab_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Standardize prim_slab coordinates using mean and std.
        
        Args:
            coords: (B, N, 3) raw prim_slab coordinates in Angstrom
            
        Returns:
            (B, N, 3) standardized coordinates
        """
        mean = self.prim_slab_coord_mean.to(coords.device, coords.dtype)
        std = self.prim_slab_coord_std.to(coords.device, coords.dtype)
        return (coords - mean) / (std + 1e-8)

    def denormalize_prim_slab_coords(self, coords_normalized: torch.Tensor) -> torch.Tensor:
        """Destandardize prim_slab coordinates back to original space.
        
        Args:
            coords_normalized: (B, N, 3) standardized coordinates
            
        Returns:
            (B, N, 3) denormalized coordinates in Angstrom
        """
        mean = self.prim_slab_coord_mean.to(coords_normalized.device, coords_normalized.dtype)
        std = self.prim_slab_coord_std.to(coords_normalized.device, coords_normalized.dtype)
        return coords_normalized * std + mean

    def normalize_ads_center(self, ads_center: torch.Tensor) -> torch.Tensor:
        """Standardize adsorbate center (center of mass) using mean and std.
        
        Args:
            ads_center: (B, 3) raw adsorbate center in Angstrom
            
        Returns:
            (B, 3) standardized center
        """
        mean = self.ads_center_mean.to(ads_center.device, ads_center.dtype)
        std = self.ads_center_std.to(ads_center.device, ads_center.dtype)
        return (ads_center - mean) / (std + 1e-8)

    def denormalize_ads_center(self, ads_center_normalized: torch.Tensor) -> torch.Tensor:
        """Destandardize adsorbate center back to original space.
        
        Args:
            ads_center_normalized: (B, 3) standardized center
            
        Returns:
            (B, 3) denormalized center in Angstrom
        """
        mean = self.ads_center_mean.to(ads_center_normalized.device, ads_center_normalized.dtype)
        std = self.ads_center_std.to(ads_center_normalized.device, ads_center_normalized.dtype)
        return ads_center_normalized * std + mean

    def normalize_ads_rel_pos(self, ads_rel_pos: torch.Tensor) -> torch.Tensor:
        """Standardize adsorbate relative positions using mean and std.
        
        Args:
            ads_rel_pos: (B, M, 3) raw adsorbate relative positions in Angstrom
            
        Returns:
            (B, M, 3) standardized relative positions
        """
        mean = self.ads_rel_pos_mean.to(ads_rel_pos.device, ads_rel_pos.dtype)
        std = self.ads_rel_pos_std.to(ads_rel_pos.device, ads_rel_pos.dtype)
        return (ads_rel_pos - mean) / (std + 1e-8)

    def denormalize_ads_rel_pos(self, ads_rel_pos_normalized: torch.Tensor) -> torch.Tensor:
        """Destandardize adsorbate relative positions back to original space.
        
        Args:
            ads_rel_pos_normalized: (B, M, 3) standardized relative positions
            
        Returns:
            (B, M, 3) denormalized relative positions in Angstrom
        """
        mean = self.ads_rel_pos_mean.to(ads_rel_pos_normalized.device, ads_rel_pos_normalized.dtype)
        std = self.ads_rel_pos_std.to(ads_rel_pos_normalized.device, ads_rel_pos_normalized.dtype)
        return ads_rel_pos_normalized * std + mean
