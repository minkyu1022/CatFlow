"""
LMDB Dataset with RAM caching and dynamic batch padding.

LMDB format:
- key: idx (int)
- value: dict {
    "primitive_slab": ASE.Atoms,
    "supercell_matrix": numpy array (float),
    "n_slab": int,
    "n_vac": int,
    "ads_atomic_numbers": numpy array (int),
    "ads_pos": numpy array (float),
    "ref_energy": float (optional)
}
"""

import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from pymatgen.core import Lattice

from src.data.pad import pad_to_max


def cell_to_lattice_params(cell: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 cell matrix to lattice parameters (a, b, c, alpha, beta, gamma).
    
    Args:
        cell: 3x3 cell matrix (row vectors)
    
    Returns:
        1D array of [a, b, c, alpha, beta, gamma] where angles are in degrees
    """
    lattice = Lattice(cell)
    # Returns (a, b, c, alpha, beta, gamma)
    return np.array(lattice.parameters, dtype=np.float32)


class LMDBCachedDataset(Dataset):
    """
    LMDB Dataset that loads all data into RAM for fast access.

    Args:
        lmdb_path: Path to the LMDB file
        preload_to_ram: If True, load all data into RAM at initialization
    """

    def __init__(
        self,
        lmdb_path: str,
        preload_to_ram: bool = True,
    ):
        super().__init__()
        self.lmdb_path = Path(lmdb_path)
        self.preload_to_ram = preload_to_ram

        # Open LMDB and get keys
        self.env = lmdb.open(
            str(self.lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )

        # Get all keys
        with self.env.begin() as txn:
            self._keys = []
            cursor = txn.cursor()
            for key, _ in cursor:
                # Skip 'length' key if present
                key_str = key.decode("ascii")
                if key_str != "length":
                    self._keys.append(key_str)

        # Sort keys numerically if they are integers
        try:
            self._keys = sorted(self._keys, key=int)
        except ValueError:
            self._keys = sorted(self._keys)

        self.num_samples = len(self._keys)

        # Preload all data to RAM
        self.cached_data: Optional[List[Dict[str, Any]]] = None
        if self.preload_to_ram:
            self._preload_to_ram()

    def _preload_to_ram(self):
        """Load all LMDB data into RAM."""
        print(f"Loading {self.num_samples} samples from LMDB to RAM...")
        self.cached_data = []

        with self.env.begin() as txn:
            for key in tqdm(self._keys, desc="Loading LMDB to RAM"):
                value = txn.get(key.encode("ascii"))
                data_dict = pickle.loads(value)
                self.cached_data.append(data_dict)

        # Close LMDB environment after loading to RAM
        self.env.close()
        self.env = None
        print(f"Successfully loaded {len(self.cached_data)} samples to RAM.")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary with:
        - primitive_slab: ASE.Atoms object
        - supercell_matrix: numpy array (3x3 or flattened)
        - n_slab: int
        - n_vac: int
        - ads_atomic_numbers: numpy array (int)
        - ads_pos: numpy array (float, shape: n_adsorbate x 3)
        - ref_energy: float (optional)
        """
        if self.cached_data is not None:
            return self.cached_data[idx]
        else:
            # Fallback to direct LMDB access
            with self.env.begin() as txn:
                value = txn.get(self._keys[idx].encode("ascii"))
                return pickle.loads(value)

    def close(self):
        """Close LMDB environment if still open."""
        if self.env is not None:
            self.env.close()
            self.env = None


def collate_fn_with_dynamic_padding(
    batch: List[Dict[str, Any]],
    pad_value: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function with dynamic padding for variable-length data.

    For each batch, computes n_atoms_max and pads:
    - primitive_slab.numbers -> prim_slab_atomic_numbers
    - primitive_slab.positions -> prim_slab_positions
    - ads_atomic_numbers -> adsorbate_atomic_numbers
    - ads_pos -> adsorbate_pos

    Args:
        batch: List of data dictionaries from LMDBCachedDataset
        pad_value: Value to use for padding (default: 0)

    Returns:
        Dictionary with padded tensors and masks
    """
    # Extract data from ASE.Atoms
    prim_slab_numbers_list = []
    prim_slab_positions_list = []
    lattice_params_list = []  # 6-dimensional: (a, b, c, alpha, beta, gamma)
    adsorbate_numbers_list = []
    adsorbate_pos_list = []
    
    ref_ads_pos_list = []
    bind_ads_atom_list = []

    supercell_matrices = []
    n_slabs = []
    n_vacs = []
    ref_energies = []

    for sample in batch:
        # Extract from primitive_slab (ASE.Atoms)
        primitive_slab = sample["primitive_slab"]
        prim_slab_numbers_list.append(
            torch.tensor(primitive_slab.numbers, dtype=torch.long)
        )
        prim_slab_positions_list.append(
            torch.tensor(primitive_slab.positions, dtype=torch.float)
        )
        # Convert 3x3 cell matrix to 6-dimensional lattice parameters
        lattice_params = cell_to_lattice_params(primitive_slab.cell[:])
        lattice_params_list.append(torch.tensor(lattice_params, dtype=torch.float))

        # Extract other fields (LMDB key names: ads_atomic_numbers, ads_pos)
        ads_nums = np.array(sample["ads_atomic_numbers"])
        ads_pos = np.array(sample["ads_pos"])
        
        ref_pos = np.array(sample["ref_ads_pos"])
        bind_ads_atom_list.append(sample.get("bind_ads_atom_symbol", 0))
        
        ref_ads_pos_list.append(torch.tensor(ref_pos.reshape(-1, 3), dtype=torch.float))

        adsorbate_numbers_list.append(torch.tensor(ads_nums, dtype=torch.long))
        adsorbate_pos_list.append(
            torch.tensor(ads_pos.reshape(-1, 3), dtype=torch.float)
        )

        supercell_matrices.append(
            torch.tensor(sample["supercell_matrix"], dtype=torch.float)
        )
        n_slabs.append(sample["n_slab"])
        n_vacs.append(sample["n_vac"])
        # Extract ref_energy if available, otherwise use 0.0
        ref_energy = sample.get("ref_energy", None)
        ref_energies.append(ref_energy)

    # Compute scaling_factor = (n_vac + n_slab) / n_slab for each sample
    scaling_factors = []
    for n_slab, n_vac in zip(n_slabs, n_vacs):
        scaling_factors.append((n_vac + n_slab) / n_slab)

    # Use pad_to_max for dynamic padding
    prim_slab_atomic_numbers, prim_slab_num_mask = pad_to_max(
        prim_slab_numbers_list, value=pad_value
    )
    prim_slab_positions, _ = pad_to_max(prim_slab_positions_list, value=0.0)

    # Handle adsorbate (may have empty tensors)
    # Ensure at least 1 dimension for adsorbate
    if all(len(t) == 0 for t in adsorbate_numbers_list):
        # All empty - create dummy tensor with size 1
        batch_size = len(batch)
        adsorbate_atomic_numbers = torch.full(
            (batch_size, 1), pad_value, dtype=torch.long
        )
        adsorbate_pos = torch.zeros((batch_size, 1, 3), dtype=torch.float)
        ref_ads_pos = torch.zeros((batch_size, 1, 3), dtype=torch.float)
        adsorbate_mask = torch.zeros((batch_size, 1), dtype=torch.bool)
        n_adsorbate_atoms_max = 1
    else:
        # Pad non-empty adsorbate tensors
        # Add dummy element to empty tensors for pad_to_max compatibility
        padded_ads_nums = []
        padded_ads_pos = []
        padded_ref_ads_pos = []
        for nums, pos, ref_pos in zip(adsorbate_numbers_list, adsorbate_pos_list, ref_ads_pos_list):
            if len(nums) == 0:
                padded_ads_nums.append(torch.tensor([pad_value], dtype=torch.long))
                padded_ads_pos.append(torch.zeros((1, 3), dtype=torch.float))
                padded_ref_ads_pos.append(torch.zeros((1, 3), dtype=torch.float))
            else:
                padded_ads_nums.append(nums)
                padded_ads_pos.append(pos)
                padded_ref_ads_pos.append(ref_pos)

        adsorbate_atomic_numbers, ads_num_mask = pad_to_max(
            padded_ads_nums, value=pad_value
        )
        adsorbate_pos, _ = pad_to_max(padded_ads_pos, value=0.0)
        ref_ads_pos, _ = pad_to_max(padded_ref_ads_pos, value=0.0)

        # Create proper mask (original empty tensors should be all False)
        n_adsorbate_atoms_max = adsorbate_atomic_numbers.shape[1]
        adsorbate_mask = torch.zeros(
            (len(batch), n_adsorbate_atoms_max), dtype=torch.bool
        )
        for i, nums in enumerate(adsorbate_numbers_list):
            if len(nums) > 0:
                adsorbate_mask[i, : len(nums)] = True

    # Create prim_slab mask from padding result
    if isinstance(prim_slab_num_mask, torch.Tensor):
        prim_slab_mask = prim_slab_num_mask.bool()
    else:
        # No padding needed (all same size)
        prim_slab_mask = torch.ones(prim_slab_atomic_numbers.shape, dtype=torch.bool)

    # Stack fixed-size tensors
    lattice_params = torch.stack(lattice_params_list)  # [B, 6]
    supercell_matrix = torch.stack(supercell_matrices)

    n_prim_slab_atoms_max = prim_slab_atomic_numbers.shape[1]

    return {
        # Padded prim_slab data
        "prim_slab_atomic_numbers": prim_slab_atomic_numbers,  # [B, n_prim_slab_max]
        "ref_prim_slab_element": prim_slab_atomic_numbers,  # [B, n_prim_slab_max]
        "n_prim_slab_atoms": torch.tensor(
            [len(t) for t in prim_slab_numbers_list], dtype=torch.long
        ),  # [B]
        # Padded adsorbate data
        "adsorbate_atomic_numbers": adsorbate_atomic_numbers,  # [B, n_adsorbate_max]
        "adsorbate_pos": adsorbate_pos,  # [B, n_adsorbate_max, 3]
        "ref_ads_pos": ref_ads_pos,  # [B, n_adsorbate_max, 3]
        "adsorbate_mask": adsorbate_mask,  # [B, n_adsorbate_max]
        "bind_ads_atom": torch.tensor(bind_ads_atom_list, dtype=torch.long),
        "n_adsorbate_atoms": torch.tensor(
            [len(t) for t in adsorbate_numbers_list], dtype=torch.long
        ),  # [B]
        # Other fields (already fixed size)
        "supercell_matrix": supercell_matrix,  # [B, 3, 3] or [B, 9]
        "n_slab": torch.tensor(n_slabs, dtype=torch.long),  # [B]
        "n_vac": torch.tensor(n_vacs, dtype=torch.long),  # [B]
        # Batch info
        "n_prim_slab_atoms_max": n_prim_slab_atoms_max,
        "n_adsorbate_atoms_max": n_adsorbate_atoms_max,
        # Model input fields - Primitive slab
        "lattice": lattice_params,  # [B, 6] - (a, b, c, alpha, beta, gamma)
        "prim_slab_cart_coords": prim_slab_positions,  # [B, n_prim_slab_max, 3]
        
        "prim_slab_atom_pad_mask": prim_slab_mask.float(),  # [B, n_prim_slab_max] - float for masking
        # Token fields for prim_slab (each atom = individual token)
        "prim_slab_atom_to_token": torch.eye(n_prim_slab_atoms_max).unsqueeze(0).expand(len(batch), -1, -1).clone(),  # [B, N, N] identity
        "prim_slab_token_pad_mask": prim_slab_mask.float(),  # [B, N] - same as prim_slab_atom_pad_mask
        # Model input fields - Adsorbate
        "ads_cart_coords": adsorbate_pos,  # [B, n_adsorbate_max, 3]
        "ref_ads_element": adsorbate_atomic_numbers,  # [B, n_adsorbate_max]
        "ads_atom_pad_mask": adsorbate_mask.float(),  # [B, n_adsorbate_max] - float for masking
        # Token fields for adsorbate (each atom = individual token)
        "ads_atom_to_token": torch.eye(n_adsorbate_atoms_max).unsqueeze(0).expand(len(batch), -1, -1).clone(),  # [B, M, M] identity
        "ads_token_pad_mask": adsorbate_mask.float(),  # [B, M] - same as ads_atom_pad_mask
        # Scaling factor: (n_vac + n_slab) / n_slab
        "scaling_factor": torch.tensor(scaling_factors, dtype=torch.float),  # [B]
        # Reference energy
        "ref_energy": torch.tensor(ref_energies, dtype=torch.float),  # [B]
    }


class LMDBCachedDatasetWithTransform(LMDBCachedDataset):
    """
    Extended LMDB Dataset that transforms data to PyTorch Geometric Data format.
    """

    def __init__(
        self,
        lmdb_path: str,
        preload_to_ram: bool = True,
    ):
        super().__init__(lmdb_path, preload_to_ram)

    def __getitem__(self, idx: int) -> Data:
        """
        Returns a PyTorch Geometric Data object.
        Note: For variable-length data, use collate_pyg_with_dynamic_padding
        """
        raw_data = super().__getitem__(idx)

        primitive_slab = raw_data["primitive_slab"]

        # Convert 3x3 cell matrix to 6-dimensional lattice parameters
        lattice_params = cell_to_lattice_params(primitive_slab.cell[:])

        # Create PyG Data object with raw numpy arrays
        # Padding will be applied in collate_fn
        data = Data(
            # Prim slab info (variable length - will be padded in collate)
            prim_slab_atomic_numbers=torch.tensor(
                primitive_slab.numbers, dtype=torch.long
            ),
            prim_slab_positions=torch.tensor(
                primitive_slab.positions, dtype=torch.float
            ),
            lattice_params=torch.tensor(lattice_params, dtype=torch.float),  # [6]
            # Adsorbate info (variable length - will be padded in collate)
            adsorbate_atomic_numbers=torch.tensor(
                raw_data["ads_atomic_numbers"], dtype=torch.long
            ),
            adsorbate_pos=torch.tensor(
                raw_data["ads_pos"].reshape(-1, 3), dtype=torch.float
            ),
            ref_ads_pos=torch.tensor(
                raw_data["ref_ads_pos"].reshape(-1, 3), dtype=torch.float
            ),
            bind_ads_atom=raw_data.get("bind_ads_atom", 0),   
            # Fixed size data
            supercell_matrix=torch.tensor(  # [3, 3]
                raw_data["supercell_matrix"], dtype=torch.float
            ),
            n_slab=raw_data["n_slab"],  # scalar
            n_vac=raw_data["n_vac"],  # scalar
            scaling_factor=(raw_data["n_vac"] + raw_data["n_slab"]) / raw_data["n_slab"],  # scalar
            # Number of atoms for batching
            num_prim_slab_atoms=len(primitive_slab.numbers),  # scalar
            num_adsorbate_atoms=len(raw_data["ads_atomic_numbers"]),  # scalar
            # Reference energy (optional, default to 0.0 if not present)
            ref_energy=raw_data.get("ref_energy", None),  # scalar
        )

        return data


def collate_pyg_with_dynamic_padding(
    data_list: List[Data],
    pad_value: int = 0,
) -> Data:
    """
    Custom collate function for PyTorch Geometric Data objects with dynamic padding.

    Args:
        data_list: List of Data objects from LMDBCachedDatasetWithTransform
        pad_value: Value to use for padding (default: 0)

    Returns:
        Batched Data object with padded tensors
    """
    batch_size = len(data_list)

    # Extract lists for padding
    prim_slab_nums_list = [d.prim_slab_atomic_numbers for d in data_list]
    prim_slab_pos_list = [d.prim_slab_positions for d in data_list]
    adsorbate_nums_list = [d.adsorbate_atomic_numbers for d in data_list]
    adsorbate_pos_list = [d.adsorbate_pos for d in data_list]
    ref_ads_pos_list = [d.ref_ads_pos for d in data_list]
    
    bind_ads_atom_list = [
        d.bind_ads_atom for d in data_list
    ]

    # Use pad_to_max for prim_slab
    prim_slab_atomic_numbers, prim_slab_num_mask = pad_to_max(
        prim_slab_nums_list, value=pad_value
    )
    prim_slab_positions, _ = pad_to_max(prim_slab_pos_list, value=0.0)

    # Create prim_slab mask
    if isinstance(prim_slab_num_mask, torch.Tensor):
        prim_slab_mask = prim_slab_num_mask.bool()
    else:
        prim_slab_mask = torch.ones(prim_slab_atomic_numbers.shape, dtype=torch.bool)

    # Handle adsorbate (may have empty tensors)
    if all(len(t) == 0 for t in adsorbate_nums_list):
        adsorbate_atomic_numbers = torch.full(
            (batch_size, 1), pad_value, dtype=torch.long
        )
        adsorbate_pos = torch.zeros((batch_size, 1, 3), dtype=torch.float)
        ref_ads_pos = torch.zeros((batch_size, 1, 3), dtype=torch.float)
        adsorbate_mask = torch.zeros((batch_size, 1), dtype=torch.bool)
        n_adsorbate_atoms_max = 1
    else:
        # Pad non-empty adsorbate tensors
        padded_ads_nums = []
        padded_ads_pos = []
        padded_ref_ads_pos = []
        for nums, pos, ref_pos in zip(adsorbate_nums_list, adsorbate_pos_list, ref_ads_pos_list):
            if len(nums) == 0:
                padded_ads_nums.append(torch.tensor([pad_value], dtype=torch.long))
                padded_ads_pos.append(torch.zeros((1, 3), dtype=torch.float))
                padded_ref_ads_pos.append(torch.zeros((1, 3), dtype=torch.float))
            else:
                padded_ads_nums.append(nums)
                padded_ads_pos.append(pos)
                padded_ref_ads_pos.append(ref_pos)

        adsorbate_atomic_numbers, _ = pad_to_max(padded_ads_nums, value=pad_value)
        adsorbate_pos, _ = pad_to_max(padded_ads_pos, value=0.0)
        ref_ads_pos, _ = pad_to_max(padded_ref_ads_pos, value=0.0)

        n_adsorbate_atoms_max = adsorbate_atomic_numbers.shape[1]
        adsorbate_mask = torch.zeros(
            (batch_size, n_adsorbate_atoms_max), dtype=torch.bool
        )
        for i, nums in enumerate(adsorbate_nums_list):
            if len(nums) > 0:
                adsorbate_mask[i, : len(nums)] = True

    n_prim_slab_atoms_max = prim_slab_atomic_numbers.shape[1]
    lattice_params = torch.stack([d.lattice_params for d in data_list])  # [B, 6]

    # Create batched Data object
    batched_data = Data(
        # Padded prim_slab data
        prim_slab_atomic_numbers=prim_slab_atomic_numbers,  # [B, n_prim_slab_max]
        prim_slab_positions=prim_slab_positions,  # [B, n_prim_slab_max, 3]
        prim_slab_mask=prim_slab_mask,  # [B, n_prim_slab_max]
        n_prim_slab_atoms=torch.tensor(
            [d.num_prim_slab_atoms for d in data_list], dtype=torch.long
        ),  # [B]
        # Padded adsorbate data
        adsorbate_atomic_numbers=adsorbate_atomic_numbers,  # [B, n_adsorbate_max]
        adsorbate_pos=adsorbate_pos,  # [B, n_adsorbate_max, 3]
        ref_ads_pos=ref_ads_pos,  # [B, n_adsorbate_max, 3]
        adsorbate_mask=adsorbate_mask,  # [B, n_adsorbate_max]
        
        bind_ads_atom=torch.tensor(bind_ads_atom_list, dtype=torch.long),
        
        n_adsorbate_atoms=torch.tensor(
            [d.num_adsorbate_atoms for d in data_list], dtype=torch.long
        ),  # [B]
        # Stacked fixed size data
        supercell_matrix=torch.stack(
            [d.supercell_matrix for d in data_list]
        ),  # [B, ...]
        n_slab=torch.tensor([d.n_slab for d in data_list], dtype=torch.long),  # [B]
        n_vac=torch.tensor([d.n_vac for d in data_list], dtype=torch.long),  # [B]
        # Batch info
        batch_size=batch_size,
        n_prim_slab_atoms_max=n_prim_slab_atoms_max,
        n_adsorbate_atoms_max=n_adsorbate_atoms_max,
        # Model input fields - Primitive slab
        lattice=lattice_params,  # [B, 6] - (a, b, c, alpha, beta, gamma)
        prim_slab_cart_coords=prim_slab_positions,  # [B, n_prim_slab_max, 3]
        ref_prim_slab_element=prim_slab_atomic_numbers,  # [B, n_prim_slab_max]
        prim_slab_atom_pad_mask=prim_slab_mask.float(),  # [B, n_prim_slab_max] - float for masking
        # Token fields for prim_slab (each atom = individual token)
        prim_slab_atom_to_token=torch.eye(n_prim_slab_atoms_max).unsqueeze(0).expand(batch_size, -1, -1).clone(),  # [B, N, N] identity
        prim_slab_token_pad_mask=prim_slab_mask.float(),  # [B, N] - same as prim_slab_atom_pad_mask
        # Model input fields - Adsorbate
        ads_cart_coords=adsorbate_pos,  # [B, n_adsorbate_max, 3]
        ref_ads_element=adsorbate_atomic_numbers,  # [B, n_adsorbate_max]
        ads_atom_pad_mask=adsorbate_mask.float(),  # [B, n_adsorbate_max] - float for masking
        # Token fields for adsorbate (each atom = individual token)
        ads_atom_to_token=torch.eye(n_adsorbate_atoms_max).unsqueeze(0).expand(batch_size, -1, -1).clone(),  # [B, M, M] identity
        ads_token_pad_mask=adsorbate_mask.float(),  # [B, M] - same as ads_atom_pad_mask
        # Scaling factor: (n_vac + n_slab) / n_slab
        scaling_factor=torch.tensor([d.scaling_factor for d in data_list], dtype=torch.float),  # [B]
        
        ref_energy=torch.tensor([d.ref_energy for d in data_list], dtype=torch.float),  # [B]
    )

    return batched_data
