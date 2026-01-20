"""
Module to reconstruct catalyst system structures using model prediction output.

prediction_output structure:
    {
        "exception": False,
        "generated_prim_slab_coords": (B*M, N, 3) - cartesian coordinates
        "generated_ads_coords": (B*M, A, 3) - cartesian coordinates
        "generated_lattices": (B*M, 6) - (a, b, c, alpha, beta, gamma)
        "generated_supercell_matrices": (B*M, 3, 3) or (B*M, 9)
        "generated_scaling_factors": (B*M,)
        "prim_slab_atom_types": (B, N) - atomic numbers
        "ads_atom_types": (B, A) - atomic numbers
        "prim_slab_atom_mask": (B, N) - padding mask
        "ads_atom_mask": (B, A) - padding mask
        "tags": optional
    }
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
from ase import Atoms
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import DummySpecie, Element
from pymatgen.io.ase import AseAtomsAdaptor
from ase.geometry import cellpar_to_cell
from ase.cell import Cell

import ase
import numpy as np
from sympy import Matrix
from itertools import product

def tag_surface_atoms(
    slab_atoms: ase.Atoms = None,
):
    """
    Sets the tags of an `ase.Atoms` object. Any atom that we consider a "bulk"
    atom will have a tag of 0, and any atom that we consider a "surface" atom
    will have a tag of 1. We use a combination of Voronoi neighbor algorithms
    (adapted from `pymatgen.core.surface.Slab.get_surface_sites`; see
    https://pymatgen.org/pymatgen.core.surface.html) and a distance cutoff.

    Arguments
    ---------
    slab_atoms: ase.Atoms
        The slab where you are trying to find surface sites.
    bulk_atoms: ase.Atoms
        The bulk structure that the surface was cut from.

    Returns
    -------
    slab_atoms: ase.Atoms
        A copy of the slab atoms with the surface atoms tagged as 1.
    """
    assert slab_atoms is not None
    slab_atoms = slab_atoms.copy()

    height_tags = find_surface_atoms_by_height(slab_atoms)

    tags = height_tags

    slab_atoms.set_tags(tags)

    return slab_atoms

def find_surface_atoms_by_height(surface_atoms):
    """
    As discussed in the docstring for `find_surface_atoms_with_voronoi`,
    sometimes we might accidentally tag a surface atom as a bulk atom if there
    are multiple coordination environments for that atom type within the bulk.
    One heuristic that we use to address this is to simply figure out if an
    atom is close to the surface. This function will figure that out.

    Specifically:  We consider an atom a surface atom if it is within 2
    Angstroms of the heighest atom in the z-direction (or more accurately, the
    direction of the 3rd unit cell vector).

    Arguments
    ---------
    surface_atoms: ase.Atoms

    Returns
    -------
    tags: list
        A list that contains the indices of the surface atoms.
    """
    unit_cell_height = np.linalg.norm(surface_atoms.cell[2])
    scaled_positions = surface_atoms.get_scaled_positions()
    scaled_max_height = max(scaled_position[2] for scaled_position in scaled_positions)
    scaled_threshold = scaled_max_height - 2.0 / unit_cell_height

    return [
        0 if scaled_position[2] < scaled_threshold else 1
        for scaled_position in scaled_positions
    ]


def assemble(
    generated_prim_slab_coords: np.ndarray,
    generated_ads_coords: np.ndarray,
    generated_lattice: np.ndarray,
    generated_supercell_matrix: np.ndarray,
    generated_scaling_factor: float,
    prim_slab_atom_types: np.ndarray,
    ads_atom_types: np.ndarray,
    prim_slab_atom_mask: Optional[np.ndarray] = None,
    ads_atom_mask: Optional[np.ndarray] = None,
) -> Atoms:
    """
    Reconstruct a single catalyst system structure using model prediction output.
    
    Args:
        generated_prim_slab_coords: (N, 3) cartesian coordinates of primitive slab
        generated_ads_coords: (A, 3) cartesian coordinates of adsorbate
        generated_lattice: (6,) lattice parameters (a, b, c, alpha, beta, gamma)
        generated_supercell_matrix: (3, 3) or (9,) supercell transformation matrix
        generated_scaling_factor: z-direction scaling factor
        prim_slab_atom_types: (N,) atomic numbers of primitive slab
        ads_atom_types: (A,) atomic numbers of adsorbate
        prim_slab_atom_mask: (N,) optional, valid atom mask (True = valid)
        ads_atom_mask: (A,) optional, valid atom mask (True = valid)
    
    Returns:
        Reconstructed ASE Atoms object of catalyst system
    """

    # Create AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()
    
    # Apply mask to select only valid atoms
    if prim_slab_atom_mask is not None:
        mask = prim_slab_atom_mask.astype(bool)
        prim_slab_coords = generated_prim_slab_coords[mask]
        prim_slab_types = [DummySpecie("X") if z == 0 else Element.from_Z(z) for z in prim_slab_atom_types[mask]]
    else:
        prim_slab_coords = generated_prim_slab_coords
        prim_slab_types = [DummySpecie("X") if z == 0 else Element.from_Z(z) for z in prim_slab_atom_types]
    
    if ads_atom_mask is not None:
        ads_mask = ads_atom_mask.astype(bool)
        ads_coords = generated_ads_coords[ads_mask]
        ads_types = ads_atom_types[ads_mask]
    else:
        ads_coords = generated_ads_coords
        ads_types = ads_atom_types
    
    # Create Lattice object from lattice parameters
    # generated_lattice: (a, b, c, alpha, beta, gamma)
    a, b, c, alpha, beta, gamma = generated_lattice
    
    prim_slab_struct = Structure(
        lattice=Lattice.from_parameters(a, b, c, alpha, beta, gamma),
        species=prim_slab_types,
        coords=prim_slab_coords,
        coords_are_cartesian=True
    )
    
    if generated_supercell_matrix.shape == (9,):
        supercell_matrix = np.round(generated_supercell_matrix.reshape(3, 3)).astype(int)
    else:
        supercell_matrix = np.round(generated_supercell_matrix).astype(int)

    # Create supercell
    supercell_slab_struct = prim_slab_struct.copy()
    supercell_slab_struct.make_supercell(supercell_matrix, to_unit_cell=False)
    
    # Convert Pymatgen Structure to ASE Atoms
    recon_tight_slab = adaptor.get_atoms(supercell_slab_struct)
    
    # Apply z-direction scaling (expand vacuum region)
    recon_slab = recon_tight_slab.copy()
    recon_cell = recon_slab.get_cell()
    recon_cell[2] = recon_cell[2] * generated_scaling_factor
    recon_slab.set_cell(recon_cell)
    
    # Set tags for slab atoms: tag 0
    # recon_slab.set_tags(np.zeros(len(recon_slab), dtype=int))
    recon_slab = tag_surface_atoms(slab_atoms=recon_slab)
    
    # Create adsorbate Atoms
    ads_atoms = Atoms(
        numbers=ads_types,
        positions=ads_coords,
        cell=recon_cell,
        pbc=True
    )
    # Set tags for adsorbate atoms: tag 2
    ads_atoms.set_tags(np.full(len(ads_atoms), 2, dtype=int))
    
    # Combine slab and adsorbate
    recon_system = recon_slab + ads_atoms
    
    return recon_system, recon_slab


def assemble_batch(
    prediction_output: Dict[str, Any],
    num_samples: int = 1,
    return_error_indices: bool = False,
) -> Union[List[Atoms], Tuple[List[Atoms], List[int]]]:
    """
    Reconstruct all structures from batched prediction output.
    
    Args:
        prediction_output: Dictionary returned from predict_step
        num_samples: Number of samples per input (multiplicity)
        return_error_indices: If True, also return list of sample indices that had errors
    
    Returns:
        If return_error_indices=False: List of reconstructed Atoms objects (length: batch_size * num_samples)
        If return_error_indices=True: Tuple of (atoms_list, error_indices)
    """
    if prediction_output.get("exception", False):
        if return_error_indices:
            return [], []
        return []
    
    # Convert tensors to numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x
    
    generated_prim_slab_coords = to_numpy(prediction_output["generated_prim_slab_coords"])
    generated_ads_coords = to_numpy(prediction_output["generated_ads_coords"])
    generated_lattices = to_numpy(prediction_output["generated_lattices"])
    generated_supercell_matrices = to_numpy(prediction_output["generated_supercell_matrices"])
    generated_scaling_factors = to_numpy(prediction_output["generated_scaling_factors"])
    prim_slab_atom_types = to_numpy(prediction_output["prim_slab_atom_types"])
    ads_atom_types = to_numpy(prediction_output["ads_atom_types"])
    prim_slab_atom_mask = to_numpy(prediction_output.get("prim_slab_atom_mask"))
    ads_atom_mask = to_numpy(prediction_output.get("ads_atom_mask"))
    
    # Calculate shapes: (B*M, ...) -> B*M samples
    total_samples = generated_prim_slab_coords.shape[0]
    batch_size = prim_slab_atom_types.shape[0]
    
    # Calculate num_samples (check if it matches given value)
    calculated_num_samples = total_samples // batch_size
    if num_samples != calculated_num_samples:
        num_samples = calculated_num_samples
    
    atoms_list = []
    error_indices = []
    
    for i in range(total_samples):
        # Calculate original batch index
        batch_idx = i // num_samples
        
        try:
            atoms, _ = assemble(
                generated_prim_slab_coords=generated_prim_slab_coords[i],
                generated_ads_coords=generated_ads_coords[i],
                generated_lattice=generated_lattices[i],
                generated_supercell_matrix=generated_supercell_matrices[i],
                generated_scaling_factor=generated_scaling_factors[i],
                prim_slab_atom_types=prim_slab_atom_types[batch_idx],
                ads_atom_types=ads_atom_types[batch_idx],
                prim_slab_atom_mask=prim_slab_atom_mask[batch_idx] if prim_slab_atom_mask is not None else None,
                ads_atom_mask=ads_atom_mask[batch_idx] if ads_atom_mask is not None else None,
            )
            atoms_list.append(atoms)
        except Exception as e:
            print(f"[WARNING] Failed to reconstruct sample {i}: {e}")
            print(f"  - Problematic Supercell Matrix:\n{generated_supercell_matrices[i]}")
            atoms_list.append(None)
            error_indices.append(i)
    
    if return_error_indices:
        return atoms_list, error_indices
    return atoms_list