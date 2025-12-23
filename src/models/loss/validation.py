"""Validation metrics and structure matching utilities."""

from typing import List, Optional, Tuple, Dict, Any
import io
import contextlib

import numpy as np
from pymatgen.core import Structure, Lattice  # Used for find_best_match_rmsd
from pymatgen.analysis.structure_matcher import StructureMatcher
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase import Atoms
import sys
from ase.optimize import LBFGS
import warnings

from scripts.assemble import assemble
import smact
from smact.screening import pauling_test
import itertools

# Suppress all warnings for JSON parsing in subprocesses
warnings.filterwarnings('ignore')

# Global calculator for reuse across processes
_GLOBAL_CALC = None

OC20_GAS_PHASE_ENERGIES = {
    'H': -3.477,
    'O': -7.204,
    'C': -7.282,
    'N': -8.083,
}

from pymatgen.io.ase import AseAtomsAdaptor
adaptor = AseAtomsAdaptor()

def get_adsorbate_energy_from_table(atoms_obj):

    total_energy = 0.0
    symbols = atoms_obj.get_chemical_symbols()
    print(f"Adsorbate composition: {', '.join(symbols)}", file=sys.stderr)
    
    for atom_symbol in symbols:
        try:
            total_energy += OC20_GAS_PHASE_ENERGIES[atom_symbol]
        except KeyError:
            raise ValueError(
                f"Energy table does not contain '{atom_symbol}' atom. "
                f"Currently supported atoms are {list(OC20_GAS_PHASE_ENERGIES.keys())}."
            )
            
    return total_energy

def get_uma_calculator(model_name: str = "uma-m-1p1", device: str = "cuda") -> FAIRChemCalculator:
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    return FAIRChemCalculator(predictor, task_name="oc20")

def relaxation_and_compute_adsorption_energy(calc: FAIRChemCalculator, system: Atoms, slab: Atoms, adsorbate: Atoms):
    try:
        system.calc = calc
        slab.calc = calc
        
        # Relax system
        opt = LBFGS(system)
        opt.run(0.05, 100)
        
        # Relax slab
        opt = LBFGS(slab)
        opt.run(0.05, 100)
        
        # Energy calculation
        e_sys = system.get_potential_energy()
        e_slab = slab.get_potential_energy()
        
        # Adsorbate energy is looked up in the table
        e_adsorbate = get_adsorbate_energy_from_table(adsorbate)
        
        # Adsorption energy calculation
        e_ads = e_sys - (e_slab + e_adsorbate)
        
        return e_ads, e_sys, e_slab, e_adsorbate

    except Exception as e:
        # UMA calculation failed, return default value (e.g. No edges found)
        print(f"WARNING: UMA calculation failed for a sample. Error: {e}", file=sys.stderr)
        e_adsorbate = get_adsorbate_energy_from_table(adsorbate)
        return 999.0, float('nan'), float('nan'), e_adsorbate

def find_best_match_rmsd_prim(
    args: Tuple[
        np.ndarray,  # sampled_coords: (n_samples, n_atoms, 3)
        np.ndarray,  # sampled_lattices: (n_samples, 6) - lattice parameters (a, b, c, alpha, beta, gamma)
        np.ndarray,  # true_coords: (n_atoms, 3)
        np.ndarray,  # true_lattices: (6,) - lattice parameters (a, b, c, alpha, beta, gamma)
        np.ndarray,  # atom_types: (n_atoms,)
        np.ndarray,  # atom_mask: (n_atoms,) bool
        Dict[str, Any],  # matcher_kwargs
    ],
) -> List[Optional[float]]:
    """
    Find the best RMSD match between sampled structures and the true structure.

    This function is designed to be run in parallel with ProcessPool.

    Args:
        args: Tuple containing:
            - sampled_coords: Sampled atom coordinates (n_samples, n_atoms, 3)
            - sampled_lattices: Sampled lattice parameters (n_samples, 6) - (a, b, c, alpha, beta, gamma)
            - true_coords: True atom coordinates (n_atoms, 3)
            - true_lattices: True lattice parameters (6,) - (a, b, c, alpha, beta, gamma)
            - atom_types: Atomic numbers for each atom (n_atoms,)
            - atom_mask: Boolean mask for valid atoms (n_atoms,)
            - matcher_kwargs: Keyword arguments for StructureMatcher

    Returns:
        List of RMSD values for each sample. None if no match found.
    """
    (
        sampled_coords,
        sampled_lattices,
        true_coords,
        true_lattices,
        atom_types,
        atom_mask,
        matcher_kwargs,
    ) = args

    n_samples = sampled_coords.shape[0]

    # Filter valid atoms
    valid_atom_types = atom_types[atom_mask]
    true_valid_coords = true_coords[atom_mask]

    # Create true structure from lattice parameters (a, b, c, alpha, beta, gamma)
    try:
        true_lattice = Lattice.from_parameters(
            a=true_lattices[0],
            b=true_lattices[1],
            c=true_lattices[2],
            alpha=true_lattices[3],
            beta=true_lattices[4],
            gamma=true_lattices[5],
        )
        true_structure = Structure(
            lattice=true_lattice,
            species=valid_atom_types.tolist(),
            coords=true_valid_coords,
            coords_are_cartesian=True,
        )
    except Exception:
        return [None] * n_samples

    # Create matcher
    matcher = StructureMatcher.from_dict(matcher_kwargs)

    results = []
    for i in range(n_samples):
        try:
            # Create sampled structure from lattice parameters (a, b, c, alpha, beta, gamma)
            sampled_lattice = Lattice.from_parameters(
                a=sampled_lattices[i, 0],
                b=sampled_lattices[i, 1],
                c=sampled_lattices[i, 2],
                alpha=sampled_lattices[i, 3],
                beta=sampled_lattices[i, 4],
                gamma=sampled_lattices[i, 5],
            )
            sampled_valid_coords = sampled_coords[i][atom_mask]

            sampled_structure = Structure(
                lattice=sampled_lattice,
                species=valid_atom_types.tolist(),
                coords=sampled_valid_coords,
                coords_are_cartesian=True,
            )

            # Check if structures match
            if matcher.fit(true_structure, sampled_structure):
                # Compute RMSD if match found
                rmsd = matcher.get_rms_dist(true_structure, sampled_structure)
                if rmsd is not None:
                    results.append(rmsd[0])  # get_rms_dist returns (rms_dist, max_dist)
                else:
                    results.append(None)
            else:
                results.append(None)

        except Exception:
            results.append(None)

    return results


def find_best_match_rmsd_slab(
    args: Tuple[
        np.ndarray,  # sampled_prim_slab_coords: (n_samples, n_prim_slab_atoms, 3)
        np.ndarray,  # sampled_ads_coords: (n_samples, n_ads_atoms, 3)
        np.ndarray,  # sampled_lattices: (n_samples, 6)
        np.ndarray,  # sampled_supercell_matrices: (n_samples, 3, 3) or (n_samples, 9)
        np.ndarray,  # sampled_scaling_factors: (n_samples,)
        np.ndarray,  # true_prim_slab_coords: (n_prim_slab_atoms, 3)
        np.ndarray,  # true_ads_coords: (n_ads_atoms, 3)
        np.ndarray,  # true_lattices: (6,)
        np.ndarray,  # true_supercell_matrix: (3, 3) or (9,)
        float,       # true_scaling_factor
        np.ndarray,  # prim_slab_atom_types: (n_prim_slab_atoms,)
        np.ndarray,  # ads_atom_types: (n_ads_atoms,)
        np.ndarray,  # prim_slab_atom_mask: (n_prim_slab_atoms,) bool
        np.ndarray,  # ads_atom_mask: (n_ads_atoms,) bool
        Dict[str, Any],  # matcher_kwargs
    ],
) -> List[Optional[float]]:
    """
    Find the best RMSD match between sampled full systems and the true system.
    
    This function assembles the full system (supercell slab + adsorbate) and computes RMSD.
    Designed to be run in parallel with ProcessPool.
    
    Args:
        args: Tuple containing all necessary data for system assembly and RMSD computation.
    
    Returns:
        List of RMSD values for each sample. None if no match found or assembly failed.
    """
    (
        sampled_prim_slab_coords,
        sampled_ads_coords,
        sampled_lattices,
        sampled_supercell_matrices,
        sampled_scaling_factors,
        true_prim_slab_coords,
        true_ads_coords,
        true_lattices,
        true_supercell_matrix,
        true_scaling_factor,
        prim_slab_atom_types,
        ads_atom_types,
        prim_slab_atom_mask,
        ads_atom_mask,
        matcher_kwargs,
    ) = args
    
    n_samples = sampled_prim_slab_coords.shape[0]
    
    # Assemble true system first
    try:
        true_system, true_slab = assemble(
            generated_prim_slab_coords=true_prim_slab_coords,
            generated_ads_coords=true_ads_coords,
            generated_lattice=true_lattices,
            generated_supercell_matrix=true_supercell_matrix.reshape(3, 3),
            generated_scaling_factor=float(true_scaling_factor),
            prim_slab_atom_types=prim_slab_atom_types,
            ads_atom_types=ads_atom_types,
            prim_slab_atom_mask=prim_slab_atom_mask,
            ads_atom_mask=ads_atom_mask,
        )
        
        true_structure = adaptor.get_structure(true_slab)
        
    except Exception:
        return [None] * n_samples
    
    # Create matcher
    matcher = StructureMatcher.from_dict(matcher_kwargs)
    
    results = []
    for i in range(n_samples):
        try:
            # Assemble sampled system
            sampled_system, sampled_slab = assemble(
                generated_prim_slab_coords=sampled_prim_slab_coords[i],
                generated_ads_coords=sampled_ads_coords[i],
                generated_lattice=sampled_lattices[i],
                generated_supercell_matrix=sampled_supercell_matrices[i].reshape(3, 3),
                generated_scaling_factor=float(sampled_scaling_factors[i]),
                prim_slab_atom_types=prim_slab_atom_types,
                ads_atom_types=ads_atom_types,
                prim_slab_atom_mask=prim_slab_atom_mask,
                ads_atom_mask=ads_atom_mask,
            )
            
            sampled_structure = adaptor.get_structure(sampled_slab)
            
            # Check if slabs match
            if matcher.fit(true_structure, sampled_structure):
                # Compute RMSD if match found
                rmsd = matcher.get_rms_dist(true_structure, sampled_structure)
                if rmsd is not None:
                    results.append(rmsd[0])  # get_rms_dist returns (rms_dist, max_dist)
                else:
                    results.append(None)
            else:
                results.append(None)
                
        except Exception:
            results.append(None)
    
    return results


def _struct_comp_validity(atoms: Atoms) -> bool:
    """Check structural validity of an Atoms object."""
    # 1. Check cell volume
    try:
        vol = float(atoms.get_volume())
        vol_ok = vol >= 0.1
    except Exception:
        vol_ok = False

    # 2. Check atom clash
    try:
        if len(atoms) > 1:
            dists = atoms.get_all_distances()
            # Exclude self-distance (0)
            min_dist = np.min(dists[np.nonzero(dists)])
            dist_ok = min_dist >= 0.5
            if dist_ok is False:
                print("Atom clash check failed.")
        else:
            dist_ok = True  # Skip distance check for single atom
    except Exception:
        dist_ok = False
    
    # 3. Check min width of cell (a, b axes)
    min_ab = 8.0
    
    a_length = np.linalg.norm(atoms.cell[0])
    b_length = np.linalg.norm(atoms.cell[1])
    
    if a_length < min_ab or b_length < min_ab:
        print("Width check failed.")
        width_ok = False
    else:
        width_ok = True
    
    # 4. Check min height of cell (c projected onto normal of a×b plane)
    min_height = 20.0
    
    a_vec = atoms.cell[0]
    b_vec = atoms.cell[1]
    c_vec = atoms.cell[2]
    
    # normal = unit vector of (a × b)
    cross_ab = np.cross(a_vec, b_vec)
    cross_ab_norm = np.linalg.norm(cross_ab)
    
    if cross_ab_norm < 1e-10:
        # Degenerate case: a and b are parallel
        print("Height check failed.")
        height_ok = False
    else:
        normal = cross_ab / cross_ab_norm
        proj_height = abs(np.dot(normal, c_vec))
        height_ok = proj_height >= min_height

    # 5. Basic validity check
    basic_valid = bool(vol_ok and dist_ok and width_ok and height_ok)
    
    if not basic_valid:
        print("")
        return False

    # 6. SMACT validity (charge neutrality etc)
    try:
        smact_ok = smact_validity(atoms)
    except Exception:
        smact_ok = False

    # 7. Crystal validity (connectivity check)
    try:
        crystal_ok = crystal_validity(atoms)
    except Exception:
        crystal_ok = False

    return bool(basic_valid and smact_ok and crystal_ok)

def _structural_validity(atoms: Atoms) -> bool:
    """Check structural validity of an Atoms object."""
    # 1. Check cell volume
    try:
        vol = float(atoms.get_volume())
        vol_ok = vol >= 0.1
    except Exception:
        vol_ok = False

    # 2. Check atom clash
    try:
        if len(atoms) > 1:
            dists = atoms.get_all_distances()
            # Exclude self-distance (0)
            min_dist = np.min(dists[np.nonzero(dists)])
            dist_ok = min_dist >= 0.5
  
        else:
            dist_ok = True  # Skip distance check for single atom
    except Exception:
        dist_ok = False
    
    # 3. Check min width of cell (a, b axes)
    min_ab = 8.0
    
    a_length = np.linalg.norm(atoms.cell[0])
    b_length = np.linalg.norm(atoms.cell[1])
    
    if a_length < min_ab or b_length < min_ab:
        print("Width check failed.")
        width_ok = False
    else:
        width_ok = True
    
    # 4. Check min height of cell (c projected onto normal of a×b plane)
    min_height = 20.0
    
    a_vec = atoms.cell[0]
    b_vec = atoms.cell[1]
    c_vec = atoms.cell[2]
    
    # normal = unit vector of (a × b)
    cross_ab = np.cross(a_vec, b_vec)
    cross_ab_norm = np.linalg.norm(cross_ab)
    
    if cross_ab_norm < 1e-10:
        # Degenerate case: a and b are parallel
        print("Height check failed.")
        height_ok = False
    else:
        normal = cross_ab / cross_ab_norm
        proj_height = abs(np.dot(normal, c_vec))
        height_ok = proj_height >= min_height

    # 5. Basic validity check
    basic_valid = bool(vol_ok and dist_ok and width_ok and height_ok)

    return basic_valid

def compute_structural_validity_single(
    args: Tuple[
        np.ndarray,  # sampled_prim_slab_coords: (n_samples, n_prim_slab_atoms, 3)
        np.ndarray,  # sampled_ads_coords: (n_samples, n_ads_atoms, 3)
        np.ndarray,  # sampled_lattices: (n_samples, 6)
        np.ndarray,  # sampled_supercell_matrices: (n_samples, 3, 3) or (n_samples, 9)
        np.ndarray,  # sampled_scaling_factors: (n_samples,)
        np.ndarray,  # prim_slab_atom_types: (n_prim_slab_atoms,)
        np.ndarray,  # ads_atom_types: (n_ads_atoms,)
        np.ndarray,  # prim_slab_atom_mask: (n_prim_slab_atoms,) bool
        np.ndarray,  # ads_atom_mask: (n_ads_atoms,) bool
    ],
) -> List[bool]:
    """
    Compute structural validity for sampled structures (without adsorption energy calculation).
    
    This function reconstructs the full system from prim_slab using supercell_matrix
    and scaling_factor, then checks structural validity.
    
    Designed to be run in parallel with ProcessPool.
    
    Args:
        args: Tuple containing:
            - sampled_prim_slab_coords: Sampled prim slab coordinates (n_samples, n_atoms, 3)
            - sampled_ads_coords: Sampled adsorbate coordinates (n_samples, n_ads_atoms, 3)
            - sampled_lattices: Sampled lattice parameters (n_samples, 6)
            - sampled_supercell_matrices: Supercell transformation matrices (n_samples, 3, 3) or (n_samples, 9)
            - sampled_scaling_factors: Z-direction scaling factors (n_samples,)
            - prim_slab_atom_types: Atomic numbers for prim slab atoms
            - ads_atom_types: Atomic numbers for adsorbate atoms
            - prim_slab_atom_mask: Boolean mask for valid prim slab atoms
            - ads_atom_mask: Boolean mask for valid adsorbate atoms
    
    Returns:
        List of structural validity booleans for each sample.
    """
    (
        sampled_prim_slab_coords,
        sampled_ads_coords,
        sampled_lattices,
        sampled_supercell_matrices,
        sampled_scaling_factors,
        prim_slab_atom_types,
        ads_atom_types,
        prim_slab_atom_mask,
        ads_atom_mask,
    ) = args
    
    n_samples = sampled_prim_slab_coords.shape[0]
    
    results = []
    for i in range(n_samples):
        try:
            # Use assemble function to reconstruct the system
            recon_system, recon_slab = assemble(
                generated_prim_slab_coords=sampled_prim_slab_coords[i],
                generated_ads_coords=sampled_ads_coords[i],
                generated_lattice=sampled_lattices[i],
                generated_supercell_matrix=sampled_supercell_matrices[i].reshape(3, 3),
                generated_scaling_factor=sampled_scaling_factors[i],
                prim_slab_atom_types=prim_slab_atom_types,
                ads_atom_types=ads_atom_types,
                prim_slab_atom_mask=prim_slab_atom_mask,
                ads_atom_mask=ads_atom_mask,
            )
            
            # Check structural validity
            struct_valid = _structural_validity(recon_system)
            results.append(struct_valid)
                
        except Exception as e:
            print(f"WARNING: Failed to compute structural validity for sample {i}: {e}", file=sys.stderr)
            results.append(False)
    
    return results


def compute_adsorption_and_validity_single(
    sampled_prim_slab_coords: np.ndarray,  # (n_samples, n_prim_slab_atoms, 3)
    sampled_ads_coords: np.ndarray,  # (n_samples, n_ads_atoms, 3)
    sampled_lattices: np.ndarray,  # (n_samples, 6)
    sampled_supercell_matrices: np.ndarray,  # (n_samples, 3, 3) or (n_samples, 9)
    sampled_scaling_factors: np.ndarray,  # (n_samples,)
    prim_slab_atom_types: np.ndarray,  # (n_prim_slab_atoms,)
    ads_atom_types: np.ndarray,  # (n_ads_atoms,)
    prim_slab_atom_mask: np.ndarray,  # (n_prim_slab_atoms,) bool
    ads_atom_mask: np.ndarray,  # (n_ads_atoms,) bool
    calc: FAIRChemCalculator,  # Pre-initialized calculator
) -> List[Dict[str, Any]]:
    """
    Compute adsorption energy and structural validity for sampled structures.
    
    This function reconstructs the full system from prim_slab using supercell_matrix
    and scaling_factor, following the same logic as scripts/assemble.py.
    
    Args:
        sampled_prim_slab_coords: Sampled prim slab coordinates (n_samples, n_atoms, 3)
        sampled_ads_coords: Sampled adsorbate coordinates (n_samples, n_ads_atoms, 3)
        sampled_lattices: Sampled lattice parameters (n_samples, 6)
        sampled_supercell_matrices: Supercell transformation matrices (n_samples, 3, 3) or (n_samples, 9)
        sampled_scaling_factors: Z-direction scaling factors (n_samples,)
        prim_slab_atom_types: Atomic numbers for prim slab atoms
        ads_atom_types: Atomic numbers for adsorbate atoms
        prim_slab_atom_mask: Boolean mask for valid prim slab atoms
        ads_atom_mask: Boolean mask for valid adsorbate atoms
        calc: Pre-initialized FAIRChemCalculator (reused across calls)
    
    Returns:
        List of dicts containing E_adsorption and struct_valid for each sample.
    """
    n_samples = sampled_prim_slab_coords.shape[0]
    
    # Get valid adsorbate types for creating ads_atoms (for energy lookup)
    valid_ads_types = ads_atom_types[ads_atom_mask]
    
    results = []
    for i in range(n_samples):
        try:
            # Use assemble function to reconstruct the system
            recon_system, recon_slab = assemble(
                generated_prim_slab_coords=sampled_prim_slab_coords[i],
                generated_ads_coords=sampled_ads_coords[i],
                generated_lattice=sampled_lattices[i],
                generated_supercell_matrix=sampled_supercell_matrices[i].reshape(3, 3),
                generated_scaling_factor=sampled_scaling_factors[i],
                prim_slab_atom_types=prim_slab_atom_types,
                ads_atom_types=ads_atom_types,
                prim_slab_atom_mask=prim_slab_atom_mask,
                ads_atom_mask=ads_atom_mask,
            )
            
            # Create ads_atoms (adsorbate only, without cell, for energy lookup)
            valid_ads_coords = sampled_ads_coords[i][ads_atom_mask]
            ads_atoms = Atoms(
                numbers=valid_ads_types,
                positions=valid_ads_coords,
            )
            
            # Check structural validity BEFORE relaxation
            struct_valid = _structural_validity(recon_system)
            
            if not struct_valid:
                # Skip relaxation for invalid structures
                results.append({
                    "E_adsorption": float('nan'),
                    "struct_valid": False,
                })
                continue
            
            # Compute adsorption energy with relaxation (suppress LBFGS output)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                e_ads, e_sys, e_slab, e_adsorbate = relaxation_and_compute_adsorption_energy(
                    calc, recon_system, recon_slab, ads_atoms
                )
            
            # Check if calculation succeeded
            if e_ads == 999.0 or np.isnan(e_ads):
                results.append({
                    "E_adsorption": float('nan'),
                    "struct_valid": struct_valid,
                })
            else:
                results.append({
                    "E_adsorption": float(e_ads),
                    "struct_valid": struct_valid,
                })
                
        except Exception as e:
            print(f"WARNING: Failed to compute adsorption energy for sample {i}: {e}", file=sys.stderr)
            results.append({
                "E_adsorption": float('nan'),
                "struct_valid": False,
            })
    
    return results

def smact_validity(structures,
                   use_pauling_test=True,
                   include_alloys=True):
    if isinstance(structures, Atoms):
        structures = adaptor.get_structure(structures)
    elem_symbols = tuple(structures.composition.as_dict().keys())
    count = structures.composition.as_dict().values()
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(int(c),) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        print("No valid composition found in SMACT check.")
        return False

def crystal_validity(crystal, cutoff=4.1):
    if isinstance(crystal, Atoms):
        crystal = adaptor.get_structure(crystal)
    dist_mat = crystal.distance_matrix
    dist_mat += np.diag(np.ones(dist_mat.shape[0])*(cutoff + 10.))
    if dist_mat.min(axis=0).max() > cutoff:
        print("Crystal validity check failed: some atoms are not connected.")
        return False
    else:
        return True