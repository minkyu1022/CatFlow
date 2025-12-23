from __future__ import annotations

import math
import os
import pickle
import pandas as pd
import ast
import lmdb
import numpy as np
import torch
import itertools
import functools
from collections import Counter, defaultdict
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.util.coord import lattice_points_in_supercell
from ase import Atoms
from ase.io import write
from pymatgen.core import Lattice, Structure

def calculate_rmsd_pymatgen(
    struct1: Atoms | Structure,
    struct2: Atoms | Structure,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5,
    primitive_cell: bool = False,
) -> float | None:
    """
    Calculate RMSD using Pymatgen StructureMatcher.
    
    Arguments
    ---------
    struct1, struct2: Atoms | Structure
        ASE Atoms object or pymatgen Structure object
    ltol: float
        Lattice length tolerance (default: 0.2)
    stol: float
        Site distance tolerance (default: 0.3)
    angle_tol: float
        Angle tolerance in degrees (default: 5)
        
    Returns
    -------
    float | None
        RMSD value (Å) or None if structures don't match
    """
    # Convert ASE Atoms to pymatgen Structure
    if hasattr(struct1, 'get_positions'):
        struct1 = AseAtomsAdaptor.get_structure(struct1)
    if hasattr(struct2, 'get_positions'):
        struct2 = AseAtomsAdaptor.get_structure(struct2)
    
    # Create StructureMatcher
    matcher = StructureMatcher(
        primitive_cell=primitive_cell,
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol
    )
    
    # Check if structures match
    if matcher.fit(struct1, struct2):
        # Calculate RMS distance
        rms_dist, max_dist = matcher.get_rms_dist(struct1, struct2)
        return rms_dist, max_dist
    else:
        return None, None

def get_info_from_metadata(df, index):
    row = df.iloc[index]

    def parse(val, dtype=None, np_type=None):
        if isinstance(val, str):
            val = ast.literal_eval(val)
        if dtype == 'array':
            return np.array(val, dtype=np_type)
        elif dtype:
            return dtype(val)
        return val

    data = {
        'meta': {
            'sid': row['sid'],
            'bulk_src_id': row['bulk_src_id'],
            'specific_miller': row['specific_miller'],
            'shift': row['shift'],
            'top': row['top'],
        },
        'config': {
            'n_c': parse(row['n_c'], int),
            'n_vac': parse(row['n_vac'], int),
            'height': parse(row['height'], float),
        },
        'structures': {
            'true_atomic_nums': parse(row['true_system_atomic_numbers'], 'array', int),
            'true_positions': parse(row['true_system_positions'], 'array', float),
            'true_lattice': parse(row['true_lattice'], 'array', float),
            'true_tags': parse(row['true_tags'], 'array'),
            'ads_pos_relaxed': parse(row['ads_pos_relaxed'], 'array', float),
        }
    }
    
    return data

def find_vacuum_axis_ase(atoms: Atoms) -> int:
    # 1. Get fractional coordinates (wrap=True brings all atoms to 0~1 range)
    scaled_positions = atoms.get_scaled_positions(wrap=True)
    cell_lengths = atoms.cell.lengths() # [a length, b length, c length]
    
    max_gap_size = -1.0
    vacuum_axis = 2 # default value
    
    # Iterate over x(0), y(1), z(2) axes
    for i in range(3):
        # Sort coordinates only for this axis
        coords = np.sort(scaled_positions[:, i])
        
        # Calculate gaps between adjacent atoms
        gaps = np.diff(coords)
        
        # Consider PBC: include (1.0 + first atom - last atom) as gap
        boundary_gap = 1.0 + coords[0] - coords[-1]
        
        # Largest gap for this axis (fractional)
        current_axis_max_frac_gap = max(np.max(gaps), boundary_gap)
        
        # Convert to real distance (Å)
        real_gap_len = current_axis_max_frac_gap * cell_lengths[i]
        
        # Compare with other axes to find the largest
        if real_gap_len > max_gap_size:
            max_gap_size = real_gap_len
            vacuum_axis = i
            
    return vacuum_axis

def align_vacuum_to_z_axis(atoms: Atoms, vac_axis_idx: int) -> Atoms:
    """
    Change the vacuum axis to z-axis (index 2) by reordering,
    and orient it to stand in the z-direction in space.
    """
    atoms = atoms.copy()
    
    # 1. Change axis order (Permutation)
    #    If vacuum is in x or y, move it to z (index 2) position.
    if vac_axis_idx == 0:   # x is vacuum -> change to (y, z, x) order (maintain Right-handed)
        new_order = [1, 2, 0]
    elif vac_axis_idx == 1: # y is vacuum -> change to (z, x, y) order
        new_order = [2, 0, 1]
    else:                   # z is already vacuum
        new_order = [0, 1, 2]

    # Perform axis reordering if needed
    if vac_axis_idx != 2:
        old_cell = atoms.get_cell()
        new_cell = old_cell[new_order]  # Only change vector order (direction remains the same)
        
        old_scaled = atoms.get_scaled_positions()
        new_scaled = old_scaled[:, new_order] # Change column order of fractional coordinates
        
        atoms.set_cell(new_cell)
        atoms.set_scaled_positions(new_scaled)

    # ------------------------------------------------------------------------
    # [Core] 2. Reset spatial orientation (Standardization)
    # Current state: cell[2] is the vacuum vector, but still lying sideways in space.
    # Solution: Extract only lattice lengths and angles (par), then redraw in "standard orientation".
    #       ASE's standard orientation rule: a-axis is x-axis, b-axis is in xy-plane, c-axis is in remaining z-direction
    # ------------------------------------------------------------------------
    
    # Extract [a, b, c, alpha, beta, gamma]
    cell_par = atoms.cell.cellpar() 
    
    # Create "new standard cell" from extracted parameters and rotate atoms accordingly
    # scale_atoms=True rotates atoms together
    atoms.set_cell(cell_par, scale_atoms=True)
    
    # # Wrap atoms outside cell (optional)
    # atoms.wrap()
    
    return atoms