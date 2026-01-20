import os
import glob
import time
import numpy as np
from multiprocessing import Pool
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import CrystalNNFingerprint
from tqdm import tqdm
import argparse

# --- Global Featurizers ---
# 1. Composition: Magpie preset (atomic mass, electronegativity, etc.)
magpie_featurizer = ElementProperty.from_preset("magpie")

# 2. Structure: CrystalNN Fingerprint (Geometric local environment)
# "ops" preset is commonly used for Order Parameters in structure recognition
cnn_featurizer = CrystalNNFingerprint.from_preset("ops")

def process_single_file(file_path):
    """
    Reads a file and calculates both Compositional and Structural fingerprints.
    """
    try:
        # 1. Load Structure
        atoms = read(file_path)
        adaptor = AseAtomsAdaptor()
        struct = adaptor.get_structure(atoms)
        
        # 2. Calculate Composition Fingerprint
        # Pymatgen automatically handles stoichiometry (e.g., A12B12 -> AB)
        comp_fp = magpie_featurizer.featurize(struct.composition)
        
        # 3. Calculate Structural Fingerprint
        # Per the snippet: Calculate CrystalNN FP for every site, then take the mean.
        site_fps = []
        for i in range(len(struct)):
            # featurize returns the fingerprint vector for site i
            fp = cnn_featurizer.featurize(struct, i)
            site_fps.append(fp)
        
        # Average site fingerprints to get a single vector for the whole structure
        struct_fp = np.mean(site_fps, axis=0)
        
        return (comp_fp, struct_fp)

    except Exception:
        # Return None if any step (loading or featurization) fails
        return None

def calculate_diversities(base_directory, file_pattern="*.traj", num_workers=4):
    """
    Calculates both Compositional and Structural Diversity using parallel processing.
    """
    
    print(f"Searching for files in [{base_directory}]...")
    search_path = os.path.join(base_directory, "**", file_pattern)
    traj_files = glob.glob(search_path, recursive=True)
    
    total_files = len(traj_files)
    if total_files == 0:
        print("No files found.")
        return None

    print(f"Found {total_files} files.")
    print(f"Starting parallel feature extraction (Workers: {num_workers})...")
    
    start_time = time.time()

    # Parallel Processing
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, traj_files), 
                            total=total_files, 
                            desc="Extracting Features"))
    
    # Filter out failed samples
    valid_results = [r for r in results if r is not None]
    
    load_time = time.time() - start_time
    print(f"Extraction complete! (Time: {load_time:.2f}s, Valid: {len(valid_results)})")

    if not valid_results:
        return None

    # Separate fingerprints into two lists
    comp_fps = [r[0] for r in valid_results]
    struct_fps = [r[1] for r in valid_results]

    calc_start_time = time.time()

    # --- 1. Compositional Diversity ---
    print("Calculating Compositional Diversity...")
    X_comp = np.array(comp_fps)
    
    # Apply Standard Scaling to Composition FPs (as per 'CompScaler' in the snippet)
    scaler = StandardScaler()
    X_comp_scaled = scaler.fit_transform(X_comp)
    
    # Calculate mean pairwise distance
    comp_div = pdist(X_comp_scaled, metric='euclidean').mean()

    # --- 2. Structural Diversity ---
    print("Calculating Structural Diversity...")
    X_struct = np.array(struct_fps)
    
    # Note: The snippet does NOT apply scaling to structural fingerprints.
    # It directly calculates pdist on the averaged CrystalNN vectors.
    struct_div = pdist(X_struct, metric='euclidean').mean()

    calc_time = time.time() - calc_start_time

    result = {
        "total_count": len(valid_results),
        "comp_diversity": comp_div,
        "struct_diversity": struct_div,
        "load_time": load_time,
        "calc_time": calc_time
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, default="unrelaxed_samples")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    target_dir = args.target_dir
    num_workers = args.num_workers
    
    stats = calculate_diversities(target_dir, num_workers=num_workers)
    
    if stats:
        print("=" * 40)
        print(f"Directory        : {target_dir}")
        print(f"Valid Samples    : {stats['total_count']}")
        print("-" * 40)
        print(f"Comp. Diversity  : {stats['comp_diversity']:.6f}")
        print(f"Struct. Diversity: {stats['struct_diversity']:.6f}")
        print("-" * 40)
        print(f"Extraction Time  : {stats['load_time']:.2f}s")
        print(f"Calculation Time : {stats['calc_time']:.2f}s")
        print("=" * 40)