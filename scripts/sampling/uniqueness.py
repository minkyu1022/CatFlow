import os
import glob
import time
from multiprocessing import Pool
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm
import argparse

def process_single_file(file_path):
    """
    Reads a single traj file and converts it to a Pymatgen Structure object.
    Executed in a parallel worker process.
    """
    try:
        # Read with ASE
        atoms = read(file_path)
        
        tags = atoms.get_tags()
        slab_atoms = atoms[tags != 2]
        
        if len(slab_atoms) == 0:
            return None
        
        # Convert to Pymatgen Structure
        adaptor = AseAtomsAdaptor()
        struct = adaptor.get_structure(slab_atoms)
        return struct
    except Exception:
        return None

def get_uniqueness_parallel(base_directory, file_pattern="*.traj", num_workers=4):
    """
    Calculates uniqueness by reading traj files in parallel.
    
    Args:
        base_directory (str): Root directory to search.
        file_pattern (str): File extension pattern.
        num_workers (int): Number of processor cores to use.
    """
    
    print(f"Searching for files in [{base_directory}]...")
    search_path = os.path.join(base_directory, "**", file_pattern)
    traj_files = glob.glob(search_path, recursive=True)
    
    total_files = len(traj_files)
    if total_files == 0:
        print("No files found.")
        return None

    print(f"Found {total_files} files.")
    print(f"Starting parallel processing (Workers: {num_workers})...")
    
    start_time = time.time()

    # Parallel loading and conversion
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_file, traj_files), 
                            total=total_files, 
                            desc="Loading Parallel"))
    
    # Filter out failed loads (None)
    pmg_structures = [s for s in results if s is not None]
    
    load_time = time.time() - start_time
    print(f"Data loading complete! (Time: {load_time:.2f}s, Success: {len(pmg_structures)})")

    if not pmg_structures:
        return None

    # Structural duplicate check
    # StructureMatcher is run in the main process as it handles complex comparisons
    print("Checking for structural duplicates (StructureMatcher)...")
    matcher_start_time = time.time()
    
    matcher = StructureMatcher()
    grouped_structures = matcher.group_structures(pmg_structures)
    
    matcher_time = time.time() - matcher_start_time
    
    # Calculate statistics
    unique_count = len(grouped_structures)
    total_count = len(pmg_structures)
    uniqueness_score = unique_count / total_count if total_count > 0 else 0
    
    result = {
        "total_count": total_count,
        "unique_count": unique_count,
        "uniqueness": uniqueness_score,
        "load_time": load_time,
        "match_time": matcher_time
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, default="unrelaxed_samples")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()
    target_dir = args.target_dir
    num_workers = args.num_workers
    
    stats = get_uniqueness_parallel(target_dir, num_workers=num_workers)
    
    if stats:
        print("=" * 40)
        print(f"Directory      : {target_dir}")
        print(f"Total Loaded   : {stats['total_count']}")
        print(f"Unique Groups  : {stats['unique_count']}")
        print(f"Uniqueness     : {stats['uniqueness']:.6f}")
        print(f"Load Time      : {stats['load_time']:.2f}s")
        print(f"Match Time     : {stats['match_time']:.2f}s")
        print("=" * 40)