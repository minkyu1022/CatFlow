#!/usr/bin/env python3
"""
모든 traj 파일에 대해 relaxation을 수행하고 ref_energy와 비교하는 스크립트
Multi-GPU 병렬처리 지원
"""

import os
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count, get_context
import argparse
import contextlib
import io

import numpy as np
from ase import Atoms
from ase.io import read
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

OC20_GAS_PHASE_ENERGIES = {
    'H': -3.477,
    'O': -7.204,
    'C': -7.282,
    'N': -8.083,
}


def get_adsorbate_energy_from_table(atoms_obj):
    """Calculate adsorbate energy from lookup table."""
    total_energy = 0.0
    symbols = atoms_obj.get_chemical_symbols()
    
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
    """Get UMA calculator."""
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    return FAIRChemCalculator(predictor, task_name="oc20")


def relaxation_and_compute_adsorption_energy(
    calc: FAIRChemCalculator, 
    system: Atoms, 
    slab: Atoms, 
    adsorbate: Atoms
) -> Tuple[float, float, float, float, bool]:
    """
    Relax system and compute adsorption energy.
    
    Returns:
        (e_ads, e_sys, e_slab, e_adsorbate, converged_system, converged_slab)
        converged_system: True if LBFGS optimizer converged, False otherwise
        converged_slab: True if LBFGS optimizer converged, False otherwise
    """
    try:
        system.calc = calc
        slab.calc = calc
        
        # Relax system (suppress LBFGS output)
        opt = LBFGS(system, logfile=None)
        # Suppress optimizer output by redirecting stdout
        with contextlib.redirect_stdout(io.StringIO()):
            converged_system = opt.run(0.05, 100)  # Returns True if converged, False otherwise
        
        opt = LBFGS(slab, logfile=None)
        # Suppress optimizer output by redirecting stdout
        with contextlib.redirect_stdout(io.StringIO()):
            converged_slab = opt.run(0.05, 100)  # Returns True if converged, False otherwise
        
        # Energy calculation
        e_sys = system.get_potential_energy()
        e_slab = slab.get_potential_energy()
        
        # relaxed_slab = system[system.get_tags() == 0]
        # relaxed_slab.calc = calc
        # e_slab = relaxed_slab.get_potential_energy()
        
        # Adsorbate energy is looked up in the table
        e_adsorbate = get_adsorbate_energy_from_table(adsorbate)
        
        # Adsorption energy calculation
        e_ads = e_sys - (e_slab + e_adsorbate)
        
        return e_ads, e_sys, e_slab, e_adsorbate, converged_system, converged_slab

    except Exception as e:
        # UMA calculation failed, return default value
        print(f"WARNING: UMA calculation failed for a sample. Error: {e}", file=sys.stderr)
        e_adsorbate = get_adsorbate_energy_from_table(adsorbate)
        return 999.0, float('nan'), float('nan'), e_adsorbate, False, False


# Global variables for each worker process
_worker_gpu_id = None
_worker_calculator = None

def init_worker(gpu_id: int):
    """Initialize worker process with specific GPU and shared calculator."""
    global _worker_gpu_id, _worker_calculator
    _worker_gpu_id = gpu_id
    
    if gpu_id >= 0:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                device_str = "cuda"
            else:
                device_str = "cpu"
        except ImportError:
            device_str = "cpu"
    else:
        device_str = "cpu"
    
    # Initialize calculator once per worker process (shared across all tasks in this worker)
    try:
        _worker_calculator = get_uma_calculator(device=device_str)
    except Exception as e:
        print(f"WARNING: Failed to initialize calculator in worker: {e}", file=sys.stderr)
        _worker_calculator = None


def process_single_traj(args: Tuple[str, int]) -> Dict:
    """
    Process a single traj file.
    
    Args:
        args: (traj_path, index)
    
    Returns:
        Dict with results: {'index': int, 'success': bool, 'e_ads': float, 'ref_energy': float, 'error': str or None}
    """
    traj_path, index = args
    
    result = {
        'index': index,
        'success': False,
        'e_ads': None,
        'ref_energy': None,
        'converged': False,
        'error': None
    }
    
    try:
        # Use the GPU ID set during worker initialization
        gpu_id = _worker_gpu_id if _worker_gpu_id is not None else -1
        
        # Set CUDA device for this process if using GPU
        device_str = "cpu"
        if gpu_id >= 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_id)
                    device_str = "cuda"  # fairchem expects "cuda" or "cpu"
                else:
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device_str = "cpu"
        
        # Load generated catalyst
        generated_catalyst = read(traj_path)
        
        # Split into system, slab, and adsorbate
        system = generated_catalyst.copy()
        slab = generated_catalyst.copy()[generated_catalyst.get_tags() != 2]
        adsorbate = generated_catalyst.copy()[generated_catalyst.get_tags() == 2]
        
        # Use shared calculator from worker initialization (reuse across all tasks in this worker)
        calc = _worker_calculator
        if calc is None:
            # Fallback: create calculator if initialization failed
            calc = get_uma_calculator(device=device_str)
        
        # Compute adsorption energy
        e_ads, e_sys, e_slab, e_adsorbate, converged_system, converged_slab = relaxation_and_compute_adsorption_energy(
            calc, system, slab, adsorbate
        )
        
        # Load reference energy
        ref_json_path = traj_path.replace('.traj', '_ref_E.json')
        with open(ref_json_path, 'r') as f:
            ref_data = json.load(f)
        ref_energy = ref_data['ref_energy']
        
        # Check if e_ads <= ref_energy
        result['e_ads'] = e_ads
        result['ref_energy'] = ref_energy
        result['success'] = abs(e_ads - ref_energy) <= 0.1
        result['converged_system'] = converged_system
        result['converged_slab'] = converged_slab
        
        gpu_label = f"GPU {gpu_id}" if gpu_id >= 0 else "CPU"
        print(f"[{gpu_label}] Index {index}: e_ads={e_ads:.6f}, ref_energy={ref_energy:.6f}, success={result['success']}, converged_system={converged_system}, converged_slab={converged_slab}", 
              file=sys.stderr)
        
    except Exception as e:
        gpu_label = f"GPU {gpu_id}" if gpu_id >= 0 else "CPU"
        result['error'] = str(e)
        print(f"[{gpu_label}] Index {index}: ERROR - {e}", file=sys.stderr)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate relaxation results against reference energies"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="unrelaxed_samples/structure_prediction",
        help="Directory containing traj files and ref_E.json files"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available GPUs or CPU if no GPU)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: num_gpus if specified, else CPU count)"
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force using CPU instead of GPU"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save statistics as JSON file (optional)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Find all traj files
    traj_files = sorted(data_dir.glob("*.traj"))
    
    if len(traj_files) == 0:
        print(f"No .traj files found in {data_dir}")
        return
    
    print(f"Found {len(traj_files)} traj files")
    
    # Determine number of GPUs and workers
    if args.use_cpu:
        num_gpus = 0
        num_workers = args.num_workers or cpu_count()
        print(f"Using CPU with {num_workers} workers")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                num_gpus = args.num_gpus if args.num_gpus is not None else available_gpus
                num_workers = args.num_workers if args.num_workers is not None else min(num_gpus, available_gpus)
                print(f"Using {num_gpus} GPU(s) with {num_workers} workers")
            else:
                num_gpus = 0
                num_workers = args.num_workers or cpu_count()
                print(f"No CUDA available, using CPU with {num_workers} workers")
        except ImportError:
            print("PyTorch not available, using CPU")
            num_gpus = 0
            num_workers = args.num_workers or cpu_count()
    
    # Prepare arguments for each traj file
    tasks = []
    for traj_file in traj_files:
        index_str = traj_file.stem  # filename without extension
        try:
            index = int(index_str)
        except ValueError:
            print(f"Warning: Could not parse index from {traj_file.name}, skipping", file=sys.stderr)
            continue
        
        # Check if corresponding ref_E.json exists
        ref_json_path = traj_file.with_name(f"{index_str}_ref_E.json")
        if not ref_json_path.exists():
            print(f"Warning: {ref_json_path} not found, skipping {traj_file.name}", file=sys.stderr)
            continue
        
        # Add task (GPU assignment is done per-worker, not per-task)
        tasks.append((str(traj_file), index))
    
    print(f"Processing {len(tasks)} valid traj files...")
    
    # Process tasks in parallel
    results = []
    if num_workers == 1:
        # Single process (useful for debugging)
        # Set worker GPU ID for single process
        init_worker(0 if num_gpus > 0 else -1)
        for task in tqdm(tasks, desc="Processing samples", unit="sample"):
            results.append(process_single_traj(task))
    else:
        # Multi-process - use 'spawn' method for CUDA compatibility
        # Use 'spawn' context to avoid CUDA initialization issues with fork
        ctx = get_context('spawn')
        
        # Strategy: Create separate pools for each GPU to ensure proper GPU assignment
        # Distribute workers evenly across GPUs
        if num_gpus > 0:
            try:
                import torch
                available_gpus = torch.cuda.device_count()
                actual_gpus = min(num_gpus, available_gpus)
                workers_per_gpu = max(1, num_workers // actual_gpus)
                print(f"Using {workers_per_gpu} workers per GPU (total {actual_gpus} GPUs)")
            except ImportError:
                available_gpus = 0
                workers_per_gpu = num_workers
                actual_gpus = 0
        else:
            actual_gpus = 0
            workers_per_gpu = num_workers
        
        if num_gpus > 0 and actual_gpus > 0:
            # Distribute tasks across GPUs (round-robin)
            tasks_per_gpu = [[] for _ in range(actual_gpus)]
            for i, task in enumerate(tasks):
                gpu_idx = i % actual_gpus
                tasks_per_gpu[gpu_idx].append(task)
            
            print(f"Tasks per GPU: {[len(t) for t in tasks_per_gpu]}")
            
            # Process each GPU's tasks in parallel using separate pools
            all_results = []
            pools = []
            
            try:
                # Create a pool for each GPU
                for gpu_id in range(actual_gpus):
                    pool = ctx.Pool(
                        processes=workers_per_gpu,
                        initializer=init_worker,
                        initargs=(gpu_id,)
                    )
                    pools.append((pool, tasks_per_gpu[gpu_id]))
                
                # Process tasks for each GPU pool with progress tracking
                import threading
                from threading import Lock
                
                results_list = [None] * actual_gpus
                progress_lock = Lock()
                completed_count = [0]  # Use list to allow modification in nested functions
                total_tasks = len(tasks)
                
                # Create a shared progress bar
                pbar = tqdm(total=total_tasks, desc="Processing samples", unit="sample")
                
                def process_gpu_tasks(pool_idx, pool, gpu_tasks):
                    """Process tasks for a GPU pool and update progress."""
                    gpu_results = []
                    for result in pool.imap_unordered(process_single_traj, gpu_tasks):
                        gpu_results.append(result)
                        with progress_lock:
                            completed_count[0] += 1
                            pbar.update(1)
                    results_list[pool_idx] = gpu_results
                
                threads = []
                for pool_idx, (pool, gpu_tasks) in enumerate(pools):
                    thread = threading.Thread(target=process_gpu_tasks, args=(pool_idx, pool, gpu_tasks))
                    thread.start()
                    threads.append(thread)
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                pbar.close()
                
                # Combine results from all GPUs
                for gpu_results in results_list:
                    if gpu_results:
                        all_results.extend(gpu_results)
                
            finally:
                # Close all pools
                for pool, _ in pools:
                    pool.close()
                    pool.join()
            
            results = all_results
        else:
            # CPU mode: single pool
            with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(-1,)) as pool:
                results = list(tqdm(
                    pool.imap_unordered(process_single_traj, tasks),
                    total=len(tasks),
                    desc="Processing samples",
                    unit="sample"
                ))
    
    # Aggregate results
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    error_count = sum(1 for r in results if r['error'] is not None)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    # Aggregate convergence results
    converged_system_count = sum(1 for r in results if r.get('converged_system', False))
    converged_slab_count = sum(1 for r in results if r.get('converged_slab', False))
    converged_all_count = sum(1 for r in results if r.get('converged_system', False) and r.get('converged_slab', False))
    converged_system_rate = (converged_system_count / total_count * 100) if total_count > 0 else 0
    converged_slab_rate = (converged_slab_count / total_count * 100) if total_count > 0 else 0
    converged_all_rate = (converged_all_count / total_count * 100) if total_count > 0 else 0
    
    # Success statistics
    success_and_converged_all_count = sum(1 for r in results if r['success'] and r.get('converged_system', False) and r.get('converged_slab', False))
    success_and_converged_all_rate = (success_and_converged_all_count / total_count * 100) if total_count > 0 else 0
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples processed: {total_count}")
    print(f"Successful samples (abs(e_ads - ref_energy) <= 0.1): {success_count}")
    print(f"Failed samples (abs(e_ads - ref_energy) > 0.1): {total_count - success_count - error_count}")
    print(f"Error samples: {error_count}")
    print(f"Success rate: {success_rate:.2f}%")
    print("-"*60)
    print(f"Converged system (LBFGS converged): {converged_system_count}")
    print(f"Non-converged system (LBFGS not converged): {total_count - converged_system_count - error_count}")
    print(f"System convergence rate: {converged_system_rate:.2f}%")
    print("-"*60)
    print(f"Converged slab (LBFGS converged): {converged_slab_count}")
    print(f"Non-converged slab (LBFGS not converged): {total_count - converged_slab_count - error_count}")
    print(f"Slab convergence rate: {converged_slab_rate:.2f}%")
    print("-"*60)
    print(f"Converged all (both system and slab): {converged_all_count}")
    print(f"Non-converged all: {total_count - converged_all_count - error_count}")
    print(f"Converged all rate: {converged_all_rate:.2f}%")
    print("-"*60)
    print(f"Success and converged all: {success_and_converged_all_count}")
    print(f"Success and converged all rate: {success_and_converged_all_rate:.2f}%")
    print("="*60)
    
    # Print detailed statistics
    if success_count > 0:
        success_e_ads = [r['e_ads'] for r in results if r['success'] and r['e_ads'] is not None]
        if success_e_ads:
            print(f"\nSuccessful samples - e_ads statistics:")
            print(f"  Mean: {np.mean(success_e_ads):.6f}")
            print(f"  Min: {np.min(success_e_ads):.6f}")
            print(f"  Max: {np.max(success_e_ads):.6f}")
    
    failed_results = [r for r in results if not r['success'] and r['error'] is None]
    if failed_results:
        failed_e_ads = [r['e_ads'] for r in failed_results if r['e_ads'] is not None]
        if failed_e_ads:
            print(f"\nFailed samples - e_ads statistics:")
            print(f"  Mean: {np.mean(failed_e_ads):.6f}")
            print(f"  Min: {np.min(failed_e_ads):.6f}")
            print(f"  Max: {np.max(failed_e_ads):.6f}")
    
    # Prepare statistics dictionary for JSON output
    stats_dict = {
        'total_samples': total_count,
        'success': {
            'count': success_count,
            'rate_percent': success_rate,
        },
        'failed': {
            'count': total_count - success_count - error_count,
        },
        'error': {
            'count': error_count,
        },
        'convergence': {
            'system': {
                'count': converged_system_count,
                'rate_percent': converged_system_rate,
            },
            'slab': {
                'count': converged_slab_count,
                'rate_percent': converged_slab_rate,
            },
            'all': {
                'count': converged_all_count,
                'rate_percent': converged_all_rate,
            },
        },
        'success_and_converged_all': {
            'count': success_and_converged_all_count,
            'rate_percent': success_and_converged_all_rate,
        },
    }
    
    # Add detailed e_ads statistics if available
    if success_count > 0:
        success_e_ads = [r['e_ads'] for r in results if r['success'] and r['e_ads'] is not None]
        if success_e_ads:
            stats_dict['success']['e_ads_statistics'] = {
                'mean': float(np.mean(success_e_ads)),
                'min': float(np.min(success_e_ads)),
                'max': float(np.max(success_e_ads)),
            }
    
    if failed_results:
        failed_e_ads = [r['e_ads'] for r in failed_results if r['e_ads'] is not None]
        if failed_e_ads:
            stats_dict['failed']['e_ads_statistics'] = {
                'mean': float(np.mean(failed_e_ads)),
                'min': float(np.min(failed_e_ads)),
                'max': float(np.max(failed_e_ads)),
            }
    
    
    # Save to JSON file if specified
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        print(f"\nStatistics saved to: {output_path}")
        
    # Save all e_ads results to ads_E_results.json
    # Sort results by index for consistent ordering
    sorted_results = sorted(results, key=lambda x: x['index'])
    
    # Prepare results data (all samples, regardless of success)
    ads_e_results = []
    for r in sorted_results:
        result_entry = {
            'index': r['index'],
            'e_ads': r['e_ads'] if r['e_ads'] is not None else None,
            'ref_energy': r['ref_energy'] if r['ref_energy'] is not None else None,
            'success': bool(r['success']),  # Convert to Python bool for JSON serialization
            'converged_system': bool(r.get('converged_system', False)),  # Convert to Python bool
            'converged_slab': bool(r.get('converged_slab', False)),  # Convert to Python bool
        }
        if r['error'] is not None:
            result_entry['error'] = r['error']
        ads_e_results.append(result_entry)
    
    # Save to ads_E_results.json in data_dir
    ads_e_output_path = data_dir / "ads_E_results.json"
    with open(ads_e_output_path, 'w') as f:
        json.dump(ads_e_results, f, indent=2)
    print(f"\nAll e_ads results saved to: {ads_e_output_path}")


if __name__ == "__main__":
    main()

