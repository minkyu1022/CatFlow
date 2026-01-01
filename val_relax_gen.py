import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple
from multiprocessing import cpu_count, get_context
import argparse
import contextlib
import io

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


def get_uma_calculator(model_name: str = "uma-s-1p1", device: str = "cuda") -> FAIRChemCalculator:
    """Get UMA calculator."""
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    return FAIRChemCalculator(predictor, task_name="oc20")


def relaxation_and_compute_adsorption_energy(
    calc: FAIRChemCalculator, 
    system: Atoms, 
    slab: Atoms, 
    adsorbate: Atoms
) -> Tuple[float, float, float, float, bool, bool]:
    try:
        system.calc = calc
        slab.calc = calc
        
        opt = LBFGS(system, logfile=None)
        with contextlib.redirect_stdout(io.StringIO()):
            converged_system = opt.run(0.05, 100)
        
        opt = LBFGS(slab, logfile=None)
        with contextlib.redirect_stdout(io.StringIO()):
            converged_slab = opt.run(0.05, 100)
        
        e_sys = system.get_potential_energy()
        e_slab = slab.get_potential_energy()
        e_adsorbate = get_adsorbate_energy_from_table(adsorbate)
        e_ads = e_sys - (e_slab + e_adsorbate)
        
        return e_ads, e_sys, e_slab, e_adsorbate, converged_system, converged_slab

    except Exception as e:
        print(f"WARNING: UMA calculation failed for a sample. Error: {e}", file=sys.stderr)
        e_adsorbate = get_adsorbate_energy_from_table(adsorbate)
        return 999.0, float('nan'), float('nan'), e_adsorbate, False, False


# Global variables for each worker process
_worker_gpu_id = None
_worker_calculator = None

def init_worker(gpu_id: int):
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
    
    try:
        _worker_calculator = get_uma_calculator(device=device_str)
    except Exception as e:
        print(f"WARNING: Failed to initialize calculator in worker: {e}", file=sys.stderr)
        _worker_calculator = None


def process_single_traj(args: Tuple[str, int]) -> Dict:
    traj_path, index = args
    
    result = {
        'index': index,
        'e_ads': None,
        'e_sys': None,
        'e_slab': None,
        'e_adsorbate': None,
        'converged_system': False,
        'converged_slab': False,
        'error': None
    }
    
    try:
        gpu_id = _worker_gpu_id if _worker_gpu_id is not None else -1
        
        device_str = "cpu"
        if gpu_id >= 0:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(gpu_id)
                    device_str = "cuda"
                else:
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device_str = "cpu"
        
        generated_catalyst = read(traj_path)
        system = generated_catalyst.copy()
        slab = generated_catalyst.copy()[generated_catalyst.get_tags() != 2]
        adsorbate = generated_catalyst.copy()[generated_catalyst.get_tags() == 2]
        
        calc = _worker_calculator
        if calc is None:
            calc = get_uma_calculator(device=device_str)
        
        e_ads, e_sys, e_slab, e_adsorbate, converged_system, converged_slab = relaxation_and_compute_adsorption_energy(
            calc, system, slab, adsorbate
        )
        
        result['e_ads'] = e_ads
        result['e_sys'] = e_sys
        result['e_slab'] = e_slab
        result['e_adsorbate'] = e_adsorbate
        result['converged_system'] = converged_system
        result['converged_slab'] = converged_slab
        
    except Exception as e:
        result['error'] = str(e)
        print(f"Index {index}: ERROR - {e}", file=sys.stderr)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute relaxation energies and convergence statistics"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="unrelaxed_samples/C2H4O",
        help="Directory containing traj files"
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
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Find all traj files recursively in subdirectories
    traj_files = sorted(data_dir.rglob("*.traj"))
    
    if len(traj_files) == 0:
        print(f"No .traj files found in {data_dir} or its subdirectories")
        return
    
    print(f"Found {len(traj_files)} traj files in {data_dir} and subdirectories")
    
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
    
    tasks = []
    for traj_file in traj_files:
        # Get relative path from data_dir to create unique index
        try:
            relative_path = traj_file.relative_to(data_dir)
            # Create unique index by combining subdirectory and filename
            # e.g., "0/123.traj" -> "0_123", "1/123.traj" -> "1_123"
            if relative_path.parent == Path('.'):
                # File is directly in data_dir
                index_str = relative_path.stem
                # Try to parse as integer
                try:
                    index = int(index_str)
                except ValueError:
                    index = index_str
            else:
                # File is in a subdirectory - use string index to avoid conflicts
                index = f"{relative_path.parent}_{relative_path.stem}"
        except ValueError:
            print(f"Warning: Could not create index from {traj_file}, skipping", file=sys.stderr)
            continue
        tasks.append((str(traj_file), index))
    
    print(f"Processing {len(tasks)} traj files...")
    
    results = []
    if num_workers == 1:
        init_worker(0 if num_gpus > 0 else -1)
        for task in tqdm(tasks, desc="Processing samples", unit="sample"):
            results.append(process_single_traj(task))
    else:
        ctx = get_context('spawn')
        
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
            tasks_per_gpu = [[] for _ in range(actual_gpus)]
            for i, task in enumerate(tasks):
                gpu_idx = i % actual_gpus
                tasks_per_gpu[gpu_idx].append(task)
            
            print(f"Tasks per GPU: {[len(t) for t in tasks_per_gpu]}")
            
            all_results = []
            pools = []
            
            try:
                for gpu_id in range(actual_gpus):
                    pool = ctx.Pool(
                        processes=workers_per_gpu,
                        initializer=init_worker,
                        initargs=(gpu_id,)
                    )
                    pools.append((pool, tasks_per_gpu[gpu_id]))
                
                import threading
                from threading import Lock
                
                results_list = [None] * actual_gpus
                progress_lock = Lock()
                completed_count = [0]
                total_tasks = len(tasks)
                
                pbar = tqdm(total=total_tasks, desc="Processing samples", unit="sample")
                
                def process_gpu_tasks(pool_idx, pool, gpu_tasks):
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
                
                for thread in threads:
                    thread.join()
                
                pbar.close()
                
                for gpu_results in results_list:
                    if gpu_results:
                        all_results.extend(gpu_results)
                
            finally:
                for pool, _ in pools:
                    pool.close()
                    pool.join()
            
            results = all_results
        else:
            with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(-1,)) as pool:
                results = list(tqdm(
                    pool.imap_unordered(process_single_traj, tasks),
                    total=len(tasks),
                    desc="Processing samples",
                    unit="sample"
                ))
    
    total_count = len(results)
    error_count = sum(1 for r in results if r['error'] is not None)
    
    converged_system_count = sum(1 for r in results if r.get('converged_system', False))
    converged_slab_count = sum(1 for r in results if r.get('converged_slab', False))
    converged_all_count = sum(1 for r in results if r.get('converged_system', False) and r.get('converged_slab', False))
    converged_system_rate = (converged_system_count / total_count * 100) if total_count > 0 else 0
    converged_slab_rate = (converged_slab_count / total_count * 100) if total_count > 0 else 0
    converged_all_rate = (converged_all_count / total_count * 100) if total_count > 0 else 0
    
    converged_all_indices = [r['index'] for r in results if r.get('converged_system', False) and r.get('converged_slab', False) and r['error'] is None]
    converged_all_indices.sort()
    
    stats_dict = {
        'total_samples': total_count,
        'error_count': error_count,
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
                'indices': converged_all_indices,
            },
        },
    }
    
    convergence_output_path = data_dir / "convergence_stats.json"
    with open(convergence_output_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\nConvergence statistics saved to: {convergence_output_path}")
    
    # Sort results: convert index to string for consistent sorting
    sorted_results = sorted(results, key=lambda x: str(x['index']))
    
    energy_results = []
    for r in sorted_results:
        result_entry = {
            'index': r['index'],
            'e_ads': r['e_ads'] if r['e_ads'] is not None else None,
            'e_sys': r['e_sys'] if r['e_sys'] is not None else None,
            'e_slab': r['e_slab'] if r['e_slab'] is not None else None,
            'e_adsorbate': r['e_adsorbate'] if r['e_adsorbate'] is not None else None,
            'converged_system': bool(r.get('converged_system', False)),
            'converged_slab': bool(r.get('converged_slab', False)),
        }
        if r['error'] is not None:
            result_entry['error'] = r['error']
        energy_results.append(result_entry)
    
    energy_output_path = data_dir / "energy_results.json"
    with open(energy_output_path, 'w') as f:
        json.dump(energy_results, f, indent=2)
    print(f"All energy results saved to: {energy_output_path}")


if __name__ == "__main__":
    main()