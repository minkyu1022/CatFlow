import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Union
import argparse
import contextlib
import io
import time
import os
import functools
import multiprocessing

import torch
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.calculate import InferenceBatcher
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

OC20_GAS_PHASE_ENERGIES = {
    'H': -3.48483361833793,
    'O': -7.185616160375758,
    'C': -7.232295041080779,
    'N': -8.09079187764214,
}
# computed by UMA with reference molecules with adsorbate_E_table.py
# chemical potential (C) = E(CO) - E(H2O) + E(H2)
# chemical potential (H) = 0.5*E(H2)
# chemical potential (N) = 0.5*E(N2)
# chemical potential (O) = E(H2O) - E(H2)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def get_adsorbate_energy_from_table(atoms_obj):
    total_energy = 0.0
    symbols = atoms_obj.get_chemical_symbols()
    for atom_symbol in symbols:
        try:
            total_energy += OC20_GAS_PHASE_ENERGIES[atom_symbol]
        except KeyError:
            raise ValueError(f"Energy table does not contain '{atom_symbol}'")
    return total_energy

def relaxation_and_compute_adsorption_energy(calc, system, slab, adsorbate):
    try:
        system.calc = calc
        slab.calc = calc
        
        e_sys_unrelaxed = system.get_potential_energy()
        
        opt_sys = LBFGS(system, logfile=None)
        with contextlib.redirect_stdout(io.StringIO()):
            converged_system = opt_sys.run(0.05, 500)
        steps_system = opt_sys.get_number_of_steps()
        e_sys_relaxed = system.get_potential_energy()
        
        opt_slab = LBFGS(slab, logfile=None)
        with contextlib.redirect_stdout(io.StringIO()):
            converged_slab = opt_slab.run(0.05, 500)
        steps_slab = opt_slab.get_number_of_steps()
        e_slab_relaxed = slab.get_potential_energy()
        
        e_adsorbate = get_adsorbate_energy_from_table(adsorbate)
        
        initial_e_ads = e_sys_unrelaxed - (e_slab_relaxed + e_adsorbate)
        relaxed_e_ads = e_sys_relaxed - (e_slab_relaxed + e_adsorbate)
        
        return (initial_e_ads, relaxed_e_ads, e_sys_unrelaxed, e_sys_relaxed, 
                e_slab_relaxed, e_adsorbate, converged_system, converged_slab, 
                steps_system, steps_slab)
    except Exception as e:
        try: e_adsorbate = get_adsorbate_energy_from_table(adsorbate)
        except: e_adsorbate = 0.0
        return 999.0, 999.0, float('nan'), float('nan'), float('nan'), e_adsorbate, False, False, 0, 0

def process_single_traj(args, batch_predict_unit=None) -> Dict:
    traj_path, index, structure_dir_str = args
    result = {
        'index': index, 'initial_e_ads': None, 'relaxed_e_ads': None,
        'e_sys_unrelaxed': None, 'e_sys_relaxed': None, 'e_slab_relaxed': None, 'e_adsorbate': None,
        'converged_system': False, 'converged_slab': False, 'steps_system': 0, 'steps_slab': 0, 'error': None
    }
    try:
        try: generated_catalyst = read(traj_path)
        except Exception as read_err:
            result['error'] = f"Read Error: {read_err}"; return result

        generated_catalyst.center()
        system = generated_catalyst.copy()
        if 0 in system.get_tags(): system.set_constraint(FixAtoms(indices=[atom.index for atom in system if atom.tag == 0]))
        slab = generated_catalyst.copy()[generated_catalyst.get_tags() != 2]
        if 0 in slab.get_tags(): slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.tag == 0]))
        adsorbate = generated_catalyst.copy()[generated_catalyst.get_tags() == 2]
        if len(adsorbate) == 0: result['error'] = "No adsorbate atoms"; return result

        calc = FAIRChemCalculator(batch_predict_unit, task_name="oc20")
        
        (initial_e_ads, relaxed_e_ads, e_sys_unrelaxed, e_sys_relaxed, 
         e_slab_relaxed, e_adsorbate, converged_system, converged_slab, 
         steps_sys, steps_slab) = relaxation_and_compute_adsorption_energy(calc, system, slab, adsorbate)

        if relaxed_e_ads != 999.0:
            try:
                save_path = Path(structure_dir_str) / f"{index}.traj"
                system.calc = None 
                system.write(save_path)
            except Exception: pass
        
        result.update({
            'initial_e_ads': initial_e_ads, 'relaxed_e_ads': relaxed_e_ads,
            'e_sys_unrelaxed': e_sys_unrelaxed, 'e_sys_relaxed': e_sys_relaxed,
            'e_slab_relaxed': e_slab_relaxed, 'e_adsorbate': e_adsorbate,
            'converged_system': converged_system, 'converged_slab': converged_slab,
            'steps_system': steps_sys, 'steps_slab': steps_slab
        })
    except Exception as e:
        result['error'] = str(e)
    return result


def run_gpu_process(gpu_id, tasks, num_workers_per_gpu):

    try:
        if gpu_id >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device_str = "cuda"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            device_str = "cpu"

        predict_unit = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device_str)
        
        batcher = InferenceBatcher(
            predict_unit, 
            concurrency_backend_options={"max_workers": num_workers_per_gpu}
        )
        
        process_func = functools.partial(process_single_traj, batch_predict_unit=batcher.batch_predict_unit)
        results = list(batcher.executor.map(process_func, tasks))
        return results

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"CRITICAL ERROR in GPU {gpu_id} Process: {e}\n{error_msg}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description="Compute relaxation energies")
    parser.add_argument("--data_dir", type=str, default="unrelaxed_samples")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--output_dir", type=str, default="ref_energies")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist.")
        return

    traj_files = sorted(data_dir.rglob("*.traj"))
    if not traj_files: return
    
    subset_name = data_dir.name
    energy_export_dir = Path(args.output_dir) / subset_name
    energy_export_dir.mkdir(parents=True, exist_ok=True)
    structure_output_dir = energy_export_dir / "relaxed_structures"
    structure_output_dir.mkdir(parents=True, exist_ok=True)

    initial_jsonl = energy_export_dir / "initial_energy_results.jsonl"
    relaxed_jsonl = energy_export_dir / "relaxed_energy_results.jsonl"
    stats_json = energy_export_dir / "convergence_stats.json"
    
    with open(initial_jsonl, 'w') as f: pass
    with open(relaxed_jsonl, 'w') as f: pass
    
    print(f"Found {len(traj_files)} files. Processing {subset_name}...")

    num_gpus = 0
    if not args.use_cpu and torch.cuda.is_available():
        num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
    
    all_tasks = []
    for traj_file in traj_files:
        try:
            relative_path = traj_file.relative_to(data_dir)
            if relative_path.parent == Path('.'):
                index = relative_path.stem
                with contextlib.suppress(ValueError): index = int(index)
            else:
                index = f"{relative_path.parent}_{relative_path.stem}"
            all_tasks.append((str(traj_file), index, str(structure_output_dir)))
        except ValueError: continue
            
    final_results = []
    
    with open(initial_jsonl, 'w', buffering=1) as f_init, \
         open(relaxed_jsonl, 'w', buffering=1) as f_rel:

        if num_gpus > 0:
            print(f"Using {num_gpus} GPUs. Each GPU handles {args.num_workers} concurrent simulations.")
            print(f"Total concurrent capacity: {num_gpus * args.num_workers}")

            tasks_per_gpu = [[] for _ in range(num_gpus)]
            for i, task in enumerate(all_tasks):
                tasks_per_gpu[i % num_gpus].append(task)
                
            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=num_gpus) as pool:
                async_results = []
                for gpu_id in range(num_gpus):
                    if tasks_per_gpu[gpu_id]:
                        res = pool.apply_async(run_gpu_process, (gpu_id, tasks_per_gpu[gpu_id], args.num_workers))
                        async_results.append(res)
                
                print("Processing started on GPUs...")

                for res in tqdm(async_results, desc="Waiting for GPU batches"):
                    batch_results = res.get()
                    if not batch_results: print("[WARNING] No results from a worker.")
                    
                    for r in batch_results:
                        final_results.append(r)
                        
                        base = {k: r.get(k) for k in ['index', 'converged_system', 'converged_slab', 'steps_system', 'steps_slab', 'e_slab_relaxed', 'e_adsorbate', 'error']}
                        
                        init_entry = base.copy()
                        init_entry.update({'e_ads': r['initial_e_ads'], 'e_sys': r['e_sys_unrelaxed']})
                        
                        rel_entry = base.copy()
                        rel_entry.update({'e_ads': r['relaxed_e_ads'], 'e_sys': r['e_sys_relaxed']})
                        
                        f_init.write(json.dumps(init_entry, cls=NumpyEncoder) + "\n")
                        f_rel.write(json.dumps(rel_entry, cls=NumpyEncoder) + "\n")
                        
        else:
            print(f"Using CPU with {args.num_workers} workers.")
            final_results = run_gpu_process(-1, all_tasks, args.num_workers)
            for r in final_results:
                base = {k: r.get(k) for k in ['index', 'converged_system', 'converged_slab', 'steps_system', 'steps_slab', 'e_slab_relaxed', 'e_adsorbate', 'error']}
                init_entry = base.copy(); init_entry.update({'e_ads': r['initial_e_ads'], 'e_sys': r['e_sys_unrelaxed']})
                rel_entry = base.copy(); rel_entry.update({'e_ads': r['relaxed_e_ads'], 'e_sys': r['e_sys_relaxed']})
                
                f_init.write(json.dumps(init_entry, cls=NumpyEncoder) + "\n")
                f_rel.write(json.dumps(rel_entry, cls=NumpyEncoder) + "\n")

    total_processed = len(final_results)
    error_count = sum(1 for r in final_results if r['error'] is not None)
    converged_system_count = sum(1 for r in final_results if r.get('converged_system'))
    converged_slab_count = sum(1 for r in final_results if r.get('converged_slab'))
    converged_all_count = sum(1 for r in final_results if r.get('converged_system') and r.get('converged_slab'))

    stats_dict = {
        'total_samples': total_processed, 'error_count': error_count,
        'convergence': {'system_count': converged_system_count, 'slab_count': converged_slab_count, 'all_count': converged_all_count}
    }
    with open(stats_json, 'w') as f: json.dump(stats_dict, f, indent=2, cls=NumpyEncoder)
    
    print("\n[INFO] Processing Complete.")
    print("[INFO] Converting JSONL to final JSON format...")
    try:
        def jsonl_to_json(jsonl_path, json_path):
            data_list = []
            if os.path.exists(jsonl_path):
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        if line.strip(): data_list.append(json.loads(line))
                with contextlib.suppress(Exception): data_list.sort(key=lambda x: str(x.get('index', '')))
                with open(json_path, 'w') as f: json.dump(data_list, f, indent=2, cls=NumpyEncoder)
        jsonl_to_json(initial_jsonl, energy_export_dir / "initial_energy_results.json")
        jsonl_to_json(relaxed_jsonl, energy_export_dir / "relaxed_energy_results.json")
        print("Success! Final JSON files created.")
    except Exception as e: print(f"[WARNING] Failed to convert JSONL to JSON: {e}")

if __name__ == "__main__":
    main()