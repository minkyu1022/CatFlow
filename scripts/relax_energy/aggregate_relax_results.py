import json
import glob
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_json_as_dict(file_path, key_field='index'):

    if not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

            if isinstance(data, list):
                return {str(item.get(key_field)): item for item in data}
            return {}
    except Exception as e:
        print(f"[WARNING] Failed to load {file_path}: {e}")
        return {}

def aggregate_results(base_dir):
    """
    - base_dir/relaxed_energies/{adsorbate_symbol}.json
    - base_dir/initial_energies/{adsorbate_symbol}.json
    - base_dir/convergence_stats/{adsorbate_symbol}.json
    """
    relaxed_dir = os.path.join(base_dir, "relaxed_energies")
    initial_dir = os.path.join(base_dir, "initial_energies")
    stats_dir = os.path.join(base_dir, "convergence_stats")
    
    if not os.path.exists(relaxed_dir):
        print(f"[ERROR] Directory not found: {relaxed_dir}")
        print(f"       Make sure you are pointing to the correct directory (e.g., 'model_relax_results/CatGPT')")
        return
    
    search_pattern = os.path.join(relaxed_dir, "*.json")
    relaxed_files = glob.glob(search_pattern)
    
    if not relaxed_files:
        print(f"[WARNING] No JSON files found in {relaxed_dir}")
        return

    print(f"[INFO] Found {len(relaxed_files)} adsorbate subsets. Aggregating results (Strict Mode: Converged Only)...")

    csv_data_list = []
    json_data_dict = {}
    
    # Global Weighted Average Accumulators
    global_stats = {
        "weighted_steps_system": 0.0,
        "weighted_steps_slab": 0.0,
        "weighted_energy_gain": 0.0,
        "weighted_initial_energy": 0.0,
        "weighted_final_energy": 0.0,
        "total_converged_system_count": 0,
        "total_converged_slab_count": 0,
        "total_energy_valid_count": 0
    }
    
    for relaxed_file in relaxed_files:
        path_obj = Path(relaxed_file)
        adsorbate_name = path_obj.stem 
        
        # 1. Load Relaxed Energy Results
        relaxed_data = load_json_as_dict(relaxed_file, 'index')
        
        # 2. Load Initial Energy Results 
        initial_file = os.path.join(initial_dir, f"{adsorbate_name}.json")
        initial_data = load_json_as_dict(initial_file, 'index')
        
        # 3. Load Convergence Stats 
        stats_file = os.path.join(stats_dir, f"{adsorbate_name}.json")
        conv_data = {}
        if os.path.exists(stats_file):
            try:
                with open(stats_file, 'r') as f:
                    conv_data = json.load(f)
            except Exception as e:
                print(f"[WARNING] Failed to read stats for {adsorbate_name}: {e}")
        else:
            print(f"[WARNING] Convergence stats not found for {adsorbate_name}")
        
        subset_energy_gains = []
        subset_initial_energies = []
        subset_final_energies = []
        subset_system_steps = []
        subset_slab_steps = []
        
        # Converged system count
        subset_converged_system_count = 0
    
        for idx, r_item in relaxed_data.items():

            i_item = initial_data.get(idx)
            
            if r_item.get('converged_system'):
                subset_converged_system_count += 1
            
            if r_item.get('converged_system'):
                steps = r_item.get('steps_system', 0)
                subset_system_steps.append(steps)
            
            if r_item.get('converged_slab'):
                steps = r_item.get('steps_slab', 0)
                subset_slab_steps.append(steps)
            
            if i_item and r_item:
                e_init = i_item.get('e_ads')
                e_final = r_item.get('e_ads')
                
                is_converged = r_item.get('converged_system', False)
                
                if (e_init is not None and e_final is not None and 
                    e_init != 999.0 and e_final != 999.0 and
                    is_converged):
                    
                    gain = e_init - e_final
                    subset_energy_gains.append(gain)
                    subset_initial_energies.append(e_init)
                    subset_final_energies.append(e_final)

        # Steps
        avg_sys_steps = np.mean(subset_system_steps) if subset_system_steps else 0.0
        avg_slab_steps = np.mean(subset_slab_steps) if subset_slab_steps else 0.0
        
        # Energy
        avg_gain = np.mean(subset_energy_gains) if subset_energy_gains else None
        avg_initial_eads = np.mean(subset_initial_energies) if subset_initial_energies else None
        avg_final_eads = np.mean(subset_final_energies) if subset_final_energies else None
        valid_energy_count = len(subset_energy_gains) 
        

        # System Steps
        sys_conv_count = len(subset_system_steps)
        if sys_conv_count > 0:
            global_stats["weighted_steps_system"] += (avg_sys_steps * sys_conv_count)
            global_stats["total_converged_system_count"] += sys_conv_count
            
        # Slab Steps
        slab_conv_count = len(subset_slab_steps)
        if slab_conv_count > 0:
            global_stats["weighted_steps_slab"] += (avg_slab_steps * slab_conv_count)
            global_stats["total_converged_slab_count"] += slab_conv_count

        # Energy
        if valid_energy_count > 0 and avg_gain is not None:
            global_stats["weighted_energy_gain"] += (avg_gain * valid_energy_count)
            global_stats["weighted_initial_energy"] += (avg_initial_eads * valid_energy_count)
            global_stats["weighted_final_energy"] += (avg_final_eads * valid_energy_count)
            global_stats["total_energy_valid_count"] += valid_energy_count

        # 6. Convergence Rates 
        total_samples = conv_data.get("total_samples", len(relaxed_data))
        conv_info = conv_data.get("convergence", {})
        
        sys_count_stat = subset_converged_system_count  
        
        rate_sys = (sys_count_stat / total_samples * 100) if total_samples > 0 else 0.0
        
        error_count = conv_data.get("error_count", 0)
        
        extracted_data = {
            "total_samples": total_samples,
            "error_count": error_count,
            "valid_count": sys_count_stat, 
            "convergence_rates": {
                "system": rate_sys
            },
            "avg_steps": {
                "system": avg_sys_steps,
                "slab": avg_slab_steps
            },
            "energy_stats": {
                "avg_relaxation_gain": avg_gain,
                "avg_initial_eads": avg_initial_eads,
                "avg_final_eads": avg_final_eads,
                "valid_count": valid_energy_count 
            }
        }

        json_data_dict[adsorbate_name] = extracted_data
        
        row = {
            "Subset": adsorbate_name,
            "Total Samples": total_samples,
            "Errors": extracted_data["error_count"],
            "Valid (Conv) Count": sys_count_stat,
            "Conv Rate (System) %": rate_sys,
            "Avg Steps (System)": avg_sys_steps,
            "Avg Relax Gain (eV)": avg_gain,
            "Avg Initial E_ads (eV)": avg_initial_eads,
            "Avg Final E_ads (eV)": avg_final_eads
        }
        csv_data_list.append(row)

    global_avg_sys_steps = 0.0
    if global_stats["total_converged_system_count"] > 0:
        global_avg_sys_steps = global_stats["weighted_steps_system"] / global_stats["total_converged_system_count"]

    global_avg_slab_steps = 0.0
    if global_stats["total_converged_slab_count"] > 0:
        global_avg_slab_steps = global_stats["weighted_steps_slab"] / global_stats["total_converged_slab_count"]

    global_avg_gain = 0.0
    global_avg_initial_eads = 0.0
    global_avg_final_eads = 0.0
    if global_stats["total_energy_valid_count"] > 0:
        global_avg_gain = global_stats["weighted_energy_gain"] / global_stats["total_energy_valid_count"]
        global_avg_initial_eads = global_stats["weighted_initial_energy"] / global_stats["total_energy_valid_count"]
        global_avg_final_eads = global_stats["weighted_final_energy"] / global_stats["total_energy_valid_count"]

    json_data_dict["_GLOBAL_STATS"] = {
        "global_average_steps_system": global_avg_sys_steps,
        "global_average_steps_slab": global_avg_slab_steps,
        "global_average_relaxation_gain": global_avg_gain,
        "global_average_initial_eads": global_avg_initial_eads,
        "global_average_final_eads": global_avg_final_eads,
        "total_system_converged_count": global_stats["total_converged_system_count"],
        "total_slab_converged_count": global_stats["total_converged_slab_count"],
        "total_energy_valid_count": global_stats["total_energy_valid_count"]
    }

    output_json = os.path.join(base_dir, "final_relaxation_summary.json")
    try:
        with open(output_json, 'w') as f:
            json.dump(json_data_dict, f, indent=4, cls=NumpyEncoder)
        print(f"[SUCCESS] JSON summary saved to: {output_json}")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")

    if csv_data_list:
        df = pd.DataFrame(csv_data_list)
        df = df.sort_values("Subset")
        
        summary_row = {
            "Subset": "TOTAL / GLOBAL AVG",
            "Total Samples": df["Total Samples"].sum(),
            "Errors": df["Errors"].sum(),
            "Valid (Conv) Count": df["Valid (Conv) Count"].sum(),
            "Conv Rate (System) %": df["Conv Rate (System) %"].mean(),
            "Avg Steps (System)": global_avg_sys_steps,      
            "Avg Relax Gain (eV)": global_avg_gain,          
            "Avg Initial E_ads (eV)": global_avg_initial_eads,  
            "Avg Final E_ads (eV)": global_avg_final_eads       
        }
        
        df_summary = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        print("\n" + "="*120)
        print("AGGREGATED RELAXATION RESULTS (Strict Mode: Converged Only)")
        print("="*120)

        pd.set_option('display.max_rows', None)
        print(df_summary.to_string(index=False, float_format=lambda x: "{:.4f}".format(x) if isinstance(x, (float, int)) and not pd.isna(x) else str(x)))
        print("="*120)
        
        print(f"\n[SUMMARY]")
        print(f"  - Global Avg Steps (System): {global_avg_sys_steps:.4f}")
        print(f"  - Global Avg Steps (Slab): {global_avg_slab_steps:.4f}")
        print(f"  - Global Avg Relaxation Gain: {global_avg_gain:.4f} eV")
        print(f"  - Global Avg Initial E_ads: {global_avg_initial_eads:.4f} eV")
        print(f"  - Global Avg Final E_ads: {global_avg_final_eads:.4f} eV")
    else:
        print("[WARNING] No valid data extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate relaxation results from new folder structure.")
    parser.add_argument("--base_dir", type=str, required=True, 
                        help="Base directory containing relaxed_energies, initial_energies, and convergence_stats folders (e.g., model_relax_results/CatGPT)")
    args = parser.parse_args()
    
    if not os.path.exists(args.base_dir):
        print(f"[ERROR] Directory not found: {args.base_dir}")
    else:
        aggregate_results(args.base_dir)