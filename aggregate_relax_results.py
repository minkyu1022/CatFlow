import json
import glob
import os
import argparse
import pandas as pd
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def aggregate_results(base_dir):

    search_pattern = os.path.join(base_dir, "*", "convergence_stats.json")
    stat_files = glob.glob(search_pattern)
    
    if not stat_files:
        print(f"[WARNING] No convergence_stats.json files found in {base_dir}")
        return

    print(f"[INFO] Found {len(stat_files)} statistics files. Aggregating...")

    csv_data_list = []
    
    json_data_dict = {}
    
    for fpath in stat_files:
        path_obj = Path(fpath)
        subset_name = path_obj.parent.name  
        
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            extracted_data = {
                "total_samples": data.get("total_samples", 0),
                "error_count": data.get("error_count", 0),
                "valid_count": data["convergence"]["all"]["count"],
                "convergence_rates": {
                    "all": data["convergence"]["all"]["rate_percent"],
                    "system": data["convergence"]["system"]["rate_percent"],
                    "slab": data["convergence"]["slab"]["rate_percent"],
                },
            }

            json_data_dict[subset_name] = extracted_data
            
            row = {
                "Subset": subset_name,
                "Total Samples": extracted_data["total_samples"],
                "Errors": extracted_data["error_count"],
                "Valid (Conv) Count": extracted_data["valid_count"],
                "Conv Rate (All) %": extracted_data["convergence_rates"]["all"],
                "Conv Rate (System) %": extracted_data["convergence_rates"]["system"],
                "Conv Rate (Slab) %": extracted_data["convergence_rates"]["slab"],
            }
            csv_data_list.append(row)
            
        except Exception as e:
            print(f"[ERROR] Failed to read {subset_name}: {e}")

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
        
        output_csv = os.path.join(base_dir, "final_relaxation_summary.csv")
        df.to_csv(output_csv, index=False, float_format="%.2f")
        
        print(f"[SUCCESS] CSV summary saved to: {output_csv}")
        print("\n" + "="*60)
        print(df.to_string(index=False))
        print("="*60)
    else:
        print("[WARNING] No valid data extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate relaxation results from subdirectories.")
    parser.add_argument("--base_dir", type=str, default="unrelaxed_samples/de_novo_generation", 
                        help="Base directory containing subset folders")
    args = parser.parse_args()
    
    aggregate_results(args.base_dir)