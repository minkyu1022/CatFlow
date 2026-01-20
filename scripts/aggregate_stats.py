import argparse
import json
import os
from pathlib import Path
import glob
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def aggregate_results(base_output_dir, output_filename="all_results_summary.json"):
    base_path = Path(base_output_dir)
    if not base_path.exists():
        print(f"[ERROR] Base directory not found: {base_path}")
        return

    print(f"[INFO] Aggregating results from: {base_path}")

    stats_files = list(base_path.glob("**/stats.json"))
    
    all_data = {}
    summary_total = {
        "total_processed": 0,
        "total_valid": 0,
        "total_invalid": 0,
        "total_failed": 0,
        "overall_validity_rate": 0.0,
        "subsets_count": 0,
        "validity_breakdown": {
            "vol_failed": 0,
            "dist_failed": 0,
            "height_failed": 0,
            "assemble_failed": 0,
        }
    }

    for stat_file in stats_files:
        try:
            with open(stat_file, 'r') as f:
                data = json.load(f)
    
            subset_name = stat_file.parent.name
            
            all_data[subset_name] = data
            
            summary_total["subsets_count"] += 1
            summary_total["total_processed"] += data.get("total_processed", 0)
            summary_total["total_valid"] += data.get("total_valid", 0)
            summary_total["total_invalid"] += data.get("total_invalid", 0)
            summary_total["total_failed"] += data.get("total_failed", 0)
            
            if "validity_stats" in data:
                for key in summary_total["validity_breakdown"]:
                    summary_total["validity_breakdown"][key] += data["validity_stats"].get(key, 0)

        except Exception as e:
            print(f"[WARNING] Could not read {stat_file}: {e}")

    if summary_total["total_processed"] > 0:
        summary_total["overall_validity_rate"] = (summary_total["total_valid"] / summary_total["total_processed"]) * 100
    
    final_output = {
        "summary": summary_total,
        "details_per_subset": all_data
    }

    output_path = base_path / output_filename
    try:
        with open(output_path, 'w') as f:
            json.dump(final_output, f, indent=4, cls=NumpyEncoder)
        print(f"[SUCCESS] Aggregated results saved to: {output_path}")
        
        print("="*50)
        print(f"Total Subsets: {summary_total['subsets_count']}")
        print(f"Total Valid Samples: {summary_total['total_valid']} / {summary_total['total_processed']}")
        print(f"Overall Validity Rate: {summary_total['overall_validity_rate']:.2f}%")
        print("="*50)
        
    except Exception as e:
        print(f"[ERROR] Failed to save aggregated json: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate stats.json files from subdirectories.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base output directory containing subset folders")
    args = parser.parse_args()
    
    aggregate_results(args.base_dir)