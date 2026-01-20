import os
import json
import lmdb
import pickle
from glob import glob

lmdb_path = "/home/jovyan/mk-catgen-data/dataset/val_id/dataset.lmdb"
traj_dir = "/home/jovyan/MinCatFlow/unrelaxed_samples/sp_all"

env = lmdb.open(
    lmdb_path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)

with env.begin() as txn:
    traj_files = glob(os.path.join(traj_dir, "*.traj"))

    print(f"Found {len(traj_files)} traj files")

    for traj_path in traj_files:
        basename = os.path.basename(traj_path)
        index = os.path.splitext(basename)[0] 

        key = index.encode("utf-8")

        value = txn.get(key)
        if value is None:
            print(f"[WARN] index {index} not found in lmdb")
            continue

        data = pickle.loads(value)

        if "ref_energy" not in data:
            print(f"[WARN] ref_energy missing for index {index}")
            continue

        ref_energy = data["ref_energy"]

        out_path = os.path.join(traj_dir, f"{index}_ref_E.json")
        with open(out_path, "w") as f:
            json.dump(
                {"ref_energy": ref_energy},
                f,
                indent=2
            )

        print(f"[OK] Saved {out_path}")
