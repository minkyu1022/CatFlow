import os
import json
import lmdb
import pickle
from glob import glob

# 경로 설정
lmdb_path = "dataset/val_id/dataset.lmdb"
traj_dir = "unrelaxed_samples/de_novo_generation"

# lmdb 열기 (read-only)
env = lmdb.open(
    lmdb_path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
)

with env.begin() as txn:
    # 모든 traj 파일 순회
    traj_files = glob(os.path.join(traj_dir, "*.traj"))

    print(f"Found {len(traj_files)} traj files")

    for traj_path in traj_files:
        # index 추출
        basename = os.path.basename(traj_path)
        index = os.path.splitext(basename)[0]   # "{index}.traj" -> "{index}"

        key = index.encode("utf-8")

        value = txn.get(key)
        if value is None:
            print(f"[WARN] index {index} not found in lmdb")
            continue

        # lmdb value는 보통 pickle
        data = pickle.loads(value)

        if "ref_energy" not in data:
            print(f"[WARN] ref_energy missing for index {index}")
            continue

        ref_energy = data["ref_energy"]

        # json 저장
        out_path = os.path.join(traj_dir, f"{index}_ref_E.json")
        with open(out_path, "w") as f:
            json.dump(
                {"ref_energy": ref_energy},
                f,
                indent=2
            )

        print(f"[OK] Saved {out_path}")
