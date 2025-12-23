import lmdb
import pickle
import numpy as np
import json
from tqdm import tqdm

# ===== Read Primitive dataset =====
lmdb_path_prim = "/home/minkyu/EfficientCatGen/dataset/train/dataset.lmdb"

print("Reading primitive_slab data...")
env_prim = lmdb.open(
    lmdb_path_prim,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=True,
    meminit=False,
    max_readers=1,
)

num_atoms_list_prim = []

with env_prim.begin() as txn:
    cursor = txn.cursor()
    for key, value in tqdm(cursor, desc="Processing primitive_slab"):
        key_str = key.decode("ascii")
        if key_str == "length":
            continue
        
        data_dict = pickle.loads(value)
        primitive_slab = data_dict["primitive_slab"]
        num_atoms = len(primitive_slab)
        num_atoms_list_prim.append(num_atoms)

env_prim.close()

num_atoms_array_prim = np.array(num_atoms_list_prim)

# ===== Print statistics =====
print(f"\n=== Primitive Slab Atom Count Statistics ===")
print(f"Total samples: {len(num_atoms_array_prim)}")
print(f"Minimum: {np.min(num_atoms_array_prim)}")
print(f"Maximum: {np.max(num_atoms_array_prim)}")
print(f"Mean: {np.mean(num_atoms_array_prim):.2f}")
print(f"Median: {np.median(num_atoms_array_prim):.2f}")
print(f"Standard deviation: {np.std(num_atoms_array_prim):.2f}")

# ===== Calculate atom count distribution (index = atom count, value = probability) =====
max_atoms = int(np.max(num_atoms_array_prim))
min_atoms = int(np.min(num_atoms_array_prim))

# Calculate frequency for indices from 0 to max_atoms
distribution = np.zeros(max_atoms + 1, dtype=np.float64)

for num_atoms in num_atoms_array_prim:
    distribution[int(num_atoms)] += 1

# Normalize to probabilities
total_samples = len(num_atoms_array_prim)
distribution = distribution / total_samples

# Convert to list (for JSON saving)
distribution_list = distribution.tolist()

# ===== Save to JSON file =====
output_dict = {
    'primitive_dataset': distribution_list
}

output_file = 'primitive_atom_distribution.json'
with open(output_file, 'w') as f:
    json.dump(output_dict, f, indent=2)

print(f"\nAtom count distribution saved to '{output_file}'.")
print(f"Distribution range: 0 ~ {max_atoms} (total {len(distribution_list)} indices)")
print(f"Probability sum: {np.sum(distribution):.10f}")

# Print first few values (for verification)
print(f"\nFirst 10 values (indices 0~9):")
for i in range(min(10, len(distribution_list))):
    print(f"  Index {i}: {distribution_list[i]:.10f}")

