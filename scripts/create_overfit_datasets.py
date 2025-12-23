#!/usr/bin/env python3
"""
Randomly select 10 samples from LMDB dataset:
1. overfit_val.lmdb: Save the selected 10 samples as is
2. overfit_train.lmdb: Save the selected 10 samples replicated 10000 times
"""

import lmdb
import pickle
import random
import os
from tqdm import tqdm

def create_overfit_datasets(
    source_lmdb_path: str,
    output_dir: str,
    num_samples: int = 10,
    train_multiplier: int = 10000,
    seed: int = 42
):
    """
    Arguments
    ---------
    source_lmdb_path: str
        Path to source LMDB file
    output_dir: str
        Output directory path
    num_samples: int
        Number of samples to select (default: 10)
    train_multiplier: int
        Multiplier for replicating train dataset (default: 10000)
    seed: int
        Random seed (default: 42)
    """
    # Set random seed
    random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open source LMDB
    print(f"Opening source LMDB file: {source_lmdb_path}")
    source_env = lmdb.open(
        source_lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )
    
    # Get all keys
    print("Getting all keys...")
    with source_env.begin() as txn:
        keys = []
        cursor = txn.cursor()
        for key, _ in cursor:
            key_str = key.decode("ascii")
            if key_str != "length":
                keys.append(key_str)
    
    # Sort keys numerically
    try:
        keys = sorted(keys, key=int)
    except ValueError:
        keys = sorted(keys)
    
    total_samples = len(keys)
    print(f"Total number of samples: {total_samples}")
    
    if total_samples < num_samples:
        raise ValueError(f"Requested number of samples ({num_samples}) exceeds total number of samples ({total_samples}).")
    
    # Randomly select num_samples
    selected_indices = random.sample(range(total_samples), num_samples)
    selected_keys = [keys[i] for i in selected_indices]
    print(f"Selected indices: {selected_indices}")
    
    # Read selected sample data
    print("Reading selected sample data...")
    selected_data = []
    with source_env.begin() as txn:
        for key in tqdm(selected_keys, desc="Reading data"):
            value = txn.get(key.encode("ascii"))
            data_dict = pickle.loads(value)
            selected_data.append(data_dict)
    
    source_env.close()
    
    # 1. Create overfit_val.lmdb (10 samples as is)
    val_lmdb_path = os.path.join(output_dir, "overfit_val.lmdb")
    print(f"\nCreating {val_lmdb_path}...")
    val_env = lmdb.open(
        val_lmdb_path,
        map_size=1099511627776,  # 1TB max size
        subdir=False,
        meminit=False,
        map_async=True,
    )
    
    with val_env.begin(write=True) as txn:
        for idx, data_dict in enumerate(tqdm(selected_data, desc="Saving val")):
            txn.put(
                f"{idx}".encode("ascii"),
                pickle.dumps(data_dict, protocol=-1)
            )
    
    # Save length
    with val_env.begin(write=True) as txn:
        txn.put("length".encode("ascii"), pickle.dumps(len(selected_data), protocol=-1))
    
    val_env.sync()
    val_env.close()
    print(f"✓ {val_lmdb_path} created ({len(selected_data)} samples)")
    
    # 2. Create overfit_train.lmdb (10 samples replicated 10000 times)
    train_lmdb_path = os.path.join(output_dir, "overfit_train.lmdb")
    print(f"\nCreating {train_lmdb_path}...")
    train_env = lmdb.open(
        train_lmdb_path,
        map_size=1099511627776 * 2,  # 2TB max size
        subdir=False,
        meminit=False,
        map_async=True,
    )
    
    total_train_samples = len(selected_data) * train_multiplier
    print(f"Total {total_train_samples} samples to be saved ({len(selected_data)} samples × {train_multiplier} times)")
    
    train_idx = 0
    with train_env.begin(write=True) as txn:
        for _ in tqdm(range(train_multiplier), desc="Replicating"):
            for data_dict in selected_data:
                txn.put(
                    f"{train_idx}".encode("ascii"),
                    pickle.dumps(data_dict, protocol=-1)
                )
                train_idx += 1
    
    # Save length
    with train_env.begin(write=True) as txn:
        txn.put("length".encode("ascii"), pickle.dumps(total_train_samples, protocol=-1))
    
    train_env.sync()
    train_env.close()
    print(f"✓ {train_lmdb_path} created ({total_train_samples} samples)")
    
    print("\nDone!")
    print(f"  - Val dataset: {val_lmdb_path} ({len(selected_data)} samples)")
    print(f"  - Train dataset: {train_lmdb_path} ({total_train_samples} samples)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create overfit dataset")
    parser.add_argument(
        "--source",
        type=str,
        default="dataset/train/dataset.lmdb",
        help="Path to source LMDB file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/overfit",
        help="Output directory path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to select (default: 10)"
    )
    parser.add_argument(
        "--train-multiplier",
        type=int,
        default=10000,
        help="Multiplier for replicating train dataset (default: 10000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    create_overfit_datasets(
        source_lmdb_path=args.source,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        train_multiplier=args.train_multiplier,
        seed=args.seed
    )

