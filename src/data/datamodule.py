"""
DataModule for LMDB datasets with RAM caching and dynamic batch padding.
"""

import random
import math
import heapq
from collections import deque
from typing import Optional, Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import (
    Sampler,
    BatchSampler,
    SequentialSampler,
    RandomSampler,
)
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist

import lightning.pytorch as pl
from omegaconf import DictConfig

# Import LMDB dataset and collate functions
from src.data.lmdb_dataset import (
    LMDBCachedDataset,
    LMDBCachedDatasetWithTransform,
    collate_fn_with_dynamic_padding,
    collate_pyg_with_dynamic_padding,
)


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class OverfitSampler(Sampler):
    """Sampler for overfitting on a small subset of data."""

    def __init__(self, num_samples: int, dataset_len: int):
        self.num_samples = num_samples
        self.dataset_len = dataset_len

    def __iter__(self):
        idx_list = list(range(self.dataset_len))
        num_repeats = math.ceil(self.num_samples / self.dataset_len)
        full_list = (idx_list * num_repeats)[: self.num_samples]
        return iter(random.sample(full_list, len(full_list)))

    def __len__(self):
        return self.num_samples


class DynamicBatchSampler(BatchSampler):
    """
    Batch sampler that dynamically groups samples into batches based on a
    user-defined size metric (e.g., number of atoms, tokens, nodes).

    Each batch is constructed to stay within a maximum total unit count
    (`max_batch_units`), allowing for efficient batching of variable-sized data.

    Examples:
    - For molecular graphs: sort_key = lambda i: dataset[i]['n_atoms']
    - For general graphs: sort_key = lambda i: dataset[i].num_nodes
    """

    def __init__(
        self,
        dataset,
        max_batch_units: int = 1000,
        max_batch_size: Optional[int] = None,
        drop_last: bool = False,
        distributed: bool = False,
        sort_key: Callable = None,
        buffer_size_multiplier: int = 100,
        shuffle: bool = False,
        use_heap: bool = False,
    ):
        self.distributed = distributed
        if distributed:
            self.sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            self.sampler = (
                RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            )

        super().__init__(self.sampler, batch_size=1, drop_last=drop_last)

        self.max_batch_units = max_batch_units
        self.max_batch_size = max_batch_size
        self.sort_key = sort_key
        self.max_buffer_size = max_batch_units * buffer_size_multiplier
        self._epoch = 0
        self.shuffle = shuffle
        self.use_heap = use_heap
        self.drop_last = drop_last
        self.bucket_batches = []

    def __len__(self):
        if not self.bucket_batches:
            self._build_batches()
        return len(self.bucket_batches)

    def __iter__(self):
        self._build_batches()
        for batch, _ in self.bucket_batches:
            yield batch

    def _build_batches(self):
        buffer = []
        buffer_deque = deque()
        buffer_size = 0
        batch = []
        batch_units = 0
        bucket_batches = []

        indices = list(self.sampler)
        for index in indices:
            num_units = self.sort_key(index)
            if self.use_heap:
                heapq.heappush(buffer, (-num_units, index))
            else:
                buffer_deque.append((num_units, index))
            buffer_size += num_units

            while buffer_size > self.max_buffer_size:
                if self.use_heap:
                    neg_units, index = heapq.heappop(buffer)
                    num_units = -neg_units
                else:
                    num_units, index = buffer_deque.popleft()
                buffer_size -= num_units

                if (batch_units + num_units > self.max_batch_units) or (
                    self.max_batch_size and len(batch) >= self.max_batch_size
                ):
                    bucket_batches.append((batch, batch_units))
                    batch, batch_units = [], 0
                batch.append(index)
                batch_units += num_units

        while buffer if self.use_heap else buffer_deque:
            if self.use_heap:
                neg_units, index = heapq.heappop(buffer)
                num_units = -neg_units
            else:
                num_units, index = buffer_deque.popleft()
            if (batch_units + num_units > self.max_batch_units) or (
                self.max_batch_size and len(batch) >= self.max_batch_size
            ):
                bucket_batches.append((batch, batch_units))
                batch, batch_units = [], 0

            batch.append(index)
            batch_units += num_units

        if batch and not self.drop_last:
            bucket_batches.append((batch, batch_units))

        if self.shuffle and self.use_heap:
            np.random.shuffle(bucket_batches)

        if self.distributed:
            num_batches = torch.tensor(len(bucket_batches), device="cuda")
            dist.all_reduce(num_batches, op=dist.ReduceOp.MIN)
            num_batches = num_batches.item()
            if len(bucket_batches) > num_batches:
                bucket_batches = bucket_batches[:num_batches]

        self.bucket_batches = bucket_batches

    def set_epoch(self, epoch):
        self._epoch = epoch
        if self.distributed:
            self.sampler.set_epoch(epoch)


class LMDBDataModule(pl.LightningDataModule):
    """
    DataModule for LMDB datasets with RAM caching and dynamic batch padding.

    Args:
        train_lmdb_path: Path to train LMDB file
        val_lmdb_path: Path to validation LMDB file (optional)
        test_lmdb_path: Path to test LMDB file (optional)
        batch_size: Batch size configuration (DictConfig with train/val/test keys)
        num_workers: Number of workers configuration (DictConfig with train/val/test keys)
        preload_to_ram: Whether to preload all data to RAM
        use_pyg: Whether to use PyTorch Geometric Data format
    """

    def __init__(
        self,
        train_lmdb_path: str,
        val_lmdb_path: Optional[str] = None,
        test_lmdb_path: Optional[str] = None,
        batch_size: DictConfig = None,
        num_workers: DictConfig = None,
        preload_to_ram: bool = True,
        use_pyg: bool = False,
    ):
        super().__init__()
        self.train_lmdb_path = train_lmdb_path
        self.val_lmdb_path = val_lmdb_path
        self.test_lmdb_path = test_lmdb_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preload_to_ram = preload_to_ram
        self.use_pyg = use_pyg

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        DatasetClass = (
            LMDBCachedDatasetWithTransform if self.use_pyg else LMDBCachedDataset
        )

        if stage is None or stage == "fit":
            if self.train_lmdb_path:
                self.train_dataset = DatasetClass(
                    lmdb_path=self.train_lmdb_path,
                    preload_to_ram=self.preload_to_ram,
                )
            if self.val_lmdb_path:
                self.val_dataset = DatasetClass(
                    lmdb_path=self.val_lmdb_path,
                    preload_to_ram=self.preload_to_ram,
                )

        if stage is None or stage == "test":
            if self.test_lmdb_path:
                self.test_dataset = DatasetClass(
                    lmdb_path=self.test_lmdb_path,
                    preload_to_ram=self.preload_to_ram,
                )

    def _get_collate_fn(self):
        """Get the appropriate collate function."""
        if self.use_pyg:
            return collate_pyg_with_dynamic_padding
        else:
            return collate_fn_with_dynamic_padding

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size.train,
            shuffle=shuffle,
            num_workers=self.num_workers.train,
            collate_fn=self._get_collate_fn(),
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size.val,
            shuffle=False,
            num_workers=self.num_workers.val,
            collate_fn=self._get_collate_fn(),
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size.test,
            shuffle=False,
            num_workers=self.num_workers.test,
            collate_fn=self._get_collate_fn(),
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"train_lmdb_path={self.train_lmdb_path}, "
            f"val_lmdb_path={self.val_lmdb_path}, "
            f"test_lmdb_path={self.test_lmdb_path}, "
            f"batch_size={self.batch_size}, "
            f"preload_to_ram={self.preload_to_ram})"
        )
