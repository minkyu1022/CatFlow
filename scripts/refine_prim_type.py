"""
Primitive slab element refinement module.

This module provides vocabulary-based filtering for element logits during sampling.
Elements not in the vocabulary (e.g., He, Li, Be) are excluded from argmax decoding.
"""

import torch
from torch import Tensor

# Allowed atomic numbers for primitive slab atoms
# These are the elements that appear in the training data
PRIM_SLAB_ELEMENT_VOCAB = [
    1, 5, 6, 7, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]


def create_vocab_mask(num_elements: int, vocab: list[int], device: torch.device) -> Tensor:

    mask = torch.zeros(num_elements, dtype=torch.bool, device=device)
    for atomic_num in vocab:
        if 1 <= atomic_num <= num_elements:
            # atomic_num is 1-indexed, mask index is 0-indexed
            mask[atomic_num - 1] = True
    return mask


def refine_prim_element_logits(
    logits: Tensor, 
    vocab: list[int] | None = None,
    vocab_mask: Tensor | None = None,
) -> Tensor:

    if vocab_mask is None:
        if vocab is None:
            vocab = PRIM_SLAB_ELEMENT_VOCAB
        vocab_mask = create_vocab_mask(logits.shape[-1], vocab, logits.device)
    
    # Apply mask: set non-vocab logits to -inf so they won't be selected by argmax
    masked_logits = logits.clone()
    masked_logits[..., ~vocab_mask] = float('-inf')
    
    return masked_logits
