"""
Block-Level CV Fold Splitting

Shared by analysis/grid_search.py and inference/comparison.py.
Splits trial-level data into folds by session (block), preserving
sequential structure within each fold.

Usage:
    from utils.fold_utils import split_folds_by_block

    folds = split_folds_by_block(block_ids, n_folds=2)
    for train_mask, test_mask in folds:
        train_stim = stimuli[train_mask]
        test_stim = stimuli[test_mask]
"""

import numpy as np
from typing import List, Tuple


def merge_smallest_adjacent(
    block_sizes: list,
    labels: list,
    n_folds: int,
) -> List[List]:
    """
    Merge adjacent blocks until we have exactly n_folds groups.

    Greedily merges the smallest block with its smallest neighbour.
    Preserves temporal contiguity (only adjacent blocks are merged).

    Args:
        block_sizes: List of trial counts per block
        labels: List of block identifiers (same length as block_sizes)
        n_folds: Target number of groups

    Returns:
        List of lists, each containing the block labels in that fold
    """
    labeled_blocks = [[label] for label in labels]
    sizes = list(block_sizes)

    while len(sizes) > n_folds:
        min_idx = min(range(len(sizes)), key=lambda i: sizes[i])

        if min_idx == 0:
            adj = 1
        elif min_idx == len(sizes) - 1:
            adj = len(sizes) - 2
        else:
            adj = (min_idx - 1
                   if sizes[min_idx - 1] < sizes[min_idx + 1]
                   else min_idx + 1)

        if min_idx < adj:
            labeled_blocks[min_idx].extend(labeled_blocks.pop(adj))
            sizes[min_idx] += sizes.pop(adj)
        else:
            labeled_blocks[adj].extend(labeled_blocks.pop(min_idx))
            sizes[adj] += sizes.pop(min_idx)

    return labeled_blocks


def split_folds_by_block(
    block_ids: np.ndarray,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split trial indices into CV folds by block grouping.

    Preserves sequential trial order within each fold.
    Uses merge_smallest_adjacent to balance fold sizes.

    Args:
        block_ids: Per-trial block identifier array
        n_folds: Number of folds

    Returns:
        List of (train_mask, test_mask) boolean arrays

    Raises:
        ValueError: If fewer than 2 blocks available
    """
    unique_blocks = np.unique(block_ids)
    block_sizes = [np.sum(block_ids == b) for b in unique_blocks]

    n_actual_folds = min(n_folds, len(unique_blocks))
    if n_actual_folds < 2:
        raise ValueError(
            f"Need at least 2 blocks for CV, got {len(unique_blocks)}"
        )

    fold_groups = merge_smallest_adjacent(
        block_sizes, list(unique_blocks), n_actual_folds,
    )

    folds = []
    for test_fold_idx in range(n_actual_folds):
        test_blocks = set(fold_groups[test_fold_idx])
        test_mask = np.isin(block_ids, list(test_blocks))
        train_mask = ~test_mask
        folds.append((train_mask, test_mask))

    return folds
