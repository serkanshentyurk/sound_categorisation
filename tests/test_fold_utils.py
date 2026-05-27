"""
Tests for utils/fold_utils.py.

Block-aware CV fold construction.
"""

import numpy as np
import pytest


class TestSplitFoldsByBlock:
    """split_folds_by_block takes per-row block_ids, returns (train_mask, test_mask) pairs."""

    def test_returns_list_of_pairs(self):
        from utils.fold_utils import split_folds_by_block
        block_ids = np.repeat([0, 1, 2, 3], 50)
        folds = split_folds_by_block(block_ids, n_folds=2)
        assert isinstance(folds, list)
        assert len(folds) == 2
        for fold in folds:
            assert len(fold) == 2  # (train_mask, test_mask)

    def test_train_test_complementary(self):
        """train_mask and test_mask are complementary boolean arrays."""
        from utils.fold_utils import split_folds_by_block
        block_ids = np.repeat([0, 1, 2, 3], 50)
        folds = split_folds_by_block(block_ids, n_folds=2)
        for train_mask, test_mask in folds:
            assert train_mask.dtype == bool
            assert test_mask.dtype == bool
            # mutually exclusive + total
            assert np.all(train_mask ^ test_mask)
            assert np.all(train_mask | test_mask)

    def test_test_indices_match_one_or_more_blocks(self):
        """Test fold corresponds to whole blocks (preserves block structure)."""
        from utils.fold_utils import split_folds_by_block
        block_ids = np.repeat([0, 1, 2, 3], 50)
        folds = split_folds_by_block(block_ids, n_folds=2)
        for train_mask, test_mask in folds:
            # Blocks appearing in test should not appear in train
            test_blocks = set(np.unique(block_ids[test_mask]))
            train_blocks = set(np.unique(block_ids[train_mask]))
            assert test_blocks.isdisjoint(train_blocks)


class TestMergeSmallestAdjacent:
    """merge_smallest_adjacent groups adjacent blocks into n_folds groups."""

    def test_returns_groups_of_labels(self):
        from utils.fold_utils import merge_smallest_adjacent
        groups = merge_smallest_adjacent(
            block_sizes=[100, 30, 80, 20, 60],
            labels=['A', 'B', 'C', 'D', 'E'],
            n_folds=2,
        )
        assert isinstance(groups, list)
        assert len(groups) == 2
        # Each group is a list of labels
        for g in groups:
            assert isinstance(g, list)

    def test_all_labels_preserved(self):
        """Every input label appears in exactly one group."""
        from utils.fold_utils import merge_smallest_adjacent
        labels = ['A', 'B', 'C', 'D', 'E']
        groups = merge_smallest_adjacent(
            block_sizes=[100, 30, 80, 20, 60],
            labels=labels,
            n_folds=3,
        )
        all_in_groups = [label for g in groups for label in g]
        assert sorted(all_in_groups) == sorted(labels)
