from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analysis/run_cifar10_semantic_overlap_confirmation.py"
SPEC = importlib.util.spec_from_file_location("cifar10_semantic_overlap_confirmation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_membership_matrix_allows_nested_and_crossing_groups() -> None:
    labels = np.array([0, 1, 2, 3, 4])
    groups = [
        {"name": "broad", "class_ids": [0, 1, 2, 3]},
        {"name": "nested", "class_ids": [1, 2]},
        {"name": "crossing", "class_ids": [2, 3, 4]},
    ]
    matrix, names = MODULE.membership_matrix(labels, groups)
    assert names == ["broad", "nested", "crossing"]
    assert matrix.astype(int).tolist() == [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
    ]


def test_membership_matrix_rejects_duplicate_names() -> None:
    groups = [
        {"name": "same", "class_ids": [0]},
        {"name": "same", "class_ids": [1]},
    ]
    with pytest.raises(ValueError, match="unique"):
        MODULE.membership_matrix(np.array([0, 1]), groups)


def test_membership_matrix_rejects_out_of_range_cifar_class() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        MODULE.membership_matrix(
            np.array([0, 1]), [{"name": "bad", "class_ids": [0, 10]}]
        )
