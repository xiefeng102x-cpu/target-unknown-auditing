#!/usr/bin/env python3
"""Run a pre-draw SRAC-aware boundary-seeking GTSRB attack.

The attack is fixed before victim training and before any protected SRAC draw.
It mixes a declared fraction of strong global-blend poison copies with weak
copies.  The development stage may select that fraction to place detector
recall near the frozen SRAC recall target.  Confirmation seeds must then use
the selected fraction without retuning.

This wrapper deliberately reuses the audited native-GTSRB training and
detector implementation.  It changes only poison strength assignment and
adds strength metadata to the per-sample table.  The previously frozen runner
is never edited.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import run_tifs_native_gtsrb as native  # noqa: E402


_ORIGINAL_READ_JSON = native.read_json
_ORIGINAL_BUILD_SEED_DATA = native.build_seed_data
_ORIGINAL_VALIDATE_SEED_DATA = native.validate_seed_data
_ORIGINAL_DETECTOR_SAMPLE_ROWS = native.detector_sample_rows


def _validate_boundary_protocol(protocol: dict[str, Any]) -> None:
    study_id = str(protocol.get("study_id", ""))
    if not study_id.startswith("srac_gtsrb_boundary_attack_"):
        raise RuntimeError("unexpected SRAC boundary-attack study_id")
    attack = protocol.get("attack", {})
    strong_fraction = float(attack["strong_fraction"])
    strong_alpha = float(attack["patch_alpha"])
    weak_alpha = float(attack["weak_alpha"])
    if not 0.0 <= strong_fraction <= 1.0:
        raise RuntimeError("strong_fraction must be in [0,1]")
    if not 0.0 <= weak_alpha < strong_alpha <= 1.0:
        raise RuntimeError("require 0 <= weak_alpha < patch_alpha <= 1")
    if int(attack.get("strength_assignment_rng_base", -1)) <= 0:
        raise RuntimeError("strength_assignment_rng_base must be a positive integer")
    endpoint = protocol.get("srac_attack_target", protocol.get("srac_evaluation", {}))
    if float(endpoint.get("recall_target", -1.0)) != 0.8:
        raise RuntimeError("boundary attack must declare the frozen recall target 0.8")
    if endpoint.get("protected_draw_unknown_to_attacker") is not True:
        raise RuntimeError("protected draw secrecy must be explicit")


def read_json(path: Path) -> Any:
    protocol = _ORIGINAL_READ_JSON(path)
    _validate_boundary_protocol(protocol)
    return protocol


def _strength_mask(n_poison: int, protocol: dict[str, Any], seed: int) -> np.ndarray:
    attack = protocol["attack"]
    strong_count = int(round(float(attack["strong_fraction"]) * n_poison))
    rng = np.random.default_rng(int(attack["strength_assignment_rng_base"]) + seed)
    strong = np.zeros(n_poison, dtype=bool)
    if strong_count:
        strong[rng.permutation(n_poison)[:strong_count]] = True
    return strong


def build_seed_data(
    images: np.ndarray, labels: np.ndarray, protocol: dict[str, Any], seed: int
) -> native.SeedData:
    data = _ORIGINAL_BUILD_SEED_DATA(images, labels, protocol, seed)
    n_poison = len(data.poison_base_indices)
    strong = _strength_mask(n_poison, protocol, seed)
    weak = ~strong
    weak_protocol = copy.deepcopy(protocol)
    weak_protocol["attack"]["patch_alpha"] = float(protocol["attack"]["weak_alpha"])
    weak_images = native.patch_uint8(images[data.poison_base_indices[weak]], weak_protocol)
    n_clean = len(data.train_indices)
    n_target_clean = int((~data.candidate_truth).sum())
    data.train_images[n_clean + np.flatnonzero(weak)] = weak_images
    data.candidate_images[n_target_clean + np.flatnonzero(weak)] = weak_images
    candidate_strength = np.full(len(data.candidate_truth), "clean", dtype=object)
    poison_strength = np.where(strong, "strong", "weak")
    candidate_strength[n_target_clean:] = poison_strength
    data.poison_strong_mask = strong
    data.candidate_strength = candidate_strength
    return data


def validate_seed_data(
    data: native.SeedData,
    labels: np.ndarray,
    protocol: dict[str, Any],
) -> None:
    _ORIGINAL_VALIDATE_SEED_DATA(data, labels, protocol)
    n_poison = len(data.poison_base_indices)
    expected = _strength_mask(n_poison, protocol, int(protocol.get("_active_seed", -1)))
    observed = np.asarray(data.poison_strong_mask, dtype=bool)
    if len(observed) != n_poison:
        raise RuntimeError("poison strength metadata length mismatch")
    if int(protocol.get("_active_seed", -1)) >= 0 and not np.array_equal(observed, expected):
        raise RuntimeError("poison strength assignment is not reproducible")
    expected_strong = int(round(float(protocol["attack"]["strong_fraction"]) * n_poison))
    if int(observed.sum()) != expected_strong:
        raise RuntimeError("strong poison count mismatch")
    if len(data.candidate_strength) != len(data.candidate_truth):
        raise RuntimeError("candidate strength metadata length mismatch")


def detector_sample_rows(
    seed: int,
    method: str,
    data: native.SeedData,
    result: dict[str, Any],
    target: int,
) -> list[dict[str, Any]]:
    rows = _ORIGINAL_DETECTOR_SAMPLE_ROWS(seed, method, data, result, target)
    strengths = np.asarray(data.candidate_strength, dtype=object)
    for row, strength in zip(rows, strengths):
        row["poison_strength"] = str(strength)
    return rows


def _build_seed_data_with_active_seed(
    images: np.ndarray, labels: np.ndarray, protocol: dict[str, Any], seed: int
) -> native.SeedData:
    # The active seed is recorded only in the in-memory copy so validation can
    # independently reconstruct the assignment.  The frozen JSON is untouched.
    protocol["_active_seed"] = int(seed)
    try:
        return build_seed_data(images, labels, protocol, seed)
    finally:
        # Keep it through validate_seed_data, which immediately follows build in
        # the native runner; removal happens in the validation wrapper below.
        pass


def _validate_and_clear_seed(
    data: native.SeedData, labels: np.ndarray, protocol: dict[str, Any]
) -> None:
    try:
        validate_seed_data(data, labels, protocol)
    finally:
        protocol.pop("_active_seed", None)


def main() -> None:
    native.read_json = read_json
    native.build_seed_data = _build_seed_data_with_active_seed
    native.validate_seed_data = _validate_and_clear_seed
    native.detector_sample_rows = detector_sample_rows
    # The native manifest and MPS receipts must bind this wrapper, which is the
    # actual executable defining the adaptive poison construction.
    native.__file__ = str(HERE)
    native.main()


if __name__ == "__main__":
    main()
