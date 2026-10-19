#!/usr/bin/env python3
"""Build a full-population SRAC input from the pre-frozen GTSRB block.

The original GTSRB detector files contain the label-conditional score domain:
target-class clean members and all added poison records.  The frozen protocol
states that every other clean training member is unflagged by this screening
rule.  This adapter reconstructs the complete training population, validates
that statement against the original split RNG and candidate positions, and
writes the same blind_scores/private_truth interface used by the SRAC replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
SOURCE = PAPER14 / "results" / "tifs_native_cifar_gtsrb_v1" / "gtsrb"
PROTOCOL = PAPER14 / "configs" / "tifs_native_gtsrb_confirmation_v1_1.json"
CACHE = PAPER14 / "data" / "gtsrb_official" / "gtsrb_32x32_uint8_v1.npz"
OUTPUT = ROOT / "derived_inputs" / "gtsrb_tinycnn_prefrozen_v1"
METHODS = ("cf_cmd_u", "crossfit_spectral", "target_centroid_distance")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_indices(labels: np.ndarray, seed: int, fraction: float) -> np.ndarray:
    parts: list[np.ndarray] = []
    for label in range(43):
        indices = np.flatnonzero(labels == label)
        order = np.random.default_rng(2026100600 + seed + label * 1009).permutation(indices)
        parts.append(order[: int(np.floor(fraction * len(order)))])
    return np.concatenate(parts)


def build_seed(
    seed: int,
    labels: np.ndarray,
    protocol: dict[str, object],
    source_root: Path = SOURCE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fraction = float(protocol["split"]["train_fraction_per_class"])
    target = int(protocol["attack"]["target_class"])
    clean_indices = training_indices(labels, seed, fraction)
    clean_ids = [f"train_clean_{int(index)}" for index in clean_indices]
    score_path = source_root / "per_sample" / f"seed_{seed}_scores.csv"
    source = pd.read_csv(score_path)
    if set(source["method"].unique()) != set(METHODS):
        raise RuntimeError(f"unexpected detector set in {score_path}")

    canonical = source.loc[source["method"].eq(METHODS[0])].sort_values("candidate_position")
    if canonical["record_id"].duplicated().any():
        raise RuntimeError(f"duplicate candidate record in {score_path}")
    poison = canonical.loc[canonical["poison_truth"].eq(1)].copy()
    poison_ids = poison["record_id"].tolist()
    sample_ids = clean_ids + poison_ids
    n_clean = len(clean_ids)
    n_poison = len(poison_ids)
    n_total = len(sample_ids)
    declared = protocol["attack"].get("expected_counts_per_seed", {})
    expected_clean = int(declared.get("clean", 23976))
    expected_poison = int(declared.get("poison", n_poison))
    expected_total = int(declared.get("total", expected_clean + expected_poison))
    if (n_clean, n_poison, n_total) != (expected_clean, expected_poison, expected_total):
        raise RuntimeError(f"population count mismatch for seed {seed}")
    if len(set(sample_ids)) != n_total:
        raise RuntimeError(f"full-population record IDs are not unique for seed {seed}")

    id_by_position = dict(enumerate(sample_ids))
    target_clean = canonical.loc[canonical["poison_truth"].eq(0)]
    expected_target = set(clean_indices[labels[clean_indices] == target].tolist())
    observed_target = set(target_clean["base_train_index"].astype(int).tolist())
    if expected_target != observed_target or len(expected_target) != 1350:
        raise RuntimeError(f"target-clean reconstruction mismatch for seed {seed}")
    for row in canonical.itertuples(index=False):
        position = int(row.candidate_position)
        if id_by_position.get(position) != row.record_id:
            raise RuntimeError(f"candidate position mismatch for seed {seed}: {position}")

    victim = f"gtsrb_patch_tinycnn_seed{seed}"
    truth_values = np.zeros(n_total, dtype=np.int8)
    truth_values[n_clean:] = 1
    truth = pd.DataFrame(
        {
            "victim_id": victim,
            "sample_id": sample_ids,
            "role": "train",
            "is_poison": truth_values,
        }
    )

    score_frames: list[pd.DataFrame] = []
    for method in METHODS:
        table = source.loc[source["method"].eq(method)].sort_values("candidate_position")
        if table["candidate_position"].duplicated().any() or len(table) != 1350 + n_poison:
            raise RuntimeError(f"invalid candidate coverage for seed {seed}/{method}")
        flags = np.zeros(n_total, dtype=np.int8)
        positions = table["candidate_position"].to_numpy(dtype=np.int64)
        flags[positions] = table["flag"].to_numpy(dtype=np.int8)
        score_frames.append(
            pd.DataFrame(
                {
                    "victim_id": victim,
                    "sample_id": sample_ids,
                    "role": "train",
                    "fold": 0,
                    "detector": method,
                    "conventional_flag": flags,
                }
            )
        )
    receipt = {
        "seed": seed,
        "source_scores": str(score_path.relative_to(PAPER14)),
        "source_scores_sha256": sha256(score_path),
        "clean_rows": n_clean,
        "poison_rows": n_poison,
        "population_rows": n_total,
        "prevalence": n_poison / n_total,
        "detectors": list(METHODS),
    }
    return pd.concat(score_frames, ignore_index=True), truth, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source_root = args.source.resolve()
    protocol_path = args.protocol.resolve()
    cache_path = args.cache.resolve()
    output = args.output.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") not in {
        "tifs_native_gtsrb_confirmation_v1",
        "tifs_native_gtsrb_confirmation_v1_1",
    }:
        raise RuntimeError("unexpected GTSRB source protocol")
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["seeds"]]
    if manifest.get("completed_seeds") != seeds:
        raise RuntimeError("GTSRB confirmation is incomplete")
    with np.load(cache_path) as cache:
        labels = cache["train_labels"].copy()
    score_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    seed_receipts: list[dict[str, object]] = []
    for seed in seeds:
        scores, truth, receipt = build_seed(seed, labels, protocol, source_root)
        score_frames.append(scores)
        truth_frames.append(truth)
        seed_receipts.append(receipt)
    scores = pd.concat(score_frames, ignore_index=True)
    truth = pd.concat(truth_frames, ignore_index=True)
    population_per_seed = int(seed_receipts[0]["population_rows"])
    expected_scores = len(seeds) * population_per_seed * len(METHODS)
    expected_truth = len(seeds) * population_per_seed
    if len(scores) != expected_scores or len(truth) != expected_truth:
        raise RuntimeError("derived row-count invariant failed")
    output.mkdir(parents=True, exist_ok=True)
    score_output = output / "blind_scores.csv"
    truth_output = output / "private_truth.csv"
    scores.to_csv(score_output, index=False)
    truth.to_csv(truth_output, index=False)
    receipt = {
        "status": "PASS",
        "scope": "derived full-population SRAC input from pre-frozen GTSRB confirmation",
        "protocol": str(protocol_path.relative_to(PAPER14)),
        "protocol_sha256": sha256(protocol_path),
        "source_manifest": str((source_root / "manifest.json").relative_to(PAPER14)),
        "source_manifest_sha256": sha256(source_root / "manifest.json"),
        "cache_sha256": sha256(cache_path),
        "seeds": seeds,
        "methods": list(METHODS),
        "forced_unflagged_rule": "all non-target clean training members",
        "blind_score_rows": len(scores),
        "private_truth_rows": len(truth),
        "blind_scores_sha256": sha256(score_output),
        "private_truth_sha256": sha256(truth_output),
        "seed_receipts": seed_receipts,
    }
    receipt_path = output / "adapter_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "seed_receipts"}, indent=2))


if __name__ == "__main__":
    main()
