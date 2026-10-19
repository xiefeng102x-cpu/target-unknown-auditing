#!/usr/bin/env python3
"""Replay CIFAR-10/VGG16 checkpoints, features, and detector score handoffs."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

import run_cifar10_vgg16_bn_development as development
import run_cifar10_vgg16_bn_srac_confirmation as runner
from cifar10_vgg16_bn import CIFAR10VGG16BN


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/cifar10_vgg16_bn_srac_confirmation_v1.json"
DEFAULT_RESULTS = ROOT / "results/cifar10_vgg16_bn_srac_confirmation_v1"


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    receipts = json.loads((args.results / "seed_receipts.json").read_text(encoding="utf-8"))
    receipt_by_seed = {int(row["seed"]): row for row in receipts}
    seeds = [int(seed) for seed in protocol["formal_seeds"]]
    if sorted(receipt_by_seed) != seeds:
        raise RuntimeError("receipt seeds differ from frozen confirmation seeds")

    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    train_images, train_labels, test_images, test_labels, _ = development.resolve_dataset(protocol)
    train_indices, validation_indices = development.fixed_split(
        train_labels,
        rows_per_class=int(protocol["dataset"]["validation_rows_per_class"]),
        seed=int(protocol["dataset"]["validation_split_seed"]),
    )
    target = int(protocol["attack"]["target_class"])
    target_validation = validation_indices[train_labels[validation_indices] == target]
    order = np.random.default_rng(int(protocol["reference"]["split_seed"])).permutation(
        target_validation
    )
    reference_ids = np.sort(order[:250])
    calibration_ids = np.sort(order[250:500])
    fresh_ids = np.flatnonzero(test_labels == target).astype(np.int64)

    errors: list[str] = []
    replayed_score_rows = 0
    for seed in seeds:
        receipt = receipt_by_seed[seed]
        poison_indices = development.poison_plan(
            train_labels,
            train_indices,
            target=target,
            count=int(protocol["attack"]["poison_count"]),
            seed=seed,
        )
        if runner.array_hash(poison_indices) != receipt["poison_indices_sha256"]:
            errors.append(f"seed {seed}: poison plan mismatch")
            continue

        checkpoint = Path(receipt["model_file"])
        model = CIFAR10VGG16BN().to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device)
        metrics = development.evaluate(
            model,
            test_images,
            test_labels,
            device,
            target,
            int(protocol["victim"]["batch_size"]),
        )
        for key, value in receipt["victim_metrics"].items():
            if not close(metrics[key], value):
                errors.append(f"seed {seed}: victim metric mismatch for {key}")

        assigned = train_labels.copy()
        assigned[poison_indices] = target
        candidate_ids = train_indices[assigned[train_indices] == target]
        poisoned = set(int(value) for value in poison_indices)
        _, candidate_features = runner.model_batch(
            model, train_images, candidate_ids, device, poisoned=poisoned
        )
        _, reference = runner.model_batch(model, train_images, reference_ids, device)
        _, calibration = runner.model_batch(model, train_images, calibration_ids, device)
        _, fresh_features = runner.model_batch(model, test_images, fresh_ids, device)

        with np.load(Path(receipt["feature_file"])) as archived:
            arrays = {
                "candidate_ids": candidate_ids,
                "candidate_features": candidate_features.astype(np.float32),
                "reference": reference.astype(np.float32),
                "calibration": calibration.astype(np.float32),
                "fresh_ids": fresh_ids,
                "fresh_features": fresh_features.astype(np.float32),
            }
            for name, observed in arrays.items():
                expected = archived[name]
                if expected.shape != observed.shape or not np.array_equal(expected, observed):
                    errors.append(f"seed {seed}: archived feature mismatch for {name}")

        with tempfile.TemporaryDirectory(prefix=f"cifar-srac-replay-{seed}-") as directory:
            score_path, truth_path, trusted_path, _ = runner.write_scores(
                Path(directory),
                protocol,
                seed,
                candidate_ids,
                candidate_features,
                reference,
                calibration,
                fresh_ids,
                fresh_features,
                poison_indices,
                train_labels,
            )
            expected_hashes = {
                score_path: receipt["blind_score_part_sha256"],
                truth_path: receipt["private_truth_part_sha256"],
                trusted_path: receipt["trusted_member_pool_part_sha256"],
            }
            for path, expected in expected_hashes.items():
                if runner.sha256(path) != expected:
                    errors.append(f"seed {seed}: replayed handoff hash mismatch for {path.name}")
            replayed_score_rows += sum(1 for _ in score_path.open("r", encoding="utf-8")) - 1

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "verified_seeds": seeds,
        "checkpoint_count": len(seeds),
        "replayed_score_rows": replayed_score_rows,
        "device": str(device),
        "compatibility_trusted_pool_used_by_srac": False,
        "statement": "The active SRAC endpoint reads only blind_scores.csv and private_truth.csv; the replayed trusted-member file is a compatibility artifact and is not an SRAC input.",
    }
    output = args.results / "checkpoint_score_replay_verification.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
