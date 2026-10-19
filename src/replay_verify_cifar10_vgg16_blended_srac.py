#!/usr/bin/env python3
"""Prospective semantic replay for the CIFAR/VGG Blended confirmation."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from cifar10_vgg16_bn import CIFAR10VGG16BN
from prospective_replay_contract import ReplayPolicy, compare_csv
import run_cifar10_vgg16_bn_development as development
import run_cifar10_vgg16_bn_blended_srac_confirmation as runner


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/cifar10_vgg16_bn_blended_srac_confirmation_v1.json"
DEFAULT_RESULTS = ROOT / "results/cifar10_vgg16_bn_blended_srac_confirmation_v1"


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-10)


def policy_from_protocol(protocol: dict) -> ReplayPolicy:
    frozen = protocol["prospective_replay_contract"]
    if frozen.get("contract_version") != "prospective-semantic-replay-v1":
        raise RuntimeError("unexpected semantic replay contract")
    if float(frozen.get("relative_tolerance", -1.0)) != 0.0:
        raise RuntimeError("relative tolerance must be zero")
    return ReplayPolicy(
        key_fields=tuple(frozen["key_fields"]),
        exact_fields=tuple(frozen["exact_fields"]),
        score_fields=tuple(frozen["score_fields"]),
        absolute_tolerance=float(frozen["absolute_tolerance"]),
    )


def run(protocol_path: Path, results: Path, device: torch.device) -> dict:
    runner.install()
    protocol, _ = runner.load_and_verify()
    if protocol_path.resolve() != runner.CONFIG.resolve():
        raise RuntimeError("protocol path differs from the frozen adapter path")
    policy = policy_from_protocol(protocol)
    receipts = json.loads((results / "seed_receipts.json").read_text(encoding="utf-8"))
    receipt_by_seed = {int(row["seed"]): row for row in receipts}
    seeds = [int(seed) for seed in protocol["formal_seeds"]]
    if sorted(receipt_by_seed) != seeds:
        raise RuntimeError("receipt seeds differ from frozen confirmation seeds")
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("the frozen replay requires MPS")

    train_images, train_labels, test_images, test_labels, _ = development.resolve_dataset(protocol)
    train_indices, validation_indices = development.fixed_split(
        train_labels,
        rows_per_class=int(protocol["dataset"]["validation_rows_per_class"]),
        seed=int(protocol["dataset"]["validation_split_seed"]),
    )
    target = int(protocol["attack"]["target_class"])
    target_validation = validation_indices[train_labels[validation_indices] == target]
    order = np.random.default_rng(int(protocol["reference"]["split_seed"])).permutation(target_validation)
    reference_ids = np.sort(order[:250])
    calibration_ids = np.sort(order[250:500])
    fresh_ids = np.flatnonzero(test_labels == target).astype(np.int64)

    errors: list[str] = []
    per_seed: list[dict] = []
    for seed in seeds:
        receipt = receipt_by_seed[seed]
        poison_indices = development.poison_plan(
            train_labels, train_indices, target=target,
            count=int(protocol["attack"]["poison_count"]), seed=seed,
        )
        if runner.engine.array_hash(poison_indices) != receipt["poison_indices_sha256"]:
            errors.append(f"seed {seed}: poison plan mismatch")
            continue

        model = CIFAR10VGG16BN().to(device)
        model.load_state_dict(
            torch.load(Path(receipt["model_file"]), map_location="cpu", weights_only=True),
            strict=True,
        )
        model.to(device)
        metrics = development.evaluate(
            model, test_images, test_labels, device, target,
            int(protocol["victim"]["batch_size"]),
        )
        for key, value in receipt["victim_metrics"].items():
            if not close(metrics[key], value):
                errors.append(f"seed {seed}: victim metric mismatch for {key}")

        assigned = train_labels.copy()
        assigned[poison_indices] = target
        candidate_ids = train_indices[assigned[train_indices] == target]
        poisoned = set(int(value) for value in poison_indices)
        _, candidate_features = runner.engine.model_batch(
            model, train_images, candidate_ids, device, poisoned=poisoned
        )
        _, reference = runner.engine.model_batch(model, train_images, reference_ids, device)
        _, calibration = runner.engine.model_batch(model, train_images, calibration_ids, device)
        _, fresh_features = runner.engine.model_batch(model, test_images, fresh_ids, device)

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

            with tempfile.TemporaryDirectory(prefix=f"cifar-blended-replay-{seed}-") as directory:
                score_path, truth_path, trusted_path, _ = runner.write_scores(
                    Path(directory), protocol, seed,
                    archived["candidate_ids"], archived["candidate_features"],
                    archived["reference"], archived["calibration"],
                    archived["fresh_ids"], archived["fresh_features"],
                    poison_indices, train_labels,
                )
                score_report = compare_csv(
                    results / "blind_scores_parts" / f"seed_{seed}.csv",
                    score_path,
                    policy,
                )
                if score_report["status"] != "PASS":
                    errors.append(f"seed {seed}: semantic score replay failed")
                truth_exact = runner.engine.sha256(truth_path) == receipt["private_truth_part_sha256"]
                trusted_exact = runner.engine.sha256(trusted_path) == receipt["trusted_member_pool_part_sha256"]
                if not truth_exact:
                    errors.append(f"seed {seed}: truth hash mismatch")
                if not trusted_exact:
                    errors.append(f"seed {seed}: compatibility-pool hash mismatch")
                per_seed.append({
                    "seed": seed,
                    "score_replay": score_report,
                    "truth_hash_exact": truth_exact,
                    "compatibility_pool_hash_exact": trusted_exact,
                })

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "verified_seeds": seeds,
        "checkpoint_count": len(seeds),
        "device": str(device),
        "semantic_contract": policy.as_dict(),
        "per_seed": per_seed,
        "raw_score_hashes_are_diagnostic": True,
        "exact_endpoint_invariants": [
            "row keys", "conventional flags", "private truth", "protected-draw inputs"
        ],
    }
    output = results / "checkpoint_score_semantic_replay_verification.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", choices=("mps",), default="mps")
    args = parser.parse_args()
    report = run(args.protocol, args.results, torch.device(args.device))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
