#!/usr/bin/env python3
"""Independently verify post-filter GTSRB retraining artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_tifs_native_cifar10 as common
import run_tifs_native_gtsrb as gtsrb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/srac_gtsrb_global_blend_postfilter_confirmation_v1.json"
DEFAULT_OUTPUT = ROOT / "results/srac_gtsrb_global_blend_postfilter_confirmation_v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def source_mask(source: Path, seed: int, method: str, total: int) -> np.ndarray:
    rows = [
        row
        for row in read_csv(source / "per_sample" / f"seed_{seed}_scores.csv")
        if row["method"] == method
    ]
    positions = [int(row["candidate_position"]) for row in rows]
    if len(rows) != 2482 or len(set(positions)) != 2482:
        raise RuntimeError(f"seed {seed}/{method}: invalid source score table")
    mask = np.zeros(total, dtype=bool)
    for row, position in zip(rows, positions):
        mask[position] = bool(int(row["flag"]))
    return mask


def expected_mask(
    condition: str,
    method_index: dict[str, int],
    source: Path,
    seed: int,
    total: int,
    clean: int,
) -> np.ndarray:
    if condition.startswith("detector_"):
        return source_mask(source, seed, condition.removeprefix("detector_"), total)
    if condition.startswith("random_matched_"):
        method = condition.removeprefix("random_matched_")
        count = int(source_mask(source, seed, method, total).sum())
        order = np.random.default_rng(
            202610130000 + seed * 101 + method_index[method] * 1009
        ).permutation(total)
        mask = np.zeros(total, dtype=bool)
        mask[order[:count]] = True
        return mask
    if condition == "oracle_all_poison":
        mask = np.zeros(total, dtype=bool)
        mask[clean:] = True
        return mask
    if condition == "original_poisoned":
        return np.zeros(total, dtype=bool)
    raise RuntimeError(f"unknown condition: {condition}")


def bootstrap_interval(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def verify(protocol_path: Path, output: Path, device_name: str) -> dict[str, Any]:
    errors: list[str] = []
    protocol = read_json(protocol_path)
    manifest = read_json(output / "manifest.json")
    summary = read_json(output / "summary.json")
    rows = read_csv(output / "per_seed_metrics.csv")
    victim_protocol_path = (ROOT / protocol["victim_protocol_relative_to_paper14"]).resolve()
    victim_protocol = read_json(victim_protocol_path)
    source = (ROOT / protocol["source_result_relative_to_paper14"]).resolve()
    seeds = [int(seed) for seed in protocol["seeds"]]
    conditions = list(protocol["conditions"])
    methods = list(protocol["detectors"])
    method_index = {method: index for index, method in enumerate(methods)}
    expected_keys = {(seed, condition) for seed in seeds for condition in conditions}
    row_by_key = {(int(row["seed"]), row["condition"]): row for row in rows}
    if len(rows) != len(expected_keys) or set(row_by_key) != expected_keys:
        errors.append("per-seed table is incomplete or duplicated")
    if manifest.get("status") != "complete":
        errors.append("manifest is not complete")
    if manifest.get("protocol_sha256") != common.sha256_file(protocol_path):
        errors.append("manifest protocol hash mismatch")
    runner = ROOT / "src/run_gtsrb_global_blend_postfilter.py"
    if manifest.get("runner_sha256") != common.sha256_file(runner):
        errors.append("manifest runner hash mismatch")
    declared_runner_hash = protocol.get("runner_sha256")
    if declared_runner_hash is not None and declared_runner_hash != common.sha256_file(runner):
        errors.append("frozen runner hash mismatch")
    if common.sha256_file(victim_protocol_path) != protocol["victim_protocol_sha256"]:
        errors.append("source victim protocol hash mismatch")

    cache_path = (ROOT / victim_protocol["dataset"]["root_relative_to_paper14"]).resolve()
    cache_path = cache_path / "gtsrb_32x32_uint8_v1.npz"
    with np.load(cache_path) as archive:
        cache = {key: archive[key].copy() for key in archive.files}
    source_victims = {
        int(row["seed"]): row for row in read_csv(source / "victim_metrics.csv")
    }
    device = gtsrb.resolve_device(device_name)
    declared_counts = victim_protocol["attack"].get("expected_counts_per_seed", {})
    clean = int(declared_counts.get("clean", 23976))
    poison = int(declared_counts.get("poison", 1132))
    total = int(declared_counts.get("total", clean + poison))

    checked_models = 0
    for seed in seeds:
        for condition in conditions:
            row = row_by_key.get((seed, condition))
            if row is None:
                continue
            remove = expected_mask(condition, method_index, source, seed, total, clean)
            expected_clean_removed = int(remove[:clean].sum())
            expected_poison_removed = int(remove[clean:].sum())
            integer_checks = {
                "rows_removed": int(remove.sum()),
                "clean_removed": expected_clean_removed,
                "poison_removed": expected_poison_removed,
                "poison_remaining": poison - expected_poison_removed,
            }
            for name, expected in integer_checks.items():
                if int(row[name]) != expected:
                    errors.append(f"seed {seed}/{condition}: {name} mismatch")
            if condition == "original_poisoned":
                source_row = source_victims[seed]
                for name in ("balanced_clean_accuracy", "attack_success_rate"):
                    if not close(float(row[name]), float(source_row[name])):
                        errors.append(f"seed {seed}/{condition}: source {name} mismatch")
                if row["checkpoint_sha256"] != source_row["checkpoint_sha256"]:
                    errors.append(f"seed {seed}/{condition}: source checkpoint hash mismatch")
                continue

            if row.get("remove_mask_sha256") != sha256_array(remove.astype(np.uint8)):
                errors.append(f"seed {seed}/{condition}: removal-mask hash mismatch")
            checkpoint = Path(row["checkpoint_path"])
            if not checkpoint.is_file() or common.sha256_file(checkpoint) != row["checkpoint_sha256"]:
                errors.append(f"seed {seed}/{condition}: checkpoint hash mismatch")
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("remove_mask_sha256") != sha256_array(remove.astype(np.uint8)):
                errors.append(f"seed {seed}/{condition}: checkpoint removal binding mismatch")
            if payload.get("state_sha256") != common.state_hash(payload["state_dict"]):
                errors.append(f"seed {seed}/{condition}: state hash mismatch")
            model = gtsrb.GTSRBTinyCNN().to(device)
            model.load_state_dict(payload["state_dict"])
            metrics = gtsrb.victim_metrics(model, cache, victim_protocol, 512, device)
            gtsrb.synchronize_device(device)
            checked_models += 1
            for name in ("balanced_clean_accuracy", "attack_success_rate"):
                if not close(float(row[name]), float(metrics[name])):
                    errors.append(f"seed {seed}/{condition}: replayed {name} mismatch")

    for condition in conditions:
        block = [row_by_key[(seed, condition)] for seed in seeds if (seed, condition) in row_by_key]
        recorded = summary.get("conditions", {}).get(condition, {})
        checks = {
            "seed_count": len(block),
            "mean_clean_removed": np.mean([int(row["clean_removed"]) for row in block]),
            "mean_poison_removed": np.mean([int(row["poison_removed"]) for row in block]),
            "mean_balanced_clean_accuracy": np.mean(
                [float(row["balanced_clean_accuracy"]) for row in block]
            ),
            "mean_attack_success_rate": np.mean(
                [float(row["attack_success_rate"]) for row in block]
            ),
        }
        for name, expected in checks.items():
            if name not in recorded or not close(float(recorded[name]), float(expected)):
                errors.append(f"summary {condition}/{name} mismatch")

    bootstrap = protocol["analysis"]["bootstrap"]
    for index, method in enumerate(methods):
        detected = [row_by_key[(seed, f"detector_{method}")] for seed in seeds]
        random = [row_by_key[(seed, f"random_matched_{method}")] for seed in seeds]
        asr_delta = np.asarray(
            [
                float(det["attack_success_rate"]) - float(rnd["attack_success_rate"])
                for det, rnd in zip(detected, random)
            ]
        )
        bca_delta = np.asarray(
            [
                float(det["balanced_clean_accuracy"])
                - float(rnd["balanced_clean_accuracy"])
                for det, rnd in zip(detected, random)
            ]
        )
        recorded = summary.get("paired_contrasts", {}).get(method, {})
        scalar_checks = {
            "mean_asr_delta_detector_minus_random": asr_delta.mean(),
            "median_asr_delta_detector_minus_random": np.median(asr_delta),
            "detector_lower_asr_seed_count": int(np.sum(asr_delta < 0)),
            "mean_bca_delta_detector_minus_random": bca_delta.mean(),
        }
        for name, expected in scalar_checks.items():
            if name not in recorded or not close(float(recorded[name]), float(expected)):
                errors.append(f"contrast {method}/{name} mismatch")
        interval_checks = {
            "asr_delta_mean_bootstrap_95": bootstrap_interval(
                asr_delta, int(bootstrap["seed"]) + index, int(bootstrap["draws"])
            ),
            "bca_delta_mean_bootstrap_95": bootstrap_interval(
                bca_delta, int(bootstrap["seed"]) + 100 + index, int(bootstrap["draws"])
            ),
        }
        for name, expected in interval_checks.items():
            actual = recorded.get(name, [])
            if len(actual) != 2 or any(not close(a, b) for a, b in zip(actual, expected)):
                errors.append(f"contrast {method}/{name} mismatch")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "protocol_sha256": common.sha256_file(protocol_path),
        "runner_sha256": common.sha256_file(ROOT / "src/run_gtsrb_global_blend_postfilter.py"),
        "verified_seeds": seeds,
        "verified_conditions": conditions,
        "verified_rows": len(rows),
        "replayed_checkpoint_models": checked_models,
        "actual_device": device.type,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
    args = parser.parse_args()
    receipt = verify(args.protocol.resolve(), args.output.resolve(), args.device)
    receipt_path = args.receipt.resolve() if args.receipt else args.output.resolve() / "verification_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
