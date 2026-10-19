#!/usr/bin/env python3
"""Independently verify the candidate-pool random-removal addendum."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_gtsrb_global_blend_candidate_random as candidate
import run_gtsrb_global_blend_postfilter as base
import run_tifs_native_cifar10 as common
import run_tifs_native_gtsrb as gtsrb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/srac_gtsrb_global_blend_candidate_random_development_v1.json"
DEFAULT_OUTPUT = ROOT / "results/srac_gtsrb_global_blend_candidate_random_development_v1"


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def verify(protocol_path: Path, output: Path, device_name: str) -> dict[str, Any]:
    errors: list[str] = []
    protocol = base.read_json(protocol_path)
    manifest = base.read_json(output / "manifest.json")
    summary = base.read_json(output / "summary.json")
    rows = base.read_csv(output / "per_seed_metrics.csv")
    source = (ROOT / protocol["source_result_relative_to_paper14"]).resolve()
    victim_protocol_path = (ROOT / protocol["victim_protocol_relative_to_paper14"]).resolve()
    victim_protocol = base.read_json(victim_protocol_path)
    seeds = [int(seed) for seed in protocol["seeds"]]
    methods = list(protocol["detectors"])
    conditions = list(protocol["conditions"])
    expected_keys = {
        (seed, f"candidate_random_matched_{method}")
        for seed in seeds
        for method in methods
    }
    row_by_key = {(int(row["seed"]), row["condition"]): row for row in rows}
    if len(rows) != len(expected_keys) or set(row_by_key) != expected_keys:
        errors.append("per-seed table is incomplete or duplicated")
    if manifest.get("status") != "complete":
        errors.append("manifest is not complete")
    hash_checks = {
        "protocol": (manifest.get("protocol_sha256"), common.sha256_file(protocol_path)),
        "runner": (
            manifest.get("runner_sha256"),
            common.sha256_file(ROOT / "src/run_gtsrb_global_blend_candidate_random.py"),
        ),
        "victim protocol": (
            common.sha256_file(victim_protocol_path),
            protocol["victim_protocol_sha256"],
        ),
        "source manifest": (
            common.sha256_file(source / "manifest.json"),
            protocol["source_manifest_sha256"],
        ),
    }
    for name, (actual, expected) in hash_checks.items():
        if actual != expected:
            errors.append(f"{name} hash mismatch")
    if protocol.get("runner_sha256") != hash_checks["runner"][1]:
        errors.append("frozen runner hash mismatch")

    data_root, _ = gtsrb.validate_dataset(victim_protocol)
    verification_log = gtsrb.RunLog(output / "verification_cache.log")
    try:
        cache, _ = gtsrb.build_resized_cache(data_root, verification_log)
    finally:
        verification_log.close()
    device = gtsrb.resolve_device(device_name)
    checked = 0
    for seed in seeds:
        data = gtsrb.build_seed_data(
            cache["train_images"], cache["train_labels"], victim_protocol, seed
        )
        gtsrb.validate_seed_data(data, cache["train_labels"], victim_protocol)
        detector_masks = base.frozen_removal_masks(
            base.source_rows(source, seed), data, tuple(methods)
        )
        for method_index, method in enumerate(methods):
            condition = f"candidate_random_matched_{method}"
            row = row_by_key.get((seed, condition))
            if row is None:
                continue
            remove = candidate.candidate_random_mask(
                len(data.train_images),
                data.candidate_positions,
                int(detector_masks[method].sum()),
                seed,
                method_index,
            )
            expected_clean = int(remove[: len(data.train_indices)].sum())
            expected_poison = int(remove[len(data.train_indices) :].sum())
            checks = {
                "rows_removed": int(remove.sum()),
                "clean_removed": expected_clean,
                "poison_removed": expected_poison,
                "poison_remaining": len(data.poison_base_indices) - expected_poison,
            }
            for name, expected in checks.items():
                if int(row[name]) != expected:
                    errors.append(f"seed {seed}/{condition}: {name} mismatch")
            expected_mask_hash = base.sha256_array(remove.astype(np.uint8))
            if row.get("remove_mask_sha256") != expected_mask_hash:
                errors.append(f"seed {seed}/{condition}: removal-mask hash mismatch")
            checkpoint = Path(row["checkpoint_path"])
            if not checkpoint.is_file() or common.sha256_file(checkpoint) != row["checkpoint_sha256"]:
                errors.append(f"seed {seed}/{condition}: checkpoint hash mismatch")
                continue
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("remove_mask_sha256") != expected_mask_hash:
                errors.append(f"seed {seed}/{condition}: checkpoint mask binding mismatch")
            if payload.get("state_sha256") != common.state_hash(payload["state_dict"]):
                errors.append(f"seed {seed}/{condition}: state hash mismatch")
            model = gtsrb.GTSRBTinyCNN().to(device)
            model.load_state_dict(payload["state_dict"])
            metrics = gtsrb.victim_metrics(model, cache, victim_protocol, 512, device)
            gtsrb.synchronize_device(device)
            checked += 1
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

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "protocol_sha256": common.sha256_file(protocol_path),
        "runner_sha256": common.sha256_file(
            ROOT / "src/run_gtsrb_global_blend_candidate_random.py"
        ),
        "verifier_sha256": common.sha256_file(Path(__file__).resolve()),
        "verified_seeds": seeds,
        "verified_conditions": conditions,
        "verified_rows": len(rows),
        "replayed_checkpoint_models": checked,
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
    receipt_path = (
        args.receipt.resolve()
        if args.receipt
        else args.output.resolve() / "verification_receipt.json"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
