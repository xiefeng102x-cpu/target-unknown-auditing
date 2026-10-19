#!/usr/bin/env python3
"""Replay-verify the frozen GTSRB SRAC boundary-attack confirmation.

The verifier reconstructs every mixed-strength training population, reloads
each checkpoint, replays clean/attack inference and all three detector rules,
and compares the regenerated flags with the committed per-sample tables.  It
does not retrain or alter a decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_gtsrb_srac_boundary_attack as boundary  # noqa: E402
import run_tifs_native_gtsrb as native  # noqa: E402
import run_tifs_native_cifar10 as common  # noqa: E402


METHODS = ("cf_cmd_u", "target_centroid_distance", "crossfit_spectral")
DEFAULT_PROTOCOL = ROOT / "configs/srac_gtsrb_boundary_attack_confirmation_v1.json"
DEFAULT_OUTPUT = ROOT / "results/srac_gtsrb_boundary_attack_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left: float, right: float, tol: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify(protocol_path: Path, output: Path, device_name: str) -> dict[str, Any]:
    errors: list[str] = []
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary._validate_boundary_protocol(protocol)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    victims = read_csv(output / "victim_metrics.csv")
    detectors = read_csv(output / "detector_metrics.csv")
    seeds = [int(value) for value in protocol["seeds"]]
    protocol_hash = sha256(protocol_path)
    runner = ROOT / "src/run_gtsrb_srac_boundary_attack.py"
    if protocol.get("stage") != "confirmation":
        errors.append("protocol is not confirmation stage")
    if protocol.get("protocol_id") != "tifs_native_gtsrb_confirmation_v1":
        errors.append("unexpected native protocol id")
    if len(seeds) != 10 or len(set(seeds)) != 10:
        errors.append("confirmation must contain ten unique seeds")
    if sha256(output / "frozen_protocol.json") != protocol_hash:
        errors.append("archived protocol differs from invoked protocol")
    if protocol.get("runner", {}).get("sha256") != sha256(runner):
        errors.append("runner hash differs from frozen protocol")
    if manifest.get("protocol_sha256") != protocol_hash:
        errors.append("manifest protocol hash mismatch")
    if manifest.get("runner_sha256") != sha256(runner):
        errors.append("manifest runner hash mismatch")
    if manifest.get("completed_seeds") != seeds:
        errors.append("manifest seed block mismatch")
    if manifest.get("requested_device") != "mps":
        errors.append("formal training did not request MPS")
    if manifest.get("actual_device_by_seed") != {str(seed): "mps" for seed in seeds}:
        errors.append("formal training did not record MPS for every seed")

    victim_by_seed = {int(row["seed"]): row for row in victims}
    detector_by_key = {(int(row["seed"]), row["method"]): row for row in detectors}
    if sorted(victim_by_seed) != seeds or len(victims) != 10:
        errors.append("victim table does not equal the frozen seed block")
    if set(detector_by_key) != {(seed, method) for seed in seeds for method in METHODS}:
        errors.append("detector table is incomplete or duplicated")

    class QuietLog:
        def __call__(self, _message: str) -> None:
            return None

    data_root, _ = native.validate_dataset(protocol)
    cache, _ = native.build_resized_cache(data_root, QuietLog())
    device = native.resolve_device(device_name, protocol)
    alpha = float(protocol["detectors"]["primary"]["alpha"])
    target = int(protocol["attack"]["target_class"])
    replay_rows = 0
    eligible: list[int] = []
    subgroup: dict[str, dict[str, int]] = {
        method: {"strong_tp": 0, "strong_n": 0, "weak_tp": 0, "weak_n": 0}
        for method in METHODS
    }
    for seed in seeds:
        victim = victim_by_seed.get(seed)
        if victim is None:
            continue
        protocol["_active_seed"] = seed
        data = boundary.build_seed_data(cache["train_images"], cache["train_labels"], protocol, seed)
        try:
            boundary.validate_seed_data(data, cache["train_labels"], protocol)
        finally:
            protocol.pop("_active_seed", None)
        checkpoint = Path(victim["checkpoint_path"])
        if not checkpoint.is_file() or sha256(checkpoint) != victim["checkpoint_sha256"]:
            errors.append(f"seed {seed}: checkpoint hash mismatch")
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("protocol_sha256") != protocol_hash or int(payload.get("seed", -1)) != seed:
            errors.append(f"seed {seed}: checkpoint provenance mismatch")
        if common.state_hash(payload["state_dict"]) != victim["state_sha256"]:
            errors.append(f"seed {seed}: state hash mismatch")
        model = native.GTSRBTinyCNN().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        metrics = native.victim_metrics(model, cache, protocol, 512, device)
        for name in ("clean_accuracy", "balanced_clean_accuracy", "attack_success_rate"):
            if not close(metrics[name], float(victim[name])):
                errors.append(f"seed {seed}: {name} inference replay mismatch")
        if int(metrics["eligible"]) != int(victim["eligible"]):
            errors.append(f"seed {seed}: eligibility replay mismatch")
        if int(metrics["eligible"]):
            eligible.append(seed)

        _, candidate_features = native.predict_and_features(
            model, data.candidate_images, 512, device, True
        )
        reference, calibration, _, _ = native.trusted_features(
            model, cache, protocol, seed, 512, device
        )
        assert candidate_features is not None
        regenerated = {
            "cf_cmd_u": common.cf_cmd_u(reference, calibration, candidate_features, seed, alpha),
            "target_centroid_distance": common.centroid_distance(
                reference, calibration, candidate_features, alpha
            ),
            "crossfit_spectral": common.crossfit_spectral(
                reference, calibration, candidate_features, seed, alpha
            ),
        }
        sample_path = output / "per_sample" / f"seed_{seed}_scores.csv"
        receipt = json.loads((output / "receipts" / f"seed_{seed}.json").read_text())
        if sha256(sample_path) != receipt.get("per_sample_sha256"):
            errors.append(f"seed {seed}: per-sample hash mismatch")
            continue
        committed = read_csv(sample_path)
        for method in METHODS:
            block = [row for row in committed if row["method"] == method]
            if len(block) != len(data.candidate_truth):
                errors.append(f"seed {seed}/{method}: per-sample row count mismatch")
                continue
            if [row["record_id"] for row in block] != data.candidate_record_ids:
                errors.append(f"seed {seed}/{method}: record ordering mismatch")
            strengths = [row.get("poison_strength", "") for row in block]
            if strengths != [str(value) for value in data.candidate_strength]:
                errors.append(f"seed {seed}/{method}: strength metadata mismatch")
            flags = np.asarray([int(row["flag"]) for row in block], dtype=np.int8)
            expected_flags = np.asarray(regenerated[method]["flags"], dtype=np.int8)
            if not np.array_equal(flags, expected_flags):
                errors.append(f"seed {seed}/{method}: detector flags do not replay")
            truth = np.asarray(data.candidate_truth, dtype=bool)
            strong = np.asarray(data.candidate_strength) == "strong"
            weak = np.asarray(data.candidate_strength) == "weak"
            tp = int(np.sum(flags[truth] == 1))
            fp = int(np.sum(flags[~truth] == 1))
            expected_counts = {
                "tp": tp,
                "fn": int(truth.sum()) - tp,
                "fp": fp,
                "tn_target": int((~truth).sum()) - fp,
            }
            row = detector_by_key[(seed, method)]
            for name, value in expected_counts.items():
                if int(row[name]) != value:
                    errors.append(f"seed {seed}/{method}: {name} mismatch")
            subgroup[method]["strong_tp"] += int(flags[strong].sum())
            subgroup[method]["strong_n"] += int(strong.sum())
            subgroup[method]["weak_tp"] += int(flags[weak].sum())
            subgroup[method]["weak_n"] += int(weak.sum())
            replay_rows += len(block)
    native.synchronize_device(device)

    pooled = summary.get("methods", {})
    target_method = str(protocol["adaptive_attack_endpoint"]["target_detector"])
    target_tpr = pooled.get(target_method, {}).get("pooled_tpr")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "formal_evidence": not errors,
        "protocol_sha256": protocol_hash,
        "runner_sha256": sha256(runner),
        "verified_seeds": seeds,
        "eligible_seeds": eligible,
        "eligible_seed_count": len(eligible),
        "replayed_per_sample_rows": replay_rows,
        "target_detector": target_method,
        "target_detector_pooled_recall": target_tpr,
        "target_detector_evasion_observed": bool(
            target_tpr is not None
            and float(target_tpr) < float(protocol["srac_evaluation"]["recall_target"])
        ),
        "poison_strength_subgroups": subgroup,
        "verification_device": device.type,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("mps", "cpu", "auto"), default="mps")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = args.receipt or args.output / "verification_receipt.json"
    result = verify(args.protocol.resolve(), args.output.resolve(), args.device)
    write_json(receipt.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
