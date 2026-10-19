#!/usr/bin/env python3
"""Retrain GTSRB global-blend victims after candidate-pool random removal.

This is a fairness-control addendum to the frozen post-filter experiment.  For
each detector it removes the same number of records, sampled uniformly from the
detector's eligible candidate pool rather than from the full training set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_gtsrb_global_blend_postfilter as base
import run_tifs_native_cifar10 as common
import run_tifs_native_gtsrb as gtsrb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/srac_gtsrb_global_blend_candidate_random_development_v1.json"
DEFAULT_OUTPUT = ROOT / "results/srac_gtsrb_global_blend_candidate_random_development_v1"


def candidate_random_mask(
    total: int,
    candidate_positions: np.ndarray,
    count: int,
    seed: int,
    method_index: int,
) -> np.ndarray:
    if count > len(candidate_positions):
        raise ValueError("removal count exceeds the eligible candidate pool")
    order = np.random.default_rng(
        202610130500 + seed * 103 + method_index * 1013
    ).permutation(np.asarray(candidate_positions, dtype=np.int64))
    remove = np.zeros(total, dtype=bool)
    remove[order[:count]] = True
    return remove


def summarize(rows: list[dict[str, Any]], conditions: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in conditions:
        block = [row for row in rows if row["condition"] == condition]
        output[condition] = {
            "seeds": [int(row["seed"]) for row in block],
            "seed_count": len(block),
            "mean_clean_removed": float(np.mean([row["clean_removed"] for row in block])),
            "mean_poison_removed": float(np.mean([row["poison_removed"] for row in block])),
            "mean_balanced_clean_accuracy": float(
                np.mean([row["balanced_clean_accuracy"] for row in block])
            ),
            "mean_attack_success_rate": float(
                np.mean([row["attack_success_rate"] for row in block])
            ),
        } if block else {"seeds": [], "seed_count": 0, "status": "pending"}
    return {"conditions": output}


def self_test() -> None:
    candidates = np.asarray([2, 4, 5, 8, 9])
    first = candidate_random_mask(12, candidates, 3, 201, 0)
    second = candidate_random_mask(12, candidates, 3, 201, 0)
    assert np.array_equal(first, second)
    assert int(first.sum()) == 3
    assert set(np.flatnonzero(first)).issubset(set(candidates.tolist()))
    print("GTSRB_GLOBAL_BLEND_CANDIDATE_RANDOM_SELF_TEST_PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("mps", "cpu", "auto"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    protocol_path = args.protocol.resolve()
    protocol = base.read_json(protocol_path)
    if protocol.get("protocol_id") not in {
        "srac_gtsrb_global_blend_candidate_random_development_v1",
        "srac_gtsrb_global_blend_candidate_random_confirmation_v1",
    }:
        raise RuntimeError("unexpected candidate-random protocol")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (ROOT / protocol["source_result_relative_to_paper14"]).resolve()
    victim_protocol_path = (ROOT / protocol["victim_protocol_relative_to_paper14"]).resolve()
    victim_protocol = base.read_json(victim_protocol_path)
    if common.sha256_file(source / "manifest.json") != protocol["source_manifest_sha256"]:
        raise RuntimeError("source manifest hash mismatch")
    if common.sha256_file(victim_protocol_path) != protocol["victim_protocol_sha256"]:
        raise RuntimeError("victim protocol hash mismatch")
    if common.sha256_file(Path(__file__).resolve()) != protocol["runner_sha256"]:
        raise RuntimeError("runner hash mismatch")
    source_manifest = base.read_json(source / "manifest.json")
    seeds = [int(seed) for seed in protocol["seeds"]]
    if source_manifest.get("completed_seeds") != seeds:
        raise RuntimeError("source victim block does not match frozen seeds")
    methods = tuple(protocol["detectors"])
    conditions = [f"candidate_random_matched_{method}" for method in methods]
    if list(protocol["conditions"]) != conditions:
        raise RuntimeError("condition order or coverage mismatch")

    device = gtsrb.resolve_device(args.device, victim_protocol)
    data_root, dataset_receipt = gtsrb.validate_dataset(victim_protocol)
    log = gtsrb.RunLog(output / "run.log")
    cache, cache_receipt = gtsrb.build_resized_cache(data_root, log)
    rows = base.read_csv(output / "per_seed_metrics.csv")
    completed = {(int(row["seed"]), row["condition"]) for row in rows}
    started = time.time()
    try:
        for seed in seeds:
            data = gtsrb.build_seed_data(
                cache["train_images"], cache["train_labels"], victim_protocol, seed
            )
            gtsrb.validate_seed_data(data, cache["train_labels"], victim_protocol)
            detector_masks = base.frozen_removal_masks(
                base.source_rows(source, seed), data, methods
            )
            for method_index, method in enumerate(methods):
                condition = f"candidate_random_matched_{method}"
                key = (seed, condition)
                if key in completed:
                    log(f"seed={seed} condition={condition} skip existing result")
                    continue
                remove = candidate_random_mask(
                    len(data.train_images),
                    data.candidate_positions,
                    int(detector_masks[method].sum()),
                    seed,
                    method_index,
                )
                clean_removed = int(remove[: len(data.train_indices)].sum())
                poison_removed = int(remove[len(data.train_indices) :].sum())
                condition_started = time.time()
                log(
                    f"seed={seed} condition={condition} start removed="
                    f"{clean_removed}/{poison_removed}"
                )
                sanitized = base.sanitized_data(data, remove)
                model, training = gtsrb.train_model(
                    sanitized, victim_protocol, seed, args.threads, device, log
                )
                metrics = gtsrb.victim_metrics(
                    model, cache, victim_protocol, args.eval_batch_size, device
                )
                gtsrb.synchronize_device(device)
                checkpoint = output / "models" / condition / f"seed_{seed}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "candidate_random_protocol_sha256": common.sha256_file(protocol_path),
                    "victim_protocol_sha256": common.sha256_file(victim_protocol_path),
                    "seed": seed,
                    "condition": condition,
                    "detector": method,
                    "candidate_positions_sha256": base.sha256_array(
                        np.asarray(data.candidate_positions, dtype=np.int64)
                    ),
                    "remove_mask_sha256": base.sha256_array(remove.astype(np.uint8)),
                    "state_sha256": training["state_sha256"],
                    "state_dict": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                }
                temporary = checkpoint.with_name(checkpoint.name + ".tmp")
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "detector": method,
                        "rows_removed": int(remove.sum()),
                        "clean_removed": clean_removed,
                        "poison_removed": poison_removed,
                        "poison_remaining": len(data.poison_base_indices) - poison_removed,
                        "balanced_clean_accuracy": metrics["balanced_clean_accuracy"],
                        "attack_success_rate": metrics["attack_success_rate"],
                        "best_epoch": training["best_epoch"],
                        "checkpoint_path": str(checkpoint),
                        "checkpoint_sha256": common.sha256_file(checkpoint),
                        "state_sha256": training["state_sha256"],
                        "remove_mask_sha256": base.sha256_array(remove.astype(np.uint8)),
                        "training_seconds": time.time() - condition_started,
                    }
                )
                completed.add(key)
                base.write_csv(output / "per_seed_metrics.csv", rows)
                base.write_json(output / "summary.json", summarize(rows, conditions))
                log(
                    f"seed={seed} condition={condition} committed BCA="
                    f"{metrics['balanced_clean_accuracy']:.6f} ASR="
                    f"{metrics['attack_success_rate']:.6f}"
                )
        rows = sorted(rows, key=lambda row: (int(row["seed"]), conditions.index(row["condition"])))
        base.write_csv(output / "per_seed_metrics.csv", rows)
        summary = summarize(rows, conditions)
        base.write_json(output / "summary.json", summary)
        base.write_json(
            output / "manifest.json",
            {
                "status": "complete",
                "protocol_sha256": common.sha256_file(protocol_path),
                "runner_sha256": common.sha256_file(Path(__file__).resolve()),
                "victim_protocol_sha256": common.sha256_file(victim_protocol_path),
                "source_manifest_sha256": common.sha256_file(source / "manifest.json"),
                "dataset": dataset_receipt,
                "cache": cache_receipt,
                "seeds": seeds,
                "conditions": conditions,
                "row_count": len(rows),
                "expected_row_count": len(seeds) * len(conditions),
                "actual_device": device.type,
                "elapsed_seconds_this_invocation": time.time() - started,
                "summary": summary,
            },
        )
    finally:
        log.close()


if __name__ == "__main__":
    main()
