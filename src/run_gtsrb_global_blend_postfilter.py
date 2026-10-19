#!/usr/bin/env python3
"""Retrain GTSRB global-blend victims after frozen detector removal.

The script compares each frozen high-recall detector with a size-matched uniform
removal rule.  Oracle removal of every poison is the clean-training reference.
The original suspect checkpoint is reported without retraining.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_tifs_native_cifar10 as common
import run_tifs_native_gtsrb as gtsrb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/srac_gtsrb_global_blend_postfilter_development_v1.json"
DEFAULT_OUTPUT = ROOT / "results/srac_gtsrb_global_blend_postfilter_development_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def source_rows(source: Path, seed: int) -> list[dict[str, str]]:
    path = source / "per_sample" / f"seed_{seed}_scores.csv"
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"missing frozen score table: {path}")
    return rows


def frozen_removal_masks(
    rows: list[dict[str, str]],
    data: gtsrb.SeedData,
    methods: tuple[str, ...],
) -> dict[str, np.ndarray]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    expected_ids = {
        int(position): record_id
        for position, record_id in zip(data.candidate_positions, data.candidate_record_ids)
    }
    masks: dict[str, np.ndarray] = {}
    for method in methods:
        block = by_method.get(method, [])
        if len(block) != len(data.candidate_positions):
            raise RuntimeError(f"{method}: incomplete candidate table")
        positions = [int(row["candidate_position"]) for row in block]
        if len(set(positions)) != len(positions):
            raise RuntimeError(f"{method}: duplicate candidate position")
        remove = np.zeros(len(data.train_images), dtype=bool)
        for row, position in zip(block, positions):
            if expected_ids.get(position) != row["record_id"]:
                raise RuntimeError(f"{method}: record-position mismatch")
            expected_truth = int(position >= len(data.train_indices))
            if int(row["poison_truth"]) != expected_truth:
                raise RuntimeError(f"{method}: truth-position mismatch")
            remove[position] = bool(int(row["flag"]))
        masks[method] = remove
    return masks


def matched_random_mask(total: int, count: int, seed: int, method_index: int) -> np.ndarray:
    remove = np.zeros(total, dtype=bool)
    order = np.random.default_rng(202610130000 + seed * 101 + method_index * 1009).permutation(total)
    remove[order[:count]] = True
    return remove


def sanitized_data(data: gtsrb.SeedData, remove: np.ndarray) -> gtsrb.SeedData:
    n_clean = len(data.train_indices)
    clean_keep = ~remove[:n_clean]
    poison_keep = ~remove[n_clean:]
    return dataclasses.replace(
        data,
        train_indices=data.train_indices[clean_keep],
        poison_base_indices=data.poison_base_indices[poison_keep],
        poison_sources=data.poison_sources[poison_keep],
        train_images=data.train_images[~remove],
        train_labels=data.train_labels[~remove],
    )


def bootstrap_interval(values: np.ndarray, seed: int, draws: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    conditions = list(protocol["conditions"])
    summaries: dict[str, Any] = {}
    for condition in conditions:
        block = [row for row in rows if row["condition"] == condition]
        if not block:
            summaries[condition] = {"seeds": [], "seed_count": 0, "status": "pending"}
            continue
        summaries[condition] = {
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
        }
    contrasts: dict[str, Any] = {}
    bootstrap = protocol["analysis"]["bootstrap"]
    for method_index, method in enumerate(protocol["detectors"]):
        detected_name = f"detector_{method}"
        random_name = f"random_matched_{method}"
        detected = {int(row["seed"]): row for row in rows if row["condition"] == detected_name}
        random = {int(row["seed"]): row for row in rows if row["condition"] == random_name}
        seeds = sorted(set(detected) & set(random))
        if not seeds:
            contrasts[method] = {"seeds": [], "status": "pending"}
            continue
        asr_delta = np.asarray(
            [detected[seed]["attack_success_rate"] - random[seed]["attack_success_rate"] for seed in seeds],
            dtype=np.float64,
        )
        bca_delta = np.asarray(
            [
                detected[seed]["balanced_clean_accuracy"]
                - random[seed]["balanced_clean_accuracy"]
                for seed in seeds
            ],
            dtype=np.float64,
        )
        contrasts[method] = {
            "seeds": seeds,
            "mean_asr_delta_detector_minus_random": float(asr_delta.mean()),
            "median_asr_delta_detector_minus_random": float(np.median(asr_delta)),
            "asr_delta_mean_bootstrap_95": bootstrap_interval(
                asr_delta,
                int(bootstrap["seed"]) + method_index,
                int(bootstrap["draws"]),
            ),
            "detector_lower_asr_seed_count": int(np.sum(asr_delta < 0)),
            "mean_bca_delta_detector_minus_random": float(bca_delta.mean()),
            "bca_delta_mean_bootstrap_95": bootstrap_interval(
                bca_delta,
                int(bootstrap["seed"]) + 100 + method_index,
                int(bootstrap["draws"]),
            ),
        }
    return {"conditions": summaries, "paired_contrasts": contrasts}


def self_test() -> None:
    total = 100
    mask = matched_random_mask(total, 17, 3, 1)
    assert len(mask) == total and int(mask.sum()) == 17
    values = np.asarray([-0.4, -0.2, -0.1])
    interval = bootstrap_interval(values, 7, 1000)
    assert interval[1] < 0
    print("GTSRB_GLOBAL_BLEND_POSTFILTER_SELF_TEST_PASS")


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
    protocol = read_json(protocol_path)
    if protocol.get("protocol_id") not in {
        "srac_gtsrb_global_blend_postfilter_development_v1",
        "srac_gtsrb_global_blend_postfilter_confirmation_v1",
    }:
        raise RuntimeError("unexpected post-filter protocol")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (ROOT / protocol["source_result_relative_to_paper14"]).resolve()
    victim_protocol_path = (ROOT / protocol["victim_protocol_relative_to_paper14"]).resolve()
    victim_protocol = read_json(victim_protocol_path)
    if common.sha256_file(victim_protocol_path) != protocol["victim_protocol_sha256"]:
        raise RuntimeError("victim protocol hash mismatch")
    manifest = read_json(source / "manifest.json")
    seeds = [int(seed) for seed in protocol["seeds"]]
    if manifest.get("completed_seeds") != seeds:
        raise RuntimeError("source victim block does not match frozen seeds")
    methods = tuple(protocol["detectors"])
    conditions = list(protocol["conditions"])
    expected_conditions = ["original_poisoned"]
    for method in methods:
        expected_conditions.extend([f"detector_{method}", f"random_matched_{method}"])
    expected_conditions.append("oracle_all_poison")
    if conditions != expected_conditions:
        raise RuntimeError("condition order or coverage mismatch")

    device = gtsrb.resolve_device(args.device, victim_protocol)
    data_root, dataset_receipt = gtsrb.validate_dataset(victim_protocol)
    log = gtsrb.RunLog(output / "run.log")
    cache, cache_receipt = gtsrb.build_resized_cache(data_root, log)
    source_victims = {int(row["seed"]): row for row in read_csv(source / "victim_metrics.csv")}
    rows = read_csv(output / "per_seed_metrics.csv")
    completed = {(int(row["seed"]), row["condition"]) for row in rows}
    started = time.time()
    try:
        for seed in seeds:
            seed_data = gtsrb.build_seed_data(
                cache["train_images"], cache["train_labels"], victim_protocol, seed
            )
            gtsrb.validate_seed_data(seed_data, cache["train_labels"], victim_protocol)
            source_row = source_victims[seed]
            source_checkpoint = Path(source_row["checkpoint_path"])
            if common.sha256_file(source_checkpoint) != source_row["checkpoint_sha256"]:
                raise RuntimeError(f"seed {seed}: source checkpoint hash mismatch")
            original_key = (seed, "original_poisoned")
            if original_key not in completed:
                rows.append(
                    {
                        "seed": seed,
                        "condition": "original_poisoned",
                        "detector": "none",
                        "rows_removed": 0,
                        "clean_removed": 0,
                        "poison_removed": 0,
                        "poison_remaining": len(seed_data.poison_base_indices),
                        "balanced_clean_accuracy": float(source_row["balanced_clean_accuracy"]),
                        "attack_success_rate": float(source_row["attack_success_rate"]),
                        "best_epoch": int(source_row["best_epoch"]),
                        "checkpoint_path": str(source_checkpoint),
                        "checkpoint_sha256": source_row["checkpoint_sha256"],
                        "state_sha256": source_row["state_sha256"],
                        "training_seconds": 0.0,
                    }
                )
                completed.add(original_key)
                write_csv(output / "per_seed_metrics.csv", rows)

            detector_masks = frozen_removal_masks(source_rows(source, seed), seed_data, methods)
            masks: dict[str, tuple[str, np.ndarray]] = {}
            for method_index, method in enumerate(methods):
                masks[f"detector_{method}"] = (method, detector_masks[method])
                masks[f"random_matched_{method}"] = (
                    method,
                    matched_random_mask(
                        len(seed_data.train_images),
                        int(detector_masks[method].sum()),
                        seed,
                        method_index,
                    ),
                )
            oracle = np.zeros(len(seed_data.train_images), dtype=bool)
            oracle[len(seed_data.train_indices) :] = True
            masks["oracle_all_poison"] = ("oracle", oracle)

            for condition in conditions[1:]:
                key = (seed, condition)
                if key in completed:
                    log(f"seed={seed} condition={condition} skip existing result")
                    continue
                condition_started = time.time()
                detector, remove = masks[condition]
                clean_removed = int(remove[: len(seed_data.train_indices)].sum())
                poison_removed = int(remove[len(seed_data.train_indices) :].sum())
                sanitized = sanitized_data(seed_data, remove)
                log(
                    f"seed={seed} condition={condition} start removed="
                    f"{clean_removed}/{poison_removed}"
                )
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
                    "postfilter_protocol_sha256": common.sha256_file(protocol_path),
                    "victim_protocol_sha256": common.sha256_file(victim_protocol_path),
                    "seed": seed,
                    "condition": condition,
                    "detector": detector,
                    "remove_mask_sha256": sha256_array(remove.astype(np.uint8)),
                    "state_sha256": training["state_sha256"],
                    "state_dict": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                }
                temporary = checkpoint.with_name(checkpoint.name + ".tmp")
                torch.save(payload, temporary)
                temporary.replace(checkpoint)
                row = {
                    "seed": seed,
                    "condition": condition,
                    "detector": detector,
                    "rows_removed": int(remove.sum()),
                    "clean_removed": clean_removed,
                    "poison_removed": poison_removed,
                    "poison_remaining": len(seed_data.poison_base_indices) - poison_removed,
                    "balanced_clean_accuracy": metrics["balanced_clean_accuracy"],
                    "attack_success_rate": metrics["attack_success_rate"],
                    "best_epoch": training["best_epoch"],
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": common.sha256_file(checkpoint),
                    "state_sha256": training["state_sha256"],
                    "remove_mask_sha256": sha256_array(remove.astype(np.uint8)),
                    "training_seconds": time.time() - condition_started,
                }
                rows.append(row)
                completed.add(key)
                write_csv(output / "per_seed_metrics.csv", rows)
                write_json(output / "summary.json", summarize(rows, protocol))
                log(
                    f"seed={seed} condition={condition} committed BCA="
                    f"{metrics['balanced_clean_accuracy']:.6f} ASR="
                    f"{metrics['attack_success_rate']:.6f}"
                )
        rows = sorted(rows, key=lambda row: (int(row["seed"]), conditions.index(row["condition"])))
        write_csv(output / "per_seed_metrics.csv", rows)
        summary = summarize(rows, protocol)
        write_json(output / "summary.json", summary)
        write_json(
            output / "manifest.json",
            {
                "status": "complete",
                "protocol_path": str(protocol_path),
                "protocol_sha256": common.sha256_file(protocol_path),
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256": common.sha256_file(Path(__file__).resolve()),
                "victim_protocol_path": str(victim_protocol_path),
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
