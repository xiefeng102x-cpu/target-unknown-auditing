#!/usr/bin/env python3
"""Frozen full-population SCAn+LDA handoff for CIFAR-10 semantic groups.

This stage never changes the victim or detector.  It reconstructs the frozen
training split and poison plan, extracts one feature vector for every training
member, and emits only the binary SCAn screen needed by the protected-draw
endpoint.  Poison identities are carried in the private handoff solely for
post-draw adjudication and endpoint verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

import backdoorbench_feature_adapter as adapter
import run_cifar10_vgg16_bn_blended_scan_lda_confirmation as victim
import scan_lda_expansion as expansion
from cifar10_vgg16_bn import CIFAR10VGG16BN


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DEFAULT = (
    ROOT / "auditability_certificate_v1/PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json"
)
VICTIM_RESULTS = ROOT / "results/cifar10_vgg16_bn_blended_scan_lda_confirmation_v1"
OUTPUT_DEFAULT = ROOT / "results/cifar10_scan_lda_scores_confirmation_v1"
ADAPTER = ROOT / "src/backdoorbench_feature_adapter.py"
EXPANSION = ROOT / "src/scan_lda_expansion.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def expected_score_hashes() -> dict[str, str]:
    return {
        "score_runner_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(ADAPTER),
        "expansion_sha256": sha256(EXPANSION),
    }


def reference_ids(
    validation_indices: np.ndarray,
    labels: np.ndarray,
    *,
    rows_per_class: int,
    seed_base: int,
) -> np.ndarray:
    selected: list[np.ndarray] = []
    for label in range(10):
        candidates = validation_indices[labels[validation_indices] == label]
        if len(candidates) < rows_per_class:
            raise RuntimeError(f"insufficient class-{label} reference rows")
        order = np.random.default_rng(seed_base + 1009 * label).permutation(candidates)
        selected.append(np.sort(order[:rows_per_class]).astype(np.int64))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--victims", type=Path, default=VICTIM_RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    protocol_hash = sha256(args.protocol)
    if protocol_hash != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("protocol sidecar mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("hashes", {}).get("score_stage") != expected_score_hashes():
        raise RuntimeError("score-stage frozen hashes mismatch")
    if protocol.get("hashes", {}).get("victim_stage") != victim.expected_hashes():
        raise RuntimeError("victim-stage frozen hashes mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("the frozen protocol requires MPS")
    device = torch.device("mps")

    config = victim.merged_config()
    victim.install()
    manifest_path = args.victims / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE":
        raise RuntimeError("victim training is incomplete")
    if manifest.get("freeze_manifest_sha256") != protocol_hash:
        raise RuntimeError("victim manifest protocol mismatch")
    if manifest.get("completed_seeds") != protocol["formal_seeds"]:
        raise RuntimeError("victim manifest seed mismatch")

    development = victim.engine.development
    train_images, train_labels, _, _, dataset_receipt = development.resolve_dataset(config)
    train_indices, validation_indices = development.fixed_split(
        train_labels,
        rows_per_class=int(config["dataset"]["validation_rows_per_class"]),
        seed=int(config["dataset"]["validation_split_seed"]),
    )
    refs = reference_ids(
        validation_indices,
        train_labels,
        rows_per_class=int(protocol["reference_rows_per_class"]),
        seed_base=int(protocol["reference_seed_base"]),
    )

    receipts_by_seed = {
        int(row["seed"]): row
        for row in json.loads((args.victims / "seed_receipts.json").read_text(encoding="utf-8"))
    }
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, object]] = []
    for seed in map(int, protocol["formal_seeds"]):
        seed_receipt = receipts_by_seed.get(seed)
        if seed_receipt is None:
            raise RuntimeError(f"missing victim receipt for seed {seed}")
        checkpoint = args.victims / f"models/seed_{seed}_suspect.pt"
        if sha256(checkpoint) != seed_receipt["model_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for seed {seed}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = CIFAR10VGG16BN()
        model.load_state_dict(state, strict=True)
        model.to(device).eval()

        poison_indices = development.poison_plan(
            train_labels,
            train_indices,
            target=int(config["attack"]["target_class"]),
            count=int(config["attack"]["poison_count"]),
            seed=seed,
        )
        if array_sha256(poison_indices) != seed_receipt["poison_indices_sha256"]:
            raise RuntimeError(f"poison-plan mismatch for seed {seed}")
        poisoned = set(map(int, poison_indices))
        assigned = train_labels[train_indices].copy()
        is_poison = np.isin(train_indices, poison_indices, assume_unique=True)
        assigned[is_poison] = int(config["attack"]["target_class"])

        _, candidate_features = victim.engine.model_batch(
            model,
            train_images,
            train_indices,
            device,
            poisoned=poisoned,
            batch_size=args.batch_size,
        )
        _, trusted_features = victim.engine.model_batch(
            model,
            train_images,
            refs,
            device,
            batch_size=args.batch_size,
        )
        result = adapter.scan(
            trusted_features,
            train_labels[refs],
            candidate_features,
            assigned,
            n_classes=10,
            seed=int(protocol["scan_seed_base"]) + seed,
        )
        scan_flags = np.asarray(result.suspicious_mask, dtype=bool)
        parameters = protocol["detector_parameters"]
        if parameters != expansion.FROZEN_PARAMETERS:
            raise RuntimeError("expansion parameters differ from freeze")
        expanded = scan_flags.copy()
        expansion_receipts = {}
        for label in map(int, result.receipt["flagged_classes"]):
            members = np.flatnonzero(assigned == label)
            class_refs = refs[train_labels[refs] == label]
            class_validation = validation_indices[train_labels[validation_indices] == label]
            class_calibration = np.sort(np.setdiff1d(class_validation, refs)).astype(np.int64)
            if len(class_calibration) != int(protocol["calibration_rows_per_class"]):
                raise RuntimeError(f"unexpected calibration size for class {label}")
            _, calibration_features = victim.engine.model_batch(
                model, train_images, class_calibration, device, batch_size=args.batch_size,
            )
            reference_features = trusted_features[train_labels[refs] == label]
            class_flags, info = expansion.expand_class(
                reference_features, calibration_features, candidate_features[members],
                scan_flags[members], **parameters,
            )
            expanded[members] = class_flags
            info.update({"reference_ids_sha256": array_sha256(class_refs),
                         "calibration_ids_sha256": array_sha256(class_calibration)})
            expansion_receipts[str(label)] = info
        flags = expanded.astype(np.int8)
        eligible = bool(seed_receipt["eligible"])
        victim_id = f"cifar10_blended_scan_lda_vgg16bn_seed{seed}"
        frames.append(
            pd.DataFrame(
                {
                    "victim_id": victim_id,
                    "seed": seed,
                    "sample_id": train_indices.astype(np.int64),
                    "assigned_group": assigned.astype(np.int64),
                    "flag": flags,
                    "scan_flag": scan_flags.astype(np.int8),
                    "is_poison": is_poison.astype(np.int8),
                    "victim_eligible": int(eligible),
                }
            )
        )
        receipts.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha256(checkpoint),
                "eligible": eligible,
                "balanced_clean_accuracy": float(seed_receipt["victim_metrics"]["balanced_accuracy"]),
                "attack_success_rate": float(seed_receipt["victim_metrics"]["attack_success_rate"]),
                "recall": float(flags[is_poison].mean()),
                "scan_only_recall": float(scan_flags[is_poison].mean()),
                "expansion_receipts": expansion_receipts,
                "all_clean_flag_rate": float(flags[~is_poison].mean()),
                "population_rows": int(len(train_indices)),
                "poison_rows": int(is_poison.sum()),
                "reference_rows": int(len(refs)),
                "reference_ids_sha256": array_sha256(refs),
                "adapter_receipt": result.receipt,
            }
        )
        print(
            f"seed={seed} eligible={eligible} recall={receipts[-1]['recall']:.6f} "
            f"clean_rate={receipts[-1]['all_clean_flag_rate']:.6f}",
            flush=True,
        )

    args.output.mkdir(parents=True)
    flags_path = args.output / "flags.csv"
    pd.concat(frames, ignore_index=True).to_csv(flags_path, index=False)
    output_manifest = {
        "status": "CIFAR10_SCAN_LDA_HANDOFF_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "score_runner_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(ADAPTER),
        "expansion_sha256": sha256(EXPANSION),
        "victim_manifest_sha256": sha256(manifest_path),
        "dataset": dataset_receipt,
        "device": device.type,
        "flags_sha256": sha256(flags_path),
        "seed_count": len(receipts),
        "eligible_seed_count": sum(int(row["eligible"]) for row in receipts),
        "seeds": receipts,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in output_manifest.items() if key != "seeds"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
