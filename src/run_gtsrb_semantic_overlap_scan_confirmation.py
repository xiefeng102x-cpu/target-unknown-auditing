#!/usr/bin/env python3
"""Frozen SCAn handoff for the GTSRB semantic-overlap confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DEFAULT = ROOT / "auditability_certificate_v1/PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V1.json"
VICTIM_PROTOCOL = ROOT / "configs/srac_gtsrb_semantic_overlap_confirmation_v1.json"
VICTIM_RESULTS = ROOT / "results/srac_gtsrb_semantic_overlap_confirmation_v1"
CACHE = ROOT / "data/gtsrb_official/gtsrb_32x32_uint8_v1.npz"
NATIVE_RUNNER = ROOT / "src/run_tifs_native_gtsrb.py"
ADAPTER = ROOT / "src/backdoorbench_feature_adapter.py"
OUTPUT_DEFAULT = ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_ids(data) -> list[str]:
    return [f"train_clean_{int(index)}" for index in data.train_indices] + [
        f"train_poison_source{int(source)}_base{int(index)}"
        for source, index in zip(data.poison_sources, data.poison_base_indices)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", choices=("mps",), required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    protocol_hash = sha256(args.protocol)
    if protocol_hash != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("protocol sidecar mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    bindings = {
        "runner_sha256": Path(__file__),
        "victim_protocol_sha256": VICTIM_PROTOCOL,
        "victim_runner_sha256": NATIVE_RUNNER,
        "adapter_sha256": ADAPTER,
    }
    for key, path in bindings.items():
        if protocol[key] != sha256(path):
            raise RuntimeError(f"frozen binding mismatch: {key}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device("mps")

    victim_manifest_path = VICTIM_RESULTS / "manifest.json"
    victim_manifest = json.loads(victim_manifest_path.read_text(encoding="utf-8"))
    if victim_manifest.get("status") != "complete":
        raise RuntimeError("victim training incomplete")
    if victim_manifest.get("protocol_sha256") != protocol["victim_protocol_sha256"]:
        raise RuntimeError("victim protocol hash mismatch")
    if victim_manifest.get("runner_sha256") != protocol["victim_runner_sha256"]:
        raise RuntimeError("victim runner hash mismatch")
    if victim_manifest.get("completed_seeds") != protocol["seeds"]:
        raise RuntimeError("victim seed mismatch")

    native = load_module(NATIVE_RUNNER, "gtsrb_semantic_overlap_native")
    adapter = load_module(ADAPTER, "gtsrb_semantic_overlap_adapter")
    victim_protocol = json.loads(VICTIM_PROTOCOL.read_text(encoding="utf-8"))
    victim_metrics = pd.read_csv(VICTIM_RESULTS / "victim_metrics.csv")
    with np.load(CACHE) as loaded:
        cache = {name: loaded[name].copy() for name in loaded.files}

    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, object]] = []
    for seed in map(int, protocol["seeds"]):
        data = native.build_seed_data(
            cache["train_images"], cache["train_labels"], victim_protocol, seed
        )
        native.validate_seed_data(data, cache["train_labels"], victim_protocol)
        checkpoint = VICTIM_RESULTS / f"models/seed_{seed}_suspect.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = native.GTSRBTinyCNN(43)
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        _, train_features = native.predict_and_features(
            model, data.train_images, args.batch_size, device, True
        )
        _, test_features = native.predict_and_features(
            model, cache["test_images"], args.batch_size, device, True
        )
        assert train_features is not None and test_features is not None

        reference_indices: list[int] = []
        for group in range(43):
            candidates = np.flatnonzero(cache["test_labels"] == group)
            order = np.random.default_rng(
                int(protocol["reference_seed_base"]) + seed + group * 1009
            ).permutation(candidates)
            reference_indices.extend(order[: len(order) // 2].tolist())
        reference = np.asarray(reference_indices, dtype=np.int64)
        result = adapter.scan(
            test_features[reference],
            cache["test_labels"][reference],
            train_features,
            data.train_labels,
            n_classes=43,
            seed=int(protocol["scan_seed_base"]) + seed,
        )
        truth = np.concatenate(
            [
                np.zeros(len(data.train_indices), dtype=np.int8),
                np.ones(len(data.poison_base_indices), dtype=np.int8),
            ]
        )
        flags = np.asarray(result.suspicious_mask, dtype=np.int8)
        assigned = data.train_labels.astype(np.int64)
        metric = victim_metrics.loc[victim_metrics.seed.eq(seed)]
        if len(metric) != 1:
            raise RuntimeError(f"victim metric mismatch for seed {seed}")
        eligible = bool(int(metric.iloc[0]["eligible"]))
        frames.append(
            pd.DataFrame(
                {
                    "victim_id": f"gtsrb_semantic_overlap_seed{seed}",
                    "seed": seed,
                    "sample_id": sample_ids(data),
                    "assigned_group": assigned,
                    "flag": flags,
                    "is_poison": truth,
                    "victim_eligible": int(eligible),
                }
            )
        )
        poison = truth.astype(bool)
        receipts.append(
            {
                "seed": seed,
                "checkpoint_sha256": sha256(checkpoint),
                "eligible": eligible,
                "balanced_clean_accuracy": float(metric.iloc[0]["balanced_clean_accuracy"]),
                "attack_success_rate": float(metric.iloc[0]["attack_success_rate"]),
                "recall": float(flags[poison].mean()),
                "all_clean_flag_rate": float(flags[~poison].mean()),
                "reference_rows": len(reference),
                "reference_indices_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
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
    manifest = {
        "status": "SEMANTIC_OVERLAP_SCAN_HANDOFF_COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "runner_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(ADAPTER),
        "victim_manifest_sha256": sha256(victim_manifest_path),
        "device": device.type,
        "flags_sha256": sha256(flags_path),
        "seed_count": len(receipts),
        "eligible_seed_count": sum(int(row["eligible"]) for row in receipts),
        "seeds": receipts,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
