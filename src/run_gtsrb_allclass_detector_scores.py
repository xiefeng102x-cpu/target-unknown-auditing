#!/usr/bin/env python3
"""Generate target-unknown all-class detector flags for frozen GTSRB victims.

The runner never receives an attack target for detector calibration.  It applies
the same three detector families independently to every assigned-label group.
Attack metadata is used only to reconstruct the already frozen victim population
and private truth for evaluation.
"""

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
PROTOCOL_DEFAULT = ROOT / "auditability_certificate_v1/PROTOCOL_GTSRB_ALLCLASS_GROUP_SAFE_V1.json"
VICTIM_PROTOCOL = ROOT / "configs/srac_gtsrb_global_blend_confirmation_v1.json"
VICTIM_RESULTS = ROOT / "results/srac_gtsrb_global_blend_confirmation_v1"
CACHE = ROOT / "data/gtsrb_official/gtsrb_32x32_uint8_v1.npz"
OUTPUT_DEFAULT = ROOT / "results/gtsrb_allclass_detector_scores_v1"
NATIVE_RUNNER = ROOT / "src/run_tifs_native_gtsrb.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_ids(data) -> list[str]:
    clean = [f"train_clean_{int(index)}" for index in data.train_indices]
    poison = [
        f"train_poison_source{int(source)}_base{int(index)}"
        for source, index in zip(data.poison_sources, data.poison_base_indices)
    ]
    return clean + poison


def synthetic_self_test(native) -> None:
    rng = np.random.default_rng(20261015)
    reference = rng.normal(size=(30, 64))
    calibration = rng.normal(size=(30, 64))
    candidates = rng.normal(size=(80, 64))
    for builder in (
        lambda: native.common.cf_cmd_u(reference, calibration, candidates, 11, 0.05),
        lambda: native.common.centroid_distance(reference, calibration, candidates, 0.05),
        lambda: native.common.crossfit_spectral(reference, calibration, candidates, 11, 0.05),
    ):
        result = builder()
        if len(result["flags"]) != 80 or not np.isfinite(result["scores"]).all():
            raise RuntimeError("all-class detector synthetic self-test failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    native = load_module(NATIVE_RUNNER, "native_gtsrb_allclass_dependency")
    if args.self_test:
        synthetic_self_test(native)
        print("GTSRB_ALLCLASS_STATIC_SELF_TEST_PASS")
        return

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    expected_protocol_hash = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256(args.protocol) != expected_protocol_hash:
        raise RuntimeError("protocol sidecar mismatch")
    if protocol.get("study_id") != "gtsrb_allclass_group_safe_v1":
        raise RuntimeError("unexpected all-class protocol")
    if protocol.get("runner_sha256") != sha256(Path(__file__)):
        raise RuntimeError("runner hash differs from frozen protocol")
    if protocol.get("victim_protocol_sha256") != sha256(VICTIM_PROTOCOL):
        raise RuntimeError("victim protocol hash mismatch")
    if protocol.get("victim_manifest_sha256") != sha256(VICTIM_RESULTS / "manifest.json"):
        raise RuntimeError("victim manifest hash mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "per_seed").mkdir()

    victim_protocol = json.loads(VICTIM_PROTOCOL.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    with np.load(CACHE) as loaded:
        cache = {name: loaded[name].copy() for name in loaded.files}
    seeds = [int(seed) for seed in protocol["seeds"]]
    alpha = float(protocol["detector_alpha"])
    outputs: list[dict[str, object]] = []

    for seed in seeds:
        data = native.build_seed_data(cache["train_images"], cache["train_labels"], victim_protocol, seed)
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
        ids = np.asarray(sample_ids(data), dtype=object)
        assigned = data.train_labels.astype(np.int64)
        truth = np.concatenate(
            [np.zeros(len(data.train_indices), dtype=np.int8), np.ones(len(data.poison_base_indices), dtype=np.int8)]
        )
        if not (len(ids) == len(assigned) == len(truth) == len(train_features)):
            raise RuntimeError(f"population alignment failed for seed {seed}")

        seed_rows: list[pd.DataFrame] = []
        group_receipts: list[dict[str, object]] = []
        for group in range(43):
            trusted_indices = np.flatnonzero(cache["test_labels"] == group)
            order = np.random.default_rng(
                int(protocol["trusted_split_seed_base"]) + seed + group * 1009
            ).permutation(trusted_indices)
            cut = len(order) // 2
            if cut < int(protocol["minimum_reference_rows_per_group"]):
                raise RuntimeError(f"insufficient trusted rows for group {group}: {len(order)}")
            reference = test_features[order[:cut]]
            calibration = test_features[order[cut:]]
            candidate_positions = np.flatnonzero(assigned == group)
            candidates = train_features[candidate_positions]
            detector_seed = int(protocol["detector_seed_base"]) + seed * 1000 + group
            results = {
                "cf_cmd_u": native.common.cf_cmd_u(
                    reference, calibration, candidates, detector_seed, alpha
                ),
                "target_centroid_distance": native.common.centroid_distance(
                    reference, calibration, candidates, alpha
                ),
                "crossfit_spectral": native.common.crossfit_spectral(
                    reference, calibration, candidates, detector_seed, alpha
                ),
            }
            for method, result in results.items():
                seed_rows.append(
                    pd.DataFrame(
                        {
                            "victim_id": f"gtsrb_patch_tinycnn_seed{seed}",
                            "sample_id": ids[candidate_positions],
                            "assigned_group": group,
                            "detector": method,
                            "score": np.asarray(result["scores"], dtype=float),
                            "flag": np.asarray(result["flags"], dtype=np.int8),
                            "is_poison": truth[candidate_positions],
                        }
                    )
                )
            group_receipts.append(
                {
                    "group": group,
                    "trusted_rows": len(order),
                    "reference_rows": cut,
                    "calibration_rows": len(order) - cut,
                    "candidate_rows": len(candidate_positions),
                    "poison_rows": int(truth[candidate_positions].sum()),
                }
            )
        table = pd.concat(seed_rows, ignore_index=True)
        expected = len(ids) * 3
        if len(table) != expected:
            raise RuntimeError(f"all-class score row count mismatch for seed {seed}")
        key = ["victim_id", "sample_id", "detector"]
        if table.duplicated(key).any():
            raise RuntimeError(f"duplicate all-class score key for seed {seed}")
        output = args.output / "per_seed" / f"seed_{seed}_allclass_scores.csv"
        table.to_csv(output, index=False)
        outputs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "rows": len(table),
                "score_sha256": sha256(output),
                "groups": group_receipts,
            }
        )
        print(
            f"seed={seed} rows={len(table)} device={device.type} "
            f"flags={int(table['flag'].sum())}",
            flush=True,
        )

    manifest = {
        "status": "ALLCLASS_SCORE_HANDOFF_COMPLETE",
        "study_id": protocol["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(args.protocol),
        "runner_sha256": sha256(Path(__file__)),
        "device": device.type,
        "cache_sha256": sha256(CACHE),
        "seed_count": len(outputs),
        "total_score_rows": int(sum(int(row["rows"]) for row in outputs)),
        "seeds": outputs,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
