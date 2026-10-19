#!/usr/bin/env python3
"""Combine detector, full-population random, and candidate-random controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/srac_gtsrb_global_blend_candidate_random_confirmation_v1.json"
DEFAULT_CANDIDATE = ROOT / "results/srac_gtsrb_global_blend_candidate_random_confirmation_v1"
DEFAULT_DETECTOR = ROOT / "results/srac_gtsrb_global_blend_postfilter_confirmation_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sampled.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def condition_summary(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "seed_count": int(len(rows)),
        "mean_clean_removed": float(rows["clean_removed"].mean()),
        "mean_poison_removed": float(rows["poison_removed"].mean()),
        "mean_balanced_clean_accuracy": float(rows["balanced_clean_accuracy"].mean()),
        "mean_attack_success_rate": float(rows["attack_success_rate"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    fixed = protocol["fixed_detector_postfilter_comparator"]
    if sha256(ROOT / "configs/srac_gtsrb_global_blend_postfilter_confirmation_v1.json") != fixed["protocol_sha256"]:
        raise RuntimeError("fixed comparator protocol hash mismatch")
    if sha256(args.detector / "manifest.json") != fixed["manifest_sha256"]:
        raise RuntimeError("fixed comparator manifest hash mismatch")
    if sha256(args.detector / "verification_receipt.json") != fixed["verification_receipt_sha256"]:
        raise RuntimeError("fixed comparator verification hash mismatch")
    candidate_receipt = json.loads(
        (args.candidate / "verification_receipt.json").read_text(encoding="utf-8")
    )
    if candidate_receipt.get("status") != "PASS":
        raise RuntimeError("candidate-random verification did not pass")

    detected = pd.read_csv(args.detector / "per_seed_metrics.csv")
    candidate = pd.read_csv(args.candidate / "per_seed_metrics.csv")
    seeds = [int(seed) for seed in protocol["seeds"]]
    conditions: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    bootstrap_spec = protocol["analysis"]["bootstrap"]
    for name in ("original_poisoned", "oracle_all_poison"):
        conditions[name] = condition_summary(detected.loc[detected["condition"].eq(name)])
    for index, method in enumerate(protocol["detectors"]):
        names = {
            "detector": f"detector_{method}",
            "full_population_random": f"random_matched_{method}",
            "candidate_random": f"candidate_random_matched_{method}",
        }
        frames = {
            "detector": detected.loc[detected["condition"].eq(names["detector"])].set_index("seed"),
            "full_population_random": detected.loc[
                detected["condition"].eq(names["full_population_random"])
            ].set_index("seed"),
            "candidate_random": candidate.loc[
                candidate["condition"].eq(names["candidate_random"])
            ].set_index("seed"),
        }
        for label, frame in frames.items():
            if list(sorted(map(int, frame.index))) != seeds:
                raise RuntimeError(f"{method}/{label}: seed coverage mismatch")
            conditions[f"{method}/{label}"] = condition_summary(frame.reset_index())
        asr_delta = (
            frames["detector"]["attack_success_rate"]
            - frames["candidate_random"]["attack_success_rate"]
        ).to_numpy(float)
        bca_delta = (
            frames["detector"]["balanced_clean_accuracy"]
            - frames["candidate_random"]["balanced_clean_accuracy"]
        ).to_numpy(float)
        contrasts[method] = {
            "seeds": seeds,
            "mean_asr_delta_detector_minus_candidate_random": float(asr_delta.mean()),
            "median_asr_delta_detector_minus_candidate_random": float(np.median(asr_delta)),
            "asr_delta_mean_bootstrap_95": bootstrap(
                asr_delta,
                int(bootstrap_spec["seed"]) + index,
                int(bootstrap_spec["draws"]),
            ),
            "detector_lower_asr_seed_count": int(np.sum(asr_delta < 0)),
            "mean_bca_delta_detector_minus_candidate_random": float(bca_delta.mean()),
            "bca_delta_mean_bootstrap_95": bootstrap(
                bca_delta,
                int(bootstrap_spec["seed"]) + 100 + index,
                int(bootstrap_spec["draws"]),
            ),
        }
        contrasts[method]["support_rule"] = bool(
            contrasts[method]["asr_delta_mean_bootstrap_95"][1] < 0
            and contrasts[method]["mean_bca_delta_detector_minus_candidate_random"] >= -0.05
        )

    result = {
        "status": "PASS",
        "conditions": conditions,
        "paired_detector_minus_candidate_random": contrasts,
        "input_hashes": {
            "protocol": sha256(args.protocol),
            "candidate_manifest": sha256(args.candidate / "manifest.json"),
            "candidate_metrics": sha256(args.candidate / "per_seed_metrics.csv"),
            "candidate_verification": sha256(args.candidate / "verification_receipt.json"),
            "detector_manifest": sha256(args.detector / "manifest.json"),
            "detector_metrics": sha256(args.detector / "per_seed_metrics.csv"),
            "detector_verification": sha256(args.detector / "verification_receipt.json"),
        },
    }
    output = args.output or (args.candidate / "comparison_with_detector.json")
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
