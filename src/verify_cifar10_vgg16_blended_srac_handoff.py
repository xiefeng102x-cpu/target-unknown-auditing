#!/usr/bin/env python3
"""Verify structural invariants of the CIFAR/VGG Blended SRAC handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/cifar10_vgg16_bn_blended_srac_confirmation_v1.json"
DEFAULT_RESULTS = ROOT / "results/cifar10_vgg16_bn_blended_srac_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(protocol_path: Path, results: Path) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    receipts = json.loads((results / "seed_receipts.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    seeds = [int(seed) for seed in protocol["formal_seeds"]]
    if protocol.get("protocol_id") != "cifar10_vgg16_bn_blended_srac_confirmation_v1":
        errors.append("unexpected protocol id")
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE":
        errors.append("score handoff is incomplete")
    if manifest.get("completed_seeds") != seeds or not manifest.get("all_victims_eligible"):
        errors.append("completed or eligible seed block differs from freeze")
    if [int(row["seed"]) for row in receipts] != seeds:
        errors.append("seed receipt block is incomplete or reordered")

    paths = {
        "scores": results / "blind_scores.csv",
        "truth": results / "private_truth.csv",
        "trusted": results / "trusted_member_pool.csv",
        "receipts": results / "seed_receipts.json",
    }
    declarations = {
        "scores": manifest.get("blind_scores_sha256"),
        "truth": manifest.get("private_truth_sha256"),
        "trusted": manifest.get("trusted_member_pool_sha256"),
        "receipts": manifest.get("seed_receipts_sha256"),
    }
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != declarations[name]:
            errors.append(f"{name} hash mismatch")

    score_rows = truth_rows = 0
    if not errors:
        scores = pd.read_csv(paths["scores"])
        truth = pd.read_csv(paths["truth"])
        score_rows, truth_rows = len(scores), len(truth)
        train_scores = scores.loc[scores["role"].eq("train")]
        train_truth = truth.loc[truth["role"].eq("train")]
        joined = train_scores.merge(
            train_truth[["victim_id", "sample_id", "role", "is_poison"]],
            on=["victim_id", "sample_id", "role"], how="left", validate="many_to_one",
        )
        if joined["is_poison"].isna().any():
            errors.append("incomplete truth join")
        if set(joined["conventional_flag"].unique()) - {0, 1}:
            errors.append("non-binary conventional flag")
        if set(joined["detector"].unique()) != set(protocol["detectors"]):
            errors.append("detector set mismatch")
        if set(joined["fold"].astype(int).unique()) != {0, 1}:
            errors.append("fold set mismatch")
        for victim, group in train_truth.groupby("victim_id", sort=True):
            if not str(victim).startswith("cifar10_blended_vgg16bn_seed"):
                errors.append(f"unexpected victim tag: {victim}")
            if int(group["is_poison"].sum()) != int(protocol["attack"]["poison_count"]):
                errors.append(f"poison count mismatch for {victim}")
            if len(group) != int(protocol["crossfit"]["expected_rows"]):
                errors.append(f"target-stratum size mismatch for {victim}")

    for receipt in receipts:
        checkpoint = Path(receipt["model_file"])
        feature = Path(receipt["feature_file"])
        if not checkpoint.is_file() or sha256(checkpoint) != receipt["model_sha256"]:
            errors.append(f"checkpoint hash mismatch for seed {receipt['seed']}")
        if not feature.is_file() or sha256(feature) != receipt["feature_sha256"]:
            errors.append(f"feature hash mismatch for seed {receipt['seed']}")
        if not receipt.get("eligible"):
            errors.append(f"ineligible seed {receipt['seed']}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "protocol_sha256": sha256(protocol_path),
        "verified_seeds": seeds,
        "checkpoint_count": len(receipts),
        "score_rows": int(score_rows),
        "truth_rows": int(truth_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    report = verify(args.protocol, args.results)
    receipt = args.results / "srac_handoff_verification.json"
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
