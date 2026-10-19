#!/usr/bin/env python3
"""Verify hashes and structural invariants of the CIFAR-10/VGG16 SRAC handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/cifar10_vgg16_bn_srac_confirmation_v1.json"
DEFAULT_RESULTS = ROOT / "results/cifar10_vgg16_bn_srac_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    receipts = json.loads((args.results / "seed_receipts.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    seeds = [int(seed) for seed in protocol["formal_seeds"]]

    if protocol.get("protocol_id") != "cifar10_vgg16_bn_srac_confirmation_v1":
        errors.append("unexpected protocol id")
    if manifest.get("status") != "SCORE_HANDOFF_COMPLETE":
        errors.append("score handoff is incomplete")
    if manifest.get("completed_seeds") != seeds:
        errors.append("completed seed block differs from freeze")
    if not manifest.get("all_victims_eligible"):
        errors.append("at least one frozen victim is ineligible")
    if len(receipts) != len(seeds) or [int(row["seed"]) for row in receipts] != seeds:
        errors.append("seed receipt block is incomplete or reordered")

    paths = {
        "scores": args.results / "blind_scores.csv",
        "truth": args.results / "private_truth.csv",
        "trusted": args.results / "trusted_member_pool.csv",
        "receipts": args.results / "seed_receipts.json",
    }
    declared = {
        "scores": manifest.get("blind_scores_sha256"),
        "truth": manifest.get("private_truth_sha256"),
        "trusted": manifest.get("trusted_member_pool_sha256"),
        "receipts": manifest.get("seed_receipts_sha256"),
    }
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != declared[name]:
            errors.append(f"{name} hash mismatch")

    if not errors:
        scores = pd.read_csv(paths["scores"])
        truth = pd.read_csv(paths["truth"])
        train_scores = scores.loc[scores["role"].eq("train")].copy()
        train_truth = truth.loc[truth["role"].eq("train")].copy()
        joined = train_scores.merge(
            train_truth[["victim_id", "sample_id", "role", "is_poison"]],
            on=["victim_id", "sample_id", "role"],
            how="left",
            validate="many_to_one",
        )
        if joined["is_poison"].isna().any():
            errors.append("incomplete truth join")
        if set(joined["conventional_flag"].unique()) - {0, 1}:
            errors.append("non-binary conventional flag")
        expected_methods = set(protocol["detectors"])
        if set(joined["detector"].unique()) != expected_methods:
            errors.append("detector set mismatch")
        if set(joined["fold"].astype(int).unique()) != {0, 1}:
            errors.append("fold set mismatch")
        for victim, group in train_truth.groupby("victim_id", sort=True):
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

    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "protocol_sha256": sha256(args.protocol),
        "verified_seeds": seeds,
        "checkpoint_count": len(receipts),
        "score_rows": int(len(pd.read_csv(paths["scores"]))) if paths["scores"].is_file() else 0,
        "truth_rows": int(len(pd.read_csv(paths["truth"]))) if paths["truth"].is_file() else 0,
    }
    receipt_path = args.results / "srac_handoff_verification.json"
    receipt_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
