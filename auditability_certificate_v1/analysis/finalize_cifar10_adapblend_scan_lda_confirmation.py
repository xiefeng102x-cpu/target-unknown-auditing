#!/usr/bin/env python3
"""Issue an outcome-neutral audit receipt for the frozen Adap-Blend confirmation (block 13)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
DEFAULT_PROTOCOL = CERT / "PROTOCOL_CIFAR10_ADAPBLEND_SCAN_LDA_CONFIRMATION_V1.json"
DEFAULT_VICTIMS = ROOT / "results/cifar10_vgg16_bn_adapblend_scan_lda_confirmation_v1"
DEFAULT_SCORES = ROOT / "results/cifar10_adapblend_scan_lda_scores_confirmation_v1"
DEFAULT_ENDPOINT = CERT / "results_cifar10_adapblend_scan_lda_confirmation_v1"
DEFAULT_OUTPUT = DEFAULT_ENDPOINT / "audit_receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finalize(protocol_path: Path, victims: Path, scores: Path, endpoint: Path) -> dict:
    protocol_hash = sha256(protocol_path)
    sidecar = protocol_path.with_suffix(protocol_path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != protocol_hash:
        raise RuntimeError("protocol sidecar mismatch")
    protocol = load_json(protocol_path)

    victim_manifest_path = victims / "manifest.json"
    score_manifest_path = scores / "manifest.json"
    endpoint_report_path = endpoint / "report.json"
    decisions_path = endpoint / "decisions.csv"
    summary_path = endpoint / "summary.csv"
    truth_path = endpoint / "truth_by_victim.csv"
    victim_manifest = load_json(victim_manifest_path)
    score_manifest = load_json(score_manifest_path)
    endpoint_report = load_json(endpoint_report_path)

    if victim_manifest.get("status") != "SCORE_HANDOFF_COMPLETE":
        raise RuntimeError("victim block is incomplete")
    if victim_manifest.get("freeze_manifest_sha256") != protocol_hash:
        raise RuntimeError("victim block uses a different protocol")
    if victim_manifest.get("completed_seeds") != protocol["formal_seeds"]:
        raise RuntimeError("victim seed set differs from freeze")
    if score_manifest.get("status") != "CIFAR10_ADAPBLEND_SCAN_LDA_HANDOFF_COMPLETE":
        raise RuntimeError("SCAn handoff is incomplete")
    if score_manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("SCAn handoff uses a different protocol")
    allowed_status = {
        "PASS_CIFAR10_ADAPBLEND_SCAN_LDA_CONFIRMATION",
        "FAIL_CIFAR10_ADAPBLEND_SCAN_LDA_CONFIRMATION",
    }
    if endpoint_report.get("status") not in allowed_status:
        raise RuntimeError("endpoint report has an unknown or incomplete status")
    if endpoint_report.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("endpoint uses a different protocol")

    decisions = pd.read_csv(decisions_path)
    summary = pd.read_csv(summary_path)
    truth = pd.read_csv(truth_path)
    eligible = int(endpoint_report["eligible_victims"])
    expected_rows = (
        eligible
        * len(protocol["certificate"]["budgets"])
        * int(protocol["certificate"]["protected_draws_per_seed_budget"])
    )
    if len(decisions) != expected_rows:
        raise RuntimeError("unexpected endpoint decision count")
    keys = ["victim_id", "k", "draw"]
    duplicate_keys = int(decisions.duplicated(keys).sum())
    if duplicate_keys:
        raise RuntimeError("duplicate endpoint decision keys")
    if int(truth.victim_id.nunique()) != eligible or len(truth) != eligible:
        raise RuntimeError("truth table does not cover each eligible victim exactly once")

    recomputed: list[dict[str, object]] = []
    for budget, rows in decisions.groupby("k", sort=True):
        recomputed.append(
            {
                "k": int(budget),
                "decisions": int(len(rows)),
                "victims": int(rows.victim_id.nunique()),
                "go_rate": float(rows.decision.eq("GO").mean()),
                "nogo_rate": float(rows.decision.eq("NO_GO").mean()),
                "abstain_rate": float(rows.decision.eq("ABSTAIN").mean()),
                "victims_with_go": int(rows.loc[rows.decision.eq("GO"), "victim_id"].nunique()),
                "false_go_count": int(rows.false_go.sum()),
                "false_nogo_count": int(rows.false_nogo.sum()),
            }
        )
    recomputed_frame = pd.DataFrame(recomputed)
    merged = summary.merge(recomputed_frame, on="k", suffixes=("_reported", "_recomputed"), validate="one_to_one")
    for column in (
        "decisions",
        "victims",
        "victims_with_go",
        "false_go_count",
        "false_nogo_count",
    ):
        if not (merged[f"{column}_reported"] == merged[f"{column}_recomputed"]).all():
            raise RuntimeError(f"summary integer mismatch: {column}")
    for column in ("go_rate", "nogo_rate", "abstain_rate"):
        if not ((merged[f"{column}_reported"] - merged[f"{column}_recomputed"]).abs() <= 1e-12).all():
            raise RuntimeError(f"summary rate mismatch: {column}")

    victim_receipts = load_json(victims / "seed_receipts.json")
    receipt_seeds = [int(row["seed"]) for row in victim_receipts]
    if receipt_seeds != protocol["formal_seeds"]:
        raise RuntimeError("victim receipt order or membership differs from freeze")
    victim_metrics = pd.DataFrame(
        {
            "seed": int(row["seed"]),
            "eligible": bool(row["eligible"]),
            "balanced_clean_accuracy": float(row["victim_metrics"]["balanced_accuracy"]),
            "attack_success_rate": float(row["victim_metrics"]["attack_success_rate"]),
        }
        for row in victim_receipts
    )
    false_go = int(decisions.false_go.sum())
    false_nogo = int(decisions.false_nogo.sum())
    return {
        "status": endpoint_report["status"],
        "outcome_neutral_finalizer": True,
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "chronology": {
            "internally_frozen_before_seed_execution": True,
            "external_timestamp": False,
            "formal_seeds": protocol["formal_seeds"],
            "no_seed_replacement": True,
        },
        "victims": {
            "attempted": len(victim_receipts),
            "eligible": int(victim_metrics.eligible.sum()),
            "balanced_clean_accuracy_mean": float(victim_metrics.balanced_clean_accuracy.mean()),
            "balanced_clean_accuracy_min": float(victim_metrics.balanced_clean_accuracy.min()),
            "balanced_clean_accuracy_max": float(victim_metrics.balanced_clean_accuracy.max()),
            "attack_success_rate_mean": float(victim_metrics.attack_success_rate.mean()),
            "attack_success_rate_min": float(victim_metrics.attack_success_rate.min()),
            "attack_success_rate_max": float(victim_metrics.attack_success_rate.max()),
        },
        "policy_truth": {
            "true_pass_victims": int(truth.true_policy_pass.sum()),
            "actual_recall_min": float(truth.actual_recall.min()),
            "actual_recall_max": float(truth.actual_recall.max()),
            "actual_max_group_clean_rate_max": float(truth.actual_max_group_clean_rate.max()),
        },
        "endpoint": {
            "decision_rows": int(len(decisions)),
            "duplicate_decision_keys": duplicate_keys,
            "false_go_count": false_go,
            "false_nogo_count": false_nogo,
            "by_budget": recomputed,
            "support_checks": endpoint_report["support_checks"],
        },
        "claim_boundary": (
            "This block applies the frozen block-12 SCAn+LDA screen, unchanged, to one "
            "Adap-Blend attack on CIFAR-10/VGG-16-BN under an internal pre-execution freeze. "
            "It does not establish universal attack, architecture, detector, or external-"
            "preregistration transfer."
        ),
        "sha256": {
            "victim_manifest": sha256(victim_manifest_path),
            "victim_seed_receipts": sha256(victims / "seed_receipts.json"),
            "score_manifest": sha256(score_manifest_path),
            "flags": sha256(scores / "flags.csv"),
            "endpoint_report": sha256(endpoint_report_path),
            "endpoint_decisions": sha256(decisions_path),
            "endpoint_summary": sha256(summary_path),
            "endpoint_truth": sha256(truth_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--victims", type=Path, default=DEFAULT_VICTIMS)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--endpoint", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = finalize(args.protocol, args.victims, args.scores, args.endpoint)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
