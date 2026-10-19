#!/usr/bin/env python3
"""Issue a fail-closed receipt for the frozen CIFAR/VGG Blended block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
DEFAULT_PROTOCOL = PAPER14 / "configs/cifar10_vgg16_bn_blended_srac_confirmation_v1.json"
DEFAULT_RESULTS = PAPER14 / "results/cifar10_vgg16_bn_blended_srac_confirmation_v1"
DEFAULT_ENDPOINT = ROOT / "results_cifar10_vgg16_blended_srac_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require_status(record: dict, expected: str, path: Path) -> None:
    if record.get("status") != expected:
        raise RuntimeError(f"{path} has status {record.get('status')!r}; expected {expected!r}")


def finalize(protocol_path: Path, results: Path, endpoint: Path) -> dict:
    protocol = load_json(protocol_path)
    manifest_path = results / "manifest.json"
    handoff_path = results / "srac_handoff_verification.json"
    replay_path = results / "checkpoint_score_semantic_replay_verification.json"
    endpoint_path = endpoint / "report.json"
    decisions_path = endpoint / "decisions.csv"
    summary_path = endpoint / "summary.csv"
    ranges_path = endpoint / "actual_ranges.csv"
    manifest = load_json(manifest_path)
    handoff = load_json(handoff_path)
    replay = load_json(replay_path)
    endpoint_report = load_json(endpoint_path)
    require_status(manifest, "SCORE_HANDOFF_COMPLETE", manifest_path)
    require_status(handoff, "PASS", handoff_path)
    require_status(replay, "PASS", replay_path)
    require_status(endpoint_report, "COMPLETE_FROZEN_ENDPOINT", endpoint_path)

    seeds = [int(seed) for seed in protocol["formal_seeds"]]
    if manifest.get("completed_seeds") != seeds or not manifest.get("all_victims_eligible"):
        raise RuntimeError("formal victim block is incomplete or ineligible")
    if handoff.get("verified_seeds") != seeds or replay.get("verified_seeds") != seeds:
        raise RuntimeError("verification receipts do not cover every frozen seed")
    if endpoint_report.get("duplicate_decision_keys") != 0:
        raise RuntimeError("duplicate SRAC decision keys")

    receipts = load_json(results / "seed_receipts.json")
    if [int(row["seed"]) for row in receipts] != seeds:
        raise RuntimeError("seed receipt order or membership differs from freeze")
    decisions = pd.read_csv(decisions_path)
    summary = pd.read_csv(summary_path)
    ranges = pd.read_csv(ranges_path)
    expected_rows = (
        len(seeds)
        * int(protocol["crossfit"]["folds"])
        * len(protocol["detectors"])
        * len(protocol["srac_evaluation"]["audit_budgets"])
        * int(protocol["srac_evaluation"]["protected_draws_per_seed_fold_budget"])
    )
    if len(decisions) != expected_rows or endpoint_report.get("decision_rows") != expected_rows:
        raise RuntimeError("frozen endpoint produced an unexpected decision count")

    detectors = {}
    for detector, group in decisions.groupby("detector", sort=True):
        detectors[str(detector)] = {
            "actual_recall_mean": float(group["actual_recall"].mean()),
            "actual_recall_min": float(group["actual_recall"].min()),
            "actual_recall_max": float(group["actual_recall"].max()),
            "actual_clean_flag_rate_mean": float(group["actual_clean_flag_rate"].mean()),
            "go_rate": float(group["decision"].eq("GO").mean()),
            "nogo_rate": float(group["decision"].eq("NO_GO").mean()),
            "abstain_rate": float(group["decision"].eq("ABSTAIN").mean()),
            "false_go_count": int(group["false_go"].sum()),
            "false_nogo_count": int(group["false_nogo"].sum()),
        }
    victim_metrics = pd.DataFrame([row["victim_metrics"] for row in receipts])
    raw_byte_exact = all(
        row["score_replay"]["raw_sha256"]["byte_exact"] for row in replay["per_seed"]
    )
    score_max = max(
        max(row["score_replay"]["score_max_absolute_difference"].values())
        for row in replay["per_seed"]
    )
    report = {
        "status": "PASS",
        "scope": "locally pre-frozen CIFAR-10/VGG16-BN/Blended-variant confirmation and whole-population SRAC endpoint",
        "chronology": {
            "confirmation_seeds": seeds,
            "confirmation_seeds_disjoint_from_development_seed": True,
            "development_information_boundary": "BCA and ASR only",
            "protected_draws_started_after_local_freeze": True,
            "external_timestamp_or_independent_custodian": False,
        },
        "victims": {
            "eligible": len(receipts),
            "balanced_clean_accuracy_mean": float(victim_metrics["balanced_accuracy"].mean()),
            "balanced_clean_accuracy_min": float(victim_metrics["balanced_accuracy"].min()),
            "balanced_clean_accuracy_max": float(victim_metrics["balanced_accuracy"].max()),
            "attack_success_rate_mean": float(victim_metrics["attack_success_rate"].mean()),
            "attack_success_rate_min": float(victim_metrics["attack_success_rate"].min()),
            "attack_success_rate_max": float(victim_metrics["attack_success_rate"].max()),
        },
        "attack": {
            "name": protocol["attack"]["implementation_name"],
            "fidelity_boundary": protocol["attack"]["fidelity_boundary"],
        },
        "audit_universe": {
            "definition": protocol["srac_evaluation"]["audit_universe"],
            "rows_per_victim": int(protocol["crossfit"]["expected_rows"]),
            "clean_rows_per_victim": int(protocol["crossfit"]["expected_clean_rows"]),
            "poison_rows_per_victim": int(protocol["crossfit"]["expected_poison_rows"]),
            "poison_prevalence": float(
                protocol["crossfit"]["expected_poison_rows"] / protocol["crossfit"]["expected_rows"]
            ),
        },
        "detectors": detectors,
        "srac_endpoint": {
            "decision_rows": int(len(decisions)),
            "false_go_count": int(decisions["false_go"].sum()),
            "false_nogo_count": int(decisions["false_nogo"].sum()),
            "by_budget": endpoint_report["by_budget"],
        },
        "verification": {
            "structural_handoff": handoff["status"],
            "prospective_semantic_replay": replay["status"],
            "raw_score_tables_byte_exact": raw_byte_exact,
            "maximum_absolute_score_difference": score_max,
            "exact_field_mismatch_count": int(sum(
                sum(row["score_replay"]["exact_field_mismatches"].values())
                for row in replay["per_seed"]
            )),
        },
        "claim_boundary": "This block tests cross-attack transfer from localized BadNets patches to a distributed Blended variant. It does not establish exact upstream BackdoorBench fidelity, universal attack transfer, external preregistration, or independence among repeated protected draws.",
        "sha256": {
            "protocol": sha256(protocol_path),
            "training_manifest": sha256(manifest_path),
            "handoff_verification": sha256(handoff_path),
            "semantic_replay_verification": sha256(replay_path),
            "endpoint_report": sha256(endpoint_path),
            "endpoint_decisions": sha256(decisions_path),
            "endpoint_summary": sha256(summary_path),
            "endpoint_actual_ranges": sha256(ranges_path),
        },
        "table_shapes": {
            "decisions": list(decisions.shape),
            "summary": list(summary.shape),
            "actual_ranges": list(ranges.shape),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--endpoint", type=Path, default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    report = finalize(args.protocol, args.results, args.endpoint)
    output = args.endpoint / "audit_receipt_2026-10-14.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
