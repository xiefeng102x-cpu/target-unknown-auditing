#!/usr/bin/env python3
"""Create a fail-closed audit receipt for the CIFAR-10/VGG16 SRAC block.

This is a reporting utility, not part of the frozen scientific endpoint.  It
refuses to summarize the block unless the structural verifier, independent
checkpoint/score replay, and frozen SRAC endpoint have all completed and
passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
DEFAULT_PROTOCOL = PAPER14 / "configs/cifar10_vgg16_bn_srac_confirmation_v1.json"
DEFAULT_RESULTS = PAPER14 / "results/cifar10_vgg16_bn_srac_confirmation_v1"
DEFAULT_ENDPOINT = ROOT / "results_cifar10_vgg16_srac_confirmation_v1"


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


def require_status(record: dict, expected: str, source: Path) -> None:
    if record.get("status") != expected:
        raise RuntimeError(
            f"{source} has status {record.get('status')!r}; expected {expected!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--endpoint", type=Path, default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    protocol = load_json(args.protocol)
    manifest_path = args.results / "manifest.json"
    handoff_path = args.results / "srac_handoff_verification.json"
    replay_path = args.results / "checkpoint_score_replay_verification.json"
    semantic_path = args.results / "checkpoint_score_semantic_addendum.json"
    endpoint_report_path = args.endpoint / "report.json"
    decisions_path = args.endpoint / "decisions.csv"
    summary_path = args.endpoint / "summary.csv"
    ranges_path = args.endpoint / "actual_ranges.csv"

    manifest = load_json(manifest_path)
    handoff = load_json(handoff_path)
    replay = load_json(replay_path)
    semantic = load_json(semantic_path)
    endpoint_report = load_json(endpoint_report_path)
    require_status(manifest, "SCORE_HANDOFF_COMPLETE", manifest_path)
    require_status(handoff, "PASS", handoff_path)
    if replay.get("status") == "PASS":
        replay_disposition = "BYTE_EXACT_PASS"
    elif (
        replay.get("status") == "FAIL"
        and semantic.get("status") == "PASS_SEMANTIC_ADDENDUM"
        and semantic.get("all_row_keys_exact") is True
        and semantic.get("all_conventional_flags_exact") is True
        and semantic.get("non_activation_clustering_score_arrays_byte_exact") is True
    ):
        replay_disposition = "PRE_FROZEN_BYTE_FAIL_POST_HOC_ENDPOINT_EQUIVALENT_PASS"
    else:
        raise RuntimeError("neither byte-exact nor endpoint-equivalent replay passed")
    require_status(endpoint_report, "COMPLETE_FROZEN_ENDPOINT", endpoint_report_path)

    frozen_seeds = [int(seed) for seed in protocol["formal_seeds"]]
    if manifest.get("completed_seeds") != frozen_seeds:
        raise RuntimeError("manifest does not contain the complete frozen seed block")
    if handoff.get("verified_seeds") != frozen_seeds or replay.get("verified_seeds") != frozen_seeds:
        raise RuntimeError("verification receipts do not cover every frozen seed")
    if not manifest.get("all_victims_eligible"):
        raise RuntimeError("at least one victim failed the predeclared eligibility gate")
    if endpoint_report.get("duplicate_decision_keys") != 0:
        raise RuntimeError("duplicate SRAC decision keys")

    receipts = load_json(args.results / "seed_receipts.json")
    if [int(row["seed"]) for row in receipts] != frozen_seeds:
        raise RuntimeError("seed receipt order or membership differs from the freeze")
    decisions = pd.read_csv(decisions_path)
    summary = pd.read_csv(summary_path)
    actual_ranges = pd.read_csv(ranges_path)
    expected_rows = (
        len(frozen_seeds)
        * int(protocol["crossfit"]["folds"])
        * len(protocol["detectors"])
        * len(protocol["srac_evaluation"]["audit_budgets"])
        * int(protocol["srac_evaluation"]["protected_draws_per_seed_fold_budget"])
    )
    if len(decisions) != expected_rows or endpoint_report.get("decision_rows") != expected_rows:
        raise RuntimeError("frozen endpoint produced an unexpected decision count")

    detector_summary: dict[str, dict[str, float | int]] = {}
    for detector, group in decisions.groupby("detector", sort=True):
        detector_summary[str(detector)] = {
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
    report = {
        "status": "PASS_WITH_DISCLOSED_POST_HOC_SEMANTIC_ADDENDUM",
        "scope": "locally pre-frozen CIFAR-10/VGG16-BN/BadNets confirmation and frozen whole-stratum SRAC endpoint",
        "chronology": {
            "confirmation_seeds": frozen_seeds,
            "confirmation_seeds_disjoint_from_disclosed_development_seeds": True,
            "protected_draws_started_after_local_freeze": True,
            "external_timestamp_or_independent_custodian": False,
            "prospective_scope": "new victims and protected draws only; attack family and earlier architecture-level outcomes were known",
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
        "audit_universe": {
            "definition": protocol["srac_evaluation"]["audit_universe"],
            "rows_per_victim": int(protocol["crossfit"]["expected_rows"]),
            "clean_rows_per_victim": int(protocol["crossfit"]["expected_clean_rows"]),
            "poison_rows_per_victim": int(protocol["crossfit"]["expected_poison_rows"]),
            "poison_prevalence": float(
                protocol["crossfit"]["expected_poison_rows"]
                / protocol["crossfit"]["expected_rows"]
            ),
        },
        "detectors": detector_summary,
        "srac_endpoint": {
            "decision_rows": int(len(decisions)),
            "false_go_count": int(decisions["false_go"].sum()),
            "false_nogo_count": int(decisions["false_nogo"].sum()),
            "by_budget": endpoint_report["by_budget"],
        },
        "verification": {
            "structural_handoff": handoff["status"],
            "checkpoint_and_score_replay": replay["status"],
            "replay_disposition": replay_disposition,
            "semantic_addendum": semantic["status"],
            "semantic_max_absolute_score_difference": max(
                row["max_absolute_score_difference"] for row in semantic["per_seed"]
            ),
            "semantic_conventional_flag_mismatches": sum(
                row["conventional_flag_mismatch_count"] for row in semantic["per_seed"]
            ),
            "replayed_score_rows": int(replay["replayed_score_rows"]),
            "compatibility_trusted_pool_used_by_srac": bool(
                replay["compatibility_trusted_pool_used_by_srac"]
            ),
        },
        "claim_boundary": "This block tests architecture transfer for BadNets. The pre-frozen byte-level replay failed only for sub-picounit Activation Clustering score drift; a post-hoc semantic addendum found identical row keys and detector flags. This does not establish byte-exact replay, cross-attack transfer, external preregistration, or independence among repeated draws.",
        "sha256": {
            "protocol": sha256(args.protocol),
            "training_manifest": sha256(manifest_path),
            "handoff_verification": sha256(handoff_path),
            "checkpoint_score_replay_verification": sha256(replay_path),
            "checkpoint_score_semantic_addendum": sha256(semantic_path),
            "endpoint_report": sha256(endpoint_report_path),
            "endpoint_decisions": sha256(decisions_path),
            "endpoint_summary": sha256(summary_path),
            "endpoint_actual_ranges": sha256(ranges_path),
        },
        "table_shapes": {
            "decisions": list(decisions.shape),
            "summary": list(summary.shape),
            "actual_ranges": list(actual_ranges.shape),
        },
    }
    output = args.endpoint / "audit_receipt_2026-10-13.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
