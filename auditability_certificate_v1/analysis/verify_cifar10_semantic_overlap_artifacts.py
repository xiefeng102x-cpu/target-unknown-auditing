#!/usr/bin/env python3
"""Independently verify the completed CIFAR semantic-overlap artifact chain.

This verifier is deliberately outside the frozen decision implementation.  It
does not regenerate scores or decisions.  It checks hashes, row contracts,
truth calculations, error labels, summaries, support gates, and the reported
terminal status from the already-written artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
DEFAULT_PROTOCOL = CERT / "PROTOCOL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION_V1.json"
DEFAULT_VICTIMS = ROOT / "results/cifar10_vgg16_bn_blended_semantic_overlap_confirmation_v1"
DEFAULT_SCORES = ROOT / "results/cifar10_semantic_overlap_scan_scores_confirmation_v1"
DEFAULT_ENDPOINT = CERT / "results_cifar10_semantic_overlap_confirmation_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required artifact is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def group_membership(labels: np.ndarray, groups: list[dict]) -> np.ndarray:
    columns = [np.isin(labels, np.asarray(row["class_ids"], dtype=int)) for row in groups]
    require(bool(columns), "protected-group family is empty")
    return np.column_stack(columns)


def recompute_truth(flags: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = flags.loc[flags.victim_eligible.eq(1)].copy()
    for victim_id, frame in eligible.groupby("victim_id", sort=True):
        frame = frame.reset_index(drop=True)
        poison = frame.is_poison.to_numpy(dtype=bool)
        require(bool(np.any(poison)), f"eligible victim has no poison rows: {victim_id}")
        frozen_flags = frame.flag.to_numpy(dtype=np.int8)
        require(bool(np.isin(frozen_flags, [0, 1]).all()), f"nonbinary flag: {victim_id}")
        memberships = group_membership(
            frame.assigned_group.to_numpy(dtype=int), protocol["protected_groups"]
        )
        clean_rates: list[float] = []
        for index in range(memberships.shape[1]):
            clean = (~poison) & memberships[:, index]
            require(bool(np.any(clean)), f"protected group has no clean row: {victim_id}/{index}")
            clean_rates.append(float(frozen_flags[clean].mean()))
        recall = float(frozen_flags[poison].mean())
        max_clean = max(clean_rates)
        true_pass = bool(
            recall >= float(protocol["certificate"]["recall_target"])
            and max_clean
            <= float(protocol["certificate"]["max_group_clean_flag_rate_cap"])
        )
        rows.append(
            {
                "victim_id": victim_id,
                "actual_recall": recall,
                "actual_max_group_clean_rate": max_clean,
                "true_policy_pass": int(true_pass),
            }
        )
    require(bool(rows), "no eligible victims in score table")
    return pd.DataFrame(rows)


def compare_frame(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str], label: str) -> None:
    require(not actual.duplicated(keys).any(), f"duplicate keys in {label}")
    require(not expected.duplicated(keys).any(), f"duplicate recomputed keys in {label}")
    merged = actual.merge(expected, on=keys, suffixes=("_reported", "_recomputed"), validate="one_to_one")
    require(len(merged) == len(actual) == len(expected), f"key coverage mismatch in {label}")
    for column in expected.columns:
        if column in keys:
            continue
        left = merged[f"{column}_reported"]
        right = merged[f"{column}_recomputed"]
        if pd.api.types.is_numeric_dtype(right):
            require(
                bool(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=0, atol=1e-12)),
                f"numeric mismatch in {label}.{column}",
            )
        else:
            require(bool(left.astype(str).eq(right.astype(str)).all()), f"value mismatch in {label}.{column}")


def verify(protocol_path: Path, victims: Path, scores: Path, endpoint: Path) -> dict:
    protocol_hash = sha256(protocol_path)
    sidecar = protocol_path.with_suffix(protocol_path.suffix + ".sha256")
    require(sidecar.is_file(), "protocol sidecar is missing")
    require(sidecar.read_text(encoding="utf-8").split()[0] == protocol_hash, "protocol sidecar mismatch")
    protocol = load_json(protocol_path)

    victim_manifest_path = victims / "manifest.json"
    victim_receipts_path = victims / "seed_receipts.json"
    score_manifest_path = scores / "manifest.json"
    flags_path = scores / "flags.csv"
    report_path = endpoint / "report.json"
    decisions_path = endpoint / "decisions.csv"
    truth_path = endpoint / "truth_by_victim.csv"
    summary_path = endpoint / "summary.csv"

    victim_manifest = load_json(victim_manifest_path)
    victim_receipts = load_json(victim_receipts_path)
    score_manifest = load_json(score_manifest_path)
    report = load_json(report_path)

    require(victim_manifest.get("status") == "SCORE_HANDOFF_COMPLETE", "victim block incomplete")
    require(victim_manifest.get("freeze_manifest_sha256") == protocol_hash, "victim protocol mismatch")
    require(victim_manifest.get("completed_seeds") == protocol["formal_seeds"], "victim seed mismatch")
    require(victim_manifest.get("seed_receipts_sha256") == sha256(victim_receipts_path), "victim receipts hash mismatch")
    require([int(row["seed"]) for row in victim_receipts] == protocol["formal_seeds"], "victim receipt order mismatch")

    expected_score_status = (
        "CIFAR10_SCAN_LDA_HANDOFF_COMPLETE"
        if protocol.get("schema") == "cifar10-scan-lda-semantic-overlap-confirmation-v1"
        else "CIFAR10_SEMANTIC_OVERLAP_SCAN_HANDOFF_COMPLETE"
    )
    require(score_manifest.get("status") == expected_score_status, "score block incomplete")
    require(score_manifest.get("protocol_sha256") == protocol_hash, "score protocol mismatch")
    require(score_manifest.get("victim_manifest_sha256") == sha256(victim_manifest_path), "score-to-victim hash mismatch")
    require(score_manifest.get("flags_sha256") == sha256(flags_path), "score flags hash mismatch")

    require(report.get("protocol_sha256") == protocol_hash, "endpoint protocol mismatch")
    require(report.get("score_manifest_sha256") == sha256(score_manifest_path), "endpoint-to-score hash mismatch")
    require(report.get("flags_sha256") == sha256(flags_path), "endpoint-to-flags hash mismatch")
    output_hashes = report.get("outputs", {})
    require(output_hashes.get("decisions_sha256") == sha256(decisions_path), "decision hash mismatch")
    require(output_hashes.get("truth_sha256") == sha256(truth_path), "truth hash mismatch")
    require(output_hashes.get("summary_sha256") == sha256(summary_path), "summary hash mismatch")

    flags = pd.read_csv(flags_path)
    required_flag_columns = {
        "victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"
    }
    require(required_flag_columns.issubset(flags.columns), "score table schema mismatch")
    require(not flags.duplicated(["victim_id", "sample_id"]).any(), "duplicate score rows")
    require(bool(np.isin(flags.flag, [0, 1]).all()), "score flags are not binary")
    require(bool(np.isin(flags.is_poison, [0, 1]).all()), "poison truth is not binary")

    recomputed_truth = recompute_truth(flags, protocol)
    reported_truth = pd.read_csv(truth_path)
    compare_frame(reported_truth, recomputed_truth, ["victim_id"], "truth")
    require(int(report.get("eligible_victims", -1)) == len(recomputed_truth), "eligible-victim count mismatch")
    require(int(report.get("true_policy_pass_victims", -1)) == int(recomputed_truth.true_policy_pass.sum()), "true-pass count mismatch")

    decisions = pd.read_csv(decisions_path)
    keys = ["victim_id", "k", "draw"]
    require(not decisions.duplicated(keys).any(), "duplicate decision keys")
    eligible_ids = set(recomputed_truth.victim_id.astype(str))
    require(set(decisions.victim_id.astype(str)) == eligible_ids, "decision victim coverage mismatch")
    expected_budgets = set(map(int, protocol["certificate"]["budgets"]))
    require(set(decisions.k.astype(int)) == expected_budgets, "decision budget coverage mismatch")
    draws = int(protocol["certificate"]["protected_draws_per_seed_budget"])
    expected_draws = set(range(draws))
    for (_, _), frame in decisions.groupby(["victim_id", "k"], sort=False):
        require(set(frame.draw.astype(int)) == expected_draws, "decision draw coverage mismatch")

    truth_map = recomputed_truth.set_index("victim_id").true_policy_pass.astype(bool).to_dict()
    expected_false_go = decisions.apply(
        lambda row: int(row.decision == "GO" and not truth_map[str(row.victim_id)]), axis=1
    )
    expected_false_nogo = decisions.apply(
        lambda row: int(row.decision == "NO_GO" and truth_map[str(row.victim_id)]), axis=1
    )
    require(bool(decisions.false_go.astype(int).eq(expected_false_go).all()), "false-GO labels mismatch")
    require(bool(decisions.false_nogo.astype(int).eq(expected_false_nogo).all()), "false-NO-GO labels mismatch")

    recomputed_summary = (
        decisions.groupby("k", as_index=False)
        .agg(
            decisions=("decision", "size"),
            victims=("victim_id", "nunique"),
            go_rate=("decision", lambda x: float((x == "GO").mean())),
            nogo_rate=("decision", lambda x: float((x == "NO_GO").mean())),
            abstain_rate=("decision", lambda x: float((x == "ABSTAIN").mean())),
            false_go_count=("false_go", "sum"),
            false_nogo_count=("false_nogo", "sum"),
        )
    )
    victims_with_go = (
        decisions.loc[decisions.decision.eq("GO")]
        .groupby("k").victim_id.nunique()
        .reindex(recomputed_summary.k, fill_value=0)
        .to_numpy(dtype=int)
    )
    recomputed_summary.insert(5, "victims_with_go", victims_with_go)
    compare_frame(pd.read_csv(summary_path), recomputed_summary, ["k"], "summary")

    gate = protocol["support_gate"]
    by_k = recomputed_summary.set_index("k")
    require(8000 in by_k.index, "support-gate budget k=8000 is absent")
    checks = {
        "minimum_eligible_victims": len(recomputed_truth) >= int(gate["minimum_eligible_victims"]),
        "minimum_true_policy_pass_victims": int(recomputed_truth.true_policy_pass.sum())
        >= int(gate["minimum_true_policy_pass_victims"]),
        "zero_false_go": (not bool(gate.get("zero_false_go", True))) or int(decisions.false_go.sum()) == 0,
        "zero_false_nogo": (not bool(gate.get("zero_false_nogo", True))) or int(decisions.false_nogo.sum()) == 0,
        "minimum_victims_with_go_at_k_8000": int(by_k.loc[8000, "victims_with_go"])
        >= int(gate["minimum_victims_with_go_at_k_8000"]),
    }
    if "minimum_victims_with_go_at_k_2000" in gate:
        require(2000 in by_k.index, "support-gate budget k=2000 is absent")
        checks["minimum_victims_with_go_at_k_2000"] = int(
            by_k.loc[2000, "victims_with_go"]
        ) >= int(gate["minimum_victims_with_go_at_k_2000"])
    require(report.get("support_checks") == checks, "reported support checks mismatch")
    if protocol.get("schema") == "cifar10-scan-lda-semantic-overlap-confirmation-v1":
        expected_status = (
            "PASS_CIFAR10_SCAN_LDA_CONFIRMATION"
            if all(checks.values())
            else "FAIL_CIFAR10_SCAN_LDA_CONFIRMATION"
        )
    else:
        expected_status = (
            "PASS_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION"
            if all(checks.values())
            else "FAIL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION"
        )
    require(report.get("status") == expected_status, "terminal status mismatch")

    return {
        "status": "VERIFIED",
        "terminal_status": expected_status,
        "protocol_sha256": protocol_hash,
        "eligible_victims": len(recomputed_truth),
        "true_policy_pass_victims": int(recomputed_truth.true_policy_pass.sum()),
        "decision_rows": len(decisions),
        "false_go_count": int(decisions.false_go.sum()),
        "false_nogo_count": int(decisions.false_nogo.sum()),
        "support_checks": checks,
        "verified_sha256": {
            "victim_manifest": sha256(victim_manifest_path),
            "victim_receipts": sha256(victim_receipts_path),
            "score_manifest": sha256(score_manifest_path),
            "flags": sha256(flags_path),
            "endpoint_report": sha256(report_path),
            "decisions": sha256(decisions_path),
            "truth": sha256(truth_path),
            "summary": sha256(summary_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--victims", type=Path, default=DEFAULT_VICTIMS)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--endpoint", type=Path, default=DEFAULT_ENDPOINT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = verify(args.protocol, args.victims, args.scores, args.endpoint)
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
