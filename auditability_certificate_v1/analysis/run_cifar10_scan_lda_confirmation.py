#!/usr/bin/env python3
"""Evaluate the frozen CIFAR-10 semantic overlapping-group confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
PROTOCOL_DEFAULT = CERT / "PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json"
SCORES_DEFAULT = ROOT / "results/cifar10_scan_lda_scores_confirmation_v1"
OUTPUT_DEFAULT = CERT / "results_cifar10_scan_lda_confirmation_v1"
CORE = CERT / "analysis/srac_family_core_scan_lda_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location("cifar10_scan_lda_srac_core", CORE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def membership_matrix(labels: np.ndarray, groups: list[dict]) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for row in groups:
        classes = np.asarray(row["class_ids"], dtype=int)
        if len(classes) == 0 or len(np.unique(classes)) != len(classes):
            raise ValueError(f"invalid class list for {row['name']}")
        if np.any((classes < 0) | (classes > 9)):
            raise ValueError(f"out-of-range class id for {row['name']}")
        mask = np.isin(labels, classes)
        if not np.any(mask):
            raise ValueError(f"empty protected group in population: {row['name']}")
        columns.append(mask)
        names.append(str(row["name"]))
    if len(set(names)) != len(names):
        raise ValueError("protected-group names must be unique")
    return np.column_stack(columns), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--scores", type=Path, default=SCORES_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    protocol_hash = sha256(args.protocol)
    if protocol_hash != sidecar.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("protocol sidecar mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    endpoint_hashes = {
        "endpoint_core_sha256": sha256(CORE),
        "endpoint_runner_sha256": sha256(Path(__file__)),
    }
    if protocol.get("hashes", {}).get("endpoint_stage") != endpoint_hashes:
        raise RuntimeError("endpoint-stage frozen hashes mismatch")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    manifest_path = args.scores / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "CIFAR10_SCAN_LDA_HANDOFF_COMPLETE":
        raise RuntimeError("score handoff incomplete")
    if manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("score handoff protocol mismatch")
    flags_path = args.scores / "flags.csv"
    if manifest.get("flags_sha256") != sha256(flags_path):
        raise RuntimeError("score table hash mismatch")

    core = load_core()
    flags = pd.read_csv(flags_path)
    eligible = flags[flags.victim_eligible.eq(1)].copy()
    outputs: list[pd.DataFrame] = []
    truth_rows: list[dict[str, object]] = []
    for victim_id, victim_frame in eligible.groupby("victim_id", sort=True):
        victim_frame = victim_frame.reset_index(drop=True)
        memberships, names = membership_matrix(
            victim_frame.assigned_group.to_numpy(dtype=int),
            protocol["protected_groups"],
        )
        poison = victim_frame.is_poison.to_numpy(dtype=bool)
        frozen_flags = victim_frame.flag.to_numpy(dtype=np.int8)
        clean_rates = [
            float(frozen_flags[(~poison) & memberships[:, index]].mean())
            for index in range(memberships.shape[1])
            if np.any((~poison) & memberships[:, index])
        ]
        actual_recall = float(frozen_flags[poison].mean())
        actual_max_clean = max(clean_rates, default=0.0)
        true_pass = bool(
            actual_recall >= float(protocol["certificate"]["recall_target"])
            and actual_max_clean
            <= float(protocol["certificate"]["max_group_clean_flag_rate_cap"])
        )
        truth_rows.append(
            {
                "victim_id": victim_id,
                "actual_recall": actual_recall,
                "actual_max_group_clean_rate": actual_max_clean,
                "true_policy_pass": int(true_pass),
            }
        )
        decisions = core.audit_fixed_population_family(
            flags=frozen_flags,
            poison=poison,
            memberships=memberships,
            group_names=names,
            budgets=protocol["certificate"]["budgets"],
            draws=int(protocol["certificate"]["protected_draws_per_seed_budget"]),
            delta=float(protocol["certificate"]["delta_per_decision"]),
            tau=float(protocol["certificate"]["recall_target"]),
            alpha_cap=float(protocol["certificate"]["max_group_clean_flag_rate_cap"]),
            namespace=f"cifar10-scan-lda-confirm-v1|{victim_id}",
            flagged_share=float(protocol["certificate"]["active_stratum_budget_share"]),
        )
        decisions.insert(0, "victim_id", victim_id)
        outputs.append(decisions)

    if not outputs:
        raise RuntimeError("no eligible victim reached the endpoint")
    decisions = pd.concat(outputs, ignore_index=True)
    truth = pd.DataFrame(truth_rows)
    summary = (
        decisions.groupby("k", as_index=False)
        .agg(
            decisions=("decision", "size"),
            victims=("victim_id", "nunique"),
            go_rate=("decision", lambda values: float((values == "GO").mean())),
            nogo_rate=("decision", lambda values: float((values == "NO_GO").mean())),
            abstain_rate=("decision", lambda values: float((values == "ABSTAIN").mean())),
            victims_with_go=(
                "victim_id",
                lambda values: int(
                    decisions.loc[values.index]
                    .query("decision == 'GO'")
                    .victim_id.nunique()
                ),
            ),
            false_go_count=("false_go", "sum"),
            false_nogo_count=("false_nogo", "sum"),
        )
    )
    args.output.mkdir(parents=True)
    decisions_path = args.output / "decisions.csv"
    truth_path = args.output / "truth_by_victim.csv"
    summary_path = args.output / "summary.csv"
    decisions.to_csv(decisions_path, index=False)
    truth.to_csv(truth_path, index=False)
    summary.to_csv(summary_path, index=False)

    gate = protocol["support_gate"]
    eligible_count = int(truth.victim_id.nunique())
    true_pass_count = int(truth.true_policy_pass.sum())
    by_k = summary.set_index("k")
    checks = {
        "minimum_eligible_victims": eligible_count >= int(gate["minimum_eligible_victims"]),
        "minimum_true_policy_pass_victims": true_pass_count
        >= int(gate["minimum_true_policy_pass_victims"]),
        "zero_false_go": int(decisions.false_go.sum()) == 0,
        "zero_false_nogo": int(decisions.false_nogo.sum()) == 0,
        "minimum_victims_with_go_at_k_2000": int(by_k.loc[2000, "victims_with_go"])
        >= int(gate["minimum_victims_with_go_at_k_2000"]),
        "minimum_victims_with_go_at_k_8000": int(by_k.loc[8000, "victims_with_go"])
        >= int(gate["minimum_victims_with_go_at_k_8000"]),
    }
    status = (
        "PASS_CIFAR10_SCAN_LDA_CONFIRMATION"
        if all(checks.values())
        else "FAIL_CIFAR10_SCAN_LDA_CONFIRMATION"
    )
    report = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": protocol["study_id"],
        "protocol_sha256": protocol_hash,
        "score_manifest_sha256": sha256(manifest_path),
        "flags_sha256": sha256(flags_path),
        "endpoint_core_sha256": sha256(CORE),
        "endpoint_runner_sha256": sha256(Path(__file__)),
        "eligible_victims": eligible_count,
        "true_policy_pass_victims": true_pass_count,
        "protected_group_count": len(protocol["protected_groups"]),
        "support_checks": checks,
        "outputs": {
            "decisions_sha256": sha256(decisions_path),
            "truth_sha256": sha256(truth_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
