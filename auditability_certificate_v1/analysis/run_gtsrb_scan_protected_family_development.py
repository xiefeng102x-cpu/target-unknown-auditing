#!/usr/bin/env python3
"""Retrospective protected-group-family stress test on frozen GTSRB SCAn flags.

This analysis changes no victim, attack, detector flag, or poison identity.  It
reuses the disjoint confirmation handoff to measure the power cost of broader
predeclared group families.  Because the families were designed after the SCAn
confirmation was observed, every output is explicitly developmental and must
not be described as prospective confirmation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
INPUT = ROOT / "results/gtsrb_allclass_scan_scores_confirmation_v1/flags.csv"
OUTPUT = CERT / "results_gtsrb_scan_protected_family_development_v1"
CORE_PATH = CERT / "analysis/run_protected_group_family_srac.py"

SPEC = importlib.util.spec_from_file_location("protected_group_family_srac", CORE_PATH)
pf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pf
SPEC.loader.exec_module(pf)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_cohort(sample_id: str, modulus: int) -> int:
    digest = hashlib.sha256(f"srac-cohort-v1|{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def family_matrix(frame: pd.DataFrame, family: str) -> tuple[np.ndarray, list[str]]:
    labels = frame.assigned_group.to_numpy(dtype=int)
    classes = sorted(np.unique(labels).tolist())
    columns: list[np.ndarray] = []
    names: list[str] = []

    if family != "five_label_buckets":
        for label in classes:
            columns.append(labels == label)
            names.append(f"class_{label}")

    if family in {"five_label_buckets", "class_plus_five_label_buckets"}:
        # Numerical label buckets are a granularity stress test, not semantic
        # traffic-sign superclasses.
        buckets = np.minimum(labels // 9, 4)
        for bucket in range(5):
            mask = buckets == bucket
            if np.any(mask):
                columns.append(mask)
                names.append(f"label_bucket_{bucket}")

    if family == "class_plus_two_way_cohorts":
        cohorts = np.array(
            [stable_cohort(value, 2) for value in frame.sample_id.astype(str)], dtype=int
        )
        for label in classes:
            for cohort in range(2):
                mask = (labels == label) & (cohorts == cohort)
                if np.any(mask):
                    columns.append(mask)
                    names.append(f"class_{label}_cohort_{cohort}")

    if not columns:
        raise ValueError(f"unknown or empty family: {family}")
    return np.column_stack(columns), names


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    frame = frame[frame.victim_eligible == 1].copy()
    families = [
        "five_label_buckets",
        "class43",
        "class_plus_five_label_buckets",
        "class_plus_two_way_cohorts",
    ]
    outputs: list[pd.DataFrame] = []
    truth_rows: list[dict[str, object]] = []
    for victim_id, victim in frame.groupby("victim_id", sort=True):
        victim = victim.reset_index(drop=True)
        flags = victim.flag.to_numpy(dtype=np.int8)
        poison = victim.is_poison.to_numpy(dtype=bool)
        for family in families:
            memberships, names = family_matrix(victim, family)
            clean_rates = []
            for index in range(memberships.shape[1]):
                clean = (~poison) & memberships[:, index]
                if np.any(clean):
                    clean_rates.append(float(flags[clean].mean()))
            actual_recall = float(flags[poison].mean())
            actual_max_clean = max(clean_rates, default=0.0)
            truth_rows.append(
                {
                    "victim_id": victim_id,
                    "family": family,
                    "groups": memberships.shape[1],
                    "actual_recall": actual_recall,
                    "actual_max_group_clean_rate": actual_max_clean,
                    "true_policy_pass": int(actual_recall >= 0.8 and actual_max_clean <= 0.1),
                }
            )
            result = pf.audit_fixed_population_family(
                flags=flags,
                poison=poison,
                memberships=memberships,
                group_names=names,
                budgets=[2000, 8000, 16000],
                draws=50,
                delta=0.005,
                tau=0.8,
                alpha_cap=0.1,
                namespace=f"gtsrb-scan-family-dev-v1|{victim_id}|{family}",
                flagged_share=0.5,
            )
            result.insert(0, "family", family)
            result.insert(0, "victim_id", victim_id)
            outputs.append(result)

    decisions = pd.concat(outputs, ignore_index=True)
    truth = pd.DataFrame(truth_rows)
    summary = (
        decisions.groupby(["family", "k"], as_index=False)
        .agg(
            decisions=("decision", "size"),
            victims=("victim_id", "nunique"),
            groups_min=("groups", "min"),
            groups_max=("groups", "max"),
            go_rate=("decision", lambda x: float((x == "GO").mean())),
            nogo_rate=("decision", lambda x: float((x == "NO_GO").mean())),
            abstain_rate=("decision", lambda x: float((x == "ABSTAIN").mean())),
            false_go_count=("false_go", "sum"),
            false_nogo_count=("false_nogo", "sum"),
        )
    )
    decisions.to_csv(OUTPUT / "decisions.csv", index=False)
    truth.to_csv(OUTPUT / "truth_by_victim.csv", index=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    report = {
        "status": "DEVELOPMENT_ONLY_NOT_CONFIRMATORY",
        "study_id": "gtsrb_scan_protected_family_development_v1",
        "reason": "protected group families were designed after the SCAn confirmation was observed",
        "victims": int(truth.victim_id.nunique()),
        "families": families,
        "decision_rows": int(len(decisions)),
        "false_go_count": int(decisions.false_go.sum()),
        "false_nogo_count": int(decisions.false_nogo.sum()),
        "input_sha256": sha256(INPUT),
        "core_sha256": sha256(CORE_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "outputs": {
            "decisions_sha256": sha256(OUTPUT / "decisions.csv"),
            "truth_sha256": sha256(OUTPUT / "truth_by_victim.csv"),
            "summary_sha256": sha256(OUTPUT / "summary.csv"),
        },
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

