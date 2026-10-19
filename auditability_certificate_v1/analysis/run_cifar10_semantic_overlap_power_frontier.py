#!/usr/bin/env python3
"""Post-hoc audit-power frontier on frozen CIFAR semantic-overlap flags.

This analysis never retrains a victim or detector.  It compares allocation and
protected-family choices on one immutable flag/truth table.  It is descriptive
and must not be reported as part of the blinded confirmation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
PROTOCOL_DEFAULT = CERT / "PROTOCOL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION_V1.json"
SCORES_DEFAULT = ROOT / "results/cifar10_semantic_overlap_scan_scores_confirmation_v1"
OUTPUT_DEFAULT = CERT / "results_cifar10_semantic_overlap_power_frontier_v1"
CORE = CERT / "analysis/run_protected_group_family_srac.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location("posthoc_power_frontier_core", CORE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def family_definitions(protocol: dict) -> dict[str, list[dict]]:
    return {
        "global": [{"name": "all", "class_ids": list(range(10))}],
        "singleton_classes": [
            {"name": f"class_{label}", "class_ids": [label]} for label in range(10)
        ],
        "semantic_overlap": protocol["protected_groups"],
    }


def membership(labels: np.ndarray, groups: list[dict]) -> tuple[np.ndarray, list[str]]:
    columns = [np.isin(labels, np.asarray(row["class_ids"], dtype=int)) for row in groups]
    names = [str(row["name"]) for row in groups]
    if not columns or len(set(names)) != len(names):
        raise ValueError("invalid protected family")
    matrix = np.column_stack(columns)
    if np.any(matrix.sum(axis=0) == 0):
        raise ValueError("protected family contains an empty group")
    return matrix, names


def allocation_grid(flags: np.ndarray, frozen_share: float) -> list[tuple[str, float]]:
    proposed = [
        ("flag_share_0.25", 0.25),
        ("frozen_flag_share", float(frozen_share)),
        ("flag_share_0.75", 0.75),
        ("proportional_to_flag_prevalence", float(flags.mean())),
    ]
    output: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for name, share in proposed:
        key = (name, round(share, 15))
        if key not in seen:
            output.append((name, share))
            seen.add(key)
    return output


def evaluate(flags: pd.DataFrame, protocol: dict, *, draws: int | None = None) -> pd.DataFrame:
    core = load_core()
    draws = int(draws or protocol["certificate"]["protected_draws_per_seed_budget"])
    frames: list[pd.DataFrame] = []
    eligible = flags.loc[flags.victim_eligible.eq(1)].copy()
    for victim_id, victim_frame in eligible.groupby("victim_id", sort=True):
        victim_frame = victim_frame.reset_index(drop=True)
        frozen_flags = victim_frame.flag.to_numpy(dtype=np.int8)
        poison = victim_frame.is_poison.to_numpy(dtype=bool)
        labels = victim_frame.assigned_group.to_numpy(dtype=int)
        for allocation_name, share in allocation_grid(
            frozen_flags, float(protocol["certificate"]["active_stratum_budget_share"])
        ):
            namespace = f"cifar-semantic-posthoc-power-v1|{victim_id}|{allocation_name}"
            for family_name, groups in family_definitions(protocol).items():
                memberships, names = membership(labels, groups)
                decisions = core.audit_fixed_population_family(
                    flags=frozen_flags,
                    poison=poison,
                    memberships=memberships,
                    group_names=names,
                    budgets=protocol["certificate"]["budgets"],
                    draws=draws,
                    delta=float(protocol["certificate"]["delta_per_decision"]),
                    tau=float(protocol["certificate"]["recall_target"]),
                    alpha_cap=float(protocol["certificate"]["max_group_clean_flag_rate_cap"]),
                    namespace=namespace,
                    flagged_share=share,
                )
                decisions.insert(0, "family", family_name)
                decisions.insert(0, "flagged_share", share)
                decisions.insert(0, "allocation", allocation_name)
                decisions.insert(0, "victim_id", victim_id)
                decisions.insert(0, "analysis_status", "POSTHOC_AFTER_SEED_4411_SCORE")
                frames.append(decisions)
    if not frames:
        raise RuntimeError("no eligible victim in frozen score table")
    return pd.concat(frames, ignore_index=True)


def summarize(decisions: pd.DataFrame, population_rows: int) -> pd.DataFrame:
    keys = ["allocation", "flagged_share", "family", "k"]
    rows: list[dict[str, object]] = []
    for key, frame in decisions.groupby(keys, sort=True):
        true_pass = frame.actual_recall.ge(0.8) & frame.actual_max_group_clean_rate.le(0.1)
        passed = frame.loc[true_pass]
        failed = frame.loc[~true_pass]
        rows.append(
            {
                **dict(zip(keys, key)),
                "audit_fraction": float(key[-1] / population_rows),
                "decisions": int(len(frame)),
                "true_pass_decisions": int(len(passed)),
                "true_fail_decisions": int(len(failed)),
                "go_power_given_pass": float(passed.decision.eq("GO").mean()) if len(passed) else np.nan,
                "nogo_power_given_fail": float(failed.decision.eq("NO_GO").mean()) if len(failed) else np.nan,
                "false_go_rate_given_fail": float(failed.false_go.mean()) if len(failed) else np.nan,
                "false_nogo_rate_given_pass": float(passed.false_nogo.mean()) if len(passed) else np.nan,
                "abstain_rate": float(frame.decision.eq("ABSTAIN").mean()),
                "victims_with_go": int(frame.loc[frame.decision.eq("GO"), "victim_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--scores", type=Path, default=SCORES_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--draws", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol_hash = sha256(args.protocol)
    sidecar = args.protocol.with_suffix(args.protocol.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != protocol_hash:
        raise RuntimeError("protocol sidecar mismatch")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest_path = args.scores / "manifest.json"
    flags_path = args.scores / "flags.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("score/protocol mismatch")
    if manifest.get("flags_sha256") != sha256(flags_path):
        raise RuntimeError("score table hash mismatch")
    flags = pd.read_csv(flags_path)
    decisions = evaluate(flags, protocol, draws=args.draws)
    population_rows = int(flags.groupby("victim_id").size().iloc[0])
    summary = summarize(decisions, population_rows)
    args.output.mkdir(parents=True)
    decisions_path = args.output / "decisions.csv"
    summary_path = args.output / "summary.csv"
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    receipt = {
        "status": "POSTHOC_POWER_FRONTIER_COMPLETE",
        "disclosure": "Designed after observing the seed-4411 SCAn endpoint; not blinded confirmation evidence.",
        "protocol_sha256": protocol_hash,
        "score_manifest_sha256": sha256(manifest_path),
        "flags_sha256": sha256(flags_path),
        "core_sha256": sha256(CORE),
        "script_sha256": sha256(Path(__file__)),
        "decision_rows": len(decisions),
        "outputs": {
            "decisions_sha256": sha256(decisions_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
