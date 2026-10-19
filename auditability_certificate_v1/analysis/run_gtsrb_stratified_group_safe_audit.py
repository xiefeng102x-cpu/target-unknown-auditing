#!/usr/bin/env python3
"""Exploratory exact stratified group-safe audit on frozen GTSRB victims."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER14 = ROOT.parent
POLICY_PATH = ROOT / "analysis/run_stratified_group_safe_srac.py"
ADAPTER_PATH = ROOT / "analysis/run_gtsrb_group_safe_policy_audit.py"
INPUT = ROOT / "derived_inputs/gtsrb_global_blend_prefrozen_v1"
CACHE = PAPER14 / "data/gtsrb_official/gtsrb_32x32_uint8_v1.npz"
DEFAULT_OUTPUT = ROOT / "results_gtsrb_stratified_group_safe_audit_v1"
METHODS = ("cf_cmd_u", "crossfit_spectral", "target_centroid_distance", "flag_entire_target_class")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--active-share", type=float, default=0.5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    policy = load_module(POLICY_PATH, "stratified_group_safe_srac")
    adapter = load_module(ADAPTER_PATH, "gtsrb_group_adapter")
    with np.load(CACHE) as cache:
        train_labels = cache["train_labels"].copy()

    frames: list[pd.DataFrame] = []
    for seed in range(202, 212):
        population = adapter.population_for_seed(seed, train_labels)
        poison = population["is_poison"].to_numpy(dtype=np.int8).astype(bool)
        groups = population["group"].to_numpy(dtype=np.int64)
        for detector in METHODS:
            rows = policy.audit_fixed_population(
                flags=population[detector].to_numpy(dtype=np.int8),
                poison=poison,
                groups=groups,
                budgets=(2000, 8000, 16000),
                draws=args.draws,
                delta=0.005,
                tau=0.80,
                alpha_cap=0.10,
                namespace=f"gtsrb-exact-stratified-v1|seed={seed}|detector={detector}",
                active_share=args.active_share,
            )
            rows.insert(0, "detector", detector)
            rows.insert(0, "victim_id", f"gtsrb_patch_tinycnn_seed{seed}")
            frames.append(rows)
    decisions = pd.concat(frames, ignore_index=True)
    summary = (
        decisions.groupby(["detector", "k"], sort=True)
        .agg(
            decisions=("decision", "size"),
            victims=("victim_id", "nunique"),
            actual_recall_mean=("actual_recall", "mean"),
            actual_max_group_clean_rate_mean=("actual_max_group_clean_rate", "mean"),
            active_sample=("active_sample", "mean"),
            inactive_sample=("inactive_sample", "mean"),
            go_rate=("decision", lambda values: float((values == "GO").mean())),
            nogo_rate=("decision", lambda values: float((values == "NO_GO").mean())),
            abstain_rate=("decision", lambda values: float((values == "ABSTAIN").mean())),
            false_go_count=("false_go", "sum"),
            false_nogo_count=("false_nogo", "sum"),
        )
        .reset_index()
    )
    decisions_path = args.output / "decisions.csv"
    summary_path = args.output / "summary.csv"
    decisions.to_csv(decisions_path, index=False)
    summary.to_csv(summary_path, index=False)
    baseline = decisions.loc[decisions["detector"].eq("flag_entire_target_class")]
    useful = decisions.loc[~decisions["detector"].eq("flag_entire_target_class")]
    report = {
        "status": "COMPLETE_EXPLORATORY_EXACT_STRATIFIED_REPAIR",
        "prospective": False,
        "victims": int(decisions["victim_id"].nunique()),
        "decision_rows": int(len(decisions)),
        "active_share": args.active_share,
        "target_group_needed_by_certificate": False,
        "baseline_go_count": int(baseline["decision"].eq("GO").sum()),
        "nontrivial_go_count": int(useful["decision"].eq("GO").sum()),
        "false_go_count": int(decisions["false_go"].sum()),
        "false_nogo_count": int(decisions["false_nogo"].sum()),
        "interpretation": (
            "Post-hoc policy-repair evidence. Exact poison-count intervals are inverted "
            "inside pre-draw flag/group strata. A prospective freeze is required before "
            "using this result as confirmation evidence."
        ),
        "inputs": {
            "policy_sha256": sha256(POLICY_PATH),
            "adapter_sha256": sha256(ADAPTER_PATH),
            "truth_sha256": sha256(INPUT / "private_truth.csv"),
            "scores_sha256": sha256(INPUT / "blind_scores.csv"),
            "cache_sha256": sha256(CACHE),
        },
        "outputs": {
            "decisions_sha256": sha256(decisions_path),
            "summary_sha256": sha256(summary_path),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
