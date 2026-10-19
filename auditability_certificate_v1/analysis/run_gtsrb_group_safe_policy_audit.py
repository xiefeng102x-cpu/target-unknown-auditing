#!/usr/bin/env python3
"""Apply group-safe SRAC to frozen GTSRB detectors and the target-class baseline."""

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
GROUP_SAFE_PATH = ROOT / "analysis/run_group_safe_srac.py"
INPUT = ROOT / "derived_inputs/gtsrb_global_blend_prefrozen_v1"
SOURCE = PAPER14 / "results/srac_gtsrb_global_blend_confirmation_v1"
CACHE = PAPER14 / "data/gtsrb_official/gtsrb_32x32_uint8_v1.npz"
DEFAULT_OUTPUT = ROOT / "results_gtsrb_group_safe_policy_audit_v1"
METHODS = ("cf_cmd_u", "crossfit_spectral", "target_centroid_distance")


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


def population_for_seed(seed: int, train_labels: np.ndarray) -> pd.DataFrame:
    victim = f"gtsrb_patch_tinycnn_seed{seed}"
    truth = pd.read_csv(INPUT / "private_truth.csv")
    truth = truth.loc[truth["victim_id"].eq(victim) & truth["role"].eq("train")].copy()
    source = pd.read_csv(SOURCE / f"per_sample/seed_{seed}_scores.csv")
    canonical = source.loc[source["method"].eq("cf_cmd_u")].copy()
    assigned = canonical.set_index("record_id")["assigned_label"].astype(int).to_dict()

    def group_for(sample_id: str) -> int:
        if sample_id.startswith("train_clean_"):
            return int(train_labels[int(sample_id.rsplit("_", 1)[1])])
        return int(assigned[sample_id])

    truth["group"] = truth["sample_id"].astype(str).map(group_for)
    if truth["group"].isna().any() or truth["group"].nunique() != 43:
        raise RuntimeError(f"incomplete assigned-label groups for seed {seed}")
    candidate_ids = set(canonical["record_id"].astype(str))
    truth["flag_entire_target_class"] = truth["sample_id"].astype(str).isin(candidate_ids).astype(int)
    scores = pd.read_csv(INPUT / "blind_scores.csv")
    scores = scores.loc[scores["victim_id"].eq(victim) & scores["role"].eq("train")]
    wide = scores.pivot(index="sample_id", columns="detector", values="conventional_flag")
    truth = truth.set_index("sample_id").join(wide[list(METHODS)], how="left")
    if truth[list(METHODS)].isna().any().any():
        raise RuntimeError(f"incomplete detector flags for seed {seed}")
    return truth.reset_index().sort_values("sample_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gs = load_module(GROUP_SAFE_PATH, "group_safe_srac")
    with np.load(CACHE) as cache:
        train_labels = cache["train_labels"].copy()

    frames: list[pd.DataFrame] = []
    detectors = (*METHODS, "flag_entire_target_class")
    for seed in range(202, 212):
        population = population_for_seed(seed, train_labels)
        poison = population["is_poison"].to_numpy(dtype=np.int8).astype(bool)
        groups = population["group"].to_numpy(dtype=np.int64)
        for detector in detectors:
            rows = gs.audit_fixed_population(
                flags=population[detector].to_numpy(dtype=np.int8),
                poison=poison,
                groups=groups,
                budgets=(2000, 8000, 16000),
                draws=200,
                delta=0.005,
                tau=0.80,
                alpha_cap=0.10,
                namespace=f"gtsrb-group-safe-v1|seed={seed}",
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
            mean_sampled_poison=("sampled_poison", "mean"),
            mean_min_sampled_clean_per_group=("min_sampled_clean_per_group", "mean"),
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
    report = {
        "status": "COMPLETE_EXPLORATORY_POLICY_REPAIR_AUDIT",
        "scope": "Post-hoc group-safe replay on ten frozen GTSRB global-blend victims",
        "prospective": False,
        "target_group_needed_by_certificate": False,
        "declared_groups": 43,
        "detectors": list(detectors),
        "decision_rows": int(len(decisions)),
        "false_go_count": int(decisions["false_go"].sum()),
        "false_nogo_count": int(decisions["false_nogo"].sum()),
        "baseline_rejected_without_false_go": bool(
            not decisions.loc[decisions["detector"].eq("flag_entire_target_class"), "decision"].eq("GO").any()
        ),
        "interpretation": (
            "Exploratory repair audit only. GO requires recall and every assigned-label "
            "clean-rate cap; NO-GO can be triggered by either failure. Coverage must be "
            "inspected before this policy replaces the frozen manuscript endpoint."
        ),
        "inputs": {
            "group_safe_core_sha256": sha256(GROUP_SAFE_PATH),
            "truth_sha256": sha256(INPUT / "private_truth.csv"),
            "scores_sha256": sha256(INPUT / "blind_scores.csv"),
            "gtsrb_cache_sha256": sha256(CACHE),
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
