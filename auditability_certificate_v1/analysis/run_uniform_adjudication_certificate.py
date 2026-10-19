#!/usr/bin/env python3
"""Replay a sparse-adjudication recall certificate on frozen score tables.

The audit sample is drawn uniformly from the *entire* declared fold.  An
independent adjudicator supplies poison/clean labels only for sampled records.
The score rule and its binary flag are fixed before the draw.  Conditional on
the number of sampled poison (clean) records, those records are a simple random
sample without replacement from the fixed poison (clean) population.  Hoeffding
bounds for sampling without replacement therefore give distribution-free
recall and clean-flag-rate intervals without a complete clean registry or a
poison-count declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_uniform_adjudication_v1"

BLOCKS = {
    "pathmnist_resnet18": ROOT.parent / "results" / "pathmnist_prospective_protocol_correct_v2_source",
    "pathmnist_resnet50": ROOT.parent / "results" / "pathmnist_blind_strong_resnet50_certificate_v1",
    "pathmnist_adaptive3200": ROOT.parent / "results" / "pathmnist_blind_adaptive_multiscore_certificate_v1",
    "pathmnist_certaware5300": ROOT.parent / "results" / "pathmnist_certaware_redteam_source_v2",
    "cifar10_vgg16": ROOT.parent / "results" / "cifar10_vgg16_bn_blind_certificate_v1",
    "gtsrb_tinycnn_prefrozen": ROOT / "derived_inputs" / "gtsrb_tinycnn_prefrozen_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def radius(n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    return math.sqrt(math.log(3.0 / delta) / (2.0 * n))


def interval(successes: int, total: int, delta: float) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    center = successes / total
    eps = radius(total, delta)
    return max(0.0, center - eps), min(1.0, center + eps)


def robust_bounds(
    observed_poison: int,
    observed_caught_poison: int,
    observed_clean: int,
    observed_flagged_clean: int,
    max_label_errors: int,
    delta: float,
) -> tuple[float, float, float]:
    """Conservative bounds when at most B sampled identity labels are wrong."""
    if max_label_errors < 0:
        raise ValueError("max_label_errors must be nonnegative")
    k = observed_poison + observed_clean
    b = max_label_errors
    if observed_poison <= b:
        recall_lo, recall_hi = 0.0, 1.0
    else:
        denominator_lo = observed_poison - b
        recall_lo = max(
            0.0,
            max(0, observed_caught_poison - b) / min(k, observed_poison + b)
            - radius(denominator_lo, delta),
        )
        recall_hi = min(
            1.0,
            min(k, observed_caught_poison + b) / denominator_lo
            + radius(denominator_lo, delta),
        )
    if observed_clean <= b:
        clean_hi = 1.0
    else:
        denominator_clean = observed_clean - b
        clean_hi = min(
            1.0,
            min(k, observed_flagged_clean + b) / denominator_clean
            + radius(denominator_clean, delta),
        )
    return recall_lo, recall_hi, clean_hi


def decide(recall_lo: float, recall_hi: float, clean_rate_hi: float, tau: float, alpha_cap: float) -> str:
    if recall_lo >= tau and clean_rate_hi <= alpha_cap:
        return "GO"
    if recall_hi < tau:
        return "NO_GO"
    return "ABSTAIN"


def load_block(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    score_path = path / "blind_scores.csv"
    truth_path = path / "private_truth.csv"
    scores = pd.read_csv(score_path)
    truth = pd.read_csv(truth_path)
    scores = scores.loc[scores["role"].eq("train")].copy()
    truth = truth.loc[truth["role"].eq("train")].copy()
    joined = scores.merge(
        truth[["victim_id", "sample_id", "role", "is_poison"]],
        on=["victim_id", "sample_id", "role"],
        how="left",
        validate="many_to_one",
    )
    if joined["is_poison"].isna().any():
        raise RuntimeError(f"Incomplete truth join in {path}")
    if "conventional_flag" not in joined:
        raise RuntimeError(f"Frozen conventional_flag is missing in {score_path}")
    hashes = {"scores": sha256(score_path), "truth": sha256(truth_path)}
    return joined, hashes


def audit_block(
    block: str,
    table: pd.DataFrame,
    budgets: tuple[int, ...],
    draws: int,
    delta_family: float,
    tau: float,
    alpha_cap: float,
) -> pd.DataFrame:
    detectors = sorted(table["detector"].unique())
    folds = sorted(map(int, table["fold"].unique()))
    # The original five-detector/two-fold family fixed 0.005 per decision.  Keep
    # that cap for smaller families so adding a block cannot silently relax the
    # deployed certificate.
    delta_cell = min(0.005, delta_family / (len(detectors) * len(folds)))
    output: list[dict[str, object]] = []

    for (victim, fold), group in table.groupby(["victim_id", "fold"], sort=True):
        wide = group.pivot(index="sample_id", columns="detector", values="conventional_flag")
        labels = group.drop_duplicates("sample_id").set_index("sample_id")["is_poison"]
        wide = wide.join(labels.rename("is_poison"), how="inner").sort_index()
        if wide[detectors].isna().any().any():
            raise RuntimeError(f"Detector coverage differs in {block}/{victim}/fold{fold}")
        flags = wide[detectors].to_numpy(np.int8)
        poison = wide["is_poison"].to_numpy(np.int8).astype(bool)
        n_total = len(wide)
        n_poison = int(poison.sum())
        n_clean = n_total - n_poison
        actual_recall = flags[poison].mean(axis=0) if n_poison else np.full(len(detectors), np.nan)
        actual_fpr = flags[~poison].mean(axis=0) if n_clean else np.full(len(detectors), np.nan)

        for k in budgets:
            if k > n_total:
                continue
            for draw in range(draws):
                rng = np.random.default_rng(stable_seed("uniform-adjudication-v1", block, victim, fold, k, draw))
                selected = rng.choice(n_total, size=k, replace=False)
                selected_poison = poison[selected]
                m_poison = int(selected_poison.sum())
                m_clean = k - m_poison
                selected_flags = flags[selected]
                poison_hits = selected_flags[selected_poison].sum(axis=0) if m_poison else np.zeros(len(detectors), dtype=int)
                clean_hits = selected_flags[~selected_poison].sum(axis=0) if m_clean else np.zeros(len(detectors), dtype=int)

                for d_index, detector in enumerate(detectors):
                    recall_lo, recall_hi = interval(int(poison_hits[d_index]), m_poison, delta_cell)
                    _, fpr_hi = interval(int(clean_hits[d_index]), m_clean, delta_cell)
                    decision = decide(recall_lo, recall_hi, fpr_hi, tau, alpha_cap)
                    false_go = decision == "GO" and (
                        float(actual_recall[d_index]) < tau or float(actual_fpr[d_index]) > alpha_cap
                    )
                    false_nogo = decision == "NO_GO" and float(actual_recall[d_index]) >= tau
                    output.append(
                        {
                            "block": block,
                            "victim_id": victim,
                            "fold": int(fold),
                            "detector": detector,
                            "k": k,
                            "draw": draw,
                            "n_total": n_total,
                            "n_poison": n_poison,
                            "poison_prevalence": n_poison / n_total,
                            "sampled_poison": m_poison,
                            "sampled_clean": m_clean,
                            "sampled_caught_poison": int(poison_hits[d_index]),
                            "sampled_flagged_clean": int(clean_hits[d_index]),
                            "recall_lo": recall_lo,
                            "recall_hi": recall_hi,
                            "clean_flag_rate_hi": fpr_hi,
                            "decision": decision,
                            "actual_recall": float(actual_recall[d_index]),
                            "actual_clean_flag_rate": float(actual_fpr[d_index]),
                            "false_go": int(false_go),
                            "false_nogo": int(false_nogo),
                            "delta_cell": delta_cell,
                        }
                    )
    return pd.DataFrame(output)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["block", "detector", "k"], sort=True)
        .agg(
            rows=("decision", "size"),
            victims=("victim_id", "nunique"),
            mean_prevalence=("poison_prevalence", "mean"),
            mean_sampled_poison=("sampled_poison", "mean"),
            mean_actual_recall=("actual_recall", "mean"),
            mean_actual_clean_flag_rate=("actual_clean_flag_rate", "mean"),
            go_rate=("decision", lambda x: float((x == "GO").mean())),
            nogo_rate=("decision", lambda x: float((x == "NO_GO").mean())),
            abstain_rate=("decision", lambda x: float((x == "ABSTAIN").mean())),
            false_go_count=("false_go", "sum"),
            false_nogo_count=("false_nogo", "sum"),
        )
        .reset_index()
    )
    summary["false_go_rate"] = summary["false_go_count"] / summary["rows"]
    summary["false_nogo_rate"] = summary["false_nogo_count"] / summary["rows"]
    return summary


def summarize_actual_ranges(rows: pd.DataFrame) -> pd.DataFrame:
    actual = rows[
        ["block", "victim_id", "fold", "detector", "actual_recall", "actual_clean_flag_rate"]
    ].drop_duplicates()
    return (
        actual.groupby(["block", "detector"], sort=True)
        .agg(
            victim_fold_rows=("actual_recall", "size"),
            recall_min=("actual_recall", "min"),
            recall_median=("actual_recall", "median"),
            recall_max=("actual_recall", "max"),
            clean_rate_min=("actual_clean_flag_rate", "min"),
            clean_rate_median=("actual_clean_flag_rate", "median"),
            clean_rate_max=("actual_clean_flag_rate", "max"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--draws", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    budgets = (120, 500, 2000)
    delta_family = 0.05
    tau = 0.80
    alpha_cap = 0.10
    frames: list[pd.DataFrame] = []
    source_hashes: dict[str, dict[str, str]] = {}
    for name, path in BLOCKS.items():
        table, hashes = load_block(path)
        frames.append(audit_block(name, table, budgets, args.draws, delta_family, tau, alpha_cap))
        source_hashes[name] = hashes

    rows = pd.concat(frames, ignore_index=True)
    summary = summarize(rows)
    actual_ranges = summarize_actual_ranges(rows)
    rows_path = args.output / "uniform_adjudication_decisions.csv"
    summary_path = args.output / "uniform_adjudication_summary.csv"
    ranges_path = args.output / "victim_fold_actual_ranges.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    actual_ranges.to_csv(ranges_path, index=False)

    primary = summary.loc[summary["k"].eq(2000)]
    report = {
        "status": "COMPLETE_CORRECTIVE_REPLAY",
        "protocol": "uniform whole-population draw; score-blind independent adjudication; fixed pre-draw flags",
        "scope": (
            "certificate replay over six immutable blocks; five victim blocks predate SRAC, "
            "while the GTSRB victim-generation protocol was frozen before confirmation training"
        ),
        "parameters": {
            "budgets": list(budgets),
            "draws_per_victim_fold_budget": args.draws,
            "delta_family": delta_family,
            "delta_per_decision_cap": 0.005,
            "delta_values_observed": sorted(map(float, rows["delta_cell"].unique())),
            "tau": tau,
            "alpha_cap": alpha_cap,
            "tail_allocation": "three simultaneous Hoeffding-without-replacement tails per detector-fold",
        },
        "primary_k2000": {
            "rows": int(primary["rows"].sum()),
            "go_rate": float((primary["go_rate"] * primary["rows"]).sum() / primary["rows"].sum()),
            "nogo_rate": float((primary["nogo_rate"] * primary["rows"]).sum() / primary["rows"].sum()),
            "abstain_rate": float((primary["abstain_rate"] * primary["rows"]).sum() / primary["rows"].sum()),
            "false_go_count": int(primary["false_go_count"].sum()),
            "false_nogo_count": int(primary["false_nogo_count"].sum()),
        },
        "source_hashes": source_hashes,
        "output_hashes": {
            "decisions": sha256(rows_path),
            "summary": sha256(summary_path),
            "victim_fold_actual_ranges": sha256(ranges_path),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
