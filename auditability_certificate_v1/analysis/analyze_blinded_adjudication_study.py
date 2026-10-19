#!/usr/bin/env python3
"""Privately score a completed blinded duplicate-adjudication packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binom


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cp_upper(errors: int, trials: int, delta: float) -> float:
    if trials <= 0:
        return 1.0
    if errors >= trials:
        return 1.0
    return float(beta.ppf(1.0 - delta, errors + 1, trials - errors))


def analyze(public: pd.DataFrame, key: pd.DataFrame, *, delta: float, audit_budgets: list[int]) -> dict:
    if not 0.0 < delta < 1.0:
        raise RuntimeError("delta must lie in (0,1)")
    # Bonferroni composition: first bound the unknown field error probability,
    # then bound the number of errors in a future audit draw.
    delta_rate = delta / 2.0
    delta_future_draw = delta / 2.0
    required_public = {"presentation_id", "adjudicated_is_poison", "adjudicator_id", "adjudicated_at"}
    required_key = {"presentation_id", "unit_token", "study_stratum", "gold_is_poison"}
    if not required_public.issubset(public.columns) or not required_key.issubset(key.columns):
        raise RuntimeError("input schema mismatch")
    if public["presentation_id"].duplicated().any() or key["presentation_id"].duplicated().any():
        raise RuntimeError("duplicate presentation id")
    joined = public.merge(key, on="presentation_id", how="inner", validate="one_to_one")
    if len(joined) != len(public) or len(joined) != len(key):
        raise RuntimeError("public/private packet mismatch")
    labels = pd.to_numeric(joined["adjudicated_is_poison"], errors="coerce")
    if labels.isna().any() or set(labels.astype(int).unique()) - {0, 1}:
        raise RuntimeError("every presentation requires a binary adjudication")
    if joined["adjudicator_id"].astype(str).str.strip().eq("").any():
        raise RuntimeError("adjudicator_id is required")
    if joined["adjudicated_at"].astype(str).str.strip().eq("").any():
        raise RuntimeError("adjudicated_at is required")
    joined["observed"] = labels.astype(int)
    joined["error"] = joined["observed"].ne(joined["gold_is_poison"].astype(int))

    duplicate_groups = joined.groupby("unit_token", sort=False).filter(lambda group: len(group) == 2)
    duplicate_pivot = duplicate_groups.groupby("unit_token")["observed"].agg(list)
    disagreements = int(sum(values[0] != values[1] for values in duplicate_pivot))
    duplicate_units = int(len(duplicate_pivot))

    strata: dict[str, dict] = {}
    for name, group in joined.groupby("study_stratum", sort=True):
        errors = int(group["error"].sum())
        trials = int(len(group))
        strata[str(name)] = {
            "presentations": trials,
            "errors": errors,
            "error_rate": errors / trials if trials else None,
            "one_sided_error_rate_upper": cp_upper(errors, trials, delta_rate),
        }

    # Use only uniformly sampled, first-presentation units for the field-mixture
    # error bound. Duplicate and challenge presentations are diagnostics.
    uniform = joined.loc[joined["study_stratum"].eq("uniform")].sort_values("presentation_id")
    uniform = uniform.drop_duplicates("unit_token", keep="first")
    field_errors = int(uniform["error"].sum())
    field_trials = int(len(uniform))
    error_upper = cp_upper(field_errors, field_trials, delta_rate)
    b_by_budget = {
        str(int(k)): int(binom.ppf(1.0 - delta_future_draw, int(k), error_upper))
        for k in audit_budgets
    }
    return {
        "status": "COMPLETE_PRIVATE_ADJUDICATION_AUDIT",
        "delta_total": delta,
        "delta_error_rate_bound": delta_rate,
        "delta_future_draw_bound": delta_future_draw,
        "uniform_unique_units": field_trials,
        "uniform_errors": field_errors,
        "uniform_error_rate": field_errors / field_trials if field_trials else None,
        "uniform_error_rate_one_sided_upper": error_upper,
        "declared_B_by_audit_budget_under_iid_error_model": b_by_budget,
        "duplicate_units": duplicate_units,
        "duplicate_disagreements": disagreements,
        "duplicate_disagreement_rate": disagreements / duplicate_units if duplicate_units else None,
        "strata": strata,
        "assumption": "B values require exchangeable independent adjudication errors after blinding; correlated or adversarial errors require a separate deterministic B or stronger controls.",
        "failure_budget_statement": "Bonferroni composition allocates half of delta to the field error-rate upper bound and half to the future audit-draw error count, so the declared B(k) fails with probability at most delta under the stated model.",
        "interpretation_guard": "Duplicate agreement measures repeatability, not correctness. Gold controls and the uniform gold audit measure correctness.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-public", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=float, default=0.005)
    parser.add_argument("--audit-budgets", type=int, nargs="+", default=[120, 500, 2000])
    args = parser.parse_args()
    report = analyze(
        pd.read_csv(args.completed_public),
        pd.read_csv(args.private_key),
        delta=args.delta,
        audit_budgets=args.audit_budgets,
    )
    report["sha256"] = {
        "completed_public": sha256(args.completed_public),
        "private_key": sha256(args.private_key),
        "analyzer": sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
