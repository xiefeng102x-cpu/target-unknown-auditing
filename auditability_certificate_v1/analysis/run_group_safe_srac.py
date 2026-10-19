#!/usr/bin/env python3
"""Target-unknown group-safe sparse random adjudication certificate.

The fixed-data audit object contains a known, pre-draw group label for every
record (for example, the assigned training label), a hidden poison identity,
and a frozen binary screen decision.  GO requires both adequate poison recall
and an acceptable clean flag rate in *every* declared group.  NO-GO can be
issued for inadequate recall or for a certified group-level burden violation.

The implementation uses simultaneous Hoeffding bounds for sampling without
replacement.  It is deliberately conservative and distribution free.  It
does not assume that the auditor knows the attack target group.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def simultaneous_tail_budget(delta: float, group_count: int) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if group_count <= 0:
        raise ValueError("at least one declared group is required")
    # Recall lower/upper plus clean-rate lower/upper for every group.
    return delta / (2 + 2 * group_count)


def hoeffding_interval(successes: int, total: int, eta: float) -> Interval:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("invalid Bernoulli counts")
    if total == 0:
        return Interval(0.0, 1.0)
    radius = math.sqrt(math.log(1.0 / eta) / (2.0 * total))
    center = successes / total
    return Interval(max(0.0, center - radius), min(1.0, center + radius))


def decide(
    recall: Interval,
    clean_by_group: dict[Hashable, Interval],
    tau: float,
    alpha_cap: float,
) -> str:
    if recall.lower >= tau and all(bound.upper <= alpha_cap for bound in clean_by_group.values()):
        return "GO"
    if recall.upper < tau or any(bound.lower > alpha_cap for bound in clean_by_group.values()):
        return "NO_GO"
    return "ABSTAIN"


def audit_fixed_population(
    *,
    flags: np.ndarray,
    poison: np.ndarray,
    groups: np.ndarray,
    budgets: Iterable[int],
    draws: int,
    delta: float,
    tau: float,
    alpha_cap: float,
    namespace: str,
) -> pd.DataFrame:
    flags = np.asarray(flags, dtype=np.int8)
    poison = np.asarray(poison, dtype=bool)
    groups = np.asarray(groups)
    if flags.ndim != 1 or poison.ndim != 1 or groups.ndim != 1:
        raise ValueError("flags, poison, and groups must be one-dimensional")
    if not (len(flags) == len(poison) == len(groups)):
        raise ValueError("population arrays must have equal length")
    if len(flags) == 0 or not set(np.unique(flags)).issubset({0, 1}):
        raise ValueError("flags must be a nonempty binary array")
    declared_groups = sorted(pd.unique(groups).tolist(), key=str)
    eta = simultaneous_tail_budget(delta, len(declared_groups))
    n_total = len(flags)
    n_poison = int(poison.sum())
    if n_poison == 0 or n_poison == n_total:
        raise ValueError("both clean and poison records are required")

    actual_recall = float(flags[poison].mean())
    actual_clean = {
        group: float(flags[(~poison) & (groups == group)].mean())
        if np.any((~poison) & (groups == group))
        else math.nan
        for group in declared_groups
    }
    group_sizes = {group: int(np.sum(groups == group)) for group in declared_groups}
    output: list[dict[str, object]] = []
    for budget in map(int, budgets):
        if budget <= 0 or budget > n_total:
            raise ValueError(f"invalid audit budget {budget} for population {n_total}")
        for draw in range(int(draws)):
            rng = np.random.default_rng(stable_seed(namespace, budget, draw))
            selected = rng.choice(n_total, size=budget, replace=False)
            selected_poison = poison[selected]
            sampled_poison = int(selected_poison.sum())
            poison_hits = int(flags[selected][selected_poison].sum())
            if budget == n_total:
                recall_bound = Interval(actual_recall, actual_recall)
            else:
                recall_bound = hoeffding_interval(poison_hits, sampled_poison, eta)

            group_bounds: dict[Hashable, Interval] = {}
            sampled_clean_by_group: dict[Hashable, int] = {}
            sampled_flagged_clean_by_group: dict[Hashable, int] = {}
            selected_groups = groups[selected]
            selected_flags = flags[selected]
            for group in declared_groups:
                group_selected = selected_groups == group
                group_clean = group_selected & (~selected_poison)
                clean_count = int(group_clean.sum())
                clean_hits = int(selected_flags[group_clean].sum())
                sampled_clean_by_group[group] = clean_count
                sampled_flagged_clean_by_group[group] = clean_hits
                if int(group_selected.sum()) == group_sizes[group]:
                    exact = actual_clean[group]
                    group_bounds[group] = Interval(exact, exact)
                else:
                    group_bounds[group] = hoeffding_interval(clean_hits, clean_count, eta)

            decision = decide(recall_bound, group_bounds, tau, alpha_cap)
            true_policy_pass = actual_recall >= tau and all(
                (not math.isnan(rate)) and rate <= alpha_cap for rate in actual_clean.values()
            )
            worst_group = max(
                declared_groups,
                key=lambda group: -math.inf if math.isnan(actual_clean[group]) else actual_clean[group],
            )
            output.append(
                {
                    "k": budget,
                    "draw": draw,
                    "n_total": n_total,
                    "n_poison": n_poison,
                    "group_count": len(declared_groups),
                    "eta_per_tail": eta,
                    "sampled_poison": sampled_poison,
                    "sampled_caught_poison": poison_hits,
                    "recall_lo": recall_bound.lower,
                    "recall_hi": recall_bound.upper,
                    "max_clean_rate_lo": max(bound.lower for bound in group_bounds.values()),
                    "max_clean_rate_hi": max(bound.upper for bound in group_bounds.values()),
                    "actual_recall": actual_recall,
                    "actual_max_group_clean_rate": actual_clean[worst_group],
                    "actual_worst_group": str(worst_group),
                    "min_sampled_clean_per_group": min(sampled_clean_by_group.values()),
                    "decision": decision,
                    "false_go": int(decision == "GO" and not true_policy_pass),
                    "false_nogo": int(decision == "NO_GO" and true_policy_pass),
                }
            )
    return pd.DataFrame(output)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("k", sort=True)
        .agg(
            decisions=("decision", "size"),
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
