#!/usr/bin/env python3
"""Exact SRAC for an arbitrary predeclared family of protected groups.

Groups may overlap and need not cover the population.  The protected draw is
stratified only by the frozen binary screen flag.  Conditional on the number
of sampled records falling in any fixed protected group, those records are a
uniform sample without replacement from that group/flag cell.  Exact
hypergeometric count intervals can therefore be union-bounded even when the
groups and their sampled records overlap.

This module is an audit primitive, not an experiment runner.  Poison labels are
read only after each protected draw to emulate independent adjudication.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Hashable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


@dataclass(frozen=True)
class CountInterval:
    lower: int
    upper: int


@dataclass(frozen=True)
class RateInterval:
    lower: float
    upper: float


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@lru_cache(maxsize=None)
def exact_hypergeom_count_interval(
    population: int,
    sampled: int,
    observed_poison: int,
    eta: float,
) -> CountInterval:
    """Two-sided exact interval with at most ``eta`` failure in each tail."""
    if population < 0 or sampled < 0 or sampled > population:
        raise ValueError("invalid population/sample sizes")
    if observed_poison < 0 or observed_poison > sampled:
        raise ValueError("invalid observed poison count")
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie in (0,1)")
    if sampled == 0:
        return CountInterval(0, population)
    if sampled == population:
        return CountInterval(observed_poison, observed_poison)

    feasible_low = observed_poison
    feasible_high = population - (sampled - observed_poison)

    lo, hi = feasible_low, feasible_high
    while lo < hi:
        mid = (lo + hi) // 2
        if float(hypergeom.sf(observed_poison - 1, population, mid, sampled)) > eta:
            hi = mid
        else:
            lo = mid + 1
    lower = lo

    lo, hi = feasible_low, feasible_high
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if float(hypergeom.cdf(observed_poison, population, mid, sampled)) > eta:
            lo = mid
        else:
            hi = mid - 1
    upper = lo
    if lower > upper:
        raise RuntimeError("empty exact confidence set")
    return CountInterval(lower, upper)


def _allocate_flag_budget(flags: np.ndarray, budget: int, flagged_share: float) -> tuple[int, int]:
    if not 0.0 <= flagged_share <= 1.0:
        raise ValueError("flagged_share must lie in [0,1]")
    if budget <= 0 or budget > len(flags):
        raise ValueError("invalid total budget")
    capacities = [int(np.sum(flags == 1)), int(np.sum(flags == 0))]
    flagged = min(capacities[0], int(round(budget * flagged_share)))
    unflagged = min(capacities[1], budget - flagged)
    if flagged + unflagged < budget:
        flagged = min(capacities[0], budget - unflagged)
    if flagged + unflagged < budget:
        unflagged = min(capacities[1], budget - flagged)
    if flagged + unflagged != budget:
        raise RuntimeError("failed to allocate complete budget")
    return flagged, unflagged


def _ratio(numerator: int, other: int, *, empty: float) -> float:
    denominator = numerator + other
    return numerator / denominator if denominator else empty


def _validate_inputs(
    flags: np.ndarray,
    poison: np.ndarray,
    memberships: np.ndarray,
    group_names: Sequence[Hashable],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Hashable, ...]]:
    flags = np.asarray(flags, dtype=np.int8)
    poison = np.asarray(poison, dtype=bool)
    memberships = np.asarray(memberships, dtype=bool)
    names = tuple(group_names)
    if flags.ndim != 1 or poison.ndim != 1 or memberships.ndim != 2:
        raise ValueError("flags/poison must be vectors and memberships a matrix")
    if len(flags) == 0 or len(flags) != len(poison) or memberships.shape[0] != len(flags):
        raise ValueError("population arrays must be nonempty and aligned")
    if memberships.shape[1] != len(names) or not names:
        raise ValueError("one unique name is required for every protected group")
    if len(set(names)) != len(names):
        raise ValueError("protected group names must be unique")
    if not set(np.unique(flags)).issubset({0, 1}):
        raise ValueError("flags must be binary")
    if np.any(memberships.sum(axis=0) == 0):
        raise ValueError("protected groups must be nonempty")
    return flags, poison, memberships, names


def audit_fixed_population_family(
    *,
    flags: np.ndarray,
    poison: np.ndarray,
    memberships: np.ndarray,
    group_names: Sequence[Hashable],
    budgets: Iterable[int],
    draws: int,
    delta: float,
    tau: float,
    alpha_cap: float,
    namespace: str,
    flagged_share: float = 0.5,
) -> pd.DataFrame:
    """Audit recall and every protected-group clean flag rate.

    ``memberships[i, j]`` declares whether record ``i`` belongs to protected
    group ``j``.  Memberships, flags, and the sampling allocation are frozen
    before the protected draw.  The same sample can support many overlapping
    groups; dependence is allowed because validity uses a union bound.
    """
    flags, poison, memberships, names = _validate_inputs(
        flags, poison, memberships, group_names
    )
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if not 0.0 <= tau <= 1.0 or not 0.0 <= alpha_cap <= 1.0:
        raise ValueError("rate thresholds must lie in [0,1]")

    # Two global flag cells identify recall.  Each nonempty group/flag cell
    # adds one interval for a protected clean-rate denominator or numerator.
    cells: list[tuple[int, int, np.ndarray]] = []
    for group_index in range(len(names)):
        for flag in (1, 0):
            indices = np.flatnonzero(memberships[:, group_index] & (flags == flag))
            if len(indices):
                cells.append((group_index, flag, indices))
    interval_count = 2 + len(cells)
    eta = delta / (2 * interval_count)

    actual_recall = float(flags[poison].mean()) if np.any(poison) else 1.0
    actual_group_clean: dict[Hashable, float] = {}
    for index, name in enumerate(names):
        clean = (~poison) & memberships[:, index]
        if np.any(clean):
            actual_group_clean[name] = float(flags[clean].mean())
    actual_max_group_clean = max(actual_group_clean.values(), default=0.0)
    true_pass = actual_recall >= tau and actual_max_group_clean <= alpha_cap

    flag_indices = {flag: np.flatnonzero(flags == flag) for flag in (1, 0)}
    output: list[dict[str, object]] = []
    for budget in map(int, budgets):
        flagged_sample, unflagged_sample = _allocate_flag_budget(
            flags, budget, flagged_share
        )
        allocation = {1: flagged_sample, 0: unflagged_sample}
        for draw in range(int(draws)):
            chosen: dict[int, np.ndarray] = {}
            global_bounds: dict[int, CountInterval] = {}
            for flag in (1, 0):
                rng = np.random.default_rng(stable_seed(namespace, budget, draw, flag))
                selected = rng.choice(
                    flag_indices[flag], size=allocation[flag], replace=False
                ) if allocation[flag] else np.empty(0, dtype=int)
                chosen[flag] = selected
                global_bounds[flag] = exact_hypergeom_count_interval(
                    len(flag_indices[flag]),
                    len(selected),
                    int(poison[selected].sum()),
                    eta,
                )

            cell_bounds: dict[tuple[int, int], CountInterval] = {}
            for group_index, flag, indices in cells:
                in_group = memberships[chosen[flag], group_index]
                sampled_indices = chosen[flag][in_group]
                cell_bounds[(group_index, flag)] = exact_hypergeom_count_interval(
                    len(indices),
                    len(sampled_indices),
                    int(poison[sampled_indices].sum()),
                    eta,
                )

            caught = global_bounds[1]
            missed = global_bounds[0]
            recall_lo = _ratio(caught.lower, missed.upper, empty=0.0)
            recall_hi = _ratio(caught.upper, missed.lower, empty=1.0)

            group_bounds: dict[Hashable, RateInterval] = {}
            for group_index, name in enumerate(names):
                flagged_size = int(np.sum(memberships[:, group_index] & (flags == 1)))
                unflagged_size = int(np.sum(memberships[:, group_index] & (flags == 0)))
                if flagged_size == 0:
                    group_bounds[name] = RateInterval(0.0, 0.0)
                    continue
                flagged_poison = cell_bounds[(group_index, 1)]
                unflagged_poison = cell_bounds.get(
                    (group_index, 0), CountInterval(0, 0)
                )
                clean_flagged_lo = flagged_size - flagged_poison.upper
                clean_flagged_hi = flagged_size - flagged_poison.lower
                clean_unflagged_lo = unflagged_size - unflagged_poison.upper
                clean_unflagged_hi = unflagged_size - unflagged_poison.lower
                lower = _ratio(clean_flagged_lo, clean_unflagged_hi, empty=0.0)
                upper = _ratio(clean_flagged_hi, clean_unflagged_lo, empty=1.0)
                group_bounds[name] = RateInterval(lower, upper)

            if recall_lo >= tau and all(
                bound.upper <= alpha_cap for bound in group_bounds.values()
            ):
                decision = "GO"
            elif recall_hi < tau or any(
                bound.lower > alpha_cap for bound in group_bounds.values()
            ):
                decision = "NO_GO"
            else:
                decision = "ABSTAIN"
            binding_name, binding_bound = max(
                group_bounds.items(), key=lambda item: item[1].upper
            )
            output.append(
                {
                    "k": budget,
                    "draw": draw,
                    "groups": len(names),
                    "intervals": interval_count,
                    "eta_per_tail": eta,
                    "flagged_sample": flagged_sample,
                    "unflagged_sample": unflagged_sample,
                    "observed_poison": int(
                        poison[chosen[1]].sum() + poison[chosen[0]].sum()
                    ),
                    "recall_lo": recall_lo,
                    "recall_hi": recall_hi,
                    "max_group_clean_lo": max(
                        bound.lower for bound in group_bounds.values()
                    ),
                    "max_group_clean_hi": binding_bound.upper,
                    "binding_group_hi": str(binding_name),
                    "actual_recall": actual_recall,
                    "actual_max_group_clean_rate": actual_max_group_clean,
                    "decision": decision,
                    "false_go": int(decision == "GO" and not true_pass),
                    "false_nogo": int(decision == "NO_GO" and true_pass),
                }
            )
    return pd.DataFrame(output)

