#!/usr/bin/env python3
"""Exact flag/group-stratified group-safe SRAC.

Sampling strata depend only on public pre-draw group labels and frozen screen
flags.  Poison identities are used only after the protected sample is drawn.
Each stratum's unknown poison count receives an exact hypergeometric confidence
interval.  The intervals are composed into bounds on global poison recall and
the clean flag rate of every group touched by the screen.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Hashable, Iterable

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


@dataclass(frozen=True)
class Stratum:
    name: str
    indices: np.ndarray
    group: Hashable | None
    flag: int
    active: bool


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def exact_hypergeom_count_interval(
    population: int,
    sampled: int,
    observed_poison: int,
    eta: float,
) -> CountInterval:
    """Two-sided exact interval with at most eta failure in each tail."""
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

    # Small K is rejected when observing at least x poisons is too unlikely.
    lo, hi = feasible_low, feasible_high
    while lo < hi:
        mid = (lo + hi) // 2
        if float(hypergeom.sf(observed_poison - 1, population, mid, sampled)) > eta:
            hi = mid
        else:
            lo = mid + 1
    lower = lo

    # Large K is rejected when observing at most x poisons is too unlikely.
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


def build_strata(flags: np.ndarray, groups: np.ndarray) -> list[Stratum]:
    flags = np.asarray(flags, dtype=np.int8)
    groups = np.asarray(groups)
    active_groups = [
        group for group in sorted(pd.unique(groups).tolist(), key=str)
        if np.any((groups == group) & (flags == 1))
    ]
    strata: list[Stratum] = []
    for group in active_groups:
        for flag in (1, 0):
            indices = np.flatnonzero((groups == group) & (flags == flag))
            if len(indices):
                strata.append(Stratum(f"group={group}|flag={flag}", indices, group, flag, True))
    inactive = np.flatnonzero((flags == 0) & (~np.isin(groups, active_groups)))
    if len(inactive):
        strata.append(Stratum("inactive_groups|flag=0", inactive, None, 0, False))
    if not strata or sum(len(stratum.indices) for stratum in strata) != len(flags):
        raise RuntimeError("strata do not partition the population")
    return strata


def _capped_equal_allocation(total: int, capacities: list[int]) -> list[int]:
    allocation = [0] * len(capacities)
    remaining = total
    open_indices = {index for index, cap in enumerate(capacities) if cap > 0}
    while remaining and open_indices:
        share = max(1, remaining // len(open_indices))
        progressed = False
        for index in sorted(open_indices):
            room = capacities[index] - allocation[index]
            take = min(room, share, remaining)
            if take:
                allocation[index] += take
                remaining -= take
                progressed = True
            if allocation[index] == capacities[index]:
                open_indices.discard(index)
            if remaining == 0:
                break
        if not progressed:
            break
    if remaining:
        raise RuntimeError("allocation capacity exhausted")
    return allocation


def allocate_budget(strata: list[Stratum], budget: int, active_share: float = 0.5) -> list[int]:
    if budget <= 0 or budget > sum(len(stratum.indices) for stratum in strata):
        raise ValueError("invalid total budget")
    active_indices = [index for index, stratum in enumerate(strata) if stratum.active]
    inactive_indices = [index for index, stratum in enumerate(strata) if not stratum.active]
    active_capacity = sum(len(strata[index].indices) for index in active_indices)
    inactive_capacity = sum(len(strata[index].indices) for index in inactive_indices)
    active_target = min(active_capacity, int(round(budget * active_share)))
    inactive_target = min(inactive_capacity, budget - active_target)
    if active_target + inactive_target < budget:
        active_target = min(active_capacity, budget - inactive_target)
    if active_target + inactive_target < budget:
        inactive_target = min(inactive_capacity, budget - active_target)
    if active_target + inactive_target != budget:
        raise RuntimeError("failed to allocate complete budget")

    allocation = [0] * len(strata)
    active_alloc = _capped_equal_allocation(
        active_target, [len(strata[index].indices) for index in active_indices]
    )
    inactive_alloc = _capped_equal_allocation(
        inactive_target, [len(strata[index].indices) for index in inactive_indices]
    ) if inactive_indices else []
    for index, value in zip(active_indices, active_alloc):
        allocation[index] = value
    for index, value in zip(inactive_indices, inactive_alloc):
        allocation[index] = value
    return allocation


def _ratio(numerator: int, other: int, *, vacuous: tuple[float, float]) -> float:
    denominator = numerator + other
    if denominator <= 0:
        return vacuous[0]
    return numerator / denominator


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
    active_share: float = 0.5,
) -> pd.DataFrame:
    flags = np.asarray(flags, dtype=np.int8)
    poison = np.asarray(poison, dtype=bool)
    groups = np.asarray(groups)
    if not (flags.ndim == poison.ndim == groups.ndim == 1):
        raise ValueError("population arrays must be one-dimensional")
    if not (len(flags) == len(poison) == len(groups)) or len(flags) == 0:
        raise ValueError("population arrays must be nonempty and aligned")
    if not set(np.unique(flags)).issubset({0, 1}):
        raise ValueError("flags must be binary")
    strata = build_strata(flags, groups)
    # Two tails per unknown poison count; the union bound covers every stratum.
    eta = delta / (2 * len(strata))
    declared_groups = sorted(pd.unique(groups).tolist(), key=str)
    actual_recall = float(flags[poison].mean())
    actual_group_clean = {
        group: float(flags[(~poison) & (groups == group)].mean())
        for group in declared_groups if np.any((~poison) & (groups == group))
    }
    true_pass = actual_recall >= tau and max(actual_group_clean.values()) <= alpha_cap
    output: list[dict[str, object]] = []

    for budget in map(int, budgets):
        allocation = allocate_budget(strata, budget, active_share)
        for draw in range(int(draws)):
            intervals: dict[str, CountInterval] = {}
            observed: dict[str, int] = {}
            for stratum, sample_size in zip(strata, allocation):
                if sample_size:
                    rng = np.random.default_rng(
                        stable_seed(namespace, budget, draw, stratum.name)
                    )
                    chosen = rng.choice(stratum.indices, size=sample_size, replace=False)
                    poisoned = int(poison[chosen].sum())
                else:
                    poisoned = 0
                observed[stratum.name] = poisoned
                intervals[stratum.name] = exact_hypergeom_count_interval(
                    len(stratum.indices), sample_size, poisoned, eta
                )

            caught_lo = caught_hi = missed_lo = missed_hi = 0
            by_group_flag: dict[tuple[Hashable, int], tuple[int, CountInterval]] = {}
            for stratum in strata:
                bound = intervals[stratum.name]
                if stratum.flag:
                    caught_lo += bound.lower
                    caught_hi += bound.upper
                else:
                    missed_lo += bound.lower
                    missed_hi += bound.upper
                if stratum.group is not None:
                    by_group_flag[(stratum.group, stratum.flag)] = (
                        len(stratum.indices), bound
                    )

            recall_lo = _ratio(caught_lo, missed_hi, vacuous=(0.0, 1.0))
            recall_hi = _ratio(caught_hi, missed_lo, vacuous=(1.0, 1.0))
            group_bounds: dict[Hashable, RateInterval] = {}
            for group in declared_groups:
                flagged_size = int(np.sum((groups == group) & (flags == 1)))
                if flagged_size == 0:
                    group_bounds[group] = RateInterval(0.0, 0.0)
                    continue
                unflagged_size = int(np.sum((groups == group) & (flags == 0)))
                flagged_poison = by_group_flag[(group, 1)][1]
                unflagged_poison = by_group_flag.get(
                    (group, 0), (0, CountInterval(0, 0))
                )[1]
                clean_flagged_lo = flagged_size - flagged_poison.upper
                clean_flagged_hi = flagged_size - flagged_poison.lower
                clean_unflagged_lo = unflagged_size - unflagged_poison.upper
                clean_unflagged_hi = unflagged_size - unflagged_poison.lower
                lower_denominator = clean_flagged_lo + clean_unflagged_hi
                upper_denominator = clean_flagged_hi + clean_unflagged_lo
                lower = clean_flagged_lo / lower_denominator if lower_denominator else 0.0
                upper = clean_flagged_hi / upper_denominator if upper_denominator else 1.0
                group_bounds[group] = RateInterval(lower, upper)

            if recall_lo >= tau and all(bound.upper <= alpha_cap for bound in group_bounds.values()):
                decision = "GO"
            elif recall_hi < tau or any(bound.lower > alpha_cap for bound in group_bounds.values()):
                decision = "NO_GO"
            else:
                decision = "ABSTAIN"
            output.append(
                {
                    "k": budget,
                    "draw": draw,
                    "strata": len(strata),
                    "eta_per_tail": eta,
                    "active_sample": int(sum(value for value, stratum in zip(allocation, strata) if stratum.active)),
                    "inactive_sample": int(sum(value for value, stratum in zip(allocation, strata) if not stratum.active)),
                    "observed_poison": int(sum(observed.values())),
                    "recall_lo": recall_lo,
                    "recall_hi": recall_hi,
                    "max_group_clean_lo": max(bound.lower for bound in group_bounds.values()),
                    "max_group_clean_hi": max(bound.upper for bound in group_bounds.values()),
                    "actual_recall": actual_recall,
                    "actual_max_group_clean_rate": max(actual_group_clean.values()),
                    "decision": decision,
                    "false_go": int(decision == "GO" and not true_pass),
                    "false_nogo": int(decision == "NO_GO" and true_pass),
                }
            )
    return pd.DataFrame(output)
