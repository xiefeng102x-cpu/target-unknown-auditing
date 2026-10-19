#!/usr/bin/env python3
"""Bounded-error robust hull for exact finite-population count intervals.

This module is an executable counterpart of Proposition ``family-robust`` in
the manuscript.  It is intentionally separate from every frozen experiment
runner.  Adding it cannot change an already-frozen score or endpoint.
"""

from __future__ import annotations

from run_protected_group_family_srac import CountInterval, exact_hypergeom_count_interval


def robust_hypergeom_count_hull(
    population: int,
    sampled: int,
    observed_poison: int,
    eta: float,
    max_label_errors: int,
) -> CountInterval:
    """Return the union hull over all correct counts within the error budget.

    ``max_label_errors`` is a global upper bound on erroneous labels in the
    protected draw.  Consequently, the correct poison count in every cell is
    within that many units of the observed count, including when cells overlap.
    Using the same global bound in each cell is conservative but requires no
    independence assumption about errors or group memberships.
    """
    if population < 0 or sampled < 0 or sampled > population:
        raise ValueError("invalid population/sample sizes")
    if observed_poison < 0 or observed_poison > sampled:
        raise ValueError("invalid observed poison count")
    if max_label_errors < 0:
        raise ValueError("max_label_errors must be nonnegative")
    low = max(0, observed_poison - max_label_errors)
    high = min(sampled, observed_poison + max_label_errors)
    intervals = [
        exact_hypergeom_count_interval(population, sampled, correct_count, eta)
        for correct_count in range(low, high + 1)
    ]
    return CountInterval(
        lower=min(interval.lower for interval in intervals),
        upper=max(interval.upper for interval in intervals),
    )


__all__ = ["robust_hypergeom_count_hull"]
