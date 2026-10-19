from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "auditability_certificate_v1/analysis"
sys.path.insert(0, str(ANALYSIS))
SCRIPT = ANALYSIS / "robust_interval_hull.py"
SPEC = importlib.util.spec_from_file_location("robust_interval_hull", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from run_protected_group_family_srac import exact_hypergeom_count_interval


def test_zero_error_hull_equals_exact_interval() -> None:
    exact = exact_hypergeom_count_interval(40, 12, 4, 0.01)
    robust = MODULE.robust_hypergeom_count_hull(40, 12, 4, 0.01, 0)
    assert robust == exact


def test_hull_contains_every_feasible_correct_count_interval() -> None:
    hull = MODULE.robust_hypergeom_count_hull(60, 15, 7, 0.005, 3)
    for correct_count in range(4, 11):
        interval = exact_hypergeom_count_interval(60, 15, correct_count, 0.005)
        assert hull.lower <= interval.lower
        assert hull.upper >= interval.upper


def test_hull_can_only_widen_with_error_budget() -> None:
    intervals = [
        MODULE.robust_hypergeom_count_hull(80, 20, 6, 0.01, budget)
        for budget in range(5)
    ]
    for previous, current in zip(intervals, intervals[1:]):
        assert current.lower <= previous.lower
        assert current.upper >= previous.upper


@pytest.mark.parametrize("population,sampled", [(8, 3), (10, 5), (12, 7)])
def test_adversarial_corruption_failure_is_bounded_by_exact_interval_failure(
    population: int, sampled: int
) -> None:
    eta = 0.025
    error_budget = 1
    for poison_total in range(population + 1):
        support_low = max(0, sampled - (population - poison_total))
        support_high = min(sampled, poison_total)
        worst_case_failure = 0.0
        for correct_count in range(support_low, support_high + 1):
            probability = float(hypergeom.pmf(correct_count, population, poison_total, sampled))
            adversary_can_break = False
            for observed_count in range(
                max(0, correct_count - error_budget),
                min(sampled, correct_count + error_budget) + 1,
            ):
                hull = MODULE.robust_hypergeom_count_hull(
                    population, sampled, observed_count, eta, error_budget
                )
                if not (hull.lower <= poison_total <= hull.upper):
                    adversary_can_break = True
                    break
            if adversary_can_break:
                worst_case_failure += probability
        assert worst_case_failure <= 2 * eta + 1e-12


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, 0, 0.01, 0),
        (10, 11, 0, 0.01, 0),
        (10, 5, 6, 0.01, 0),
        (10, 5, 2, 0.01, -1),
    ],
)
def test_invalid_inputs_fail_closed(args: tuple[int, int, int, float, int]) -> None:
    with pytest.raises(ValueError):
        MODULE.robust_hypergeom_count_hull(*args)
