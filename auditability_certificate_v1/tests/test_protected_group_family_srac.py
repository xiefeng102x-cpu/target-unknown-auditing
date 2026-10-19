from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.stats import hypergeom


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analysis/run_protected_group_family_srac.py"
SPEC = importlib.util.spec_from_file_location("protected_group_family_srac", MODULE_PATH)
pf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pf
SPEC.loader.exec_module(pf)


def overlapping_memberships(size: int) -> tuple[np.ndarray, list[str]]:
    index = np.arange(size)
    memberships = np.column_stack(
        [
            index < (3 * size // 4),
            index >= (size // 4),
            (index % 2) == 0,
        ]
    )
    return memberships, ["left", "right", "even"]


def test_full_census_recovers_overlapping_group_quantities() -> None:
    size = 120
    memberships, names = overlapping_memberships(size)
    poison = np.zeros(size, dtype=bool)
    poison[::10] = True
    flags = np.zeros(size, dtype=np.int8)
    flags[poison] = 1
    flags[[1, 31, 61, 91]] = 1
    rows = pf.audit_fixed_population_family(
        flags=flags,
        poison=poison,
        memberships=memberships,
        group_names=names,
        budgets=[size],
        draws=1,
        delta=0.005,
        tau=0.8,
        alpha_cap=0.1,
        namespace="family-census",
    )
    row = rows.iloc[0]
    assert row.recall_lo == row.actual_recall == row.recall_hi
    assert row.max_group_clean_lo == row.actual_max_group_clean_rate
    assert row.max_group_clean_hi == row.actual_max_group_clean_rate
    assert row.decision == "GO"
    assert row.false_go == 0


def test_whole_protected_group_negative_control_is_rejected() -> None:
    size = 200
    memberships, names = overlapping_memberships(size)
    poison = np.zeros(size, dtype=bool)
    poison[150:] = True
    flags = memberships[:, 0].astype(np.int8)
    rows = pf.audit_fixed_population_family(
        flags=flags,
        poison=poison,
        memberships=memberships,
        group_names=names,
        budgets=[size],
        draws=1,
        delta=0.005,
        tau=0.8,
        alpha_cap=0.1,
        namespace="family-negative-control",
    )
    row = rows.iloc[0]
    assert row.actual_max_group_clean_rate == 1.0
    assert row.decision == "NO_GO"
    assert row.false_go == 0


def test_protected_family_draw_is_reproducible() -> None:
    size = 1000
    memberships, names = overlapping_memberships(size)
    poison = np.zeros(size, dtype=bool)
    poison[::20] = True
    flags = poison.astype(np.int8)
    kwargs = dict(
        flags=flags,
        poison=poison,
        memberships=memberships,
        group_names=names,
        budgets=[200],
        draws=5,
        delta=0.005,
        tau=0.8,
        alpha_cap=0.1,
        namespace="family-replay",
    )
    first = pf.audit_fixed_population_family(**kwargs)
    second = pf.audit_fixed_population_family(**kwargs)
    assert first.equals(second)


def test_group_names_must_be_unique() -> None:
    memberships, _ = overlapping_memberships(20)
    with np.testing.assert_raises(ValueError):
        pf.audit_fixed_population_family(
            flags=np.zeros(20, dtype=np.int8),
            poison=np.zeros(20, dtype=bool),
            memberships=memberships,
            group_names=["same", "same", "third"],
            budgets=[10],
            draws=1,
            delta=0.005,
            tau=0.8,
            alpha_cap=0.1,
            namespace="bad-names",
        )


def test_zero_poison_population_keeps_recall_vacuous_and_abstains() -> None:
    size = 80
    memberships, names = overlapping_memberships(size)
    rows = pf.audit_fixed_population_family(
        flags=np.zeros(size, dtype=np.int8),
        poison=np.zeros(size, dtype=bool),
        memberships=memberships,
        group_names=names,
        budgets=[size],
        draws=1,
        delta=0.005,
        tau=0.8,
        alpha_cap=0.1,
        namespace="zero-poison",
    )
    row = rows.iloc[0]
    assert not row.recall_defined
    assert np.isnan(row.actual_recall)
    assert row.recall_lo == 0.0
    assert row.recall_hi == 1.0
    assert row.decision == "ABSTAIN"


def test_copied_exact_interval_controls_each_tail_by_enumeration() -> None:
    population = 14
    sampled = 6
    eta = 0.025
    for true_poison in range(population + 1):
        lower_miss = 0.0
        upper_miss = 0.0
        for observed in range(sampled + 1):
            probability = float(
                hypergeom.pmf(observed, population, true_poison, sampled)
            )
            if probability == 0.0:
                continue
            interval = pf.exact_hypergeom_count_interval(
                population, sampled, observed, eta
            )
            lower_miss += probability * (interval.lower > true_poison)
            upper_miss += probability * (interval.upper < true_poison)
        assert lower_miss <= eta + 1e-12
        assert upper_miss <= eta + 1e-12
