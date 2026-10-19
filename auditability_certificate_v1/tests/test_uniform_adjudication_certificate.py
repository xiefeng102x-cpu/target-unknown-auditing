import importlib.util
import itertools
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "run_uniform_adjudication_certificate.py"
SPEC = importlib.util.spec_from_file_location("uniform_adjudication", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_empty_class_abstains_by_interval():
    assert MODULE.interval(0, 0, 0.05) == (0.0, 1.0)


def test_interval_contains_empirical_rate():
    lo, hi = MODULE.interval(90, 100, 0.05)
    assert 0.0 <= lo <= 0.9 <= hi <= 1.0


def test_more_adjudicated_records_shrink_radius():
    assert MODULE.radius(500, 0.05) < MODULE.radius(100, 0.05)


def test_decision_rule_covers_all_three_outputs():
    assert MODULE.decide(0.81, 0.95, 0.09, 0.80, 0.10) == "GO"
    assert MODULE.decide(0.10, 0.79, 0.01, 0.80, 0.10) == "NO_GO"
    assert MODULE.decide(0.79, 0.95, 0.09, 0.80, 0.10) == "ABSTAIN"
    assert MODULE.decide(0.81, 0.95, 0.11, 0.80, 0.10) == "ABSTAIN"


def test_bounded_label_error_bounds_cover_all_compatible_sample_rates():
    observed_labels = (1, 1, 0, 0, 0, 0)
    flags = (1, 0, 1, 0, 0, 0)
    b = 1
    observed_poison = sum(observed_labels)
    observed_clean = len(observed_labels) - observed_poison
    observed_caught = sum(z * a for z, a in zip(observed_labels, flags))
    observed_flagged_clean = sum((1 - z) * a for z, a in zip(observed_labels, flags))
    lo, hi, fhi = MODULE.robust_bounds(
        observed_poison,
        observed_caught,
        observed_clean,
        observed_flagged_clean,
        b,
        delta=0.05,
    )
    for flips in itertools.chain(((),), itertools.combinations(range(len(flags)), 1)):
        true_labels = list(observed_labels)
        for index in flips:
            true_labels[index] = 1 - true_labels[index]
        m1 = sum(true_labels)
        m0 = len(true_labels) - m1
        if m1:
            sample_recall = sum(z * a for z, a in zip(true_labels, flags)) / m1
            assert lo <= sample_recall + 1e-12
            assert sample_recall <= hi + 1e-12
        if m0:
            sample_clean_rate = sum((1 - z) * a for z, a in zip(true_labels, flags)) / m0
            assert sample_clean_rate <= fhi + 1e-12
