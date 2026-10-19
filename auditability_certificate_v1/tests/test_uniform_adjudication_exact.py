import importlib.util
import itertools
import math
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "audit_uniform_adjudication_exact.py"
SPEC = importlib.util.spec_from_file_location("uniform_adjudication_exact", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_hypergeometric_recurrence_matches_combinatorial_definition():
    for population, successes, draws in ((10, 3, 4), (25, 7, 9), (100, 40, 60)):
        xs, probabilities = MODULE.hypergeom_distribution(population, successes, draws)
        direct = [
            math.comb(successes, x)
            * math.comb(population - successes, draws - x)
            / math.comb(population, draws)
            for x in xs
        ]
        assert abs(sum(probabilities) - 1.0) < 1e-12
        assert max(abs(left - right) for left, right in zip(probabilities, direct)) < 1e-12


def test_selected_invalid_population_stays_below_declared_error():
    go, nogo, _ = MODULE.exact_decision_probabilities(
        n=10_000,
        n_poison=500,
        caught_poison=399,
        flagged_clean=380,
        k=2_000,
        delta=0.005,
        tau=0.80,
        alpha_cap=0.10,
    )
    assert go < 0.005
    assert nogo < 0.005


def test_exact_probabilities_match_complete_sample_enumeration():
    # Four poison records (three caught) followed by four clean records (one flagged).
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    flags = [1, 1, 1, 0, 1, 0, 0, 0]
    k = 4
    delta = 0.20
    tau = 0.50
    alpha_cap = 0.70
    counts = {"GO": 0, "NO_GO": 0, "ABSTAIN": 0}
    for sample in itertools.combinations(range(len(labels)), k):
        m1 = sum(labels[index] for index in sample)
        m0 = k - m1
        c1 = sum(labels[index] * flags[index] for index in sample)
        c0 = sum((1 - labels[index]) * flags[index] for index in sample)
        if m1:
            eps1 = MODULE.radius(m1, delta)
            lo = max(0.0, c1 / m1 - eps1)
            hi = min(1.0, c1 / m1 + eps1)
        else:
            lo, hi = 0.0, 1.0
        fhi = min(1.0, c0 / m0 + MODULE.radius(m0, delta)) if m0 else 1.0
        decision = "GO" if lo >= tau and fhi <= alpha_cap else "NO_GO" if hi < tau else "ABSTAIN"
        counts[decision] += 1

    denominator = math.comb(len(labels), k)
    exact = MODULE.exact_decision_probabilities(
        n=8,
        n_poison=4,
        caught_poison=3,
        flagged_clean=1,
        k=k,
        delta=delta,
        tau=tau,
        alpha_cap=alpha_cap,
    )
    brute = tuple(counts[name] / denominator for name in ("GO", "NO_GO", "ABSTAIN"))
    assert max(abs(left - right) for left, right in zip(exact, brute)) < 1e-12
