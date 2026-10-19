#!/usr/bin/env python3
"""Exact finite-population error audit for the SRAC decision rule.

The calculation sums the hypergeometric laws for the sampled class count and
the two class-conditional flag counts.  It does not use Monte Carlo draws.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_uniform_adjudication_v1" / "exact_boundary_audit.csv"


def radius(n: int, delta: float) -> float:
    return 1.0 if n <= 0 else math.sqrt(math.log(3.0 / delta) / (2.0 * n))


def _logchoose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_distribution(population: int, successes: int, draws: int) -> tuple[list[int], list[float]]:
    """Exact hypergeometric PMF by stable recurrence around its mode."""
    lo = max(0, draws - (population - successes))
    hi = min(draws, successes)
    mode = min(hi, max(lo, math.floor((draws + 1) * (successes + 1) / (population + 2))))
    p_mode = math.exp(
        _logchoose(successes, mode)
        + _logchoose(population - successes, draws - mode)
        - _logchoose(population, draws)
    )
    probs: dict[int, float] = {mode: p_mode}
    p = p_mode
    for x in range(mode, hi):
        numerator = (successes - x) * (draws - x)
        denominator = (x + 1) * (population - successes - draws + x + 1)
        p = 0.0 if numerator == 0 else p * numerator / denominator
        probs[x + 1] = p
    p = p_mode
    for x in range(mode, lo, -1):
        numerator = x * (population - successes - draws + x)
        denominator = (successes - x + 1) * (draws - x + 1)
        p = 0.0 if numerator == 0 else p * numerator / denominator
        probs[x - 1] = p
    xs = list(range(lo, hi + 1))
    values = [probs[x] for x in xs]
    total = sum(values)
    return xs, [v / total for v in values]


def exact_decision_probabilities(
    n: int,
    n_poison: int,
    caught_poison: int,
    flagged_clean: int,
    k: int,
    delta: float,
    tau: float,
    alpha_cap: float,
) -> tuple[float, float, float]:
    """Return exact GO, NO-GO, and ABSTAIN probabilities."""
    n_clean = n - n_poison
    p_go = 0.0
    p_nogo = 0.0

    m_values, m_probs = hypergeom_distribution(n, n_poison, k)
    for m_poison, p_m in zip(m_values, m_probs):
        m_clean = k - m_poison

        # A zero class count makes the corresponding interval vacuous.
        if m_poison > 0:
            eps_r = radius(m_poison, delta)
            recall_values, recall_probs = hypergeom_distribution(
                n_poison, caught_poison, m_poison
            )
            p_recall_go = sum(
                probability
                for caught, probability in zip(recall_values, recall_probs)
                if caught / m_poison - eps_r >= tau
            )
            p_recall_nogo = sum(
                probability
                for caught, probability in zip(recall_values, recall_probs)
                if caught / m_poison + eps_r < tau
            )
        else:
            p_recall_go = 0.0
            p_recall_nogo = 0.0

        if m_clean > 0:
            eps_f = radius(m_clean, delta)
            clean_values, clean_probs = hypergeom_distribution(
                n_clean, flagged_clean, m_clean
            )
            p_clean_go = sum(
                probability
                for flagged, probability in zip(clean_values, clean_probs)
                if flagged / m_clean + eps_f <= alpha_cap
            )
        else:
            p_clean_go = 0.0

        p_go += p_m * p_recall_go * p_clean_go
        p_nogo += p_m * p_recall_nogo

    p_abstain = max(0.0, 1.0 - p_go - p_nogo)
    return p_go, p_nogo, p_abstain


def main() -> None:
    n = 10_000
    # Match the deployed detector-fold operating point used by the real replay.
    delta = 0.005
    tau = 0.80
    alpha_cap = 0.10
    rows: list[dict[str, float | int | str]] = []

    for prevalence in (0.01, 0.05, 0.25):
        n_poison = round(n * prevalence)
        n_clean = n - n_poison
        for k in (500, 2000, 5000):
            cases = {
                # One count below the policy boundary is the closest invalid
                # recall population at this finite N.
                "recall_below": (math.ceil(tau * n_poison) - 1, round(0.04 * n_clean)),
                # One count above the clean-rate cap is the closest invalid
                # FPR population.
                "fpr_above": (round(0.90 * n_poison), math.floor(alpha_cap * n_clean) + 1),
                # The closest adequate recall population audits false NO-GO.
                "recall_at_target": (math.ceil(tau * n_poison), round(0.04 * n_clean)),
            }
            for case, (caught, flagged_clean) in cases.items():
                p_go, p_nogo, p_abstain = exact_decision_probabilities(
                    n, n_poison, caught, flagged_clean, k, delta, tau, alpha_cap
                )
                recall = caught / n_poison
                fpr = flagged_clean / n_clean
                false_go = p_go if recall < tau or fpr > alpha_cap else 0.0
                false_nogo = p_nogo if recall >= tau else 0.0
                rows.append(
                    {
                        "case": case,
                        "n": n,
                        "prevalence": prevalence,
                        "k": k,
                        "actual_recall": recall,
                        "actual_clean_flag_rate": fpr,
                        "go_probability": p_go,
                        "nogo_probability": p_nogo,
                        "abstain_probability": p_abstain,
                        "false_go_probability": false_go,
                        "false_nogo_probability": false_nogo,
                        "declared_delta": delta,
                    }
                )

    result = pd.DataFrame(rows)
    if (result["false_go_probability"] > delta + 1e-12).any():
        raise AssertionError("Exact false-GO probability exceeds the declared budget")
    if (result["false_nogo_probability"] > delta + 1e-12).any():
        raise AssertionError("Exact false-NO-GO probability exceeds the declared budget")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
    print("max false GO", result["false_go_probability"].max())
    print("max false NO-GO", result["false_nogo_probability"].max())


if __name__ == "__main__":
    main()
