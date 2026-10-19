#!/usr/bin/env python3
"""Synthetic scaling check for the uniform-adjudication certificate."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_uniform_adjudication_v1" / "synthetic_scaling.csv"


def bounds(caught: int, poisons: int, flagged_clean: int, clean: int, delta: float) -> tuple[float, float, float]:
    def rad(n: int) -> float:
        return 1.0 if n == 0 else math.sqrt(math.log(3.0 / delta) / (2.0 * n))

    rhat = caught / poisons if poisons else 0.0
    fhat = flagged_clean / clean if clean else 0.0
    return max(0.0, rhat - rad(poisons)), min(1.0, rhat + rad(poisons)), min(1.0, fhat + rad(clean))


def main() -> None:
    rng = np.random.default_rng(20261013)
    n = 100_000
    # Match the deployed detector-fold operating point used by the real replay.
    delta = 0.005
    tau = 0.80
    alpha_cap = 0.10
    draws = 5_000
    rows: list[dict[str, object]] = []
    for prevalence in (0.01, 0.02, 0.05, 0.10, 0.25):
        n_poison = round(n * prevalence)
        n_clean = n - n_poison
        for expected_poison_audits in (50, 200, 800):
            k = min(n, math.ceil(expected_poison_audits / prevalence))
            for name, recall in (("failing", 0.70), ("adequate", 0.90)):
                caught_population = round(n_poison * recall)
                flagged_clean_population = round(n_clean * 0.04)
                decisions: list[str] = []
                false_go = 0
                false_nogo = 0
                sampled_poison_total = 0
                for _ in range(draws):
                    m_poison = int(rng.hypergeometric(n_poison, n_clean, k))
                    m_clean = k - m_poison
                    caught = int(rng.hypergeometric(caught_population, n_poison - caught_population, m_poison)) if m_poison else 0
                    flagged_clean = int(rng.hypergeometric(flagged_clean_population, n_clean - flagged_clean_population, m_clean)) if m_clean else 0
                    lo, hi, fhi = bounds(caught, m_poison, flagged_clean, m_clean, delta)
                    decision = "GO" if lo >= tau and fhi <= alpha_cap else "NO_GO" if hi < tau else "ABSTAIN"
                    decisions.append(decision)
                    false_go += int(decision == "GO" and recall < tau)
                    false_nogo += int(decision == "NO_GO" and recall >= tau)
                    sampled_poison_total += m_poison
                rows.append(
                    {
                        "prevalence": prevalence,
                        "expected_poison_audits": expected_poison_audits,
                        "k": k,
                        "sampling_fraction": k / n,
                        "sparse_regime": k / n <= 0.10,
                        "condition": name,
                        "true_recall": recall,
                        "mean_sampled_poison": sampled_poison_total / draws,
                        "go_rate": decisions.count("GO") / draws,
                        "nogo_rate": decisions.count("NO_GO") / draws,
                        "abstain_rate": decisions.count("ABSTAIN") / draws,
                        "false_go_rate": false_go / draws,
                        "false_nogo_rate": false_nogo / draws,
                    }
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
