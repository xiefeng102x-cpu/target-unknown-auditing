# Floating-point precision of the SRAC-UI engine at audited sizes (post hoc check)

Status: post hoc; raised by the 2026-10-20 simulated review (R1 W3, M5). No model is trained
and no stored decision is changed. Before the plan and the script were hashed, the script was
run once with one draw per budget and a cap of a few calls (environment switch FP_DRYRUN)
to check that it executes; that run is discarded.

## Claim under test

Engine v1.1 computes hypergeometric probabilities in double precision (log-gamma pmfs for the
union-intersection statistic; scipy.stats.hypergeom sf/cdf inside the exact count intervals,
including the Berger-Boos set) and protects every comparison with a relative margin of 1e-6.
The margin is sufficient if the computed probabilities near the decision thresholds have a
relative error far below 1e-6.

## Settings (the largest populations and smallest levels used)

* SRAC-UI confirmation study (GTSRB, N = 25,108; A = 11; alpha = delta/(2A)).
* CIFAR-10 SCAn+LDA study (N = 45,000; A = 9).
* Tenfold replication of the semantic-family populations (N = 251,080).
* A stress level alpha = delta/(2*44), the smallest level of any family in the paper.
* All three budgets, flagged share 0.1; draws 0-4 for the instrumented runs (first three
  victims of each setting); for stability, draws 0-19 of every victim, except draws 0-4 for
  the tenfold populations, whose decisions take about ten seconds each.

## Measurements

1. Every call of the engine's tail function (linear_p) whose returned probability lies in
   [1e-8, 1e-1] is recomputed in 40-digit arithmetic (mpmath) from the same arguments; report
   the maximum relative error, overall and for values within a factor 10 of the threshold.
2. Every call of scipy hypergeom sf/cdf made by the exact interval routine whose value lies in
   [1e-9, 1e-1] is recomputed in 40-digit arithmetic; report the maximum relative error.
3. Decision stability: decide the stability draws again with MARGIN = 0 and MARGIN = 1e-4 and
   count decisions that differ from MARGIN = 1e-6; MARGIN = 1e-6 must reproduce the stored
   decisions.

Up to 400 calls per setting and measurement are recomputed (the first ones met, in a fixed
order), because the 40-digit reference is slow. All numbers are reported.
