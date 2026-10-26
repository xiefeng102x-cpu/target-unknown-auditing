# SRAC-UI against a stratified betting union-intersection test (post hoc, developmental)

Status: post hoc. All outcomes, the stored SRAC-UI, interval, and betting-audit decisions, and
the 2026-10-20 simulated review that asked for this comparison (R2 W1, M4) were known. No
model is trained. Before the plan and the script were hashed, the script was timed on a few
random samples of two confirmation victims (not audit draws) and its betting martingale was
checked against a reference implementation; those runs are discarded.

## Comparator

A fixed-budget test in the style of SWEETER (Spertus and Stark 2022): for each linear assertion
and direction, and for every point of the null boundary, each stratum (recall) or group cell
(clean-rate cap) receives a without-replacement betting test martingale (Waudby-Smith and
Ramdas) for its share of that null point, with the predictable aGRAPA bets of the betting
comparison audit already in the paper, applied in the order of the stored sample. The two
final values are multiplied; the assertion's null is rejected iff the minimum of the product
over the boundary is at least 1/alpha. Every null point is dominated by a boundary point and
the strata are sampled independently, so the size is at most alpha by Markov's inequality.
Exact computation over the whole boundary; no grid, no Berger-Boos restriction, no tuning.
The sequential (anytime, interleaved) form of SWEETER is not evaluated.

Same strata (flagged share 0.1), same samples and order, same assertions, same GO/NO-GO rule
as SRAC-UI. Allocations:

* sweeter: alpha = delta/(2A) per assertion and direction (SRAC-UI as reported);
* sweeter_dir: GO tests at delta/2, NO-GO tests at delta/(2A) (direction-specific, as in the
  direction-specific allocation analysis).

## Validation

Monte Carlo size check (20,000 draws per null boundary point; three small configurations,
two recall targets, three boundary points, both directions, alpha in {0.05, 0.2}); every
rejection rate must be at most alpha plus three Monte Carlo standard errors.

## Samples

SRAC-UI confirmation study, all 200 draws per budget; post-hoc GTSRB semantic-family and four
CIFAR-10 group-safe studies, draws 0-19 per budget (runtime). Budgets 2000/8000/16000.

## Reported

GO on passing and NO-GO on failing screens, false decisions, and per-decision runtime for
sweeter, sweeter_dir, SRAC-UI (stored, and direction-specific), and the betting audit (stored,
and direction-specific); contrasts SRAC-UI - sweeter at matched allocation and sweeter -
betting audit, with 95% cluster-bootstrap intervals over physical models (10,000 resamples,
seed 2026102101). Whatever the outcome, it is reported.
