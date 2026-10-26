# Direction-specific risk allocation (post hoc, developmental)

Status: post hoc. Every outcome of the studies below, all stored decisions, and the
2026-10-20 simulated review (R2 W2) that raised the question were known before this plan.
No model is trained; only frozen flags and stored draws are used. Before the plan and the
script were hashed, the script was timed on a few draws of one victim; those timing runs are
not part of the results.

## Question

A GO requires every GO-direction test to reject. If a population fails the policy, at least
one assertion is false, its GO-direction null is true, and a false GO requires that null to be
rejected. Each GO-direction test may therefore be run at level delta/2 without dividing by the
number of assertions A (intersection-union principle, Berger 1982); the false-GO probability
stays at most delta/2 for every fixed population. A NO-GO is issued when any NO-GO test rejects,
so the NO-GO direction keeps its union bound. How much GO power does this allocation add, for
SRAC-UI and for every comparator that admits it, on the very samples already decided?

## Engines (each: false GO <= delta/2, false NO-GO <= delta/2; same samples)

| name | GO direction | NO-GO direction |
|---|---|---|
| ui_full (stored) | each assertion alpha = delta/(2A) | alpha = delta/(2A) |
| ui_dir | each assertion alpha = delta/2 (gamma = 0.05 alpha) | delta/(2A) (unchanged) |
| srac (stored; frozen interval SRAC) | each cell, one tail, delta/(2Q) | each cell, one tail, delta/(2Q) |
| srac_dir | recall: two cells, one tail each, delta/4; each group: its two cells, delta/4 each | delta/(2Q) (unchanged) |
| interval_split (stored mechanism engine) | onesided_split at delta/2 | onesided_split at delta/2 |
| interval_split_dir | as srac_dir | onesided_split at delta/2 (unchanged) |
| rla_bet (stored) | each assertion alpha = delta/(2A) | delta/(2A) |
| rla_dir | each assertion alpha = delta/2 | delta/(2A) (unchanged) |

Uniform exact intervals and PPI are not re-run (PPI has no finite-sample guarantee).

## Sets and samples

* Post hoc (GTSRB semantic-family study and the four CIFAR-10 group-safe studies; blocks 8,
  11-14): the SRAC-UI samples with flagged share 0.1 (ui, interval_split), and the samples of
  the equal-budget comparison (srac with share 0.5; rla_bet on uniform samples).
* SRAC-UI confirmation (block 16): its srac_ui, srac, and rla_bet samples (same stable_seed keys).
* Budgets 2000/8000/16000, 200 draws.

## Validation gates (all must hold, otherwise no result is reported)

* The unchanged engines recomputed by the script reproduce every stored srac, rla_bet, and
  interval_split decision, and stored ui_full decisions are used for the NO-GO direction of
  ui_dir (the GO-direction result at delta/2 must contain every stored GO).
* False decisions are not a gate; they are counted and reported as found.

## Reported (all, without selection)

GO rate on passing screens and NO-GO rate on failing screens per engine, set, and budget;
false decisions; mean labels for the betting audit; paired contrasts dir - stored per engine,
and ui_dir - srac_dir, ui_dir - rla_dir at every budget, with 95% percentile cluster-bootstrap
intervals over physical models (10,000 resamples, seed 2026102001). The confirmation-study
endpoints stay as frozen; these numbers are post hoc.
