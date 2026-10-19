# Post-hoc audit-power study V1 (developmental; not confirmation)

Status: written on 2026-10-16 after the GTSRB semantic-overlap V2 and CIFAR-10
semantic-overlap V1 endpoints were observed. Every input is an immutable frozen flag
table. Results are descriptive power evidence and must never be pooled with a frozen
support gate or presented as prospective confirmation.

## Inputs (unchanged)
- results/gtsrb_semantic_overlap_scan_scores_confirmation_v2/flags.csv (10 victims)
- results/cifar10_semantic_overlap_scan_scores_confirmation_v1/flags.csv (5 victims)
- frozen protected families, tau=0.80, alpha_cap=0.10, delta=0.005 per decision,
  budgets 2000/8000/16000, 200 draws per victim and budget
- byte-identical frozen exact hypergeometric core (sha256 9171814b...)

## Validity of every compared rule
All rules use exact hypergeometric inversion within the two frozen flag strata and
their protected-set cells, with design-only allocations fixed before the draw.
- twosided: frozen rule; eta = delta/(2Q) on both tails of all Q intervals.
- onesided: per-error-type budgets; GO uses only GO-direction tails at eta = delta/Q,
  NO-GO uses the opposite tails at eta = delta/Q. False GO <= delta and false NO-GO
  <= delta separately (jointly <= 2 delta).
- onesided_split: as onesided, with delta/2 on the two recall cells (eta = delta/4)
  and delta/2 on the Q-2 group cells.
- Sampling share of the flagged stratum: 0.5 (frozen), 0.25, 0.1, proportional.
- Escalation: nested permutations per stratum; looks at 2000, 8000, 16000; each look
  uses onesided_split at delta/3 (Bonferroni over looks); stop at the first non-ABSTAIN.
  Reported for shares 0.5 and 0.25 with expected adjudications used.

## Validation
twosided + share 0.5 must reproduce both frozen decision tables exactly.

## Reporting rule
Report every method x share x budget cell and both escalation shares, including
P(GO | true pass), P(NO-GO | true fail), false GO, false NO-GO, abstention, and mean
adjudications used. No best cell is promoted. Any future endpoint change must be
frozen before new seeds are opened.
