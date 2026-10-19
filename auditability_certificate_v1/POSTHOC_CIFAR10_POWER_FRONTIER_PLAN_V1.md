# Post-hoc CIFAR-10 audit-power frontier

Status: analysis code written while the formal score pipeline was still running,
after the seed-4411 SCAn recall (0.777778) had been observed. This is not a blinded
confirmation and must never be pooled with the frozen support gate.

## Fixed inputs

- the completed five-seed `flags.csv` from the semantic-overlap confirmation;
- the original frozen budgets, confidence level, recall target, clean-rate cap,
  eligible victims, and exact hypergeometric core;
- no retraining, rescoring, threshold change, detector selection, or victim replacement.

## Comparisons

For every victim and budget, report the full cross-product of:

- frozen 50/50 flagged-stratum allocation;
- 25/75 and 75/25 allocation sensitivity;
- allocation proportional to the realized frozen flag prevalence;
- one global group, all ten singleton assigned classes, and the frozen semantic
  overlapping family.

The same protected-draw namespace is reused across group families for a fixed
allocation, so differences across families reflect the policy family rather than
different sampled identities.

## Metrics

- audit fraction `k/N`;
- `P(GO | true policy pass)`;
- `P(NO-GO | true policy fail)`;
- false-GO and false-NO-GO rates conditional on truth;
- abstention rate and number of victims receiving at least one GO.

All allocations and families are reported. No best cell is promoted as a frozen
endpoint. The purpose is to measure the price of group safety and allocation choice,
not to repair or reinterpret the confirmation outcome.
