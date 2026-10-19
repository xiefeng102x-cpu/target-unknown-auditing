# Protected Group Family Extension V1

## Status

`DEVELOPMENT_ONLY_NOT_CONFIRMATORY`

This extension was designed after the disjoint GTSRB SCAn confirmation was
observed.  It broadens and tests the mathematical audit object.  It is not a
new prospective confirmation block and must not be described as one.

## Motivation

The first group-safe implementation assigns each record to one declared group.
That is sufficient for assigned-label classes, but a real policy may need to
protect overlapping sets such as class, data source, client, acquisition site,
or their intersections.  Reverting to a population-average clean flag rate is
not safe: a detector can destroy one protected set while retaining a small
global rate.

## General audit object

Let `H_1,...,H_J` be any finite family of nonempty subsets of the frozen audit
universe.  Sets may overlap, need not form a partition, and need not cover the
universe.  For each set, certify its realized clean flag rate

`F_H = clean flagged records in H / clean records in H`.

The GO policy requires global poison recall at least `tau` and `F_H` at most
`alpha_cap` for every declared protected set with at least one clean record.
The guarantee is deliberately limited to the declared family; it does not
claim safety for every unknown subset.

## Protected sampling construction

1. Freeze the universe, detector flags, protected sets, thresholds, allocation,
   and error budget.
2. Sample uniformly without replacement inside the two global flag strata.
3. Blindly adjudicate poison identity for sampled records.
4. For every nonempty `H x flag` cell, condition on how many sampled records
   fall in the cell.  The records inside that cell are then a uniform sample
   without replacement from the fixed cell.
5. Invert the hypergeometric law for the global flag cells and every protected
   group/flag cell.
6. Apply a union bound over all nonempty cells.  No independence between groups
   or intervals is assumed.

If there are `I` nontrivial count intervals, each interval receives per-tail
budget `eta = delta/(2I)`.  Deterministic composition gives simultaneous bounds
for global recall and every protected clean flag rate.  The resulting GO and
NO-GO error probabilities are at most `delta` on arbitrary fixed data, provided
the protected sampling and adjudication assumptions hold.

## Implementation and tests

- Core: `analysis/run_protected_group_family_srac.py`
- Unit tests: `tests/test_protected_group_family_srac.py`
- Development runner: `analysis/run_gtsrb_scan_protected_family_development.py`
- Development results: `results_gtsrb_scan_protected_family_development_v1/`

The new core and the two earlier group-safe implementations pass 13 tests.  The
tests include overlapping groups, a whole-group negative control, exact census
recovery, deterministic replay, and input validation.

## Development replay on frozen GTSRB SCAn flags

The runner reuses the ten disjoint SCAn confirmation victims and changes no
victim, attack, detector flag, or poison identity.  It compares four families:

| Family | Declared groups | GO at k=2,000 | GO at k=8,000 | GO at k=16,000 |
|---|---:|---:|---:|---:|
| Five numerical label buckets | 5 | 15.4% | 80.2% | 88.2% |
| Assigned classes | 43 | 6.4% | 80.0% | 86.4% |
| Classes plus five nested buckets | 48 | 7.8% | 80.4% | 84.6% |
| Classes plus two-way within-class cohorts | 129 | 2.2% | 79.0% | 83.8% |

Each cell contains 500 protected decisions (ten victims times fifty draws).
Across all 6,000 decisions, observed false-GO and false-NO-GO counts are zero.
These zero counts are descriptive, dependent replay outcomes rather than an
independent empirical failure-probability estimate.  Validity comes from the
simultaneous finite-population construction; the replay measures power.

The numerical label buckets and deterministic cohorts are granularity stress
tests, not semantic traffic-sign groups.  A future confirmation must freeze a
domain-grounded group family before victim training and protected draws.

## Evidence-gated interpretation

The extension removes the mathematical requirement that protected groups form
a single class partition.  It does not remove the need to predeclare which
groups matter.  Broader families remain statistically feasible at larger audit
budgets, while low-budget GO coverage declines.  This is the intended
security--power tradeoff, not evidence of universal low-cost auditing.

