# GTSRB semantic-overlap confirmation — version 2

## Scope

This block tests the arbitrary protected-set extension with groups that have domain
meaning and were fixed before confirmation training. It does not test a second dataset,
architecture, attack, or detector. The freeze is locally hash-bound and has no external
timestamp.

## Freeze chronology

- Version-1 protocol SHA-256:
  `0c0545954fa2d04d7f2c1a4007d51aa7c61060f7eb2735ddb68ed62f3a309126`.
- Version 1 stopped at dataset-archive validation. The declared MD5 contained one
  transcribed character error. The already-declared SHA-256 matched the physical file.
- No confirmation seed, model training, detector scoring, or protected draw was opened
  under version 1. The fail-closed receipt is
  `GTSRB_SEMANTIC_OVERLAP_V1_FAILURE_RECEIPT.json`.
- Version-2 protocol SHA-256:
  `f4747bd77254b54bbb0b2ad20c2c315c2cbc0ee67ecbd59d3032ee8b99ae9635`.
- The version-2 protocol and sidecar predate the first runner event and seed 245 in the
  local filesystem/run log. See `GTSRB_SEMANTIC_OVERLAP_V2_FREEZE_RECEIPT.json`.

## Frozen design

- Dataset/model/attack: GTSRB, TinyCNN, full-frame global blend.
- Detector: pinned upstream BackdoorBench SCAn feature primitive.
- Confirmation seeds: 245--254, without replacement.
- Protected family: ten traffic-sign semantic sets. The family contains nested sets
  and overlaps that are not subset relations.
- Policy: global poison recall at least 0.80 and clean flag rate at most 0.10 in every
  protected set.
- Audit budgets: 2,000, 8,000, and 16,000.
- Protected draws: 200 per victim and budget.
- Per-decision error budget: 0.005.
- Endpoint: exact two-sided hypergeometric count inversion with a union-bound tail
  allocation over all count intervals.

## Results

All ten victims pass the frozen BCA/ASR eligibility gate. Nine screens satisfy the
realized joint policy. Seed 247 has recall 0.781802 and is a true policy failure.

| Audit size | Decisions | GO | NO-GO | ABSTAIN | Victims with GO | False GO | False NO-GO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 2,000 | 14.25% | 0.00% | 85.75% | 8 | 0 | 0 |
| 8,000 | 2,000 | 86.90% | 0.00% | 13.10% | 9 | 0 | 0 |
| 16,000 | 2,000 | 90.00% | 0.95% | 9.05% | 9 | 0 | 0 |

Seed 247 receives no GO at any budget. At `k=16000`, it receives 19 correct NO-GO
decisions and 181 abstentions. Every frozen support gate passes.

## Integrity bindings

- Score flags:
  `c37390f8e3252aebc67d97e49f9cc2ab7f237c6356ca3a0ae555d4d3511074e8`.
- Decisions:
  `8ae3b010a30464b0826d50ef0bcfca8ac1b6491415a9c79dd4bf43a5fcb8cd97`.
- Summary:
  `ffdad2a8578a2ca728fabdb05ac5aa191eea4157009eda1aed56978bb87c55cc`.
- Victim truth table:
  `41db076b817dee2822a3645df22705dfa6e4230619ee3f1326bc8b2a1c9ca8e2`.

The endpoint report is
`results_gtsrb_semantic_overlap_confirmation_v2/report.json`. Eleven focused tests
cover the freeze contract, semantic membership, overlapping-set endpoint, and exact
hypergeometric interval behavior.
