# Post-hoc family-resolution analysis V1 (developmental; not confirmation)

Written 2026-10-16, after the GTSRB semantic-overlap V2 (block 8) and CIFAR-10 SCAn+LDA
(block 12) endpoints were observed, in response to a review point: a group-safe GO
protects only declared groups, and with coarse semantic sets a small class can be
erased without violating any declared cap.

Script: `analysis/run_family_resolution_v1.py` (hash in the freeze receipt).
Inputs: the immutable frozen flags, protocols, and draw namespaces of blocks 8 and 12.
Frozen design kept: two-sided exact hypergeometric intervals, eta = delta/(2Q), 50/50
flag allocation, k in {2000, 8000, 16000}, delta = 0.005, tau = 0.8, cap = 0.1.

Families: F0 = frozen declared semantic family; F1 = F0 plus every singleton class.

- Part A: real screens, all 200 frozen draws, F0 and F1. F0 must reproduce both frozen
  endpoints decision by decision; otherwise the run is invalid.
- Part B: exact hidden-erasure capacity per victim and class (largest fraction of the
  class's clean records that can be additionally flagged with every F0 cap still met).
- Part C: stress screens for every class with hidden erasure >= 0.10, first m_c unflagged
  clean records in sha256(sample_id) order; F0 and F1; first 50 draw indices per budget.

Reported quantities: true policy pass under F0 and F1; GO / NO-GO / ABSTAIN rates; false
GO and false NO-GO counts; Q. All cells are reported; nothing becomes a frozen endpoint.
A timing check (10 decisions per family on one victim under a non-frozen namespace
"timing-only", decisions not inspected) and the count of classes with hidden erasure >= 0.10
on that victim (43) were observed before this plan was frozen.
