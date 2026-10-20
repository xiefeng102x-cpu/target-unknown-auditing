# Post-hoc plan: SRAC-UI under label corruption, scaling in N, exact paired tests (V1)

Status: post hoc, developmental. Frozen (this file and the scripts below hash-bound in
UI_REDTEAM_SCALING_RECEIPT_V1.json) before the analyses were run. Disclosure: during
development, before this freeze, a timing run decided draws 0-1 of one block-8 and one
block-11 victim (part A), single draws of one block-8 victim at replication 1, 10, and 40
(part B, different draw seeds), and part C was computed once; the parameters below were then
fixed for run time only. No result becomes a
prospective endpoint. Motivation: pre-submission review (2026-10-20), items R1-W3, EIC-W2,
R1-W2.

## A. Label red team against SRAC-UI (analysis/run_ui_redteam_v1.py, analysis/srac_ui_robust_v1.py)
- Blocks 8, 11, 12, 13; frozen flags and families; SRAC-UI samples with flagged share 0.1
  (namespace "srac-ui-v1"), draws 0-19 per budget (k = 2000, 8000, 16000).
- Adversary and strategy set exactly as in run_groupsafe_redteam_v1.py (function attack),
  rho in {0.1%, 0.5%}, B = ceil(rho k).
- Certificates: naive SRAC-UI (B_cert = 0) and declared-B SRAC-UI (srac_ui_robust_v1, R = B),
  alpha = delta/(2A). The robust tests must pass their exhaustive self-test before the run.
- Validation: unattacked naive decisions reproduce the stored SRAC-UI v1.1 decisions.
- Reported: false GO (failing screens) and false NO-GO (passing screens) under attack for both
  certificates; GO of the declared-B certificate on passing screens with and without attack.
- Parts: (8, 11), (12), (13), combined afterwards.

## B. Scaling in N (analysis/run_ui_scaling_v1.py)
- Block-8 populations replicated 10 times (N about 2.5 x 10^5; prevalence, recall, and group
  clean rates unchanged); SRAC-UI v1.1, flagged share 0.1, 40 fresh draws per budget, same
  draw indices on the original populations.
- Reported: GO on passing and NO-GO on failing screens per budget and replication; false
  decisions.

## C. Exact paired tests for block 16 (same script)
- Nine passing screens, k = 2000, corrected decisions: exact one-sided sign test and exact
  paired sign-flip permutation test of SRAC-UI minus interval SRAC and minus the betting audit.
