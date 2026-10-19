# Post-hoc human-adjudication design analysis V1 (developmental; not confirmation)

Written 2026-10-17, after the user chose human forensic review as the adjudication channel
(internal review ARS 2026-10-17, DA-C1 / R4). Question: which panel designs of human reviewers
give the robust group-safe certificate useful power, and at what review cost?

Model: r independent reviewers, majority vote; per-reviewer error e; a share c of records is "hard"
and receives one shared outcome (fully correlated errors). Effective per-record error
eps = c*e + (1-c)*P(majority wrong). Grid: e in {0.5%, 1%, 2%, 5%}, r in {1, 3, 5}, c in {0, 0.1}.
Designs with eps >= 0.005 are not simulated, because Table V (adjudication sensitivity V1) already
shows zero robust power at eps = 0.005 on both blocks; they are reported with that reference.

Certificate: the robust hull of `run_adjudication_sensitivity_v1.py` unchanged (delta/2 count
intervals, delta/2 error event, B = Binomial(k, eps) upper quantile), frozen flags, families, and
draws of blocks 8 and 12, k in {2000, 8000, 16000}, 200 draws. Cost: r*k reviews per audit.
Script: `analysis/run_human_adjudication_design_v1.py` (hash in the freeze receipt). All designs are
reported; nothing becomes a frozen endpoint. No reviewer error rate has been measured; e is a
planning parameter that a deployment must calibrate (gold items), not an empirical claim.
