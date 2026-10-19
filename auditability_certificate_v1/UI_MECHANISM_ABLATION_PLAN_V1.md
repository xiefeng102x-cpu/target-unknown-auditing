# SRAC-UI mechanism ablation and nearest-neighbour comparison (post hoc, developmental)

Status: post hoc. All outcomes of blocks 8, 11-14 and 16, the SRAC-UI v1.1 decisions, the block-16
same-share ablation, and the 2026-10-19 external re-evaluation were known before this plan.
No model is trained; only frozen flags and stored draws are used.

Question: on the very samples SRAC-UI v1.1 used (flagged share 0.1), which component of SRAC-UI
produces its power over interval SRAC, and how does it compare with a SUITE-style stratified
union-intersection test (Ottoboni et al. 2018) at the same risk?

Engines (all: false GO <= delta/2 and false NO-GO <= delta/2 per decision; same samples):
interval_frozen, interval_used, interval_split (interval SRAC, three error allocations);
ui_split (UI test, interval_split allocation: recall delta/4, groups delta/4 shared);
ui_nobb (UI allocation, no Berger-Boos restriction; block 16, draws 0-19 only, because the
full nuisance range is ~100x slower); ui_full (stored v1.1 decisions); suite (Fisher combination
of one-sided stratum p-values, maximized over the null boundary, alpha = delta/(2A), no
Berger-Boos; exact size verified on small populations by the script's self-test before freezing).

Sets: post hoc GTSRB (block 8), post hoc CIFAR-10 (blocks 11-14), block 16 (samples of the
prospective block; the analysis itself is post hoc).  Budgets 2000/8000/16000, 200 draws.

Reported, all without selection: GO rate on policy-passing screens, NO-GO rate on failing
screens, false decisions, for every engine; victim-level mean differences with 95% percentile
bootstrap intervals over victims (10,000 resamples, seed 2026101816) for
ui_split - interval_split (linear test), ui_full - ui_split (allocation), ui_full - ui_nobb
(Berger-Boos), ui_full - suite (exact convolution vs Fisher combination), ui_full - interval_split.

Not covered: sequential betting variants of stratified UI tests (Spertus and Stark 2022), which
are designed for sequential sampling; this analysis is at a fixed sample.
