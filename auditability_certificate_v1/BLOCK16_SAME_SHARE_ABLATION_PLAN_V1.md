# Block 16 same-share ablation (post hoc, developmental)

Status: post hoc. The block-16 outcomes and the 2026-10-18 external evaluation (which ran a
similar diagnostic on SRAC-UI v1.0 decisions) were known before this plan was written.

Question: on exactly the samples SRAC-UI drew in block 16 (flagged share 0.1), how much GO
power does the interval engine of SRAC reach under (i) the frozen error split, (ii) the split
over decision-relevant cells only, (iii) one-sided intervals with delta/2 per error direction
over those cells (the same risk as SRAC-UI)?  Compare each with SRAC-UI v1.1.

Fixed before running: script `analysis/run_block16_same_share_ablation_v1.py` (hash in
`BLOCK16_SAME_SHARE_ABLATION_RECEIPT_V1.json`); inputs: frozen block-16 protocol and flags,
`results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv`; statistics: victim-level mean
GO difference on the 9 passing screens with a 95% percentile bootstrap over victims (10,000
resamples, the frozen block-16 bootstrap seed); NO-GO rates on the failing screen.  All three
variants and all budgets are reported; none replaces a pre-declared endpoint.
