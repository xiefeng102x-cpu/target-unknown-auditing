# Post-hoc development of a flag-stratified union-intersection audit (SRAC-UI) V1

Written 2026-10-18 after the equal-budget comparison (AUDIT_BASELINES_PLAN_V1) showed that a uniform
betting comparison audit beats the frozen SRAC engine at k = 2000 (47.6% vs 16.8% GO on passing screens)
while SRAC wins at k >= 8000. Goal: an exact engine that combines flag stratification with a direct test
of each linear assertion. This is DEVELOPMENT on data whose outcomes under other engines are known; any
claim of superiority needs a later prospective confirmation on new victims.

Engine (script docstring has the formulas): each assertion (recall >= tau; F_H <= cap for every group with
a flagged record) is linear in two stratum counts. Statistic T = unbiased stratified estimate of the linear
form; exact null law = convolution of two hypergeometric laws; p = supremum over null counts consistent with
the sample, restricted to an exact 1-gamma confidence set for the nuisance count (Berger-Boos, gamma = 0.05
alpha added); branch and bound with monotone block bounds decides the supremum exactly. alpha = delta/(2A)
per assertion and direction; delta = 0.005.

Designs: srac (frozen; must reproduce 39,000 frozen decisions); srac_ui at flagged shares 0.5 (frozen SRAC
draws), 0.25 and 0.1 (fresh draws, namespace "srac-ui-v1"); srac_ui_esc: 2000 -> 8000 -> 16000, share 0.25,
alpha/3 per look. Blocks 8, 11, 12, 13, 14. All shares and designs reported; no selection among shares is
claimed as confirmatory.

Pre-run checks (no study data): exactness of the convolution against brute force (5 random cases, identical
to 1e-10); synthetic validity at the boundary (GO error 0.03, NO-GO error 0.0425 at alpha 0.05 before the
Berger-Boos restriction; after it, recall GO error 0.03-0.035 and NO-GO error 0.025-0.045, group GO error 0.015
and NO-GO error 0.025, all at alpha 0.05, 200 replicates); a synthetic power check at R = 0.90, k = 2000,
alpha = 0.0025 (GO: shares 0.5 / 0.25 / 0.1 give 9% / 19% / 33% for SRAC-UI, 9% / 11% / 15% for the SRAC
intervals, 25-30% for uniform RLA); timing on one victim per block (printed seconds only).
