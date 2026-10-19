# Post-hoc same-budget audit baselines V1 (developmental)

Written 2026-10-18 in response to an internal review (item 1): the paper must show, at the same label
budget, how SRAC compares with existing audit designs that also give design-based guarantees for a
fixed population, so that its contribution is not reduced to "the population may be adversarial".

Object: the same group-safe certificate (recall >= 0.80 and clean flag rate <= 0.10 in every declared
group), delta = 0.005 per decision split over both directions, on the frozen flags and families of
blocks 8, 11, 12, 13, 14; budgets 2000/8000/16000; 200 draws per victim and budget.

Designs (script docstring has the formulas):
- srac: frozen endpoint (validation gate: reproduce every frozen decision, 39,000 in total).
- uni_exact: simple random sample of k from the universe, exact hypergeometric intervals for the same
  joint counts, eta = delta/(2Q) (TAR-style exact uniform audit).
- rla_bet: comparison RLA with SHANGRLA-style assertions (flags as reported outcomes), uniform sample
  without replacement, Waudby-Smith--Ramdas without-replacement betting test martingales with
  predictable aGRAPA bets, level delta/(2A) per assertion and direction, anytime stopping.
- ppi_clt: prediction-powered / difference estimator with flags as predictions, uniform sample,
  finite-population-corrected CLT intervals, Bonferroni delta/(2A). Asymptotic only.
Uniform designs share one uniform draw per (victim, k, draw) from namespace "audit-baselines-v1".

Reported: decision rates by policy truth, false GO / false NO-GO counts, mean labels used (rla_bet stops
early). Pre-run checks (no study data): a synthetic validity/power check of the betting test (null
rejection 0.028 and 0.005 at alpha 0.05; power 0.955 and 1.0) and a timing run that printed only
row counts and seconds. Nothing becomes a frozen endpoint; all designs and blocks are reported.
