# Directional label-error budgets for a GO-or-ABSTAIN certificate (post hoc)

Status: post hoc. The automated-channel pilot (REFERENCE_CHANNEL_PLAN_V1.md) and all stored
decisions were known. Before this plan and the scripts were hashed: the exhaustive self-test of
srac_ui_directional_v1.py was run (7,551 observed/true pairs, 0 violations); and single
decisions on draw 0 of one confirmation victim were computed with the master labels and with
the channel's labels (oracle directional budgets), and the script was timed on draw 0 of one
victim (all settings), to check execution. Those runs are discarded.

## Construction

A GO is false only if a failing assertion's GO test rejects. Only two label-error directions
can make that test reject wrongly: a flagged clean record labelled poison (budget B1 in the
flagged sample) and an unflagged poison labelled clean (budget B0 in the unflagged sample).
srac_ui_directional_v1.py takes, for each GO test, the supremum of the engine's p-value over
every true count vector consistent with at most B1 and B0 such errors and any number of errors
in the other two directions. A GO-or-ABSTAIN certificate uses every GO test at delta/2
(intersection-union) and issues no NO-GO.

## Analyses (SRAC-UI confirmation study, flagged share 0.1, all 200 draws, all budgets)

A. Master labels, B1 = B0 = 0 (reference).
B. The automated channel's labels, with oracle directional budgets: per stratum, the upper
   delta_adj/4 hypergeometric quantile (delta_adj = 0.001) of the true number of errors of the
   bounded direction.
C. Synthetic channels (simulation of channel error rates; no human is modelled). Each sampled
   record's label is flipped independently, given its true class and stratum, so that the
   expected error counts per sampled record are: flagged clean->poison h1 (bounded),
   unflagged poison->clean g0 (bounded; 0.03% is the pilot channel's value), flagged
   poison->clean 1.1% (unbounded; the pilot channel's value), and unflagged clean->poison f0
   (unbounded, "false alarms"). Grid: f0 in {0, 0.1, 0.3, 1, 3}%, h1 in {0, 1, 3.5}%, g0 in {0, 0.03}%.
   B1 and B0 are the upper delta_adj/4 quantiles of Binomial(n1, h1) and Binomial(n0, g0).
   Flips use numpy.default_rng(stable_seed("directional-v1", victim, k, draw, f0, h1, g0)).

Reported: GO rate on the nine passing screens and false GO on the failing screen, per budget
and setting; whether the declared budgets covered the realized errors.

Whatever the outcome, it is reported.
