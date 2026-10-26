# Two-stage adjudication for the directional certificate (post hoc, simulation)

Status: post hoc and simulated; no person or model labels anything. Written after the
directional-budget analysis (DIRECTIONAL_BUDGET_PLAN_V1.md), which showed that false alarms on
unflagged clean records, an error direction that cannot cause a false GO, decide the power of
the GO-or-ABSTAIN certificate. Before this plan and its script were hashed, the script was
timed on draw 0 of one confirmation victim (all settings); that run is discarded.

## Design simulated

Stage 1 labels every sampled record. Stage 2 re-examines only the unflagged records that
stage 1 called poison (for example a second independent reviewer, a provenance check, or a
reconstructed-trigger test); a record stays poison only if stage 2 agrees. Stage 2 can only
turn poison calls into clean calls, so its own mistakes on true poisons add to the bounded
unflagged poison->clean direction (budget B0) and never to an unbounded one.

Per sampled record, independently given its true class and stratum:
- flagged clean -> poison r_FC = 5% (bounded, B1); flagged poison -> clean r_FP = 1%;
- unflagged poison -> clean at stage 1 r_UP = 1%;
- unflagged clean -> poison at stage 1 f in {1, 2, 5}%;
- stage 2 keeps a stage-1 false alarm with probability q in {0.02, 0.05, 0.1, 0.25, 1}
  (q = 1: no stage 2, or a stage 2 that always repeats stage 1; q is the conditional error of
  stage 2 given a stage-1 error, so q above the stage-2 error rate models correlated errors);
- stage 2 wrongly clears a true poison that stage 1 called poison with probability
  m in {0, 2}%.
The rates are illustrative round numbers, not measurements.

Budgets: B1 and B0 are the upper delta_adj/4 quantiles (delta_adj = 0.001) of
Binomial(n1, r_FC * C1/N1) and Binomial(n0, u * M0/N0), with u = r_UP + (1 - r_UP) * m, the
final unflagged poison->clean rate. Certificate: srac_ui_directional_v1.decide_go, GO tests
at delta/2, on all 200 draws of each budget of the SRAC-UI confirmation study (flagged share
0.1). Flips use numpy.default_rng(stable_seed("two-stage-v1", victim, k, draw, f, q, m)).

## Reported

Per (f, q, m, k): GO rate on the nine passing screens, false GO on the failing screen, budget
coverage, and the stage-2 workload (records re-examined as a share of k). Whatever the
outcome, it is reported.
