# Post-hoc used-cell error-budget analysis V1 (developmental; not confirmation)

Written 2026-10-17 in response to an internal review finding (ARS review, Phase 0.5 C5 / R1 W3):
the frozen family core divides delta over every non-empty protected-set/flag cell, including
cells of declared sets with no flagged record, which never enter a decision (F_H = 0 exactly).

Rule compared: Q_used = 2 + sum over declared H with N_H1 > 0 of (1 + 1[N_H0 > 0]). Q_used is a
function of the frozen flags and the declared family only, so it is fixed before audit randomness
and Proposition 3 (union bound over Q intervals) remains valid.

Script: `analysis/run_used_cell_budget_v1.py` (hash in the freeze receipt). Inputs: immutable frozen
flags, protocols, and draw namespaces of blocks 8, 11, and 12. Families: F0 (frozen) and F1 (F0 plus
every singleton class). Rules: "all" (frozen) and "used". Frozen design otherwise: two-sided exact
hypergeometric intervals, 50/50 flag allocation, k in {2000, 8000, 16000}, 200 draws, delta = 0.005,
tau = 0.8, cap = 0.1. F0/all must reproduce each frozen endpoint decision by decision; otherwise the
run is invalid. Reported: GO / NO-GO rates, false GO and false NO-GO counts, Q, per block, family,
rule, and budget. All cells are reported; nothing is promoted to a frozen endpoint. A future
confirmation block may adopt the used-cell rule only through a new frozen protocol.
