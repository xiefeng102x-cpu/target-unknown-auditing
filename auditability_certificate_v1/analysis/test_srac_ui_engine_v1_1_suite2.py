#!/usr/bin/env python3
"""Second test suite for the SRAC-UI engine v1.1 (analysis/srac_ui_engine_v1_1.py).

The first suite (test_srac_ui_engine_v1_1.py) is kept unchanged as the record that was
hash-bound before the v1.1 recomputation.  This suite reuses its brute-force reference and
closes the gaps found in a later review:

  * T1/T2 no longer bypass the engine on the zero-observation outcome of the recall
    assertion (no sampled poison in either stratum).  The first suite filled that cell with
    (no GO, no NO-GO) without calling the engine, so an engine that answered GO there would
    still have passed.  Every outcome is now decided by the engine.
  * Risk budgets: alpha in {2e-4, 1e-3, 0.05, 0.2}.  The two small values cover the
    per-assertion levels used in the experiments (delta / (2A) with delta = 0.005).
  * T1b: an independent Berger-Boos check.  T1 lets the reference reuse the engine's
    nuisance range so that it isolates the supremum; T1b instead restricts the reference to
    the exact rational interval and requires every engine rejection to be a reference
    rejection (valid because the engine's interval contains the exact one).
  * T5 now asserts the exact expected decision of each regression case (derived in the
    comments below), not merely the absence of a NO-GO or an exception.
  * T6 (sensitivity): the suite is rerun on a deliberately broken engine that answers GO on
    the zero-observation outcome; the suite must detect it.
T3 and T4 are those of the first suite.  Exit status 0 iff all tests pass.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from fractions import Fraction as Fr
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import srac_ui_engine_v1_1 as E  # noqa: E402
import test_srac_ui_engine_v1_1 as T0  # noqa: E402  (first suite: reference and T3, T4)

ALPHAS = (2e-4, 1e-3, 0.05, 0.2)
THRESHOLDS = (("recall", (Fr(4, 5), Fr(2, 3), Fr(1, 2))), ("group", (Fr(1, 10), Fr(1, 4), Fr(1, 3))))


def exact_bb(kind, Na, na, xa, Nb, nb, xb, gamma):
    """Exact rational Berger-Boos range of the constrained count (independent of the engine)."""
    if kind == "recall":
        return T0.exact_interval(Na, na, xa, Fr(gamma) / 2) if 0 < na < Na else (0, Na)
    return T0.exact_interval(Nb, nb, xb, Fr(gamma) / 2) if 0 < nb < Nb else (0, Nb)


def t1_t2(summary, engine_call=T0.engine_call, configs=T0.CONFIGS, alphas=ALPHAS, thresholds=THRESHOLDS, tag=""):
    fails = []; n_cmp = n_band = n_zero = 0; worst = Fr(0)
    for kind, thrs in thresholds:
        for (Na, na, Nb, nb), thr, alpha in itertools.product(configs, thrs, alphas):
            ref = T0.Ref(kind, Na, na, Nb, nb, thr)
            gamma = float(E.GAMMA_SHARE) * alpha
            lim = Fr(alpha) - Fr(gamma); band = lim * (1 - Fr(E.MARGIN))
            dec = {}
            for xa in range(na + 1):
                for xb in range(nb + 1):
                    eg, en = engine_call(kind, Na, na, xa, Nb, nb, xb, thr, alpha)
                    n_zero += kind == "recall" and xa == 0 and xb == 0
                    # T1: same nuisance range as the engine (isolates the supremum)
                    rg, rn, pg, pn = ref.decide(xa, xb, alpha, Fr(gamma), T0.engine_bb(kind, Na, na, xa, Nb, nb, xb, gamma))
                    n_cmp += 1
                    for e, r, p, name in ((eg, rg, pg, "GO"), (en, rn, pn, "NO_GO")):
                        if e != r:
                            if r and not e and band < p <= lim:
                                n_band += 1
                            else:
                                fails.append(dict(test="T1" + tag, kind=kind, cfg=(Na, na, Nb, nb), thr=str(thr), alpha=alpha,
                                                  x=(xa, xb), direction=name, engine=e, reference=r, p=float(p)))
                    # T1b: exact rational Berger-Boos range; engine rejection => reference rejection
                    ig, i_n, _, _ = ref.decide(xa, xb, alpha, Fr(gamma), exact_bb(kind, Na, na, xa, Nb, nb, xb, gamma))
                    for e, r, name in ((eg, ig, "GO"), (en, i_n, "NO_GO")):
                        if e and not r:
                            fails.append(dict(test="T1b" + tag, kind=kind, cfg=(Na, na, Nb, nb), thr=str(thr), alpha=alpha,
                                              x=(xa, xb), direction=name))
                    dec[xa, xb] = (eg, en)
            # T2: exact size of the engine's own decisions at every true count pair
            tot = comb(Na, na) * comb(Nb, nb)
            for Ka in range(Na + 1):
                wa = T0.hg_w(Na, Ka, na)
                for Kb in range(Nb + 1):
                    wb = T0.hg_w(Nb, Kb, nb); h, vac = ref.holds(Ka, Kb)
                    pgo = Fr(sum(wa[i] * wb[j] for (i, j), d in dec.items() if d[0]), tot)
                    png = Fr(sum(wa[i] * wb[j] for (i, j), d in dec.items() if d[1]), tot)
                    if not h or vac:
                        worst = max(worst, pgo / Fr(alpha))
                        if pgo > Fr(alpha):
                            fails.append(dict(test="T2" + tag, kind=kind, cfg=(Na, na, Nb, nb), thr=str(thr), alpha=alpha,
                                              truth=(Ka, Kb), false_go=float(pgo)))
                    if h or vac:
                        worst = max(worst, png / Fr(alpha))
                        if png > Fr(alpha):
                            fails.append(dict(test="T2" + tag, kind=kind, cfg=(Na, na, Nb, nb), thr=str(thr), alpha=alpha,
                                              truth=(Ka, Kb), false_nogo=float(png)))
    if not tag:
        summary["T1_comparisons"] = n_cmp; summary["T1_zero_observation_outcomes"] = n_zero
        summary["T1_margin_band_disagreements"] = n_band; summary["T2_max_error_over_alpha"] = float(worst)
    return fails


def t5(summary):
    """Regression cases with their exact expected decisions."""
    fails = []; out = []
    alpha_pop = lambda pop: 0.005 / (2 * E.n_assertions(pop))  # noqa: E731
    # Boundary population: 5 poisons, 4 of them flagged (recall exactly 4/5), no clean record flagged.
    #  (a) census of both strata: recall = 0.8 >= tau and F = 0 are known exactly -> GO.
    #  (b) 1 of 4 flagged records sampled, unflagged census (m = 1 known): c in {1,..,4};
    #      the GO null contains c <= 3 with p = c/4 <= 3/4 and the NO-GO null contains c = 4
    #      with p = 1, so neither direction rejects -> ABSTAIN.  (v1 answered NO-GO here.)
    for clean_unflagged, sample_flagged, want in ((9, 4, "GO"), (5, 1, "ABSTAIN")):
        z = np.array([True] * 5 + [False] * clean_unflagged)
        flags = np.array([1] * 4 + [0] * (1 + clean_unflagged), dtype=bool)
        M = np.ones((len(z), 1), dtype=bool)
        pop = dict(poison=z, memberships=M, gsize=[(4, 1 + clean_unflagged)], N={1: 4, 0: 1 + clean_unflagged})
        chosen = {1: np.arange(sample_flagged), 0: np.arange(4, len(z))}
        d = E.decide_ui(pop, chosen, alpha_pop(pop), 0.8, 0.1)
        out.append(dict(case="boundary", N=len(z), n_flagged=sample_flagged, decision=d, expected=want))
        if d != want:
            fails.append(dict(test="T5", case="boundary", N=len(z), decision=d, expected=want))
    cases = [
        # no flagged record; 2 of 5 sampled unflagged records are poisons -> recall 0: NO-GO only
        ("recall", (0, 0, 0, 10, 5, 2, 0.8, 0.001), (False, True)),
        # every record flagged, 2 sampled poisons -> recall 1: GO only
        ("recall", (10, 5, 2, 0, 0, 0, 0.8, 0.001), (True, False)),
        # group with no flagged record (CF = 0) and clean unflagged members -> F = 0: GO only
        ("group", (0, 0, 0, 10, 5, 2, 0.1, 0.001), (True, False)),
        # group entirely flagged with sampled clean members -> F = 1: NO-GO only
        ("group", (10, 5, 2, 0, 0, 0, 0.1, 0.001), (False, True)),
        # zero-observation recall outcomes: no sampled poison anywhere -> neither direction
        ("recall", (10, 5, 0, 20, 10, 0, 0.8, 0.001), (False, False)),
        ("recall", (4, 4, 0, 10, 10, 0, 0.8, 0.2), (False, False)),
    ]
    for kind, args, want in cases:
        try:
            got = tuple(bool(v) for v in (E.recall_tests if kind == "recall" else E.group_tests)(*args))
        except Exception as err:  # noqa: BLE001
            fails.append(dict(test="T5", case=kind, args=args, error=repr(err))); continue
        out.append(dict(case=kind, args=args, result=list(got), expected=list(want)))
        if got != want:
            fails.append(dict(test="T5", case=kind, args=args, result=list(got), expected=list(want)))
    summary["T5_cases"] = out
    return fails


def t6(summary):
    """Sensitivity: an engine that answers GO on the zero-observation outcome must fail."""
    def broken(kind, Na, na, xa, Nb, nb, xb, thr, alpha):
        if kind == "recall" and xa == 0 and xb == 0:
            return True, False
        return T0.engine_call(kind, Na, na, xa, Nb, nb, xb, thr, alpha)
    caught = t1_t2({}, engine_call=broken, configs=T0.CONFIGS[:4], alphas=(0.05,),
                   thresholds=(("recall", (Fr(4, 5),)),), tag="-mutant")
    summary["T6_mutant_failures_detected"] = len(caught)
    return [] if caught else [dict(test="T6", reason="zero-observation GO mutant not detected")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, default=None); a = ap.parse_args()
    summary = dict(engine=E.ENGINE_VERSION, margin=E.MARGIN, suite=2, alphas=list(ALPHAS))
    fails = t5(summary) + T0.t3(summary) + T0.t4(summary) + t1_t2(summary) + t6(summary)
    summary["failures"] = fails[:50]; summary["n_failures"] = len(fails); summary["passed"] = not fails
    txt = json.dumps(summary, indent=2, default=str)
    print(txt)
    if a.json:
        a.json.write_text(txt + "\n")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
