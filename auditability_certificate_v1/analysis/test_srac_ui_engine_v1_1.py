#!/usr/bin/env python3
"""Regression and exact-size tests for the SRAC-UI engine v1.1 (analysis/srac_ui_engine_v1_1.py).

Every check uses exact integer / rational arithmetic (math.comb, fractions.Fraction) and an
independent brute-force reference that enumerates *all* null count pairs (no monotone
reduction, no branch and bound).

  T1  decision equality: engine vs. brute-force reference on every outcome of every small
      configuration (recall and group assertions; tau in {4/5, 2/3, 1/2}; cap in {1/10, 1/4,
      1/3}; alpha in {0.05, 0.2}; census, partial census, empty strata).  The only permitted
      disagreement is the conservative margin band: the reference rejects with an exact
      p-value in ((alpha-gamma)(1-MARGIN), alpha-gamma] and the engine does not.
  T2  exact size: for every true count pair of every configuration, the exact probability of
      a false GO (assertion false) and of a false NO-GO (assertion true) is <= alpha.
  T3  Berger-Boos interval: the engine interval contains the exact rational interval.
  T4  census: on random small populations with overlapping groups, a census returns the true
      decision (GO iff the policy passes, NO-GO iff it fails), apart from the documented
      conservative cases (no poison; a flagged group with no clean record).
  T5  regression cases from the 2026-10-18 evaluation (boundary false NO-GO; empty strata).
Exit status 0 iff all tests pass; a JSON summary is printed and optionally written.
"""
from __future__ import annotations
import argparse, itertools, json, sys
from fractions import Fraction as Fr
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import srac_ui_engine_v1_1 as E  # noqa: E402

CONFIGS = [  # (Na, na, Nb, nb): a = flagged stratum, b = unflagged stratum
    (4, 4, 10, 10), (4, 1, 6, 6), (6, 3, 10, 5), (8, 4, 12, 6), (10, 10, 10, 3), (5, 0, 10, 5),
    (5, 3, 10, 0), (0, 0, 10, 4), (6, 2, 0, 0), (12, 6, 15, 7), (3, 2, 20, 10), (9, 9, 9, 9),
]
ALPHAS = (0.05, 0.2)


def hg_w(N, K, n):
    return [comb(K, x) * comb(N - K, n - x) for x in range(n + 1)]


def exact_interval(N, n, x, eta: Fr):
    """Exact two-sided hypergeometric count interval, P(tail) <= eta each side (rational)."""
    if n == 0:
        return 0, N
    tot = comb(N, n); feas = range(x, N - (n - x) + 1)
    lower = min(K for K in feas if Fr(sum(hg_w(N, K, n)[x:]), tot) > eta)
    upper = max(K for K in feas if Fr(sum(hg_w(N, K, n)[:x + 1]), tot) > eta)
    return lower, upper


class Ref:
    """Brute-force exact UI test of one linear assertion  wa*Ka - wb*Kb  (>= 0: 'recall' kind;
    <= 0: 'group' kind) with the engine's statistic S = A*Xa - B*Xb."""

    def __init__(self, kind, Na, na, Nb, nb, thr: Fr):
        self.kind, self.Na, self.na, self.Nb, self.nb = kind, Na, na, Nb, nb
        r = thr.numerator, thr.denominator
        if kind == "recall":
            self.wa, self.wb = r[1] - r[0], r[0]
        else:
            self.wa, self.wb = r[1] - r[0], r[0]
        self.A = self.wa * Na * max(nb, 1) if na > 0 else 0
        self.B = self.wb * Nb * max(na, 1) if nb > 0 else 0
        self.tot = comb(Na, na) * comb(Nb, nb)
        self.dist = {}
        for Ka in range(Na + 1):
            wa = hg_w(Na, Ka, na)
            for Kb in range(Nb + 1):
                wb = hg_w(Nb, Kb, nb); d = {}
                for xa, u in enumerate(wa):
                    if u:
                        for xb, v in enumerate(wb):
                            if v:
                                S = self.A * xa - self.B * xb; d[S] = d.get(S, 0) + u * v
                self.dist[Ka, Kb] = sorted(d.items())

    def holds(self, Ka, Kb):
        if self.kind == "recall":
            return Ka + Kb > 0 and self.wa * Ka - self.wb * Kb >= 0, Ka + Kb == 0   # (holds, vacuous)
        return self.wa * Ka - self.wb * Kb <= 0, Ka + Kb == 0

    def tail(self, Ka, Kb, s, side):
        return Fr(sum(w for S, w in self.dist[Ka, Kb] if (S >= s if side == "ge" else S <= s)), self.tot)

    def decide(self, xa, xb, alpha, gamma, bb):
        """Exact UI decision; bb = (lo, hi) Berger-Boos range for the constrained count
        (Ka for recall, Kb for group), taken from the engine so that T1 isolates the sup."""
        s = self.A * xa - self.B * xb
        Ka_rng = range(xa, self.Na - (self.na - xa) + 1) if self.na > 0 else range(0, self.Na + 1)
        Kb_rng = range(xb, self.Nb - (self.nb - xb) + 1) if self.nb > 0 else range(0, self.Nb + 1)
        if self.kind == "recall":
            Ka_rng = [k for k in Ka_rng if bb[0] <= k <= bb[1]]
        else:
            Kb_rng = [k for k in Kb_rng if bb[0] <= k <= bb[1]]
        if self.kind == "recall":   # GO rejects large S; NO-GO rejects small S
            go_side, ng_side = "ge", "le"
        else:                       # GO rejects small S; NO-GO rejects large S
            go_side, ng_side = "le", "ge"
        p_go = p_ng = Fr(0)
        for Ka in Ka_rng:
            for Kb in Kb_rng:
                h, vac = self.holds(Ka, Kb)
                in_go_null = (not h) or vac
                in_ng_null = h or vac
                if in_go_null:
                    p_go = max(p_go, self.tail(Ka, Kb, s, go_side))
                if in_ng_null:
                    p_ng = max(p_ng, self.tail(Ka, Kb, s, ng_side))
        lim = Fr(alpha) - Fr(gamma)
        return p_go <= lim, p_ng <= lim, p_go, p_ng


def engine_bb(kind, Na, na, xa, Nb, nb, xb, gamma):
    if kind == "recall":
        if 0 < na < Na:
            r = E._bb_interval(Na, na, xa, gamma); return r.lower, r.upper
        return 0, Na
    if 0 < nb < Nb:
        r = E._bb_interval(Nb, nb, xb, gamma); return r.lower, r.upper
    return 0, Nb


def engine_call(kind, Na, na, xa, Nb, nb, xb, thr, alpha):
    if kind == "recall":
        return E.recall_tests(Na, na, xa, Nb, nb, xb, thr, alpha)
    return E.group_tests(Na, na, xa, Nb, nb, xb, thr, alpha)


def t1_t2(summary):
    fails = []; n_cmp = n_band = 0; worst = Fr(0)
    for kind, thrs in (("recall", (Fr(4, 5), Fr(2, 3), Fr(1, 2))), ("group", (Fr(1, 10), Fr(1, 4), Fr(1, 3)))):
        for (Na, na, Nb, nb), thr, alpha in itertools.product(CONFIGS, thrs, ALPHAS):
            ref = Ref(kind, Na, na, Nb, nb, thr)
            gamma = float(E.GAMMA_SHARE) * alpha
            dec = {}
            for xa in range(na + 1):
                for xb in range(nb + 1):
                    if kind == "recall" and xa == 0 and xb == 0:
                        dec[xa, xb] = (False, False); continue
                    eg, en = engine_call(kind, Na, na, xa, Nb, nb, xb, thr, alpha)
                    bb = engine_bb(kind, Na, na, xa, Nb, nb, xb, gamma)
                    rg, rn, pg, pn = ref.decide(xa, xb, alpha, Fr(gamma), bb)
                    n_cmp += 1
                    lim = Fr(alpha) - Fr(gamma); band = lim * (1 - Fr(E.MARGIN))
                    for e, r, p, name in ((eg, rg, pg, "GO"), (en, rn, pn, "NO_GO")):
                        if e != r:
                            if r and not e and band < p <= lim:
                                n_band += 1
                            else:
                                fails.append(dict(test="T1", kind=kind, cfg=(Na, na, Nb, nb), thr=str(thr), alpha=alpha,
                                                  x=(xa, xb), direction=name, engine=e, reference=r, p=float(p)))
                    dec[xa, xb] = (eg, en)
            # T2 exact size of the engine decisions
            tot = comb(Na, na) * comb(Nb, nb)
            for Ka in range(Na + 1):
                wa = hg_w(Na, Ka, na)
                for Kb in range(Nb + 1):
                    wb = hg_w(Nb, Kb, nb); h, vac = ref.holds(Ka, Kb)
                    pgo = Fr(sum(wa[i] * wb[j] for (i, j), d in dec.items() if d[0]), tot)
                    png = Fr(sum(wa[i] * wb[j] for (i, j), d in dec.items() if d[1]), tot)
                    if (not h or vac):
                        worst = max(worst, pgo / Fr(alpha))
                        if pgo > Fr(alpha):
                            fails.append(dict(test="T2", kind=kind, cfg=(Na, na, Nb, nb), thr=str(thr), alpha=alpha, truth=(Ka, Kb), false_go=float(pgo)))
                    if h or vac:
                        worst = max(worst, png / Fr(alpha))
                        if png > Fr(alpha):
                            fails.append(dict(test="T2", kind=kind, cfg=(Na, na, Nb, nb), thr=str(thr), alpha=alpha, truth=(Ka, Kb), false_nogo=float(png)))
    summary["T1_comparisons"] = n_cmp; summary["T1_margin_band_disagreements"] = n_band
    summary["T2_max_error_over_alpha"] = float(worst)
    return fails


def t3(summary):
    fails = []; n = 0
    for N in range(1, 31):
        for m in range(1, N):
            for x in range(m + 1):
                for gamma in (1e-3, 0.0125, 0.1):
                    lo, hi = exact_interval(N, m, x, Fr(gamma) / 2)
                    r = E._bb_interval(N, m, x, gamma); n += 1
                    if not (r.lower <= lo and r.upper >= hi):
                        fails.append(dict(test="T3", N=N, n=m, x=x, gamma=gamma, engine=(r.lower, r.upper), exact=(lo, hi)))
    summary["T3_intervals"] = n
    return fails


def t4(summary, n_pops=400, seed=20261018):
    rng = np.random.default_rng(seed); fails = []; checked = cons = 0
    for _ in range(n_pops):
        N = int(rng.integers(6, 40)); flags = rng.random(N) < rng.uniform(0.1, 0.6)
        poison = rng.random(N) < rng.uniform(0.05, 0.5); labels = rng.integers(0, 4, N)
        groups = [np.isin(labels, g) for g in ([0, 1], [1, 2], [3], [0, 2, 3])]
        M = np.column_stack(groups)
        gsize = [(int(np.sum(M[:, g] & flags)), int(np.sum(M[:, g] & ~flags))) for g in range(M.shape[1])]
        pop = dict(poison=poison, memberships=M, gsize=gsize, N={1: int(flags.sum()), 0: int((~flags).sum())})
        chosen = {1: np.flatnonzero(flags), 0: np.flatnonzero(~flags)}
        alpha = 0.005 / (2 * E.n_assertions(pop))
        d = E.decide_ui(pop, chosen, alpha, 0.8, 0.1)
        npois = int(poison.sum()); rec = (poison & flags).sum() / npois if npois else None
        rates = [ (M[:, g] & ~poison & flags).sum() / (M[:, g] & ~poison).sum() for g in range(M.shape[1]) if (M[:, g] & ~poison).any()]
        passes = rec is not None and Fr(int((poison & flags).sum()), npois) >= Fr(4, 5) and all(
            Fr(int((M[:, g] & ~poison & flags).sum()), int((M[:, g] & ~poison).sum())) <= Fr(1, 10)
            for g in range(M.shape[1]) if (M[:, g] & ~poison).any())
        conservative = npois == 0 or any(gsize[g][0] > 0 and not (M[:, g] & ~poison).any() for g in range(M.shape[1]))
        checked += 1
        want = "GO" if passes else "NO_GO"
        if d != want:
            if conservative and d != ("NO_GO" if passes else "GO"):
                cons += 1
            else:
                fails.append(dict(test="T4", N=N, decision=d, truth_pass=bool(passes), recall=None if rec is None else float(rec)))
    summary["T4_census_populations"] = checked; summary["T4_conservative_exceptions"] = cons
    return fails


def t5(summary):
    import pandas as pd
    fails = []; out = []
    for clean_unflagged, sample_flagged in ((9, 4), (5, 1)):
        z = np.array([True] * 5 + [False] * clean_unflagged)
        flags = np.array([1] * 4 + [0] * (1 + clean_unflagged), dtype=bool)
        M = np.ones((len(z), 1), dtype=bool)
        pop = dict(poison=z, memberships=M, gsize=[(4, 1 + clean_unflagged)], N={1: 4, 0: 1 + clean_unflagged})
        chosen = {1: np.arange(sample_flagged), 0: np.arange(4, len(z))}
        d = E.decide_ui(pop, chosen, 0.005 / (2 * E.n_assertions(pop)), 0.8, 0.1)
        out.append(dict(N=len(z), n_flagged=sample_flagged, decision=d))
        if d == "NO_GO":
            fails.append(dict(test="T5", case="boundary", N=len(z), decision=d))
    for args in ((0, 0, 0, 10, 5, 2, 0.8, 0.001), (10, 5, 2, 0, 0, 0, 0.8, 0.001)):
        try:
            out.append(dict(args=args, result=list(E.recall_tests(*args))))
        except Exception as err:  # noqa: BLE001
            fails.append(dict(test="T5", case="empty", args=args, error=repr(err)))
    for args in ((0, 0, 0, 10, 5, 2, 0.1, 0.001), (10, 5, 2, 0, 0, 0, 0.1, 0.001)):
        try:
            out.append(dict(args=args, result=list(E.group_tests(*args))))
        except Exception as err:  # noqa: BLE001
            fails.append(dict(test="T5", case="empty-group", args=args, error=repr(err)))
    summary["T5_cases"] = out
    return fails


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", type=Path, default=None); a = ap.parse_args()
    summary = dict(engine=E.ENGINE_VERSION, margin=E.MARGIN)
    fails = t5(summary) + t3(summary) + t4(summary) + t1_t2(summary)
    summary["failures"] = fails[:50]; summary["n_failures"] = len(fails); summary["passed"] = not fails
    txt = json.dumps(summary, indent=2, default=str)
    print(txt)
    if a.json:
        a.json.write_text(txt + "\n")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
