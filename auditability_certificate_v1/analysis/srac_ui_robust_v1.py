"""SRAC-UI with up to R wrong sampled labels (declared-B robust tests), built on engine v1.1.

Section V-C of the manuscript states that maximizing the p-value over every count vector
within R label flips of the observed one preserves validity.  This module implements a
superset of that maximization that needs one supremum per test:

  * a label flip changes the observed count of exactly one stratum by one, so the true
    sample counts (x1*, x0*) lie in the L1 ball of radius R around the observed (x1, x0),
    clipped to [0, n1] x [0, n0];
  * the engine's p-value at x* is a supremum, over null counts consistent with x* and inside
    the Berger-Boos range of x*, of a tail probability at the statistic s(x*);
  * here the supremum runs over the union of the consistency sets and of the Berger-Boos
    ranges over the ball (the exact interval endpoints are nondecreasing in the observed
    count), and the tail is evaluated at the least favourable statistic over the ball.
    The tail is monotone in the threshold, so the returned p-value is at least the engine's
    p-value at every x* in the ball, and a rejection here implies a rejection by the
    engine at the (unknown) true counts.  Hence every test keeps its size alpha, and
    SRAC-UI keeps false GO <= delta/2 and false NO-GO <= delta/2, whenever at most R
    sampled labels are wrong.
  * if the ball contains the vacuous outcome of the recall assertion (no sampled poison),
    neither direction rejects, as in the engine.
With R = 0 every function reduces exactly to the engine (checked in selftest()).
"""
from __future__ import annotations
import numpy as np
import srac_ui_engine_v1_1 as E


def _range(n, x, R):
    """Observed count x of a stratum with n sampled records, moved by at most R."""
    return max(0, x - R), min(n, x + R)


def _extreme(A, x1, n1, B, x0, n0, R, lowest):
    """min (lowest=True) or max of A*x1' - B*x0' over the clipped L1 ball of radius R."""
    best = None
    for d in range(R + 1):            # d flips in stratum 1, R - d in stratum 0
        if lowest:
            v = A * max(0, x1 - d) - B * min(n0, x0 + (R - d))
        else:
            v = A * min(n1, x1 + d) - B * max(0, x0 - (R - d))
        best = v if best is None else (min(best, v) if lowest else max(best, v))
    return best


def _bb_union(N, n, x, R, gamma):
    lo, hi = _range(n, x, R)
    return E._bb_interval(N, n, lo, gamma).lower, E._bb_interval(N, n, hi, gamma).upper


def recall_tests(N1, n1, x1, N0, n0, x0, tau, alpha, R=0, gamma=None):
    if R == 0:
        return E.recall_tests(N1, n1, x1, N0, n0, x0, tau, alpha, gamma)
    tau = E.rational(tau); P, Q = tau.numerator, tau.denominator
    w1, w0 = Q - P, P
    N1, n1, x1, N0, n0, x0, R = map(int, (N1, n1, x1, N0, n0, x0, R))
    if x1 + x0 <= R:                 # the ball contains the vacuous outcome
        return False, False
    A = w1 * N1 * max(n0, 1) if n1 > 0 else 0
    B = w0 * N0 * max(n1, 1) if n0 > 0 else 0
    s_lo = _extreme(A, x1, n1, B, x0, n0, R, True)
    s_hi = _extreme(A, x1, n1, B, x0, n0, R, False)
    alpha, gamma, thr = E._alpha_gamma(alpha, gamma)
    x1lo, x1hi = _range(n1, x1, R) if n1 > 0 else (0, 0)
    x0lo, x0hi = _range(n0, x0, R) if n0 > 0 else (0, 0)
    c_lo, c_hi = (x1lo, N1 - (n1 - x1hi)) if n1 > 0 else (0, N1)
    m_lo = x0lo if n0 > 0 else 0
    m_max = N0 - (n0 - x0hi) if n0 > 0 else N0
    if 0 < n1 < N1:
        bl, bh = _bb_union(N1, n1, x1, R, gamma); c_lo, c_hi = max(c_lo, bl), min(c_hi, bh)
    m_go = lambda c: np.maximum((w1 * np.asarray(c, dtype=np.int64)) // w0 + 1, m_lo)
    m_ng = lambda c: np.minimum((w1 * np.asarray(c, dtype=np.int64)) // w0, m_max)
    go_hi = min(c_hi, E._ceil_div(w0 * m_max, w1) - 1)
    go = E.sup_at_most(c_lo, go_hi, lambda a, b: E.linear_p(N1, n1, b, A, N0, n0, m_go(a), B, s_lo, "ge"), thr)
    ng_lo = max(c_lo, 1, E._ceil_div(w0 * m_lo, w1))
    ng = E.sup_at_most(ng_lo, c_hi, lambda a, b: E.linear_p(N1, n1, a, A, N0, n0, m_ng(b), B, s_hi, "le"), thr)
    return go, ng


def group_tests(fs, n1, y1, us, n0, y0, cap, alpha, R=0, gamma=None):
    if R == 0:
        return E.group_tests(fs, n1, y1, us, n0, y0, cap, alpha, gamma)
    cap = E.rational(cap); a_, b_ = cap.numerator, cap.denominator
    w1, w0 = b_ - a_, a_
    fs, n1, y1, us, n0, y0, R = map(int, (fs, n1, y1, us, n0, y0, R))
    A = w1 * fs * max(n0, 1) if n1 > 0 else 0
    B = w0 * us * max(n1, 1) if n0 > 0 else 0
    s_lo = _extreme(A, y1, n1, B, y0, n0, R, True)
    s_hi = _extreme(A, y1, n1, B, y0, n0, R, False)
    alpha, gamma, thr = E._alpha_gamma(alpha, gamma)
    y1lo, y1hi = _range(n1, y1, R) if n1 > 0 else (0, 0)
    y0lo, y0hi = _range(n0, y0, R) if n0 > 0 else (0, 0)
    cu_lo, cu_hi = (y0lo, us - (n0 - y0hi)) if n0 > 0 else (0, us)
    if 0 < n0 < us:
        bl, bh = _bb_union(us, n0, y0, R, gamma); cu_lo, cu_hi = max(cu_lo, bl), min(cu_hi, bh)
    cf_min, cf_max = (y1lo, fs - (n1 - y1hi)) if n1 > 0 else (0, fs)
    cf_go = lambda cu: np.maximum((w0 * np.asarray(cu, dtype=np.int64)) // w1 + 1, cf_min)
    cf_ng = lambda cu: np.minimum((w0 * np.asarray(cu, dtype=np.int64)) // w1, cf_max)
    go_hi = min(cu_hi, E._ceil_div(w1 * cf_max, w0) - 1)
    go = E.sup_at_most(cu_lo, go_hi, lambda lo, hi: E.linear_p(fs, n1, cf_go(lo), A, us, n0, hi, B, s_hi, "le"), thr)
    if cu_lo == 0 and cf_min == 0:
        go = False
    ng_lo = max(cu_lo, E._ceil_div(w1 * cf_min, w0))
    ng = E.sup_at_most(ng_lo, cu_hi, lambda lo, hi: E.linear_p(fs, n1, cf_ng(hi), A, us, n0, lo, B, s_lo, "ge"), thr)
    return go, ng


def decide(pop, chosen, labels, alpha, tau, cap, R=0):
    """SRAC-UI decision from the adjudicated labels `labels` (True = poison) of the sampled
    records, declaring that at most R of them are wrong."""
    M = pop["memberships"]; s1, s0 = chosen[1], chosen[0]
    go, ng = recall_tests(pop["N"][1], len(s1), int(labels[s1].sum()), pop["N"][0], len(s0), int(labels[s0].sum()), tau, alpha, R)
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        a1 = s1[M[s1, g]]; a0 = s0[M[s0, g]]
        g_go, g_ng = group_tests(fs, len(a1), int((~labels[a1]).sum()), us, len(a0), int((~labels[a0]).sum()), cap, alpha, R)
        go &= g_go; ng |= g_ng
    return "GO" if go else ("NO_GO" if ng else "ABSTAIN")


def selftest():
    """Exhaustive check on small configurations: for R in {1, 2}, every observed outcome, and
    every true outcome within R flips, a rejection by the robust test implies a rejection by
    the engine at the true outcome (both directions, both assertion kinds)."""
    import itertools
    from fractions import Fraction as Fr
    configs = [(4, 4, 10, 10), (6, 3, 10, 5), (8, 4, 12, 6), (10, 10, 10, 3), (5, 0, 10, 5), (5, 3, 10, 0), (12, 6, 15, 7)]
    checked = viol = 0
    for kind, thrs in (("recall", (Fr(4, 5), Fr(1, 2))), ("group", (Fr(1, 10), Fr(1, 3)))):
        f = recall_tests if kind == "recall" else group_tests
        for (Na, na, Nb, nb), thr, alpha, R in itertools.product(configs, thrs, (0.05, 0.2), (1, 2)):
            eng = {(a, b): f(Na, na, a, Nb, nb, b, thr, alpha) for a in range(na + 1) for b in range(nb + 1)}
            for (a, b) in eng:
                rg, rn = f(Na, na, a, Nb, nb, b, thr, alpha, R)
                for (a2, b2), (eg, en) in eng.items():
                    if abs(a2 - a) + abs(b2 - b) <= R:
                        checked += 1
                        viol += (rg and not eg) + (rn and not en)
    zero = all(recall_tests(10, 5, x, 20, 10, y, Fr(4, 5), 0.05, 0) == E.recall_tests(10, 5, x, 20, 10, y, Fr(4, 5), 0.05)
               for x in range(6) for y in range(11))
    return dict(pairs_checked=checked, violations=viol, reduces_to_engine_at_R0=zero, passed=viol == 0 and zero)


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))
