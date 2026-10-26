"""GO-or-ABSTAIN SRAC-UI with directional label-error budgets (built on engine v1.1).

Only label errors that can help a GO need a bound:
  * flagged stratum, clean labelled poison  (inflates sampled poisons x1; deflates the clean
    count y1 of every group's flagged cell)                       at most B1 in the sample;
  * unflagged stratum, poison labelled clean (deflates sampled misses x0; inflates the clean
    count y0 of every group's unflagged cell)                     at most B0 in the sample.
Errors in the other two directions are unbounded.  Hence the true sample counts lie in
  recall: x1* in [x1 - B1, n1],  x0* in [0, x0 + B0];
  group:  y1* in [0, y1 + B1],    y0* in [y0 - B0, n0]  (clipped to the feasible range),
and each GO test takes the supremum of the engine's GO p-value over this rectangle in the
same way as the declared-B tests (srac_ui_robust_v1): union of the consistency sets and of the
Berger-Boos ranges (the exact interval endpoints are nondecreasing in the observed count), the
least favourable statistic, and no rejection when the rectangle contains the outcome with no
sampled poison.  A rejection therefore implies a rejection by the engine at the true counts,
so every GO test keeps its level and false GO stays at most the level of the failing
assertion's test.  NO-GO is not issued: errors that help a NO-GO are not bounded.
"""
from __future__ import annotations
import numpy as np
import srac_ui_engine_v1_1 as E


def _bb_union(N, n, lo, hi, gamma):
    return E._bb_interval(N, n, lo, gamma).lower, E._bb_interval(N, n, hi, gamma).upper


def recall_go(N1, n1, x1, N0, n0, x0, tau, alpha, B1=0, B0=0, gamma=None):
    tau = E.rational(tau); w1, w0 = tau.denominator - tau.numerator, tau.numerator
    N1, n1, x1, N0, n0, x0, B1, B0 = map(int, (N1, n1, x1, N0, n0, x0, B1, B0))
    x1lo, x1hi = (max(0, x1 - B1), n1) if n1 > 0 else (0, 0)
    x0lo, x0hi = (0, min(n0, x0 + B0)) if n0 > 0 else (0, 0)
    if x1lo == 0 and x0lo == 0:          # the rectangle contains the vacuous outcome
        return False
    A = w1 * N1 * max(n0, 1) if n1 > 0 else 0
    B = w0 * N0 * max(n1, 1) if n0 > 0 else 0
    s_lo = A * x1lo - B * x0hi
    alpha, gamma, thr = E._alpha_gamma(alpha, gamma)
    c_lo, c_hi = (x1lo, N1 - (n1 - x1hi)) if n1 > 0 else (0, N1)
    m_lo = x0lo if n0 > 0 else 0
    m_max = N0 - (n0 - x0hi) if n0 > 0 else N0
    if 0 < n1 < N1:
        bl, bh = _bb_union(N1, n1, x1lo, x1hi, gamma); c_lo, c_hi = max(c_lo, bl), min(c_hi, bh)
    m_go = lambda c: np.maximum((w1 * np.asarray(c, dtype=np.int64)) // w0 + 1, m_lo)
    go_hi = min(c_hi, E._ceil_div(w0 * m_max, w1) - 1)
    return E.sup_at_most(c_lo, go_hi, lambda a, b: E.linear_p(N1, n1, b, A, N0, n0, m_go(a), B, s_lo, "ge"), thr)


def group_go(fs, n1, y1, us, n0, y0, cap, alpha, B1=0, B0=0, gamma=None):
    cap = E.rational(cap); w1, w0 = cap.denominator - cap.numerator, cap.numerator
    fs, n1, y1, us, n0, y0, B1, B0 = map(int, (fs, n1, y1, us, n0, y0, B1, B0))
    y1lo, y1hi = (0, min(n1, y1 + B1)) if n1 > 0 else (0, 0)
    y0lo, y0hi = (max(0, y0 - B0), n0) if n0 > 0 else (0, 0)
    A = w1 * fs * max(n0, 1) if n1 > 0 else 0
    B = w0 * us * max(n1, 1) if n0 > 0 else 0
    s_hi = A * y1hi - B * y0lo
    alpha, gamma, thr = E._alpha_gamma(alpha, gamma)
    cu_lo, cu_hi = (y0lo, us - (n0 - y0hi)) if n0 > 0 else (0, us)
    if 0 < n0 < us:
        bl, bh = _bb_union(us, n0, y0lo, y0hi, gamma); cu_lo, cu_hi = max(cu_lo, bl), min(cu_hi, bh)
    cf_min, cf_max = (y1lo, fs - (n1 - y1hi)) if n1 > 0 else (0, fs)
    cf_go = lambda cu: np.maximum((w0 * np.asarray(cu, dtype=np.int64)) // w1 + 1, cf_min)
    go_hi = min(cu_hi, E._ceil_div(w1 * cf_max, w0) - 1)
    go = E.sup_at_most(cu_lo, go_hi, lambda lo, hi: E.linear_p(fs, n1, cf_go(lo), A, us, n0, hi, B, s_hi, "le"), thr)
    if cu_lo == 0 and cf_min == 0:
        go = False
    return go


def decide_go(pop, chosen, labels, alpha, tau, cap, B1=0, B0=0):
    """GO or ABSTAIN from adjudicated labels (True = poison) of the sampled records."""
    M = pop["memberships"]; s1, s0 = chosen[1], chosen[0]
    if not recall_go(pop["N"][1], len(s1), int(labels[s1].sum()), pop["N"][0], len(s0), int(labels[s0].sum()), tau, alpha, B1, B0):
        return "ABSTAIN"
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        a1 = s1[M[s1, g]]; a0 = s0[M[s0, g]]
        if not group_go(fs, len(a1), int((~labels[a1]).sum()), us, len(a0), int((~labels[a0]).sum()), cap, alpha,
                        min(B1, len(a1)), min(B0, len(a0))):
            return "ABSTAIN"
    return "GO"


def selftest():
    """Exhaustive check: for B1, B0 in {0, 1, 2}, every observed outcome, and every true outcome
    consistent with the directional error model, a directional GO rejection implies a GO
    rejection by the engine at the true outcome (both assertion kinds)."""
    import itertools
    from fractions import Fraction as Fr
    configs = [(4, 4, 10, 10), (6, 3, 10, 5), (8, 4, 12, 6), (10, 10, 10, 3), (5, 0, 10, 5), (5, 3, 10, 0), (12, 6, 15, 7)]
    checked = viol = 0
    for kind, thrs in (("recall", (Fr(4, 5), Fr(1, 2))), ("group", (Fr(1, 10), Fr(1, 3)))):
        for (Na, na, Nb, nb), thr, alpha, B1, B0 in itertools.product(configs, thrs, (0.05, 0.2), (0, 1, 2), (0, 1, 2)):
            if kind == "recall":
                eng = {(a, b): E.recall_tests(Na, na, a, Nb, nb, b, thr, alpha)[0] for a in range(na + 1) for b in range(nb + 1)}
                dirf = lambda a, b: recall_go(Na, na, a, Nb, nb, b, thr, alpha, B1, B0)
                # true (a*, b*) given observed (a, b): a* in [a - B1, na], b* in [0, b + B0]
                ok = lambda a, b, a2, b2: a - B1 <= a2 and b2 <= b + B0
            else:
                eng = {(a, b): E.group_tests(Na, na, a, Nb, nb, b, thr, alpha)[0] for a in range(na + 1) for b in range(nb + 1)}
                dirf = lambda a, b: group_go(Na, na, a, Nb, nb, b, thr, alpha, B1, B0)
                ok = lambda a, b, a2, b2: a2 <= a + B1 and b - B0 <= b2
            for (a, b) in eng:
                if not dirf(a, b):
                    continue
                for (a2, b2), eg in eng.items():
                    if ok(a, b, a2, b2):
                        checked += 1
                        viol += not eg
    zero_r = all(recall_go(10, 5, x, 20, 10, y, Fr(4, 5), 0.05) <= E.recall_tests(10, 5, x, 20, 10, y, Fr(4, 5), 0.05)[0]
                 for x in range(6) for y in range(11))
    return dict(pairs_checked=checked, violations=viol, never_more_than_engine_at_B0=zero_r, passed=viol == 0 and zero_r)


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))
