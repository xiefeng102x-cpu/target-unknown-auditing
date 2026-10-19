"""SRAC-UI engine, version 1.1 (erratum release of the v1 engine).

The v1 engine lives inside analysis/run_stratified_ui_audit_v1.py (amended sha256 9dcff664...)
and is kept unchanged, together with every result it produced.  The statistical construction
(Proposition 4: union-intersection over null counts, exact convolution of two hypergeometric
laws, Berger-Boos restriction of the nuisance count, monotone block bounds, branch and bound)
is unchanged.  Version 1.1 changes only arithmetic and the handling of degenerate strata:

  E1  Policy thresholds are rationals (tau = 4/5 and cap = 1/10, parsed from their decimal
      strings).  The null boundaries  m <= (1-tau)/tau * c  and  CF <= cap/(1-cap) * CU  are
      evaluated by integer floor division.  v1 evaluated (1-0.8)/0.8 = 0.24999999999999994 in
      floating point, so whenever 4 | c the NO-GO null omitted its boundary point m = c/4.
      That is anti-conservative: a population with recall exactly 0.8 could be declared
      NO-GO with probability above alpha (probability 1 under a census).  The group ratio
      0.1/0.9 rounds upward and had no effect (checked to 1e5 in the audit).
  E2  The statistic is evaluated on an integer scale, S = A*Xa - B*Xb with integers A, B, and
      tail thresholds are integer ceil/floor divisions (v1: float division with a 1e-9
      tolerance, which can misplace a threshold when the rational gap is below 1e-9).
  E3  An empty stratum (n = 0) contributes 0 to the statistic; the corresponding count is
      unconstrained by the sample (v1 divided by n and raised ZeroDivisionError).
  E4  Floating-point pmfs.  Log-gamma pmfs carry a relative error of order 1e-9 at N ~ 5e4.
      A p-value is therefore compared with (alpha - gamma) * (1 - MARGIN) and the Berger-Boos
      interval is computed at level (gamma / 2) * (1 - MARGIN), MARGIN = 1e-6.  Both margins
      can only make a test reject less often.
"""
from __future__ import annotations

from fractions import Fraction
import numpy as np
from scipy.special import gammaln

ENGINE_VERSION = "srac-ui-1.1"
GAMMA_SHARE = Fraction(1, 20)     # Berger-Boos: gamma = 0.05 alpha
MARGIN = 1e-6
_I64 = 2 ** 62


def rational(x) -> Fraction:
    """Exact rational from a decimal policy value (0.8 -> 4/5); never from binary floats."""
    return x if isinstance(x, Fraction) else Fraction(str(x))


def _ceil_div(a, b):
    return -((-a) // b)


def hg_pmf(x, N, K, n):
    """Hypergeometric pmf by log-gamma differences (vectorized; zero outside the support)."""
    x = np.asarray(x, dtype=np.float64); K = np.asarray(K, dtype=np.float64)
    ok = (x >= 0) & (x <= K) & (n - x >= 0) & (n - x <= N - K)
    xs = np.where(ok, x, 0.0); Ks = np.where(ok, K, n)
    lg = (gammaln(Ks + 1) - gammaln(xs + 1) - gammaln(Ks - xs + 1) + gammaln(N - Ks + 1) - gammaln(n - xs + 1)
          - gammaln(N - Ks - n + xs + 1) - (gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1)))
    return np.where(ok, np.exp(lg), 0.0)


def linear_p(Na, na, Ka, A, Nb, nb, Kb, B, s, side):
    """Vectorized over parameter pairs (Ka[i], Kb[i]):
        P(A*Xa - B*Xb >= s)  (side 'ge')   or   P(A*Xa - B*Xb <= s)  (side 'le'),
    Xa ~ HG(Na, Ka, na), Xb ~ HG(Nb, Kb, nb) independent, A, B >= 0 and s integers.
    A stratum with n = 0 is degenerate at 0.  Thresholds are exact integer divisions."""
    A, B, s = int(A), int(B), int(s)
    if max(abs(A) * (na + 1), abs(B) * (nb + 1), abs(s)) >= _I64:
        raise OverflowError("statistic scale exceeds int64")
    Ka = np.atleast_1d(np.asarray(Ka, dtype=np.int64)); Kb = np.atleast_1d(np.asarray(Kb, dtype=np.int64))
    P = max(len(Ka), len(Kb)); Ka = np.broadcast_to(Ka, (P,)); Kb = np.broadcast_to(Kb, (P,))
    if nb > 0:
        lo = max(0, nb - (Nb - int(Kb.min()))); hi = min(nb, int(Kb.max()))
        xb = np.arange(lo, hi + 1, dtype=np.int64)
        pb = hg_pmf(xb[None, :], Nb, Kb[:, None], nb)
    else:
        xb = np.zeros(1, dtype=np.int64); pb = np.ones((P, 1))
    rhs = s + B * xb                                          # event: A*Xa >= rhs  (or <= rhs)
    if na > 0 and A > 0:
        pa = hg_pmf(np.arange(na + 1)[None, :], Na, Ka[:, None], na)
        if side == "ge":
            tail = np.concatenate([np.cumsum(pa[:, ::-1], axis=1)[:, ::-1], np.zeros((P, 1))], axis=1)
            thr = _ceil_div(rhs, A)
            q = np.where(thr[None, :] <= 0, 1.0, tail[:, np.clip(thr, 0, na + 1)])
        else:
            cdf = np.cumsum(pa, axis=1)
            thr = rhs // A
            q = np.where(thr[None, :] < 0, 0.0, np.where(thr[None, :] >= na, 1.0, cdf[:, np.clip(thr, 0, na)]))
    else:                                                     # A*Xa == 0
        ev = (0 >= rhs) if side == "ge" else (0 <= rhs)
        q = np.broadcast_to(ev.astype(float)[None, :], pb.shape)
    return np.minimum((pb * q).sum(axis=1), 1.0)


def sup_at_most(lo, hi, bound, thr, fan=24):
    """True iff max over integers u in [lo, hi] of p(u) <= thr, where bound(a, b) returns valid
    upper bounds of max p over each sub-range [a, b] that are exact when a == b."""
    if lo > hi:
        return True
    blocks = [(lo, hi)]
    while blocks:
        A, Bb = [], []
        for a, b in blocks:
            edges = np.unique(np.linspace(a, b + 1, min(fan, b - a + 1) + 1).astype(np.int64))
            A.extend(edges[:-1]); Bb.extend(edges[1:] - 1)
        A = np.asarray(A, dtype=np.int64); Bb = np.asarray(Bb, dtype=np.int64)
        val = bound(A, Bb)
        bad = ~(val <= thr)                                   # NaN counts as a violation
        if np.any(bad & (A == Bb)):
            return False
        blocks = list(zip(A[bad], Bb[bad]))
    return True


def _bb_interval(N, n, x, gamma):
    """Exact two-sided hypergeometric interval for the count, failure <= gamma/2 per tail."""
    lvl = float(gamma) / 2 * (1 - MARGIN)
    return _core().exact_hypergeom_count_interval(int(N), int(n), int(x), lvl)


_CORE = None


def _core():
    """The frozen exact-interval routine (same code as SRAC); loaded lazily by hash."""
    global _CORE
    if _CORE is None:
        import hashlib, importlib.util, sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        p = root / "auditability_certificate_v1/analysis/frozen/cifar10_semantic_overlap_v1/run_protected_group_family_srac.py"
        if hashlib.sha256(p.read_bytes()).hexdigest() != "9171814bd8be311461cb6c3a493be1b8faba58c06dbd76f497d51c47bd9d5422":
            raise RuntimeError("frozen core hash mismatch")
        spec = importlib.util.spec_from_file_location("frozen_family_core_v11", p)
        m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
        _CORE = m
    return _CORE


def _alpha_gamma(alpha, gamma):
    alpha = float(alpha)
    gamma = float(GAMMA_SHARE) * alpha if gamma is None else float(gamma)
    return alpha, gamma, (alpha - gamma) * (1 - MARGIN)


def recall_tests(N1, n1, x1, N0, n0, x0, tau, alpha, gamma=None):
    """UI tests of the linear assertion (1-tau) c - tau m >= 0, tau = p/q, i.e. (q-p) c - p m >= 0.
    c: poisons among the N1 flagged records, m: poisons among the N0 unflagged records.
    Statistic S = A x1 - B x0 with A = (q-p) N1 n0', B = p N0 n1' (n' = max(n, 1)); a stratum
    with n = 0 drops out.  S is a positive multiple of the stratified estimate of (q-p) c - p m.
    GO   (H0: recall < tau, i.e. m > (q-p) c / p): p(c) = P(S >= s | c, m_go(c)); increasing in c,
         decreasing in m, m_go increasing, so max over [a, b] <= P(S >= s | b, m_go(a)).
    NO-GO (H0: recall >= tau, i.e. m <= (q-p) c / p): p(c) = P(S <= s | c, m_ng(c)) <=
         P(S <= s | a, m_ng(b)) over [a, b].
    The population with c = m = 0 is in both nulls (vacuous recall)."""
    tau = rational(tau); P, Q = tau.numerator, tau.denominator
    if not 0 < tau < 1:
        raise ValueError("tau must lie in (0, 1)")
    w1, w0 = Q - P, P
    N1, n1, x1, N0, n0, x0 = map(int, (N1, n1, x1, N0, n0, x0))
    if x1 == 0 and x0 == 0:
        return False, False
    A = w1 * N1 * max(n0, 1) if n1 > 0 else 0
    B = w0 * N0 * max(n1, 1) if n0 > 0 else 0
    s = A * x1 - B * x0
    alpha, gamma, thr = _alpha_gamma(alpha, gamma)
    c_lo, c_hi = (x1, N1 - (n1 - x1)) if n1 > 0 else (0, N1)
    m_max = N0 - (n0 - x0) if n0 > 0 else N0
    if 0 < n1 < N1:
        r = _bb_interval(N1, n1, x1, gamma); c_lo, c_hi = max(c_lo, r.lower), min(c_hi, r.upper)
    m_go = lambda c: np.maximum((w1 * np.asarray(c, dtype=np.int64)) // w0 + 1, x0)
    m_ng = lambda c: np.minimum((w1 * np.asarray(c, dtype=np.int64)) // w0, m_max)
    # GO-null points need m_go(c) <= m_max  <=>  w1 c < w0 m_max
    go_hi = min(c_hi, _ceil_div(w0 * m_max, w1) - 1)
    go = sup_at_most(c_lo, go_hi, lambda a, b: linear_p(N1, n1, b, A, N0, n0, m_go(a), B, s, "ge"), thr)
    # NO-GO-null points need m_ng(c) >= x0  <=>  w1 c >= w0 x0   (and c >= 1 unless c = m = 0)
    ng_lo = max(c_lo, 1, _ceil_div(w0 * x0, w1))
    ng = sup_at_most(ng_lo, c_hi, lambda a, b: linear_p(N1, n1, a, A, N0, n0, m_ng(b), B, s, "le"), thr)
    return go, ng


def group_tests(fs, n1, y1, us, n0, y0, cap, alpha, gamma=None):
    """UI tests of the clean-flag-rate assertion CF / (CF + CU) <= cap, cap = a/b, i.e.
    (b-a) CF - a CU <= 0.  CF/CU: clean records of group H in the flagged/unflagged stratum
    (sizes fs/us); y1/y0: clean among the n1/n0 sampled members of H in each stratum.
    Statistic S = A y1 - B y0, A = (b-a) fs n0', B = a us n1'.
    GO   (H0: F > cap, CF >= cf_go(CU)): p = P(S <= s | cf_go(CU), CU) <= P(S <= s | cf_go(lo), hi).
    NO-GO (H0: F <= cap, CF <= cf_ng(CU)): p = P(S >= s | cf_ng(CU), CU) <= P(S >= s | cf_ng(hi), lo).
    A group with no clean records is conservatively kept in the GO null."""
    cap = rational(cap); a_, b_ = cap.numerator, cap.denominator
    if not 0 < cap < 1:
        raise ValueError("cap must lie in (0, 1)")
    w1, w0 = b_ - a_, a_
    fs, n1, y1, us, n0, y0 = map(int, (fs, n1, y1, us, n0, y0))
    A = w1 * fs * max(n0, 1) if n1 > 0 else 0
    B = w0 * us * max(n1, 1) if n0 > 0 else 0
    s = A * y1 - B * y0
    alpha, gamma, thr = _alpha_gamma(alpha, gamma)
    cu_lo, cu_hi = (y0, us - (n0 - y0)) if n0 > 0 else (0, us)
    if 0 < n0 < us:
        r = _bb_interval(us, n0, y0, gamma); cu_lo, cu_hi = max(cu_lo, r.lower), min(cu_hi, r.upper)
    cf_min, cf_max = (y1, fs - (n1 - y1)) if n1 > 0 else (0, fs)
    cf_go = lambda cu: np.maximum((w0 * np.asarray(cu, dtype=np.int64)) // w1 + 1, cf_min)
    cf_ng = lambda cu: np.minimum((w0 * np.asarray(cu, dtype=np.int64)) // w1, cf_max)
    # GO-null points need cf_go(CU) <= cf_max  <=>  w0 CU < w1 cf_max
    go_hi = min(cu_hi, _ceil_div(w1 * cf_max, w0) - 1)
    go = sup_at_most(cu_lo, go_hi, lambda lo, hi: linear_p(fs, n1, cf_go(lo), A, us, n0, hi, B, s, "le"), thr)
    if cu_lo == 0 and cf_min == 0:
        go = False
    # NO-GO-null points need cf_ng(CU) >= cf_min  <=>  w0 CU >= w1 cf_min
    ng_lo = max(cu_lo, _ceil_div(w1 * cf_min, w0))
    ng = sup_at_most(ng_lo, cu_hi, lambda lo, hi: linear_p(fs, n1, cf_ng(hi), A, us, n0, lo, B, s, "ge"), thr)
    return go, ng


def decide_ui(pop, chosen, alpha, tau, cap):
    Z = pop["poison"]; M = pop["memberships"]; s1, s0 = chosen[1], chosen[0]
    go, ng = recall_tests(pop["N"][1], len(s1), int(Z[s1].sum()), pop["N"][0], len(s0), int(Z[s0].sum()), tau, alpha)
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            continue
        a1 = s1[M[s1, g]]; a0 = s0[M[s0, g]]
        g_go, g_ng = group_tests(fs, len(a1), int((~Z[a1]).sum()), us, len(a0), int((~Z[a0]).sum()), cap, alpha)
        go &= g_go; ng |= g_ng
    return "GO" if go else ("NO_GO" if ng else "ABSTAIN")


def n_assertions(pop):
    return 1 + sum(1 for fs, _ in pop["gsize"] if fs > 0)
