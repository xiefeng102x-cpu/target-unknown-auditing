"""Target-unknown second stage for SCAn: clean-calibrated LDA expansion.

For every class that SCAn flags, fit a regularised linear discriminant between
the trusted clean references of that class and the SCAn-flagged members of that
class.  Score every member assigned to the class, convert scores to conformal
p-values against a disjoint trusted clean calibration set of the same class, and
add members with p <= q to the SCAn flags.  No attack-target identity is used:
the class comes from SCAn and the threshold from clean held-out data.
"""
from __future__ import annotations

import numpy as np

FROZEN_PARAMETERS = {"pca_dimension": 32, "relative_ridge": 0.1, "calibration_q": 0.05}


def lda_scores(reference: np.ndarray, flagged: np.ndarray, values: np.ndarray, *,
               pca_dimension: int, relative_ridge: float) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    flagged = np.asarray(flagged, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(reference) < 2 or len(flagged) < 2:
        raise ValueError("LDA expansion needs at least two reference and two flagged rows")
    mu_ref = reference.mean(axis=0)
    pooled = np.vstack([reference - mu_ref, flagged - flagged.mean(axis=0)])
    _, singular, vt = np.linalg.svd(pooled, full_matrices=False)
    dim = int(min(pca_dimension, vt.shape[0]))
    basis = vt[:dim].T
    variance = singular[:dim] ** 2 / (len(pooled) - 1)
    variance = variance + float(relative_ridge) * variance.mean()
    weights = ((flagged.mean(axis=0) - mu_ref) @ basis) / variance
    return ((values - mu_ref) @ basis) @ weights


def conformal_pvalues(calibration_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    calibration_scores = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    at_least = len(calibration_scores) - np.searchsorted(calibration_scores, scores, side="left")
    return (1.0 + at_least) / (len(calibration_scores) + 1.0)


def expand_class(reference: np.ndarray, calibration: np.ndarray, members: np.ndarray,
                 scan_flags: np.ndarray, *, pca_dimension: int, relative_ridge: float,
                 calibration_q: float) -> tuple[np.ndarray, dict]:
    """Return expanded flags (SCAn OR LDA) for the members of one SCAn-flagged class."""
    scan_flags = np.asarray(scan_flags, dtype=bool)
    if scan_flags.sum() < 2:
        return scan_flags.copy(), {"expanded": False, "reason": "fewer_than_two_scan_flags"}
    score = lambda x: lda_scores(reference, members[scan_flags], x,
                                 pca_dimension=pca_dimension, relative_ridge=relative_ridge)
    pvalues = conformal_pvalues(score(calibration), score(members))
    added = (pvalues <= calibration_q) & ~scan_flags
    return scan_flags | added, {"expanded": True, "added": int(added.sum()),
                                "calibration_rows": int(len(calibration)),
                                "reference_rows": int(len(reference))}


def selftest() -> None:
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, (800, 20)); poison = rng.normal(0, 1, (300, 20)); poison[:, 0] += 4
    reference, calibration, members_clean = clean[:250], clean[250:500], clean[500:]
    members = np.vstack([members_clean, poison]); truth = np.r_[np.zeros(300, bool), np.ones(300, bool)]
    scan = truth & (rng.random(600) < 0.6)
    flags, info = expand_class(reference, calibration, members, scan, **FROZEN_PARAMETERS)
    assert flags[truth].mean() > 0.95 and flags[~truth].mean() < 0.12, (flags[truth].mean(), flags[~truth].mean())
    p = conformal_pvalues(np.arange(10.0), np.array([-1.0, 4.5, 100.0]))
    assert np.allclose(p, [11 / 11, 6 / 11, 1 / 11])
    print("scan_lda_expansion selftest PASS", info)


if __name__ == "__main__":
    selftest()
