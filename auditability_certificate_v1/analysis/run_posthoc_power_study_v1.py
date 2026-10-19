"""Post-hoc audit-power study on immutable frozen flags (developmental, not confirmation).

Compares, on the same frozen populations, (i) tail/budget allocation rules and
(ii) design-only sampling allocations, plus (iii) a nested anytime-valid
escalation 2000 -> 8000 -> 16000 with a Bonferroni split over looks.
Every rule is a valid fixed-data certificate for its declared error budget.
All combinations are reported; none is promoted to a confirmation endpoint.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
FROZEN_CORE = ROOT / "auditability_certificate_v1/analysis/frozen/cifar10_semantic_overlap_v1/run_protected_group_family_srac.py"
FROZEN_CORE_SHA = "9171814bd8be311461cb6c3a493be1b8faba58c06dbd76f497d51c47bd9d5422"

DATASETS = {
    "gtsrb_semantic_v2": dict(
        protocol=CERT / "PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V2.json",
        flags=ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v2/flags.csv",
        frozen_decisions=CERT / "results_gtsrb_semantic_overlap_confirmation_v2/decisions.csv",
        namespace="gtsrb-semantic-overlap-confirm-v1|{victim}",
        n_classes=43,
    ),
    "cifar10_semantic_v1": dict(
        protocol=CERT / "PROTOCOL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION_V1.json",
        flags=ROOT / "results/cifar10_semantic_overlap_scan_scores_confirmation_v1/flags.csv",
        frozen_decisions=CERT / "results_cifar10_semantic_overlap_confirmation_v1/decisions.csv",
        namespace="cifar10-semantic-overlap-confirm-v1|{victim}",
        n_classes=10,
    ),
}
METHODS = ("twosided", "onesided", "onesided_split")
SHARES = ("0.5", "0.25", "0.1", "prop")
ESCALATION_SHARES = ("0.5", "0.25")
ESCALATION_METHOD = "onesided_split"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core():
    if sha256(FROZEN_CORE) != FROZEN_CORE_SHA:
        raise RuntimeError("frozen core hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_family_core", FROZEN_CORE)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CORE = load_core()
interval = CORE.exact_hypergeom_count_interval
stable_seed = CORE.stable_seed


def etas(method: str, delta: float, q: int) -> dict[str, float]:
    """Per-tail failure probability for recall (R) and group-cell (G) intervals."""
    if method == "twosided":      # frozen rule: both tails of all Q intervals share delta
        e = delta / (2 * q); return dict(R=e, G=e)
    if method == "onesided":      # false GO <= delta and false NO-GO <= delta separately
        e = delta / q; return dict(R=e, G=e)
    if method == "onesided_split":  # per error type; delta/2 to the 2 recall cells, delta/2 to group cells
        return dict(R=delta / 4, G=delta / (2 * (q - 2)) if q > 2 else delta / 2)
    raise ValueError(method)


def allocate(n1_cap: int, n0_cap: int, k: int, share: str) -> tuple[int, int]:
    s = (n1_cap / (n1_cap + n0_cap)) if share == "prop" else float(share)
    n1 = min(n1_cap, int(round(k * s)))
    n0 = min(n0_cap, k - n1)
    if n1 + n0 < k:
        n1 = min(n1_cap, k - n0)
    if n1 + n0 != k:
        raise RuntimeError("allocation failed")
    return n1, n0


def ratio(a, b, empty):
    d = a + b
    return a / d if d else empty


def decide(pop, chosen, eta, tau, cap):
    """pop: dict with flags, poison, memberships, cells. chosen: {1: idx, 0: idx}."""
    poison, M = pop["poison"], pop["memberships"]
    def side(n_pop, idx, et, which):
        iv = interval(int(n_pop), int(len(idx)), int(poison[idx].sum()), float(et))
        return iv.lower if which == "L" else iv.upper
    N1, N0 = pop["N"][1], pop["N"][0]
    # GO direction
    cL = side(N1, chosen[1], eta["R"], "L"); mU = side(N0, chosen[0], eta["R"], "U")
    rec_lo = ratio(cL, mU, 0.0)
    # NO-GO direction
    cU = side(N1, chosen[1], eta["R"], "U"); mL = side(N0, chosen[0], eta["R"], "L")
    rec_hi = ratio(cU, mL, 1.0)
    if cL + mL == 0:  # zero-poison population admissible -> vacuous recall
        rec_lo, rec_hi = 0.0, 1.0
    go_groups, nogo_groups, max_hi = True, False, 0.0
    for g in range(M.shape[1]):
        f_size, u_size = pop["gsize"][g]
        if f_size == 0:
            continue
        in1 = chosen[1][M[chosen[1], g]]; in0 = chosen[0][M[chosen[0], g]]
        fL = side(f_size, in1, eta["G"], "L"); fU = side(f_size, in1, eta["G"], "U")
        if u_size:
            uL = side(u_size, in0, eta["G"], "L"); uU = side(u_size, in0, eta["G"], "U")
        else:
            uL = uU = 0
        hi = ratio(f_size - fL, u_size - uU, 1.0)
        lo = ratio(f_size - fU, u_size - uL, 0.0)
        max_hi = max(max_hi, hi)
        go_groups &= hi <= cap
        nogo_groups |= lo > cap
    if rec_lo >= tau and go_groups:
        d = "GO"
    elif rec_hi < tau or nogo_groups:
        d = "NO_GO"
    else:
        d = "ABSTAIN"
    return d, rec_lo, rec_hi, max_hi


def build_population(frame, groups):
    flags = frame.flag.to_numpy(dtype=np.int8)
    poison = frame.is_poison.to_numpy(dtype=bool)
    labels = frame.assigned_group.to_numpy(dtype=int)
    M = np.column_stack([np.isin(labels, g["class_ids"]) for g in groups])
    cells = sum(1 for g in range(M.shape[1]) for f in (1, 0) if np.any(M[:, g] & (flags == f)))
    gsize = [(int(np.sum(M[:, g] & (flags == 1))), int(np.sum(M[:, g] & (flags == 0)))) for g in range(M.shape[1])]
    idx = {f: np.flatnonzero(flags == f) for f in (1, 0)}
    clean_rates = [flags[(~poison) & M[:, g]].mean() for g in range(M.shape[1]) if np.any((~poison) & M[:, g])]
    recall = float(flags[poison].mean())
    return dict(flags=flags, poison=poison, memberships=M, q=2 + cells, gsize=gsize, idx=idx,
                N={1: len(idx[1]), 0: len(idx[0])}, recall=recall, max_clean=float(max(clean_rates)))


def run_victim(args):
    ds, victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text())
    c = proto["certificate"]; delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    budgets, draws = c["budgets"], c["protected_draws_per_seed_budget"]
    pop = build_population(frame, proto["protected_groups"])
    true_pass = pop["recall"] >= tau and pop["max_clean"] <= cap
    ns = cfg["namespace"].format(victim=victim)
    rows = []
    # fixed-budget grid
    for method in METHODS:
        eta = etas(method, delta, pop["q"])
        for share in SHARES:
            for k in budgets:
                n1, n0 = allocate(pop["N"][1], pop["N"][0], k, share)
                for draw in range(draws):
                    chosen = {}
                    for f, n in ((1, n1), (0, n0)):
                        rng = np.random.default_rng(stable_seed(ns, k, draw, f))
                        chosen[f] = rng.choice(pop["idx"][f], size=n, replace=False) if n else np.empty(0, int)
                    d, lo, hi, mh = decide(pop, chosen, eta, tau, cap)
                    rows.append(dict(dataset=ds, victim_id=victim, design="fixed", method=method, share=share, k=k,
                                     draw=draw, n1=n1, n0=n0, decision=d, recall_lo=lo, recall_hi=hi, max_group_clean_hi=mh,
                                     true_pass=int(true_pass), actual_recall=pop["recall"], actual_max_group_clean=pop["max_clean"],
                                     k_used=k))
    # nested anytime-valid escalation: Bonferroni delta/len(budgets) per look, stop at first non-ABSTAIN
    eta = etas(ESCALATION_METHOD, delta / len(budgets), pop["q"])
    for share in ESCALATION_SHARES:
        for draw in range(draws):
            perm = {f: np.random.default_rng(stable_seed("posthoc-escalation-v1", ns, share, draw, f)).permutation(pop["idx"][f]) for f in (1, 0)}
            final, used, lo, hi = "ABSTAIN", budgets[-1], None, None
            for k in budgets:
                n1, n0 = allocate(pop["N"][1], pop["N"][0], k, share)
                d, lo, hi, mh = decide(pop, {1: perm[1][:n1], 0: perm[0][:n0]}, eta, tau, cap)
                if d != "ABSTAIN":
                    final, used = d, k
                    break
            rows.append(dict(dataset=ds, victim_id=victim, design="escalation", method=ESCALATION_METHOD, share=share,
                             k=budgets[-1], draw=draw, n1=None, n0=None, decision=final, recall_lo=lo, recall_hi=hi,
                             max_group_clean_hi=None, true_pass=int(true_pass), actual_recall=pop["recall"],
                             actual_max_group_clean=pop["max_clean"], k_used=used))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=4); a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    jobs = []
    for ds, cfg in DATASETS.items():
        f = pd.read_csv(cfg["flags"]); f = f[f.victim_eligible.eq(1)]
        for victim, frame in f.groupby("victim_id", sort=True):
            jobs.append((ds, victim, frame.reset_index(drop=True), {k: str(v) for k, v in cfg.items()}))
    with Pool(a.procs) as p:
        out = pd.concat(p.map(run_victim, jobs), ignore_index=True)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    # validation: twosided + 0.5 must equal frozen decisions
    val = {}
    for ds, cfg in DATASETS.items():
        fz = pd.read_csv(cfg["frozen_decisions"])[["victim_id", "k", "draw", "decision"]]
        mine = out[(out.dataset == ds) & (out.design == "fixed") & (out.method == "twosided") & (out.share == "0.5")][["victim_id", "k", "draw", "decision"]]
        m = fz.merge(mine, on=["victim_id", "k", "draw"], suffixes=("_frozen", "_mine"))
        val[ds] = dict(rows_frozen=len(fz), rows_matched=len(m), identical=bool(len(m) == len(fz) and (m.decision_frozen == m.decision_mine).all()))
    a.output.mkdir(parents=True)
    out.to_csv(a.output / "decisions.csv", index=False)
    g = out.groupby(["dataset", "design", "method", "share", "k", "true_pass"])
    summ = g.agg(decisions=("decision", "size"), go=("decision", lambda s: (s == "GO").mean()),
                 nogo=("decision", lambda s: (s == "NO_GO").mean()), abstain=("decision", lambda s: (s == "ABSTAIN").mean()),
                 false_go=("false_go", "sum"), false_nogo=("false_nogo", "sum"), mean_k_used=("k_used", "mean")).reset_index()
    summ.to_csv(a.output / "summary.csv", index=False)
    rep = dict(validation=val, total_rows=len(out), false_go_total=int(out.false_go.sum()), false_nogo_total=int(out.false_nogo.sum()),
               script_sha256=sha256(Path(__file__)), frozen_core_sha256=FROZEN_CORE_SHA,
               decisions_sha256=sha256(a.output / "decisions.csv"), summary_sha256=sha256(a.output / "summary.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
