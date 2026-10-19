"""Post-hoc used-cell error-budget analysis V1 (developmental; not confirmation).

The frozen family core sets eta = delta / (2Q) with Q = 2 + every non-empty
protected-set/flag cell.  A declared set with no flagged record has F_H = 0 exactly,
and its cells never enter a decision.  Because flags are frozen before the draw,
counting only the cells that can enter a decision,

    Q_used = 2 + sum over declared H with N_H1 > 0 of (1 + 1[N_H0 > 0]),

is fixed before audit randomness and keeps Proposition 3 valid.  This script compares
the frozen rule ("all") with the used-cell rule ("used") on the frozen flags and draw
namespaces of blocks 8, 11, and 12, for the frozen family F0 and for F0 plus every
singleton class (F1).  Everything else is the frozen design: two-sided exact
hypergeometric intervals, 50/50 flag allocation, k in {2000, 8000, 16000}, 200 draws,
delta = 0.005.  F0/all must reproduce each frozen endpoint decision by decision.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; CERT = ROOT / "auditability_certificate_v1"
spec = importlib.util.spec_from_file_location("ph", CERT / "analysis/run_posthoc_power_study_v1.py")
H = importlib.util.module_from_spec(spec); sys.modules["ph"] = H; spec.loader.exec_module(H)
DATASETS = {
    "block8_gtsrb_semantic_v2": dict(protocol=CERT / "PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V2.json",
        flags=ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v2/flags.csv",
        frozen=CERT / "results_gtsrb_semantic_overlap_confirmation_v2/decisions.csv",
        ns="gtsrb-semantic-overlap-confirm-v1|{v}"),
    "block11_cifar10_semantic_v1": dict(protocol=CERT / "PROTOCOL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION_V1.json",
        flags=ROOT / "results/cifar10_semantic_overlap_scan_scores_confirmation_v1/flags.csv",
        frozen=CERT / "results_cifar10_semantic_overlap_confirmation_v1/decisions.csv",
        ns="cifar10-semantic-overlap-confirm-v1|{v}"),
    "block12_cifar10_scan_lda_v1": dict(protocol=CERT / "PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json",
        flags=ROOT / "results/cifar10_scan_lda_scores_confirmation_v1/flags.csv",
        frozen=CERT / "results_cifar10_scan_lda_confirmation_v1/decisions.csv",
        ns="cifar10-scan-lda-confirm-v1|{v}"),
}


def q_used(pop):
    return 2 + sum((1 + (1 if us else 0)) for fs, us in pop["gsize"] if fs > 0)


def run_victim(args):
    ds, victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    f0 = list(proto["protected_groups"])
    f1 = f0 + [dict(name=f"class_{k}", class_ids=[k]) for k in sorted(set(int(x) for x in frame.assigned_group))]
    ns = cfg["ns"].format(v=victim); rows = []
    for fam, groups in (("F0", f0), ("F1", f1)):
        pop = H.build_population(frame, groups)
        tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
        for rule, q in (("all", pop["q"]), ("used", q_used(pop))):
            e = delta / (2 * q)
            for k in c["budgets"]:
                n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
                for draw in range(c["protected_draws_per_seed_budget"]):
                    chosen = {}
                    for f, n in ((1, n1), (0, n0)):
                        rng = np.random.default_rng(H.stable_seed(ns, k, draw, f))
                        chosen[f] = rng.choice(pop["idx"][f], size=n, replace=False) if n else np.empty(0, int)
                    d, _, _, _ = H.decide(pop, chosen, dict(R=e, G=e), tau, cap)
                    rows.append(dict(dataset=ds, victim_id=victim, family=fam, rule=rule, Q=q, k=k, draw=draw,
                                     decision=d, true_pass=tp))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    jobs = []
    for ds, cfg in DATASETS.items():
        f = pd.read_csv(cfg["flags"]); f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            jobs.append((ds, v, fr.reset_index(drop=True)[["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]],
                         {k: str(x) for k, x in cfg.items()}))
    with Pool(a.procs) as p:
        out = pd.concat(p.map(run_victim, jobs), ignore_index=True)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    val = {}
    for ds, cfg in DATASETS.items():
        fz = pd.read_csv(cfg["frozen"])[["victim_id", "k", "draw", "decision"]]
        mine = out[(out.dataset == ds) & (out.family == "F0") & (out.rule == "all")][["victim_id", "k", "draw", "decision"]]
        m = fz.merge(mine, on=["victim_id", "k", "draw"], suffixes=("_f", "_m"))
        val[ds] = dict(rows=len(fz), matched=len(m), identical=bool(len(m) == len(fz) and (m.decision_f == m.decision_m).all()))
    a.output.mkdir(parents=True); out.to_csv(a.output / "decisions.csv", index=False)
    g = lambda x, s: (x == s).mean()
    s = out.groupby(["dataset", "family", "rule", "k", "true_pass"]).agg(n=("decision", "size"), Q=("Q", "max"),
        go=("decision", lambda x: g(x, "GO")), nogo=("decision", lambda x: g(x, "NO_GO")),
        false_go=("false_go", "sum"), false_nogo=("false_nogo", "sum")).reset_index()
    s.to_csv(a.output / "summary.csv", index=False)
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rep = dict(validation=val, rows=len(out), script_sha256=h(__file__), decisions_sha256=h(a.output / "decisions.csv"),
               summary_sha256=h(a.output / "summary.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
