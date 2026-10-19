"""Post-hoc family-resolution analysis V1 (developmental; not confirmation).

Question: what does the declared protected family actually protect, and what does
adding every singleton class to the family cost?

Inputs are the immutable frozen flags, protocols, and draw namespaces of the GTSRB
semantic-overlap V2 block (block 8) and the CIFAR-10 SCAn+LDA block (block 12).
Everything except the family is frozen: two-sided exact hypergeometric intervals,
eta = delta / (2Q), 50/50 flag-stratum allocation, budgets 2000/8000/16000,
delta = 0.005, tau = 0.8, cap = 0.1.

Families
  F0  frozen declared semantic family.
  F1  F0 plus one singleton group per class that has at least one record.

Part A (real screens): all 200 frozen draws per victim and budget under F0 and F1.
  F0 must reproduce the frozen endpoint decision by decision.
Part B (hidden-erasure capacity, exact, no sampling): for every victim and class c,
  m_c is the largest number of currently unflagged clean records of c whose
  additional flagging keeps every F0 group's true clean flag rate <= cap.
  Reported as the fraction m_c / (clean records of c) ("hidden erasure").
Part C (stress screens): for every victim and every class whose hidden erasure is
  >= 0.10 (so that the stressed screen violates the F1 singleton cap), flag the first
  m_c unflagged clean records of c in sha256(sample_id) order, then certify under F0
  and F1 with the first 50 frozen draw indices per budget.  New flags change the
  strata, so these are fresh protected draws from the same seed namespace.
All cells are reported; nothing is promoted to a frozen endpoint.
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
    "gtsrb_semantic_v2": dict(protocol=CERT / "PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V2.json",
        flags=ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v2/flags.csv",
        frozen=CERT / "results_gtsrb_semantic_overlap_confirmation_v2/decisions.csv",
        ns="gtsrb-semantic-overlap-confirm-v1|{v}", n_classes=43),
    "cifar10_scan_lda_v1": dict(protocol=CERT / "PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json",
        flags=ROOT / "results/cifar10_scan_lda_scores_confirmation_v1/flags.csv",
        frozen=CERT / "results_cifar10_scan_lda_confirmation_v1/decisions.csv",
        ns="cifar10-scan-lda-confirm-v1|{v}", n_classes=10),
}
STRESS_DRAWS = 50
STRESS_MIN_FRACTION = 0.10


def families(proto, frame):
    f0 = list(proto["protected_groups"])
    present = sorted(set(int(x) for x in frame.assigned_group))
    f1 = f0 + [dict(name=f"class_{c}", class_ids=[c]) for c in present]
    return dict(F0=f0, F1=f1)


def certify(pop, ns, k, draw, delta, tau, cap):
    n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
    chosen = {}
    for f, n in ((1, n1), (0, n0)):
        rng = np.random.default_rng(H.stable_seed(ns, k, draw, f))
        chosen[f] = rng.choice(pop["idx"][f], size=n, replace=False) if n else np.empty(0, int)
    e = delta / (2 * pop["q"])
    d, _, _, mh = H.decide(pop, chosen, dict(R=e, G=e), tau, cap)
    return d, mh


def hidden_erasure(frame, f0, cap):
    lab = frame.assigned_group.to_numpy(int); flag = frame.flag.to_numpy(int); clean = frame.is_poison.to_numpy(int) == 0
    out = {}
    for c in sorted(set(lab)):
        avail = int(np.sum((lab == c) & clean & (flag == 0))); n_clean = int(np.sum((lab == c) & clean))
        m = avail
        for g in f0:
            if c not in g["class_ids"]:
                continue
            ing = np.isin(lab, g["class_ids"]) & clean
            room = int(np.floor(cap * ing.sum() + 1e-9)) - int(np.sum(ing & (flag == 1)))
            m = min(m, max(0, room))
        out[c] = dict(n_clean=n_clean, already_flagged=int(np.sum((lab == c) & clean & (flag == 1))), hidden_m=m,
                      hidden_fraction=m / n_clean if n_clean else 0.0)
    return out


def run_victim(args):
    ds, victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    ns = cfg["ns"].format(v=victim); fams = families(proto, frame)
    rows, cap_rows = [], []
    for fam, groups in fams.items():
        pop = H.build_population(frame, groups)
        tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
        for k in c["budgets"]:
            for draw in range(c["protected_draws_per_seed_budget"]):
                d, mh = certify(pop, ns, k, draw, delta, tau, cap)
                rows.append(dict(part="A", dataset=ds, victim_id=victim, family=fam, Q=pop["q"], stress_class=-1,
                                 stress_fraction=0.0, k=k, draw=draw, decision=d, true_pass=tp,
                                 actual_recall=pop["recall"], actual_max_clean=pop["max_clean"], max_group_clean_hi=mh))
    he = hidden_erasure(frame, fams["F0"], cap)
    for cl, v in he.items():
        cap_rows.append(dict(dataset=ds, victim_id=victim, cls=cl, **v))
    order = np.argsort([hashlib.sha256(str(s).encode()).hexdigest() for s in frame.sample_id])
    lab = frame.assigned_group.to_numpy(int); clean = frame.is_poison.to_numpy(int) == 0
    for cl, v in he.items():
        if v["hidden_fraction"] < STRESS_MIN_FRACTION:
            continue
        cand = [i for i in order if lab[i] == cl and clean[i] and frame.flag.iat[i] == 0][: v["hidden_m"]]
        st = frame.copy(); st.loc[st.index[cand], "flag"] = 1
        for fam, groups in fams.items():
            pop = H.build_population(st, groups)
            tp = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
            for k in c["budgets"]:
                for draw in range(STRESS_DRAWS):
                    d, mh = certify(pop, ns, k, draw, delta, tau, cap)
                    rows.append(dict(part="C", dataset=ds, victim_id=victim, family=fam, Q=pop["q"], stress_class=cl,
                                     stress_fraction=v["hidden_fraction"], k=k, draw=draw, decision=d, true_pass=tp,
                                     actual_recall=pop["recall"], actual_max_clean=pop["max_clean"], max_group_clean_hi=mh))
    return pd.DataFrame(rows), pd.DataFrame(cap_rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    jobs = []
    for ds, cfg in DATASETS.items():
        f = pd.read_csv(cfg["flags"]); f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            if a.only and a.only not in v:
                continue
            jobs.append((ds, v, fr.reset_index(drop=True)[["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]],
                         {k: str(x) for k, x in cfg.items()}))
    with Pool(a.procs) as p:
        res = p.map(run_victim, jobs)
    out = pd.concat([r[0] for r in res], ignore_index=True); capt = pd.concat([r[1] for r in res], ignore_index=True)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    val = {}
    for ds, cfg in DATASETS.items():
        fz = pd.read_csv(cfg["frozen"])[["victim_id", "k", "draw", "decision"]]
        mine = out[(out.dataset == ds) & (out.part == "A") & (out.family == "F0")][["victim_id", "k", "draw", "decision"]]
        m = fz.merge(mine, on=["victim_id", "k", "draw"], suffixes=("_f", "_m"))
        val[ds] = dict(rows=len(fz), matched=len(m), identical=bool(len(m) == len(fz) and (m.decision_f == m.decision_m).all()))
    a.output.mkdir(parents=True)
    out.to_csv(a.output / "decisions.csv", index=False); capt.to_csv(a.output / "hidden_erasure_capacity.csv", index=False)
    g = lambda x, s: (x == s).mean()
    s = out.groupby(["part", "dataset", "family", "k", "true_pass"]).agg(n=("decision", "size"), Q=("Q", "max"),
        go=("decision", lambda x: g(x, "GO")), nogo=("decision", lambda x: g(x, "NO_GO")),
        false_go=("false_go", "sum"), false_nogo=("false_nogo", "sum")).reset_index()
    s.to_csv(a.output / "summary.csv", index=False)
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rep = dict(validation=val, rows=len(out), script_sha256=h(__file__), decisions_sha256=h(a.output / "decisions.csv"),
               capacity_sha256=h(a.output / "hidden_erasure_capacity.csv"), summary_sha256=h(a.output / "summary.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
