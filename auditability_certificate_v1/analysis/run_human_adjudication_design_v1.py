"""Post-hoc human-adjudication design analysis V1 (developmental; not confirmation).

Adjudication by r independent human reviewers with majority vote.  Each reviewer
mislabels a record with probability e; a fraction c of records are "hard" and draw one
shared outcome for all reviewers (fully correlated errors).  The effective per-record
error of the majority label is

    eps(e, r, c) = c * e + (1 - c) * sum_{j > r/2} C(r, j) e^j (1 - e)^(r - j).

For each design with eps < 0.005 (Table V already shows zero power at eps >= 0.005), the
robust group-safe certificate of run_adjudication_sensitivity_v1.py is run on the frozen
flags, families, and draws of blocks 8 and 12: delta/2 for the count intervals, delta/2
for the error event, B = Binomial(k, eps) upper (1 - delta/2) quantile, hull
[L(x - B), U(x + B)].  Sampled labels are flipped independently with probability eps.
Cost is r * k human reviews per audit.  All designs are reported; nothing becomes a
frozen endpoint.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from math import comb
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]; CERT = ROOT / "auditability_certificate_v1"
spec = importlib.util.spec_from_file_location("adj", CERT / "analysis/run_adjudication_sensitivity_v1.py")
A = importlib.util.module_from_spec(spec); sys.modules["adj"] = A; spec.loader.exec_module(A)
H = A.H
REVIEWER_ERRORS = (0.005, 0.01, 0.02, 0.05)
PANEL_SIZES = (1, 3, 5)
CORRELATED_SHARES = (0.0, 0.1)
EPS_CEILING = 0.005


def majority_error(e: float, r: int) -> float:
    return sum(comb(r, j) * e ** j * (1 - e) ** (r - j) for j in range(r // 2 + 1, r + 1))


def designs() -> list[dict]:
    out = []
    for c in CORRELATED_SHARES:
        for e in REVIEWER_ERRORS:
            for r in PANEL_SIZES:
                eps = c * e + (1 - c) * majority_error(e, r)
                out.append(dict(c=c, e=e, r=r, eps=eps, run=bool(eps < EPS_CEILING)))
    return out


def run_victim(args):
    ds, victim, frame, cfg, plan = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); q = pop["q"]
    true_pass = pop["recall"] >= tau and pop["max_clean"] <= cap
    ns = cfg["ns"].format(v=victim); poison = pop["poison"]; rows = []
    for k in c["budgets"]:
        n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
        for draw in range(c["protected_draws_per_seed_budget"]):
            chosen = {}
            for f, n in ((1, n1), (0, n0)):
                rng = np.random.default_rng(H.stable_seed(ns, k, draw, f))
                chosen[f] = rng.choice(pop["idx"][f], size=n, replace=False)
            sampled = np.concatenate([chosen[1], chosen[0]])
            for d in plan:
                obs = poison.copy()
                flip = np.random.default_rng(H.stable_seed("human-adj-v1", ns, k, draw, d["c"], d["e"], d["r"])).random(len(sampled)) < d["eps"]
                obs[sampled[flip]] = ~obs[sampled[flip]]
                B = int(binom.ppf(1 - delta / 2, k, d["eps"]))
                dec = A.decide(pop, chosen, obs, (delta / 2) / (2 * q), B, tau, cap)
                rows.append(dict(dataset=ds, victim_id=victim, k=k, draw=draw, c=d["c"], e=d["e"], r=d["r"], eps=d["eps"],
                                 B=B, errors=int(flip.sum()), decision=dec, true_pass=int(true_pass), reviews=d["r"] * k))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    allp = designs(); plan = [d for d in allp if d["run"]]
    jobs = []
    for ds, cfg in A.DATASETS.items():
        f = pd.read_csv(cfg["flags"]); f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            jobs.append((ds, v, fr.reset_index(drop=True)[["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]],
                         {k: str(x) for k, x in cfg.items()}, plan))
    with Pool(a.procs) as p:
        out = pd.concat(p.map(run_victim, jobs), ignore_index=True)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    a.output.mkdir(parents=True)
    out.to_csv(a.output / "decisions.csv", index=False)
    pd.DataFrame(allp).to_csv(a.output / "designs.csv", index=False)
    s = out.groupby(["dataset", "c", "e", "r", "eps", "k", "true_pass"]).agg(
        n=("decision", "size"), B=("B", "max"), go=("decision", lambda x: float((x == "GO").mean())),
        nogo=("decision", lambda x: float((x == "NO_GO").mean())), false_go=("false_go", "sum"),
        false_nogo=("false_nogo", "sum"), reviews=("reviews", "max")).reset_index()
    s.to_csv(a.output / "summary.csv", index=False)
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rep = dict(rows=len(out), designs_run=len(plan), false_go=int(out.false_go.sum()), false_nogo=int(out.false_nogo.sum()),
               script_sha256=h(__file__), decisions_sha256=h(a.output / "decisions.csv"), summary_sha256=h(a.output / "summary.csv"),
               designs_sha256=h(a.output / "designs.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
