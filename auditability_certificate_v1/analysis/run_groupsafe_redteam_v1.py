"""Post-hoc outcome-aware label red team against the group-safe certificate (developmental).

Frozen flags, families, draws, and certificate of blocks 8, 11, 12, and 13.  After the
protected draw, an adversary who knows the draw, the flags, and every true label may
corrupt up to B = ceil(rho * k) sampled labels, rho in {0.1, 0.5, 1, 2}%.

  Failing screens (true policy fails): the adversary seeks a false GO and uses only
  flips that can only help GO: sampled unflagged poisons -> "clean" (shrinks the missed
  count and enlarges the clean-unflagged count of their groups) and sampled flagged
  clean records -> "poison" (enlarges the caught count and shrinks the clean-flagged
  count of their groups).
  Passing screens (true policy holds): the adversary seeks a false NO-GO with the mirror
  flips (flagged poisons -> "clean", unflagged clean -> "poison").

Strategies: the budget is split between the two flip types in shares 0, 1/4, 1/2, 3/4, 1
(unused budget of one type moves to the other).  Within a type, records are taken in a
priority order: (GO) members of the group with the largest clean-rate upper bound
first; (NO-GO) either members of the group with the largest clean-rate lower bound
first, or members of the smallest protected group first.  Ties are broken by a fixed
seeded permutation.  The adversary wins if any strategy reaches its target decision.

Certificates: naive (B_cert = 0, the frozen endpoint applied to corrupted labels) and
declared-B robust hull [L(x - B), U(x + B)] with the full delta (B is a deterministic
declared budget, so no error event is charged).  Validation: the unattacked naive
certificate must reproduce every frozen decision of the four blocks.  The unattacked
robust decision is also recorded (the power cost of declaring B).  Nothing becomes a
frozen endpoint.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; CERT = ROOT / "auditability_certificate_v1"
spec = importlib.util.spec_from_file_location("adj", CERT / "analysis/run_adjudication_sensitivity_v1.py")
A = importlib.util.module_from_spec(spec); sys.modules["adj"] = A; spec.loader.exec_module(A)
H = A.H
BLOCKS = {
    "8": dict(protocol=CERT / "PROTOCOL_GTSRB_SEMANTIC_OVERLAP_CONFIRMATION_V2.json",
              flags=ROOT / "results/gtsrb_semantic_overlap_scan_scores_confirmation_v2/flags.csv",
              frozen=CERT / "results_gtsrb_semantic_overlap_confirmation_v2/decisions.csv", ns="gtsrb-semantic-overlap-confirm-v1|{v}"),
    "11": dict(protocol=CERT / "PROTOCOL_CIFAR10_SEMANTIC_OVERLAP_CONFIRMATION_V1.json",
               flags=ROOT / "results/cifar10_semantic_overlap_scan_scores_confirmation_v1/flags.csv",
               frozen=CERT / "results_cifar10_semantic_overlap_confirmation_v1/decisions.csv", ns="cifar10-semantic-overlap-confirm-v1|{v}"),
    "12": dict(protocol=CERT / "PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json",
               flags=ROOT / "results/cifar10_scan_lda_scores_confirmation_v1/flags.csv",
               frozen=CERT / "results_cifar10_scan_lda_confirmation_v1/decisions.csv", ns="cifar10-scan-lda-confirm-v1|{v}"),
    "13": dict(protocol=CERT / "PROTOCOL_CIFAR10_ADAPBLEND_SCAN_LDA_CONFIRMATION_V1.json",
               flags=ROOT / "results/cifar10_adapblend_scan_lda_scores_confirmation_v1/flags.csv",
               frozen=CERT / "results_cifar10_adapblend_scan_lda_confirmation_v1/decisions.csv", ns="cifar10-adapblend-scan-lda-confirm-v1|{v}"),
}
RHOS = (0.001, 0.005, 0.01, 0.02)
SHARES = (0.0, 0.25, 0.5, 0.75, 1.0)
RANK = {"GO": {"GO": 0, "ABSTAIN": 1, "NO_GO": 2}, "NO_GO": {"NO_GO": 0, "ABSTAIN": 1, "GO": 2}}


def group_bounds(pop, chosen, obs, eta, B):
    """Per-group (lo, hi) clean-rate bounds of the hull certificate; NaN for groups without flags."""
    M = pop["memberships"]; out = []
    def iv(n_pop, idx):
        x = int(obs[idx].sum()); n = len(idx)
        return (H.interval(int(n_pop), n, max(0, x - B), float(eta)).lower, H.interval(int(n_pop), n, min(n, x + B), float(eta)).upper)
    for g in range(M.shape[1]):
        fs, us = pop["gsize"][g]
        if fs == 0:
            out.append((np.nan, np.nan)); continue
        f_lo, f_hi = iv(fs, chosen[1][M[chosen[1], g]])
        u_lo, u_hi = iv(us, chosen[0][M[chosen[0], g]]) if us else (0, 0)
        out.append((H.ratio(fs - f_hi, us - u_lo, 0.0), H.ratio(fs - f_lo, us - u_hi, 1.0)))
    return np.array(out, dtype=float)


def ordered(cands, M, g, perm_rank):
    if len(cands) == 0:
        return cands
    key = np.lexsort((perm_rank[cands], ~M[cands, g]))   # members of g first, then fixed permutation
    return cands[key]


def attack(pop, chosen, eta, Bcert, budget, target, rng_seed, tau, cap):
    """Best decision (for the adversary) over the strategy set; returns (decision, strategy, flips)."""
    poison, M = pop["poison"], pop["memberships"]
    s1, s0 = chosen[1], chosen[0]
    if target == "GO":
        typeA = s0[poison[s0]]        # unflagged poisons -> clean
        typeB = s1[~poison[s1]]       # flagged clean -> poison
    else:
        typeA = s1[poison[s1]]        # flagged poisons -> clean
        typeB = s0[~poison[s0]]       # unflagged clean -> poison
    base = group_bounds(pop, chosen, poison, eta, Bcert)
    valid = ~np.isnan(base[:, 0])
    perm_rank = np.empty(len(poison), dtype=np.int64)
    perm_rank[np.random.default_rng(rng_seed).permutation(len(poison))] = np.arange(len(poison))
    if target == "GO":
        prios = [("bind", int(np.nanargmax(np.where(valid, base[:, 1], -1))))]
    else:
        sizes = np.array([sum(pop["gsize"][g]) if valid[g] else np.inf for g in range(M.shape[1])])
        prios = [("bind", int(np.nanargmax(np.where(valid, base[:, 0], -1)))), ("small", int(np.argmin(sizes)))]
        if prios[1][1] == prios[0][1]:
            prios = prios[:1]          # identical orderings; skip the duplicate
    best = None
    for pname, g in prios:
        oa, ob = ordered(typeA, M, g, perm_rank), ordered(typeB, M, g, perm_rank)
        for share in SHARES:
            nb = min(len(ob), int(round(share * budget))); na = min(len(oa), budget - nb); nb = min(len(ob), budget - na)
            obs = poison.copy()
            fa, fb = oa[:na], ob[:nb]
            obs[fa] = ~obs[fa]; obs[fb] = ~obs[fb]
            d = A.decide(pop, chosen, obs, eta, Bcert, tau, cap)
            cand = (RANK[target][d], d, f"{pname}:{share}", na + nb)
            if best is None or cand[0] < best[0]:
                best = cand
            if d == target:
                return best[1], best[2], best[3]
    return best[1], best[2], best[3]


def run_victim(args):
    block, victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); q = pop["q"]; eta = delta / (2 * q)
    true_pass = bool(pop["recall"] >= tau and pop["max_clean"] <= cap)
    target = "NO_GO" if true_pass else "GO"
    ns = cfg["ns"].format(v=victim); rows = []
    for k in c["budgets"]:
        n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, "0.5")
        for draw in range(c["protected_draws_per_seed_budget"]):
            chosen = {}
            for f, n in ((1, n1), (0, n0)):
                rng = np.random.default_rng(H.stable_seed(ns, k, draw, f))
                chosen[f] = rng.choice(pop["idx"][f], size=n, replace=False)
            base = A.decide(pop, chosen, pop["poison"], eta, 0, tau, cap)
            for rho in RHOS:
                B = int(math.ceil(rho * k - 1e-9))
                seed = H.stable_seed("groupsafe-redteam-v1", ns, k, draw, rho)
                rob0 = A.decide(pop, chosen, pop["poison"], eta, B, tau, cap)
                dn, sn, fn = attack(pop, chosen, eta, 0, B, target, seed, tau, cap)
                dr, sr, fr = attack(pop, chosen, eta, B, B, target, seed, tau, cap)
                rows.append(dict(block=block, victim_id=victim, k=k, draw=draw, rho=rho, B=B, true_pass=int(true_pass),
                                 recall=pop["recall"], max_clean=pop["max_clean"], Q=q, target=target,
                                 unattacked_naive=base, unattacked_robust=rob0,
                                 attacked_naive=dn, naive_strategy=sn, naive_flips=fn,
                                 attacked_robust=dr, robust_strategy=sr, robust_flips=fr))
    return pd.DataFrame(rows)


def run_part(blocks, output, procs):
    if output.exists():
        raise FileExistsError(output)
    jobs = []
    for block in blocks:
        cfg = BLOCKS[block]
        f = pd.read_csv(cfg["flags"], usecols=["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"])
        f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            jobs.append((block, v, fr.reset_index(drop=True), {k: str(x) for k, x in cfg.items()}))
    with Pool(procs) as p:
        out = pd.concat(p.imap_unordered(run_victim, jobs), ignore_index=True)
    output.mkdir(parents=True)
    out.sort_values(["block", "victim_id", "k", "draw", "rho"]).to_csv(output / "decisions.csv", index=False)
    print(json.dumps(dict(part=str(output), blocks=blocks, rows=len(out))))


def combine(parts, output):
    if output.exists():
        raise FileExistsError(output)
    out = pd.concat([pd.read_csv(p / "decisions.csv", dtype={"block": str}) for p in parts], ignore_index=True)
    out = out.sort_values(["block", "victim_id", "k", "draw", "rho"]).reset_index(drop=True)
    if sorted(out.block.unique()) != sorted(BLOCKS) or out.duplicated(["block", "victim_id", "k", "draw", "rho"]).any():
        raise RuntimeError("parts do not cover every block exactly once")
    val = {}
    for block, cfg in BLOCKS.items():
        fz = pd.read_csv(cfg["frozen"])[["victim_id", "k", "draw", "decision"]]
        mine = out[(out.block == block) & (out.rho == RHOS[0])][["victim_id", "k", "draw", "unattacked_naive"]]
        m = fz.merge(mine, on=["victim_id", "k", "draw"])
        val[block] = dict(frozen_rows=len(fz), matched=len(m), identical=bool(len(m) == len(fz) and (m.decision == m.unattacked_naive).all()))
    for col in ("unattacked_naive", "unattacked_robust", "attacked_naive", "attacked_robust"):
        out[f"{col}_false_go"] = ((out[col] == "GO") & (out.true_pass == 0)).astype(int)
        out[f"{col}_false_nogo"] = ((out[col] == "NO_GO") & (out.true_pass == 1)).astype(int)
    output.mkdir(parents=True)
    out.to_csv(output / "decisions.csv", index=False)
    g = out.groupby(["block", "true_pass", "rho", "k"])
    s = g.agg(n=("draw", "size"), B=("B", "max"),
              naive_false_go=("attacked_naive_false_go", "sum"), robust_false_go=("attacked_robust_false_go", "sum"),
              naive_false_nogo=("attacked_naive_false_nogo", "sum"), robust_false_nogo=("attacked_robust_false_nogo", "sum"),
              unattacked_naive_go=("unattacked_naive", lambda x: float((x == "GO").mean())),
              unattacked_robust_go=("unattacked_robust", lambda x: float((x == "GO").mean())),
              unattacked_naive_nogo=("unattacked_naive", lambda x: float((x == "NO_GO").mean())),
              unattacked_robust_nogo=("unattacked_robust", lambda x: float((x == "NO_GO").mean())),
              attacked_robust_go=("attacked_robust", lambda x: float((x == "GO").mean())),
              attacked_robust_nogo=("attacked_robust", lambda x: float((x == "NO_GO").mean()))).reset_index()
    s.to_csv(output / "summary.csv", index=False)
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    tot = {c: int(out[c].sum()) for c in out.columns if c.endswith("_false_go") or c.endswith("_false_nogo")}
    rep = dict(validation=val, rows=len(out), totals=tot, script_sha256=h(__file__),
               decisions_sha256=h(output / "decisions.csv"), summary_sha256=h(output / "summary.csv"))
    (output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--blocks", default=",".join(BLOCKS)); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--procs", type=int, default=2)
    c = sub.add_parser("combine"); c.add_argument("--parts", type=Path, nargs="+", required=True); c.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        blocks = a.blocks.split(",")
        if not set(blocks) <= set(BLOCKS):
            raise ValueError(blocks)
        run_part(blocks, a.output, a.procs)
    else:
        combine(a.parts, a.output)


if __name__ == "__main__":
    main()
