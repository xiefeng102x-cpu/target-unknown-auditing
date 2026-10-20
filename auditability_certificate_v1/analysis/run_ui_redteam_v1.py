#!/usr/bin/env python3
"""Post-hoc outcome-aware label red team against SRAC-UI (plan UI_REDTEAM_SCALING_PLAN_V1.md, part A).

Same blocks (8, 11, 12, 13), frozen flags and families, and the same SRAC-UI samples
(flagged share 0.1, namespace "srac-ui-v1") as the post-hoc SRAC-UI analysis.  The adversary
and its strategy set are those of the group-safe red team (run_groupsafe_redteam_v1.py,
function attack): after the draw it corrupts up to B = ceil(rho k) sampled labels, rho in
{0.1%, 0.5%}; on failing screens it seeks a false GO (unflagged poisons -> clean, flagged
clean -> poison), on passing screens a false NO-GO (mirror flips); the budget is split in
shares 0, 1/4, 1/2, 3/4, 1 between the two flip types, records are prioritized by the binding
group (and, for NO-GO, also by the smallest group), and the adversary wins if any strategy
reaches its target.  Certificates: naive SRAC-UI (B_cert = 0) and declared-B SRAC-UI
(srac_ui_robust_v1, R = B), both at alpha = delta/(2A).  Validation: the unattacked naive
decision must reproduce every stored SRAC-UI v1.1 decision (share 0.1) of the four blocks.

usage: run_ui_redteam_v1.py run --output PART_DIR --blocks 8,11 [--procs 2] [--draws N]
       run_ui_redteam_v1.py combine --parts PART_DIR [PART_DIR ...] --output DIR
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
spec = importlib.util.spec_from_file_location("gs_redteam", HERE / "run_groupsafe_redteam_v1.py")
RT = importlib.util.module_from_spec(spec); sys.modules["gs_redteam"] = RT; spec.loader.exec_module(RT)
import srac_ui_engine_v1_1 as E  # noqa: E402
import srac_ui_robust_v1 as UR  # noqa: E402
H = V1.H
BLOCKS = ("8", "11", "12", "13")
RHOS = (0.001, 0.005)
SHARES = RT.SHARES; RANK = RT.RANK
STORED = CERT / "results_stratified_ui_audit_v1_1/decisions.csv"


def attack(pop, chosen, alpha, eta, R, budget, target, seed, tau, cap):
    """RT.attack with the SRAC-UI decision (declared R) in place of the interval certificate."""
    poison, M = pop["poison"], pop["memberships"]
    s1, s0 = chosen[1], chosen[0]
    typeA, typeB = (s0[poison[s0]], s1[~poison[s1]]) if target == "GO" else (s1[poison[s1]], s0[~poison[s0]])
    base = RT.group_bounds(pop, chosen, poison, eta, 0)
    valid = ~np.isnan(base[:, 0])
    perm_rank = np.empty(len(poison), dtype=np.int64)
    perm_rank[np.random.default_rng(seed).permutation(len(poison))] = np.arange(len(poison))
    if target == "GO":
        prios = [("bind", int(np.nanargmax(np.where(valid, base[:, 1], -1))))]
    else:
        sizes = np.array([sum(pop["gsize"][g]) if valid[g] else np.inf for g in range(M.shape[1])])
        prios = [("bind", int(np.nanargmax(np.where(valid, base[:, 0], -1)))), ("small", int(np.argmin(sizes)))]
        if prios[1][1] == prios[0][1]:
            prios = prios[:1]
    best = None
    for pname, g in prios:
        oa, ob = RT.ordered(typeA, M, g, perm_rank), RT.ordered(typeB, M, g, perm_rank)
        for share in SHARES:
            nb = min(len(ob), int(round(share * budget))); na = min(len(oa), budget - nb); nb = min(len(ob), budget - na)
            obs = poison.copy(); fa, fb = oa[:na], ob[:nb]
            obs[fa] = ~obs[fa]; obs[fb] = ~obs[fb]
            d = UR.decide(pop, chosen, obs, alpha, tau, cap, R)
            cand = (RANK[target][d], d, f"{pname}:{share}", na + nb)
            if best is None or cand[0] < best[0]:
                best = cand
            if d == target:
                break
        if best[1] == target:
            break
    return best[1], best[2], best[3]


def run_victim(args):
    block, victim, frame, cfg, draws = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"]); eta = delta / (2 * pop["q"])
    alpha = delta / (2 * E.n_assertions(pop))
    true_pass = bool(pop["recall"] >= tau and pop["max_clean"] <= cap); target = "NO_GO" if true_pass else "GO"
    ns = cfg["ns"].format(v=victim); rows = []
    for k in c["budgets"]:
        for draw in range(draws):
            ch = V1.draw_strata(pop, ns, k, draw, "0.1")
            base = UR.decide(pop, ch, pop["poison"], alpha, tau, cap, 0)
            for rho in RHOS:
                B = int(math.ceil(rho * k - 1e-9)); seed = H.stable_seed("ui-redteam-v1", ns, k, draw, rho)
                rob0 = UR.decide(pop, ch, pop["poison"], alpha, tau, cap, B)
                dn, sn, fn = attack(pop, ch, alpha, eta, 0, B, target, seed, tau, cap)
                dr, sr, fr = attack(pop, ch, alpha, eta, B, B, target, seed, tau, cap)
                rows.append(dict(block=block, victim_id=victim, k=k, draw=draw, rho=rho, B=B, true_pass=int(true_pass),
                                 recall=pop["recall"], max_clean=pop["max_clean"], A=E.n_assertions(pop), target=target,
                                 unattacked_naive=base, unattacked_robust=rob0, attacked_naive=dn, naive_strategy=sn,
                                 naive_flips=fn, attacked_robust=dr, robust_strategy=sr, robust_flips=fr))
    return pd.DataFrame(rows)


def jobs(draws, blocks=BLOCKS):
    out = []
    for block in blocks:
        cfg = V1.BLOCKS[block]
        f = pd.read_csv(cfg["flags"], usecols=["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"])
        for v, fr in f[f.victim_eligible.eq(1)].groupby("victim_id", sort=True):
            out.append((block, v, fr.reset_index(drop=True), {k: str(x) for k, x in cfg.items()}, draws))
    return out


def summarize(d, draws):
    st = pd.read_csv(STORED, dtype={"block": str, "share": str})
    st = st[(st.design == "srac_ui") & (st.share == "0.1") & st.block.isin(BLOCKS) & (st.draw < draws)]
    m = d[d.rho == RHOS[0]].merge(st[["block", "victim_id", "k", "draw", "decision"]], on=["block", "victim_id", "k", "draw"], how="left")
    rec = dict(compared=len(m), mismatches=int((m.unattacked_naive != m.decision).sum()), missing=int(m.decision.isna().sum()))
    summ = []
    for rho, g in d.groupby("rho"):
        f, p = g[g.true_pass == 0], g[g.true_pass == 1]
        by = lambda x, col, val: {int(k): float((x[x.k == k][col] == val).mean()) for k in sorted(x.k.unique())}
        summ.append(dict(rho=rho, failing_audits=len(f), passing_audits=len(p),
                         failing_screens=int(f.victim_id.nunique()), passing_screens=int(p.victim_id.nunique()),
                         naive_false_go=float((f.attacked_naive == "GO").mean()), robust_false_go=float((f.attacked_robust == "GO").mean()),
                         naive_false_nogo=float((p.attacked_naive == "NO_GO").mean()), robust_false_nogo=float((p.attacked_robust == "NO_GO").mean()),
                         naive_false_go_count=int((f.attacked_naive == "GO").sum()), robust_false_go_count=int((f.attacked_robust == "GO").sum()),
                         robust_false_nogo_count=int((p.attacked_robust == "NO_GO").sum()),
                         robust_go_attacked_by_k=by(p, "attacked_robust", "GO"), robust_go_unattacked_by_k=by(p, "unattacked_robust", "GO"),
                         naive_go_unattacked_by_k=by(p, "unattacked_naive", "GO"),
                         robust_nogo_attacked_failing_by_k=by(f, "attacked_robust", "NO_GO")))
    return rec, summ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "combine"]); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=2); ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--blocks", default=",".join(BLOCKS)); ap.add_argument("--parts", type=Path, nargs="*")
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.cmd == "run":
        sel = UR.selftest()
        if not sel["passed"]:
            raise RuntimeError(f"robust selftest failed: {sel}")
        blocks = tuple(a.blocks.split(","))
        with Pool(a.procs) as p:
            d = pd.concat(p.imap_unordered(run_victim, jobs(a.draws, blocks)), ignore_index=True)
        a.output.mkdir(parents=True)
        d.to_csv(a.output / "decisions.csv", index=False)
        (a.output / "part.json").write_text(json.dumps(dict(blocks=blocks, draws=a.draws, robust_selftest=sel)) + "\n")
        return
    parts = [json.loads((p / "part.json").read_text()) for p in a.parts]
    draws = {x["draws"] for x in parts}
    if len(draws) != 1 or sorted(b for x in parts for b in x["blocks"]) != sorted(BLOCKS):
        raise RuntimeError("parts do not cover the blocks with one draw count")
    d = pd.concat([pd.read_csv(p / "decisions.csv", dtype={"block": str}) for p in a.parts], ignore_index=True)
    d = d.sort_values(["block", "victim_id", "k", "draw", "rho"]).reset_index(drop=True)
    rec, summ = summarize(d, draws.pop())
    a.output.mkdir(parents=True)
    d.to_csv(a.output / "decisions.csv", index=False)
    rep = dict(status="POST_HOC_DEVELOPMENTAL", robust_selftest=parts[0]["robust_selftest"], reproduction=rec,
               draws_per_budget=parts[0]["draws"], summary=summ,
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               robust_module_sha256=hashlib.sha256((HERE / "srac_ui_robust_v1.py").read_bytes()).hexdigest(),
               decisions_sha256=hashlib.sha256((a.output / "decisions.csv").read_bytes()).hexdigest())
    (a.output / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
