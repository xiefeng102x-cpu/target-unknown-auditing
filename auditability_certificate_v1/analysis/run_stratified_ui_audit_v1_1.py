#!/usr/bin/env python3
"""Erratum recomputation of the post-hoc SRAC-UI analysis with engine v1.1.

Identical design to analysis/run_stratified_ui_audit_v1.py (same frozen flags, families,
budgets, shares, draw namespaces, escalation permutations, alpha = delta/(2A), alpha/3 per
look); the only change is the engine (analysis/srac_ui_engine_v1_1.py; see its docstring,
items E1-E4).  The frozen SRAC rows are not recomputed: they are copied from the v1 output,
whose validation reproduced every frozen SRAC decision.  The v1 results stay in place.

Output (new directory): decisions.csv (all designs), summary.csv, changes_vs_v1.csv (every row
whose decision differs from v1), report.json.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; CERT = HERE.parent
spec = importlib.util.spec_from_file_location("ui_v1", HERE / "run_stratified_ui_audit_v1.py")
V1 = importlib.util.module_from_spec(spec); sys.modules["ui_v1"] = V1; spec.loader.exec_module(V1)
sys.path.insert(0, str(HERE))
import srac_ui_engine_v1_1 as E  # noqa: E402
H = V1.H; BLOCKS = V1.BLOCKS; BUDGETS = V1.BUDGETS
V1_OUT = CERT / "results_stratified_ui_audit_v1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_victim(args):
    block, victim, frame, cfg = args
    proto = json.loads(Path(cfg["protocol"]).read_text()); c = proto["certificate"]
    delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    pop = H.build_population(frame, proto["protected_groups"])
    A = E.n_assertions(pop); alpha = delta / (2 * A)
    true_pass = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
    ns = cfg["ns"].format(v=victim); rows = []
    for k in c["budgets"]:
        for draw in range(c["protected_draws_per_seed_budget"]):
            for share in V1.SHARES:
                chosen = V1.draw_strata(pop, ns, k, draw, share)
                rows.append(dict(design="srac_ui", share=share, k=k, draw=draw, decision=E.decide_ui(pop, chosen, alpha, tau, cap), labels=k))
    for draw in range(c["protected_draws_per_seed_budget"]):
        perm = {f: np.random.default_rng(H.stable_seed("srac-ui-escalation-v1", ns, draw, f)).permutation(pop["idx"][f]) for f in (1, 0)}
        final, used = "ABSTAIN", BUDGETS[-1]
        for k in BUDGETS:
            n1, n0 = H.allocate(pop["N"][1], pop["N"][0], k, V1.ESC_SHARE)
            d = E.decide_ui(pop, {1: perm[1][:n1], 0: perm[0][:n0]}, alpha / len(BUDGETS), tau, cap)
            if d != "ABSTAIN":
                final, used = d, k; break
        rows.append(dict(design="srac_ui_esc", share=V1.ESC_SHARE, k=0, draw=draw, decision=final, labels=used))
    out = pd.DataFrame(rows)
    out.insert(0, "victim_id", victim); out.insert(0, "block", block)
    out["true_pass"] = true_pass; out["recall"] = pop["recall"]; out["max_clean"] = pop["max_clean"]; out["A"] = A; out["Q"] = pop["q"]
    return out


def jobs_for(blocks):
    jobs = []
    for block, cfg in BLOCKS.items():
        if blocks and block not in blocks:
            continue
        f = pd.read_csv(cfg["flags"], usecols=["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"])
        f = f[f.victim_eligible.eq(1)]
        for v, fr in f.groupby("victim_id", sort=True):
            jobs.append((block, v, fr.reset_index(drop=True), {k: str(x) for k, x in cfg.items()}))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, required=True); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--blocks", default="", help="comma-separated block ids to compute (default all)")
    ap.add_argument("--combine", type=Path, default=None, help="write the combined output to this new directory")
    a = ap.parse_args()
    blocks = [b for b in a.blocks.split(",") if b]
    a.parts.mkdir(parents=True, exist_ok=True)
    part = lambda b, v: a.parts / f"{b}__{v}.csv"
    if a.combine is None:
        todo = [j for j in jobs_for(blocks) if not part(j[0], j[1]).exists()]
        print(f"{len(todo)} victims to compute", flush=True)
        with Pool(a.procs) as p:
            for df in p.imap_unordered(run_victim, todo):
                df.to_csv(part(df.block.iloc[0], df.victim_id.iloc[0]), index=False)
                print(f"part done: {df.block.iloc[0]} {df.victim_id.iloc[0]}", flush=True)
        return
    if a.combine.exists():
        raise FileExistsError(a.combine)
    v1 = pd.read_csv(V1_OUT / "decisions.csv", dtype={"block": str, "share": str})
    keys = v1[v1.design != "srac"][["block", "victim_id"]].drop_duplicates()
    parts = []
    for b, v in keys.itertuples(index=False):
        p = part(b, v)
        if not p.exists():
            raise FileNotFoundError(p)
        parts.append(pd.read_csv(p, dtype={"block": str, "share": str}))
    new = pd.concat(parts + [v1[v1.design == "srac"].drop(columns=["false_go", "false_nogo"])], ignore_index=True)
    new = new.sort_values(["block", "victim_id", "design", "share", "k", "draw"]).reset_index(drop=True)
    new["false_go"] = ((new.decision == "GO") & (new.true_pass == 0)).astype(int)
    new["false_nogo"] = ((new.decision == "NO_GO") & (new.true_pass == 1)).astype(int)
    key = ["block", "victim_id", "design", "share", "k", "draw"]
    m = new.merge(v1[key + ["decision", "labels"]], on=key, suffixes=("", "_v1"), how="outer", indicator=True)
    if not (m._merge == "both").all():
        raise RuntimeError("row sets differ from v1")
    ch = m[(m.decision != m.decision_v1) | (m.labels != m.labels_v1)].drop(columns="_merge")
    a.combine.mkdir(parents=True)
    new.to_csv(a.combine / "decisions.csv", index=False)
    ch.to_csv(a.combine / "changes_vs_v1.csv", index=False)
    s = new.groupby(["design", "share", "true_pass", "k"]).agg(n=("decision", "size"), go=("decision", lambda x: float((x == "GO").mean())),
            nogo=("decision", lambda x: float((x == "NO_GO").mean())), false_go=("false_go", "sum"), false_nogo=("false_nogo", "sum"),
            mean_labels=("labels", "mean")).reset_index()
    s.to_csv(a.combine / "summary.csv", index=False)
    trans = ch.groupby(["design", "share", "k", "decision_v1", "decision"]).size().rename("rows").reset_index()
    rep = dict(engine=E.ENGINE_VERSION, rows=len(new), changed_rows=len(ch), transitions=trans.to_dict("records"),
               totals=new.groupby("design")[["false_go", "false_nogo"]].sum().to_dict("index"),
               engine_sha256=sha(HERE / "srac_ui_engine_v1_1.py"), runner_sha256=sha(__file__),
               v1_decisions_sha256=sha(V1_OUT / "decisions.csv"),
               decisions_sha256=sha(a.combine / "decisions.csv"), summary_sha256=sha(a.combine / "summary.csv"),
               changes_sha256=sha(a.combine / "changes_vs_v1.csv"))
    (a.combine / "report.json").write_text(json.dumps(rep, indent=2) + "\n"); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
