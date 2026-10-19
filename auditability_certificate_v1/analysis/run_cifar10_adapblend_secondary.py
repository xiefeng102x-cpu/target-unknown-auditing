"""Pre-declared secondary analyses for the Adap-Blend SCAn+LDA confirmation (block 13).

Reported descriptively; none is part of the frozen support gate.
  S1  Nested escalation 2000 -> 8000 -> 16000, delta/3 per look, one-sided per-error-type
      tails (delta/2 recall cells, delta/2 group cells), flagged share 0.25 (as in block 12).
  S2  Used-cell error budget (Q counts only cells of sets with a flagged record) for the
      frozen semantic family F0 and for F0 plus every singleton class (F1), fixed budgets,
      frozen draw namespace, two-sided exact intervals, 50/50 allocation.
  S3  Recall including cover rows and the cover flag rate, from the score manifest.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
PROTOCOL = CERT / "PROTOCOL_CIFAR10_ADAPBLEND_SCAN_LDA_CONFIRMATION_V1.json"
SCORES = ROOT / "results/cifar10_adapblend_scan_lda_scores_confirmation_v1"
OUTPUT = CERT / "results_cifar10_adapblend_secondary_v1"
HELPERS = CERT / "analysis/run_posthoc_power_study_v1.py"
HELPERS_SHA = "71eb676df806de0172dd99ba2e7413c100851a9d290e252669dcc8f7f36a1096"
NAMESPACE = "cifar10-adapblend-scan-lda-confirm-v1|{v}"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def q_used(pop) -> int:
    return 2 + sum((1 + (1 if us else 0)) for fs, us in pop["gsize"] if fs > 0)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=OUTPUT); a = ap.parse_args()
    if sha256(HELPERS) != HELPERS_SHA:
        raise RuntimeError("helper hash mismatch")
    ph = sha256(PROTOCOL)
    if PROTOCOL.with_suffix(".json.sha256").read_text().split()[0] != ph:
        raise RuntimeError("protocol sidecar mismatch")
    proto = json.loads(PROTOCOL.read_text())
    if proto["secondary_analysis"]["script_sha256"] != sha256(Path(__file__)):
        raise RuntimeError("secondary script hash mismatch")
    man = json.loads((SCORES / "manifest.json").read_text())
    if man.get("protocol_sha256") != ph or man.get("flags_sha256") != sha256(SCORES / "flags.csv"):
        raise RuntimeError("score handoff mismatch")
    if a.output.exists():
        raise FileExistsError(a.output)
    spec = importlib.util.spec_from_file_location("posthoc_helpers", HELPERS)
    h = importlib.util.module_from_spec(spec); sys.modules[spec.name] = h; spec.loader.exec_module(h)
    c = proto["certificate"]; delta, tau, cap = c["delta_per_decision"], c["recall_target"], c["max_group_clean_flag_rate_cap"]
    budgets, draws = c["budgets"], c["protected_draws_per_seed_budget"]
    flags = pd.read_csv(SCORES / "flags.csv"); flags = flags[flags.victim_eligible.eq(1)]
    esc, fixed = [], []
    for victim, frame in flags.groupby("victim_id", sort=True):
        frame = frame.reset_index(drop=True)[["victim_id", "seed", "sample_id", "assigned_group", "flag", "is_poison", "victim_eligible"]]
        f0 = list(proto["protected_groups"])
        f1 = f0 + [dict(name=f"class_{k}", class_ids=[k]) for k in sorted(set(int(x) for x in frame.assigned_group))]
        pop = h.build_population(frame, f0)
        tp0 = int(pop["recall"] >= tau and pop["max_clean"] <= cap)
        eta = h.etas("onesided_split", delta / len(budgets), pop["q"])
        for draw in range(draws):
            perm = {f: np.random.default_rng(h.stable_seed("cifar10-adapblend-escalation-v1", victim, draw, f)).permutation(pop["idx"][f]) for f in (1, 0)}
            final, used = "ABSTAIN", budgets[-1]
            for k in budgets:
                n1, n0 = h.allocate(pop["N"][1], pop["N"][0], k, "0.25")
                d, *_ = h.decide(pop, {1: perm[1][:n1], 0: perm[0][:n0]}, eta, tau, cap)
                if d != "ABSTAIN":
                    final, used = d, k; break
            esc.append(dict(victim_id=victim, draw=draw, decision=final, k_used=used, true_pass=tp0))
        for fam, groups in (("F0", f0), ("F1", f1)):
            pp = h.build_population(frame, groups)
            tp = int(pp["recall"] >= tau and pp["max_clean"] <= cap)
            q = q_used(pp); e = delta / (2 * q)
            for k in budgets:
                n1, n0 = h.allocate(pp["N"][1], pp["N"][0], k, "0.5")
                for draw in range(draws):
                    chosen = {}
                    for f, n in ((1, n1), (0, n0)):
                        rng = np.random.default_rng(h.stable_seed(NAMESPACE.format(v=victim), k, draw, f))
                        chosen[f] = rng.choice(pp["idx"][f], size=n, replace=False) if n else np.empty(0, int)
                    d, *_ = h.decide(pp, chosen, dict(R=e, G=e), tau, cap)
                    fixed.append(dict(victim_id=victim, family=fam, Q_used=q, k=k, draw=draw, decision=d, true_pass=tp))
    esc, fixed = pd.DataFrame(esc), pd.DataFrame(fixed)
    for df in (esc, fixed):
        df["false_go"] = ((df.decision == "GO") & (df.true_pass == 0)).astype(int)
        df["false_nogo"] = ((df.decision == "NO_GO") & (df.true_pass == 1)).astype(int)
    a.output.mkdir(parents=True)
    esc.to_csv(a.output / "escalation_decisions.csv", index=False); fixed.to_csv(a.output / "used_cell_decisions.csv", index=False)
    cover = [dict(seed=r["seed"], recall=r["recall"], recall_including_cover=r["recall_including_cover"],
                  cover_flag_rate=r["cover_flag_rate"], scan_only_recall=r["scan_only_recall"]) for r in man["seeds"]]
    pd.DataFrame(cover).to_csv(a.output / "cover_descriptive.csv", index=False)
    summ = fixed.groupby(["family", "k", "true_pass"]).agg(n=("decision", "size"), go=("decision", lambda x: float((x == "GO").mean())),
                                                           nogo=("decision", lambda x: float((x == "NO_GO").mean()))).reset_index()
    summ.to_csv(a.output / "used_cell_summary.csv", index=False)
    rep = dict(status="SECONDARY_COMPLETE", protocol_sha256=ph,
               escalation=dict(decisions=len(esc), go_rate_true_pass=float(esc[esc.true_pass == 1].decision.eq("GO").mean()) if (esc.true_pass == 1).any() else None,
                               nogo_rate_true_fail=float(esc[esc.true_pass == 0].decision.eq("NO_GO").mean()) if (esc.true_pass == 0).any() else None,
                               mean_k_used=float(esc.k_used.mean()), false_go=int(esc.false_go.sum()), false_nogo=int(esc.false_nogo.sum())),
               used_cell=dict(decisions=len(fixed), false_go=int(fixed.false_go.sum()), false_nogo=int(fixed.false_nogo.sum())),
               sha256={p.name: sha256(p) for p in sorted(a.output.glob("*.csv"))})
    (a.output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
