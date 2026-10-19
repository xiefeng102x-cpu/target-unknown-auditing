"""Pre-declared secondary analysis for the CIFAR-10 SCAn+LDA confirmation.

Nested escalation 2000 -> 8000 -> 16000 with a Bonferroni split delta/3 per look,
one-sided per-error-type tails with delta/2 on the two recall cells and delta/2 on the
protected-group cells, flagged-stratum share 0.25, stop at the first non-ABSTAIN.
Reported descriptively; it is not part of the frozen support gate.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "auditability_certificate_v1"
PROTOCOL = CERT / "PROTOCOL_CIFAR10_SCAN_LDA_CONFIRMATION_V1.json"
SCORES = ROOT / "results/cifar10_scan_lda_scores_confirmation_v1"
OUTPUT = CERT / "results_cifar10_scan_lda_secondary_escalation_v1"
HELPERS = CERT / "analysis/run_posthoc_power_study_v1.py"
HELPERS_SHA = "71eb676df806de0172dd99ba2e7413c100851a9d290e252669dcc8f7f36a1096"
SHARE = "0.25"; METHOD = "onesided_split"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=OUTPUT); a = ap.parse_args()
    if sha256(HELPERS) != HELPERS_SHA:
        raise RuntimeError("helper hash mismatch")
    ph = sha256(PROTOCOL)
    if (PROTOCOL.with_suffix(".json.sha256")).read_text().split()[0] != ph:
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
    rows = []
    for victim, frame in flags.groupby("victim_id", sort=True):
        pop = h.build_population(frame.reset_index(drop=True), proto["protected_groups"])
        true_pass = pop["recall"] >= tau and pop["max_clean"] <= cap
        eta = h.etas(METHOD, delta / len(budgets), pop["q"])
        for draw in range(draws):
            perm = {f: np.random.default_rng(h.stable_seed("cifar10-scan-lda-escalation-v1", victim, draw, f)).permutation(pop["idx"][f]) for f in (1, 0)}
            final, used = "ABSTAIN", budgets[-1]
            for k in budgets:
                n1, n0 = h.allocate(pop["N"][1], pop["N"][0], k, SHARE)
                d, *_ = h.decide(pop, {1: perm[1][:n1], 0: perm[0][:n0]}, eta, tau, cap)
                if d != "ABSTAIN":
                    final, used = d, k; break
            rows.append(dict(victim_id=victim, draw=draw, decision=final, k_used=used, true_pass=int(true_pass),
                             actual_recall=pop["recall"], actual_max_group_clean=pop["max_clean"]))
    out = pd.DataFrame(rows)
    out["false_go"] = ((out.decision == "GO") & (out.true_pass == 0)).astype(int)
    out["false_nogo"] = ((out.decision == "NO_GO") & (out.true_pass == 1)).astype(int)
    a.output.mkdir(parents=True); out.to_csv(a.output / "decisions.csv", index=False)
    rep = dict(status="SECONDARY_ESCALATION_COMPLETE", protocol_sha256=ph, decisions=len(out),
               go_rate_true_pass=float(out[out.true_pass == 1].decision.eq("GO").mean()) if (out.true_pass == 1).any() else None,
               nogo_rate_true_fail=float(out[out.true_pass == 0].decision.eq("NO_GO").mean()) if (out.true_pass == 0).any() else None,
               mean_k_used=float(out.k_used.mean()), false_go=int(out.false_go.sum()), false_nogo=int(out.false_nogo.sum()),
               decisions_sha256=sha256(a.output / "decisions.csv"))
    (a.output / "report.json").write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
