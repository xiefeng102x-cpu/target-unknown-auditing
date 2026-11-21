#!/usr/bin/env python3
"""Recompute study readouts from a separate aggregate attachment."""

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import beta


DESIGNS = {
    "previous_gold_ui": "Two-stratum SRAC-UI",
    "previous_two_phase": "Nested two-stage",
    "score_count": "Score strata / count",
    "score_joint": "Score strata / joint",
    "random_count": "Random strata / count",
    "flag_count": "Flag only / count",
    "score_fisher": "Score strata / Fisher",
    "random_fisher": "Random strata / Fisher",
    "flag_fisher": "Flag only / Fisher",
    "random_joint": "Random strata / joint",
    "flag_joint": "Flag only / joint",
}
PRIMARY_DESIGNS = tuple(DESIGNS)[:4]


def load(attachment_dir, name):
    return json.loads((attachment_dir / "data" / f"{name}.json").read_text(encoding="utf-8"))


def row(*cells):
    return "| " + " | ".join(map(str, cells)) + " |"


def table(headers):
    return [row(*headers), row(*(["---"] * len(headers)))]


def go(n, denominator):
    return f"{n:,}/{denominator:,} ({n / denominator:.2%})"


def replay_tables(data):
    arms, budgets = data["designs"], [str(k) for k in data["budgets"]]
    assert arms == list(DESIGNS) and budgets == ["500", "1000", "2000", "4000"]
    populations = data["per_population"]
    assert [p["seed"] for p in populations] == list(range(255, 265))
    for pop in populations:
        assert pop["passes_policy"] == (
            pop["true_recall"] >= data["recall_target"]
            and pop["max_group_clean_deletion"] <= data["group_clean_deletion_cap"]
        )
    passed = [p for p in populations if p["passes_policy"]]
    failed = [p for p in populations if not p["passes_policy"]]
    assert len(passed) == 9 and len(failed) == 1 and failed[0]["seed"] == 259
    draws = data["draws_per_population_per_budget"]
    n_passed, n_failed = len(passed) * draws, len(failed) * draws
    assert draws == n_failed == 200
    assert all(
        0 <= p["go"][k][arm] <= draws
        for p in populations for k in budgets for arm in arms
    )
    passing_totals = {k: {arm: sum(p["go"][k][arm] for p in passed) for arm in arms} for k in budgets}
    failing_totals = {k: {arm: sum(p["go"][k][arm] for p in failed) for arm in arms} for k in budgets}
    assert passing_totals == data["archived_pass_totals"]
    assert failing_totals == data["archived_fail_totals"]
    assert all(n == 0 for totals in failing_totals.values() for n in totals.values())
    full = data["archived_full_gold_reference"]
    lines = [
        "# Auxiliary-score audit replay",
        "",
        "Ten frozen GTSRB populations, with nine passing and one failing the policy.",
        "Each design and fixed budget has 200 draws per population.",
        "The measurements use archived truth as simulated gold labels.",
        "",
        "## Certification by budget",
        "",
        *table(("Design", *(f"{int(k):,} gold" for k in budgets))),
    ]
    for arm in arms:
        lines.append(row(DESIGNS[arm], *(go(passing_totals[k][arm], n_passed) for k in budgets)))
    lines += [
        "",
        f"Every design had 0/{n_failed} observed false GO at each fixed budget.",
        "The false-GO denominator represents one failing population.",
        f"The archived {full['gold_budget']:,}-gold reference had {go(full['go_passing'], full['passing_audits'])}.",
        "This reference used a larger budget than the matched comparisons.",
        "",
        "## Paired contrasts at 2,000 gold labels",
        "",
        *table(("Contrast", "Difference (percentage points)", "Descriptive 95% interval")),
    ]
    settings = data["primary_paired_bootstrap"]
    primary = str(settings["budget"])
    rng = np.random.default_rng(settings["seed"])
    indices = rng.integers(0, len(passed), size=(settings["resamples"], len(passed)))
    for contrast in settings["contrasts"]:
        left, right = contrast["left"], contrast["right"]
        delta = np.array([(p["go"][primary][left] - p["go"][primary][right]) / draws for p in passed])
        lo, hi = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        np.testing.assert_allclose([delta.mean(), lo, hi],
                                   [contrast["difference"], *contrast["ci95_percentile"]], atol=1e-12, rtol=0)
        lines.append(row(f"{DESIGNS[left]} minus {DESIGNS[right]}",
                         f"{100 * delta.mean():+.2f}", f"[{100 * lo:+.2f}, {100 * hi:+.2f}]"))
    lines += [
        "",
        "Each bootstrap resample draws whole passing populations with replacement.",
        "Within each population, differences are between methods evaluated on the same draw IDs.",
        "",
        "### First unmet requirement at 2,000 gold labels",
        "",
        *table(("Design", "GO", "Recall", "Group cap")),
    ]
    for arm, values in data["primary_first_stop_on_passing_populations"].items():
        assert passing_totals[primary][arm] == values["go"] and sum(values.values()) == n_passed
        lines.append(row(DESIGNS[arm], values["go"], values["recall"], values["group"]))
    lines += [
        "",
        "Recall was tested before the group caps. These counts record the first stop.",
        "",
        "## Population-level draws",
        "",
        "Each GO count uses 200 paired draws per population and budget.",
    ]
    for k in [primary, *(k for k in budgets if k != primary)]:
        lines += [
            "", f"### {int(k):,} gold labels", "",
            *table(("Seed", "True recall", "Max. clean deletion", "Policy met", *(DESIGNS[arm] for arm in PRIMARY_DESIGNS))),
        ]
        for pop in populations:
            lines.append(row(pop["seed"], f"{pop['true_recall']:.2%}", f"{pop['max_group_clean_deletion']:.2%}",
                             "Yes" if pop["passes_policy"] else "No",
                             *(f"{pop['go'][k][arm]}/{draws}" for arm in PRIMARY_DESIGNS)))
    lines += [
        "",
        "## Review budget and risk",
        "",
        "Each nested draw used 8,000 scores, reviewed 400 prioritized records and sampled another",
        "100, 600, 1,600 or 3,600 labels to reach its gold budget. Score/count and score/joint",
        "reuse the same gold records. Score/joint and random/joint hold testing and allocation quotas fixed.",
        "Only gold labels count toward the reported budget. Score computation and human time were unmeasured",
        "in this replay. The nominal false-GO level for a single fixed budget is 0.25% under correct gold",
        "labels and the sampling design.",
    ]
    return "\n".join(lines) + "\n"


def model_readout(data):
    historical = data["historical_seed_results"]
    conditions = ("plain_scan", "standardized_scan", "standardized_all")
    assert {(r["seed"], r["condition"]) for r in historical} == {
        (seed, condition) for seed in range(235, 245) for condition in conditions
    } and len(historical) == 30
    historical_means = {
        c: mean(r["attack_success_rate"] for r in historical if r["condition"] == c) for c in conditions
    }
    assert f"{historical_means['plain_scan']:.2%}" == "74.37%"
    assert f"{historical_means['standardized_scan']:.2%}" == "0.90%"
    print(f"Historical ASR (plain / standardized SCAn): {historical_means['plain_scan']:.2%} / {historical_means['standardized_scan']:.2%}")

    fresh = data["fresh_seed_results"]
    boots = data["fresh_bootstrap"]
    conditions = ("original_poisoned", *conditions)
    rng = np.random.default_rng(boots["seed"])
    indices = rng.integers(0, 5, size=(boots["draws"], 5))
    assert data["fresh_no_seed_filtering"] and not data["fresh_certificate_draws_run"]
    for endpoint in ("full", "heldout"):
        cells = {c: sorted((r for r in fresh if r["endpoint"] == endpoint and r["condition"] == c),
                           key=lambda r: r["seed"]) for c in conditions}
        for cell in cells.values():
            assert [r["seed"] for r in cell] == list(range(5181, 5186))
            for record in cell:
                assert len(record["clean_recall_by_class"]) == 43
                np.testing.assert_allclose(record["balanced_clean_accuracy"], mean(record["clean_recall_by_class"]))
        for contrast in data["fresh_contrasts"][endpoint]:
            left, right = cells[contrast["left"]], cells[contrast["right"]]
            for outcome, name in (("attack_success_rate", "asr"), ("balanced_clean_accuracy", "bca")):
                delta = np.array([a[outcome] - b[outcome] for a, b in zip(left, right)])
                np.testing.assert_allclose(delta.mean(), contrast[f"mean_{name}_difference"], atol=1e-12, rtol=0)
                np.testing.assert_allclose(np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]),
                                           contrast[f"{name}_difference_ci95_percentile"], atol=1e-12, rtol=0)
        plain, standardized = cells["plain_scan"], cells["standardized_scan"]
        asr = [mean(r["attack_success_rate"] for r in cell) for cell in (plain, standardized)]
        bca = [mean(r["balanced_clean_accuracy"] for r in cell) for cell in (plain, standardized)]
        class40 = [mean(r["clean_recall_by_class"][40] for r in cell) for cell in (plain, standardized)]
        expected = {"full": ["69.81%", "0.99%"], "heldout": ["69.56%", "0.91%"]}[endpoint]
        assert [f"{v:.2%}" for v in asr] == expected
        print(f"Fresh {endpoint} ASR: {asr[0]:.2%} / {asr[1]:.2%}; BCA: {bca[0]:.2%} / {bca[1]:.2%}; "
              f"class-40 change: {(class40[1] - class40[0]) * 100:+.2f} points")
    assert len(data["fresh_screen_policy"]) == 5 and sum(r["true_policy_pass"] for r in data["fresh_screen_policy"]) == 3
    assert not data["spatial_attack_success_rate_measured"]
    for endpoint in ("unseen_track", "official_test"):
        values = []
        for condition in ("raw_rgb", "channel_standardized", "spatial"):
            cells = [r for r in data["spatial_clean_metrics_by_fold"] if r["endpoint"] == endpoint and r["condition"] == condition]
            assert len(cells) == len({r["fold"] for r in cells}) == 8
            values.append(mean(r["class40"] for r in cells))
        expected = {"unseen_track": ["44.58%", "89.17%"], "official_test": ["59.31%", "78.89%"]}[endpoint]
        assert [f"{v:.2%}" for v in values[1:]] == expected
        print(f"Spatial {endpoint} clean class-40 recall: {values[1]:.2%} / {values[2]:.2%}")


def human_readout(data):
    expected_sizes = {"FC": 200, "FP": 100, "UP": 300, "UC": 1500}
    assert data["scored_records"] == sum(expected_sizes.values()) and not data["pair_stage_evaluated"]
    for cell, n in expected_sizes.items():
        values = data["cells"][cell]
        raw = values["raw_answer_counts"]
        assert values["n"] == n == sum(raw.values())
        if cell == "FC":
            errors = raw.get("POISON", 0)
        elif cell == "FP":
            errors = raw.get("CLEAN", 0) + raw.get("UNSURE", 0)
        elif cell == "UP":
            errors = raw.get("CLEAN", 0)
        else:
            errors = raw.get("POISON", 0) + raw.get("UNSURE", 0)
        assert errors == values["errors"]
        lo = beta.ppf(0.025, errors, n - errors + 1) if errors else 0
        hi = beta.ppf(0.975, errors + 1, n - errors) if errors < n else 1
        np.testing.assert_allclose([errors / n, lo, hi], [values["rate"], *values["ci95"]], atol=1e-12, rtol=0)
        print(f"Human {cell}: {errors}/{n} ({errors/n:.2%}), 95% CP interval [{lo:.2%}, {hi:.2%}]")
    for archive in data["simulation_projection"]:
        assert archive["passing_go"] == archive["false_go"] == 0
        assert archive["passing_audits"] == 1800 and archive["failing_audits"] == 200
        print(f"Archived human-error simulation at {archive['gold_budget']:,}: "
              f"{archive['passing_go']}/{archive['passing_audits']} passing GO, "
              f"{archive['false_go']}/{archive['failing_audits']} false GO")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attachment-dir", type=Path, required=True,
        help="Directory containing the unpacked reproducibility attachment",
    )
    parser.add_argument("--print-replay-tables", action="store_true", help="Print the reconstructed Markdown tables")
    args = parser.parse_args()
    input_names = ["data/auxiliary_scores.json"]
    if not args.print_replay_tables:
        input_names.extend(["replay_tables.md", "data/model_followup.json", "data/human_review.json"])
    missing = [name for name in input_names if not (args.attachment_dir / name).is_file()]
    if missing:
        parser.error(f"missing files in {args.attachment_dir}: {', '.join(missing)}")
    tables = replay_tables(load(args.attachment_dir, "auxiliary_scores"))
    if args.print_replay_tables:
        print(tables, end="")
        return
    assert tables == (args.attachment_dir / "replay_tables.md").read_text(encoding="utf-8")
    print("Auxiliary replay: full tables, per-population totals and paired bootstrap verified.")
    model_readout(load(args.attachment_dir, "model_followup"))
    human_readout(load(args.attachment_dir, "human_review"))
    print("Verification complete for the released aggregates.")


if __name__ == "__main__":
    main()
