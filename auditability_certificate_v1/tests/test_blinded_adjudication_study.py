from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = load("blind_builder", ROOT / "analysis/build_blinded_adjudication_study.py")
analyzer = load("blind_analyzer", ROOT / "analysis/analyze_blinded_adjudication_study.py")


def test_packet_hides_gold_and_analysis_separates_repeatability_from_correctness():
    truth = pd.DataFrame({
        "victim_id": ["v"] * 1000,
        "sample_id": range(1000),
        "role": ["train"] * 1000,
        "is_poison": [0] * 800 + [1] * 200,
    })
    public, key, receipt = builder.build(
        truth,
        uniform_rows=200,
        challenge_rows_per_class=40,
        duplicate_fraction=0.25,
        seed=17,
    )
    assert "gold_is_poison" not in public.columns
    assert "unit_token" not in public.columns
    assert "victim_id" not in public.columns
    assert "sample_id" not in public.columns
    assert "record_locator" not in public.columns
    assert public["record_handle"].is_unique
    assert receipt["unique_study_units"] == 280
    assert receipt["duplicate_units"] == 70
    completed = public.copy()
    gold = key.set_index("presentation_id")["gold_is_poison"]
    completed["adjudicated_is_poison"] = completed["presentation_id"].map(gold)
    completed["adjudicator_id"] = "rater-a"
    completed["adjudicated_at"] = "2026-10-13T00:00:00Z"
    report = analyzer.analyze(completed, key, delta=0.005, audit_budgets=[120, 500])
    assert report["uniform_errors"] == 0
    assert report["duplicate_disagreements"] == 0
    assert report["delta_error_rate_bound"] + report["delta_future_draw_bound"] == report["delta_total"]
    assert report["declared_B_by_audit_budget_under_iid_error_model"]["500"] >= 0


def test_one_sided_error_bound_increases_with_observed_errors():
    assert analyzer.cp_upper(2, 100, 0.01) > analyzer.cp_upper(0, 100, 0.01)
