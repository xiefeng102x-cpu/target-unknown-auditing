import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "prepare_gtsrb_uniform_adjudication.py"
)
SPEC = importlib.util.spec_from_file_location("gtsrb_uniform_adapter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_gtsrb_adapter_reconstructs_full_population_without_new_flags():
    protocol = json.loads(MODULE.PROTOCOL.read_text(encoding="utf-8"))
    with np.load(MODULE.CACHE) as cache:
        labels = cache["train_labels"].copy()
    scores, truth, receipt = MODULE.build_seed(101, labels, protocol)
    assert len(truth) == 26240
    assert int(truth["is_poison"].sum()) == 2264
    assert len(scores) == 26240 * 3
    assert set(scores["detector"].unique()) == set(MODULE.METHODS)
    source = pd.read_csv(MODULE.SOURCE / "per_sample" / "seed_101_scores.csv")
    expected = source.groupby("method")["flag"].sum().to_dict()
    actual = scores.groupby("detector")["conventional_flag"].sum().to_dict()
    assert actual == expected
    assert receipt["prevalence"] == 2264 / 26240
