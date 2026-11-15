"""Model-class exploration, verified novelty, variants and the scipy/sklearn unlock (no pack, no LLM, no network)."""
from __future__ import annotations

import json

from lab.memory import (MODEL_CLASSES, PROTECTED_BLOCK, SEED_MODEL_CLASS, exploration_directive,
                        model_class_exploration_required, normalize_model_class, registry_row, verified_novelty)
from lab.rsi_controller import AGENT_CONTRACT, PROPOSAL_SCHEMA, WORKBENCH_PROTOCOL, AgentToolbox
from lab.sandbox import audit_forecaster_source
from tests.test_memory_loop import cand, ctrl  # noqa: F401  (pytest fixture)

BIG = "\n    ".join(f"w{i} = {i}" for i in range(60))


def test_model_class_normalisation_and_verified_novelty():
    assert normalize_model_class("Gradient Boosting") == "gradient_boosting"
    assert normalize_model_class("quantile") == "quantile_regression"
    assert normalize_model_class("") == "unspecified" and normalize_model_class("lstm").startswith("other:")
    small, big = {"added": 9, "removed": 3}, {"added": 120, "removed": 80}
    assert verified_novelty("bold", small, SEED_MODEL_CLASS, SEED_MODEL_CLASS) == "incremental"
    assert verified_novelty("bold", big, SEED_MODEL_CLASS, SEED_MODEL_CLASS) == "bold"
    assert verified_novelty("bold", small, "gradient_boosting", SEED_MODEL_CLASS) == "bold"
    assert verified_novelty("incremental", big, "state_space", SEED_MODEL_CLASS) == "incremental"


def test_bold_label_on_a_knob_tweak_is_recorded_as_incremental(ctrl):
    r = ctrl.propose_and_test_forecaster("h1 tail 0.95", "", cand(0.251, 0.24, 0.78, "t = 0.95"), "interval_calibration",
                                         "bold")
    assert r["novelty"] == "incremental" and r["novelty_claimed"] == "bold" and r["model_class"] == SEED_MODEL_CLASS
    row = ctrl.experiment_registry()[-1]
    assert row["model_class"] == SEED_MODEL_CLASS and row["novelty_claimed"] == "bold"


def test_same_class_streak_forces_model_class_exploration(ctrl):
    for i in range(2):
        ctrl.propose_and_test_forecaster(f"tweak {i}", "", cand(0.251, 0.24, 0.78, f"a = {i}"), "point_forecast",
                                         "incremental")
    assert not model_class_exploration_required(ctrl.experiment_registry())
    ctrl.propose_and_test_forecaster("tweak 2", "", cand(0.251, 0.24, 0.78, "a = 2"), "ensemble", "incremental")
    reg = ctrl.experiment_registry()
    assert model_class_exploration_required(reg)
    d = ctrl.inspect_current_state()["exploration_directive"]
    assert "MODEL-CLASS EXPLORATION" in d and "gradient_boosting" in d and "closest near-miss" not in d
    tb = AgentToolbox(ctrl)
    assert tb.model_class_exploration and tb.max_dry_runs == 2 * ctrl.cfg.max_discovery_probes
    ctrl.propose_and_test_forecaster("gbm core", "", cand(0.251, 0.24, 0.78, BIG), "point_forecast", "bold",
                                     model_class="gradient_boosting")
    reg = ctrl.experiment_registry()
    assert reg[-1]["model_class"] == "gradient_boosting" and reg[-1]["novelty"] == "bold"
    assert not model_class_exploration_required(reg)                    # a different class resets the streak
    assert ctrl.inspect_current_state()["model_class_stats"]["gradient_boosting"]["tried"] == 1


def test_legacy_rows_without_a_tag_count_as_the_seed_class():
    rows = [registry_row({"iteration": i, "status": "REVERTED", "hypothesis": "x"}) for i in range(3)]
    assert all(r["model_class"] == SEED_MODEL_CLASS and r["model_class_inferred"] for r in rows)
    assert "MODEL-CLASS EXPLORATION" in exploration_directive(rows)


def test_variants_let_one_iteration_compare_several_forecasters(ctrl):
    tb = AgentToolbox(ctrl)
    tb.write_candidate(cand(0.26, 0.25, 0.78, "kind = 'qreg'"))
    assert json.loads(tb.save_variant("qreg"))["ok"]
    tb.write_candidate(cand(0.27, 0.25, 0.78, "kind = 'gbm'"))
    tb.save_variant("gbm")
    assert set(json.loads(tb.list_variants())) == {"qreg", "gbm"}
    tb.load_variant("qreg")
    assert "qreg" in tb.wb.code
    assert not json.loads(tb.load_variant("nope"))["ok"]
    r = json.loads(tb.submit_candidate_for_holdout("qreg core", "", "point_forecast", "bold", -1, "quantile_regression"))
    assert r["model_class"] == "quantile_regression"


def test_firewall_allows_any_computational_library_but_not_io():
    head = "class Forecaster:\n    def predict_origin(self, s, o):\n        return None\n"
    ok = ["from scipy.optimize import linprog", "import scipy.stats as st", "from scipy import stats", "import scipy",
          "import sklearn", "from sklearn.ensemble import HistGradientBoostingRegressor", "import torch",
          "import torch.nn as nn\nnet = nn.Linear(3, 1)", "import lightgbm as lgb\nm = lgb.LGBMRegressor()",
          "import statistics, heapq"]
    bad = ["from scipy import io", "import scipy.io", "from sklearn import datasets", "import sklearn.datasets",
           "import joblib", "import os", "import pickle", "from torch import hub", "import torch.hub",
           "import scipy\nx = scipy.io.loadmat('f')", "import torch\nm = torch.load('m.pt')",
           "import requests", "import importlib", "x = getattr(object, 'a')"]
    for line in ok:
        assert audit_forecaster_source(line + "\n" + head) == [], line
    for line in bad:
        assert audit_forecaster_source(line + "\n" + head), line


def test_prompts_carry_model_class_and_rewrite_guidance():
    assert "model_class" in AGENT_CONTRACT and "from scratch" in AGENT_CONTRACT
    assert PROPOSAL_SCHEMA["properties"]["model_class"]["enum"] == list(MODEL_CLASSES)
    assert "save_variant" in WORKBENCH_PROTOCOL and "keep each edit small" not in WORKBENCH_PROTOCOL
    assert "Prefer building on the closest near-miss" not in PROTECTED_BLOCK
    assert "two independent implementations" in PROTECTED_BLOCK
    assert {"save_variant", "load_variant", "list_variants"} <= set(AgentToolbox.TOOL_NAMES)
