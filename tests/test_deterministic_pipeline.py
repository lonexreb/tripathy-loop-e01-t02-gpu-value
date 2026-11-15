"""Deterministic (no LLM, no network) tests of the lab: Tasks 2-4 of the specification."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from lab.config import BENCHMARK_ORIGIN, DISCOVERY_ORIGINS, ROOT, LabConfig, check_partition
from lab.evaluator import HOLDOUT_KEYS, Evaluator, load_forecaster
from lab.rsi_controller import RSIController
from lab.sandbox import (TIME_INDEXED, PointInTimeSandbox, assert_point_in_time, audit_forecaster_source,
                         exa_cache_key, run_build_series)

FIX = Path(__file__).parent / "fixtures"
CACHE = ROOT / "outputs" / "cache"
SEED = FIX / "seed_drift.py"


@pytest.fixture(scope="session")
def sandbox():
    return PointInTimeSandbox(LabConfig(cache_dir=CACHE))


@pytest.fixture(scope="session")
def evaluator(sandbox):
    return Evaluator(sandbox.cfg, sandbox)


# ------------------------------------------------------------------------------ Task 2 ----
def test_partition_invariants():
    check_partition()
    assert len(DISCOVERY_ORIGINS) == 11


def test_slice_truncates_all_tables(sandbox):
    o = "2026-09-23"
    s = sandbox.get_pack_slice(o)
    for name in TIME_INDEXED:
        assert len(s[name]) > 0, name
        assert s[name]["available_date"].max() <= o, name
    assert s["series_history"]["date"].max() == o
    assert s["aggregator_medians"]["week_start"].max() <= "2026-09-17"   # only complete weeks
    assert len(s["targets"]) == 97
    assert "last_value_to_as_of" not in s["catalog"].columns             # no as-of statistics leak through the catalog


def test_slice_matches_official_build_series_cutoff(sandbox, tmp_path):
    o = "2026-09-23"
    out, cat = tmp_path / f"history_{o}.csv", tmp_path / f"catalog_{o}.csv"
    run_build_series(sandbox.cfg, out, cat, cutoff=o, members_from=sandbox.catalog_path)
    official = pd.read_csv(out, dtype={"series_id": str, "date": str})[["series_id", "date", "value"]]
    mine = sandbox.get_pack_slice(o)["series_history"][["series_id", "date", "value"]]
    m = official.merge(mine, on=["series_id", "date"], how="outer", suffixes=("_off", "_lab"), indicator=True)
    assert (m["_merge"] == "both").all()
    assert (m["value_off"] - m["value_lab"]).abs().max() < 1e-9


def test_point_in_time_assertion_catches_leak(sandbox):
    s = sandbox.get_pack_slice("2026-09-23")
    leak = s["daily_statistics"].iloc[[0]].copy()
    leak["date"] = leak["available_date"] = "2026-09-24"
    s["daily_statistics"] = pd.concat([s["daily_statistics"], leak])
    with pytest.raises(AssertionError, match="leakage"):
        assert_point_in_time(s, "2026-09-23")


def test_exa_blocked_when_disabled(sandbox):
    with pytest.raises(PermissionError):
        sandbox.search_exa_point_in_time("H100 rental price", "2026-09-23")


def test_exa_cache_is_point_in_time_and_offline(tmp_path):
    cfg = LabConfig(root=tmp_path, enable_exa=True, exa_cache_only=True, cache_dir=CACHE)
    cfg.outputs_dir.mkdir()
    key = exa_cache_key("h100 price", "2026-09-23", None, 8)
    rec = {"key": key, "query": "h100 price", "origin_date": "2026-09-23", "results": [
        {"title": "old", "url": "u1", "published_date": "2026-09-20T00:00:00Z", "highlights": ["a"]},
        {"title": "future", "url": "u2", "published_date": "2026-10-02T00:00:00Z", "highlights": ["b"]},
        {"title": "undated", "url": "u3", "published_date": None}]}
    cfg.exa_cache_path.write_text(json.dumps(rec) + "\n")
    sb = PointInTimeSandbox(cfg)
    assert [r["title"] for r in sb.search_exa_point_in_time("h100 price", "2026-09-23")] == ["old"]
    assert sb.search_exa_point_in_time("never cached", "2026-09-23") == []   # cache-only: no network call


def test_source_audit():
    assert audit_forecaster_source(SEED.read_text()) == []
    assert audit_forecaster_source((FIX / "candidate_h123.py").read_text()) == []
    bad = "import os\nimport pandas as pd\nclass Forecaster:\n    def predict_origin(self, *a):\n        return pd.read_csv(open('x'))\n"
    probs = " ".join(audit_forecaster_source(bad))
    assert "'os'" in probs and "'open'" in probs and ".read_csv" in probs


# ------------------------------------------------------------------------------ Task 3 ----
def test_evaluator_matches_official_scorer_frozen(evaluator):
    f = load_forecaster(SEED)
    fc, _ = evaluator.forecast(f, [BENCHMARK_ORIGIN])
    mine = evaluator.aggregates(evaluator.score(fc, [BENCHMARK_ORIGIN])[0])
    off = evaluator.official_score(fc, [BENCHMARK_ORIGIN])["overall"]
    assert mine["pooled_skill"] == pytest.approx(off["skill_pooled"], abs=1e-6)
    assert mine["pooled_skill"] == pytest.approx(0.200409, abs=1e-6)      # starter README: drift +0.20 at 2026-10-24
    assert mine["median_skill"] == pytest.approx(off["skill_median_of_series"], abs=1e-6)
    assert mine["coverage_80"] == pytest.approx(off["coverage"]["within_q10_q90"], abs=1e-6)


def test_seed_matches_starter_drift_baseline(evaluator, sandbox, tmp_path):
    """The seed forecaster reproduces starter/baselines.py --method drift row for row on Discovery."""
    o = list(DISCOVERY_ORIGINS)
    fc, _ = evaluator.forecast(load_forecaster(SEED), o)
    cmd = [sys.executable, "-I", "-B", str(sandbox.cfg.starter_dir / "baselines.py"), "--history", str(sandbox.series_path),
           "--origins", ",".join(o), "--method", "drift", "--horizons", "1,3,7", "--catalog", str(sandbox.catalog_path),
           "--tier", "headline", "--out", str(tmp_path / "drift.csv")]
    subprocess.run(cmd, check=True, capture_output=True)
    ref = pd.read_csv(tmp_path / "drift.csv", dtype={"origin_date": str})
    m = ref.merge(fc, on=["series_id", "origin_date", "horizon_days"], suffixes=("_ref", ""))
    assert len(m) == len(ref) == len(fc)
    for q in ("q10", "q50", "q90"):
        assert (m[q] - m[f"{q}_ref"]).abs().max() < 1e-8


def test_evaluator_matches_official_backtest(evaluator):
    f = load_forecaster(SEED)
    o = list(DISCOVERY_ORIGINS)
    fc, _ = evaluator.forecast(f, o)
    mine = evaluator.aggregates(evaluator.score(fc, o)[0])
    off = evaluator.official_score(fc, o)["overall"]
    assert mine["pooled_skill"] == pytest.approx(off["skill_pooled"], abs=1e-6)
    assert mine["coverage_80"] == pytest.approx(off["coverage"]["within_q10_q90"], abs=1e-6)


def test_holdout_information_firewall(evaluator):
    f = load_forecaster(SEED)
    hold = evaluator.evaluate(f, list(evaluator.cfg.holdout_origins), expose_diagnostics=False)
    assert set(hold) == set(HOLDOUT_KEYS)
    assert set(hold["skill_by_horizon"]) == {"1", "3", "7"}
    disc = evaluator.evaluate(f, list(DISCOVERY_ORIGINS), expose_diagnostics=True)
    assert {"worst_slices", "worst_series", "q10_breach_rate", "q90_breach_rate"} <= set(disc)


def test_missing_cells_filled_with_last_value(evaluator):
    class Empty:
        def predict_origin(self, *a, **k):
            return pd.DataFrame(columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])
    d = evaluator.evaluate(Empty(), [BENCHMARK_ORIGIN], expose_diagnostics=True)
    assert d["pooled_skill"] == pytest.approx(0.0)
    assert d["scoring"]["n_cells_filled_from_baseline"] == d["scoring"]["n_cells"]


# ------------------------------------------------------------------------------ Task 4 ----
@pytest.fixture(scope="module")
def ctrl(tmp_path_factory):
    root = tmp_path_factory.mktemp("labroot")
    shutil.copytree(ROOT / "lab", root / "lab", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "workspace", root / "workspace")
    (root / "gpu-price-pack-v0.1.1-handover").symlink_to((ROOT / "gpu-price-pack-v0.1.1-handover").resolve())
    shutil.copyfile(SEED, root / "workspace" / "forecaster.py")
    (root / "outputs").mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(git + ["add", "lab", "workspace"], cwd=root, check=True)
    subprocess.run(git + ["commit", "-qm", "init"], cwd=root, check=True)
    c = RSIController(LabConfig(root=root, cache_dir=CACHE), driver_name="test")
    c.bootstrap()
    return c


def _ledger_last(c):
    return c.ledger_entries()[-1]


def test_runtime_error_syntax_is_restored(ctrl):
    before = ctrl.cfg.forecaster_path.read_text()
    r = ctrl.propose_and_test_forecaster("broken syntax", "notes", "class Forecaster(:\n  pass")
    assert r["status"] == "RUNTIME_ERROR"
    assert ctrl.cfg.forecaster_path.read_text() == before
    assert _ledger_last(ctrl)["status"] == "RUNTIME_ERROR"


def test_runtime_error_exception_is_restored(ctrl):
    before = ctrl.cfg.forecaster_path.read_text()
    code = "class Forecaster:\n    def predict_origin(self, pack_slice, origin_date, horizons, external_signals=None):\n        raise ValueError('boom')\n"
    r = ctrl.propose_and_test_forecaster("raises", "notes", code)
    assert r["status"] == "RUNTIME_ERROR" and "boom" in r["error"]
    assert ctrl.cfg.forecaster_path.read_text() == before
    assert "boom" in ctrl.cfg.theory_path.read_text()          # the failure persists in the notebook


def test_firewall_rejects_file_io(ctrl):
    code = "import pandas as pd\nclass Forecaster:\n    def predict_origin(self, s, o, h, e=None):\n        return pd.read_csv('outputs/cache/series_daily.csv')\n"
    assert ctrl.propose_and_test_forecaster("peek at truth", "", code)["status"] == "REJECTED_FIREWALL"


def test_degraded_forecaster_screened_before_holdout(ctrl):
    before = ctrl.cfg.forecaster_path.read_text()
    r = ctrl.propose_and_test_forecaster("5x wider bands", "notes", (FIX / "degraded_5x.py").read_text())
    # q10 floored at 1% of q50 (100x below the last value) trips the sanity screen; it is also worse on Discovery.
    # Either way it never spends a Holdout evaluation.
    assert r["status"] == "REJECTED_SANITY"
    assert set(r["screen"]["failed"]) == {"sanity", "discovery_gain"} and "gates" not in r and "holdout" not in r
    assert r["discovery_diagnostics"]["coverage_80"] > 0.86 and r["discovery_diagnostics"]["sanity"]["n_extreme_rows"] > 0
    assert ctrl.cfg.forecaster_path.read_text() == before
    assert _ledger_last(ctrl)["status"] == "REJECTED_SANITY"


def test_improved_candidate_promoted_and_committed(ctrl):
    code = (FIX / "candidate_h123.py").read_text()
    r = ctrl.propose_and_test_forecaster("H0+H1+H3 multi-regime", "multi-regime notes", code)
    assert r["status"] == "PROMOTED", r.get("gates")
    assert r["gates"]["all_pass"]
    assert r["holdout"]["pooled_skill"] > 0.2
    assert ctrl.cfg.forecaster_path.read_text() == code == ctrl.backup_path.read_text()
    assert ctrl.champion["code_sha256"] == r["code_sha256"]
    head = subprocess.run(["git", "show", "HEAD:workspace/forecaster.py"], cwd=ctrl.root, capture_output=True, text=True).stdout
    assert head == code
    assert "PROMOTED" in subprocess.run(["git", "log", "-1", "--format=%s"], cwd=ctrl.root, capture_output=True, text=True).stdout


def test_lab_is_immutable(ctrl):
    p = ctrl.root / "lab" / "config.py"
    p.write_text(p.read_text() + "\n# tampered\n")
    with pytest.raises(PermissionError):
        ctrl.propose_and_test_forecaster("tamper", "", (FIX / "degraded_5x.py").read_text())
    assert "# tampered" not in p.read_text()


def test_slice_gpu_family_joins_aws_targets_to_side_tables(sandbox):
    s = sandbox.get_pack_slice("2026-09-23")
    aws = s["targets"][s["targets"]["series_id"].str.startswith("aws_spot.")]
    assert len(aws) == 80 and not aws["gpu_model"].isin(set(s["daily_statistics"]["gpu_model"])).any()
    for name in ("daily_statistics", "aggregator_medians", "index_prints", "hardware_fundamentals", "capability_units"):
        assert aws["gpu_family"].isin(set(s[name]["gpu_family"])).all(), name


def test_paired_backtest_on_dense_pseudo_origins(evaluator):
    from lab.evaluator import backtest_origins

    o = backtest_origins(evaluator.cfg, 1, "2026-09-20", "2026-09-23")
    r = evaluator.paired_backtest(FIX / "candidate_h123.py", SEED, o, ["does.not.exist"])
    assert r["n_origins"] == 4 and r["n_cells"] == 4 * 3 * 97
    assert r["pooled_skill_delta"] == pytest.approx(r["candidate"]["pooled_skill"] - r["reference"]["pooled_skill"])
    assert r["delta_se_origin_bootstrap"] >= 0 and 0 <= r["share_origins_improved"] <= 1
    assert set(r["delta_by_horizon"]) == {"1", "3", "7"} and r["series_cells"] == {"does.not.exist": []}
    same = evaluator.paired_backtest(SEED, SEED, o[:2])
    assert same["pooled_skill_delta"] == 0 and same["share_cells_improved"] == 0
    with pytest.raises(PermissionError):
        evaluator.paired_backtest(SEED, SEED, ["2026-09-30"])


def test_profile_pack_slice_reports_tables_joins_and_target_quality(sandbox):
    from lab.workbench import profile_pack_slice

    p = profile_pack_slice(sandbox.get_pack_slice("2026-09-23"), "2026-09-23")
    assert p["tables"]["series_history"]["date_range"][1] == "2026-09-23"
    assert p["target_joins_by_gpu_family"]["daily_statistics"]["targets_with_rows"] == 97
    assert p["missing_targets"] == [] and p["flagged_target_series"]["n_rows"] > 0
    json.dumps(p)
