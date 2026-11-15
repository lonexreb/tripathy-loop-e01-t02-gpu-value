"""Evaluation / promotion harness tests that need no data pack, LLM or network (synthetic data + recorded ledger)."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lab.config import ROOT, LabConfig
from lab.evaluator import Evaluator, finite_or_none, sanity_report
from lab.rsi_controller import RSIController, compare_discovery, evaluate_gates, screen_candidate, sha256_text

LEDGER = ROOT / "outputs" / "experiment_ledger.jsonl"
CHAMP = ROOT / "outputs" / "champions" / "PACK_ONLY" / "champion_state.json"
CFG = LabConfig()


def _ledger_rows():
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]


def _live(it):
    return next(r for r in _ledger_rows() if r.get("iteration") == it and r.get("status") == "REVERTED")


# --------------------------------------------------------------------------- Gate 3 / gates ----
def _hold(pooled=0.30, median=0.26, cov=0.75, h=(0.40, 0.32, 0.27)):
    return {"pooled_skill": pooled, "median_skill": median, "coverage_80": cov,
            "skill_by_horizon": dict(zip(("1", "3", "7"), h))}


def test_gate3_champion_at_floor_is_not_a_trap():
    champ = json.loads(CHAMP.read_text())["holdout"]          # coverage 0.7408 = 1049/1416 cells, 1 cell above 0.74
    it3 = _live(3)
    g = evaluate_gates(it3["holdout"], champ, CFG)            # 0.7380 = 1045/1416 cells
    assert g["gate3_calibration"]["in_band"] is False
    assert g["gate3_calibration"]["pass"] is True             # within champion's distance to 0.80 + 0.005
    assert g["gate1_pooled_gain"]["pass"] is False            # +0.0004 is still not a promotion
    assert g["all_pass"] is False


def test_gate3_relative_rule_is_bounded():
    champ = _hold(cov=0.7408)
    assert evaluate_gates(_hold(pooled=0.31, cov=0.7300), champ, CFG)["gate3_calibration"]["pass"] is False
    assert evaluate_gates(_hold(pooled=0.31, cov=0.7360), champ, CFG)["gate3_calibration"]["pass"] is True
    low = _hold(cov=0.705)
    assert evaluate_gates(_hold(pooled=0.31, cov=0.699), low, CFG)["gate3_calibration"]["pass"] is False  # hard floor
    assert evaluate_gates(_hold(pooled=0.31, cov=0.80), low, CFG)["all_pass"] is True
    # an over-wide champion does not license an under-covering candidate, and vice versa
    assert evaluate_gates(_hold(pooled=0.31, cov=0.72), _hold(cov=0.88), CFG)["gate3_calibration"]["pass"] is True
    assert evaluate_gates(_hold(pooled=0.31, cov=0.91), _hold(cov=0.88), CFG)["gate3_calibration"]["pass"] is False


@pytest.mark.parametrize("bad", [None, float("nan"), float("-inf"), "x"])
def test_gates_fail_closed_on_missing_or_nonfinite(bad):
    champ = _hold()
    for k in ("pooled_skill", "median_skill", "coverage_80"):
        h = _hold(pooled=0.4)
        h[k] = bad
        g = evaluate_gates(h, champ, CFG)
        assert g["all_pass"] is False
    h = _hold(pooled=0.4, h=(0.4, bad, 0.3))
    assert evaluate_gates(h, champ, CFG)["gate4_horizon_floor"]["pass"] is False


def test_gates_recorded_promotion_still_passes():
    rows = _ledger_rows()
    seed = next(r for r in rows if r["status"] == "SEED")
    promo = next(r for r in rows if r["status"] == "PROMOTED")
    assert evaluate_gates(promo["holdout"], seed["holdout"], CFG)["all_pass"] is True


# ------------------------------------------------------------------- pre-Holdout screen ----
def _with_sanity(disc, ok=True):
    return {**disc, "sanity": {"ok": ok, "n_rows": 3084, "n_nonfinite_rows": 0, "n_extreme_rows": 0 if ok else 1,
                                "n_crossing_rows": 0, "n_nonpositive_rows": 0},
            "scoring": {"n_cells": 3084, "n_cells_filled_from_baseline": 0}}


def test_two_stage_screen_replays_the_live_iterations():
    """None of the three live LLM candidates would have consumed a Holdout evaluation."""
    champ = json.loads(CHAMP.read_text())["discovery"]
    for it in (2, 3, 4):
        s = screen_candidate(_with_sanity(_live(it)["discovery"]), champ, CFG)
        assert s["pass"] is False and s["status"] == "REJECTED_DISCOVERY", (it, s)
    promo = next(r for r in _ledger_rows() if r["status"] == "PROMOTED")
    seed = next(r for r in _ledger_rows() if r["status"] == "SEED")
    assert screen_candidate(_with_sanity(promo["discovery"]), seed["discovery"], CFG)["pass"] is True


def test_screen_sanity_and_invalid_share():
    champ = {"pooled_skill": 0.1}
    good = _with_sanity({"pooled_skill": 0.2})
    assert screen_candidate(good, champ, CFG)["pass"]
    s = screen_candidate(_with_sanity({"pooled_skill": 0.2}, ok=False), champ, CFG)
    assert s["status"] == "REJECTED_SANITY" and s["failed"] == ["sanity"]
    filled = {**good, "scoring": {"n_cells": 1000, "n_cells_filled_from_baseline": 60}}
    assert screen_candidate(filled, champ, CFG)["failed"] == ["invalid_share"]
    assert screen_candidate({**good, "pooled_skill": None}, champ, CFG)["status"] == "REJECTED_DISCOVERY"


def test_compare_discovery_is_paired():
    champ = json.loads(CHAMP.read_text())["discovery"]
    cand = json.loads(json.dumps(champ))
    o = sorted(cand["skill_by_origin"])
    cand["skill_by_origin"][o[0]] += 0.01
    cand["skill_by_origin"][o[1]] -= 0.02
    cand["slices"][0]["excess_loss_sum"] -= 1.0
    v = compare_discovery(cand, champ)
    assert v["n_origins_improved"] == 1 and v["n_origins_worse"] == 1
    assert v["skill_by_origin_delta"][o[1]] == pytest.approx(-0.02)
    assert v["slice_deltas_best_first"][0]["excess_loss_sum_delta"] == pytest.approx(-1.0)


# -------------------------------------------------------------------------- sanity report ----
def test_sanity_report_flags_explosions_but_not_normal_bands():
    fc = pd.DataFrame({
        "series_id": ["a", "a", "b", "b", "c", "c"], "origin_date": ["2026-09-01"] * 6,
        "horizon_days": [1, 7, 1, 7, 1, 7],
        "q10": [1.8, 1.7, 0.9, 0.95, 4.0, 4.0],
        "q50": [2.0, 2.0, 1.0, 1.0, 5.0, 5.0],
        "q90": [2.2, 1e84, 1.1, np.inf, 6.0, 4.5],
    })
    lv = pd.Series({("a", "2026-09-01"): 2.0, ("b", "2026-09-01"): 1.0, ("c", "2026-09-01"): 5.0})
    r = sanity_report(fc, lv, CFG.gates.sanity_max_abs_log_ratio)
    assert r["ok"] is False
    assert r["n_nonfinite_rows"] == 1 and r["n_extreme_rows"] == 1 and r["n_crossing_rows"] == 1
    assert json.dumps(r, allow_nan=False)
    ok = sanity_report(fc.iloc[[0, 2, 4]], lv, CFG.gates.sanity_max_abs_log_ratio)
    assert ok["ok"] is True and ok["max_abs_log_ratio_vs_last"] == pytest.approx(np.log(1.25), abs=1e-4)


def test_finite_or_none_makes_results_json_safe():
    res = finite_or_none({"pooled_skill": float("-inf"), "skill_by_horizon": {"7": np.float64("nan")}, "n": 3,
                          "l": [1.0, float("inf")]})
    assert json.dumps(res, allow_nan=False) == '{"pooled_skill": null, "skill_by_horizon": {"7": null}, "n": 3, "l": [1.0, null]}'


# ------------------------------------------------------ evaluator on a synthetic mini-pack ----
class _Store:
    def __init__(self, df):
        self.d = {(r.series_id, int(r.day)): float(r.value) for r in df.itertuples()}
        self.days = {}
        for (s, day) in self.d:
            self.days.setdefault(s, []).append(day)

    def last_value(self, sid, day):
        ds = [x for x in self.days.get(sid, []) if x <= day]
        return None if not ds else (self.d[(sid, max(ds))], max(ds))

    def value_at(self, sid, day):
        return self.d.get((sid, day))


def _pin(y, q, t):
    return max(t * (y - q), (t - 1) * (y - q))


class _Common:
    """Minimal stand-in for starter/_common.py with the semantics documented in lab/evaluator.py."""
    SeriesStore = _Store

    @staticmethod
    def parse_date(s):
        return dt.date.fromisoformat(str(s)[:10])

    @staticmethod
    def day_of(d):
        return d.toordinal()

    @staticmethod
    def cell_pinball(y, q10, q50, q90):
        return (_pin(y, q10, 0.1) + _pin(y, q50, 0.5) + _pin(y, q90, 0.9)) / 3

    @classmethod
    def validate_forecast(cls, df, horizons):
        df = df.copy()
        q = df[["q10", "q50", "q90"]].astype(float)
        ok = np.isfinite(q).all(axis=1) & (q["q10"] > 0) & (q["q10"] <= q["q50"]) & (q["q50"] <= q["q90"])
        df["origin_day"] = df["origin_date"].map(lambda s: cls.day_of(cls.parse_date(s)))
        return df[ok], [f"invalid row {i}" for i in df.index[~ok]]


ORIGINS_D, ORIGINS_H = ("2026-07-15", "2026-07-22"), ("2026-09-30",)


@pytest.fixture()
def mini_eval():
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-06-01", "2026-10-10", freq="D")
    rows = []
    for i, sid in enumerate(["s0", "s1", "s2", "s3"]):
        v = (1.0 + i) * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
        rows += [(sid, d.strftime("%Y-%m-%d"), float(x)) for d, x in zip(days, v)]
    series = pd.DataFrame(rows, columns=["series_id", "date", "value"])
    catalog = pd.DataFrame({"series_id": ["s0", "s1", "s2", "s3"], "instrument_class": ["hyperscaler_spot"] * 3 + ["neocloud_rate_card"],
                            "pricing_tier": ["spot"] * 3 + ["on_demand"], "source_id": "x", "gpu_model": "H100"})
    cfg = LabConfig(discovery_origins=ORIGINS_D, holdout_origins=ORIGINS_H)
    sb = SimpleNamespace(series_full=series, catalog=catalog, scored_ids=["s0", "s1", "s2", "s3"])
    ev = Evaluator.__new__(Evaluator)
    ev.cfg, ev.sandbox, ev._c, ev._b = cfg, sb, _Common, None
    return ev, series


class _Band:
    def __init__(self, series, width=0.05, explode=False):
        self.s, self.w, self.explode, self.seen = series, width, explode, []

    def predict_origin(self, sl, o, horizons, sig=None):
        self.seen.append(o)
        last = self.s[self.s["date"] <= o].groupby("series_id")["value"].last()
        out = []
        for sid, lv in last.items():
            for h in horizons:
                q90 = 1e84 if self.explode and h == 7 and sid == "s0" else lv * (1 + self.w * np.sqrt(h))
                out.append((sid, o, h, lv * (1 - self.w * np.sqrt(h)), lv, q90))
        return pd.DataFrame(out, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])


def _forecaster(ev, f, origins):
    ev.sandbox.get_pack_slice = lambda o, ids=None: {}
    fc, _ = ev.forecast(f, list(origins))
    return fc


def test_diagnostics_components_and_no_groupby_apply(mini_eval):
    ev, series = mini_eval
    f = _Band(series)
    fc = _forecaster(ev, f, ORIGINS_D)
    cells, info = ev.score(fc, list(ORIGINS_D))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = ev.diagnostics(cells)
    assert info["n_cells"] == 4 * 2 * 3
    assert sum(d["loss_decomposition"].values()) == pytest.approx(1.0, abs=1e-3)
    for h in ("1", "3", "7"):
        bh = d["by_horizon"][h]
        assert {"q10_breach_rate", "q90_breach_rate", "mean_width_rel", "mean_bias_rel", "loss_components"} <= set(bh)
        assert bh["mean_width_rel"] == pytest.approx(2 * 0.05 * np.sqrt(int(h)), rel=1e-3)
    assert sum(d["by_horizon"][h]["share_of_total_loss"] for h in ("1", "3", "7")) == pytest.approx(1.0, abs=1e-3)
    assert d["largest_loss_series"][0]["share_of_total_loss"] >= d["largest_loss_series"][-1]["share_of_total_loss"]
    assert {"q10_breach_rate", "q90_breach_rate", "mean_width_rel"} <= set(d["slices"][0])
    # pooled skill unchanged by the diagnostics: 1 - mean loss / mean baseline loss over all cells
    assert d["pooled_skill"] == pytest.approx(1 - cells["loss"].mean() / cells["baseline_loss"].mean())


def test_discovery_phase_never_touches_holdout(mini_eval, tmp_path, monkeypatch):
    ev, series = mini_eval
    f = _Band(series)
    monkeypatch.setattr("lab.evaluator.load_forecaster", lambda p: f)
    ev.sandbox.get_pack_slice = lambda o, ids=None: {}
    out = ev.full_run(tmp_path / "x.py", phase="discovery")
    assert set(f.seen) == set(ORIGINS_D) and "holdout" not in out
    assert out["discovery"]["sanity"]["ok"] is True
    f.seen.clear()
    out = ev.full_run(tmp_path / "x.py", phase="holdout")
    assert set(f.seen) <= set(ORIGINS_H) | {"2026-10-24"} and "discovery" not in out and out["holdout_sanity_ok"]


def test_exploding_quantile_is_caught_and_result_is_json_safe(mini_eval):
    ev, series = mini_eval
    fc = _forecaster(ev, _Band(series, explode=True), ORIGINS_D)
    d = ev.evaluate(_Band(series, explode=True), list(ORIGINS_D), True)
    assert d["sanity"]["ok"] is False and d["sanity"]["n_extreme_rows"] == 2
    assert d["pooled_skill"] < -1e70                      # the official scoring definition is unchanged
    inf = fc.assign(q90=np.where(fc["horizon_days"] == 7, np.inf, fc["q90"]))
    assert ev.sanity(inf)["n_nonfinite_rows"] == 8
    json.dumps(finite_or_none(d), allow_nan=False, default=str)


# --------------------------------------------------------- controller two-stage state machine ----
@pytest.fixture()
def ctrl(tmp_path):
    root = tmp_path / "lab_root"
    shutil.copytree(ROOT / "lab", root / "lab", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "workspace").mkdir()
    champ_code = "# champion\n"
    (root / "workspace" / "forecaster.py").write_text(champ_code)
    (root / "workspace" / "theory_of_market.md").write_text("")
    cd = root / "outputs" / "champions" / "PACK_ONLY"
    cd.mkdir(parents=True)
    (cd / "forecaster.py").write_text(champ_code)
    state = json.loads(CHAMP.read_text())
    state["code_sha256"] = sha256_text(champ_code)
    (cd / "champion_state.json").write_text(json.dumps(state))
    c = RSIController(LabConfig(root=root), driver_name="test", use_git=False)
    c.calls = []
    c.disc_result = {"pooled_skill": state["discovery"]["pooled_skill"] + 0.01}
    c.hold_result = {**state["holdout"], "pooled_skill": state["holdout"]["pooled_skill"] + 0.01}

    def fake_eval(path, phase="full"):
        c.calls.append((phase, Path(path).read_text()))
        if phase == "discovery":
            return {"ok": True, "discovery": {**state["discovery"], **_with_sanity({}), **c.disc_result}}
        return {"ok": True, "holdout": c.hold_result, "holdout_sanity_ok": True, "benchmark_2026_10_24": state["holdout"]}
    c.run_evaluation = fake_eval
    return c


CODE = "import pandas as pd\nclass Forecaster:\n    def predict_origin(self, s, o, h, e=None):\n        return pd.DataFrame()\n"


def test_discovery_regression_never_reaches_holdout(ctrl):
    ctrl.disc_result = {"pooled_skill": 0.44279393247897514}       # iteration 4's Discovery (-0.00006)
    r = ctrl.propose_and_test_forecaster("shrink beta", "notes", CODE)
    assert r["status"] == "REJECTED_DISCOVERY"
    assert [p for p, _ in ctrl.calls] == ["discovery"]
    assert ctrl.cfg.forecaster_path.read_text() == "# champion\n"
    assert "vs_champion_discovery" in r and "pre-Holdout screen failed" in ctrl.cfg.theory_path.read_text()
    again = ctrl.propose_and_test_forecaster("same code", "", CODE)
    assert again["status"] == "REJECTED_DUPLICATE" and len(ctrl.calls) == 1
    assert ctrl.next_iteration() == 3


def test_insane_outputs_rejected_before_holdout(ctrl):
    ctrl.disc_result = {"pooled_skill": 0.9, "sanity": {"ok": False, "n_rows": 3084, "n_extreme_rows": 3}}
    r = ctrl.propose_and_test_forecaster("explode", "", CODE)
    assert r["status"] == "REJECTED_SANITY" and [p for p, _ in ctrl.calls] == ["discovery"]


def test_two_stage_promotion_with_relative_gate3(ctrl):
    ctrl.hold_result = {**ctrl.hold_result, "coverage_80": 0.738}
    r = ctrl.propose_and_test_forecaster("better", "notes", CODE)
    assert [p for p, _ in ctrl.calls] == ["discovery", "holdout"]
    assert r["status"] == "PROMOTED", r["gates"]
    assert ctrl.backup_path.read_text() == CODE == ctrl.cfg.forecaster_path.read_text()
    assert ctrl.champion["holdout"]["coverage_80"] == 0.738
    assert ctrl.propose_and_test_forecaster("again", "", CODE)["status"] == "REJECTED_DUPLICATE"


def test_holdout_insanity_blocks_promotion(ctrl):
    ev = ctrl.run_evaluation

    def bad_holdout(path, phase="full"):
        r = ev(path, phase)
        return {**r, "holdout_sanity_ok": False} if phase == "holdout" else r
    ctrl.run_evaluation = bad_holdout
    r = ctrl.propose_and_test_forecaster("better on paper", "", CODE)
    assert r["status"] == "REVERTED" and r["gates"]["holdout_sanity"]["pass"] is False


def test_discovery_dry_run_is_budgeted_and_side_effect_free(ctrl):
    from lab.rsi_controller import AgentToolbox
    tb = AgentToolbox(ctrl)
    assert tb.max_dry_runs == ctrl.cfg.max_discovery_probes
    for i in range(tb.max_dry_runs):
        tb.wb.write(CODE + f"# {i}\n")
        r = json.loads(tb.dry_run_discovery())
        assert r["status"] == "DRY_RUN_OK" and r["would_reach_holdout"] is True
        assert r["vs_champion_discovery"]["pooled_skill_delta"] == pytest.approx(0.01)
    assert json.loads(tb.dry_run_discovery())["status"] == "DRY_RUN_BUDGET_EXHAUSTED"
    assert {p for p, _ in ctrl.calls} == {"discovery"}
    assert ctrl.cfg.forecaster_path.read_text() == "# champion\n"
    assert ctrl.next_iteration() == 1                                  # dry runs are not experiments
    assert ctrl.propose_and_test_forecaster("submit", "", CODE)["status"] == "PROMOTED"
    assert ctrl.dry_run_discovery("import os\nclass Forecaster: pass\n")["status"] == "DRY_RUN_REJECTED"
