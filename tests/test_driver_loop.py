"""Deterministic tests of the RSI driver loop (no LLM, no network, no data pack)."""
import json
import shutil
from pathlib import Path

import pytest

from lab.config import DISCOVERY_ORIGINS, HOLDOUT_ORIGINS, LabConfig
from lab.rsi_controller import (AgentToolbox, DevinAPIDriver, RSIController, StrandsAgentDriver, devin_poll_action,
                                devin_status, restore_champion_file, run_iterations)
from lab.workbench import CandidateWorkbench, audit_analysis_source, clip_json, run_analysis, to_jsonable

ROOT = Path(__file__).resolve().parents[1]
CHAMP = ("import numpy as np\nimport pandas as pd\n\n\nclass Forecaster:\n    WIDTH = 0.05\n\n"
         "    def predict_origin(self, pack_slice, origin_date, horizons, external_signals=None):\n"
         "        return pd.DataFrame()\n")
CAND = CHAMP.replace("WIDTH = 0.05", "WIDTH = 0.07")


def _disc(skill):
    return {"pooled_skill": skill, "median_skill": skill - 0.04, "coverage_80": 0.75, "q10_breach_rate": 0.1,
            "q90_breach_rate": 0.15, "skill_by_horizon": {"1": skill + 0.1, "3": skill, "7": skill - 0.03},
            "sanity": {"ok": True, "n_rows": 100}, "scoring": {"n_cells": 100}}


HOLD = {"pooled_skill": 0.30, "median_skill": 0.25, "coverage_80": 0.75, "skill_by_horizon": {"1": 0.4, "3": 0.3, "7": 0.27}}


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "lab", tmp_path / "lab", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "forecaster.py").write_text(CHAMP)
    (tmp_path / "workspace" / "scientist_prompt.md").write_text("playbook " * 40)
    c = RSIController(LabConfig(root=tmp_path), driver_name="test", use_git=False)
    c.champ_dir.mkdir(parents=True)
    c.backup_path.write_text(CHAMP)
    c.state_path.write_text(json.dumps({"stage": c.cfg.stage, "iteration": 0, "code_sha256": "x",
                                        "discovery": _disc(0.25), "holdout": HOLD, "benchmark_2026_10_24": {}}))
    calls = []

    def fake_child(module_args, timeout_s):
        calls.append(module_args)
        if module_args[0] == "lab.evaluator":
            code = Path(module_args[2]).read_text()
            if "raise" in code:
                return {"ok": False, "error": "ValueError: boom"}
            return {"ok": True, "discovery": _disc(0.26), "holdout": {**HOLD, "pooled_skill": 0.29},
                    "benchmark_2026_10_24": {}}
        return {"ok": True, "results": {o: {"n": 1} for o in module_args[module_args.index("--origins") + 1].split(",")}}

    monkeypatch.setattr(c, "_run_lab_child", fake_child)
    c.child_calls = calls
    return c


def _statuses(c):
    return [r["status"] for r in c.ledger_entries()]


# ------------------------------------------------------------------ iteration accounting / main loop ----
class _Boom:
    name = "boom"

    def __init__(self, ctrl, fail=99):
        self.ctrl, self.fail, self.n = ctrl, fail, 0

    def run_iteration(self, ctrl, i):
        self.n += 1
        ctrl.cfg.forecaster_path.write_text("garbage")  # simulate a half-written candidate
        if self.n <= self.fail:
            raise RuntimeError("MaxTokensReachedException: simulated")
        return ctrl.propose_and_test_forecaster("h", "", CAND)


def test_driver_errors_restore_champion_and_stop_after_consecutive_errors(ctrl):
    d = _Boom(ctrl)
    n_err = run_iterations(ctrl, d, 10, max_consecutive_errors=3)
    assert n_err == 3 and d.n == 3
    assert ctrl.cfg.forecaster_path.read_text() == CHAMP
    errs = [r for r in ctrl.ledger_entries() if r["status"] == "DRIVER_ERROR"]
    assert [r["attempt"] for r in errs] == [1, 2, 3]
    assert all(r["iteration"] == 1 and r["counts_as_experiment"] is False for r in errs)
    assert ctrl.next_iteration() == 1


def test_driver_error_then_experiment_keeps_iteration_numbers(ctrl):
    run_iterations(ctrl, _Boom(ctrl, fail=1), 2)
    exp = [r for r in ctrl.ledger_entries() if r["status"] in ("PROMOTED", "REVERTED")]
    assert len(exp) == 1 and exp[0]["iteration"] == 1 and ctrl.next_iteration() == 2


def test_restore_champion_file_handles_deleted_forecaster(ctrl):
    ctrl.cfg.forecaster_path.unlink()
    assert restore_champion_file(ctrl) is True
    assert ctrl.cfg.forecaster_path.read_text() == CHAMP
    assert restore_champion_file(ctrl) is False


# --------------------------------------------------------------------------- Discovery-only tools ----
def test_dry_run_is_discovery_only_and_not_an_experiment(ctrl):
    r = ctrl.dry_run_discovery(CAND)
    assert r["status"] == "DRY_RUN_OK"
    assert r["vs_champion_discovery"]["pooled_skill_delta"] == pytest.approx(0.01)
    assert r["would_reach_holdout"] is True
    assert "holdout" not in r and "gates" not in r
    assert ctrl.child_calls[-1][0] == "lab.evaluator" and ctrl.child_calls[-1][-2:] == ["--phase", "discovery"]
    assert ctrl.cfg.forecaster_path.read_text() == CHAMP and ctrl.next_iteration() == 1
    assert ctrl.dry_run_discovery("import os\n" + CAND)["status"] == "DRY_RUN_REJECTED"


def test_analysis_rejects_holdout_origins_and_firewall_violations(ctrl):
    ok = "def analyze(pack_slice, origin_date):\n    return {'n': 1}\n"
    assert ctrl.run_discovery_analysis(ok, [HOLDOUT_ORIGINS[0]])["status"] == "ANALYSIS_REJECTED"
    assert ctrl.run_discovery_analysis(ok, [])["status"] == "ANALYSIS_REJECTED"
    assert ctrl.run_discovery_analysis("import os\n" + ok, [DISCOVERY_ORIGINS[0]])["status"] == "ANALYSIS_REJECTED"
    r = ctrl.run_discovery_analysis(ok, [DISCOVERY_ORIGINS[0]])
    assert r["status"] == "ANALYSIS_OK" and DISCOVERY_ORIGINS[0] in r["results"]
    assert ctrl.next_iteration() == 1


def test_audit_analysis_source():
    assert audit_analysis_source("import pandas as pd\ndef analyze(pack_slice, origin_date):\n    return 1\n") == []
    assert any("analyze" in p for p in audit_analysis_source("x = 1\n"))
    assert any("open" in p for p in audit_analysis_source("def analyze(s, o):\n    return open('f').read()\n"))


def test_run_analysis_uses_point_in_time_slices_and_refuses_holdout(tmp_path, monkeypatch):
    import pandas as pd

    import lab.sandbox

    class FakeSandbox:
        def __init__(self, cfg):
            pass

        def get_pack_slice(self, origin):
            return {"series_history": pd.DataFrame({"date": [origin], "value": [1.5]})}

    monkeypatch.setattr(lab.sandbox, "PointInTimeSandbox", FakeSandbox)
    code = tmp_path / "a.py"
    code.write_text("def analyze(pack_slice, origin_date):\n    return pack_slice['series_history']\n")
    cfg = LabConfig(root=tmp_path)
    out = run_analysis(code, [DISCOVERY_ORIGINS[0]], cfg)
    assert out["results"][DISCOVERY_ORIGINS[0]]["records"] == [{"date": DISCOVERY_ORIGINS[0], "value": 1.5}]
    with pytest.raises(PermissionError):
        run_analysis(code, [HOLDOUT_ORIGINS[0]], cfg)


def test_to_jsonable_and_clip_json():
    import numpy as np

    assert to_jsonable({"a": np.float64("nan"), "b": np.int64(3)}) == {"a": None, "b": 3}
    s = clip_json({"x": "y" * 5000}, 1000)
    assert len(s) <= 1000 and json.loads(s)["truncated"] is True


# ------------------------------------------------------------------------------- candidate editing ----
def test_workbench_exact_unique_edits():
    wb = CandidateWorkbench(CHAMP)
    assert wb.replace("WIDTH = 0.05", "WIDTH = 0.07")["ok"] and wb.code == CAND
    assert not wb.replace("does not exist", "x")["ok"]
    assert "found 2 matches" in CandidateWorkbench("a\na\n").replace("a", "b")["error"]
    assert "+    WIDTH = 0.07" in wb.diff() and "   7|" in wb.view(7, 7)
    assert wb.reset()["identical_to_champion"]


def test_toolbox_budgets_and_single_submission(ctrl):
    tb = AgentToolbox(ctrl, max_dry_runs=1)
    assert json.loads(tb.dry_run_discovery())["status"] == "DRY_RUN_SKIPPED"
    assert json.loads(tb.submit_candidate_for_holdout("h", ""))["status"] == "SUBMISSION_REJECTED"
    assert json.loads(tb.edit_candidate("WIDTH = 0.05", "WIDTH = 0.07"))["ok"]
    assert json.loads(tb.dry_run_discovery())["status"] == "DRY_RUN_OK"
    assert json.loads(tb.dry_run_discovery())["status"] == "DRY_RUN_BUDGET_EXHAUSTED"
    first = json.loads(tb.submit_candidate_for_holdout("wider", "notes"))
    assert first["status"] in ("PROMOTED", "REVERTED") and first["iteration"] == 1
    assert json.loads(tb.propose_and_test_forecaster("again", "", CAND))["status"] == "SUBMISSION_BUDGET_EXHAUSTED"
    assert len(tb.results) == 1
    assert json.loads(tb.update_scientist_prompt("new playbook " * 30, "r"))["iteration"] == 1
    assert _statuses(ctrl).count("DRY_RUN_OK") == 1 and ctrl.next_iteration() == 2


def test_inspect_state_tool_returns_valid_json(ctrl):
    tb = AgentToolbox(ctrl)
    assert json.loads(tb.inspect_current_state())["next_iteration"] == 1


# ------------------------------------------------------------------- Strands max-tokens recovery ----
class _FakeModel:
    def __init__(self):
        self.params = {"max_tokens": 4000}

    def get_config(self):
        return {"params": dict(self.params)}

    def update_config(self, params):
        self.params = params


def _fake_agent_factory(script):
    from strands.types.exceptions import MaxTokensReachedException

    class FakeAgent:
        prompts = []

        def __init__(self, model, tools, system_prompt, callback_handler):
            self.tools = {t.tool_name: t for t in tools}

        def __call__(self, prompt):
            FakeAgent.prompts.append(prompt)
            step = script.pop(0)
            if step == "max_tokens":
                raise MaxTokensReachedException("Model stopped generating due to maximum token limit")
            if step == "submit":
                self.tools["edit_candidate"](old_str="WIDTH = 0.05", new_str="WIDTH = 0.07")
                self.tools["submit_candidate_for_holdout"](hypothesis="h", theory_update_md="")
            if step == "submit_then_max_tokens":
                self.tools["propose_and_test_forecaster"](hypothesis="h", theory_update_md="", new_forecaster_code=CAND)
                raise MaxTokensReachedException("limit during playbook update")

    return FakeAgent


def _driver():
    d = StrandsAgentDriver.__new__(StrandsAgentDriver)
    d.model, d.name = _FakeModel(), "strands_fake"
    return d


def test_strands_driver_recovers_from_max_tokens(ctrl, monkeypatch):
    import strands

    agent = _fake_agent_factory(["max_tokens", "submit"])
    monkeypatch.setattr(strands, "Agent", agent)
    d = _driver()
    r = d.run_iteration(ctrl, 0)
    assert r["status"] in ("PROMOTED", "REVERTED") and d.model.params["max_tokens"] == 8000
    assert "output-token limit" in agent.prompts[1]


def test_strands_driver_keeps_result_if_limit_hits_after_submission(ctrl, monkeypatch):
    import strands

    monkeypatch.setattr(strands, "Agent", _fake_agent_factory(["submit_then_max_tokens"]))
    assert _driver().run_iteration(ctrl, 0)["iteration"] == 1


def test_strands_driver_nudges_then_gives_up(ctrl, monkeypatch):
    import strands
    from strands.types.exceptions import MaxTokensReachedException

    monkeypatch.setattr(strands, "Agent", _fake_agent_factory(["noop", "submit"]))
    assert _driver().run_iteration(ctrl, 0)["iteration"] == 1
    monkeypatch.setenv("LLM_MAX_TOKEN_RETRIES", "1")
    monkeypatch.setattr(strands, "Agent", _fake_agent_factory(["max_tokens", "max_tokens"]))
    with pytest.raises(MaxTokensReachedException):
        _driver().run_iteration(ctrl, 0)


# ---------------------------------------------------------------------------- Devin API polling ----
SO = {"hypothesis": "h", "new_forecaster_code": CAND}


@pytest.mark.parametrize("payload,so,prev,idle_s,expected", [
    ({"status": "running", "status_detail": "working"}, None, None, 0, "wait"),
    ({"status": "exit", "status_detail": "finished"}, None, None, 0, "fail"),
    ({"status_enum": "finished"}, None, None, 0, "fail"),
    ({"status_enum": "expired"}, {"hypothesis": "h"}, None, 0, "fail"),
    ({"status": "running", "status_detail": "waiting_for_user"}, None, None, 60, "wait"),
    ({"status_enum": "blocked"}, None, None, 601, "fail"),
    ({"status": "running", "status_detail": "working"}, SO, None, 0, "wait"),
    ({"status": "running", "status_detail": "working"}, SO, SO, 0, "return"),
    ({"status_enum": "blocked"}, SO, None, 0, "return"),
    ({"status": "exit", "status_detail": "finished"}, SO, None, 0, "return"),
])
def test_devin_poll_action(payload, so, prev, idle_s, expected):
    assert devin_poll_action(devin_status(payload), so, prev, idle_s) == expected


def _devin(payloads, timeout_s=10_000):
    d = DevinAPIDriver.__new__(DevinAPIDriver)
    d.poll_s, d.timeout_s, d.sessions, d.org = 0, timeout_s, [], None
    d._create = lambda prompt, title: ("v1", "sid")
    d._sleep = lambda s: None
    it = iter(payloads)

    def get(api, sid):
        p = next(it)
        if isinstance(p, Exception):
            raise p
        return p

    d._get = get
    return d


def test_devin_propose_fails_fast_on_exit_without_output():
    with pytest.raises(RuntimeError, match="without a complete structured output"):
        _devin([{"status_enum": "working"}, {"status_enum": "finished"}]).propose("p", "t")


def test_devin_propose_tolerates_transient_poll_errors():
    d = _devin([ConnectionError("blip"), {"status_enum": "working", "structured_output": SO},
                {"status_enum": "blocked", "structured_output": SO, "acus_consumed": 1.5}])
    assert d.propose("p", "t") == SO and d.sessions[0]["acus"] == 1.5


def test_devin_propose_gives_up_after_repeated_poll_errors():
    with pytest.raises(ConnectionError):
        _devin([ConnectionError("down")] * 5).propose("p", "t")
