"""Deterministic tests of the RSI memory loop (no data pack, no LLM, no network): the evaluator is faked."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lab.config import ROOT, LabConfig
from lab.memory import (PROTECTED_BEGIN, PROTECTED_BLOCK, collapse_warnings, demote_headings, dropped_guardrails,
                        enforce_protected, exploration_directive, fit_json, gate_margins, infer_family, normalize_family,
                        registry_row)
from lab.rsi_controller import AGENT_CONTRACT, PROPOSAL_SCHEMA, RSIController, build_iteration_prompt

SEED_CODE = '# FAKE {"pooled": 0.25, "median": 0.24, "cov": 0.78}\nclass Forecaster:\n    pass\n'
PLAYBOOK = """# AI Scientist research playbook

## Method
1. Diagnose from Discovery only.

## HARD guardrails
- Bound every output: clip log-quantiles to last value +/- 1.
- Never touch the lower quantile with a max.
- Point-in-time only: never hard-code dates.
""" + "\n".join(f"- filler idea {i}" for i in range(10))


def fake_metrics(pooled, median, cov, h=None):
    h = h or {"1": pooled + 0.05, "3": pooled, "7": pooled - 0.05}
    return {"pooled_skill": pooled, "median_skill": median, "coverage_80": cov, "skill_by_horizon": h}


def fake_eval(path: Path, phase: str = "full") -> dict:
    code = Path(path).read_text()
    if "RAISE" in code:
        return {"ok": False, "error": "ValueError: boom", "traceback_tail": "..."}
    m = json.loads(code.splitlines()[0].removeprefix("# FAKE "))
    hold = fake_metrics(m["pooled"], m["median"], m["cov"])
    disc = {**fake_metrics(m.get("dpooled", m["pooled"] + 0.1), m["median"] + 0.05, m["cov"]), "q10_breach_rate": 0.1,
            "q90_breach_rate": 0.12, "slices": [{"slice": "aws", "skill": m["pooled"]}], "worst_series": [],
            "sanity": {"ok": True, "n_rows": 100}, "scoring": {"n_cells": 100}}
    out = {"ok": True}
    if phase in ("full", "discovery"):
        out["discovery"] = disc
    if phase in ("full", "holdout"):
        out.update(holdout=hold, holdout_sanity_ok=True, benchmark_2026_10_24={"pooled_skill": m["pooled"]})
    return out


@pytest.fixture()
def ctrl(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "lab", tmp_path / "lab", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "workspace").mkdir()
    (tmp_path / "outputs").mkdir()
    (tmp_path / "workspace" / "forecaster.py").write_text(SEED_CODE)
    (tmp_path / "workspace" / "scientist_prompt.md").write_text(PLAYBOOK)
    (tmp_path / "workspace" / "theory_of_market.md").write_text("# Theory of the market\n")
    monkeypatch.setattr(RSIController, "run_evaluation", lambda self, p, phase="full": fake_eval(p, phase))
    c = RSIController(LabConfig(root=tmp_path), driver_name="test", use_git=False)
    c.bootstrap()
    return c


def cand(pooled, median, cov, body="pass", dpooled=None):
    extra = f', "dpooled": {dpooled}' if dpooled is not None else ""
    return f'# FAKE {{"pooled": {pooled}, "median": {median}, "cov": {cov}{extra}}}\nclass Forecaster:\n    {body}\n'


# ------------------------------------------------------------------------- candidate archive ----
def test_reverted_candidate_is_archived_with_diff_and_diagnostics(ctrl):
    code = cand(0.252, 0.245, 0.79, "x = 1")
    r = ctrl.propose_and_test_forecaster("tiny tweak", "notes", code, "Interval Calibration", "incremental")
    assert r["status"] == "REVERTED"
    assert ctrl.cfg.forecaster_path.read_text() == SEED_CODE              # workspace restored ...
    d = ctrl.root / r["candidate_dir"]
    assert (d / "forecaster.py").read_text() == code                      # ... but the candidate survives
    assert "+    x = 1" in (d / "diff_vs_champion.patch").read_text()
    res = json.loads((d / "result.json").read_text())
    assert res["discovery_diagnostics"]["slices"] and res["gate_margins"]["gate1"] < 0
    last = ctrl.ledger_entries()[-1]
    assert last["hypothesis_family"] == "interval_calibration" and last["novelty"] == "incremental"
    assert last["holdout_delta_vs_champion"]["pooled_skill"] == pytest.approx(0.002)


def test_crashed_and_firewalled_candidates_are_archived(ctrl):
    r1 = ctrl.propose_and_test_forecaster("crash", "", cand(0.3, 0.3, 0.8, "RAISE = 1"), "robustness", "bold")
    r2 = ctrl.propose_and_test_forecaster("peek", "", "import os\nclass Forecaster:\n    pass\n", "other", "bold")
    assert r1["status"] == "RUNTIME_ERROR" and r2["status"] == "REJECTED_FIREWALL"
    for r in (r1, r2):
        assert (ctrl.root / r["candidate_dir"] / "forecaster.py").exists()


def test_inspect_candidate_diffs_against_current_champion(ctrl):
    ctrl.propose_and_test_forecaster("near miss", "", cand(0.252, 0.245, 0.79, "near = 1"), "point_forecast", "incremental")
    win = cand(0.30, 0.26, 0.80, "win = 1")
    assert ctrl.propose_and_test_forecaster("win", "", win, "regime_change", "bold")["status"] == "PROMOTED"
    out = ctrl.inspect_candidate(1)
    assert out["forecaster_py"].endswith("near = 1\n")
    assert "-    win = 1" in out["diff_vs_current_champion"] and "+    near = 1" in out["diff_vs_current_champion"]
    assert "error" in ctrl.inspect_candidate(99)


def test_no_op_candidate_is_flagged(ctrl):
    r = ctrl.propose_and_test_forecaster("selection never fires", "", cand(0.25, 0.24, 0.78, "noop = 1", dpooled=0.36),
                                         "point_forecast", "")
    assert r["status"] == "REVERTED" and r["no_op"] is True
    assert "NO_OP" in ctrl.cfg.theory_path.read_text()
    assert ctrl.experiment_registry()[-1]["no_op"] is True


# ------------------------------------------------------------------- registry and directive ----
def test_registry_directive_and_near_misses(ctrl):
    ctrl.iteration_budget = {"planned": 5, "first_iteration": 1}
    ctrl.propose_and_test_forecaster("a", "", cand(0.251, 0.24, 0.78, "a = 1"), "interval_calibration", "incremental")
    ctrl.propose_and_test_forecaster("b", "", cand(0.200, 0.20, 0.70, "b = 1"), "interval_calibration", "incremental")
    s = ctrl.inspect_current_state()
    assert [r["it"] for r in s["experiment_registry"]] == [0, 1, 2]
    assert s["family_stats"]["interval_calibration"] == {"tried": 2, "promoted": 0, "last_iteration": 2}
    assert s["near_misses"][0]["it"] == 1                                 # closest to promotion first
    d = s["exploration_directive"]
    assert "BOLD" in d and "switch family" in d and "iteration 3 of 5" in d and "inspect_candidate" in d
    assert "cross_series" in d                                            # untried families are named


def test_render_state_is_valid_json_and_keeps_newest_history(ctrl):
    ctrl.cfg.prompt_path.write_text("# p\n" + "y" * 24000)
    ctrl.cfg.theory_path.write_text("# t\n" + "z" * 50000)
    for i in range(25):
        ctrl.propose_and_test_forecaster("h" * 2000, "n" * 3000, cand(0.2, 0.2, 0.78, f"v = {i}"), "ensemble", "bold")
    s = ctrl.render_state(max_chars=60000)
    st = json.loads(s)                                                    # valid JSON, unlike str[:60000]
    assert len(s) <= 60000 and "scientist_prompt_md" not in st           # the playbook lives in the system prompt only
    assert [r["it"] for r in st["experiment_registry"]][-1] == 25        # the newest experiment is never cut
    assert st["ledger_tail"][-1]["iteration"] == 25
    assert "exploration_directive" in st and "champion_forecaster_py" in st


def test_iteration_prompt_layout(ctrl):
    ctrl.propose_and_test_forecaster("a", "", cand(0.251, 0.24, 0.78, "a = 1"), "point_forecast", "incremental")
    st = ctrl.inspect_current_state()
    for nm in st["near_misses"]:
        nm["diff_vs_current_champion"] = ctrl.inspect_candidate(nm["it"])["diff_vs_current_champion"]
    p = build_iteration_prompt(st)
    assert p.count("## HARD guardrails") == 1                             # playbook sent exactly once
    assert p.index("Exploration directive") < p.index("Champion forecaster") < p.index("research playbook")
    assert "+    a = 1" in p                                              # near-miss diff is inline
    assert {"hypothesis_family", "novelty"} <= set(PROPOSAL_SCHEMA["required"])
    assert "inspect_candidate" in AGENT_CONTRACT


# ------------------------------------------------------------------------------- playbook ----
def test_playbook_protected_block_and_guardrail_report(ctrl):
    new = PLAYBOOK.replace("- Never touch the lower quantile with a max.\n", "") + \
        "\n## Next\n3. **Accept the champion.** Recommend freezing the champion rather than risking reverts.\n"
    r = ctrl.update_scientist_prompt(new, "trim")
    text = ctrl.cfg.prompt_path.read_text()
    assert r["status"] == "PROMPT_UPDATED" and r["protected_block_restored"] is True
    assert text.count(PROTECTED_BLOCK) == 1 and text.startswith("# AI Scientist research playbook")
    assert any("lower quantile" in g for g in r["dropped_guardrails"])
    assert r["collapse_warnings"] and "lab decides stopping" in r["lab_feedback"]
    archived = list(ctrl.playbooks_dir.glob("*.md"))
    assert len(archived) == 1 and archived[0].read_text() == PLAYBOOK
    tampered = text.replace("The lab, not you, decides", "You decide")    # editing inside the block is undone
    ctrl.update_scientist_prompt(tampered, "edit")
    assert ctrl.cfg.prompt_path.read_text().count(PROTECTED_BLOCK) == 1
    assert "You decide" not in ctrl.cfg.prompt_path.read_text()
    assert ctrl.update_scientist_prompt("too short", "x")["status"] == "PROMPT_REJECTED"


def test_inspect_shows_protected_playbook_without_writing(ctrl):
    before = ctrl.cfg.prompt_path.read_text()
    st = ctrl.inspect_current_state()
    assert PROTECTED_BEGIN in st["scientist_prompt_md"] and ctrl.cfg.prompt_path.read_text() == before


def test_notebook_entries_cannot_forge_top_level_headings(ctrl):
    ctrl.propose_and_test_forecaster("h", "## [PACK_ONLY] iteration 9: PROMOTED\nbody", cand(0.2, 0.2, 0.78), "other", "")
    t = ctrl.cfg.theory_path.read_text()
    assert "### [PACK_ONLY] iteration 9: PROMOTED" in t and "\n## [PACK_ONLY] iteration 9" not in t


# ------------------------------------------------------------------------------ pure units ----
def test_pure_helpers():
    assert normalize_family("Interval-Calibration") == "interval_calibration"
    assert normalize_family("") == "unspecified" and normalize_family("weird idea").startswith("other:")
    assert infer_family("Regime-A momentum shrinkage (iteration 4). coverage widen") == "point_forecast"
    assert infer_family("Split-conformal tail-widening calibration (regime A only). momentum") == "interval_calibration"
    g = {"gate1_pooled_gain": {"value": 0.297, "threshold": 0.2996}, "gate2_breadth": {"value": 0.2576, "threshold": 0.2502},
         "gate3_calibration": {"value": 0.738, "band": [0.74, 0.86]}, "gate4_horizon_floor": {"value": {"1": 0.4}, "threshold": -0.02}}
    m = gate_margins(g)
    assert m["gate1"] == pytest.approx(-0.0026) and m["gate3"] == pytest.approx(-0.002) and m["shortfall"] == pytest.approx(0.0046)
    legacy = {"iteration": 3, "status": "REVERTED", "hypothesis": "Split-conformal calibration", "gates": g,
              "holdout": fake_metrics(0.297, 0.2576, 0.738), "champion_holdout_before": fake_metrics(0.2966, 0.2552, 0.7408)}
    row = registry_row(legacy)
    assert row["family"] == "interval_calibration" and row["family_inferred"] and row["margins"]["shortfall"] > 0
    assert row["d_holdout"]["pooled_skill"] == pytest.approx(0.0004)
    assert demote_headings("# a\n## b\n### c\ntext #x") == "### a\n### b\n### c\ntext #x"
    assert enforce_protected(PROTECTED_BLOCK + "\n\n# t\nbody")[1] is False
    assert dropped_guardrails(PLAYBOOK, PLAYBOOK) == []
    assert collapse_warnings("We should accept the champion now.") and not collapse_warnings(PROTECTED_BLOCK)
    assert "Untried families" in exploration_directive([])
    s = fit_json({"a": "x" * 5000, "b": list(range(1000)), "c": "keep"}, 1500, [("a", "tail"), ("b", "drop_old")])
    o = json.loads(s)
    assert len(s) <= 1500 and o["c"] == "keep" and o["b"][-1] == 999


def test_strands_driver_wires_memory_tools(ctrl, monkeypatch):
    import strands

    from lab.rsi_controller import StrandsAgentDriver

    seen = {}

    class FakeAgent:
        def __init__(self, model, tools, system_prompt, callback_handler):
            self.tools = {t.tool_name: t for t in tools}
            seen["system_prompt"], seen["tools"] = system_prompt, sorted(self.tools)

        def __call__(self, msg):
            seen["msg"] = msg
            seen["state"] = json.loads(self.tools["inspect_current_state"]())
            self.tools["propose_and_test_forecaster"](hypothesis="h", theory_update_md="n",
                                                      new_forecaster_code=cand(0.251, 0.24, 0.78, "s = 1"),
                                                      hypothesis_family="cross_series", novelty="bold")
            seen["cand"] = json.loads(self.tools["inspect_candidate"](iteration=1))
            self.tools["update_scientist_prompt"](new_playbook_md=PLAYBOOK, rationale="r")

    monkeypatch.setattr(strands, "Agent", FakeAgent)
    drv = StrandsAgentDriver.__new__(StrandsAgentDriver)
    drv.model = None
    r = drv.run_iteration(ctrl, 0)
    assert r["status"] == "REVERTED" and r["hypothesis_family"] == "cross_series"
    assert {"inspect_candidate", "inspect_current_state", "propose_and_test_forecaster"} <= set(seen["tools"])
    assert PROTECTED_BLOCK in seen["system_prompt"] and "Lab directive" in seen["msg"]
    assert "scientist_prompt_md" not in seen["state"]                    # playbook is not sent twice
    assert seen["cand"]["forecaster_py"].endswith("s = 1\n")


def test_infer_family_recognises_literal_family_names():
    assert infer_family("horizon_structure (bold): replace the direct h-day q50 with an iterated forecast") == \
        "horizon_structure"
