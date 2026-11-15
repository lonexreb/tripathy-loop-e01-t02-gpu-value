"""Architecture 1 (memory-based RSI): versioned procedure, revision validation, admission gate, forecast locks,
evidence/episode memory and one full generation with a scripted driver (no LLM, no network, no data pack)."""
import json
import shutil
from pathlib import Path

import pytest

from lab.config import LabConfig
from lab.memory import PROTECTED_BEGIN
from lab.procedure import (CONTROL_TOTAL, DEFAULT_CONTROLS, MAX_COMPONENT_CHARS, ProcedureRegistry, admission_rule,
                           append_jsonl, procedure_controls, read_jsonl, validate_card, validate_revision, write_lock)
from lab.rsi_controller import (REVISION_PROTOCOL, AgentToolbox, RSIController, improver_prompt, parse_json_object,
                                run_generation, traced_tools)
from tests.test_driver_loop import CAND, CHAMP, HOLD, _disc

ROOT = Path(__file__).resolve().parents[1]
POLICY = "Research policy: diagnose the largest Discovery errors first, then test a different model class. " * 4
CARD = {"mechanism": "AWS spot prices reprice in steps after capacity shifts",
        "expected_observation": "h7 pinball loss falls on post-jump origins",
        "competing_explanation": "the collector's provider sample changed instead",
        "test": "dry run vs champion on all Discovery origins",
        "rejection_condition": "no pooled-skill gain or coverage outside 0.70-0.90"}


@pytest.fixture
def lab(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "lab", tmp_path / "lab", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace" / "forecaster.py").write_text(CHAMP)
    (tmp_path / "workspace" / "scientist_prompt.md").write_text(POLICY)
    c = RSIController(LabConfig(root=tmp_path), driver_name="test", use_git=False)
    c.champ_dir.mkdir(parents=True)
    c.backup_path.write_text(CHAMP)
    c.state_path.write_text(json.dumps({"stage": c.cfg.stage, "iteration": 0, "code_sha256": "x",
                                        "discovery": _disc(0.25), "holdout": HOLD, "benchmark_2026_10_24": {}}))

    def fake_child(module_args, timeout_s):
        return {"ok": True, "discovery": _disc(0.26), "holdout": {**HOLD, "pooled_skill": 0.29},
                "benchmark_2026_10_24": {}}

    monkeypatch.setattr(c, "_run_lab_child", fake_child)
    return c


def _rev(active, **kw):
    r = {"parent_version": active["version"], "target_component": "research_policy",
         "proposed_by": f"improver_{active['version']}", "change": "profile the pack before choosing a model class",
         "supporting_episode_ids": ["iter_008"], "expected_behavior_change": "profile_pack is called before any dry run",
         "development_comparison": "same-start episode vs frozen v000", "rejection_condition": "no Discovery gain",
         "status": "proposed", "new_component_text": POLICY + "\n- Always call profile_pack first."}
    r.update(kw)
    return r


def test_registry_lineage_admit_and_reject(tmp_path):
    reg = ProcedureRegistry(tmp_path / "proc")
    v0 = reg.ensure_seed(POLICY)
    assert v0["version"] == "v000" and PROTECTED_BEGIN in v0["components"]["research_policy"]
    assert reg.ensure_seed("other " * 50)["version"] == "v000"
    c1 = reg.add_candidate(_rev(v0))
    assert c1["parent_version"] == "v000" and c1["status"] == "candidate" and reg.active()["version"] == "v000"
    reg.decide(c1["version"], {"admitted": False})
    assert reg.active()["version"] == "v000" and reg.get("v001")["revision"]["status"] == "rejected"
    c2 = reg.add_candidate(_rev(v0, target_component="improver_policy", new_component_text="Improver. " * 40))
    reg.decide(c2["version"], {"admitted": True})
    assert reg.active()["version"] == "v002" and reg.get("v000")["status"] == "superseded"
    assert reg.active()["components"]["research_policy"] == v0["components"]["research_policy"]
    assert "v001" in reg.lineage_md() and "rejected" in reg.lineage_md()


def test_validate_revision_rejects_bad_objects(tmp_path):
    active = ProcedureRegistry(tmp_path / "p").ensure_seed(POLICY)
    assert validate_revision(_rev(active), active, set()) == []
    assert validate_revision(None, active, set())
    assert any("parent_version" in x for x in validate_revision(_rev(active, parent_version="v009"), active, set()))
    assert any("target_component" in x for x in validate_revision(_rev(active, target_component="scorer"), active, set()))
    assert any("identical" in x for x in validate_revision(
        _rev(active, new_component_text=active["components"]["research_policy"]), active, set()))
    assert any("cite" in x for x in validate_revision(_rev(active, supporting_episode_ids=["nope"]), active, {"g001_frozen"}))
    assert validate_revision(_rev(active, supporting_episode_ids=["g001_frozen"]), active, {"g001_frozen"}) == []
    assert any("status" in x for x in validate_revision(_rev(active, status="admitted"), active, set()))


def test_admission_rule_is_fixed_and_conservative():
    ep = lambda d, ok=True: {"lock": {"screen_pass": ok, "discovery_pooled_delta": d}}  # noqa: E731
    assert admission_rule(ep(0.001), ep(0.004))["admitted"]
    assert not admission_rule(ep(0.004), ep(0.004))["admitted"]           # ties keep I_g
    assert not admission_rule({"lock": None}, ep(0.05, ok=False))["admitted"]  # failed screen scores 0
    assert admission_rule({"lock": None}, ep(0.002))["admitted"]


def test_lock_is_immutable_and_card_is_required(tmp_path):
    write_lock(tmp_path, "g001_frozen", {"code_sha256": "a"})
    with pytest.raises(FileExistsError):
        write_lock(tmp_path, "g001_frozen", {"code_sha256": "b"})
    assert validate_card(CARD) == []
    assert set(validate_card({"mechanism": "x"})) == {"mechanism", "expected_observation", "competing_explanation",
                                                       "test", "rejection_condition"}
    assert parse_json_object('blah ```json\n{"a": 1}\n``` tail') == {"a": 1}
    assert parse_json_object("no json here") is None


def test_episode_toolbox_commit_exa_budget_and_trace(lab, monkeypatch):
    tb = AgentToolbox(lab, mode="episode", episode_id="g001_frozen", procedure_version="v000")
    assert "commit_forecast" in tb.EPISODE_TOOL_NAMES and "submit_candidate_for_holdout" not in tb.EPISODE_TOOL_NAMES
    assert json.loads(tb.commit_forecast("h", "x", "", "", "", ""))["status"] == "COMMIT_REJECTED"
    assert "identical" in json.loads(tb.commit_forecast("h", **CARD))["error"]
    tb.write_candidate(CAND)
    out = json.loads(tb.commit_forecast("wider bands", **CARD, model_class="pooled_linear_momentum"))
    assert out["status"] == "LOCKED" and out["screen_pass"] and abs(out["discovery_pooled_delta"] - 0.01) < 1e-9
    lock = json.loads((lab.locks_dir / "g001_frozen.json").read_text())
    assert lock["procedure_version"] == "v000" and lock["hypothesis_card"] == CARD
    assert (lab.locks_dir / "g001_frozen.py").read_text() == CAND
    assert json.loads(tb.commit_forecast("again", **CARD))["status"] == "ALREADY_LOCKED"
    monkeypatch.setattr(lab, "search_exa_signals", lambda q, o: [{"url": "u", "title": "t", "publishedDate": o}])
    assert tb.max_exa == 2
    for _ in range(2):
        assert json.loads(tb.search_exa_signals("GPU price cut", "2026-08-01", "repricing steps"))[0]["url"] == "u"
    assert json.loads(tb.search_exa_signals("q", "2026-08-01"))["status"] == "EXA_BUDGET_EXHAUSTED"
    ev = read_jsonl(lab.evidence_path)
    assert len(ev) == 2 and ev[0]["hypothesis_link"] == "repricing steps" and ev[0]["episode_id"] == "g001_frozen"
    t = traced_tools(tb, ["list_variants"])[0]
    t()
    assert tb.trace[-1]["tool"] == "list_variants"
    assert json.loads(tb.recall_memory())["evidence_memory"]


class FakeDriver:
    """Scripted improver + episodes: the candidate procedure locks CAND, the frozen one commits nothing."""

    def __init__(self, revs):
        self.revs = list(revs)

    def propose_revision(self, ctrl, active, digest, known):
        rev = self.revs.pop(0)(active)
        p = validate_revision(rev, active, known)
        return (None, p) if p else (rev, [])

    def run_episode(self, ctrl, version, episode_id, arm):
        tb = AgentToolbox(ctrl, mode="episode", episode_id=episode_id, procedure_version=version["version"])
        tb.trace.append({"tool": "inspect_current_state", "status": None, "args": {}})
        if arm == "candidate":
            tb.write_candidate(CAND)
            tb.commit_forecast("wider bands", **CARD)
        return tb


def test_generation_admits_candidate_procedure_and_spends_one_holdout_shot(lab):
    drv = FakeDriver([lambda a: _rev(a), lambda a: _rev(a, supporting_episode_ids=["unknown"])])
    row = run_generation(lab, drv, pairs=1)
    st = [r["status"] for r in lab.ledger_entries()]
    assert st.count("REVISION_PROPOSED") == 1 and "PROCEDURE_ADMITTED" in st and st[-1] == "GENERATION"
    assert row["active_after"] == "v001" and row["episode_scores"] == {"frozen": 0.0, "candidate": 0.01}
    assert row["holdout_shot"]["status"] == "REVERTED"      # fake Holdout 0.29 < champion 0.30
    assert "profile_pack first" in lab.cfg.prompt_path.read_text()
    eps = lab.released_episodes()
    assert [e["episode_id"] for e in eps] == ["g001_frozen", "g001_candidate"]
    assert eps[1]["holdout_result"]["status"] == "REVERTED" and eps[0]["holdout_result"] is None
    exp = [r for r in lab.ledger_entries() if r.get("status") == "REVERTED"][0]
    assert exp["hypothesis_card"]["test"] == CARD["test"]
    row2 = run_generation(lab, drv, pairs=1)                         # recursive step: the improver now runs under v001
    assert row2["active_before"] == "v001" and row2["candidate_version"] is None
    assert "REVISION_INVALID" in [r["status"] for r in lab.ledger_entries()]
    assert [e["episode_id"] for e in lab.released_episodes()][-1] == "g002_frozen"
    assert "g002_frozen" in (lab.procedures.root / "lineage.md").read_text()


def test_episode_crashes_are_refunded_and_never_locked(lab, monkeypatch):
    """A candidate or analysis that does not run costs no budget (bounded) and commit_forecast refuses to lock it."""
    tb = AgentToolbox(lab, mode="episode", episode_id="g001_frozen", procedure_version="v000")
    assert tb.max_dry_runs == 6
    ok = lab._run_lab_child
    monkeypatch.setattr(lab, "_run_lab_child", lambda a, t: {"ok": False, "error": "ValueError: boom"})
    tb.write_candidate(CAND)
    out = json.loads(tb.dry_run_discovery())
    assert out["status"] == "DRY_RUN_ERROR" and out["refunded"] and out["dry_runs_left"] == 6
    out = json.loads(tb.analyze_discovery("def analyze(pack_slice, origin_date):\n    return {}\n", ["2026-07-15"]))
    assert out["status"] == "ANALYSIS_ERROR" and out["refunded"] and out["analyses_left"] == tb.max_analyses
    out = json.loads(tb.commit_forecast("wider bands", **CARD))
    assert out["status"] == "COMMIT_REJECTED" and "boom" in out["error"] and tb.lock is None
    assert not (lab.locks_dir / "g001_frozen.json").exists()
    for _ in range(40):  # refunds are bounded: a crash loop still exhausts the budget
        last = json.loads(tb.dry_run_discovery())
    assert last["status"] == "DRY_RUN_BUDGET_EXHAUSTED"
    monkeypatch.setattr(lab, "_run_lab_child", ok)
    assert json.loads(tb.commit_forecast("wider bands", **CARD))["status"] == "LOCKED"
    rows = [r for r in lab.ledger_entries(lab.cfg.stage) if r["status"] == "ANALYSIS"]
    assert rows and "seconds" in rows[0]


def test_variants_carry_to_later_generations_only(lab):
    tb = AgentToolbox(lab, mode="episode", episode_id="g001_frozen", procedure_version="v000")
    tb.write_candidate(CAND)
    assert json.loads(tb.save_variant("wide bands"))["saved"] == "wide_bands"
    same_gen = AgentToolbox(lab, mode="episode", episode_id="g001_candidate", procedure_version="v001")
    assert same_gen.variants == {}  # both arms of one admission comparison start from the same state
    nxt = AgentToolbox(lab, mode="episode", episode_id="g002_frozen", procedure_version="v000")
    assert json.loads(nxt.list_variants())["g001_frozen__wide_bands"]["carried"]
    assert json.loads(nxt.load_variant("g001_frozen__wide_bands"))["ok"] and nxt.wb.code == CAND
    assert AgentToolbox(lab).variants == {}  # the legacy Holdout loop is unchanged


def test_backtest_output_keeps_summary_when_cells_overflow():
    from lab.rsi_controller import fit_backtest
    from lab.workbench import clip_json
    cell = {"origin_date": "2026-08-01", "horizon_days": 7, "y": 1.2345, "q10": 1.0, "q50": 1.2, "q90": 1.4,
            "loss": 0.0123, "loss_ref": 0.0234, "baseline_loss": 0.03}
    r = {"status": "BACKTEST_OK", "pooled_skill_delta": 0.01, "series_cells": {f"s{i}": [cell] * 210 for i in range(5)},
         "backtests_left": 3}
    assert json.loads(clip_json(r, 30000)).get("truncated")  # what the agent used to get: a cut string
    out = json.loads(clip_json(fit_backtest(r, 30000), 30000))
    assert out["backtests_left"] == 3 and out["pooled_skill_delta"] == 0.01 and "thinned" in out["series_cells_note"]
    assert all(3 <= len(v) < 210 for v in out["series_cells"].values())
    small = {"status": "BACKTEST_OK", "series_cells": {"s": [cell] * 5}}
    assert fit_backtest(small, 30000) is small


def test_paired_admission_rule_needs_a_gain_larger_than_its_noise():
    ep = lambda d: {"lock": {"screen_pass": True, "discovery_pooled_delta": d}}  # noqa: E731
    eps = lambda *ds: [ep(d) for d in ds]  # noqa: E731
    a = admission_rule(eps(0.01, 0.0, 0.02), eps(0.02, 0.011, 0.029))       # consistent +0.01 in every pair
    assert a["admitted"] and a["n_pairs"] == 3 and a["pairs_won"] == 3 and abs(a["difference"] - 0.01) < 1e-6
    b = admission_rule(eps(0.035, 0.0, 0.01), eps(0.027, 0.03, 0.0))        # mean +0.004 but se 0.013: noise
    assert not b["admitted"] and b["difference"] > 0.001 and b["difference_se"] > b["difference"]
    assert not admission_rule(eps(0.0, 0.0, 0.0), eps(0.0, 0.0, 0.0))["admitted"]
    assert not admission_rule(eps(0.01), [])["admitted"]                    # missing arm keeps I_g


def test_episode_controls_are_a_typed_bounded_revision_target(lab):
    v0 = lab.ensure_procedure()
    assert procedure_controls(v0) == DEFAULT_CONTROLS == procedure_controls({"components": {}})   # legacy versions
    ctl = lambda **kw: _rev(v0, target_component="episode_controls", new_component_text=json.dumps(kw))  # noqa: E731
    ok = ctl(max_dry_runs=6, max_backtests=3, max_analyses=4)
    assert sum(json.loads(ok["new_component_text"]).values()) == CONTROL_TOTAL and validate_revision(ok, v0, {"iter_008"}) == []
    assert any("sum to" in x for x in validate_revision(ctl(max_dry_runs=8, max_backtests=4, max_analyses=6), v0, set()))
    assert any("integer in" in x for x in validate_revision(ctl(max_dry_runs=0, max_backtests=7, max_analyses=6), v0, set()))
    assert any("exactly the keys" in x for x in validate_revision(ctl(max_dry_runs=13, max_exa=0), v0, set()))
    assert any("identical" in x for x in validate_revision(ctl(**DEFAULT_CONTROLS), v0, set()))
    c1 = lab.procedures.add_candidate(ok)
    tb = AgentToolbox(lab, mode="episode", episode_id="e1", procedure_version="v001", controls=procedure_controls(c1))
    assert (tb.max_dry_runs, tb.max_backtests, tb.max_analyses) == (6, 3, 4)
    assert AgentToolbox(lab, controls=procedure_controls(c1)).max_analyses == 6     # Holdout-mode budgets untouched


def test_generation_replicates_pairs_and_tests_improver_policy_through_its_follow_up(lab):
    imp = lambda a: _rev(a, target_component="improver_policy",  # noqa: E731
                         new_component_text="Improver policy: rank repeated failures by lost Discovery skill. " * 4)
    imp2 = lambda a: _rev(a, target_component="improver_policy", supporting_episode_ids=["g001_frozen"],  # noqa: E731
                          new_component_text=f"Improver policy {a['version']}: cite the worst Discovery slice first. " * 4)
    drv = FakeDriver([imp, lambda a: _rev(a), imp2, imp2])
    row = run_generation(lab, drv, pairs=3)
    eps = lab.released_episodes()
    assert [e["episode_id"] for e in eps] == ["g001_frozen", "g001_candidate", "g001_candidate_r2", "g001_frozen_r2",
                                              "g001_frozen_r3", "g001_candidate_r3"]
    adm = row["admission"]
    assert adm["n_pairs"] == 3 and adm["admitted"] and adm["candidate_scores"] == [0.01] * 3
    v1 = lab.procedures.active()
    assert v1["version"] == "v001" and v1["revision"]["target_component"] == "improver_policy"
    assert v1["revision"]["follow_up_revision"]["target_component"] == "research_policy"
    assert "rank repeated failures" in v1["components"]["improver_policy"]
    assert "profile_pack first" in v1["components"]["research_policy"] == lab.cfg.prompt_path.read_text()
    assert sum(e["holdout_result"] is not None for e in eps) == 1          # still one Holdout shot per generation
    row2 = run_generation(lab, drv, pairs=3)       # an improver_policy change with no measurable follow-up is refused
    assert row2["candidate_version"] is None and len(lab.released_episodes()) == 7
    assert "must propose a research_policy" in [r for r in lab.ledger_entries() if r["status"] == "REVISION_INVALID"][-1]["error"]


def test_improver_digest_stays_valid_json_and_carries_measured_procedural_knowledge(lab):
    lab.ensure_procedure()
    trace = [{"tool": "dry_run_discovery", "status": "DRY_RUN_OK", "args": {"note": "x" * 120}}] * 60
    for i in range(12):
        mc = "gradient_boosting" if i % 2 else "pooled_linear_momentum"
        append_jsonl(lab.episodes_path, {"episode_id": f"g{i:03d}_frozen", "generation": i, "procedure_version": "v000",
                                         "score": 0.03 if i % 2 else 0.0, "trace": trace, "tool_counts": {},
                                         "lock": {"model_class": mc, "hypothesis": "h" * 1500}, "holdout_result": None})
    lab._ledger({"status": "REVISION_INVALID", "generation": 1, "parent_version": "v000", "error": "too long"})
    d = lab.memory_digest()
    pm = d["procedural_memory"]
    assert pm["episode_scores_by_model_class"]["gradient_boosting"]["mean_score"] == 0.03
    assert pm["episode_scores_by_procedure"]["v000"]["n"] == 12 and pm["episode_scores_by_procedure"]["v000"]["sd_score"] > 0
    assert pm["invalid_revisions"][0]["error"] == "too long" and pm["episode_controls"] == DEFAULT_CONTROLS
    assert "trace" not in d["episode_memory"][0] and d["episode_memory"][0]["tool_sequence"].startswith("dry_run_discovery:DRY_RUN_OK")
    assert len(json.dumps(lab.released_episodes())) > 100_000              # the raw episodes would not fit in 45k
    body = improver_prompt(lab.procedures.active(), d).split("evidence)\n")[1].split("\n\nPropose")[0]
    assert json.loads(body)["procedural_memory"]["active_version"] == "v000" and "truncated" not in json.loads(body)
    rec = json.loads(AgentToolbox(lab, mode="episode", episode_id="e").recall_memory())
    assert rec["procedural_memory"]["episode_scores_by_model_class"] and "experiment_registry" in rec
    assert str(MAX_COMPONENT_CHARS) in REVISION_PROTOCOL and "episode_controls" in REVISION_PROTOCOL


def test_evaluator_children_get_bounded_thread_caps():
    from lab.rsi_controller import eval_thread_env
    assert eval_thread_env(cpu_count=8, env={}) == {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                                                     "OPENBLAS_NUM_THREADS": "4"}
    assert eval_thread_env(cpu_count=1, env={})["OMP_NUM_THREADS"] == "1"
    out = eval_thread_env(cpu_count=8, env={"LAB_EVAL_THREADS": "2", "MKL_NUM_THREADS": "6"})
    assert out == {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "2"}
