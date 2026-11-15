"""Architecture 1 (memory-based RSI): versioned procedure, revision validation, admission gate, forecast locks,
evidence/episode memory and one full generation with a scripted driver (no LLM, no network, no data pack)."""
import json
import shutil
from pathlib import Path

import pytest

from lab.config import LabConfig
from lab.memory import PROTECTED_BEGIN
from lab.procedure import (ProcedureRegistry, admission_rule, read_jsonl, validate_card, validate_revision,
                           write_lock)
from lab.rsi_controller import AgentToolbox, RSIController, parse_json_object, run_generation, traced_tools
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
    row = run_generation(lab, drv)
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
    row2 = run_generation(lab, drv)                         # recursive step: the improver now runs under v001
    assert row2["active_before"] == "v001" and row2["candidate_version"] is None
    assert "REVISION_INVALID" in [r["status"] for r in lab.ledger_entries()]
    assert [e["episode_id"] for e in lab.released_episodes()][-1] == "g002_frozen"
    assert "g002_frozen" in (lab.procedures.root / "lineage.md").read_text()
