"""Introspection: the scientist's diagnosis phase, the improver's capability requests, lab self-service of dated
page snapshots, and their persistence in memory and the lineage report (no LLM, no network, no data pack)."""
import json

from lab.config import LabConfig
from lab.procedure import validate_capability_request, validate_diagnosis
from lab.rsi_controller import AgentToolbox, run_generation
from lab.sandbox import ExaSignals, PointInTimeSandbox, html_to_text
from tests.test_procedure import CAND, CARD, FakeDriver, _rev, lab  # noqa: F401  (pytest fixture)

DIAG = {"limiting_factors": ["AWS spot carries 97% of pooled loss; the 7-day horizon 63% of it (state.loss_decomposition)"],
        "data_quality_issues": ["122-day AWS gap 2026-03..06 breaks every 28-day feature for post-gap origins"],
        "data_needed": ["provider list prices as they stood at each origin (dated pricing pages), to anchor spot levels"],
        "tools_needed": [],
        "hypotheses": ["a level-anchored model using on-demand price ratios beats pure momentum at h7"]}
REQ = {"kind": "data_source", "request": "point-in-time snapshots of the AWS p5 on-demand pricing page",
       "evidence": ["d001"], "expected_effect": "spot/on-demand ratio features", "how_to_test": "same-start episodes",
       "url": "https://aws.amazon.com/ec2/pricing/on-demand/", "as_of": "2026-09-01"}


def test_diagnosis_and_request_validation():
    assert validate_diagnosis(DIAG) == []
    assert any("limiting_factors" in p for p in validate_diagnosis({**DIAG, "limiting_factors": []}))
    assert any("at least one need" in p for p in validate_diagnosis(
        {**DIAG, "data_needed": [], "tools_needed": [], "hypotheses": []}))
    assert any("too short" in p for p in validate_diagnosis({**DIAG, "hypotheses": ["try gbm"]}))
    assert validate_diagnosis("free text") == ["diagnosis must be a JSON object"]
    assert validate_capability_request(REQ) == []
    assert any("kind must be" in p for p in validate_capability_request({**REQ, "kind": "magic"}))
    assert any("url must" in p for p in validate_capability_request({**REQ, "url": "ftp://x"}))
    assert any("as_of" in p for p in validate_capability_request({**REQ, "as_of": "yesterday"}))
    assert "missing field how_to_test" in validate_capability_request({k: v for k, v in REQ.items() if k != "how_to_test"})


def test_requests_are_served_redirected_queued_and_repeats_counted(lab, monkeypatch):  # noqa: F811
    lab.cfg = lab.cfg.with_(enable_exa=True)
    monkeypatch.setattr(lab, "fetch_page_as_of", lambda url, as_of, max_chars=60_000: {
        "ok": True, "url": url, "as_of": as_of, "captured_at": "2026-08-30T12:00:00Z",
        "snapshot_url": f"https://web.archive.org/web/20260830120000id_/{url}", "sha256": "ab" * 32, "n_chars": 12})
    out = lab.record_capability_requests(
        [REQ, {**REQ, "kind": "budget", "request": "more dry runs per episode please"},
         {**REQ, "kind": "tool", "request": "a residual table by series x origin x horizon", "url": None, "as_of": None},
         {"kind": "tool"}], proposed_by="improver_v000", generation=1)
    assert [r["status"] for r in out] == ["self_served", "redirected", "queued_for_maintainer", "invalid"]
    assert out[0]["captured_at"] == "2026-08-30T12:00:00Z" and out[0]["request_id"] == "cr001"
    assert lab.external_pages() == [{"url": REQ["url"], "as_of": "2026-09-01", "captured_at": "2026-08-30T12:00:00Z",
                                     "requested_by": "improver_v000"}]
    again = lab.record_capability_requests([REQ], proposed_by="improver_v000", generation=2)[0]
    assert again["repeat_of"] == "cr001" and again["times_requested"] == 2
    late = lab.record_capability_requests([{**REQ, "as_of": "2026-10-07"}], "improver_v000", 2)[0]   # Holdout-era date
    assert late["status"] == "queued_for_maintainer" and "Holdout isolation" in late["note"]
    assert [r["status"] for r in lab.ledger_entries() if r["status"] == "CAPABILITY_REQUEST"].count("CAPABILITY_REQUEST") == 6
    digest = lab.memory_digest()["procedural_memory"]
    assert digest["capability_requests"][-1]["status"] == "queued_for_maintainer"
    assert digest["external_pages_available"][0]["url"] == REQ["url"]


class IntrospectiveDriver(FakeDriver):
    """Scripted scientist that submits a diagnosis, and an improver whose revision carries a capability request."""

    def run_diagnosis(self, ctrl, version, generation):
        tb = AgentToolbox(ctrl, mode="diagnosis", episode_id=f"d{generation:03d}", procedure_version=version["version"])
        assert tb.max_dry_runs == 0 and tb.max_analyses == 4
        tb.trace.append({"tool": "profile_pack", "status": "PROFILE_OK", "args": {}})
        assert json.loads(tb.submit_diagnosis(**{**DIAG, "hypotheses": ["x"]}))["status"] == "DIAGNOSIS_REJECTED"
        assert json.loads(tb.submit_diagnosis(**DIAG))["status"] == "DIAGNOSIS_RECORDED"
        return tb

    def propose_revision(self, ctrl, active, digest, known):
        assert digest["procedural_memory"]["diagnoses"][-1]["data_needed"] == DIAG["data_needed"]
        ctrl.record_capability_requests([{**REQ, "kind": "tool", "url": None, "as_of": None}],
                                        proposed_by=f"improver_{active['version']}", generation=ctrl.next_generation())
        return super().propose_revision(ctrl, active, digest, known)


def test_generation_opens_with_a_diagnosis_and_records_the_models_requests(lab):  # noqa: F811
    drv = IntrospectiveDriver([lambda a: _rev(a)])
    row = run_generation(lab, drv, pairs=1)
    st = [r["status"] for r in lab.ledger_entries()]
    assert st.index("DIAGNOSIS") < st.index("CAPABILITY_REQUEST") < st.index("REVISION_PROPOSED")
    d = lab.diagnoses()
    assert len(d) == 1 and d[0]["diagnosis_id"] == "d001" and d[0]["tool_counts"] == {"profile_pack": 1}
    assert d[0]["diagnosis"]["limiting_factors"] == DIAG["limiting_factors"]
    assert row["active_after"] == "v001"
    md = (lab.procedures.root / "lineage.md").read_text()
    assert "## Scientist diagnoses" in md and "AWS spot carries 97%" in md
    assert "## Capability requests" in md and "| cr001 | 1 | tool |" in md and "queued_for_maintainer" in md
    assert "d001" not in [e["episode_id"] for e in lab.released_episodes()]   # a diagnosis is not an episode


def test_episode_lock_keeps_the_bottleneck_statement(lab):  # noqa: F811
    tb = AgentToolbox(lab, mode="episode", episode_id="g005_frozen", procedure_version="v000")
    tb.write_candidate(CAND)
    out = json.loads(tb.commit_forecast("wider bands", **CARD, bottleneck="only 11 Discovery origins: h7 deltas are noise"))
    assert out["status"] == "LOCKED"
    assert json.loads((lab.locks_dir / "g005_frozen.json").read_text())["bottleneck"].startswith("only 11")


def test_wayback_fetch_is_cache_only_offline_and_pinned_to_origin(tmp_path):
    cfg = LabConfig(root=tmp_path, enable_exa=True, exa_cache_only=True)
    (tmp_path / "outputs").mkdir()
    cfg.wayback_cache_path.write_text(json.dumps({
        "key": __import__("lab.sandbox", fromlist=["wayback_cache_key"]).wayback_cache_key("https://x.test/p", "2026-09-01"),
        "retrieved_at_utc": "2026-11-15T00:00:00+00:00", "ok": True, "url": "https://x.test/p", "as_of": "2026-09-01",
        "captured_at": "2026-08-01T00:00:00Z", "snapshot_url": "https://web.archive.org/web/20260801000000id_/https://x.test/p",
        "sha256": "00" * 32, "n_chars": 5, "text": "H100 $3.9/hr"}) + "\n")
    sb = PointInTimeSandbox(cfg)
    hit = sb.fetch_page_as_of("https://x.test/p", "2026-09-01")
    assert hit["ok"] and hit["text"] == "H100 $3.9/hr" and "retrieved_at_utc" not in hit
    miss = sb.fetch_page_as_of("https://x.test/p", "2026-07-01")      # a different origin is a different snapshot
    assert miss["ok"] is False and "cache-only" in miss["error"]
    page = ExaSignals(sb, "2026-09-01").page("https://x.test/p", max_chars=4)
    assert page["text"] == "H100"
    try:
        PointInTimeSandbox(LabConfig(root=tmp_path)).fetch_page_as_of("https://x.test/p", "2026-09-01")
    except PermissionError:
        pass
    else:
        raise AssertionError("PACK_ONLY must not reach external pages")


def test_html_to_text_strips_markup():
    html = "<html><head><style>p{}</style><script>x=1</script></head><body><h1>Pricing</h1><p>p5.48xlarge &amp; " \
           "<b>$98.32</b>/hr</p><!-- c --><table><tr><td>a</td></tr></table></body></html>"
    assert html_to_text(html) == "Pricing\np5.48xlarge & $98.32 /hr\na"


def test_leakage_screen_blocks_evaluation_window_literals_at_lock_time(lab):  # noqa: F811
    from lab.rsi_controller import evaluate_gates
    tb = AgentToolbox(lab, mode="episode", episode_id="g006_frozen", procedure_version="v000")
    tb.write_candidate(CAND.replace("class Forecaster:", 'SURGE = "2026-09-30"\nclass Forecaster:'))
    out = json.loads(tb.commit_forecast("date-keyed surge", **CARD))
    assert out["status"] == "COMMIT_REJECTED" and "benchmark-specific" in out["error"] and "2026-09-30" in out["error"]
    tb.write_candidate(CAND.replace("class Forecaster:", 'GAP_START = "2026-03-01"\nclass Forecaster:'))
    assert json.loads(tb.commit_forecast("gap-aware", **CARD))["status"] == "LOCKED"      # pre-window date is fine
    h = {"pooled_skill": 0.33, "median_skill": 0.30, "coverage_80": 0.80, "skill_by_horizon": {"1": 0.4, "3": 0.3, "7": 0.3}}
    ch = {**h, "pooled_skill": 0.30}
    g = evaluate_gates(h, ch, lab.cfg, None, {"candidate": 12.0, "champion": 2.0})     # 6x for +0.03: allowed 2 + 3 = 5x
    assert g["gate5_cost"]["pass"] is False and g["all_pass"] is False
    g = evaluate_gates(h, ch, lab.cfg, None, {"candidate": 9.0, "champion": 2.0})
    assert g["gate5_cost"]["pass"] is True
    assert evaluate_gates(h, ch, lab.cfg, None, {"candidate": 9.0, "champion": None})["gate5_cost"]["pass"] is True


def test_maintainer_resolution_updates_latest_request_status(lab):  # noqa: F811
    lab.record_capability_requests([{**REQ, "kind": "tool", "url": None, "as_of": None}], "improver_v000", 1)
    rec = lab.resolve_capability_request("cr001", "served", "per-series breach rates added to dry-run diagnostics")
    assert rec["status"] == "served" and rec["resolution_of"] == "cr001"
    assert [r["status"] for r in lab.memory_digest()["procedural_memory"]["capability_requests"]] == ["served"]
    lab.write_procedure_report()
    md = (lab.procedures.root / "lineage.md").read_text()
    assert md.count("| cr001 |") == 1 and "| served |" in md
