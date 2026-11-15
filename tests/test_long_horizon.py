"""Long-horizon research track: 30/90/180-day AWS spot horizons on 2023-2025 monthly origins, leak-free partition,
track propagation to evaluator children, Exa on by default, and a required qualitative market insight per lock."""
import datetime as dt
import json

from lab.config import LONG_HORIZONS, STAGE_LONG_HORIZON, LabConfig, check_partition, load_config
from lab.rsi_controller import AgentToolbox, stage_config
from tests.test_driver_loop import CAND
from tests.test_procedure import CARD, lab  # noqa: F401

INSIGHT = ("Drivers: H100 supply loosens after Blackwell ramps (dated capacity announcements), so p5 spot drifts "
           "down over 90-180 days while older p4d stays flat near its floor. Evidence: dated Exa results on AWS price "
           "cuts. A renewed demand shock or a capacity freeze would invalidate this view.")


def test_long_config_partition_and_stage(monkeypatch):
    cfg = load_config(track="long")
    assert cfg.stage == STAGE_LONG_HORIZON and cfg.is_long and cfg.horizons == LONG_HORIZONS == (30, 90, 180)
    d = [dt.date.fromisoformat(x) for x in cfg.discovery_origins]
    h = [dt.date.fromisoformat(x) for x in cfg.holdout_origins]
    assert max(d) + dt.timedelta(days=180) <= min(h)
    assert max(h) + dt.timedelta(days=180) < dt.date(2026, 3, 1)
    assert cfg.benchmark_origin == cfg.holdout_origins[-1] and cfg.gates.pooled_floor == 0.0
    monkeypatch.setenv("LAB_TRACK", "long")
    assert load_config().stage == STAGE_LONG_HORIZON
    monkeypatch.setenv("LAB_TRACK", "short")
    assert load_config().horizons == (1, 3, 7) and not load_config().is_long
    assert stage_config("long_horizon").enable_exa and stage_config("long_horizon").is_long


def test_long_partition_rejects_overlapping_origins():
    bad = load_config(track="long").__class__(**{**load_config(track="long").__dict__,
                                                 "discovery_origins": ("2024-12-01",)})
    try:
        check_partition(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("discovery targets overlapping holdout must be rejected")
    assert LabConfig().track == "short"


def test_long_commit_requires_market_insight(lab, monkeypatch):  # noqa: F811
    object.__setattr__(lab.cfg, "track", "long")
    tb = AgentToolbox(lab, mode="episode", episode_id="g001_frozen", procedure_version="v000")
    tb.write_candidate(CAND)
    out = json.loads(tb.commit_forecast("regime shift", **CARD))
    assert out["status"] == "COMMIT_REJECTED" and "market_insight" in out["error"]
    out = json.loads(tb.commit_forecast("regime shift", **CARD, market_insight=INSIGHT))
    assert out["status"] == "LOCKED"
    lock = json.loads((lab.locks_dir / "g001_frozen.json").read_text())
    assert lock["market_insight"] == INSIGHT


def test_commit_of_crashing_candidate_writes_no_lock(lab, monkeypatch):  # noqa: F811
    tb = AgentToolbox(lab, mode="episode", episode_id="g009_frozen", procedure_version="v000")
    tb.write_candidate(CAND)
    real = lab._run_lab_child
    monkeypatch.setattr(lab, "_run_lab_child", lambda a, t: {"ok": False, "error": "TypeError: boom"})
    out = json.loads(tb.commit_forecast("crash", **CARD))
    assert out["status"] == "COMMIT_REJECTED" and "boom" in out["error"]
    assert not (lab.locks_dir / "g009_frozen.json").exists() and tb.lock is None
    monkeypatch.setattr(lab, "_run_lab_child", real)
    assert json.loads(tb.commit_forecast("fixed", **CARD))["status"] == "LOCKED"
