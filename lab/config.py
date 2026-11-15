"""Temporal partition, promotion gates and stage flags of the lab (immutable to the agent)."""
from __future__ import annotations

import datetime as dt
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK_NAME = "gpu-price-pack-v0.1.1-handover"

DISCOVERY_ORIGINS: tuple[str, ...] = (
    "2026-07-15", "2026-07-22", "2026-07-29", "2026-08-05", "2026-08-12", "2026-08-19",
    "2026-08-26", "2026-09-02", "2026-09-09", "2026-09-16", "2026-09-23",
)
HOLDOUT_ORIGINS: tuple[str, ...] = ("2026-09-30", "2026-10-07", "2026-10-14", "2026-10-21", "2026-10-24")
BENCHMARK_ORIGIN = "2026-10-24"  # the starter kit's demo cutoff (drift +0.20 pooled)
FINAL_ORIGIN = "2026-11-06"      # the pack's data cutoff; the frozen submission origin
HORIZONS: tuple[int, ...] = (1, 3, 7)
CATALOG_AS_OF = "2026-10-24"     # scored set = headline tier of the starter catalog at this as_of (97 series)
SCORED_TIER = "headline"

STAGE_PACK_ONLY = "PACK_ONLY"
STAGE_PACK_PLUS_EXA = "PACK_PLUS_EXA"


@dataclass(frozen=True)
class GateThresholds:
    pooled_floor: float = 0.200          # Gate 1: beat the starter's drift demo skill ...
    pooled_min_gain: float = 0.003       # ... and the champion by at least this much
    median_tolerance: float = 0.005      # Gate 2: median-of-series skill may not drop more than this
    coverage_low: float = 0.74           # Gate 3: empirical [q10, q90] coverage band ...
    coverage_high: float = 0.86
    coverage_target: float = 0.80        # ... or no worse calibrated than the champion (|cov - target| within
    coverage_tolerance: float = 0.005    #     champion's distance + tolerance; ~7 of 1416 Holdout cells) ...
    coverage_hard_low: float = 0.70      # ... but never outside this hard band (bounds any ratchet)
    coverage_hard_high: float = 0.90
    horizon_floor: float = -0.020        # Gate 4: every horizon's pooled skill
    # Pre-Holdout screen (two-stage evaluation; lab-side only, scoring is unchanged):
    sanity_max_abs_log_ratio: float = math.log(10.0)  # any quantile beyond 10x / 0.1x the last value is rejected
    sanity_max_invalid_share: float = 0.05            # crossing / non-positive / missing cells (baseline-filled)
    discovery_min_gain: float = 0.0                   # Discovery pooled skill must beat the champion's by more


@dataclass(frozen=True)
class LabConfig:
    root: Path = ROOT
    enable_exa: bool = False
    exa_cache_only: bool = False          # True: never call the network, read outputs/exa_cache.jsonl only
    cache_dir: Path | None = None         # derived series cache (defaults to <root>/outputs/cache)
    eval_timeout_s: int = 900
    max_discovery_probes: int = 4         # Discovery-only probes the agent may run per iteration
    gates: GateThresholds = field(default_factory=GateThresholds)
    discovery_origins: tuple[str, ...] = DISCOVERY_ORIGINS
    holdout_origins: tuple[str, ...] = HOLDOUT_ORIGINS
    horizons: tuple[int, ...] = HORIZONS

    @property
    def stage(self) -> str:
        return STAGE_PACK_PLUS_EXA if self.enable_exa else STAGE_PACK_ONLY

    @property
    def pack_dir(self) -> Path:
        return self.root / PACK_NAME

    @property
    def starter_dir(self) -> Path:
        return self.pack_dir / "starter"

    @property
    def workspace_dir(self) -> Path:
        return self.root / "workspace"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def series_cache_dir(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir else self.outputs_dir / "cache"

    @property
    def forecaster_path(self) -> Path:
        return self.workspace_dir / "forecaster.py"

    @property
    def prompt_path(self) -> Path:
        return self.workspace_dir / "scientist_prompt.md"

    @property
    def theory_path(self) -> Path:
        return self.workspace_dir / "theory_of_market.md"

    @property
    def ledger_path(self) -> Path:
        return self.outputs_dir / "experiment_ledger.jsonl"

    @property
    def exa_cache_path(self) -> Path:
        return self.outputs_dir / "exa_cache.jsonl"

    def with_(self, **kw) -> "LabConfig":
        return replace(self, **kw)


def _flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def load_config(**overrides) -> LabConfig:
    """Config from the environment (LAB_ENABLE_EXA, LAB_EXA_CACHE_ONLY, LAB_ROOT, LAB_CACHE_DIR) plus overrides."""
    kw = {"enable_exa": _flag("LAB_ENABLE_EXA"), "exa_cache_only": _flag("LAB_EXA_CACHE_ONLY")}
    if os.environ.get("LAB_ROOT"):
        kw["root"] = Path(os.environ["LAB_ROOT"]).resolve()
    if os.environ.get("LAB_CACHE_DIR"):
        kw["cache_dir"] = Path(os.environ["LAB_CACHE_DIR"]).resolve()
    kw.update(overrides)
    return LabConfig(**kw)


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def check_partition(cfg: LabConfig | None = None) -> None:
    """Anti-leakage invariants of the temporal split (raises AssertionError)."""
    cfg = cfg or LabConfig()
    disc, hold = [_d(x) for x in cfg.discovery_origins], [_d(x) for x in cfg.holdout_origins]
    assert disc == sorted(disc) and hold == sorted(hold)
    last_disc_target = max(disc) + dt.timedelta(days=max(cfg.horizons))
    assert last_disc_target <= min(hold), "discovery targets overlap the holdout window"
    assert BENCHMARK_ORIGIN in cfg.holdout_origins
    assert max(hold) < _d(FINAL_ORIGIN)


check_partition()
