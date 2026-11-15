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
STAGE_LONG_HORIZON = "LONG_HORIZON"

# Long-horizon track (lab research track, not the official 1/3/7-day score): AWS EC2 spot series only (the one source
# with multi-year history), monthly origins, 30/90/180-day horizons. Every Discovery target precedes the first Holdout
# origin, and every Holdout target precedes the 2026-03..06 hole in the AWS archive.
LONG_DISCOVERY_ORIGINS: tuple[str, ...] = tuple(f"{y}-{m:02d}-01" for y, m in
                                               [(2023, m) for m in range(6, 13)] + [(2024, m) for m in range(1, 10)])
LONG_HOLDOUT_ORIGINS: tuple[str, ...] = tuple(f"2025-{m:02d}-01" for m in range(3, 9))
LONG_HORIZONS: tuple[int, ...] = (30, 90, 180)
LONG_SERIES_PREFIX = "aws_spot."


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
    # Gate 5 (complexity-aware acceptance, RRSI-style): a promoted forecaster may cost more compute per origin than
    # the champion only in proportion to its measured Holdout gain: cost_ratio - 1 <= cost_beta0 + cost_beta1 * gain.
    cost_beta0: float = 1.0              # up to 2x the champion's seconds/origin for free ...
    cost_beta1: float = 100.0            # ... plus 1x more per +0.01 pooled skill
    # Pre-Holdout screen (two-stage evaluation; lab-side only, scoring is unchanged):
    sanity_max_abs_log_ratio: float = math.log(10.0)  # any quantile beyond 10x / 0.1x the last value is rejected
    sanity_max_invalid_share: float = 0.05            # crossing / non-positive / missing cells (baseline-filled)
    discovery_min_gain: float = 0.0                   # Discovery pooled skill must beat the champion's by more
    # Opt-in noise-aware gate 1 (None = the fixed pooled_min_gain rule). When set, and the paired per-origin Holdout
    # losses of candidate and champion are available with at least pooled_gain_min_blocks independent origin blocks,
    # the required Holdout gain is pooled_gain_z paired standard errors instead of the fixed pooled_min_gain: small
    # consistent gains pass, large noisy ones do not.
    pooled_gain_z: float | None = None
    pooled_gain_min_blocks: int = 4


@dataclass(frozen=True)
class LabConfig:
    root: Path = ROOT
    enable_exa: bool = False
    exa_cache_only: bool = False          # True: never call the network, read outputs/exa_cache.jsonl only
    cache_dir: Path | None = None         # derived series cache (defaults to <root>/outputs/cache)
    eval_timeout_s: int = 3600
    max_discovery_probes: int = 4         # Discovery-only probes the agent may run per iteration
    gates: GateThresholds = field(default_factory=GateThresholds)
    discovery_origins: tuple[str, ...] = DISCOVERY_ORIGINS
    holdout_origins: tuple[str, ...] = HOLDOUT_ORIGINS
    horizons: tuple[int, ...] = HORIZONS
    track: str = "short"                  # "short": official 1/3/7-day track; "long": 30/90/180-day AWS track
    benchmark_origin: str = BENCHMARK_ORIGIN

    @property
    def stage(self) -> str:
        if self.track == "long":
            return STAGE_LONG_HORIZON
        return STAGE_PACK_PLUS_EXA if self.enable_exa else STAGE_PACK_ONLY

    @property
    def is_long(self) -> bool:
        return self.track == "long"

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

    @property
    def wayback_cache_path(self) -> Path:
        return self.outputs_dir / "wayback_cache.jsonl"

    def with_(self, **kw) -> "LabConfig":
        return replace(self, **kw)


def _flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def load_config(**overrides) -> LabConfig:
    """Config from the environment (LAB_ENABLE_EXA, LAB_EXA_CACHE_ONLY, LAB_ROOT, LAB_CACHE_DIR, LAB_POOLED_GAIN_Z) plus overrides."""
    kw = {"enable_exa": _flag("LAB_ENABLE_EXA"), "exa_cache_only": _flag("LAB_EXA_CACHE_ONLY")}
    if os.environ.get("LAB_ROOT"):
        kw["root"] = Path(os.environ["LAB_ROOT"]).resolve()
    if os.environ.get("LAB_CACHE_DIR"):
        kw["cache_dir"] = Path(os.environ["LAB_CACHE_DIR"]).resolve()
    track = overrides.pop("track", None) or os.environ.get("LAB_TRACK", "short")
    if track == "long":
        kw.update(track="long", discovery_origins=LONG_DISCOVERY_ORIGINS, holdout_origins=LONG_HOLDOUT_ORIGINS,
                  horizons=LONG_HORIZONS, benchmark_origin=LONG_HOLDOUT_ORIGINS[-1],
                  gates=GateThresholds(pooled_floor=0.0))
    kw.update(overrides)
    cfg = LabConfig(**kw)
    if os.environ.get("LAB_POOLED_GAIN_Z"):
        cfg = cfg.with_(gates=replace(cfg.gates, pooled_gain_z=float(os.environ["LAB_POOLED_GAIN_Z"])))
    check_partition(cfg)
    return cfg


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def check_partition(cfg: LabConfig | None = None) -> None:
    """Anti-leakage invariants of the temporal split (raises AssertionError)."""
    cfg = cfg or LabConfig()
    disc, hold = [_d(x) for x in cfg.discovery_origins], [_d(x) for x in cfg.holdout_origins]
    assert disc == sorted(disc) and hold == sorted(hold)
    last_disc_target = max(disc) + dt.timedelta(days=max(cfg.horizons))
    assert last_disc_target <= min(hold), "discovery targets overlap the holdout window"
    assert cfg.benchmark_origin in cfg.holdout_origins
    assert max(hold) < _d(FINAL_ORIGIN)
    if cfg.track == "long":
        assert max(hold) + dt.timedelta(days=max(cfg.horizons)) < _d("2026-03-01"), "holdout targets hit the AWS hole"


check_partition()
