"""Dual-layer RSI orchestrator: Devin API / Strands drivers, 4-gate promotion and a git commit/revert state machine.

The agent owns workspace/ (forecaster.py, scientist_prompt.md, theory_of_market.md) and talks to the lab only
through its tools: inspect_current_state, inspect_candidate, propose_and_test_forecaster, update_scientist_prompt and,
in PACK_PLUS_EXA only, search_exa_signals. Cross-iteration memory (candidate archive under outputs/candidates/,
experiment registry, exploration directive, protected playbook block, playbook versions under outputs/playbooks/)
lives in lab/memory.py. Candidates are audited (no I/O, no network), evaluated in a separate
process with a timeout, scored with full diagnostics on Discovery and only the four gate aggregates on Holdout,
then PROMOTED (champion overwritten, workspace committed) or REVERTED / RUNTIME_ERROR (forecaster.py restored
from the champion backup; the playbook and the notebook keep the agent's notes so negative results persist).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import functools
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from lab.config import (
    DISCOVERY_ORIGINS,
    HOLDOUT_ORIGINS,
    STAGE_LONG_HORIZON,
    STAGE_PACK_ONLY,
    STAGE_PACK_PLUS_EXA,
    LabConfig,
    load_config,
)
from lab.memory import (
    HYPOTHESIS_FAMILIES,
    MODEL_CLASSES,
    NOVELTY_LEVELS,
    PROTECTED_BLOCK,
    archive_candidate,
    champion_model_class,
    collapse_warnings,
    demote_headings,
    diff_stats,
    dropped_guardrails,
    enforce_protected,
    exploration_directive,
    family_stats,
    fit_json,
    gate_margins,
    infer_family,
    is_no_op,
    metric_delta,
    model_class_exploration_required,
    model_class_stats,
    near_misses,
    normalize_family,
    normalize_model_class,
    normalize_novelty,
    notebook_index,
    registry_row,
    unified_diff,
    verified_novelty,
)
from lab.procedure import (
    ADMISSION_PAIRS,
    CONTROL_BOUNDS,
    CONTROL_TOTAL,
    MAX_COMPONENT_CHARS,
    MIN_COMPONENT_CHARS,
    procedure_controls,
    HYPOTHESIS_CARD_FIELDS,
    ProcedureRegistry,
    admission_rule,
    append_jsonl,
    episode_score,
    read_jsonl,
    validate_card,
    validate_revision,
    write_lock,
)
from lab.sandbox import audit_forecaster_source, import_policy_text
from lab.workbench import CandidateWorkbench, audit_analysis_source, clip_json

WORKSPACE_FILES = ("workspace/forecaster.py", "workspace/scientist_prompt.md", "workspace/theory_of_market.md")
EXPERIMENT_STATUSES = ("PROMOTED", "REVERTED", "RUNTIME_ERROR", "REJECTED_FIREWALL", "REJECTED_SANITY",
                       "REJECTED_DISCOVERY", "REJECTED_DUPLICATE")
SCREEN_STATUSES = ("REVERTED", "REJECTED_SANITY", "REJECTED_DISCOVERY")  # deterministic verdicts: never re-test
DRY_RUN_STATUSES = ("DRY_RUN_OK", "DRY_RUN_ERROR", "DRY_RUN_REJECTED")  # Discovery-only; never counted as experiments
ANALYSIS_TIMEOUT_S = 300
GIT_IDENTITY = ["-c", "user.name=AI Scientist Lab", "-c", "user.email=ai-scientist-lab@localhost"]
MAX_PLAYBOOK_CHARS = 24_000
MAX_THEORY_UPDATE_CHARS = 6_000
MAX_STATE_CHARS = 60_000
STATE_SHRINK_ORDER = [("theory_of_market_tail_md", "tail"), ("scientist_prompt_md", "head"), ("ledger_tail", "drop_old"),
                      ("champion_discovery_diagnostics", "head"), ("near_misses", "drop_last"),
                      ("experiment_registry", "drop_old"), ("champion_forecaster_py", "head")]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _num(x) -> float | None:
    """A finite float, or None (missing, NaN, +-inf, or a non-number)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def gate_descriptions(cfg: LabConfig) -> dict:
    g = cfg.gates
    g1 = f"holdout pooled > max({g.pooled_floor:.3f}, champion + {g.pooled_min_gain:.3f})"
    if g.pooled_gain_z is not None:
        g1 = (f"holdout pooled > max({g.pooled_floor:.3f}, champion + {g.pooled_gain_z:g} paired standard errors of the "
              f"Holdout gain over origins); falls back to champion + {g.pooled_min_gain:.3f} when the error is unknown")
    return {"gate1": g1,
            "gate2": f"holdout median >= champion median - {g.median_tolerance:.3f}",
            "gate3": f"{g.coverage_hard_low:.2f} <= holdout coverage_80 <= {g.coverage_hard_high:.2f} and either "
                     f"{g.coverage_low:.2f} <= coverage_80 <= {g.coverage_high:.2f} or |coverage_80 - {g.coverage_target:.2f}| "
                     f"<= champion's distance + {g.coverage_tolerance:.3f}",
            "gate4": f"every horizon {tuple(cfg.horizons)} skill >= {g.horizon_floor:.3f}",
            "screen": f"before Holdout: finite quantiles within {math.exp(g.sanity_max_abs_log_ratio):.0f}x of the last value, "
                      f"<= {g.sanity_max_invalid_share:.0%} invalid/missing cells, and Discovery pooled > champion "
                      f"Discovery pooled + {g.discovery_min_gain:.4f}"}


def paired_origin_stats(cand_by_origin: dict | None, ref_by_origin: dict | None, cfg: LabConfig) -> dict | None:
    """Paired skill delta, standard error and z of candidate vs reference from their per-origin loss sums
    (Evaluator loss_by_origin), on the origins both have. None when either side is missing or the cells differ."""
    from lab.evaluator import overlap_block, paired_delta_stats

    cand_by_origin, ref_by_origin = cand_by_origin or {}, ref_by_origin or {}
    origins = sorted(o for o in cand_by_origin if o in ref_by_origin)
    if len(origins) < 2:
        return None
    c, r = [cand_by_origin[o] for o in origins], [ref_by_origin[o] for o in origins]
    base = [_num(x.get("baseline_sum")) for x in c]
    if any(b is None or _num(y.get("baseline_sum")) is None or abs(b - _num(y["baseline_sum"])) > 1e-9 * max(1.0, abs(b))
           for b, y in zip(base, r)):
        return None                                      # not the same scored cells: the comparison is not paired
    st = paired_delta_stats([x.get("loss_sum") for x in c], [y.get("loss_sum") for y in r], base,
                            overlap_block(origins, max(cfg.horizons)))
    return {k: (_r(v, 6) if isinstance(v, float) else v) for k, v in st.items()}


def evaluate_gates(holdout: dict, champion_holdout: dict | None, cfg: LabConfig, paired: dict | None = None) -> dict:
    """The four promotion gates, evaluated strictly on Holdout aggregates (missing / non-finite values fail).

    `paired` (paired_origin_stats of candidate vs champion on Holdout) only matters when gates.pooled_gain_z is set:
    the required gain is then pooled_gain_z paired standard errors instead of the fixed pooled_min_gain."""
    g, ch = cfg.gates, champion_holdout or {}
    champ_pooled = _num(ch.get("pooled_skill"))
    champ_median = _num(ch.get("median_skill"))
    champ_cov = _num(ch.get("coverage_80"))
    min_gain, rule = g.pooled_min_gain, "fixed"
    se = _num((paired or {}).get("delta_se"))
    if g.pooled_gain_z is not None and se is not None and (paired.get("n_independent_blocks") or 0) >= g.pooled_gain_min_blocks:
        min_gain, rule = g.pooled_gain_z * se, "paired_z"
    need = max(g.pooled_floor, (champ_pooled if champ_pooled is not None else -1e9) + min_gain)
    med_need = (champ_median if champ_median is not None else -1e9) - g.median_tolerance
    pooled, median, cov = _num(holdout.get("pooled_skill")), _num(holdout.get("median_skill")), _num(holdout.get("coverage_80"))
    by_h = {str(k): _num(v) for k, v in (holdout.get("skill_by_horizon") or {}).items()}
    in_band = cov is not None and g.coverage_low <= cov <= g.coverage_high
    in_hard = cov is not None and g.coverage_hard_low <= cov <= g.coverage_hard_high
    allowed_dist = None if champ_cov is None else abs(champ_cov - g.coverage_target) + g.coverage_tolerance
    rel_ok = cov is not None and allowed_dist is not None and abs(cov - g.coverage_target) <= allowed_dist + 1e-12
    out = {
        "gate1_pooled_gain": {"value": _r(pooled), "threshold": _r(need), "pass": pooled is not None and pooled > need,
                              "rule": rule, "required_gain": _r(min_gain, 6)},
        "gate2_breadth": {"value": _r(median), "threshold": _r(med_need), "pass": median is not None and median >= med_need},
        "gate3_calibration": {"value": _r(cov), "band": [g.coverage_low, g.coverage_high],
                              "hard_band": [g.coverage_hard_low, g.coverage_hard_high],
                              "target": g.coverage_target, "max_distance_to_target": _r(allowed_dist), "in_band": in_band,
                              "no_worse_than_champion": rel_ok, "pass": in_hard and (in_band or rel_ok)},
        "gate4_horizon_floor": {"value": {h: _r(v) for h, v in by_h.items()}, "threshold": g.horizon_floor,
                                "pass": all(by_h.get(str(h)) is not None and by_h[str(h)] >= g.horizon_floor
                                            for h in cfg.horizons)},
    }
    out["all_pass"] = all(v["pass"] for v in out.values())
    return out


def screen_candidate(discovery: dict, champion_discovery: dict | None, cfg: LabConfig) -> dict:
    """Pre-Holdout screen: output sanity, invalid/missing share and a Discovery improvement over the champion."""
    g = cfg.gates
    san = discovery.get("sanity") or {}
    sc = discovery.get("scoring") or {}
    n_cells = max(int(sc.get("n_cells") or 0), 1)
    n_rows = max(int(san.get("n_rows") or 0), 1)
    n_bad_rows = san.get("n_invalid_rows")
    if n_bad_rows is None:
        n_bad_rows = max(int(san.get("n_crossing_rows") or 0), int(san.get("n_nonpositive_rows") or 0))
    invalid = max(int(sc.get("n_cells_filled_from_baseline") or 0) / n_cells, int(n_bad_rows) / n_rows)
    cand, champ = _num(discovery.get("pooled_skill")), _num((champion_discovery or {}).get("pooled_skill"))
    need = (champ if champ is not None else -1e9) + g.discovery_min_gain
    checks = {
        "sanity": {"pass": bool(san.get("ok", False)),
                   **{k: san.get(k) for k in ("n_nonfinite_rows", "n_extreme_rows", "max_abs_log_ratio_vs_last",
                                              "limit_abs_log_ratio")}},
        "invalid_share": {"value": round(invalid, 4), "threshold": g.sanity_max_invalid_share,
                          "pass": invalid <= g.sanity_max_invalid_share},
        "discovery_gain": {"value": _r(cand, 6), "champion": _r(champ, 6), "threshold": _r(need, 6),
                           "pass": cand is not None and cand > need},
    }
    failed = [k for k, v in checks.items() if not v["pass"]]
    status = None
    if failed:
        status = "REJECTED_DISCOVERY" if failed == ["discovery_gain"] else "REJECTED_SANITY"
    return {"pass": not failed, "failed": failed, "status": status, "checks": checks}


def origin_delta_noise(by_origin_delta: dict) -> dict | None:
    """Mean, standard error and z of the per-origin skill deltas (unweighted; adjacent origins treated as independent)."""
    vals = [v for v in by_origin_delta.values() if v is not None]
    if len(vals) < 3:
        return None
    mean = sum(vals) / len(vals)
    se = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
    return {"mean_origin_delta": round(mean, 6), "se": round(se, 6), "z": round(mean / se, 2) if se > 0 else None,
            "n_origins": len(vals),
            "note": "|z| < 2: the Discovery difference is within origin-to-origin noise; do not read it as a gain or a loss"}


def compare_discovery(cand: dict, champ: dict | None, cfg: LabConfig | None = None) -> dict:
    """Paired Discovery comparison with the champion (same cells, same baseline): where the candidate wins or loses."""
    champ = champ or {}

    def delta(a, b):
        a, b = _num(a), _num(b)
        return None if a is None or b is None else round(a - b, 6)

    by_o = {o: delta(v, (champ.get("skill_by_origin") or {}).get(o)) for o, v in (cand.get("skill_by_origin") or {}).items()}
    vals = [v for v in by_o.values() if v is not None]
    key = lambda r: (r.get("instrument_class"), r.get("pricing_tier"), r.get("horizon_days"))  # noqa: E731
    champ_sl = {key(r): r for r in champ.get("slices") or []}
    sl = []
    for r in cand.get("slices") or []:
        c = champ_sl.get(key(r))
        if c:
            sl.append({"instrument_class": r.get("instrument_class"), "pricing_tier": r.get("pricing_tier"),
                       "horizon_days": r.get("horizon_days"), "skill_delta": delta(r.get("skill"), c.get("skill")),
                       "coverage_delta": delta(r.get("coverage_80"), c.get("coverage_80")),
                       "excess_loss_sum_delta": delta(r.get("excess_loss_sum"), c.get("excess_loss_sum"))})
    sl.sort(key=lambda r: r["excess_loss_sum_delta"] if r["excess_loss_sum_delta"] is not None else 0.0)
    return {
        "pooled_skill_delta": delta(cand.get("pooled_skill"), champ.get("pooled_skill")),
        "median_skill_delta": delta(cand.get("median_skill"), champ.get("median_skill")),
        "coverage_80_delta": delta(cand.get("coverage_80"), champ.get("coverage_80")),
        "skill_by_horizon_delta": {h: delta(v, (champ.get("skill_by_horizon") or {}).get(h))
                                   for h, v in (cand.get("skill_by_horizon") or {}).items()},
        "skill_by_origin_delta": by_o,
        "origin_delta_noise": origin_delta_noise(by_o),
        "paired_pooled_delta": paired_origin_stats(cand.get("loss_by_origin"), champ.get("loss_by_origin"), cfg or LabConfig()),
        "n_origins_improved": sum(v > 0 for v in vals), "n_origins_worse": sum(v < 0 for v in vals),
        "slice_deltas_best_first": sl,
    }


class RSIController:
    def __init__(self, cfg: LabConfig | None = None, driver_name: str = "manual", use_git: bool = True):
        self.cfg = cfg or load_config()
        self.root = Path(self.cfg.root)
        self.driver_name = driver_name
        self.use_git = use_git and (self.root / ".git").exists()
        self.champ_dir = self.cfg.outputs_dir / "champions" / self.cfg.stage
        self.state_path = self.champ_dir / "champion_state.json"
        self.backup_path = self.champ_dir / "forecaster.py"
        self.candidates_dir = self.cfg.outputs_dir / "candidates" / self.cfg.stage
        self.playbooks_dir = self.cfg.outputs_dir / "playbooks" / self.cfg.stage
        self.procedures = ProcedureRegistry(self.cfg.outputs_dir / "procedures" / self.cfg.stage)
        self.locks_dir = self.cfg.outputs_dir / "locks" / self.cfg.stage
        self.variants_dir = self.cfg.outputs_dir / "variants" / self.cfg.stage
        self.memory_dir = self.cfg.outputs_dir / "memory" / self.cfg.stage
        self.iteration_budget: dict | None = None  # {"planned": N, "first_iteration": i} set by the run loop
        self._pending_archive = self._pending_discovery = None
        self._lab_hash = self._hash_lab()

    # --------------------------------------------------------------------------- utilities ----
    def _hash_lab(self) -> str:
        h = hashlib.sha256()
        for p in sorted((self.root / "lab").glob("*.py")):
            h.update(p.name.encode() + p.read_bytes())
        return h.hexdigest()

    def _check_lab_intact(self) -> None:
        if self._hash_lab() != self._lab_hash:
            if self.use_git:
                self._git("checkout", "--", "lab")
            self._ledger({"status": "LAB_TAMPER", "note": "lab/ changed during an agent step; restored from git"})
            raise PermissionError("lab/ is immutable to the agent; the change was reverted")

    def _git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["git", *GIT_IDENTITY, *args], cwd=self.root, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def git_commit(self, message: str, paths=WORKSPACE_FILES) -> str | None:
        if not self.use_git:
            return None
        self._git("add", "--", *paths)
        if not self._git("diff", "--cached", "--name-only", "--", *paths):
            return self._git("rev-parse", "HEAD", check=False) or None
        self._git("commit", "-q", "-m", message, "--", *paths)
        return self._git("rev-parse", "HEAD")

    def _ledger(self, rec: dict) -> dict:
        rec = {"ts_utc": utc_now(), "stage": self.cfg.stage, "driver": self.driver_name, **rec}
        self.cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=False, default=str) + "\n")
        return rec

    def ledger_entries(self, stage: str | None = None) -> list[dict]:
        if not self.cfg.ledger_path.exists():
            return []
        rows = [json.loads(x) for x in self.cfg.ledger_path.read_text().splitlines() if x.strip()]
        return [r for r in rows if stage is None or r.get("stage") == stage]

    def next_iteration(self) -> int:
        return 1 + sum(1 for r in self.ledger_entries(self.cfg.stage) if r.get("status") in EXPERIMENT_STATUSES)

    @property
    def champion(self) -> dict | None:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    # -------------------------------------------------------------------------- evaluation ----
    def run_evaluation(self, forecaster_path: Path, phase: str = "full") -> dict:
        """Evaluate in a child process (timeout; the controller only ever receives the firewalled JSON)."""
        return self._run_lab_child(["lab.evaluator", "--forecaster", str(forecaster_path), "--phase", phase],
                                   self.cfg.eval_timeout_s)

    def _run_lab_child(self, module_args: list[str], timeout_s: int) -> dict:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "result.json"
            cmd = [sys.executable, "-B", "-m", *module_args, "--out", str(out)]
            if self.cfg.enable_exa:
                cmd.append("--enable-exa")
            if self.cfg.exa_cache_only:
                cmd.append("--exa-cache-only")
            env = {**os.environ, "PYTHONPATH": str(self.root), "PYTHONDONTWRITEBYTECODE": "1", "LAB_ROOT": str(self.root),
                   "LAB_TRACK": self.cfg.track}
            if self.cfg.cache_dir:
                env["LAB_CACHE_DIR"] = str(self.cfg.cache_dir)
            try:
                proc = subprocess.run(cmd, cwd=self.root, env=env, capture_output=True, text=True, timeout=timeout_s)
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"evaluation timed out after {timeout_s}s"}
            if not out.exists():
                return {"ok": False, "error": "evaluator produced no result (crashed before reporting)",
                        "traceback_tail": (proc.stderr or "")[-3000:]}
            return json.loads(out.read_text())

    @staticmethod
    def _discovery_summary(d: dict) -> dict:
        keys = ("pooled_skill", "median_skill", "coverage_80", "skill_by_horizon", "q10_breach_rate", "q90_breach_rate",
                "skill_by_origin")
        return {k: d.get(k) for k in keys}

    # ---------------------------------------------------------------------------- bootstrap ----
    def bootstrap(self) -> dict:
        """Make the current workspace forecaster the champion of this stage if none exists yet."""
        if self.champion:
            return self.champion
        self.champ_dir.mkdir(parents=True, exist_ok=True)
        res = self.run_evaluation(self.cfg.forecaster_path)
        if not res.get("ok"):
            raise RuntimeError(f"seed forecaster failed to evaluate: {res.get('error')}")
        code = self.cfg.forecaster_path.read_text()
        shutil.copyfile(self.cfg.forecaster_path, self.backup_path)
        commit = self.git_commit(f"[{self.cfg.stage}] seed champion")
        state = {"stage": self.cfg.stage, "iteration": 0, "code_sha256": sha256_text(code), "git_commit": commit,
                 "discovery": res["discovery"], "holdout": res["holdout"], "holdout_by_origin": res.get("holdout_by_origin"),
                 "benchmark_2026_10_24": res["benchmark_2026_10_24"], "since": utc_now()}
        self.state_path.write_text(json.dumps(state, indent=2))
        self._ledger({"status": "SEED", "iteration": 0, "hypothesis": "seed champion (current workspace forecaster)",
                      "code_sha256": state["code_sha256"], "discovery": self._discovery_summary(res["discovery"]),
                      "holdout": res["holdout"], "benchmark_2026_10_24": res["benchmark_2026_10_24"], "git_commit": commit})
        return state

    # ------------------------------------------------------------------------------- tools ----
    def budget(self) -> dict | None:
        b = self.iteration_budget
        if not b:
            return None
        this = self.next_iteration()
        done = this - b["first_iteration"]
        return {"planned": b["planned"], "this_iteration": done + 1, "remaining": max(0, b["planned"] - done - 1)}

    def experiment_registry(self) -> list[dict]:
        """Every experiment of this stage as one compact row (never truncated, unlike the full-record ledger tail)."""
        rows = []
        for r in self.ledger_entries(self.cfg.stage):
            if r.get("status") in EXPERIMENT_STATUSES + ("SEED", "FORK"):
                row = registry_row(r)
                if r.get("status") in EXPERIMENT_STATUSES:
                    row["archived"] = (self.candidates_dir / f"iter_{int(r['iteration']):03d}" / "forecaster.py").exists()
                rows.append(row)
        return rows

    def _ensure_protected_playbook(self) -> None:
        p = self.cfg.prompt_path
        text, changed = enforce_protected(p.read_text() if p.exists() else "# AI Scientist research playbook\n")
        if changed:
            p.write_text(text)

    def inspect_current_state(self, ledger_tail: int = 4, include_playbook: bool = True) -> dict:
        champ = self.bootstrap()
        playbook = enforce_protected(self.cfg.prompt_path.read_text() if self.cfg.prompt_path.exists() else "")[0]
        ledger = [{k: r.get(k) for k in ("iteration", "status", "hypothesis_family", "model_class", "novelty", "hypothesis", "discovery",
                                          "discovery_delta_vs_champion", "holdout", "screen", "gates", "gate_margins", "no_op",
                                          "candidate_dir", "benchmark_2026_10_24", "error") if r.get(k) is not None}
                  for r in self.ledger_entries(self.cfg.stage) if r.get("status") in EXPERIMENT_STATUSES + ("SEED", "FORK")]
        registry = self.experiment_registry()
        theory = self.cfg.theory_path.read_text() if self.cfg.theory_path.exists() else ""
        state = {
            "stage": self.cfg.stage, "enable_exa": self.cfg.enable_exa, "next_iteration": self.next_iteration(),
            "budget": self.budget(),
            "exploration_directive": exploration_directive(registry, self.budget()),
            "hypothesis_families": HYPOTHESIS_FAMILIES,
            "family_stats": family_stats(registry),
            "model_classes": MODEL_CLASSES,
            "champion_model_class": champion_model_class(registry),
            "model_class_stats": model_class_stats(registry),
            "playbook_lint": {"collapse_warnings": collapse_warnings(playbook)},
            "experiment_registry": registry,
            "near_misses": near_misses(registry),
            "promotion_gates": gate_descriptions(self.cfg),
            "champion_holdout_aggregates": champ["holdout"],
            "champion_discovery_diagnostics": champ["discovery"],
            "champion_forecaster_py": self.backup_path.read_text(),
            "ledger_tail": ledger[-ledger_tail:],
            "notebook_index": notebook_index(theory),
            "theory_of_market_tail_md": theory[-8000:],
        }
        if include_playbook:
            state["scientist_prompt_md"] = playbook
        return state

    def _tested_shas(self) -> set[str]:
        return {r.get("code_sha256") for r in self.ledger_entries(self.cfg.stage) if r.get("status") in SCREEN_STATUSES}

    def render_state(self, max_chars: int = MAX_STATE_CHARS, include_playbook: bool = False) -> str:
        """inspect_current_state as valid JSON within a character budget (low-value fields shrink first)."""
        return fit_json(self.inspect_current_state(include_playbook=include_playbook), max_chars, STATE_SHRINK_ORDER)

    def inspect_candidate(self, iteration: int, max_chars: int = 40_000) -> dict:
        """An archived candidate: code, diff against the CURRENT champion, gates/margins and Discovery diagnostics."""
        d = self.candidates_dir / f"iter_{int(iteration):03d}"
        if not (d / "forecaster.py").exists():
            have = sorted(x.name for x in self.candidates_dir.glob("iter_*")) if self.candidates_dir.exists() else []
            return {"error": f"no archived candidate for iteration {iteration}", "archived": have}
        code = (d / "forecaster.py").read_text()
        res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
        out = {"iteration": iteration, "status": res.get("status"), "hypothesis": res.get("hypothesis"),
               "hypothesis_family": res.get("hypothesis_family"), "gates": res.get("gates"),
               "gate_margins": res.get("gate_margins"), "holdout": res.get("holdout"),
               "holdout_delta_vs_champion": res.get("holdout_delta_vs_champion"),
               "discovery_delta_vs_champion": res.get("discovery_delta_vs_champion"), "error": res.get("error"),
               "diff_vs_current_champion": unified_diff(self.backup_path.read_text(), code),
               "discovery_diagnostics": res.get("discovery_diagnostics"), "forecaster_py": code}
        return json.loads(fit_json(out, max_chars, [("discovery_diagnostics", "head"), ("forecaster_py", "head"),
                                                    ("diff_vs_current_champion", "head")]))

    def propose_and_test_forecaster(self, hypothesis: str, theory_update_md: str, new_forecaster_code: str,
                                    hypothesis_family: str = "", novelty: str = "",
                                    builds_on_iteration: int | None = None, model_class: str = "",
                                    hypothesis_card: dict | None = None) -> dict:
        """Two-stage test: audit -> duplicate check -> Discovery (sanity + improvement screen) -> Holdout gates."""
        t0 = time.time()
        champ = self.bootstrap()
        self._check_lab_intact()
        it = self.next_iteration()
        code_sha = sha256_text(new_forecaster_code)
        champion_code = self.backup_path.read_text()
        cand_dir = self.candidates_dir / f"iter_{it:03d}"
        family = normalize_family(hypothesis_family)
        if family == "unspecified" and infer_family(hypothesis) != "unspecified":
            family = infer_family(hypothesis)
        dstats = diff_stats(unified_diff(champion_code, new_forecaster_code))
        mclass = normalize_model_class(model_class)
        champ_class = champion_model_class(self.experiment_registry())
        if mclass == "unspecified" and dstats.get("added", 0) + dstats.get("removed", 0) < 200:
            mclass = champ_class            # an untagged in-place edit of the champion stays in the champion's class
        nov = verified_novelty(novelty, dstats, mclass, champ_class)
        base = {"iteration": it, "hypothesis": hypothesis[:2000], "hypothesis_family": family, "model_class": mclass,
                "novelty": nov, "code_sha256": code_sha, "parent_code_sha256": champ["code_sha256"],
                "candidate_dir": str(cand_dir.relative_to(self.root)), "diff_vs_champion_lines": dstats}
        if nov != normalize_novelty(novelty):
            base["novelty_claimed"] = normalize_novelty(novelty)
        if builds_on_iteration is not None:
            base["builds_on_iteration"] = int(builds_on_iteration)
        if hypothesis_card:
            base["hypothesis_card"] = {k: str(hypothesis_card.get(k) or "")[:800] for k in HYPOTHESIS_CARD_FIELDS}
        self._pending_archive = (cand_dir, new_forecaster_code, champion_code)
        problems = audit_forecaster_source(new_forecaster_code)
        if problems:
            status = "RUNTIME_ERROR" if any(p.startswith("SyntaxError") for p in problems) else "REJECTED_FIREWALL"
            return self._finish(base, status, theory_update_md, t0, error="; ".join(problems)[:2000])
        if code_sha == champ["code_sha256"] or code_sha in self._tested_shas():
            return self._finish(base, "REJECTED_DUPLICATE", theory_update_md, t0,
                                error="this exact forecaster.py is the champion or was already evaluated (deterministic "
                                      "verdict); change the code before re-testing")
        self.cfg.forecaster_path.write_text(new_forecaster_code)
        res = self.run_evaluation(self.cfg.forecaster_path, phase="discovery")
        if not res.get("ok"):
            self._restore_champion()
            return self._finish(base, "RUNTIME_ERROR", theory_update_md, t0, error=res.get("error"),
                                traceback_tail=res.get("traceback_tail"))
        disc = res["discovery"]
        self._pending_discovery = disc
        screen = screen_candidate(disc, champ.get("discovery"), self.cfg)
        vs = compare_discovery(disc, champ.get("discovery"), self.cfg)
        rec = {**base, "discovery": self._discovery_summary(disc), "screen": screen,
               "discovery_delta_vs_champion": metric_delta(disc, champ.get("discovery")),
               "vs_champion_discovery": {k: vs[k] for k in vs if k != "slice_deltas_best_first"}}
        if not screen["pass"]:
            self._restore_champion()
            self._append_theory(it, hypothesis, theory_update_md, screen["status"], None,
                                f"pre-Holdout screen failed ({', '.join(screen['failed'])}): "
                                + json.dumps({k: screen["checks"][k] for k in screen["failed"]}, default=str))
            out = self._ledger({**rec, "status": screen["status"], "seconds": round(time.time() - t0, 1)})
            self._archive(out)
            return {**out, "vs_champion_discovery": vs, "discovery_diagnostics": disc}
        hres = self.run_evaluation(self.cfg.forecaster_path, phase="holdout")
        if not hres.get("ok"):
            self._restore_champion()
            return self._finish(base, "RUNTIME_ERROR", theory_update_md, t0, error=hres.get("error"),
                                traceback_tail=hres.get("traceback_tail"))
        res = {**res, **hres, "discovery": disc}
        # per-origin Holdout losses stay inside the lab (champion state only): the agent sees the paired error, not them
        by_origin = hres.get("holdout_by_origin")
        paired = paired_origin_stats(by_origin, champ.get("holdout_by_origin"), self.cfg)
        gates = evaluate_gates(res["holdout"], champ["holdout"], self.cfg, paired)
        if not res.get("holdout_sanity_ok", True):
            gates["holdout_sanity"] = {"pass": False}
            gates["all_pass"] = False
        rec.update(holdout=res["holdout"], benchmark_2026_10_24=res["benchmark_2026_10_24"], gates=gates,
                   gate_margins=gate_margins(gates),
                   holdout_delta_vs_champion=metric_delta(res["holdout"], champ["holdout"]),
                   champion_holdout_before=champ["holdout"])
        if paired:
            rec["holdout_paired_vs_champion"] = {k: paired[k] for k in ("delta_se", "delta_z", "n_origins",
                                                                         "n_independent_blocks", "share_origins_improved")}
        if is_no_op(res["holdout"], champ["holdout"]):
            rec["no_op"] = True
        if gates["all_pass"]:
            self._append_theory(it, hypothesis, theory_update_md, "PROMOTED", gates, family=base["hypothesis_family"])
            shutil.copyfile(self.cfg.forecaster_path, self.backup_path)
            commit = self.git_commit(f"[{self.cfg.stage}] iter {it} PROMOTED: {hypothesis[:60]}")
            state = {"stage": self.cfg.stage, "iteration": it, "code_sha256": code_sha, "git_commit": commit,
                     "discovery": res["discovery"], "holdout": res["holdout"], "holdout_by_origin": by_origin,
                     "benchmark_2026_10_24": res["benchmark_2026_10_24"], "since": utc_now()}
            self.state_path.write_text(json.dumps(state, indent=2))
            if self.use_git and commit:
                self._git("tag", "-f", f"{self.cfg.stage.lower()}-iter{it:03d}-promoted", commit)
            out = self._ledger({**rec, "status": "PROMOTED", "git_commit": commit, "seconds": round(time.time() - t0, 1)})
        else:
            self._restore_champion()
            self._append_theory(it, hypothesis, theory_update_md, "REVERTED", gates, family=base["hypothesis_family"],
                                no_op=bool(rec.get("no_op")))
            out = self._ledger({**rec, "status": "REVERTED", "seconds": round(time.time() - t0, 1)})
        self._archive(out)
        return {**out, "vs_champion_discovery": vs, "discovery_diagnostics": res["discovery"]}

    def _archive(self, rec: dict) -> None:
        pending, disc = getattr(self, "_pending_archive", None), getattr(self, "_pending_discovery", None)
        self._pending_archive = self._pending_discovery = None
        if pending:
            cand_dir, code, champion_code = pending
            archive_candidate(cand_dir, code, champion_code, rec, disc)

    def _finish(self, base: dict, status: str, theory_update_md: str, t0: float, **extra) -> dict:
        self._append_theory(base["iteration"], base["hypothesis"], theory_update_md, status, None, extra.get("error"),
                            family=base.get("hypothesis_family"))
        out = self._ledger({**base, "status": status, **extra, "seconds": round(time.time() - t0, 1)})
        self._archive(out)
        return out

    def _restore_champion(self) -> None:
        shutil.copyfile(self.backup_path, self.cfg.forecaster_path)

    def _append_theory(self, it: int, hypothesis: str, update_md: str, status: str, gates: dict | None,
                       error: str | None = None, family: str | None = None, no_op: bool = False) -> None:
        tag = f" [{family}]" if family and family != "unspecified" else ""
        lines = [f"\n## [{self.cfg.stage}] iteration {it}: {status}{' (NO_OP)' if no_op else ''}{tag} ({utc_now()})",
                 f"**Hypothesis.** {hypothesis.strip()}", ""]
        if update_md and update_md.strip():
            lines += [demote_headings(update_md.strip()[:MAX_THEORY_UPDATE_CHARS]), ""]
        if gates:
            g = gates
            lines.append(f"*Lab verdict:* holdout pooled {g['gate1_pooled_gain']['value']} (need > {g['gate1_pooled_gain']['threshold']}), "
                         f"median {g['gate2_breadth']['value']} (need >= {g['gate2_breadth']['threshold']}), "
                         f"coverage {g['gate3_calibration']['value']} (band {g['gate3_calibration']['band']}, or within "
                         f"{g['gate3_calibration'].get('max_distance_to_target')} of 0.80), horizons {g['gate4_horizon_floor']['value']}."
                         f" Gate margins {gate_margins(g)}.")
            if no_op:
                lines.append("*Lab verdict:* NO_OP: Holdout aggregates are identical to the champion's; the change never fired.")
        if error:
            lines.append(f"*Lab verdict:* {status}: {error[:600]}")
        with open(self.cfg.theory_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def update_scientist_prompt(self, new_playbook_md: str, rationale: str, iteration: int | None = None) -> dict:
        """Replace the playbook (`iteration` labels the ledger row). The lab-protected block is restored verbatim if the rewrite altered it; every
        previous version is archived; dropped guardrail bullets and "freeze the champion" language are reported."""
        self._check_lab_intact()
        text, restored = enforce_protected((new_playbook_md or "").strip())
        budget = MAX_PLAYBOOK_CHARS + len(PROTECTED_BLOCK)
        if len(text) < 200 + len(PROTECTED_BLOCK) or len(text) > budget:
            return {"status": "PROMPT_REJECTED",
                    "error": f"playbook (excluding the protected block) must be 200..{MAX_PLAYBOOK_CHARS} characters"}
        old = self.cfg.prompt_path.read_text() if self.cfg.prompt_path.exists() else ""
        it = self.next_iteration() - 1 if iteration is None else iteration
        if old:
            self.playbooks_dir.mkdir(parents=True, exist_ok=True)
            n = len(list(self.playbooks_dir.glob("*.md")))
            (self.playbooks_dir / f"v{n:03d}_before_iter{it:03d}_update.md").write_text(old)
        dropped = dropped_guardrails(old, text)
        warnings = collapse_warnings(text)
        self.cfg.prompt_path.write_text(text)
        rec = {"status": "PROMPT_UPDATED", "iteration": it, "rationale": (rationale or "")[:1500], "prompt_sha256": sha256_text(text),
               "protected_block_restored": restored, "dropped_guardrails": dropped, "collapse_warnings": warnings}
        out = self._ledger(rec)
        if dropped or warnings or restored:
            out["lab_feedback"] = ("Saved. " + ("The lab-protected block was restored. " if restored else "") +
                                   (f"You dropped {len(dropped)} guardrail bullet(s); re-add them unless an iteration "
                                    "falsified them (cite it). " if dropped else "") +
                                   ("Your playbook argues for stopping/freezing; the lab decides stopping, so replace it "
                                    "with ranked untried directions. " if warnings else ""))
        return out

    def _agent_file(self, kind: str, code: str) -> Path:
        d = self.cfg.outputs_dir / "runs"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{kind}_{sha256_text(code)[:12]}.py"
        p.write_text(code)
        return p

    def dry_run_discovery(self, code: str, iteration: int | None = None) -> dict:
        """Discovery-only evaluation of a candidate: pre-Holdout screen verdict, paired deltas vs the champion and full
        diagnostics. Never touches Holdout, never writes workspace/forecaster.py, never promotes; not an experiment."""
        t0 = time.time()
        self._check_lab_intact()
        champ = self.bootstrap()
        it = self.next_iteration() if iteration is None else iteration
        base = {"iteration": it, "code_sha256": sha256_text(code)}
        problems = audit_forecaster_source(code)
        if problems:
            return self._ledger({**base, "status": "DRY_RUN_REJECTED", "error": "; ".join(problems)[:2000]})
        res = self.run_evaluation(self._agent_file("dryrun", code), phase="discovery")
        if not res.get("ok"):
            return self._ledger({**base, "status": "DRY_RUN_ERROR", "error": res.get("error"),
                                 "traceback_tail": res.get("traceback_tail"), "seconds": round(time.time() - t0, 1)})
        d, cd = res["discovery"], champ.get("discovery") or {}
        screen = screen_candidate(d, cd, self.cfg)
        vs = compare_discovery(d, cd, self.cfg)
        same = all(d.get(k) == cd.get(k) for k in ("pooled_skill", "median_skill", "coverage_80"))
        rec = self._ledger({**base, "status": "DRY_RUN_OK", "discovery": self._discovery_summary(d), "screen": screen,
                            "would_reach_holdout": screen["pass"], "identical_to_champion_on_discovery": same,
                            "vs_champion_discovery": {k: vs[k] for k in vs if k != "slice_deltas_best_first"},
                            "seconds": round(time.time() - t0, 1)})
        return {**rec, "vs_champion_discovery": vs, "discovery_diagnostics": d}

    def run_discovery_analysis(self, code: str, origins: list[str], iteration: int | None = None) -> dict:
        """Run agent analysis code `analyze(pack_slice, origin_date)` on point-in-time Discovery slices (firewalled)."""
        self._check_lab_intact()
        origins = list(origins or [])
        bad = [o for o in origins if o not in self.cfg.discovery_origins]
        if not origins or bad:
            return {"status": "ANALYSIS_REJECTED", "error": f"origins must be a non-empty subset of the Discovery origins "
                                                            f"{list(self.cfg.discovery_origins)}; bad: {bad}"}
        problems = audit_analysis_source(code)
        if problems:
            return {"status": "ANALYSIS_REJECTED", "error": "; ".join(problems)[:2000]}
        t0 = time.time()
        res = self._run_lab_child(["lab.workbench", "analyze", "--code", str(self._agent_file("analysis", code)),
                                   "--origins", ",".join(origins)], ANALYSIS_TIMEOUT_S)
        it = self.next_iteration() if iteration is None else iteration
        self._ledger({"status": "ANALYSIS", "iteration": it, "code_sha256": sha256_text(code), "origins": origins,
                      "ok": bool(res.get("ok")), "error": res.get("error"), "seconds": round(time.time() - t0, 1)})
        if not res.get("ok"):
            return {"status": "ANALYSIS_ERROR", "error": res.get("error"), "traceback_tail": res.get("traceback_tail")}
        return {"status": "ANALYSIS_OK", "results": res["results"]}

    def backtest_candidate(self, code: str, reference_code: str | None = None, stride_days: int = 1,
                           series_ids: list[str] | None = None, start_date: str = "", end_date: str = "",
                           iteration: int | None = None) -> dict:
        """Dense paired backtest of a candidate vs a reference (default: the champion) on pseudo-origins inside the
        Discovery window (default: every day). Never touches Holdout; not an experiment."""
        from lab.evaluator import backtest_origins

        self._check_lab_intact()
        ref = reference_code if reference_code is not None else self.bootstrap() and self.backup_path.read_text()
        it = self.next_iteration() if iteration is None else iteration
        base = {"iteration": it, "code_sha256": sha256_text(code), "reference_sha256": sha256_text(ref)}
        problems = audit_forecaster_source(code) + [f"reference: {x}" for x in audit_forecaster_source(ref)]
        if problems:
            return self._ledger({**base, "status": "BACKTEST_REJECTED", "error": "; ".join(problems)[:2000]})
        try:
            origins = backtest_origins(self.cfg, stride_days, start_date or None, end_date or None)
        except ValueError as e:
            return {**base, "status": "BACKTEST_REJECTED", "error": f"bad start/end date: {e}"}
        if not origins:
            return {**base, "status": "BACKTEST_REJECTED", "error": "no pseudo-origins in the requested window"}
        args = ["lab.evaluator", "--forecaster", str(self._agent_file("backtest", code)), "--phase", "backtest",
                "--reference", str(self._agent_file("backtest_ref", ref)), "--origins", ",".join(origins)]
        if series_ids:
            args += ["--series", ",".join(str(x) for x in series_ids)]
        t0 = time.time()
        res = self._run_lab_child(args, self.cfg.eval_timeout_s)
        if not res.get("ok"):
            return self._ledger({**base, "status": "BACKTEST_ERROR", "error": res.get("error"),
                                 "traceback_tail": res.get("traceback_tail"), "n_origins": len(origins),
                                 "seconds": round(time.time() - t0, 1)})
        keys = ("n_origins", "pooled_skill_delta", "delta_se_origin_bootstrap", "delta_z", "share_origins_improved",
                "bootstrap_block_origins", "n_independent_blocks", "delta_se_iid_origins")
        self._ledger({**base, "status": "BACKTEST_OK", **{k: res.get(k) for k in keys},
                      "seconds": round(time.time() - t0, 1)})
        return {**base, "status": "BACKTEST_OK", **{k: v for k, v in res.items() if k != "ok"}}

    def profile_pack(self, origin_date: str) -> dict:
        """Data-quality and table-join profile of the point-in-time pack slice at origin_date (<= last Discovery)."""
        if origin_date > self.cfg.discovery_origins[-1]:
            return {"status": "PROFILE_REJECTED", "error": f"origin_date must be <= {self.cfg.discovery_origins[-1]}"}
        res = self._run_lab_child(["lab.workbench", "profile", "--origin", origin_date], ANALYSIS_TIMEOUT_S)
        if not res.get("ok"):
            return {"status": "PROFILE_ERROR", "error": res.get("error"), "traceback_tail": res.get("traceback_tail")}
        return {"status": "PROFILE_OK", **{k: v for k, v in res.items() if k != "ok"}}

    def search_exa_signals(self, query: str, origin_date: str) -> list[dict]:
        """Agent-side research search; pinned at or before the last Discovery origin so Holdout news never leaks."""
        if not self.cfg.enable_exa:
            raise PermissionError("search_exa_signals is only available in PACK_PLUS_EXA")
        last = self.cfg.discovery_origins[-1]
        if origin_date > last:
            raise PermissionError(f"agent research searches must use origin_date <= {last} (Discovery)")
        from lab.sandbox import PointInTimeSandbox
        return PointInTimeSandbox(self.cfg).search_exa_point_in_time(query, origin_date)

    # ------------------------------------------------ Architecture 1: procedure I_g, forecast locks, memory ----
    @property
    def evidence_path(self) -> Path:
        return self.memory_dir / "evidence.jsonl"

    @property
    def episodes_path(self) -> Path:
        return self.memory_dir / "episodes.jsonl"

    def ensure_procedure(self) -> dict:
        """Seed I_0 from the current playbook on first use; return the active procedure version."""
        self._ensure_protected_playbook()
        return self.procedures.ensure_seed(self.cfg.prompt_path.read_text())

    def pack_manifest_sha256(self) -> str | None:
        p = self.cfg.pack_dir / "MANIFEST.sha256"
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

    def lock_episode_forecast(self, code: str, card: dict, meta: dict, episode_id: str, procedure_version: str) -> dict:
        """Score an episode's committed forecaster on Discovery (pre-Holdout screen, Holdout untouched) and write an
        immutable lock: code / champion / pack hashes, procedure version, hypothesis card and the Discovery result."""
        champ = self.bootstrap()
        r = self.dry_run_discovery(code)
        if r.get("status") != "DRY_RUN_OK":  # a candidate that does not run is never locked: the agent can fix it
            return {"locked": False, "status": r.get("status"), "error": r.get("error"),
                    "traceback_tail": r.get("traceback_tail")}
        d, cd = r.get("discovery") or {}, champ.get("discovery") or {}
        a, b = d.get("pooled_skill"), cd.get("pooled_skill")
        delta = a - b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        (self.locks_dir / f"{episode_id}.py").write_text(code)
        rec = {"stage": self.cfg.stage, "procedure_version": procedure_version, "code_sha256": sha256_text(code),
               "champion_code_sha256": champ["code_sha256"], "pack_manifest_sha256": self.pack_manifest_sha256(),
               "origins": list(self.cfg.discovery_origins), "hypothesis_card": card, **meta,
               "status": r.get("status"), "error": r.get("error"), "screen_pass": bool(r.get("would_reach_holdout")),
               "discovery": r.get("discovery"), "discovery_pooled_delta": delta, "code_file": f"{episode_id}.py"}
        write_lock(self.locks_dir, episode_id, rec)
        return rec

    def record_evidence(self, episode_id: str, query: str, origin_date: str, results, hypothesis_link: str = "") -> dict:
        """Evidence memory: every Exa search with its point-in-time sources and the hypothesis it was meant to inform."""
        srcs = [{"url": x.get("url"), "title": str(x.get("title") or "")[:160],
                 "published": x.get("publishedDate") or x.get("published_date")}
                for x in (results or [])[:10] if isinstance(x, dict)]
        return append_jsonl(self.evidence_path, {"stage": self.cfg.stage, "episode_id": episode_id, "query": query,
                                                 "origin_date": origin_date, "hypothesis_link": (hypothesis_link or "")[:300],
                                                 "n_results": len(results or []), "sources": srcs})

    def released_episodes(self) -> list[dict]:
        return read_jsonl(self.episodes_path)

    def next_generation(self) -> int:
        return 1 + max([int(e.get("generation") or 0) for e in self.released_episodes()], default=0)

    def memory_digest(self, n_episodes: int = 12, n_evidence: int = 20) -> dict:
        """The three memories, time-filtered by construction (episodes and evidence only use Discovery-era data and
        released verdicts): procedural (I_g lineage, measured episode scores per version and per model class, rejected
        revision objects), episodic (research episodes + experiment registry), evidence."""
        active = self.ensure_procedure()
        lineage = [{k: v.get(k) for k in ("version", "parent_version", "status", "admission")}
                   | {"change": (v.get("revision") or {}).get("change"),
                      "target_component": (v.get("revision") or {}).get("target_component")}
                   for v in self.procedures.versions()]
        eps = self.released_episodes()
        invalid = [{"generation": r.get("generation"), "parent_version": r.get("parent_version"), "error": r.get("error")}
                   for r in self.ledger_entries(self.cfg.stage) if r.get("status") == "REVISION_INVALID"]
        return {"procedural_memory": {"active_version": active["version"], "lineage": lineage,
                                      "episode_controls": procedure_controls(active),
                                      "episode_scores_by_procedure": episode_stats(eps, lambda e: e.get("procedure_version")),
                                      "episode_scores_by_model_class": episode_stats(
                                          eps, lambda e: (e.get("lock") or {}).get("model_class") or "no_lock"),
                                      "invalid_revisions": invalid[-5:]},
                "episode_memory": [compact_episode(e) for e in eps[-n_episodes:]],
                "experiment_registry": self.experiment_registry(),
                "evidence_memory": read_jsonl(self.evidence_path)[-n_evidence:]}

    def write_procedure_report(self) -> Path:
        """lineage.md: procedure versions with their admission comparisons, then every episode's behavior."""
        lines = [self.procedures.lineage_md(), "## Research episodes", "",
                 "| episode | procedure | tools used | model class | locked score | Holdout shot |", "|---|---|---|---|---|---|"]
        for e in self.released_episodes():
            lock, ho = e.get("lock") or {}, e.get("holdout_result") or {}
            tools = ", ".join(f"{k}x{v}" for k, v in (e.get("tool_counts") or {}).items())
            lines.append(f"| {e['episode_id']} | {e.get('procedure_version')} | {tools} | {lock.get('model_class', '-')} | "
                         f"{e.get('score')} | {ho.get('status', '-')} {(ho.get('holdout') or {}).get('pooled_skill', '')} |")
        out = self.procedures.root / "lineage.md"
        self.procedures.root.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        return out

    # ----------------------------------------------------------------------- stage control ----
    def checkpoint_notes(self, message: str) -> str | None:
        return self.git_commit(f"[{self.cfg.stage}] {message}")

    def fork_from(self, parent_stage: str, seed_path: Path | None = None) -> dict:
        """Start this stage from another stage's champion (Stage 2 forks from the Stage 1 champion), or from a
        lab-provided seed file (the long-horizon track: 1/3/7-day momentum does not extrapolate to 180 days)."""
        parent_dir = self.cfg.outputs_dir / "champions" / parent_stage
        parent = json.loads((parent_dir / "champion_state.json").read_text())
        src = seed_path or parent_dir / "forecaster.py"
        self.champ_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, self.cfg.forecaster_path)
        shutil.copyfile(src, self.backup_path)
        if self.state_path.exists():
            self.state_path.unlink()
        if self.use_git and self.cfg.stage == STAGE_PACK_PLUS_EXA:
            branch = f"stage2-{self.cfg.stage.lower().replace('_', '-')}"
            if self._git("branch", "--list", branch):
                self._git("checkout", "-q", branch)
            else:
                self._git("checkout", "-q", "-b", branch)
        state = self.bootstrap()
        self._ledger({"status": "FORK", "iteration": 0, "parent_stage": parent_stage,
                      "parent_code_sha256": parent["code_sha256"], "parent_git_commit": parent.get("git_commit"),
                      "holdout": state["holdout"], "benchmark_2026_10_24": state["benchmark_2026_10_24"]})
        return state


# =================================================================================== drivers ====
AGENT_CONTRACT = f"""
You are an autonomous AI Scientist improving a probabilistic forecaster of GPU cloud rental prices (USD per GPU-hour).
Targets: q10/q50/q90 at horizons 1, 3 and 7 days for 97 benchmark series (84 hyperscaler_spot incl. 80 AWS EC2 spot AZ
series, 12 neocloud_rate_card, 1 marketplace_ask). Score = pooled pinball skill versus the flat last-value forecast.

CONTRACT FOR workspace/forecaster.py (it is loaded and called by an immutable lab):
  class Forecaster:
      def predict_origin(self, pack_slice: dict, origin_date: str, horizons: tuple, external_signals=None) -> pd.DataFrame
  Return columns series_id, origin_date, horizon_days, q10, q50, q90 for every series in pack_slice["targets"]
  (0 < q10 <= q50 <= q90). Missing or invalid cells are filled with the last value (zero skill).
  pack_slice holds DataFrames truncated at origin_date: series_history (series_id, date, value, n_obs, n_providers,
  is_filled, instrument_class), targets and catalog (static keys: series_id, table, source_id, instrument_class,
  statistic, gpu_model, pricing_tier, instance_type, az_global_id, members, gpu_family), daily_statistics
  (p25/median/p75/min per collector), aggregator_medians (weekly, complete weeks only), index_prints,
  hardware_fundamentals, capability_ladder, capability_units. The tables name GPUs differently ("NVIDIA H100" vs
  "H100 SXM 80GB"); join them on the lab's canonical gpu_family column (e.g. "H100"), never on gpu_model.
  aws_spot_ticks is not exposed raw: it is the source of the aws_spot.* rows of series_history (daily close,
  n_obs = ticks that day, is_filled = forward-filled day); 53 non-target AWS series (other AZs / p6) are there too.
  external_signals is None in PACK_ONLY; in PACK_PLUS_EXA it is an object with
  .origin_date and .search(query, domains=None, num_results=8) -> list of {{title, url, published_date, highlights}},
  point-in-time (published on or before origin_date) and cached.
  Imports: {import_policy_text()}. Code that breaks this is rejected (REJECTED_FIREWALL).
  CPU-only, 8 cores: keep the run under ~120 s per origin (backtest_candidate reports seconds per origin) and use
  fixed random seeds (np.random.default_rng(seed), torch.manual_seed) so replays are exact.

PROTOCOL: Discovery origins {DISCOVERY_ORIGINS[0]}..{DISCOVERY_ORIGINS[-1]} return full diagnostics; Holdout origins
{HOLDOUT_ORIGINS[0]}..{HOLDOUT_ORIGINS[-1]} return only 4 aggregates and decide promotion via 4 gates. Evaluation is
two-stage: a candidate reaches Holdout only if its outputs are sane (finite, within 10x of the last value) and its
Discovery pooled skill beats the champion's; otherwise it is REJECTED_SANITY / REJECTED_DISCOVERY. A failed
candidate is reverted, but your notebook entry and playbook persist: write down what you learned, including negative
results, and make every hypothesis falsifiable. Never hard-code dates, series-specific constants tuned to a single
origin, or anything that could only be known after an origin.

MEMORY: every candidate (including reverted and crashed ones) is archived with its code, its diff against the champion
it was tested against, its gate margins (>0 passes) and its full Discovery diagnostics; read one with
inspect_candidate(iteration). The experiment registry lists every experiment of this stage with its hypothesis family.
Tag each proposal with hypothesis_family (one of: {", ".join(HYPOTHESIS_FAMILIES)}), novelty
({" | ".join(NOVELTY_LEVELS)}) and model_class, the model behind the AWS-spot quantiles (one of:
{", ".join(MODEL_CLASSES)}); set builds_on_iteration when you start from an archived candidate instead of the champion.
"bold" is verified by the lab: it needs a different model_class or a diff of at least 40 lines, otherwise it is recorded
as incremental. The champion is one point in model space, not the template every candidate must be a tweak of: a
forecaster of a different class written from scratch is a legitimate (and, after a run of small reverts, required)
experiment.
""".strip()

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "description": "one falsifiable hypothesis tested by this candidate"},
        "theory_update_md": {"type": "string", "description": "markdown notebook entry: diagnosis, mechanism, expected effect"},
        "new_forecaster_code": {"type": "string", "description": "the complete new workspace/forecaster.py"},
        "new_playbook_md": {"type": "string", "description": "the complete revised workspace/scientist_prompt.md"},
        "playbook_rationale": {"type": "string"},
        "hypothesis_family": {"type": "string", "enum": list(HYPOTHESIS_FAMILIES)},
        "novelty": {"type": "string", "enum": list(NOVELTY_LEVELS)},
        "model_class": {"type": "string", "enum": list(MODEL_CLASSES)},
        "builds_on_iteration": {"type": ["integer", "null"],
                                "description": "archived candidate this one starts from (null = the champion)"},
    },
    "required": ["hypothesis", "theory_update_md", "new_forecaster_code", "new_playbook_md", "playbook_rationale",
                 "hypothesis_family", "novelty"],
}


def build_iteration_prompt(state: dict, max_near_miss_diff_chars: int = 6000) -> str:
    """Single-shot prompt (Devin API driver). Order: contract, directive, full registry, near-miss diffs, champion,
    playbook, notebook tail. Each section has its own budget, so no section can push another out of the prompt."""
    def j(x, n):
        s = json.dumps(x, default=str)
        return s if len(s) <= n else s[:n] + " ...[truncated by the lab]"

    near = []
    for nm in state.get("near_misses") or []:
        diff = (nm.get("diff_vs_current_champion") or "")[:max_near_miss_diff_chars]
        near.append(f"### Iteration {nm['it']} ({nm.get('family')}), gate margins {nm.get('margins')}\n"
                    f"{nm.get('hypothesis', '')}\n```diff\n{diff}\n```")
    return "\n\n".join([
        AGENT_CONTRACT,
        "## Exploration directive (from the lab)\n" + state.get("exploration_directive", ""),
        "## Experiment registry (every experiment of this stage, oldest first)\n" + j(state.get("experiment_registry"), 16000),
        "## Hypothesis families tried\n" + j(state.get("family_stats"), 2000),
        "## Closest near-misses (diff vs the current champion)\n" + ("\n\n".join(near) or "none yet"),
        "## Promotion gates\n" + j(state["promotion_gates"], 1000),
        "## Champion Holdout aggregates\n" + j(state["champion_holdout_aggregates"], 2000),
        "## Champion Discovery diagnostics\n" + j(state["champion_discovery_diagnostics"], 14000),
        "## Champion forecaster (workspace/forecaster.py)\n```python\n" + state["champion_forecaster_py"] + "\n```",
        "## Most recent experiments (full records)\n" + j(state["ledger_tail"], 9000),
        "## Your current research playbook (workspace/scientist_prompt.md)\n" + state.get("scientist_prompt_md", ""),
        "## Notebook tail (workspace/theory_of_market.md)\n" + state["theory_of_market_tail_md"],
        f"## Task (iteration {state['next_iteration']}, stage {state['stage']})\n"
        "Diagnose the Discovery errors, state ONE falsifiable hypothesis, and write the complete new forecaster.py that "
        "tests it (from the champion or from a near-miss above; set builds_on_iteration accordingly). Follow the "
        "exploration directive. Also return a revised playbook (heuristics, guardrails and ranked untried ideas; keep the "
        "lab-protected block). Work only from the material above: do not clone repositories, open pull requests or "
        "browse for GPU price data. Reply ONLY via the structured output.",
    ])


class ScriptedDriver:
    """Deterministic driver: replays a list of proposals (dicts or forecaster file paths). Used by tests and reproduce.sh."""

    name = "scripted"

    def __init__(self, proposals: list):
        self.proposals = list(proposals)

    def run_iteration(self, ctrl: RSIController, i: int) -> dict:
        p = self.proposals[i]
        if isinstance(p, (str, Path)):
            code = Path(p).read_text()
            doc = code.split('"""')[1].strip() if code.startswith('"""') else Path(p).stem
            p = {"hypothesis": doc.splitlines()[0], "theory_update_md": doc, "new_forecaster_code": code}
        if p.get("new_playbook_md"):
            ctrl.update_scientist_prompt(p["new_playbook_md"], p.get("playbook_rationale", ""), iteration=ctrl.next_iteration())
        return ctrl.propose_and_test_forecaster(p["hypothesis"], p["theory_update_md"], p["new_forecaster_code"],
                                                p.get("hypothesis_family", ""), p.get("novelty", ""),
                                                p.get("builds_on_iteration"), p.get("model_class", ""))


class DevinAPIDriver:
    """One Devin session per iteration; the proposal comes back through structured_output and is tested locally."""

    name = "devin_api"

    def __init__(self, poll_s: int = 20, timeout_s: int | None = None):
        import httpx

        self.key = os.environ.get("DEVIN_API_KEY")
        self.org = os.environ.get("DEVIN_ORG_ID")
        if not self.key:
            raise RuntimeError("DEVIN_API_KEY is not set")
        self.http = httpx.Client(base_url=os.environ.get("DEVIN_API_BASE", "https://api.devin.ai"),
                                 headers={"Authorization": f"Bearer {self.key}"}, timeout=60)
        self.poll_s = poll_s
        self.timeout_s = timeout_s or int(os.environ.get("DEVIN_SESSION_TIMEOUT_S", "2700"))
        self.max_acu = int(os.environ["DEVIN_MAX_ACU"]) if os.environ.get("DEVIN_MAX_ACU") else None
        self.sessions: list[dict] = []

    def _create(self, prompt: str, title: str) -> tuple[str, str]:
        body = {"prompt": prompt, "title": title, "structured_output_schema": PROPOSAL_SCHEMA, "tags": ["ai-scientist", "rsi"]}
        if self.max_acu:
            body["max_acu_limit"] = self.max_acu
        if self.org:
            r = self.http.post(f"/v3/organizations/{self.org}/sessions", json=body)
            if r.status_code < 300:
                return "v3", r.json()["session_id"]
            print(f"[devin] v3 create failed ({r.status_code}: {r.text[:300]}); falling back to v1", file=sys.stderr)
        r = self.http.post("/v1/sessions", json=body)
        r.raise_for_status()
        return "v1", r.json()["session_id"]

    def _get(self, api: str, sid: str) -> dict:
        path = f"/v3/organizations/{self.org}/sessions/{sid}" if api == "v3" else f"/v1/sessions/{sid}"
        r = self.http.get(path)
        r.raise_for_status()
        return r.json()

    def _sleep(self, s: float) -> None:
        time.sleep(s)

    def propose(self, prompt: str, title: str) -> dict:
        api, sid = self._create(prompt, title)
        rec = {"api": api, "session_id": sid, "created": utc_now()}
        self.sessions.append(rec)
        print(f"[devin] session {sid} ({api}) created", file=sys.stderr)
        t0, last, prev_so, http_failures, idle_since = time.time(), None, None, 0, None
        while time.time() - t0 < self.timeout_s:
            self._sleep(self.poll_s)
            try:
                s = self._get(api, sid)
                http_failures = 0
            except Exception as e:  # transient API errors must not kill a 45-minute session
                http_failures += 1
                print(f"[devin] {sid}: poll failed ({type(e).__name__}: {e}); retry {http_failures}", file=sys.stderr)
                if http_failures >= DEVIN_MAX_POLL_FAILURES:
                    raise
                continue
            so = s.get("structured_output")
            status = devin_status(s)
            if status != last:
                print(f"[devin] {sid}: {status}", file=sys.stderr)
                last = status
            idle = devin_is_idle(status)
            idle_since = (idle_since or time.time()) if idle else None
            action = devin_poll_action(status, so, prev_so, 0.0 if idle_since is None else time.time() - idle_since)
            prev_so = so
            if action == "return":
                rec.update(acus=s.get("acus_consumed"), finished=utc_now(), status=list(status))
                return so
            if action == "fail":
                rec.update(acus=s.get("acus_consumed"), finished=utc_now(), status=list(status))
                raise RuntimeError(f"Devin session {sid} ended without a complete structured output: {status}")
        raise TimeoutError(f"Devin session {sid} produced no structured output within {self.timeout_s}s")

    def run_iteration(self, ctrl: RSIController, i: int) -> dict:
        state = ctrl.inspect_current_state()
        for nm in state["near_misses"]:
            nm["diff_vs_current_champion"] = ctrl.inspect_candidate(nm["it"]).get("diff_vs_current_champion", "")
        so = self.propose(build_iteration_prompt(state), f"AI Scientist {state['stage']} iteration {state['next_iteration']}")
        res = ctrl.propose_and_test_forecaster(so["hypothesis"], so.get("theory_update_md", ""), so["new_forecaster_code"],
                                               so.get("hypothesis_family", ""), so.get("novelty", ""),
                                               so.get("builds_on_iteration"), so.get("model_class", ""))
        if so.get("new_playbook_md"):
            ctrl.update_scientist_prompt(so["new_playbook_md"], so.get("playbook_rationale", ""))
        res["devin_session"] = self.sessions[-1]
        return res


DEVIN_MAX_POLL_FAILURES = 5
DEVIN_IDLE_GRACE_S = 600
# v3: status in new/claimed/running/resuming/suspended/exit/error, status_detail e.g. working/waiting_for_user/
# waiting_for_approval/finished; v1: status_enum in working/blocked/finished/expired/suspend_requested/... .
DEVIN_TERMINAL = {"exit", "error", "finished", "expired", "stopped", "suspended"}
DEVIN_IDLE = {"blocked", "waiting_for_user", "waiting_for_approval"}


def devin_status(s: dict) -> tuple:
    """(status, detail) of a v1 or v3 session payload, lower-cased strings (None if absent)."""
    st = s.get("status")
    det = s.get("status_detail") or s.get("status_enum")
    return tuple(str(x).lower() if x is not None else None for x in (st, det))


def devin_is_idle(status: tuple) -> bool:
    return any(x in DEVIN_IDLE for x in status if x)


def devin_poll_action(status: tuple, so, prev_so, idle_s: float, idle_grace_s: float = DEVIN_IDLE_GRACE_S) -> str:
    """'return' | 'fail' | 'wait' for one poll of a Devin session.

    A complete structured output is returned once the session is terminal/idle, or once it is unchanged between two
    polls (Devin may update it incrementally). A terminal session without one fails immediately (the old loop only
    failed when the status contained 'error', so exit/finished/expired sessions burned the full timeout), and a
    session idle (blocked / waiting for the user) without one fails after `idle_grace_s`.
    """
    complete = isinstance(so, dict) and all(so.get(k) for k in ("hypothesis", "new_forecaster_code"))
    terminal = any(x in DEVIN_TERMINAL for x in status if x)
    idle = devin_is_idle(status)
    if complete and (terminal or idle or so == prev_so):
        return "return"
    if not complete and (terminal or (idle and idle_s >= idle_grace_s)):
        return "fail"
    return "wait"


class StrandsAgentDriver:
    """A Strands agent (OpenAI-compatible or LiteLLM model; no Anthropic SDK) driving the lab through @tool functions."""

    name = "strands"

    def __init__(self, provider: str | None = None):
        provider = provider or os.environ.get("STRANDS_PROVIDER", "openai")
        model_id = os.environ.get("STRANDS_MODEL_ID", "llm" if provider == "modal" else "gpt-4.1")
        self.name = f"strands_{provider}"
        if provider == "modal":
            from strands.models.openai import OpenAIModel
            base = os.environ.get("MODAL_LLM_URL", DEFAULT_MODAL_LLM_URL).rstrip("/")
            headers = modal_proxy_headers()
            api_key = os.environ.get("MODAL_LLM_API_KEY", "EMPTY")
            wait_for_health(base, api_key, headers)
            if not os.environ.get("STRANDS_MODEL_ID"):
                model_id = discover_model_id(base, api_key, headers)
            self.name = f"strands_modal:{model_id}"
            self.model = OpenAIModel(client_args={"api_key": api_key, "base_url": base + "/v1", "default_headers": headers,
                                                  "timeout": 1800, "max_retries": 2},
                                     model_id=model_id, params={"temperature": 0.2, "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "64000")),
                                                                "reasoning_effort": os.environ.get("LLM_REASONING_EFFORT", "high")})
        elif provider == "litellm":
            from strands.models.litellm import LiteLLMModel
            self.model = LiteLLMModel(model_id=model_id, params={"temperature": 0.2})
        else:
            from strands.models.openai import OpenAIModel
            client_args = {"api_key": os.environ.get("OPENAI_API_KEY")}
            if os.environ.get("OPENAI_BASE_URL"):
                client_args["base_url"] = os.environ["OPENAI_BASE_URL"]
            self.model = OpenAIModel(client_args=client_args, model_id=model_id, params={"temperature": 0.2})

    def run_iteration(self, ctrl: RSIController, i: int) -> dict:
        from strands import Agent
        from strands.types.exceptions import MaxTokensReachedException

        tb = AgentToolbox(ctrl)
        tools = traced_tools(tb, list(AgentToolbox.TOOL_NAMES) + (["search_exa_signals"] if ctrl.cfg.enable_exa else []))
        ctrl._ensure_protected_playbook()
        playbook = ctrl.cfg.prompt_path.read_text()
        registry = ctrl.experiment_registry()
        directive = exploration_directive(registry, ctrl.budget())
        explore = model_class_exploration_required(registry)
        agent = Agent(model=self.model, tools=tools, system_prompt=AGENT_CONTRACT + "\n\n" + WORKBENCH_PROTOCOL + "\n\n"
                      + playbook, callback_handler=None)
        prompt = (f"Iteration {tb.iteration} of stage {ctrl.cfg.stage}. Lab directive: {directive}\n"
                  "Call inspect_current_state and diagnose the Discovery errors (use inspect_candidate on near-misses "
                  "or on the experiments your idea relates to). "
                  + ("This is a MODEL-CLASS EXPLORATION iteration: write at least two forecasters of different model "
                     "classes from scratch (write_candidate, save_variant each), dry-run every one on Discovery, "
                     "load_variant the best " if explore else
                     "Then build the candidate (edit_candidate for a refinement, write_candidate for a different model "
                     "class, load_candidate to start from a near-miss; save_variant/load_variant to compare several), ")
                  + f"dry-run on Discovery (at most {tb.max_dry_runs} dry runs), submit_candidate_for_holdout exactly "
                  "once with one falsifiable hypothesis, its family, model_class and novelty, and finally call update_scientist_prompt with your revised "
                  "playbook (keep the lab-protected block; record what you learned and rank untried directions).")
        if ctrl.cfg.enable_exa:
            prompt += ("\n" + EXA_STAGE_DIRECTIVE)
        if ctrl.cfg.is_long:
            prompt += "\n" + LONG_STAGE_DIRECTIVE
        retries = int(os.environ.get("LLM_MAX_TOKEN_RETRIES", "2"))
        for attempt in range(retries + 2):
            try:
                agent(prompt)
            except MaxTokensReachedException as e:
                if tb.results:
                    print(f"[rsi] output limit after the Holdout submission; keeping the result ({e})", file=sys.stderr)
                    break
                if attempt >= retries:
                    raise
                self._raise_output_budget()
                prompt = CONTINUE_AFTER_MAX_TOKENS
                continue
            if tb.results or attempt >= retries + 1:
                break
            prompt = NUDGE_SUBMIT
        if not tb.results:
            raise RuntimeError("the Strands agent finished without submitting a candidate for Holdout")
        return tb.results[-1]

    def run_episode(self, ctrl: RSIController, version: dict, episode_id: str, arm: str) -> AgentToolbox:
        """One bounded research episode governed by procedure `version` (its research_policy is the playbook). The
        agent researches and commits (locks) at most one forecast; no Holdout, no playbook edits."""
        from strands import Agent
        from strands.types.exceptions import MaxTokensReachedException

        tb = AgentToolbox(ctrl, mode="episode", episode_id=episode_id, procedure_version=version["version"],
                          controls=procedure_controls(version))
        names = list(AgentToolbox.EPISODE_TOOL_NAMES) + (["search_exa_signals"] if ctrl.cfg.enable_exa else [])
        agent = Agent(model=self.model, tools=traced_tools(tb, names), callback_handler=None,
                      system_prompt=AGENT_CONTRACT + "\n\n" + EPISODE_PROTOCOL + "\n\n"
                      + version["components"]["research_policy"])
        registry = ctrl.experiment_registry()
        explore = model_class_exploration_required(registry)
        prompt = (f"Research episode {episode_id} (stage {ctrl.cfg.stage}, procedure {version['version']}). Lab directive: "
                  f"{exploration_directive(registry, None)}\nFollow your research policy (system prompt). Start with "
                  "inspect_current_state and recall_memory. Before proposing, decompose the champion's Discovery "
                  "loss (loss_decomposition, by_horizon, slices, largest_loss_series in the state; analyze_discovery "
                  "for anything finer) and aim at the largest share of it; the size of the change should match the "
                  "size of that share (a new model or feature set is allowed). Then decide one hypothesis card "
                  "(mechanism, expected observation, competing explanation, test, rejection condition). Budget: "
                  f"{tb.max_dry_runs} dry runs (a crashing candidate or analysis is refunded), {tb.max_analyses} analyses, "
                  f"{tb.max_backtests} backtests, "
                  f"{tb.max_exa if ctrl.cfg.enable_exa else 0} Exa searches. "
                  + ("This episode must compare forecasters of at least two different model classes (save_variant "
                     "each). " if explore else "")
                  + (f"Variants carried over from earlier episodes (load_variant): {sorted(tb.variants)}. "
                     if tb.variants else "")
                  + "End with commit_forecast for the best candidate, or end without committing if nothing beats "
                  "the champion.")
        if ctrl.cfg.enable_exa:
            prompt += "\n" + EXA_STAGE_DIRECTIVE
        if ctrl.cfg.is_long:
            prompt += "\n" + LONG_STAGE_DIRECTIVE
        retries = int(os.environ.get("LLM_MAX_TOKEN_RETRIES", "2"))
        for attempt in range(retries + 2):
            try:
                agent(prompt)
            except MaxTokensReachedException as e:
                if tb.lock is not None or attempt >= retries:
                    print(f"[rsi] episode {episode_id}: output limit ({e})", file=sys.stderr)
                    break
                self._raise_output_budget()
                prompt = CONTINUE_AFTER_MAX_TOKENS
                continue
            if tb.lock is not None or tb.n_dry_runs == 0 or attempt >= retries + 1:
                break
            prompt = NUDGE_COMMIT
        return tb

    def propose_revision(self, ctrl: RSIController, active: dict, digest: dict, known_ids: set[str],
                         max_attempts: int = 3) -> tuple[dict | None, list[str]]:
        """The improver (A7): a separate tool-less call governed by the active improver_policy. Returns a revision
        object that passed structural validation, or (None, problems)."""
        from strands import Agent
        from strands.types.exceptions import MaxTokensReachedException

        agent = Agent(model=self.model, tools=[], callback_handler=None,
                      system_prompt=active["components"]["improver_policy"] + "\n\n" + REVISION_PROTOCOL)
        prompt = improver_prompt(active, digest)
        problems: list[str] = []
        for _ in range(max_attempts):
            try:
                out = str(agent(prompt))
            except MaxTokensReachedException:
                self._raise_output_budget()
                prompt = ("Output limit hit. Reply with only the JSON object; new_component_text must still be the "
                          f"COMPLETE component (at most {MAX_COMPONENT_CHARS} characters), so shorten the component "
                          "itself rather than truncating it.")
                continue
            rev = parse_json_object(out)
            problems = validate_revision(rev, active, known_ids)
            if not problems:
                return rev, []
            prompt = ("The lab rejected the revision object: " + "; ".join(problems)
                      + ". Reply with the corrected JSON object only.")
        return None, problems

    def _raise_output_budget(self) -> None:
        """Double max_tokens (capped) after a MaxTokensReachedException so the retried turn can finish its tool call."""
        cfg = self.model.get_config() if hasattr(self.model, "get_config") else {}
        params = dict(cfg.get("params") or {})
        cur = params.get("max_tokens")
        if cur:
            params["max_tokens"] = min(int(cur) * 2, int(os.environ.get("LLM_MAX_TOKENS_CAP", "131072")))
            self.model.update_config(params=params)
            print(f"[rsi] output limit hit; retrying with max_tokens={params['max_tokens']}", file=sys.stderr)


WORKBENCH_PROTOCOL = """
WORKBENCH (per iteration). The candidate starts as a copy of the champion. Use view_candidate and edit_candidate
(exact, unique old_str -> new_str) for refinements of the champion; write_candidate replaces the whole file and is the
right tool for a different model class. save_variant(name) snapshots the working candidate and load_variant(name)
restores it, so one iteration can build and dry-run several competing forecasters and submit the best.
dry_run_discovery scores the current candidate on the Discovery origins
only (full diagnostics, paired deltas vs the champion and the pre-Holdout screen verdict `would_reach_holdout`;
Holdout untouched, champion untouched, nothing promoted). A submission whose Discovery pooled skill does not beat the
champion's, or whose outputs fail the sanity screen, is rejected before Holdout, so dry-run until would_reach_holdout
is true before submitting.
backtest_candidate scores the candidate against the champion (or an archived iteration) on DENSE daily pseudo-origins
inside the Discovery window (~70 origins instead of 11): paired pooled-skill delta with an origin-bootstrap standard
error (|delta_z| < 2 is noise), share of origins improved, deltas by horizon/slice/series and per-cell residuals
(y, last_value, both quantile sets) for up to 5 series_ids; use it instead of hand-written pseudo-origin loops.
profile_pack(origin_date) returns a lab-side data-quality profile: table sizes/null shares/date ranges, how many
targets each table joins to (via gpu_family) and the stale / forward-filled / gappy target series.
analyze_discovery runs your own `def analyze(pack_slice, origin_date)` (same import firewall as the forecaster, must
return JSON-able data) on point-in-time Discovery slices. submit_candidate_for_holdout runs the one Holdout evaluation
of this iteration (promote or revert); it can be called once. Keep your reasoning short and act through tools.
""".strip()

LONG_STAGE_DIRECTIVE = (
    "LONG-HORIZON TRACK (overrides the 1/3/7-day text of the contract): targets are every AWS EC2 spot series "
    "(aws_spot.*) at horizons 30, 90 and 180 days, monthly origins; Discovery origins 2023-06-01..2024-09-01, Holdout "
    "2025-03-01..2025-08-01. Score = pooled pinball skill vs the flat last value at each horizon. At these horizons "
    "momentum fades and regime drivers dominate (GPU generation launches, list-price cuts, capacity build-outs, demand "
    "shocks, instance retirement), so dated evidence matters: research it with search_exa_signals and use it inside "
    "predict_origin through external_signals.search. commit_forecast requires market_insight: a qualitative note "
    "(>= 200 characters) naming the drivers the forecast assumes, the dated evidence behind them (URLs from your "
    "searches), and what observation would change the view.")

EXA_STAGE_DIRECTIVE = (
    "Stage PACK_PLUS_EXA exists to test exogenous web signals (pack-only levers were explored in Stage 1, but only "
    "within one model class). If search_exa_signals returns nothing usable for this domain, record that and spend the "
    "iteration on a pack-only model class instead; do not dry-run a placeholder signal. "
    "Use search_exa_signals (point-in-time, origin_date <= the last Discovery origin) to find signals that "
    "plausibly move GPU rental prices (capacity announcements, price cuts, new GPU launches, provider outages), then "
    "make the forecaster use them through external_signals.search(query) inside predict_origin (results are "
    "point-in-time per origin and cached; keep queries few and fixed, and fall back to the champion's forecast when "
    "a search returns nothing). At least one dry run this iteration should test such a signal (family "
    "exogenous_signals).")
CONTINUE_AFTER_MAX_TOKENS = ("Your previous response hit the output-token limit before a tool call was made. Do not "
                             "restate your analysis. Make the next tool call now; use edit_candidate for a refinement, or "
                             "write_candidate in pieces (a short skeleton first, then edits) for a new model, then dry_run_discovery or submit_candidate_for_holdout.")
NUDGE_SUBMIT = ("You have not called submit_candidate_for_holdout yet. Finish the candidate with edit_candidate if needed, "
                "call submit_candidate_for_holdout exactly once, then update_scientist_prompt.")


def fit_backtest(r: dict, max_chars: int) -> dict:
    """Keep a backtest result inside the tool-output budget by thinning series_cells evenly over origins (every
    summary field survives); clip_json alone would cut the JSON mid-record and drop everything after it."""
    cells = r.get("series_cells") or {}
    n = max((len(v) for v in cells.values()), default=0)
    while n > 3 and len(json.dumps(r, default=str)) > max_chars:
        n = n * 2 // 3
        r = {**r, "series_cells": {k: v[:: -(-len(v) // n)] for k, v in cells.items()},
             "series_cells_note": f"thinned by the lab to <= {n} of {max(len(v) for v in cells.values())} cells per "
                                  "series (even spacing); narrow start_date/end_date or pass fewer series_ids for all"}
    return r


class AgentToolbox:
    """The in-iteration tools of the Strands driver as plain methods (wrapped with strands.tool by the driver)."""

    TOOL_NAMES = ("inspect_current_state", "inspect_candidate", "view_candidate", "edit_candidate", "write_candidate",
                  "load_candidate", "show_candidate_diff", "dry_run_discovery", "analyze_discovery",
                  "backtest_candidate", "profile_pack", "save_variant", "load_variant", "list_variants",
                  "submit_candidate_for_holdout", "propose_and_test_forecaster", "update_scientist_prompt")

    EPISODE_TOOL_NAMES = tuple(n for n in TOOL_NAMES if n not in ("submit_candidate_for_holdout",
                                                                  "propose_and_test_forecaster",
                                                                  "update_scientist_prompt")) + ("recall_memory",
                                                                                                 "commit_forecast")

    def __init__(self, ctrl: RSIController, max_dry_runs: int | None = None, max_analyses: int | None = None,
                 max_submissions: int = 1, mode: str = "holdout", episode_id: str = "", procedure_version: str = "",
                 controls: dict | None = None):
        self.ctrl = ctrl
        self.mode, self.episode_id, self.procedure_version = mode, episode_id, procedure_version
        self.lock: dict | None = None
        self.trace: list[dict] = []
        self.n_exa = 0
        self.max_exa = int(os.environ.get("RSI_EPISODE_EXA_SEARCHES", "2")) if mode == "episode" else 10 ** 6
        self.iteration = ctrl.next_iteration()
        ctrl.bootstrap()
        self.wb = CandidateWorkbench(ctrl.backup_path.read_text())
        self.model_class_exploration = model_class_exploration_required(ctrl.experiment_registry())
        default_dry = (int(os.environ.get("RSI_EXPLORE_DRY_RUNS", 2 * ctrl.cfg.max_discovery_probes))
                       if self.model_class_exploration else ctrl.cfg.max_discovery_probes)
        if mode == "episode":
            default_dry = int(os.environ.get("RSI_EPISODE_DRY_RUNS", "6"))
        self.max_dry_runs = (max_dry_runs if max_dry_runs is not None
                             else int(os.environ.get("RSI_MAX_DRY_RUNS", default_dry)))
        self.variants: dict[str, dict] = {}
        if mode == "episode":
            self._carry_variants()
        self.max_analyses = max_analyses if max_analyses is not None else int(os.environ.get("RSI_MAX_ANALYSES", "6"))
        self.max_backtests = int(os.environ.get("RSI_MAX_BACKTESTS", "4"))
        self.n_backtests = 0
        self.max_submissions = max_submissions
        self.n_dry_runs = self.n_analyses = self.n_playbook_updates = self.n_refunds = 0
        self.max_playbook_updates = int(os.environ.get("RSI_MAX_PLAYBOOK_UPDATES", "2"))
        if mode == "episode" and controls:   # I_g's typed budgets; an operator env override still wins for both arms
            for k, env in (("max_dry_runs", "RSI_EPISODE_DRY_RUNS"), ("max_backtests", "RSI_MAX_BACKTESTS"),
                           ("max_analyses", "RSI_MAX_ANALYSES")):
                if env not in os.environ and k in controls:
                    setattr(self, k, int(controls[k]))
        self.builds_on: int | None = None
        self.results: list[dict] = []

    def _carry_variants(self, keep: int = 4) -> None:
        """Variants saved by episodes of EARLIER generations (newest first, at most `keep`) start in this episode's
        variant store as "<episode>__<name>", so unfinished work is not rebuilt. Same-generation episodes are
        excluded: both arms of an admission comparison start from the same set."""
        gen = self.episode_id.split("_")[0]
        files = [p for p in sorted(self.ctrl.variants_dir.glob("*__*.py"), reverse=True)
                 if not p.name.startswith(gen + "_")] if self.ctrl.variants_dir.exists() else []
        for p in files[:keep]:
            self.variants[p.stem[:40]] = {"code": p.read_text(), "builds_on": None, "carried": True}

    def inspect_current_state(self) -> str:
        """Return the exploration directive, the full experiment registry (every experiment with family and gate
        margins), near-misses, champion code, Discovery diagnostics, Holdout aggregates, recent full ledger records
        and the notebook tail. Your playbook is already in the system prompt."""
        return self.ctrl.render_state(include_playbook=False)

    def inspect_candidate(self, iteration: int) -> str:
        """Return an archived candidate (any status): its code, its diff against the current champion, gates, gate
        margins and full Discovery diagnostics. Use it to build on near-misses instead of restarting."""
        return clip_json(self.ctrl.inspect_candidate(iteration), 45000)

    def load_candidate(self, iteration: int) -> str:
        """Replace the working candidate with the archived candidate of `iteration` (e.g. a near-miss) so you can
        refine it with edit_candidate. Pass builds_on_iteration=<iteration> when you submit."""
        d = self.ctrl.candidates_dir / f"iter_{int(iteration):03d}" / "forecaster.py"
        if not d.exists():
            return json.dumps({"ok": False, "error": f"no archived candidate for iteration {iteration}"})
        self.builds_on = int(iteration)
        return json.dumps(self.wb.write(d.read_text()))

    def save_variant(self, name: str) -> str:
        """Snapshot the working candidate under `name` (e.g. "qreg", "gbm") so several competing forecasters can be
        built and dry-run in one iteration; load_variant(name) restores one before submitting."""
        key = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:40]
        if not key:
            return json.dumps({"ok": False, "error": "give the variant a name"})
        if key not in self.variants and sum(not v.get("carried") for v in self.variants.values()) >= 8:
            return json.dumps({"ok": False, "error": "at most 8 variants per iteration"})
        self.variants[key] = {"code": self.wb.code, "builds_on": self.builds_on}
        if self.mode == "episode" and self.episode_id:  # kept for later generations (see _carry_variants)
            self.ctrl.variants_dir.mkdir(parents=True, exist_ok=True)
            (self.ctrl.variants_dir / f"{self.episode_id}__{key}.py").write_text(self.wb.code)
        return json.dumps({"ok": True, "saved": key, "variants": sorted(self.variants)})

    def load_variant(self, name: str) -> str:
        """Make the variant saved under `name` the working candidate again."""
        v = self.variants.get(re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:40])
        if v is None:
            return json.dumps({"ok": False, "error": f"no variant {name!r}; saved: {sorted(self.variants)}"})
        self.builds_on = v["builds_on"]
        return json.dumps(self.wb.write(v["code"]))

    def list_variants(self) -> str:
        """Saved variants with their size and diff vs the champion ("carried": saved by an earlier episode)."""
        return json.dumps({k: {"lines": v["code"].count("\n") + 1, "carried": bool(v.get("carried")),
                               "diff_vs_champion_lines": diff_stats(unified_diff(self.wb.base_code, v["code"]))}
                           for k, v in self.variants.items()})

    def view_candidate(self, start_line: int = 1, end_line: int = 0) -> str:
        """Show the current candidate forecaster with line numbers (end_line 0 = to the end)."""
        return self.wb.view(start_line, end_line or None)

    def edit_candidate(self, old_str: str, new_str: str) -> str:
        """Replace one exact, unique snippet old_str of the candidate with new_str. Returns status + firewall audit."""
        return json.dumps(self.wb.replace(old_str, new_str))

    def write_candidate(self, new_forecaster_code: str) -> str:
        """Replace the whole candidate forecaster (use it for a new model class or a large rewrite; edit_candidate for refinements)."""
        return json.dumps(self.wb.write(new_forecaster_code))

    def show_candidate_diff(self) -> str:
        """Unified diff of the candidate against the champion."""
        return self.wb.diff() or "(candidate is identical to the champion)"

    def dry_run_discovery(self) -> str:
        """Score the current candidate on the Discovery origins only (full diagnostics + delta vs the champion).
        Holdout is not evaluated, nothing is promoted and the champion is untouched."""
        if self.wb.code == self.wb.base_code:
            return json.dumps({"status": "DRY_RUN_SKIPPED", "error": "candidate is identical to the champion; edit it first"})
        if self.n_dry_runs >= self.max_dry_runs:
            return json.dumps({"status": "DRY_RUN_BUDGET_EXHAUSTED",
                               "error": f"{self.max_dry_runs} dry runs used; submit_candidate_for_holdout or stop"})
        self.n_dry_runs += 1
        r = self.ctrl.dry_run_discovery(self.wb.code, iteration=self.iteration)
        if r.get("status") != "DRY_RUN_OK" and self.n_refunds < self.max_dry_runs + self.max_analyses:
            # a candidate that did not run produced no evidence: refund it (bounded, so a crash loop still ends)
            self.n_dry_runs -= 1
            self.n_refunds += 1
            r = {**r, "refunded": True}
        return clip_json({**r, "dry_runs_left": self.max_dry_runs - self.n_dry_runs}, 30000)

    def analyze_discovery(self, analysis_code: str, origins: list[str]) -> str:
        """Run `def analyze(pack_slice, origin_date)` from analysis_code on point-in-time slices of the given Discovery
        origins (forecaster import firewall; return small JSON-able summaries, DataFrames are capped at 200 rows)."""
        if self.n_analyses >= self.max_analyses:
            return json.dumps({"status": "ANALYSIS_BUDGET_EXHAUSTED", "error": f"{self.max_analyses} analyses used"})
        self.n_analyses += 1
        r = self.ctrl.run_discovery_analysis(analysis_code, origins, iteration=self.iteration)
        if r.get("status") != "ANALYSIS_OK" and self.n_refunds < self.max_dry_runs + self.max_analyses:
            self.n_analyses -= 1  # failed analysis code returned no information: refunded (bounded, shared counter)
            self.n_refunds += 1
            r = {**r, "refunded": True}
        return clip_json({**r, "analyses_left": self.max_analyses - self.n_analyses}, 20000)

    def backtest_candidate(self, stride_days: int = 1, series_ids: list[str] | None = None, against_iteration: int = -1,
                           start_date: str = "", end_date: str = "") -> str:
        """Paired backtest of the current candidate vs the champion (or vs archived iteration `against_iteration`) on
        daily pseudo-origins inside the Discovery window (stride_days thins them; start/end_date narrow them). Returns
        pooled-skill delta + origin-bootstrap SE, deltas by horizon/slice/series and per-cell residuals of up to 5
        series_ids. Holdout is never touched."""
        if self.n_backtests >= self.max_backtests:
            return json.dumps({"status": "BACKTEST_BUDGET_EXHAUSTED", "error": f"{self.max_backtests} backtests used"})
        ref = None
        if against_iteration is not None and int(against_iteration) >= 0:
            p = self.ctrl.candidates_dir / f"iter_{int(against_iteration):03d}" / "forecaster.py"
            if not p.exists():
                return json.dumps({"status": "BACKTEST_REJECTED", "error": f"no archived candidate {against_iteration}"})
            ref = p.read_text()
        if self.wb.code == (ref if ref is not None else self.wb.base_code):
            return json.dumps({"status": "BACKTEST_SKIPPED", "error": "candidate is identical to the reference"})
        self.n_backtests += 1
        r = self.ctrl.backtest_candidate(self.wb.code, ref, stride_days, series_ids or [], start_date, end_date,
                                         iteration=self.iteration)
        return clip_json(fit_backtest({**r, "backtests_left": self.max_backtests - self.n_backtests}, 30000), 30000)

    def profile_pack(self, origin_date: str) -> str:
        """Data-quality / table-join profile of the point-in-time pack slice at origin_date (<= last Discovery
        origin): table sizes, null shares, date ranges, target coverage per table and stale/filled target series."""
        return clip_json(self.ctrl.profile_pack(origin_date), 20000)

    def submit_candidate_for_holdout(self, hypothesis: str, theory_update_md: str, hypothesis_family: str = "",
                                     novelty: str = "", builds_on_iteration: int = -1, model_class: str = "") -> str:
        """Evaluate the current candidate: pre-Holdout screen on Discovery, then the 4 Holdout gates; promote or revert.
        Allowed once per iteration. hypothesis_family is one of the lab's families (see inspect_current_state),
        novelty is incremental or bold, builds_on_iteration is the archived candidate you started from (-1 = the
        champion), model_class is the model behind the AWS quantiles (see model_classes). The candidate is archived
        whatever the verdict."""
        return self._submit(hypothesis, theory_update_md, self.wb.code, hypothesis_family, novelty, builds_on_iteration,
                            model_class)

    def propose_and_test_forecaster(self, hypothesis: str, theory_update_md: str, new_forecaster_code: str,
                                    hypothesis_family: str = "", novelty: str = "", builds_on_iteration: int = -1,
                                    model_class: str = "") -> str:
        """Legacy one-shot tool: replace the whole candidate with new_forecaster_code and submit it for Holdout.
        Counts as this iteration's single submission."""
        self.wb.write(new_forecaster_code)
        return self._submit(hypothesis, theory_update_md, new_forecaster_code, hypothesis_family, novelty,
                            builds_on_iteration, model_class)

    def _submit(self, hypothesis: str, theory_update_md: str, code: str, hypothesis_family: str = "",
                novelty: str = "", builds_on_iteration: int | None = -1, model_class: str = "") -> str:
        if len(self.results) >= self.max_submissions:
            return json.dumps({"status": "SUBMISSION_BUDGET_EXHAUSTED",
                               "error": "this iteration's Holdout submission was already used; call update_scientist_prompt"})
        if code == self.wb.base_code:
            return json.dumps({"status": "SUBMISSION_REJECTED", "error": "candidate is identical to the champion"})
        if builds_on_iteration is None or int(builds_on_iteration) < 0:
            builds_on_iteration = self.builds_on
        r = self.ctrl.propose_and_test_forecaster(hypothesis, theory_update_md, code, hypothesis_family, novelty,
                                                  builds_on_iteration, model_class)
        self.results.append(r)
        return fit_json(r, 30000, [("discovery_diagnostics", "head"), ("champion_holdout_before", "head")])

    def update_scientist_prompt(self, new_playbook_md: str, rationale: str) -> str:
        """Replace workspace/scientist_prompt.md (your research playbook) with a revised version. Limited per
        iteration: fold everything (including any guardrails the lab reports as dropped) into one rewrite."""
        if self.n_playbook_updates >= self.max_playbook_updates:
            return json.dumps({"status": "PLAYBOOK_BUDGET_EXHAUSTED",
                               "error": f"{self.max_playbook_updates} playbook updates used this iteration; stop now"})
        self.n_playbook_updates += 1
        r = self.ctrl.update_scientist_prompt(new_playbook_md, rationale, iteration=self.iteration)
        r["playbook_updates_left"] = self.max_playbook_updates - self.n_playbook_updates
        return json.dumps(r, default=str)

    def search_exa_signals(self, query: str, origin_date: str, hypothesis_link: str = "") -> str:
        """Point-in-time web search (results published on or before origin_date <= the last Discovery origin).
        hypothesis_link names the open question / hypothesis this search is meant to inform (kept in evidence memory).
        Budgeted per research episode: search only when you have a specific evidence question."""
        if self.n_exa >= self.max_exa:
            return json.dumps({"status": "EXA_BUDGET_EXHAUSTED", "error": f"{self.max_exa} searches used this episode"})
        self.n_exa += 1
        res = self.ctrl.search_exa_signals(query, origin_date)
        self.ctrl.record_evidence(self.episode_id or f"iter_{self.iteration:03d}", query, origin_date, res, hypothesis_link)
        return clip_json(res, 20000)

    def recall_memory(self) -> str:
        """Procedural memory (the active improvement-procedure version and its lineage with admission results),
        episode memory (past research episodes: procedure version, tools used, hypothesis card, locked Discovery score,
        Holdout outcome; failed attempts included), the experiment registry and evidence memory (past Exa searches,
        their sources and the hypotheses they informed)."""
        return fit_json(self.ctrl.memory_digest(), 30000, DIGEST_SHRINK_ORDER)

    def commit_forecast(self, hypothesis: str, mechanism: str, expected_observation: str, competing_explanation: str,
                        test: str, rejection_condition: str, model_class: str = "", hypothesis_family: str = "",
                        novelty: str = "", market_insight: str = "") -> str:
        """Lock the current candidate as this episode's forecast and end the episode. Requires a complete hypothesis
        card: mechanism, expected_observation, competing_explanation (a rival explanation of the same errors), test
        (the comparison that decides it) and rejection_condition (the result that would reject it). The lab scores the
        lock on Discovery and decides any Holdout evaluation after the episode. Allowed once. If no candidate beats
        the champion, do not commit: end the episode with a short summary (a valid research outcome)."""
        if self.lock is not None:
            return json.dumps({"status": "ALREADY_LOCKED", "error": "this episode's forecast is locked; stop now"})
        card = {"mechanism": mechanism, "expected_observation": expected_observation,
                "competing_explanation": competing_explanation, "test": test, "rejection_condition": rejection_condition}
        missing = validate_card(card)
        if missing:
            return json.dumps({"status": "COMMIT_REJECTED",
                               "error": f"hypothesis card incomplete (each field >= 15 characters): {missing}"})
        if self.ctrl.cfg.is_long and len((market_insight or "").strip()) < 200:
            return json.dumps({"status": "COMMIT_REJECTED", "error": "long-horizon track: market_insight (>= 200 "
                               "characters: drivers, dated evidence, what would change the view) is required"})
        if self.wb.code == self.wb.base_code:
            return json.dumps({"status": "COMMIT_REJECTED", "error": "candidate is identical to the champion"})
        problems = audit_forecaster_source(self.wb.code)
        if problems:
            return json.dumps({"status": "COMMIT_REJECTED", "error": "; ".join(problems)[:1500]})
        meta = {"hypothesis": hypothesis[:2000], "model_class": normalize_model_class(model_class),
                "hypothesis_family": hypothesis_family, "novelty": novelty, "builds_on_iteration": self.builds_on,
                "market_insight": (market_insight or "")[:4000]}
        rec = self.ctrl.lock_episode_forecast(self.wb.code, card, meta, self.episode_id, self.procedure_version)
        if rec.get("locked") is False:
            return clip_json({"status": "COMMIT_REJECTED", "error": "the candidate does not run on Discovery, nothing "
                              "was locked; fix it (or load_variant a working one) and commit again: "
                              f"{str(rec.get('error'))[:1200]}", "traceback_tail": rec.get("traceback_tail")}, 6000)
        self.lock = rec
        return clip_json({"status": "LOCKED", **{k: self.lock.get(k) for k in ("screen_pass", "discovery_pooled_delta",
                                                                               "error", "discovery")},
                          "note": "episode over; the lab compares procedures and decides Holdout"}, 8000)


EPISODE_PROTOCOL = WORKBENCH_PROTOCOL.split("submit_candidate_for_holdout runs")[0].rstrip() + """
RESEARCH EPISODE (Architecture 1). There is no Holdout submission and no playbook editing in an episode: your
procedure (research policy) is versioned by the lab and only changes through the improver plus an admission
comparison. recall_memory returns procedural, episode and evidence memory (failed attempts included).
commit_forecast(hypothesis + hypothesis card) locks the current candidate (immutable record, scored on Discovery)
and ends the episode; after the episode the lab compares procedures and gives the best lock the one Holdout
evaluation. Keep your reasoning short and act through tools."""

REVISION_PROTOCOL = """
OUTPUT: reply with exactly one JSON object (optionally in a ```json fence) and nothing else:
{"parent_version": "<active version>", "target_component": "research_policy" | "improver_policy" | "episode_controls",
 "proposed_by": "improver_<active version>", "change": "<the specific bounded edit>",
 "supporting_episode_ids": ["<episode ids or iter_NNN experiments that show the failure>"],
 "expected_behavior_change": "<observable difference in the next episode's actions or experiments>",
 "development_comparison": "<how the same-start comparison should reveal it>",
 "rejection_condition": "<the result that rejects the change>", "status": "proposed",
 "new_component_text": "<the COMPLETE replacement text of the target component>"}
research_policy is the scientist's playbook for research episodes (the lab re-inserts its LAB-PROTECTED block);
improver_policy is the instruction set you are following now, so revising it changes how the next improvement is
proposed. The lab admits the new version only if, from the same champion and budget, an episode run under it locks a
forecaster that beats the frozen version's episode on Discovery by the fixed margin. Change one thing.
""".strip() + f"""
A text component must be {MIN_COMPONENT_CHARS}..{MAX_COMPONENT_CHARS} characters. episode_controls is the per-episode
tool budget: its new_component_text is a JSON object with integer max_dry_runs, max_backtests and max_analyses, each
inside {CONTROL_BOUNDS}, summing to exactly {CONTROL_TOTAL} (you reallocate the budget, you cannot enlarge it).
Admission is a fixed paired rule: both versions run the same number of episodes, paired in order, and the mean paired
difference of the locked Discovery scores must exceed the margin by more than its standard error. An improver_policy
revision cannot change an episode by itself, so the lab then asks the revised improver for one research_policy or
episode_controls revision and tests and admits the two together. procedural_memory holds the lab-measured episode
scores per procedure version and model class and the errors of earlier invalid revision objects."""

NUDGE_COMMIT = ("You have not called commit_forecast. If a dry-run candidate beats the champion, finish it and call "
                "commit_forecast with the full hypothesis card; otherwise reply with a two-line summary and stop.")


def improver_prompt(active: dict, digest: dict) -> str:
    return "\n\n".join([
        f"Active improvement procedure: {active['version']} (parent {active.get('parent_version')}).",
        "## Current research_policy\n" + active["components"]["research_policy"],
        "## Current improver_policy\n" + active["components"]["improver_policy"],
        "## Released memory (procedural lineage, research episodes, experiment registry, evidence)\n"
        + fit_json(digest, 45000, DIGEST_SHRINK_ORDER),
        "Propose the one revision now (JSON object only).",
    ])


def parse_json_object(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text or "", re.S)
    cand = m.group(1) if m else (text[text.find("{"): text.rfind("}") + 1] if text and "{" in text else "")
    try:
        obj = json.loads(cand)
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


_TRACE_SKIP = {"new_forecaster_code", "analysis_code", "old_str", "new_str", "new_playbook_md", "theory_update_md"}


def traced_tools(tb: AgentToolbox, names: list[str]) -> list:
    """Wrap toolbox methods as Strands tools that append (tool, short args, status) to tb.trace."""
    from strands import tool

    def wrap(n: str):
        fn = getattr(tb, n)

        @functools.wraps(fn)
        def w(*args, **kwargs):
            out = fn(*args, **kwargs)
            status = None
            if isinstance(out, str) and out.startswith("{"):
                try:
                    status = json.loads(out).get("status")
                except (json.JSONDecodeError, AttributeError):
                    status = None
            tb.trace.append({"tool": n, "status": status,
                             "args": {k: v if isinstance(v, (int, float)) else str(v)[:100]
                                      for k, v in kwargs.items() if k not in _TRACE_SKIP}})
            return out
        return tool(w)
    return [wrap(n) for n in names]


DIGEST_SHRINK_ORDER = [("episode_memory", "drop_old"), ("evidence_memory", "drop_old"),
                       ("experiment_registry", "drop_old")]


def compact_episode(e: dict) -> dict:
    """An episode for the memory digest: the ordered tool trace becomes one short string (tool[:status] > ...)."""
    out = {k: v for k, v in e.items() if k != "trace"}
    out["tool_sequence"] = " > ".join(t["tool"] + (f":{t['status']}" if t.get("status") else "")
                                      for t in e.get("trace") or [])[:1500]
    return out


def episode_stats(eps: list[dict], key) -> dict:
    """Measured episode outcomes grouped by key(episode): n, locked-score mean/sd/list and Holdout verdicts. This is
    the lab-computed (never agent-written) record of which procedure versions and model classes worked."""
    out: dict[str, dict] = {}
    for e in eps:
        g = out.setdefault(str(key(e)), {"n": 0, "scores": [], "holdout": []})
        g["n"] += 1
        g["scores"].append(e.get("score") or 0.0)
        if e.get("holdout_result"):
            g["holdout"].append(e["holdout_result"].get("status"))
    for g in out.values():
        m = sum(g["scores"]) / g["n"]
        g["mean_score"] = round(m, 6)
        g["sd_score"] = round((sum((x - m) ** 2 for x in g["scores"]) / (g["n"] - 1)) ** 0.5, 6) if g["n"] > 1 else None
    return out


def _compact_lock(lock: dict | None) -> dict | None:
    if not lock:
        return None
    keep = ("procedure_version", "code_sha256", "code_file", "model_class", "hypothesis_family", "novelty",
            "hypothesis", "hypothesis_card", "status", "error", "screen_pass", "discovery_pooled_delta", "market_insight")
    out = {k: lock.get(k) for k in keep}
    out["discovery_pooled_skill"] = (lock.get("discovery") or {}).get("pooled_skill")
    return out


def card_markdown(card: dict, episode_id: str) -> str:
    return (f"Locked in research episode {episode_id}.\n" +
            "\n".join(f"- **{k.replace('_', ' ')}:** {card.get(k, '')}" for k in HYPOTHESIS_CARD_FIELDS))


def run_generation(ctrl: RSIController, driver, pairs: int | None = None) -> dict:
    """One Architecture-1 generation: improver proposes I_{g+1} -> frozen I_g and candidate I_{g+1} each run `pairs`
    research episodes (RSI_ADMISSION_PAIRS, default ADMISSION_PAIRS; one frozen episode when there is no candidate)
    from the same champion and budget -> fixed paired admission rule -> the best lock gets the single
    Holdout evaluation (Holdout gates decide F_{g+1}) -> episode, procedural and lineage memory are written."""
    t0 = time.time()
    active = ctrl.ensure_procedure()
    g = ctrl.next_generation()
    known = ({str(e["episode_id"]) for e in ctrl.released_episodes()}
             | {f"iter_{int(r['iteration']):03d}" for r in ctrl.experiment_registry() if r.get("iteration") is not None})
    follow_up = None
    try:
        rev, problems = driver.propose_revision(ctrl, active, ctrl.memory_digest(), known)
        if rev and rev["target_component"] == "improver_policy":
            # Episodes never read improver_policy, so the revised improver must show its effect: it proposes one
            # testable revision, and the pair is admitted or rejected together by the same episode comparison.
            follow_up, problems = driver.propose_revision(ctrl, ctrl.procedures.preview(rev), ctrl.memory_digest(), known)
            if follow_up and follow_up["target_component"] == "improver_policy":
                follow_up, problems = None, ["the revised improver_policy must propose a research_policy or "
                                             "episode_controls revision so that its effect can be measured"]
            rev = rev if follow_up else None
    except Exception as e:  # an improver failure still lets the frozen procedure run its episode
        rev, problems = None, [f"{type(e).__name__}: {e}"[:800]]
    cand = ctrl.procedures.add_candidate(rev, follow_up) if rev else None
    ctrl._ledger({"status": "REVISION_PROPOSED" if cand else "REVISION_INVALID", "generation": g,
                  "parent_version": active["version"], "procedure_version": cand and cand["version"],
                  "revision": cand and cand["revision"], "error": "; ".join(problems)[:1500] or None})
    champion_sha = ctrl.champion["code_sha256"]
    n_pairs = max(1, int(pairs if pairs is not None else os.environ.get("RSI_ADMISSION_PAIRS", ADMISSION_PAIRS)))
    episodes, toolboxes = {}, {}
    arms = [("frozen", active)] + ([("candidate", cand)] if cand else [])
    for rep in range(1, (n_pairs if cand else 1) + 1):
        for arm, ver in arms if rep % 2 else arms[::-1]:   # alternate which arm runs first
            eid = f"g{g:03d}_{arm}" + (f"_r{rep}" if rep > 1 else "")
            err, tb = None, None
            try:
                tb = driver.run_episode(ctrl, ver, eid, arm)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:800]
            restore_champion_file(ctrl)
            toolboxes[eid] = tb
            ep = {"episode_id": eid, "generation": g, "arm": arm, "replicate": rep, "procedure_version": ver["version"],
                  "stage": ctrl.cfg.stage, "champion_code_sha256": champion_sha, "error": err,
                  "tool_counts": dict(collections.Counter(t["tool"] for t in (tb.trace if tb else []))),
                  "trace": (tb.trace if tb else [])[:60], "lock": _compact_lock(tb.lock if tb else None)}
            ep["score"] = round(episode_score(ep), 6)
            episodes[eid] = ep
    by_arm = {a: [e for e in episodes.values() if e["arm"] == a] for a, _ in arms}
    admission = None
    if cand:
        admission = admission_rule(by_arm["frozen"], by_arm["candidate"])
        ctrl.procedures.decide(cand["version"], admission)
        if admission["admitted"] and cand["components"]["research_policy"] != active["components"]["research_policy"]:
            ctrl.cfg.prompt_path.write_text(cand["components"]["research_policy"])
        ctrl._ledger({"status": "PROCEDURE_ADMITTED" if admission["admitted"] else "PROCEDURE_REJECTED",
                      "generation": g, "procedure_version": cand["version"], "parent_version": active["version"],
                      "admission": admission})
    best = max((e for e in episodes.values() if episode_score(e) > 0), key=episode_score, default=None)
    holdout = None
    if best:
        lock = toolboxes[best["episode_id"]].lock
        code = (ctrl.locks_dir / lock["code_file"]).read_text()
        theory = card_markdown(lock["hypothesis_card"], best["episode_id"])
        if lock.get("market_insight"):
            theory += "\n\n**Market insight:** " + lock["market_insight"]
        r = ctrl.propose_and_test_forecaster(lock["hypothesis"], theory,
                                             code, lock.get("hypothesis_family") or "", lock.get("novelty") or "",
                                             lock.get("builds_on_iteration"), lock.get("model_class") or "",
                                             hypothesis_card=lock["hypothesis_card"])
        holdout = {k: r.get(k) for k in ("iteration", "status", "holdout", "gate_margins", "error")}
    for e in episodes.values():
        e["holdout_result"] = holdout if best is not None and e is best else None
        append_jsonl(ctrl.episodes_path, e)
    row = ctrl._ledger({"status": "GENERATION", "generation": g, "active_before": active["version"],
                        "candidate_version": cand and cand["version"], "admission": admission,
                        "active_after": ctrl.procedures.active()["version"],
                        "episode_scores": {a: round(sum(e["score"] for e in es) / len(es), 6) for a, es in by_arm.items()},
                        "episode_scores_all": {i: e["score"] for i, e in episodes.items()},
                        "holdout_shot": holdout, "seconds": round(time.time() - t0, 1)})
    ctrl.write_procedure_report()
    return row


DEFAULT_MODAL_LLM_URL = "https://ananthbashyam1--ep-glm-5-3-server.us-west.modal.direct"


def modal_proxy_headers() -> dict:
    """Modal proxy-auth headers for a protected endpoint (MODAL_PROXY_KEY / MODAL_PROXY_SECRET in .env)."""
    k = os.environ.get("MODAL_PROXY_KEY") or os.environ.get("MODAL_PROXY_TOKEN_ID")
    s = os.environ.get("MODAL_PROXY_SECRET") or os.environ.get("MODAL_PROXY_TOKEN_SECRET")
    if not (k and s):
        raise RuntimeError("set MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET (a Modal proxy auth token) in .env")
    return {"Modal-Key": k, "Modal-Secret": s}


def wait_for_health(base: str, key: str, headers: dict, timeout_s: int = 1500) -> None:
    """Block until a (possibly cold-starting) Modal endpoint answers /v1/models."""
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(base + "/v1/models", headers={**headers, "Authorization": f"Bearer {key}"}, timeout=120)
            if r.status_code == 200:
                return
            if r.status_code in (401, 403):
                raise PermissionError(f"LLM endpoint rejected credentials ({r.status_code}): {r.text[:200]}")
        except httpx.HTTPError:
            pass
        print("[modal] waiting for the LLM endpoint to warm up...", file=sys.stderr)
        time.sleep(20)
    raise TimeoutError(f"LLM endpoint {base} not healthy after {timeout_s}s")


def discover_model_id(base: str, key: str, headers: dict) -> str:
    import httpx

    r = httpx.get(base + "/v1/models", headers={**headers, "Authorization": f"Bearer {key}"}, timeout=120)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'"))


STAGE_CHOICES = ["pack_only", "pack_plus_exa", "long_horizon"]


def stage_config(stage: str, exa_cache_only: bool = False) -> LabConfig:
    """CLI stage -> config; the long-horizon track always has the point-in-time Exa gateway on."""
    if stage == "long_horizon":
        return load_config(track="long", enable_exa=True, exa_cache_only=exa_cache_only)
    return load_config(enable_exa=stage == "pack_plus_exa", exa_cache_only=exa_cache_only)


def make_driver(name: str, candidates: list[str] | None = None):
    if name == "scripted":
        return ScriptedDriver(candidates or [])
    if name == "api":
        return DevinAPIDriver()
    if name == "strands":
        return StrandsAgentDriver()
    if name == "modal":
        return StrandsAgentDriver(provider="modal")
    raise ValueError(f"unknown driver {name!r} (api, strands, modal, scripted)")


# ================================================================================ ablation ====
def write_ablation_summary(cfg: LabConfig) -> Path:
    import pandas as pd

    rows = []
    for r in RSIController(cfg, use_git=False).ledger_entries():
        if r.get("status") not in EXPERIMENT_STATUSES + ("SEED", "FORK"):
            continue
        d, h, b = r.get("discovery") or {}, r.get("holdout") or {}, r.get("benchmark_2026_10_24") or {}
        rows.append({
            "stage": r["stage"], "iteration": r.get("iteration"), "status": r["status"], "driver": r.get("driver"),
            "hypothesis": (r.get("hypothesis") or "")[:160], "hypothesis_family": r.get("hypothesis_family"),
            "novelty": r.get("novelty"), "no_op": bool(r.get("no_op")), "discovery_pooled_skill": d.get("pooled_skill"),
            "discovery_coverage_80": d.get("coverage_80"), "holdout_pooled_skill": h.get("pooled_skill"),
            "holdout_median_skill": h.get("median_skill"), "holdout_coverage_80": h.get("coverage_80"),
            **{f"holdout_skill_h{k}": v for k, v in (h.get("skill_by_horizon") or {}).items()},
            "skill_2026_10_24": b.get("pooled_skill"), "coverage_2026_10_24": b.get("coverage_80"),
            "gates_passed": (r.get("gates") or {}).get("all_pass"), "code_sha256": (r.get("code_sha256") or "")[:12],
            "git_commit": (r.get("git_commit") or "")[:10], "ts_utc": r.get("ts_utc"),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["is_stage_champion"] = False
        for st in df["stage"].unique():
            p = cfg.outputs_dir / "champions" / st / "champion_state.json"
            if p.exists():
                sha = json.loads(p.read_text())["code_sha256"][:12]
                m = (df["stage"] == st) & (df["code_sha256"] == sha) & df["status"].isin(["PROMOTED", "SEED", "FORK"])
                df.loc[m[m].index[-1:] if m.any() else [], "is_stage_champion"] = True
    out = cfg.outputs_dir / "ablation_summary.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    return out


# ===================================================================================== CLI ====
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run RSI iterations")
    run.add_argument("--stage", choices=STAGE_CHOICES, default="pack_only")
    run.add_argument("--driver", choices=["api", "strands", "modal", "scripted"], default=os.environ.get("DEVIN_MODE", "modal"))
    run.add_argument("--iterations", type=int, default=1)
    run.add_argument("--candidates", nargs="*", help="forecaster files for the scripted driver")
    run.add_argument("--exa-cache-only", action="store_true")
    fl = sub.add_parser("fork-long", help="start LONG_HORIZON (30/90/180-day AWS track) from the PACK_ONLY champion")
    fl.add_argument("--exa-cache-only", action="store_true")
    fork = sub.add_parser("fork-stage2", help="start PACK_PLUS_EXA from the PACK_ONLY champion")
    fork.add_argument("--exa-cache-only", action="store_true")
    gen = sub.add_parser("generations", help="Architecture-1 generations: improver -> same-start procedure comparison "
                                             "-> admission -> one Holdout shot")
    gen.add_argument("--stage", choices=STAGE_CHOICES, default="pack_only")
    gen.add_argument("--driver", choices=["strands", "modal"], default="modal")
    gen.add_argument("--generations", type=int, default=1)
    gen.add_argument("--exa-cache-only", action="store_true")
    sub.add_parser("status")
    sub.add_parser("ablation")
    a = ap.parse_args()
    load_dotenv(load_config().root / ".env")

    if a.cmd == "ablation":
        print(write_ablation_summary(load_config()))
        return 0
    if a.cmd == "status":
        for st in (STAGE_PACK_ONLY, STAGE_PACK_PLUS_EXA):
            c = RSIController(load_config(enable_exa=st == STAGE_PACK_PLUS_EXA), use_git=False).champion
            print(st, json.dumps({k: c[k] for k in ("iteration", "holdout", "benchmark_2026_10_24", "git_commit")}
                                 if c else None, default=str))
        return 0
    if a.cmd == "fork-long":
        st = RSIController(stage_config("long_horizon", a.exa_cache_only), driver_name="lab").fork_from(
            STAGE_PACK_ONLY, Path(__file__).resolve().parent / "seeds" / "long_horizon.py")
        print(json.dumps({"stage": STAGE_LONG_HORIZON, "discovery": (st.get("discovery") or {}).get("pooled_skill"),
                          "holdout": st.get("holdout")}, default=str))
        return 0
    if a.cmd == "fork-stage2":
        cfg = load_config(enable_exa=True, exa_cache_only=a.exa_cache_only)
        RSIController(load_config(), driver_name="lab").checkpoint_notes("notebook checkpoint before Stage 2 fork")
        st = RSIController(cfg, driver_name="lab").fork_from(STAGE_PACK_ONLY)
        print(json.dumps({"forked": True, "holdout": st["holdout"]}, default=str))
        return 0

    if a.cmd == "generations":
        cfg = stage_config(a.stage, a.exa_cache_only)
        driver = make_driver(a.driver)
        ctrl = RSIController(cfg, driver_name=driver.name)
        ctrl.bootstrap()
        ctrl.ensure_procedure()
        n_ok = 0
        for _ in range(a.generations):
            try:
                print(json.dumps(run_generation(ctrl, driver), default=str)[:4000])
                n_ok += 1
            except Exception as e:
                ctrl._ledger({"status": "DRIVER_ERROR", "generation": ctrl.next_generation(), "counts_as_experiment": False,
                              "error": f"{type(e).__name__}: {e}"[:1500]})
                print(f"[rsi] generation driver error: {type(e).__name__}: {e}", file=sys.stderr)
                restore_champion_file(ctrl)
        ctrl.checkpoint_notes(f"notebook checkpoint after {a.generations} generation(s)")
        write_ablation_summary(cfg)
        return 0 if n_ok else 1

    cfg = stage_config(a.stage, a.exa_cache_only)
    driver = make_driver(a.driver, a.candidates)
    ctrl = RSIController(cfg, driver_name=driver.name)
    ctrl.bootstrap()
    ctrl._ensure_protected_playbook()
    n = len(a.candidates) if a.driver == "scripted" and a.candidates else a.iterations
    ctrl.iteration_budget = {"planned": n, "first_iteration": ctrl.next_iteration()}
    first = ctrl.next_iteration()
    n_errors = run_iterations(ctrl, driver, n)
    ctrl.checkpoint_notes(f"notebook checkpoint after {n} iteration(s)")
    write_ablation_summary(cfg)
    return 1 if n and n_errors and ctrl.next_iteration() == first else 0


def run_iterations(ctrl: RSIController, driver, n: int, max_consecutive_errors: int | None = None) -> int:
    """Run n driver iterations; returns the number of DRIVER_ERRORs. A failed driver call is logged (a DRIVER_ERROR is
    not an experiment, so it carries the iteration number the next experiment will use plus a per-run attempt index),
    the champion file is restored, and the loop stops after `max_consecutive_errors` errors in a row."""
    if max_consecutive_errors is None:
        max_consecutive_errors = int(os.environ.get("RSI_MAX_CONSECUTIVE_DRIVER_ERRORS", "3"))
    n_errors = consecutive = 0
    for i in range(n):
        try:
            r = driver.run_iteration(ctrl, i)
            consecutive = 0
            print(json.dumps({k: r.get(k) for k in ("iteration", "status", "hypothesis", "holdout", "error")}, default=str))
        except Exception as e:
            n_errors += 1
            consecutive += 1
            ctrl._ledger({"status": "DRIVER_ERROR", "iteration": ctrl.next_iteration(), "attempt": i + 1,
                          "counts_as_experiment": False, "error": f"{type(e).__name__}: {e}"[:1500]})
            print(f"[rsi] iteration {i + 1} driver error: {type(e).__name__}: {e}", file=sys.stderr)
            restore_champion_file(ctrl)
            if max_consecutive_errors and consecutive >= max_consecutive_errors:
                print(f"[rsi] {consecutive} consecutive driver errors; stopping the run", file=sys.stderr)
                break
    return n_errors


def restore_champion_file(ctrl: RSIController) -> bool:
    """Put the champion back into workspace/forecaster.py if it was changed or deleted. Returns True if restored."""
    f = ctrl.cfg.forecaster_path
    if not ctrl.backup_path.exists():
        return False
    if not f.exists() or f.read_bytes() != ctrl.backup_path.read_bytes():
        ctrl._restore_champion()
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
