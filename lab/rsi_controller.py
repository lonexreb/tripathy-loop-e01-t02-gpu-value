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
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from lab.config import (DISCOVERY_ORIGINS, HOLDOUT_ORIGINS, STAGE_PACK_ONLY, STAGE_PACK_PLUS_EXA,
                        LabConfig, load_config)
from lab.memory import (HYPOTHESIS_FAMILIES, NOVELTY_LEVELS, PROTECTED_BLOCK, archive_candidate, collapse_warnings,
                        demote_headings, diff_stats, dropped_guardrails, enforce_protected, exploration_directive,
                        family_stats, fit_json, gate_margins, is_no_op, metric_delta, near_misses, normalize_family,
                        normalize_novelty, notebook_index, registry_row, unified_diff)
from lab.sandbox import audit_forecaster_source
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
    return {"gate1": f"holdout pooled > max({g.pooled_floor:.3f}, champion + {g.pooled_min_gain:.3f})",
            "gate2": f"holdout median >= champion median - {g.median_tolerance:.3f}",
            "gate3": f"{g.coverage_hard_low:.2f} <= holdout coverage_80 <= {g.coverage_hard_high:.2f} and either "
                     f"{g.coverage_low:.2f} <= coverage_80 <= {g.coverage_high:.2f} or |coverage_80 - {g.coverage_target:.2f}| "
                     f"<= champion's distance + {g.coverage_tolerance:.3f}",
            "gate4": f"every horizon {tuple(cfg.horizons)} skill >= {g.horizon_floor:.3f}",
            "screen": f"before Holdout: finite quantiles within {math.exp(g.sanity_max_abs_log_ratio):.0f}x of the last value, "
                      f"<= {g.sanity_max_invalid_share:.0%} invalid/missing cells, and Discovery pooled > champion "
                      f"Discovery pooled + {g.discovery_min_gain:.4f}"}


def evaluate_gates(holdout: dict, champion_holdout: dict | None, cfg: LabConfig) -> dict:
    """The four promotion gates, evaluated strictly on Holdout aggregates (missing / non-finite values fail)."""
    g, ch = cfg.gates, champion_holdout or {}
    champ_pooled = _num(ch.get("pooled_skill"))
    champ_median = _num(ch.get("median_skill"))
    champ_cov = _num(ch.get("coverage_80"))
    need = max(g.pooled_floor, (champ_pooled if champ_pooled is not None else -1e9) + g.pooled_min_gain)
    med_need = (champ_median if champ_median is not None else -1e9) - g.median_tolerance
    pooled, median, cov = _num(holdout.get("pooled_skill")), _num(holdout.get("median_skill")), _num(holdout.get("coverage_80"))
    by_h = {str(k): _num(v) for k, v in (holdout.get("skill_by_horizon") or {}).items()}
    in_band = cov is not None and g.coverage_low <= cov <= g.coverage_high
    in_hard = cov is not None and g.coverage_hard_low <= cov <= g.coverage_hard_high
    allowed_dist = None if champ_cov is None else abs(champ_cov - g.coverage_target) + g.coverage_tolerance
    rel_ok = cov is not None and allowed_dist is not None and abs(cov - g.coverage_target) <= allowed_dist + 1e-12
    out = {
        "gate1_pooled_gain": {"value": _r(pooled), "threshold": _r(need), "pass": pooled is not None and pooled > need},
        "gate2_breadth": {"value": _r(median), "threshold": _r(med_need), "pass": median is not None and median >= med_need},
        "gate3_calibration": {"value": _r(cov), "band": [g.coverage_low, g.coverage_high],
                              "hard_band": [g.coverage_hard_low, g.coverage_hard_high],
                              "max_distance_to_target": _r(allowed_dist), "in_band": in_band,
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
    invalid = max(int(sc.get("n_cells_filled_from_baseline") or 0) / n_cells,
                  (int(san.get("n_crossing_rows") or 0) + int(san.get("n_nonpositive_rows") or 0)) / n_rows)
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


def compare_discovery(cand: dict, champ: dict | None) -> dict:
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
            env = {**os.environ, "PYTHONPATH": str(self.root), "PYTHONDONTWRITEBYTECODE": "1", "LAB_ROOT": str(self.root)}
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
                 "discovery": res["discovery"], "holdout": res["holdout"], "benchmark_2026_10_24": res["benchmark_2026_10_24"],
                 "since": utc_now()}
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
        ledger = [{k: r.get(k) for k in ("iteration", "status", "hypothesis_family", "novelty", "hypothesis", "discovery",
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
                                    builds_on_iteration: int | None = None) -> dict:
        """Two-stage test: audit -> duplicate check -> Discovery (sanity + improvement screen) -> Holdout gates."""
        t0 = time.time()
        champ = self.bootstrap()
        self._check_lab_intact()
        it = self.next_iteration()
        code_sha = sha256_text(new_forecaster_code)
        champion_code = self.backup_path.read_text()
        cand_dir = self.candidates_dir / f"iter_{it:03d}"
        base = {"iteration": it, "hypothesis": hypothesis[:2000], "hypothesis_family": normalize_family(hypothesis_family),
                "novelty": normalize_novelty(novelty), "code_sha256": code_sha, "parent_code_sha256": champ["code_sha256"],
                "candidate_dir": str(cand_dir.relative_to(self.root)),
                "diff_vs_champion_lines": diff_stats(unified_diff(champion_code, new_forecaster_code))}
        if builds_on_iteration is not None:
            base["builds_on_iteration"] = int(builds_on_iteration)
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
        vs = compare_discovery(disc, champ.get("discovery"))
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
        gates = evaluate_gates(res["holdout"], champ["holdout"], self.cfg)
        if not res.get("holdout_sanity_ok", True):
            gates["holdout_sanity"] = {"pass": False}
            gates["all_pass"] = False
        rec.update(holdout=res["holdout"], benchmark_2026_10_24=res["benchmark_2026_10_24"], gates=gates,
                   gate_margins=gate_margins(gates),
                   holdout_delta_vs_champion=metric_delta(res["holdout"], champ["holdout"]),
                   champion_holdout_before=champ["holdout"])
        if is_no_op(res["holdout"], champ["holdout"]):
            rec["no_op"] = True
        if gates["all_pass"]:
            self._append_theory(it, hypothesis, theory_update_md, "PROMOTED", gates, family=base["hypothesis_family"])
            shutil.copyfile(self.cfg.forecaster_path, self.backup_path)
            commit = self.git_commit(f"[{self.cfg.stage}] iter {it} PROMOTED: {hypothesis[:60]}")
            state = {"stage": self.cfg.stage, "iteration": it, "code_sha256": code_sha, "git_commit": commit,
                     "discovery": res["discovery"], "holdout": res["holdout"],
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
        vs = compare_discovery(d, cd)
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
        res = self._run_lab_child(["lab.workbench", "analyze", "--code", str(self._agent_file("analysis", code)),
                                   "--origins", ",".join(origins)], ANALYSIS_TIMEOUT_S)
        it = self.next_iteration() if iteration is None else iteration
        self._ledger({"status": "ANALYSIS", "iteration": it, "code_sha256": sha256_text(code), "origins": origins,
                      "ok": bool(res.get("ok")), "error": res.get("error")})
        if not res.get("ok"):
            return {"status": "ANALYSIS_ERROR", "error": res.get("error"), "traceback_tail": res.get("traceback_tail")}
        return {"status": "ANALYSIS_OK", "results": res["results"]}

    def search_exa_signals(self, query: str, origin_date: str) -> list[dict]:
        """Agent-side research search; pinned at or before the last Discovery origin so Holdout news never leaks."""
        if not self.cfg.enable_exa:
            raise PermissionError("search_exa_signals is only available in PACK_PLUS_EXA")
        if origin_date > DISCOVERY_ORIGINS[-1]:
            raise PermissionError(f"agent research searches must use origin_date <= {DISCOVERY_ORIGINS[-1]} (Discovery)")
        from lab.sandbox import PointInTimeSandbox
        return PointInTimeSandbox(self.cfg).search_exa_point_in_time(query, origin_date)

    # ----------------------------------------------------------------------- stage control ----
    def checkpoint_notes(self, message: str) -> str | None:
        return self.git_commit(f"[{self.cfg.stage}] {message}")

    def fork_from(self, parent_stage: str) -> dict:
        """Start this stage from another stage's champion (Stage 2 forks from the Stage 1 champion)."""
        parent_dir = self.cfg.outputs_dir / "champions" / parent_stage
        parent = json.loads((parent_dir / "champion_state.json").read_text())
        self.champ_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent_dir / "forecaster.py", self.cfg.forecaster_path)
        shutil.copyfile(parent_dir / "forecaster.py", self.backup_path)
        if self.state_path.exists():
            self.state_path.unlink()
        if self.use_git:
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
  statistic, gpu_model, pricing_tier, instance_type, az_global_id, members), daily_statistics (p25/median/p75/min per
  collector), aggregator_medians (weekly, complete weeks only), index_prints, hardware_fundamentals,
  capability_ladder, capability_units. external_signals is None in PACK_ONLY; in PACK_PLUS_EXA it is an object with
  .origin_date and .search(query, domains=None, num_results=8) -> list of {{title, url, published_date, highlights}},
  point-in-time (published on or before origin_date) and cached.
  Allowed imports: numpy, pandas and the Python standard-library modules math, statistics, dataclasses, typing,
  collections, functools, itertools, re, datetime, warnings, bisect, heapq, operator, enum. No file or network I/O,
  no open/eval/exec, no pandas read_*/to_* calls: the lab rejects such code (REJECTED_FIREWALL).
  Keep the run fast (< 20 s per origin).

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
Tag each proposal with hypothesis_family (one of: {", ".join(HYPOTHESIS_FAMILIES)}) and novelty
({" | ".join(NOVELTY_LEVELS)}); set builds_on_iteration when you start from an archived candidate instead of the champion.
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
                                                p.get("builds_on_iteration"))


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
                                               so.get("builds_on_iteration"))
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
        from strands import Agent, tool
        from strands.types.exceptions import MaxTokensReachedException

        tb = AgentToolbox(ctrl)
        tools = [tool(getattr(tb, n)) for n in AgentToolbox.TOOL_NAMES]
        if ctrl.cfg.enable_exa:
            tools.append(tool(tb.search_exa_signals))
        ctrl._ensure_protected_playbook()
        playbook = ctrl.cfg.prompt_path.read_text()
        directive = exploration_directive(ctrl.experiment_registry(), ctrl.budget())
        agent = Agent(model=self.model, tools=tools, system_prompt=AGENT_CONTRACT + "\n\n" + WORKBENCH_PROTOCOL + "\n\n"
                      + playbook, callback_handler=None)
        prompt = (f"Iteration {tb.iteration} of stage {ctrl.cfg.stage}. Lab directive: {directive}\n"
                  "Call inspect_current_state and diagnose the Discovery errors (use inspect_candidate on near-misses "
                  "or on the experiments your idea relates to). Then edit the candidate with small edit_candidate "
                  f"changes (or start from a near-miss with load_candidate), dry-run it on Discovery (at most "
                  f"{tb.max_dry_runs} dry runs), submit_candidate_for_holdout exactly once with one falsifiable "
                  "hypothesis, its family and novelty, and finally call update_scientist_prompt with your revised "
                  "playbook (keep the lab-protected block; record what you learned and rank untried directions).")
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
(exact, unique old_str -> new_str; keep each edit small) instead of re-writing the whole file; write_candidate replaces
the whole file and is only for large rewrites. dry_run_discovery scores the current candidate on the Discovery origins
only (full diagnostics, paired deltas vs the champion and the pre-Holdout screen verdict `would_reach_holdout`;
Holdout untouched, champion untouched, nothing promoted). A submission whose Discovery pooled skill does not beat the
champion's, or whose outputs fail the sanity screen, is rejected before Holdout, so dry-run until would_reach_holdout
is true before submitting.
analyze_discovery runs your own `def analyze(pack_slice, origin_date)` (same import firewall as the forecaster, must
return JSON-able data) on point-in-time Discovery slices. submit_candidate_for_holdout runs the one Holdout evaluation
of this iteration (promote or revert); it can be called once. Keep your reasoning short and act through tools.
""".strip()

CONTINUE_AFTER_MAX_TOKENS = ("Your previous response hit the output-token limit before a tool call was made. Do not "
                             "restate your analysis. Make the next tool call now; prefer small edit_candidate edits over "
                             "write_candidate, then dry_run_discovery or submit_candidate_for_holdout.")
NUDGE_SUBMIT = ("You have not called submit_candidate_for_holdout yet. Finish the candidate with edit_candidate if needed, "
                "call submit_candidate_for_holdout exactly once, then update_scientist_prompt.")


class AgentToolbox:
    """The in-iteration tools of the Strands driver as plain methods (wrapped with strands.tool by the driver)."""

    TOOL_NAMES = ("inspect_current_state", "inspect_candidate", "view_candidate", "edit_candidate", "write_candidate",
                  "load_candidate", "show_candidate_diff", "dry_run_discovery", "analyze_discovery",
                  "submit_candidate_for_holdout", "propose_and_test_forecaster", "update_scientist_prompt")

    def __init__(self, ctrl: RSIController, max_dry_runs: int | None = None, max_analyses: int | None = None,
                 max_submissions: int = 1):
        self.ctrl = ctrl
        self.iteration = ctrl.next_iteration()
        ctrl.bootstrap()
        self.wb = CandidateWorkbench(ctrl.backup_path.read_text())
        self.max_dry_runs = (max_dry_runs if max_dry_runs is not None
                             else int(os.environ.get("RSI_MAX_DRY_RUNS", ctrl.cfg.max_discovery_probes)))
        self.max_analyses = max_analyses if max_analyses is not None else int(os.environ.get("RSI_MAX_ANALYSES", "6"))
        self.max_submissions = max_submissions
        self.n_dry_runs = self.n_analyses = 0
        self.builds_on: int | None = None
        self.results: list[dict] = []

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

    def view_candidate(self, start_line: int = 1, end_line: int = 0) -> str:
        """Show the current candidate forecaster with line numbers (end_line 0 = to the end)."""
        return self.wb.view(start_line, end_line or None)

    def edit_candidate(self, old_str: str, new_str: str) -> str:
        """Replace one exact, unique snippet old_str of the candidate with new_str. Returns status + firewall audit."""
        return json.dumps(self.wb.replace(old_str, new_str))

    def write_candidate(self, new_forecaster_code: str) -> str:
        """Replace the whole candidate forecaster (only for large rewrites; prefer edit_candidate)."""
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
        return clip_json({**r, "dry_runs_left": self.max_dry_runs - self.n_dry_runs}, 30000)

    def analyze_discovery(self, analysis_code: str, origins: list[str]) -> str:
        """Run `def analyze(pack_slice, origin_date)` from analysis_code on point-in-time slices of the given Discovery
        origins (forecaster import firewall; return small JSON-able summaries, DataFrames are capped at 200 rows)."""
        if self.n_analyses >= self.max_analyses:
            return json.dumps({"status": "ANALYSIS_BUDGET_EXHAUSTED", "error": f"{self.max_analyses} analyses used"})
        self.n_analyses += 1
        return clip_json(self.ctrl.run_discovery_analysis(analysis_code, origins, iteration=self.iteration), 20000)

    def submit_candidate_for_holdout(self, hypothesis: str, theory_update_md: str, hypothesis_family: str = "",
                                     novelty: str = "", builds_on_iteration: int = -1) -> str:
        """Evaluate the current candidate: pre-Holdout screen on Discovery, then the 4 Holdout gates; promote or revert.
        Allowed once per iteration. hypothesis_family is one of the lab's families (see inspect_current_state),
        novelty is incremental or bold, builds_on_iteration is the archived candidate you started from (-1 = the
        champion). The candidate is archived whatever the verdict."""
        return self._submit(hypothesis, theory_update_md, self.wb.code, hypothesis_family, novelty, builds_on_iteration)

    def propose_and_test_forecaster(self, hypothesis: str, theory_update_md: str, new_forecaster_code: str,
                                    hypothesis_family: str = "", novelty: str = "", builds_on_iteration: int = -1) -> str:
        """Legacy one-shot tool: replace the whole candidate with new_forecaster_code and submit it for Holdout.
        Counts as this iteration's single submission."""
        self.wb.write(new_forecaster_code)
        return self._submit(hypothesis, theory_update_md, new_forecaster_code, hypothesis_family, novelty,
                            builds_on_iteration)

    def _submit(self, hypothesis: str, theory_update_md: str, code: str, hypothesis_family: str = "",
                novelty: str = "", builds_on_iteration: int | None = -1) -> str:
        if len(self.results) >= self.max_submissions:
            return json.dumps({"status": "SUBMISSION_BUDGET_EXHAUSTED",
                               "error": "this iteration's Holdout submission was already used; call update_scientist_prompt"})
        if code == self.wb.base_code:
            return json.dumps({"status": "SUBMISSION_REJECTED", "error": "candidate is identical to the champion"})
        if builds_on_iteration is None or int(builds_on_iteration) < 0:
            builds_on_iteration = self.builds_on
        r = self.ctrl.propose_and_test_forecaster(hypothesis, theory_update_md, code, hypothesis_family, novelty,
                                                  builds_on_iteration)
        self.results.append(r)
        return fit_json(r, 30000, [("discovery_diagnostics", "head"), ("champion_holdout_before", "head")])

    def update_scientist_prompt(self, new_playbook_md: str, rationale: str) -> str:
        """Replace workspace/scientist_prompt.md (your research playbook) with a revised version."""
        return json.dumps(self.ctrl.update_scientist_prompt(new_playbook_md, rationale, iteration=self.iteration), default=str)

    def search_exa_signals(self, query: str, origin_date: str) -> str:
        """Point-in-time web search (results published on or before origin_date <= the last Discovery origin)."""
        return clip_json(self.ctrl.search_exa_signals(query, origin_date), 20000)


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
    run.add_argument("--stage", choices=["pack_only", "pack_plus_exa"], default="pack_only")
    run.add_argument("--driver", choices=["api", "strands", "modal", "scripted"], default=os.environ.get("DEVIN_MODE", "modal"))
    run.add_argument("--iterations", type=int, default=1)
    run.add_argument("--candidates", nargs="*", help="forecaster files for the scripted driver")
    run.add_argument("--exa-cache-only", action="store_true")
    fork = sub.add_parser("fork-stage2", help="start PACK_PLUS_EXA from the PACK_ONLY champion")
    fork.add_argument("--exa-cache-only", action="store_true")
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
    if a.cmd == "fork-stage2":
        cfg = load_config(enable_exa=True, exa_cache_only=a.exa_cache_only)
        RSIController(load_config(), driver_name="lab").checkpoint_notes("notebook checkpoint before Stage 2 fork")
        st = RSIController(cfg, driver_name="lab").fork_from(STAGE_PACK_ONLY)
        print(json.dumps({"forked": True, "holdout": st["holdout"]}, default=str))
        return 0

    cfg = load_config(enable_exa=a.stage == "pack_plus_exa", exa_cache_only=a.exa_cache_only)
    driver = make_driver(a.driver, a.candidates)
    ctrl = RSIController(cfg, driver_name=driver.name)
    ctrl.bootstrap()
    ctrl._ensure_protected_playbook()
    n = len(a.candidates) if a.driver == "scripted" and a.candidates else a.iterations
    ctrl.iteration_budget = {"planned": n, "first_iteration": ctrl.next_iteration()}
    n_errors = run_iterations(ctrl, driver, n)
    ctrl.checkpoint_notes(f"notebook checkpoint after {n} iteration(s)")
    write_ablation_summary(cfg)
    return 1 if n and n_errors >= n else 0


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
