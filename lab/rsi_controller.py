"""Dual-layer RSI orchestrator: Devin API / Strands drivers, 4-gate promotion and a git commit/revert state machine.

The agent owns workspace/ (forecaster.py, scientist_prompt.md, theory_of_market.md) and talks to the lab only
through four tools: inspect_current_state, propose_and_test_forecaster, update_scientist_prompt and, in
PACK_PLUS_EXA only, search_exa_signals. Candidates are audited (no I/O, no network), evaluated in a separate
process with a timeout, scored with full diagnostics on Discovery and only the four gate aggregates on Holdout,
then PROMOTED (champion overwritten, workspace committed) or REVERTED / RUNTIME_ERROR (forecaster.py restored
from the champion backup; the playbook and the notebook keep the agent's notes so negative results persist).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from lab.config import (BENCHMARK_ORIGIN, DISCOVERY_ORIGINS, HOLDOUT_ORIGINS, STAGE_PACK_ONLY, STAGE_PACK_PLUS_EXA,
                        LabConfig, load_config)
from lab.sandbox import audit_forecaster_source

WORKSPACE_FILES = ("workspace/forecaster.py", "workspace/scientist_prompt.md", "workspace/theory_of_market.md")
EXPERIMENT_STATUSES = ("PROMOTED", "REVERTED", "RUNTIME_ERROR", "REJECTED_FIREWALL")
GIT_IDENTITY = ["-c", "user.name=AI Scientist Lab", "-c", "user.email=ai-scientist-lab@localhost"]
MAX_PLAYBOOK_CHARS = 24_000
MAX_THEORY_UPDATE_CHARS = 6_000


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def evaluate_gates(holdout: dict, champion_holdout: dict | None, cfg: LabConfig) -> dict:
    """The four promotion gates, evaluated strictly on Holdout aggregates."""
    g, ch = cfg.gates, champion_holdout or {}
    champ_pooled = ch.get("pooled_skill") if ch.get("pooled_skill") is not None else -1e9
    champ_median = ch.get("median_skill") if ch.get("median_skill") is not None else -1e9
    need = max(g.pooled_floor, champ_pooled + g.pooled_min_gain)
    by_h = holdout["skill_by_horizon"]
    out = {
        "gate1_pooled_gain": {"value": _r(holdout["pooled_skill"]), "threshold": _r(need),
                              "pass": holdout["pooled_skill"] > need},
        "gate2_breadth": {"value": _r(holdout["median_skill"]), "threshold": _r(champ_median - g.median_tolerance),
                          "pass": holdout["median_skill"] >= champ_median - g.median_tolerance},
        "gate3_calibration": {"value": _r(holdout["coverage_80"]), "band": [g.coverage_low, g.coverage_high],
                              "pass": g.coverage_low <= holdout["coverage_80"] <= g.coverage_high},
        "gate4_horizon_floor": {"value": {h: _r(v) for h, v in by_h.items()}, "threshold": g.horizon_floor,
                                "pass": all(by_h.get(str(h)) is not None and by_h[str(h)] >= g.horizon_floor
                                            for h in cfg.horizons)},
    }
    out["all_pass"] = all(v["pass"] for v in out.values())
    return out


class RSIController:
    def __init__(self, cfg: LabConfig | None = None, driver_name: str = "manual", use_git: bool = True):
        self.cfg = cfg or load_config()
        self.root = Path(self.cfg.root)
        self.driver_name = driver_name
        self.use_git = use_git and (self.root / ".git").exists()
        self.champ_dir = self.cfg.outputs_dir / "champions" / self.cfg.stage
        self.state_path = self.champ_dir / "champion_state.json"
        self.backup_path = self.champ_dir / "forecaster.py"
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
    def run_evaluation(self, forecaster_path: Path) -> dict:
        """Evaluate in a child process (timeout; the controller only ever receives the firewalled JSON)."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "result.json"
            cmd = [sys.executable, "-B", "-m", "lab.evaluator", "--forecaster", str(forecaster_path), "--out", str(out)]
            if self.cfg.enable_exa:
                cmd.append("--enable-exa")
            if self.cfg.exa_cache_only:
                cmd.append("--exa-cache-only")
            env = {**os.environ, "PYTHONPATH": str(self.root), "PYTHONDONTWRITEBYTECODE": "1", "LAB_ROOT": str(self.root)}
            if self.cfg.cache_dir:
                env["LAB_CACHE_DIR"] = str(self.cfg.cache_dir)
            try:
                subprocess.run(cmd, cwd=self.root, env=env, capture_output=True, text=True, timeout=self.cfg.eval_timeout_s)
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"evaluation timed out after {self.cfg.eval_timeout_s}s"}
            if not out.exists():
                return {"ok": False, "error": "evaluator produced no result (crashed before reporting)"}
            return json.loads(out.read_text())

    @staticmethod
    def _discovery_summary(d: dict) -> dict:
        keys = ("pooled_skill", "median_skill", "coverage_80", "skill_by_horizon", "q10_breach_rate", "q90_breach_rate")
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
    def inspect_current_state(self, ledger_tail: int = 12) -> dict:
        champ = self.bootstrap()
        ledger = [{k: r.get(k) for k in ("iteration", "status", "hypothesis", "discovery", "holdout", "gates",
                                          "benchmark_2026_10_24", "error")}
                  for r in self.ledger_entries(self.cfg.stage) if r.get("status") in EXPERIMENT_STATUSES + ("SEED", "FORK")]
        theory = self.cfg.theory_path.read_text() if self.cfg.theory_path.exists() else ""
        return {
            "stage": self.cfg.stage, "enable_exa": self.cfg.enable_exa, "next_iteration": self.next_iteration(),
            "scientist_prompt_md": self.cfg.prompt_path.read_text() if self.cfg.prompt_path.exists() else "",
            "theory_of_market_tail_md": theory[-8000:],
            "champion_forecaster_py": self.backup_path.read_text(),
            "champion_discovery_diagnostics": champ["discovery"],
            "champion_holdout_aggregates": champ["holdout"],
            "promotion_gates": {"gate1": "holdout pooled > max(0.200, champion + 0.003)",
                                "gate2": "holdout median >= champion median - 0.005",
                                "gate3": "0.74 <= holdout coverage_80 <= 0.86",
                                "gate4": "every horizon (1, 3, 7) skill >= -0.020"},
            "ledger_tail": ledger[-ledger_tail:],
        }

    def propose_and_test_forecaster(self, hypothesis: str, theory_update_md: str, new_forecaster_code: str) -> dict:
        t0 = time.time()
        champ = self.bootstrap()
        self._check_lab_intact()
        it = self.next_iteration()
        code_sha = sha256_text(new_forecaster_code)
        base = {"iteration": it, "hypothesis": hypothesis[:2000], "code_sha256": code_sha,
                "parent_code_sha256": champ["code_sha256"]}
        problems = audit_forecaster_source(new_forecaster_code)
        if problems:
            status = "RUNTIME_ERROR" if any(p.startswith("SyntaxError") for p in problems) else "REJECTED_FIREWALL"
            return self._finish(base, status, theory_update_md, t0, error="; ".join(problems)[:2000])
        self.cfg.forecaster_path.write_text(new_forecaster_code)
        res = self.run_evaluation(self.cfg.forecaster_path)
        if not res.get("ok"):
            self._restore_champion()
            return self._finish(base, "RUNTIME_ERROR", theory_update_md, t0, error=res.get("error"),
                                traceback_tail=res.get("traceback_tail"))
        gates = evaluate_gates(res["holdout"], champ["holdout"], self.cfg)
        rec = {**base, "discovery": self._discovery_summary(res["discovery"]), "holdout": res["holdout"],
               "benchmark_2026_10_24": res["benchmark_2026_10_24"], "gates": gates,
               "champion_holdout_before": champ["holdout"]}
        if gates["all_pass"]:
            self._append_theory(it, hypothesis, theory_update_md, "PROMOTED", gates)
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
            self._append_theory(it, hypothesis, theory_update_md, "REVERTED", gates)
            out = self._ledger({**rec, "status": "REVERTED", "seconds": round(time.time() - t0, 1)})
        return {**out, "discovery_diagnostics": res["discovery"]}

    def _finish(self, base: dict, status: str, theory_update_md: str, t0: float, **extra) -> dict:
        self._append_theory(base["iteration"], base["hypothesis"], theory_update_md, status, None, extra.get("error"))
        return self._ledger({**base, "status": status, **extra, "seconds": round(time.time() - t0, 1)})

    def _restore_champion(self) -> None:
        shutil.copyfile(self.backup_path, self.cfg.forecaster_path)

    def _append_theory(self, it: int, hypothesis: str, update_md: str, status: str, gates: dict | None,
                       error: str | None = None) -> None:
        lines = [f"\n## [{self.cfg.stage}] iteration {it}: {status} ({utc_now()})", f"**Hypothesis.** {hypothesis.strip()}", ""]
        if update_md and update_md.strip():
            lines += [update_md.strip()[:MAX_THEORY_UPDATE_CHARS], ""]
        if gates:
            g = gates
            lines.append(f"*Lab verdict:* holdout pooled {g['gate1_pooled_gain']['value']} (need > {g['gate1_pooled_gain']['threshold']}), "
                         f"median {g['gate2_breadth']['value']} (need >= {g['gate2_breadth']['threshold']}), "
                         f"coverage {g['gate3_calibration']['value']} (band 0.74-0.86), horizons {g['gate4_horizon_floor']['value']}.")
        if error:
            lines.append(f"*Lab verdict:* {status}: {error[:600]}")
        with open(self.cfg.theory_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def update_scientist_prompt(self, new_playbook_md: str, rationale: str) -> dict:
        self._check_lab_intact()
        text = (new_playbook_md or "").strip()
        if len(text) < 200 or len(text) > MAX_PLAYBOOK_CHARS:
            return {"status": "PROMPT_REJECTED", "error": f"playbook must be 200..{MAX_PLAYBOOK_CHARS} characters"}
        self.cfg.prompt_path.write_text(text + "\n")
        return self._ledger({"status": "PROMPT_UPDATED", "iteration": self.next_iteration() - 1,
                             "rationale": rationale[:1500], "prompt_sha256": sha256_text(text)})

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
{HOLDOUT_ORIGINS[0]}..{HOLDOUT_ORIGINS[-1]} return only 4 aggregates and decide promotion via 4 gates. A failed
candidate is reverted, but your notebook entry and playbook persist: write down what you learned, including negative
results, and make every hypothesis falsifiable. Never hard-code dates, series-specific constants tuned to a single
origin, or anything that could only be known after an origin.
""".strip()

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string", "description": "one falsifiable hypothesis tested by this candidate"},
        "theory_update_md": {"type": "string", "description": "markdown notebook entry: diagnosis, mechanism, expected effect"},
        "new_forecaster_code": {"type": "string", "description": "the complete new workspace/forecaster.py"},
        "new_playbook_md": {"type": "string", "description": "the complete revised workspace/scientist_prompt.md"},
        "playbook_rationale": {"type": "string"},
    },
    "required": ["hypothesis", "theory_update_md", "new_forecaster_code", "new_playbook_md", "playbook_rationale"],
}


def build_iteration_prompt(state: dict) -> str:
    def j(x, n):
        return json.dumps(x, default=str)[:n]

    return "\n\n".join([
        AGENT_CONTRACT,
        "## Your current research playbook (workspace/scientist_prompt.md)\n" + state["scientist_prompt_md"],
        "## Notebook tail (workspace/theory_of_market.md)\n" + state["theory_of_market_tail_md"],
        "## Champion forecaster (workspace/forecaster.py)\n```python\n" + state["champion_forecaster_py"] + "\n```",
        "## Champion Discovery diagnostics\n" + j(state["champion_discovery_diagnostics"], 14000),
        "## Champion Holdout aggregates\n" + j(state["champion_holdout_aggregates"], 2000),
        "## Promotion gates\n" + j(state["promotion_gates"], 1000),
        "## Experiment ledger (most recent last)\n" + j(state["ledger_tail"], 9000),
        f"## Task (iteration {state['next_iteration']}, stage {state['stage']})\n"
        "Diagnose the Discovery errors, state ONE falsifiable hypothesis, and write the complete new forecaster.py that "
        "tests it. Also return a revised playbook (heuristics, guardrails and coding rules learned so far). Work only from "
        "the material above: do not clone repositories, open pull requests or browse for GPU price data. Reply ONLY via "
        "the structured output.",
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
            ctrl.update_scientist_prompt(p["new_playbook_md"], p.get("playbook_rationale", ""))
        return ctrl.propose_and_test_forecaster(p["hypothesis"], p["theory_update_md"], p["new_forecaster_code"])


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

    def propose(self, prompt: str, title: str) -> dict:
        api, sid = self._create(prompt, title)
        rec = {"api": api, "session_id": sid, "created": utc_now()}
        self.sessions.append(rec)
        print(f"[devin] session {sid} ({api}) created", file=sys.stderr)
        t0, last = time.time(), None
        while time.time() - t0 < self.timeout_s:
            time.sleep(self.poll_s)
            s = self._get(api, sid)
            so = s.get("structured_output")
            status = (s.get("status"), s.get("status_detail") or s.get("status_enum"))
            if status != last:
                print(f"[devin] {sid}: {status}", file=sys.stderr)
                last = status
            done = {"exit", "error", "finished", "blocked", "stopped", "waiting_for_user", "suspended", "expired"}
            if isinstance(so, dict) and all(so.get(k) for k in ("hypothesis", "new_forecaster_code")):
                if done & {str(x) for x in status} or time.time() - t0 > 120:
                    rec.update(acus=s.get("acus_consumed"), finished=utc_now())
                    return so
            elif done & {str(x) for x in status} - {"waiting_for_user"} and "error" in str(status):
                raise RuntimeError(f"Devin session {sid} ended without structured output: {status}")
        raise TimeoutError(f"Devin session {sid} produced no structured output within {self.timeout_s}s")

    def run_iteration(self, ctrl: RSIController, i: int) -> dict:
        state = ctrl.inspect_current_state()
        so = self.propose(build_iteration_prompt(state), f"AI Scientist {state['stage']} iteration {state['next_iteration']}")
        if so.get("new_playbook_md"):
            ctrl.update_scientist_prompt(so["new_playbook_md"], so.get("playbook_rationale", ""))
        res = ctrl.propose_and_test_forecaster(so["hypothesis"], so.get("theory_update_md", ""), so["new_forecaster_code"])
        res["devin_session"] = self.sessions[-1]
        return res


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
                                     model_id=model_id, params={"temperature": 0.2, "max_tokens": 12000})
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

        results: list[dict] = []

        @tool
        def inspect_current_state() -> str:
            """Return the playbook, notebook tail, champion code, Discovery diagnostics, Holdout aggregates and ledger."""
            return json.dumps(ctrl.inspect_current_state(), default=str)[:60000]

        @tool
        def propose_and_test_forecaster(hypothesis: str, theory_update_md: str, new_forecaster_code: str) -> str:
            """Write a complete new workspace/forecaster.py, evaluate it (Discovery diagnostics + 4 Holdout gates), and
            promote it or revert it. Returns the verdict and the candidate's Discovery diagnostics."""
            r = ctrl.propose_and_test_forecaster(hypothesis, theory_update_md, new_forecaster_code)
            results.append(r)
            return json.dumps(r, default=str)[:30000]

        @tool
        def update_scientist_prompt(new_playbook_md: str, rationale: str) -> str:
            """Replace workspace/scientist_prompt.md (your research playbook) with a revised version."""
            return json.dumps(ctrl.update_scientist_prompt(new_playbook_md, rationale), default=str)

        tools = [inspect_current_state, propose_and_test_forecaster, update_scientist_prompt]
        if ctrl.cfg.enable_exa:
            @tool
            def search_exa_signals(query: str, origin_date: str) -> str:
                """Point-in-time web search (results published on or before origin_date <= the last Discovery origin)."""
                return json.dumps(ctrl.search_exa_signals(query, origin_date), default=str)[:20000]
            tools.append(search_exa_signals)

        playbook = ctrl.cfg.prompt_path.read_text() if ctrl.cfg.prompt_path.exists() else ""
        agent = Agent(model=self.model, tools=tools, system_prompt=AGENT_CONTRACT + "\n\n" + playbook, callback_handler=None)
        agent(f"Iteration {ctrl.next_iteration()} of stage {ctrl.cfg.stage}. Call inspect_current_state, diagnose the "
              "Discovery errors, then call propose_and_test_forecaster exactly once with one falsifiable hypothesis and a "
              "complete forecaster. Finally call update_scientist_prompt with your revised playbook.")
        if not results:
            raise RuntimeError("the Strands agent finished without proposing a candidate")
        return results[-1]


DEFAULT_MODAL_LLM_URL = "https://ananthbashyam1--ep-glm-5-3-flash-server.us-west.modal.direct"


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
            "hypothesis": (r.get("hypothesis") or "")[:160], "discovery_pooled_skill": d.get("pooled_skill"),
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
    n = len(a.candidates) if a.driver == "scripted" and a.candidates else a.iterations
    for i in range(n):
        try:
            r = driver.run_iteration(ctrl, i)
            print(json.dumps({k: r.get(k) for k in ("iteration", "status", "hypothesis", "holdout", "error")}, default=str))
        except Exception as e:  # a failed driver call is logged and the loop continues
            ctrl._ledger({"status": "DRIVER_ERROR", "iteration": ctrl.next_iteration(), "error": f"{type(e).__name__}: {e}"[:1500]})
            print(f"[rsi] iteration {i + 1} driver error: {e}", file=sys.stderr)
            if ctrl.cfg.forecaster_path.read_text() != ctrl.backup_path.read_text():
                ctrl._restore_champion()
    ctrl.checkpoint_notes(f"notebook checkpoint after {n} iteration(s)")
    write_ablation_summary(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
