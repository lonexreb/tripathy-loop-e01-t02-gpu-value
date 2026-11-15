"""Architecture 1 (memory-based RSI): the versioned improvement procedure I_g, its revision objects and admission gate,
forecast locks, and the evidence / episode memories.

F_g is workspace/forecaster.py (promoted only through the Holdout gates). I_g is a versioned procedure with three
components: `research_policy` (the scientist's playbook, which governs research episodes), `improver_policy` (the
instructions the improver follows when it proposes the next revision) and `episode_controls` (typed per-episode tool
budgets inside lab-owned bounds). A revision changes one component (an improver_policy revision is tested together with
the one revision its new improver proposes) and is admitted only by a fixed same-start comparison: frozen I_g and
candidate I_{g+1} each run the same number of paired research episodes from the same champion with the same total
budget; agent-written claims never count as evidence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from lab.memory import PROTECTED_BLOCK, enforce_protected

COMPONENTS = ("research_policy", "improver_policy", "episode_controls")
# episode_controls is the typed part of I_g: a JSON object of per-episode tool budgets. Revisions may reallocate them
# inside lab-owned bounds and a fixed total, so both arms of an admission comparison always spend the same budget.
DEFAULT_CONTROLS = {"max_dry_runs": 3, "max_backtests": 4, "max_analyses": 6}
CONTROL_BOUNDS = {"max_dry_runs": (1, 8), "max_backtests": (0, 8), "max_analyses": (0, 10)}
CONTROL_TOTAL = sum(DEFAULT_CONTROLS.values())
REVISION_FIELDS = ("parent_version", "target_component", "proposed_by", "change", "supporting_episode_ids",
                   "expected_behavior_change", "development_comparison", "rejection_condition", "status")
HYPOTHESIS_CARD_FIELDS = ("mechanism", "expected_observation", "competing_explanation", "test", "rejection_condition")
# Introspection. A generation opens with the scientist's own diagnosis of what limits skill (data quality, missing
# data or tools, compute, ideas); the improver turns needs that no playbook edit can satisfy into typed capability
# requests. The lab records both verbatim, self-serves the safe kinds and queues the rest for a maintainer: the
# research agenda, including data sourcing and cleaning, is the model's, while scoring and partitions stay fixed.
DIAGNOSIS_FIELDS = ("limiting_factors", "data_quality_issues", "data_needed", "tools_needed", "hypotheses")
CAPABILITY_KINDS = ("data_source", "tool", "library", "budget", "diagnostic", "compute", "other")
CAPABILITY_FIELDS = ("kind", "request", "evidence", "expected_effect", "how_to_test")
SELF_SERVICE_KINDS = ("data_source",)   # a dated URL -> point-in-time Wayback snapshot, served by the lab
MIN_COMPONENT_CHARS, MAX_COMPONENT_CHARS = 200, 24_000
ADMISSION_MARGIN = 0.001   # mean paired difference (candidate - frozen locked score) must exceed this (pooled skill)
ADMISSION_PAIRS = 3        # default number of frozen/candidate episode pairs per admission comparison

SEED_IMPROVER_POLICY = """# Improver policy (I_g, improver component)

Read the released research episodes and measured results. Identify one repeated failure in the forecasting or research
procedure (for example: experiments that never left one model class, diagnostics run in an unhelpful order, evidence
searched without a testable question, hypotheses without a rejection condition, Discovery gains that did not survive
Holdout). Propose ONE bounded change, explain the behavior it should alter, and specify a comparison that could reject
it. You may revise the researcher instructions (research_policy) or these improver instructions (improver_policy).
Cite the episode ids that support the change. Never claim the change is proved: the lab runs the comparison.
Read the scientist's diagnoses (procedural_memory.diagnoses) first. When they name a need that no playbook edit can
satisfy (a data source, a tool, a library, compute, a diagnostic the lab does not provide), emit it as a typed
capability_request with the evidence behind it; the lab serves what it safely can and queues the rest, and repeated
unmet requests are a finding in their own right.
""".strip()


def _short(x, n: int = 15) -> bool:
    return len(x if isinstance(x, str) else json.dumps(x)) < n


def validate_diagnosis(d) -> list[str]:
    """Problems with a scientist's self-assessment (empty list = recorded as the generation's diagnosis)."""
    if not isinstance(d, dict):
        return ["diagnosis must be a JSON object"]
    p = []
    for k in DIAGNOSIS_FIELDS:
        v = d.get(k)
        if v is None:
            p.append(f"missing field {k} (use an empty list when there is nothing to report)")
        elif not isinstance(v, list):
            p.append(f"{k} must be a list")
        else:
            p += [f"{k}[{i}] is too short to be actionable (>= 15 characters)" for i, x in enumerate(v) if _short(x)]
    if isinstance(d.get("limiting_factors"), list) and not d["limiting_factors"]:
        p.append("limiting_factors must name at least one factor, with the evidence for it")
    if all(isinstance(d.get(k), list) for k in DIAGNOSIS_FIELDS) and not any(
            d.get(k) for k in ("data_needed", "tools_needed", "hypotheses")):
        p.append("state at least one need (data_needed / tools_needed) or hypothesis")
    return p


def validate_capability_request(req) -> list[str]:
    """Problems with a typed capability request emitted by the improver."""
    if not isinstance(req, dict):
        return ["capability request must be a JSON object"]
    p = [f"missing field {k}" for k in CAPABILITY_FIELDS if req.get(k) in (None, "", [])]
    if req.get("kind") and req["kind"] not in CAPABILITY_KINDS:
        p.append(f"kind must be one of {CAPABILITY_KINDS}")
    if req.get("request") and _short(req["request"], 20):
        p.append("request must say concretely what is needed (>= 20 characters)")
    if req.get("evidence") is not None and not isinstance(req["evidence"], list):
        p.append("evidence must be a list of episode ids, diagnosis ids or measured facts")
    url = req.get("url")
    if url is not None and not (isinstance(url, str) and url.startswith(("http://", "https://"))):
        p.append("url must start with http:// or https://")
    as_of = req.get("as_of")
    if as_of is not None:
        try:
            dt.date.fromisoformat(str(as_of))
        except ValueError:
            p.append("as_of must be an ISO date (YYYY-MM-DD)")
    return p


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def validate_card(card: dict | None) -> list[str]:
    """Names of hypothesis-card fields that are missing or too short to be falsifiable."""
    card = card or {}
    return [k for k in HYPOTHESIS_CARD_FIELDS if len(str(card.get(k) or "").strip()) < 15]


def parse_controls(text) -> tuple[dict | None, list[str]]:
    """(controls, problems) for an episode_controls component: integer budgets inside CONTROL_BOUNDS, fixed total."""
    try:
        c = json.loads(text) if isinstance(text, str) else text
    except ValueError:
        c = None
    if not isinstance(c, dict) or set(c) != set(DEFAULT_CONTROLS):
        return None, [f"episode_controls must be a JSON object with exactly the keys {sorted(DEFAULT_CONTROLS)}"]
    p = [f"{k} must be an integer in {lo}..{hi}" for k, (lo, hi) in CONTROL_BOUNDS.items()
         if type(c[k]) is not int or not lo <= c[k] <= hi]
    if not p and sum(c.values()) != CONTROL_TOTAL:
        p.append(f"episode_controls must sum to {CONTROL_TOTAL} (reallocate the fixed budget; got {sum(c.values())})")
    return (None, p) if p else ({k: c[k] for k in DEFAULT_CONTROLS}, [])


def procedure_controls(version: dict | None) -> dict:
    """The episode budgets of a procedure version (versions that predate episode_controls use the defaults)."""
    c = parse_controls(((version or {}).get("components") or {}).get("episode_controls") or "")[0]
    return c or dict(DEFAULT_CONTROLS)


def normalize_component(target: str, text: str) -> str:
    if target == "episode_controls":
        c = parse_controls(text)[0]
        return json.dumps(c, sort_keys=True) if c else str(text or "")
    text = (text or "").strip()
    if target == "research_policy":
        text = enforce_protected(text)[0]
    return text


def validate_revision(rev: dict | None, active: dict, known_episode_ids: set[str]) -> list[str]:
    """Problems with a proposed revision object (empty list = structurally admissible for the comparison)."""
    if not isinstance(rev, dict):
        return ["revision must be a JSON object"]
    p = [f"missing field {k}" for k in REVISION_FIELDS if rev.get(k) in (None, "", [])]
    if rev.get("parent_version") and rev["parent_version"] != active["version"]:
        p.append(f"parent_version must be the active version {active['version']}")
    target = rev.get("target_component")
    if target not in COMPONENTS:
        p.append(f"target_component must be one of {COMPONENTS}")
    if rev.get("status") and rev["status"] != "proposed":
        p.append("status must be 'proposed' (the lab sets admitted/rejected)")
    ids = rev.get("supporting_episode_ids") or []
    if not isinstance(ids, list):
        p.append("supporting_episode_ids must be a list")
    elif known_episode_ids and not set(map(str, ids)) & known_episode_ids:
        p.append(f"supporting_episode_ids must cite at least one released episode: {sorted(known_episode_ids)[-12:]}")
    if target == "episode_controls":
        _, cp = parse_controls(rev.get("new_component_text") or "")
        p += cp
        if not cp and normalize_component(target, rev["new_component_text"]) == json.dumps(procedure_controls(active),
                                                                                         sort_keys=True):
            p.append("new_component_text is identical to the active component (no change)")
    elif target in COMPONENTS:
        new = normalize_component(target, rev.get("new_component_text") or "")
        extra = len(PROTECTED_BLOCK) if target == "research_policy" else 0
        if not (MIN_COMPONENT_CHARS + extra <= len(new) <= MAX_COMPONENT_CHARS + extra):
            p.append(f"new_component_text must be {MIN_COMPONENT_CHARS}..{MAX_COMPONENT_CHARS} characters excluding the "
                     f"LAB-PROTECTED block (got {len(new) - extra}); send the complete revised component text")
        elif new == active["components"].get(target):
            p.append("new_component_text is identical to the active component (no change)")
    return p


def episode_score(ep: dict | None) -> float:
    """Admission score of one episode: Discovery pooled-skill delta of its locked forecaster vs the shared champion.
    An episode that locked nothing, or whose lock failed the pre-Holdout screen, scores 0 (it kept the champion)."""
    lock = (ep or {}).get("lock") or {}
    if not lock.get("screen_pass"):
        return 0.0
    d = lock.get("discovery_pooled_delta")
    return float(d) if isinstance(d, (int, float)) else 0.0


def admission_rule(base_eps, cand_eps, margin: float = ADMISSION_MARGIN) -> dict:
    """Fixed rule over paired same-start episodes (pair i = i-th frozen vs i-th candidate episode): admit I_{g+1} iff
    mean(d) - se(d) > margin, d_i = score(candidate_i) - score(frozen_i), se = sd(d)/sqrt(n) (0 for a single pair, which
    is the original one-pair rule). Ties and unpaired or missing episodes keep I_g."""
    base_eps = [base_eps] if isinstance(base_eps, dict) else list(base_eps or [])
    cand_eps = [cand_eps] if isinstance(cand_eps, dict) else list(cand_eps or [])
    n = min(len(base_eps), len(cand_eps))
    b, c = [episode_score(e) for e in base_eps[:n]], [episode_score(e) for e in cand_eps[:n]]
    d = [y - x for x, y in zip(b, c)]
    mean = sum(d) / n if n else 0.0
    se = (sum((x - mean) ** 2 for x in d) / (n - 1) / n) ** 0.5 if n > 1 else 0.0
    return {"rule": f"admit iff mean(d) - se(d) > {margin} over n paired same-start episodes (d = candidate - frozen "
                    "score; score = Discovery pooled-skill delta of the locked forecaster vs the shared champion; no "
                    "lock or failed screen = 0; se = sd(d)/sqrt(n), 0 when n = 1)",
            "n_pairs": n, "frozen_scores": [round(x, 6) for x in b], "candidate_scores": [round(x, 6) for x in c],
            "frozen_score": round(sum(b) / n, 6) if n else 0.0, "candidate_score": round(sum(c) / n, 6) if n else 0.0,
            "difference": round(mean, 6), "difference_se": round(se, 6),
            "pairs_won": sum(x > 0 for x in d), "pairs_lost": sum(x < 0 for x in d),
            "admitted": n > 0 and (mean - se) > margin}


class ProcedureRegistry:
    """Versioned I_g with parent links: <dir>/vNNN.json plus active.json; rejected versions stay on disk."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, v: str) -> Path:
        return self.root / f"{v}.json"

    def versions(self) -> list[dict]:
        if not self.root.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(self.root.glob("v[0-9][0-9][0-9].json"))]

    def get(self, v: str) -> dict:
        return json.loads(self._path(v).read_text())

    def _write(self, rec: dict) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(rec["version"]).write_text(json.dumps(rec, indent=2))
        return rec

    def ensure_seed(self, research_policy: str, improver_policy: str = SEED_IMPROVER_POLICY) -> dict:
        if (self.root / "active.json").exists():
            return self.active()
        rec = self._write({"version": "v000", "parent_version": None, "status": "active", "created_utc": utc_now(),
                           "components": {"research_policy": normalize_component("research_policy", research_policy),
                                          "improver_policy": improver_policy,
                                          "episode_controls": json.dumps(DEFAULT_CONTROLS, sort_keys=True)},
                           "revision": None, "admission": None})
        (self.root / "active.json").write_text(json.dumps({"version": "v000"}))
        return rec

    def active(self) -> dict:
        return self.get(json.loads((self.root / "active.json").read_text())["version"])

    def preview(self, rev: dict, follow_up: dict | None = None) -> dict:
        """The candidate version `rev` would create (not written). `follow_up` is the testable revision that the
        candidate's own improver_policy proposed; it is applied on top and admitted or rejected together with `rev`."""
        parent = self.active()
        comps = dict(parent["components"])
        for r in (rev, follow_up) if follow_up else (rev,):
            comps[r["target_component"]] = normalize_component(r["target_component"], r["new_component_text"])
        revision = {k: rev.get(k) for k in REVISION_FIELDS}
        revision["status"] = "proposed"
        if follow_up:
            revision["follow_up_revision"] = {k: follow_up.get(k) for k in REVISION_FIELDS}
        return {"version": f"v{len(self.versions()):03d}", "parent_version": parent["version"], "status": "candidate",
                "created_utc": utc_now(), "components": comps, "revision": revision, "admission": None}

    def add_candidate(self, rev: dict, follow_up: dict | None = None) -> dict:
        return self._write(self.preview(rev, follow_up))

    def decide(self, v: str, admission: dict) -> dict:
        rec = self.get(v)
        rec["admission"] = admission
        rec["status"] = "active" if admission["admitted"] else "rejected"
        rec["revision"]["status"] = "admitted" if admission["admitted"] else "rejected"
        if admission["admitted"]:
            old = self.active()
            old["status"] = "superseded"
            self._write(old)
            (self.root / "active.json").write_text(json.dumps({"version": v}))
        return self._write(rec)

    def lineage_md(self) -> str:
        lines = ["# Improvement-procedure lineage (I_g)", "",
                 "| version | parent | status | component | change | frozen | candidate | admitted |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in self.versions():
            rev, adm = r.get("revision") or {}, r.get("admission") or {}
            lines.append(f"| {r['version']} | {r.get('parent_version') or '-'} | {r['status']} | "
                         f"{rev.get('target_component', '-')} | {str(rev.get('change', 'seed'))[:120].replace('|', '/')} | "
                         f"{adm.get('frozen_score', '-')} | {adm.get('candidate_score', '-')} | {adm.get('admitted', '-')} |")
        return "\n".join(lines) + "\n"


def write_lock(root: Path, episode_id: str, record: dict) -> Path:
    """Immutable forecast lock: refuses to overwrite an existing lock for the same episode."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{episode_id}.json"
    if p.exists():
        raise FileExistsError(f"forecast lock {p.name} already exists (locks are immutable)")
    p.write_text(json.dumps({**record, "episode_id": episode_id, "locked_utc": utc_now()}, indent=2, default=str))
    return p


def append_jsonl(path: Path, row: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts_utc": utc_now(), **row}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
