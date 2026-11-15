"""Architecture 1 (memory-based RSI): the versioned improvement procedure I_g, its revision objects and admission gate,
forecast locks, and the evidence / episode memories.

F_g is workspace/forecaster.py (promoted only through the Holdout gates). I_g is a versioned procedure with two
components: `research_policy` (the scientist's playbook, which governs research episodes) and `improver_policy` (the
instructions the improver follows when it proposes the next revision). A revision changes exactly one component and is
admitted only by a fixed same-start comparison: frozen I_g and candidate I_{g+1} each run one research episode from the
same champion with the same budget; agent-written claims never count as evidence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from lab.memory import PROTECTED_BLOCK, enforce_protected

COMPONENTS = ("research_policy", "improver_policy")
REVISION_FIELDS = ("parent_version", "target_component", "proposed_by", "change", "supporting_episode_ids",
                   "expected_behavior_change", "development_comparison", "rejection_condition", "status")
HYPOTHESIS_CARD_FIELDS = ("mechanism", "expected_observation", "competing_explanation", "test", "rejection_condition")
MIN_COMPONENT_CHARS, MAX_COMPONENT_CHARS = 200, 12_000
ADMISSION_MARGIN = 0.001   # candidate episode must beat the frozen episode's locked score by this much (pooled skill)

SEED_IMPROVER_POLICY = """# Improver policy (I_g, improver component)

Read the released research episodes and measured results. Identify one repeated failure in the forecasting or research
procedure (for example: experiments that never left one model class, diagnostics run in an unhelpful order, evidence
searched without a testable question, hypotheses without a rejection condition, Discovery gains that did not survive
Holdout). Propose ONE bounded change, explain the behavior it should alter, and specify a comparison that could reject
it. You may revise the researcher instructions (research_policy) or these improver instructions (improver_policy).
Cite the episode ids that support the change. Never claim the change is proved: the lab runs the comparison.
""".strip()


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def validate_card(card: dict | None) -> list[str]:
    """Names of hypothesis-card fields that are missing or too short to be falsifiable."""
    card = card or {}
    return [k for k in HYPOTHESIS_CARD_FIELDS if len(str(card.get(k) or "").strip()) < 15]


def normalize_component(target: str, text: str) -> str:
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
    if target in COMPONENTS:
        new = normalize_component(target, rev.get("new_component_text") or "")
        extra = len(PROTECTED_BLOCK) if target == "research_policy" else 0
        if not (MIN_COMPONENT_CHARS + extra <= len(new) <= MAX_COMPONENT_CHARS + extra):
            p.append(f"new_component_text must be {MIN_COMPONENT_CHARS}..{MAX_COMPONENT_CHARS} characters")
        elif new == active["components"][target]:
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


def admission_rule(base_ep: dict, cand_ep: dict, margin: float = ADMISSION_MARGIN) -> dict:
    """Fixed rule: admit I_{g+1} iff its episode's locked score beats frozen I_g's by > margin. Ties keep I_g."""
    b, c = episode_score(base_ep), episode_score(cand_ep)
    return {"rule": f"admit iff score(candidate) - score(frozen) > {margin} (score = Discovery pooled-skill delta of "
                    "the locked forecaster vs the shared champion; no lock or failed screen = 0)",
            "frozen_score": round(b, 6), "candidate_score": round(c, 6), "difference": round(c - b, 6),
            "admitted": (c - b) > margin}


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
                                          "improver_policy": improver_policy},
                           "revision": None, "admission": None})
        (self.root / "active.json").write_text(json.dumps({"version": "v000"}))
        return rec

    def active(self) -> dict:
        return self.get(json.loads((self.root / "active.json").read_text())["version"])

    def add_candidate(self, rev: dict) -> dict:
        parent = self.active()
        comps = dict(parent["components"])
        comps[rev["target_component"]] = normalize_component(rev["target_component"], rev["new_component_text"])
        v = f"v{len(self.versions()):03d}"
        revision = {k: rev.get(k) for k in REVISION_FIELDS}
        revision["status"] = "proposed"
        return self._write({"version": v, "parent_version": parent["version"], "status": "candidate",
                            "created_utc": utc_now(), "components": comps, "revision": revision, "admission": None})

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
