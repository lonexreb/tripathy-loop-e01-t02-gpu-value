"""Cross-iteration research memory of the RSI loop (immutable to the agent).

What the agent can remember is exactly what the lab keeps and shows it. This module keeps it:
  * a candidate archive (outputs/candidates/<stage>/iter_NNN/: forecaster.py, diff vs the champion it was tested
    against, full Discovery diagnostics, gates and gate margins), so reverted near-misses can be read and built on;
  * a compact experiment registry derived from the ledger (one row per experiment, never truncated), with hypothesis
    families and an exploration directive that pushes the agent off exhausted families instead of letting it conclude
    "freeze the champion";
  * a lab-owned protected block in the playbook that agent rewrites cannot delete, plus a report of guardrail bullets a
    rewrite dropped, and an archive of every playbook version;
  * budgeted JSON rendering that shrinks the least valuable fields instead of slicing a JSON string mid-way.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

HYPOTHESIS_FAMILIES: dict[str, str] = {
    "point_forecast": "q50 model: trend/momentum/mean-reversion/level dynamics",
    "interval_calibration": "band width/shape: conformal, z-scaling, asymmetric tails, per-horizon widths",
    "regime_routing": "which series get which model: segmentation, routing, per-class/per-provider models",
    "cross_series": "pooling/hierarchy/consensus across AZs, GPU models, providers or collectors",
    "exogenous_signals": "other pack tables: daily_statistics, aggregator_medians, index_prints, hardware, ladder",
    "regime_change": "change points, jumps, stickiness/step detection, volatility regimes",
    "horizon_structure": "how h=1/3/7 relate: direct vs iterated, horizon-specific models",
    "ensemble": "combining several forecasters or quantile averaging",
    "robustness": "guards, clipping, fallbacks, NaN/empty handling, numerical stability",
    "other": "anything else (explain in the hypothesis)",
}
NOVELTY_LEVELS = ("incremental", "bold")

PROTECTED_BEGIN = "<!-- LAB-PROTECTED:BEGIN (owned by the lab; rewrites cannot remove or edit this block) -->"
PROTECTED_END = "<!-- LAB-PROTECTED:END -->"
PROTECTED_BODY = """## Lab rules (protected)
- The lab, not you, decides when research stops. "Accept/freeze the champion" is never a valid plan: every
  iteration must test a real, falsifiable change. A string of reverts means the explored families are exhausted,
  not that the problem is solved; move to an untried family (see the experiment registry) or make a bolder change.
- Every hypothesis is tagged with one family and a novelty (incremental | bold). Do not spend more than two
  consecutive experiments in one family without a promotion; after two reverts in a row, the next one is bold.
- Near-misses are assets: reverted candidates are archived with their code, diff and diagnostics
  (inspect_candidate). Prefer building on the closest near-miss over restarting from the champion.
- A candidate whose Holdout aggregates equal the champion's (NO_OP) changed nothing that is scored: its
  selection rule never fired. Make changes that act unconditionally or show on Discovery that they fire.
- Guardrails are only removed with evidence: cite the iteration that falsified them in the rationale.
- Keep this playbook a distilled strategy (heuristics, guardrails, ranked ideas). The full experiment history
  is rendered by the lab from the ledger; do not copy the notebook into the playbook."""
PROTECTED_BLOCK = f"{PROTECTED_BEGIN}\n{PROTECTED_BODY}\n{PROTECTED_END}"

COLLAPSE_PATTERNS = (r"\bfreez(e|ing)\b[^.\n]{0,40}\bchampion", r"\baccept(ing)?\s+the\s+champion",
                     r"\bstop\s+(iterating|experimenting|research)", r"\bno\s+(further\s+)?headroom",
                     r"\blocal\s+optimum\b", r"\bnothing\s+(left|more)\s+to\s+(try|gain)")


# ----------------------------------------------------------------------------------- tags ----
def normalize_family(family: str | None) -> str:
    f = re.sub(r"[^a-z0-9]+", "_", (family or "").strip().lower()).strip("_")
    if not f:
        return "unspecified"
    if f in HYPOTHESIS_FAMILIES:
        return f
    for k in HYPOTHESIS_FAMILIES:
        if f.startswith(k) or k.startswith(f):
            return k
    return f"other:{f[:40]}"


_FAMILY_KEYWORDS = (
    ("interval_calibration", r"calibrat|conformal|coverage|quantile width|band width|\bz ?(10|90)\b|widen|narrow"),
    ("point_forecast", r"momentum|\bbeta\b|shrink|drift|trend|mean.revert|q50"),
    ("regime_change", r"change.?point|jump|step|stick"),
    ("cross_series", r"cross.(az|series|collector)|consensus|pool(ed|ing) across|hierarch"),
    ("exogenous_signals", r"daily_statistics|aggregator|index_print|hardware|capability"),
    ("ensemble", r"ensemble|blend|averag"),
    ("regime_routing", r"regime|segment|routing"),
)


def infer_family(hypothesis: str | None) -> str:
    """Best-effort family for legacy ledger rows that predate family tags."""
    h = re.split(r"[.:;]\s", (hypothesis or "").lower(), maxsplit=1)[0][:160]
    for fam, pat in _FAMILY_KEYWORDS:
        if re.search(pat, h):
            return fam
    return "unspecified"


def normalize_novelty(novelty: str | None) -> str:
    n = (novelty or "").strip().lower()
    return n if n in NOVELTY_LEVELS else "unspecified"


# ------------------------------------------------------------------------------ verdicts ----
def gate_margins(gates: dict | None) -> dict | None:
    """Signed distance of each gate from its threshold (>0 passes, <0 fails), plus the total shortfall."""
    if not gates:
        return None

    def num(x):
        return None if x is None else float(x)

    g1, g2, g3, g4 = (gates.get(k, {}) for k in ("gate1_pooled_gain", "gate2_breadth", "gate3_calibration",
                                                  "gate4_horizon_floor"))
    m = {}
    if num(g1.get("value")) is not None and num(g1.get("threshold")) is not None:
        m["gate1"] = num(g1["value"]) - num(g1["threshold"])
    if num(g2.get("value")) is not None and num(g2.get("threshold")) is not None:
        m["gate2"] = num(g2["value"]) - num(g2["threshold"])
    if num(g3.get("value")) is not None and g3.get("band"):
        lo, hi = g3["band"]
        m["gate3"] = min(num(g3["value"]) - lo, hi - num(g3["value"]))
    hv = g4.get("value") or {}
    if hv and g4.get("threshold") is not None:
        vals = [num(v) for v in hv.values()]
        m["gate4"] = min(v if v is not None else -1e9 for v in vals) - num(g4["threshold"])
    m = {k: round(v, 6) for k, v in m.items()}
    m["shortfall"] = round(sum(-v for v in m.values() if v < 0), 6)
    return m


def metric_delta(cand: dict | None, champ: dict | None) -> dict | None:
    """Candidate minus champion for the aggregate metrics (pooled, median, coverage, breaches, per-horizon)."""
    if not cand or not champ:
        return None
    out = {}
    for k in ("pooled_skill", "median_skill", "coverage_80", "q10_breach_rate", "q90_breach_rate"):
        if isinstance(cand.get(k), (int, float)) and isinstance(champ.get(k), (int, float)):
            out[k] = round(cand[k] - champ[k], 6)
    ch, cb = champ.get("skill_by_horizon") or {}, cand.get("skill_by_horizon") or {}
    hz = {h: round(cb[h] - ch[h], 6) for h in cb if isinstance(cb.get(h), (int, float)) and isinstance(ch.get(h), (int, float))}
    if hz:
        out["skill_by_horizon"] = hz
    return out


def is_no_op(holdout: dict | None, champ_holdout: dict | None) -> bool:
    if not holdout or not champ_holdout:
        return False
    keys = ("pooled_skill", "median_skill", "coverage_80")
    return all(holdout.get(k) is not None and champ_holdout.get(k) is not None
               and abs(float(holdout[k]) - float(champ_holdout[k])) < 1e-12 for k in keys) and \
        json.dumps(holdout.get("skill_by_horizon"), sort_keys=True) == json.dumps(champ_holdout.get("skill_by_horizon"), sort_keys=True)


# ------------------------------------------------------------------------------- archive ----
def unified_diff(old: str, new: str, old_name: str = "champion/forecaster.py", new_name: str = "candidate/forecaster.py") -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), old_name, new_name, n=2))


def archive_candidate(cand_dir: Path, code: str, champion_code: str, record: dict, discovery_full: dict | None) -> Path:
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "forecaster.py").write_text(code)
    (cand_dir / "diff_vs_champion.patch").write_text(unified_diff(champion_code, code))
    (cand_dir / "result.json").write_text(json.dumps({**record, "discovery_diagnostics": discovery_full}, indent=1, default=str))
    return cand_dir


def diff_stats(diff: str) -> dict:
    lines = diff.splitlines()
    return {"added": sum(1 for x in lines if x.startswith("+") and not x.startswith("+++")),
            "removed": sum(1 for x in lines if x.startswith("-") and not x.startswith("---"))}


# ------------------------------------------------------------------------------ registry ----
def _short(s: str | None, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 3] + "..."


def registry_row(r: dict) -> dict:
    h = r.get("holdout") or {}
    fam = r.get("hypothesis_family") or "unspecified"
    row = {"it": r.get("iteration"), "status": r.get("status"), "family": fam,
           "novelty": r.get("novelty", "unspecified"), "hypothesis": _short(r.get("hypothesis"), 220)}
    if fam == "unspecified" and r.get("status") not in ("SEED", "FORK"):
        inferred = infer_family(r.get("hypothesis"))
        if inferred != "unspecified":
            row["family"], row["family_inferred"] = inferred, True
    if r.get("builds_on_iteration") is not None:
        row["builds_on"] = r["builds_on_iteration"]
    if h:
        row["holdout"] = {"pooled": _r(h.get("pooled_skill")), "median": _r(h.get("median_skill")),
                          "cov": _r(h.get("coverage_80")),
                          "h": {k: _r(v) for k, v in (h.get("skill_by_horizon") or {}).items()}}
    d_hold = r.get("holdout_delta_vs_champion") or metric_delta(h, r.get("champion_holdout_before"))
    if d_hold:
        row["d_holdout"] = {k: v for k, v in d_hold.items() if k != "skill_by_horizon"}
    margins = r.get("gate_margins") or gate_margins(r.get("gates"))
    if margins:
        row["margins"] = margins
    if r.get("no_op") or (r.get("status") == "REVERTED" and is_no_op(h, r.get("champion_holdout_before"))):
        row["no_op"] = True
    if r.get("error"):
        row["error"] = _short(r["error"], 160)
    return row


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def family_stats(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("status") in ("SEED", "FORK"):
            continue
        f = out.setdefault(r.get("family", "unspecified"), {"tried": 0, "promoted": 0, "last_iteration": None})
        f["tried"] += 1
        f["promoted"] += r.get("status") == "PROMOTED"
        f["last_iteration"] = r.get("it")
    return out


def near_misses(rows: list[dict], k: int = 3) -> list[dict]:
    """Reverted candidates closest to promotion (smallest total gate shortfall), excluding no-ops."""
    cands = [r for r in rows if r.get("status") == "REVERTED" and r.get("margins") and not r.get("no_op")]
    return sorted(cands, key=lambda r: r["margins"]["shortfall"])[:k]


def exploration_directive(rows: list[dict], budget: dict | None = None) -> str:
    exps = [r for r in rows if r.get("status") not in ("SEED", "FORK")]
    stats = family_stats(rows)
    untried = [f for f in HYPOTHESIS_FAMILIES if f not in stats and f != "other"]
    streak = 0
    for r in reversed(exps):
        if r.get("status") == "PROMOTED":
            break
        streak += 1
    recent = [r.get("family") for r in exps[-3:]]
    msgs = []
    if budget:
        msgs.append(f"Budget: this is iteration {budget.get('this_iteration')} of {budget.get('planned')} planned in this run "
                    f"({budget.get('remaining')} remaining after this one).")
    if streak >= 2:
        msgs.append(f"{streak} experiments in a row without a promotion: the next candidate must be BOLD (novelty=bold) "
                    "and should come from a family you have not exhausted.")
    if len(recent) >= 2 and len(set(recent)) == 1 and recent[0] not in (None, "unspecified"):
        msgs.append(f"The last {len(recent)} experiments were all in family '{recent[0]}': switch family this time.")
    nm = near_misses(rows, 1)
    if nm:
        how = ("read it with inspect_candidate before deciding whether to build on it" if nm[0].get("archived", True)
               else "its code predates the candidate archive; its hypothesis and margins are in the registry")
        msgs.append(f"Closest near-miss: iteration {nm[0]['it']} (gate margins {nm[0]['margins']}); {how}.")
    if untried:
        msgs.append("Untried families: " + ", ".join(untried) + ".")
    if not msgs:
        msgs.append("Pick the family with the best expected gain; an incremental refinement of the champion is acceptable.")
    return " ".join(msgs)


def notebook_index(theory_md: str, max_items: int = 60) -> list[str]:
    heads = [ln.lstrip("#").strip() for ln in theory_md.splitlines() if ln.startswith("## ")]
    return heads[-max_items:]


def demote_headings(md: str) -> str:
    """Agent notebook entries may not create top-level entries of their own (keeps the notebook index clean)."""
    return re.sub(r"^#{1,2}(?=\s)", "###", md, flags=re.M)


# ------------------------------------------------------------------------------ playbook ----
def strip_protected(text: str) -> str:
    pat = re.compile(re.escape(PROTECTED_BEGIN) + r".*?" + re.escape(PROTECTED_END), re.S)
    return pat.sub("", text).strip()


def enforce_protected(text: str) -> tuple[str, bool]:
    """Return the playbook with exactly one verbatim lab block after the title; True if the agent's copy was altered."""
    intact = text.count(PROTECTED_BLOCK) == 1 and text.count(PROTECTED_BEGIN) == 1
    if intact:
        return text.strip() + "\n", False
    body = strip_protected(text).replace(PROTECTED_BEGIN, "").replace(PROTECTED_END, "").strip()
    lines = body.splitlines()
    if lines and lines[0].startswith("# "):
        out = "\n".join([lines[0], "", PROTECTED_BLOCK, *lines[1:]])
    else:
        out = PROTECTED_BLOCK + "\n\n" + body
    return out.strip() + "\n", True


def guardrail_bullets(text: str) -> list[str]:
    """Bullets under any '##' heading whose title mentions guardrail/rule/lesson (outside the protected block)."""
    out, on = [], False
    for ln in strip_protected(text).splitlines():
        if ln.startswith("#"):
            on = bool(re.search(r"guardrail|rule|lesson|never|dead", ln, re.I))
            continue
        if on and re.match(r"\s*[-*]\s+\S", ln):
            out.append(" ".join(ln.strip().lstrip("-*").split()))
    return out


def dropped_guardrails(old: str, new: str) -> list[str]:
    new_norm = " ".join(strip_protected(new).split()).lower()
    dropped = []
    for b in guardrail_bullets(old):
        key = re.sub(r"[*`_]", "", b).lower()[:60]
        if key and key not in re.sub(r"[*`_]", "", new_norm):
            dropped.append(_short(b, 200))
    return dropped


def collapse_warnings(text: str) -> list[str]:
    body = strip_protected(text)
    hits = []
    for p in COLLAPSE_PATTERNS:
        m = re.search(p, body, re.I)
        if m:
            s, e = max(0, m.start() - 60), min(len(body), m.end() + 60)
            hits.append(" ".join(body[s:e].split()))
    return hits


# ------------------------------------------------------------------------------ rendering ----
def fit_json(obj: dict, max_chars: int, shrink_order: list[tuple[str, str]]) -> str:
    """Serialise obj to at most max_chars of VALID JSON by shrinking fields in shrink_order.

    shrink_order entries are (key, mode): mode 'tail' keeps the end of a string, 'head' keeps the start, 'drop_old'
    drops list items from the front (oldest first). A key that cannot shrink further is replaced by a marker."""
    obj = dict(obj)
    s = json.dumps(obj, default=str)
    for key, mode in shrink_order:
        while len(s) > max_chars and key in obj:
            v, excess = obj[key], len(s) - max_chars
            keep = max(200, len(v) - excess - 64) if isinstance(v, str) else 0
            if isinstance(v, str) and keep + 64 < len(v):
                obj[key] = ("[...truncated by the lab...]\n" + v[-keep:]) if mode == "tail" else (v[:keep] + "\n[...truncated by the lab...]")
            elif isinstance(v, list) and len(v) > 1:
                obj[key] = v[1:] if mode == "drop_old" else v[:-1]
            elif isinstance(v, dict) and "truncated_by_lab" not in v and len(json.dumps(v, default=str)) > 400:
                obj[key] = {"truncated_by_lab": True, "preview": json.dumps(v, default=str)[:300]}
            else:
                obj[key] = "[omitted by the lab: context budget]"
                s = json.dumps(obj, default=str)
                break
            s = json.dumps(obj, default=str)
    if len(s) > max_chars:
        raise ValueError(f"state does not fit in {max_chars} chars even after shrinking")
    return s
