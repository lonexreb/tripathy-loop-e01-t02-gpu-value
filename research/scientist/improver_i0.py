#!/usr/bin/env python3
"""I_0: the fixed, offline, rule-based improver. No model calls, no retrieval. It reads only the parent's
development-block artifacts (never checkpoint or protected), writes a diagnosis, and proposes at most two
executable candidates from a bounded library. Its rules are the artifact; they are hashed into lineage/I_0.json.

Chain per proposal: error -> unresolved question -> evidence -> hypothesis -> executable change.
"""
from __future__ import annotations
import hashlib, json, os
import numpy as np, pandas as pd

RULES = {
    "R1_interval_width": "if development coverage inside [q10,q90] < 0.75, the error is interval width: propose the two "
                         "width-targeting library members (repricing_mixture window=60, conformal_trailing window=45)",
    "R2_median_signal": "if the parent's development pinball minus last_value_empirical's is > -0.0005, the drift median adds "
                        "nothing: every candidate keeps q50 = last value",
    "R3_budget": "at most two candidates per cycle; a rule that would add a third is logged as deferred",
}
LIBRARY = [
    {"method": "repricing_mixture", "params": {"window": 60}, "targets": "interval_width",
     "hypothesis": "posted prices reprice in steps; recent repricing frequency, not the full-history frequency, sets the width"},
    {"method": "conformal_trailing", "params": {"window": 45}, "targets": "interval_width",
     "hypothesis": "recent realised |h-day log change| is a better width estimate than full-history quantiles (non-stationary volatility)"},
]


def diagnose(parent: dict, out_dir: str, root: str) -> dict:
    dev = json.load(open(os.path.join(out_dir, "report_development_" + parent["forecaster"]["method"] + ".json")))["overall"]
    lve = json.load(open(os.path.join(out_dir, "report_development_last_value_empirical.json")))["overall"]
    cells = pd.read_csv(os.path.join(root, parent["cells"]["development"]))
    cov_h = {str(h): round(float(((g.q10 <= g.y) & (g.y <= g.q90)).mean()), 3) for h, g in cells.groupby("horizon_days")}
    return {"block_read": "development", "coverage_within": dev["coverage"]["within_q10_q90"],
            "coverage_below_q10": dev["coverage"]["below_q10"], "coverage_above_q90": dev["coverage"]["above_q90"],
            "coverage_by_horizon": cov_h, "parent_pinball": dev["pinball"], "last_value_empirical_pinball": lve["pinball"],
            "median_signal": round(dev["pinball"] - lve["pinball"], 6)}


def propose(diag: dict) -> tuple[list[dict], list[str]]:
    fired, cands = [], []
    if diag["coverage_within"] < 0.75:
        fired.append("R1_interval_width")
        cands = [dict(c) for c in LIBRARY if c["targets"] == "interval_width"][:2]
    if diag["median_signal"] > -0.0005:
        fired.append("R2_median_signal")
        for c in cands:
            c["q50"] = "last_value (R2)"
    fired.append("R3_budget")
    for c in cands:
        c["chain"] = {"error": f"80% interval covers {diag['coverage_within']:.3f} of development truths (target 0.80); "
                               f"below q10 {diag['coverage_below_q10']:.3f}, above q90 {diag['coverage_above_q90']:.3f}",
                      "unresolved_question": "is the miss a width problem (volatility understated) or a location problem (median off)?",
                      "evidence": f"median_signal={diag['median_signal']:+.6f} (drift median vs flat median); coverage by horizon {diag['coverage_by_horizon']}",
                      "hypothesis": c["hypothesis"],
                      "executable_change": f"forecaster.py --method {c['method']} --params '{json.dumps(c['params'])}'"}
    return cands[:2], fired


def artifact(root: str) -> dict:
    src = open(os.path.abspath(__file__), "rb").read()
    return {"id": "I_0", "generation": 0, "parent": None, "kind": "improver", "status": "control",
            "description": "fixed rule-based improver, offline, no model, no retrieval (also the fixed-improver and no-Exa control)",
            "rules": RULES, "library": [{k: v for k, v in c.items() if k in ("method", "params", "targets")} for c in LIBRARY],
            "source": os.path.relpath(__file__, root), "source_sha256": hashlib.sha256(src).hexdigest(), "budget": {"max_candidates": 2, "queries": 0}}
