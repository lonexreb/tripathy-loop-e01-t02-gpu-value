#!/usr/bin/env python3
"""Fixed evaluator. Scores a candidate once on development and once on checkpoint with the unchanged starter scorer,
pairs it against its parent's cells (paired.py), applies PROTOCOL.md section 4 and writes the lineage artifact.
The protected block is never touched here."""
from __future__ import annotations
import datetime as dt, json, os, subprocess, sys, time
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R, OUT = os.path.join(ROOT, "research"), os.path.join(ROOT, "research", "out")
S = os.path.join(ROOT, "gpu-price-pack-v0.1.1-handover", "starter")
PY = [os.path.join(ROOT, ".venv", "bin", "python"), "-I", "-B"]
sys.path.insert(0, R)
from paired import compare  # noqa: E402
EVAL_BLOCKS = ("development", "checkpoint")


def accept(paired: dict) -> tuple[bool, str]:
    """PROTOCOL section 4: checkpoint diff < 0 with the 90% CI excluding zero AND development diff <= 0."""
    ck, dv = paired["checkpoint"], paired["development"]
    if not (ck["overall"]["mean_diff"] < 0 and ck["ci_excludes_zero"]):
        return False, f"checkpoint paired diff {ck['overall']['mean_diff']:+.6f} CI {ck['bootstrap_90ci_mean_diff']} does not exclude zero on the right side"
    if dv["overall"]["mean_diff"] > 0:
        return False, f"development paired diff {dv['overall']['mean_diff']:+.6f} is positive"
    return True, f"checkpoint diff {ck['overall']['mean_diff']:+.6f} CI {ck['bootstrap_90ci_mean_diff']}, development diff {dv['overall']['mean_diff']:+.6f}"


def run(*a):
    subprocess.run(a, check=True, stdout=subprocess.DEVNULL)


def evaluate(cid: str, cand: dict, parent: dict, blocks: dict, truth: str, ids: str, improver_id: str) -> dict:
    t0 = time.time(); scores, paired, cells = {}, {}, {}
    for b in EVAL_BLOCKS:
        fc, rep, cl = f"{OUT}/bt_{b}_{cid}.csv", f"{OUT}/report_{b}_{cid}.json", f"{OUT}/cells_{b}_{cid}.csv"
        run(*PY, f"{R}/scientist/forecaster.py", "--history", truth, "--origin-start", blocks[b]["start"], "--origin-end", blocks[b]["end"],
            "--origin-step", "1", "--horizons", "1,3,7", "--series-ids", ids, "--method", cand["method"], "--params", json.dumps(cand["params"]), "--out", fc)
        run(*PY, f"{S}/scorer.py", "--forecast", fc, "--truth", truth, "--history", truth, "--horizons", "1,3,7", "--series-ids", ids, "--out", rep, "--per-cell", cl)
        o = json.load(open(rep))["overall"]
        scores[b] = {k: o.get(k) for k in ("n_cells", "pinball", "skill_pooled", "skill_median_of_series", "coverage")}
        paired[b] = compare(pd.read_csv(cl), pd.read_csv(os.path.join(ROOT, parent["cells"][b])))
        cells[b] = os.path.relpath(cl, ROOT)
    ok, why = accept(paired)
    entry = {"id": cid, "generation": parent["generation"] + 1, "parent": parent["id"], "kind": "forecaster",
             "status": "accepted" if ok else "rejected", "decision_reason": why, "improver": improver_id,
             "method": cand["method"], "params": {**cand["params"]}, "hypothesis": cand["hypothesis"], "chain": cand["chain"],
             "forecaster": {"script": "research/scientist/forecaster.py", "method": cand["method"], "params": cand["params"]},
             "budget_used": {"evaluations": len(EVAL_BLOCKS), "blocks_read": list(EVAL_BLOCKS), "wall_clock_s": round(time.time() - t0, 1)},
             "scores": scores, "paired_vs_parent": paired, "cells": cells, "evidence_ids": [],
             "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with open(f"{R}/lineage/{cid}.json", "w") as f:
        json.dump(entry, f, indent=2)
    return entry
