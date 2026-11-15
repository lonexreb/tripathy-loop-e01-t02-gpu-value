#!/usr/bin/env python3
"""First baseline slice. One command, pack untouched, everything written to research/out and research/lineage.
1. truth series + catalog from the pack (as-of 2026-11-06), frozen history + catalog at the official origin
2. official baselines (last_value, last_value_empirical, drift) on the adapted series over every in-pack block,
   scored cell by cell, plus paired drift-vs-last_value reports
3. drift submission skeleton for every headline series at the official origin (frozen mode)
4. lineage root F_0 = starter drift, with its block scores
"""
import datetime as dt, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, OUT = os.path.join(ROOT, "research"), os.path.join(ROOT, "research", "out")
PACK = os.path.join(ROOT, "gpu-price-pack-v0.1.1-handover")
S = os.path.join(PACK, "starter")
PY = [os.path.join(ROOT, ".venv", "bin", "python"), "-I", "-B"]
TRUTH = os.path.join(OUT, "series_daily.csv")   # built here from the pack: full rows through 2026-11-06, truth and history for backtests
IDS = "@" + os.path.join(R, "adapted_series.txt")
H = "1,3,7"
OFFICIAL = "2026-11-06"
METHODS = ["last_value", "last_value_empirical", "drift"]


def run(*a):
    print("+", " ".join(os.path.relpath(x, ROOT) if os.path.isabs(x) else x for x in a[3:]), flush=True)
    subprocess.run(a, check=True)


def starter(script, *a):
    run(*PY, os.path.join(S, script), *a)


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(R, "lineage"), exist_ok=True)
    blocks = json.load(open(os.path.join(R, "origins.json")))["blocks"]
    run(*PY, os.path.join(R, "paired.py"), "--selftest")

    full_cat = f"{OUT}/series_catalog.csv"
    starter("build_series.py", "--pack", PACK, "--out", TRUTH, "--catalog", full_cat, "--as-of", OFFICIAL,
            "--report", f"{OUT}/build_series_report.json")
    hist, cat = f"{OUT}/history_{OFFICIAL}.csv", f"{OUT}/catalog_{OFFICIAL}.csv"
    starter("build_series.py", "--pack", PACK, "--cutoff", OFFICIAL, "--members-from", full_cat,
            "--out", hist, "--catalog", cat, "--report", f"{OUT}/build_history_{OFFICIAL}.json")

    scores = {}
    for b, spec in blocks.items():
        if spec["truth"] != "in_pack":
            continue
        for m in METHODS:
            fc = f"{OUT}/bt_{b}_{m}.csv"
            starter("baselines.py", "--history", TRUTH, "--origin-start", spec["start"], "--origin-end", spec["end"],
                    "--origin-step", "1", "--horizons", H, "--series-ids", IDS, "--method", m, "--out", fc)
            starter("scorer.py", "--forecast", fc, "--truth", TRUTH, "--history", TRUTH, "--horizons", H,
                    "--series-ids", IDS, "--out", f"{OUT}/report_{b}_{m}.json", "--per-cell", f"{OUT}/cells_{b}_{m}.csv")
        run(*PY, f"{R}/paired.py", "--a", f"{OUT}/cells_{b}_drift.csv", "--b", f"{OUT}/cells_{b}_last_value.csv",
            "--out", f"{OUT}/paired_{b}_drift_vs_last_value.json")
        scores[b] = {}
        for m in METHODS:
            o = json.load(open(f"{OUT}/report_{b}_{m}.json"))["overall"]
            scores[b][m] = {k: o.get(k) for k in ("n_cells", "pinball", "skill_pooled", "skill_median_of_series", "coverage")}
        scores[b]["paired_drift_vs_last_value"] = json.load(open(f"{OUT}/paired_{b}_drift_vs_last_value.json"))

    sub = f"{OUT}/submission_F0_drift_{OFFICIAL}.csv"
    starter("baselines.py", "--history", hist, "--origin", OFFICIAL, "--horizons", H, "--method", "drift",
            "--catalog", cat, "--tier", "headline", "--frozen", "--out", sub)

    f0 = {"id": "F_0", "generation": 0, "parent": None, "kind": "forecaster", "status": "accepted",
          "accepted_by": "protocol root (official starter baseline), not an evaluator decision",
          "improver": None, "method": "starter drift: baselines.py --method drift --min-samples 20, no lookback",
          "adapted_series": IDS[1:], "horizons": [1, 3, 7], "scores": scores,
          "cells": {b: os.path.relpath(f"{OUT}/cells_{b}_drift.csv", ROOT) for b in scores},
          "forecaster": {"script": "gpu-price-pack-v0.1.1-handover/starter/baselines.py", "method": "drift", "params": {"min_samples": 20}},
          "official_submission": os.path.relpath(sub, ROOT),
          "notes": ["gpurentalprices collector has no rows on 2026-10-14, 10-17, 10-23, 10-24, 10-30, 11-01: those target days are missing_truth cells; afhubbard misses only 2026-09-07",
                    "gpurentalprices.neocloud_rate_card.A100_40GB.on_demand.median does not move inside the protected block: baseline degenerate there",
                    "official-origin catalog (cutoff 2026-11-06) has 37 headline series, all daily statistics; every aws_spot series is stale>3d and drops to extended",
                    "drift and last_value_empirical differ by <0.01 skill on every block: the drift median carries no measurable signal on these series"],
          "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with open(f"{R}/lineage/F_0.json", "w") as f:
        json.dump(f0, f, indent=2)
    print("\n== block summary (adapted series, pooled skill vs last_value; paired = drift - last_value mean pinball diff, 90% block-bootstrap CI)")
    for b, s in scores.items():
        p = s["paired_drift_vs_last_value"]
        print(f"{b:12s} cells={s['drift']['n_cells']:4d} drift={s['drift']['skill_pooled']:+.4f} lve={s['last_value_empirical']['skill_pooled']:+.4f} "
              f"median_series={s['drift']['skill_median_of_series']:+.4f} cov80={s['drift']['coverage']['within_q10_q90']:.3f} "
              f"diff={p['overall']['mean_diff']:+.5f} CI={p['bootstrap_90ci_mean_diff']} excl0={p['ci_excludes_zero']}")
    print("wrote research/lineage/F_0.json")


if __name__ == "__main__":
    sys.exit(main())
