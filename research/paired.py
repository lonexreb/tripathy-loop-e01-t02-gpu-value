#!/usr/bin/env python3
"""Paired per-cell pinball comparison of two scorer --per-cell files: A minus B.
Negative mean_diff means A is better. The 90% CI is a block bootstrap over origins.
  python3 paired.py --a cells_drift.csv --b cells_last_value.csv --out paired.json
  python3 paired.py --selftest
"""
import argparse, json, sys
import numpy as np
import pandas as pd

KEY = ["series_id", "origin_date", "horizon_days"]


def r(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def summ(g):
    sb = g.loss_b.sum()
    return {"n_cells": int(len(g)), "pinball_a": r(g.loss_a.mean()), "pinball_b": r(g.loss_b.mean()),
            "mean_diff": r(g.d.mean()), "skill_a_vs_b": r(1 - g.loss_a.sum() / sb) if sb > 0 else None,
            "share_a_better": r((g.d < 0).mean()), "share_tied": r((g.d == 0).mean()),
            "coverage_a_q10_q90": r(((g.q10 <= g.y) & (g.y <= g.q90)).mean())}


def compare(a, b, reps=2000, seed=0):
    m = a[KEY + ["loss", "y", "q10", "q90"]].merge(b[KEY + ["loss"]], on=KEY, suffixes=("_a", "_b"))
    m["d"] = m.loss_a - m.loss_b
    origins = sorted(m.origin_date.unique())
    sums = m.groupby("origin_date").d.sum().reindex(origins).to_numpy()
    cnt = m.groupby("origin_date").size().reindex(origins).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(origins), size=(reps, len(origins)))
    boots = sums[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [0.05, 0.95])
    return {"overall": summ(m), "n_origins": len(origins),
            "bootstrap_90ci_mean_diff": [r(lo), r(hi)], "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "by_horizon": {str(h): summ(g) for h, g in m.groupby("horizon_days")},
            "by_series": {s: summ(g) for s, g in m.groupby("series_id")}}


def selftest():
    base = pd.DataFrame({"series_id": "s", "origin_date": [f"2026-01-0{i}" for i in range(1, 6)] * 2,
                         "horizon_days": [1] * 5 + [3] * 5, "y": 1.0, "q10": 0.9, "q90": 1.1})
    a = base.assign(loss=0.01); b = base.assign(loss=0.02)
    out = compare(a, b, reps=200)
    assert out["overall"]["mean_diff"] == -0.01 and out["ci_excludes_zero"] and out["overall"]["share_a_better"] == 1.0
    assert out["overall"]["coverage_a_q10_q90"] == 1.0 and out["n_origins"] == 5
    print("paired.py selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    out = {"a": args.a, "b": args.b, **compare(pd.read_csv(args.a), pd.read_csv(args.b), args.reps, args.seed)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    o = out["overall"]
    print(f"paired: n={o['n_cells']} A={o['pinball_a']} B={o['pinball_b']} diff={o['mean_diff']} "
          f"90%CI={out['bootstrap_90ci_mean_diff']} skill_A_vs_B={o['skill_a_vs_b']} A_better={o['share_a_better']}")


if __name__ == "__main__":
    sys.exit(main())
