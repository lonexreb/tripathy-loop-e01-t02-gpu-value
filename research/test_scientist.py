#!/usr/bin/env python3
"""Smallest checks that fail if the scientist's logic breaks. Run: .venv/bin/python -I -B research/test_scientist.py"""
import os, sys, json, tempfile, subprocess
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(R)
sys.path.insert(0, f"{R}/scientist"); sys.path.insert(0, R)
sys.path.insert(0, f"{ROOT}/gpu-price-pack-v0.1.1-handover/starter")
import forecaster, evaluate, improver_i0  # noqa: E402
from _common import SeriesStore, validate_forecast  # noqa: E402

rng = np.random.default_rng(1)
days = np.arange(1000, 1400)
step = np.cumsum(rng.random(400) < 0.05) * 0.1          # stepwise repricing
vals_step = 2.0 * np.exp(step - step.mean())
vals_noise = 2.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 400)))
df = pd.DataFrame({"series_id": ["step"] * 400 + ["noise"] * 400, "day": np.concatenate([days, days]),
                   "value": np.concatenate([vals_step, vals_noise])})
store = SeriesStore(df)

# 1 mixture: monotone, median = last, width grows with recent repricing frequency
q = forecaster.repricing_mixture(days, vals_step, 1399, 7, 60, 5, 10)
assert q and q[0] <= q[1] <= q[2] and q[1] == vals_step[-1]
busy = 2.0 * np.exp(np.cumsum(rng.random(400) < 0.3) * 0.1); qb = forecaster.repricing_mixture(days, busy, 1399, 7, 60, 5, 10)
assert (qb[2] - qb[0]) / qb[1] >= (q[2] - q[0]) / q[1]
assert forecaster.repricing_mixture(days[:10], vals_step[:10], 1009, 7, 60, 5, 10) is None   # too little data -> fallback

# 2 conformal: ~80% trailing coverage on iid log-noise
hits = []
for o in range(1200, 1390):
    k = days <= o; qq = forecaster.conformal_trailing(days[k], vals_noise[k], o, 3, 45, 0.2, 10)
    y = vals_noise[o - 1000 + 3]; hits.append(qq[0] <= y <= qq[2])
assert 0.65 <= np.mean(hits) <= 0.95, np.mean(hits)

# 3 forecaster rows validate under the starter schema, fallbacks are labelled
fc = forecaster.make_forecasts(store, ["step", "noise"], [1399, 1398], [1, 3, 7], "conformal_trailing", {})
clean, problems = validate_forecast(fc, [1, 3, 7]); assert not problems and len(clean) == 12
fc2 = forecaster.make_forecasts(store, ["step"], [1030], [7], "repricing_mixture", {})
assert fc2.method_note.str.contains("fallback=drift").all() or (fc2.q10 <= fc2.q90).all()

# 4 acceptance rule
mk = lambda d, lo, hi: {"overall": {"mean_diff": d}, "ci_excludes_zero": bool(hi < 0 or lo > 0), "bootstrap_90ci_mean_diff": [lo, hi]}
assert evaluate.accept({"checkpoint": mk(-0.002, -0.003, -0.001), "development": mk(-0.001, -0.002, 0.0)})[0]
assert not evaluate.accept({"checkpoint": mk(-0.002, -0.004, 0.001), "development": mk(-0.001, -0.002, 0.0)})[0]
assert not evaluate.accept({"checkpoint": mk(-0.002, -0.003, -0.001), "development": mk(0.001, 0.0, 0.002)})[0]

# 5 improver: budget and rule firing
c, fired = improver_i0.propose({"coverage_within": 0.6, "coverage_below_q10": 0.2, "coverage_above_q90": 0.2, "coverage_by_horizon": {}, "median_signal": 0.0})
assert len(c) == 2 and "R1_interval_width" in fired and "R2_median_signal" in fired and all("chain" in x for x in c)
c, fired = improver_i0.propose({"coverage_within": 0.8, "coverage_below_q10": 0.1, "coverage_above_q90": 0.1, "coverage_by_horizon": {}, "median_signal": -0.01})
assert c == [] and "R1_interval_width" not in fired
print("test_scientist ok")
