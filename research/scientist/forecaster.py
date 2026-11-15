#!/usr/bin/env python3
"""Candidate forecasters for the scientist. Every method uses only rows dated on or before the origin.
  drift               the starter baseline (delegated to starter/baselines.py, unchanged)
  repricing_mixture   q50 = last value; q10/q90 from a two-part mixture: P(a repricing within h days) estimated on the
                      trailing `window` days, repricing size from every nonzero historical h-day log change
  conformal_trailing  q50 = last value; symmetric log interval whose half-width is the finite-sample 0.8 quantile of
                      |h-day log change| over the trailing `window` days (split conformal on a trailing window)
When a method has too little data for a cell it falls back to drift and says so in method_note.
  python3 forecaster.py --history series.csv --origin-start 2026-10-01 --origin-end 2026-10-15 --method conformal_trailing \
      --params '{"window": 45}' --series-ids @adapted_series.txt --out fc.csv
"""
from __future__ import annotations
import argparse, json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "gpu-price-pack-v0.1.1-handover", "starter"))
from _common import SeriesStore, h_step_changes, iso, monotone, read_series_csv, day_of, parse_date, parse_horizons  # noqa: E402
from baselines import OUT_COLUMNS, forecast_one as starter_forecast_one, resolve_series_ids, resolve_origins, add_scope_args  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULTS = {"drift": {}, "repricing_mixture": {"window": 60, "min_nonzero": 5, "min_recent": 10},
            "conformal_trailing": {"window": 45, "alpha": 0.2, "min_n": 10}}


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x); cw = np.cumsum(w[o]); cw = cw / cw[-1]
    return float(x[o][min(int(np.searchsorted(cw, q, side="left")), len(x) - 1)])


def repricing_mixture(days, vals, origin_day, h, window, min_nonzero, min_recent):
    ch = h_step_changes(days, vals, h, True)
    nz = ch[ch != 0]
    if len(ch) < 20 or len(nz) < min_nonzero:
        return None
    k = days > origin_day - window
    recent = h_step_changes(days[k], vals[k], h, True)
    p = float((recent != 0).mean()) if len(recent) >= min_recent else float((ch != 0).mean())
    x = np.concatenate([[0.0], nz]); w = np.concatenate([[1.0 - p], np.full(len(nz), p / len(nz))])
    last = float(vals[-1])
    return last * math.exp(wquantile(x, w, 0.1)), last, last * math.exp(wquantile(x, w, 0.9))


def conformal_trailing(days, vals, origin_day, h, window, alpha, min_n):
    k = days > origin_day - window - h
    r = np.abs(h_step_changes(days[k], vals[k], h, True))
    n = len(r)
    if n < min_n:
        return None
    s = float(np.sort(r)[min(n, math.ceil((n + 1) * (1 - alpha))) - 1])
    last = float(vals[-1])
    return last * math.exp(-s), last, last * math.exp(s)


def forecast_one(store: SeriesStore, sid: str, origin_day: int, horizons: list[int], method: str, params: dict) -> list[dict]:
    base = starter_forecast_one(store, sid, origin_day, horizons, "drift", 20, None)
    if method == "drift" or not base:
        return base
    days, vals = store.history(sid, origin_day)
    if not (vals > 0).all():
        return base
    fn = {"repricing_mixture": repricing_mixture, "conformal_trailing": conformal_trailing}[method]
    p = {**DEFAULTS[method], **params}
    out = []
    for row in base:
        q = fn(days, vals, origin_day, row["horizon_days"], **p)
        if q is None:
            out.append({**row, "method": method, "method_note": row["method_note"] + ";fallback=drift"})
        else:
            q10, q50, q90 = monotone(*q)
            out.append({**row, "q10": q10, "q50": q50, "q90": q90, "method": method,
                        "method_note": f"last_value_date={iso(int(days[-1]))};params={json.dumps(p, sort_keys=True)}"})
    return out


def make_forecasts(store, ids, origin_days, horizons, method, params) -> pd.DataFrame:
    rows = [r for o in origin_days for sid in ids if sid in store for r in forecast_one(store, sid, o, horizons, method, params)]
    return pd.DataFrame(rows, columns=OUT_COLUMNS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True); ap.add_argument("--origin"); ap.add_argument("--origins")
    ap.add_argument("--origin-start"); ap.add_argument("--origin-end"); ap.add_argument("--origin-step", type=int, default=1)
    ap.add_argument("--horizons", default="1,3,7"); ap.add_argument("--method", default="drift", choices=sorted(DEFAULTS))
    ap.add_argument("--params", default="{}", help="JSON overrides of the method defaults")
    ap.add_argument("--frozen", action="store_true", help="refuse if the history has any row after the (single) origin")
    add_scope_args(ap); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    origins = resolve_origins(args); store = SeriesStore(read_series_csv(args.history))
    if args.frozen and (len(origins) != 1 or (store.max_day() or 0) > origins[0]):
        print("REFUSED: --frozen needs one origin and a history truncated at it", file=sys.stderr); return 2
    fc = make_forecasts(store, resolve_series_ids(args, store), origins, parse_horizons(args.horizons), args.method, json.loads(args.params))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fc.to_csv(args.out, index=False, float_format="%.10g")
    print(f"{args.method}: {len(fc)} rows | series {fc['series_id'].nunique() if len(fc) else 0} | origins {len(origins)} | "
          f"fallbacks {int(fc['method_note'].str.contains('fallback=drift').sum()) if len(fc) else 0} | wrote {os.path.basename(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
