"""Frozen 2026-11-06 submission from the best verified champion, validated with the official starter scorer."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from lab.config import FINAL_ORIGIN, STAGE_PACK_ONLY, STAGE_PACK_PLUS_EXA, LabConfig, load_config
from lab.evaluator import Evaluator, load_forecaster
from lab.sandbox import ExaSignals, PointInTimeSandbox


def best_champion(cfg: LabConfig) -> tuple[str, Path, dict]:
    """Highest Holdout pooled skill among the stage champions (Stage 2 must strictly beat Stage 1 to be used)."""
    best = None
    for st in (STAGE_PACK_ONLY, STAGE_PACK_PLUS_EXA):
        d = cfg.outputs_dir / "champions" / st
        if (d / "champion_state.json").exists():
            s = json.loads((d / "champion_state.json").read_text())
            if best is None or s["holdout"]["pooled_skill"] > best[2]["holdout"]["pooled_skill"]:
                best = (st, d / "forecaster.py", s)
    if best is None:
        raise RuntimeError("no champion yet: run the RSI controller first")
    return best


def main() -> int:
    base = load_config()
    stage, path, state = best_champion(base)
    cfg = base.with_(enable_exa=stage == STAGE_PACK_PLUS_EXA, exa_cache_only=True)
    sb = PointInTimeSandbox(cfg)
    ev = Evaluator(cfg, sb)
    ids = sb.final_ids
    sl = sb.get_pack_slice(FINAL_ORIGIN, ids)
    fc = load_forecaster(path).predict_origin(sl, FINAL_ORIGIN, tuple(cfg.horizons),
                                              ExaSignals(sb, FINAL_ORIGIN) if cfg.enable_exa else None)
    clean, problems = ev._c.validate_forecast(fc.assign(origin_date=FINAL_ORIGIN), list(cfg.horizons))
    have = set(zip(clean["series_id"], clean["horizon_days"]))
    last = sl["series_history"].groupby("series_id").agg(value=("value", "last"), date=("date", "last"))
    fill = [(s, FINAL_ORIGIN, h, *(float(last.loc[s, "value"]),) * 3) for s in ids for h in cfg.horizons
            if (s, h) not in have and s in last.index]
    out = pd.concat([clean[["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"]],
                     pd.DataFrame(fill, columns=["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"])])
    out["origin_date"] = FINAL_ORIGIN
    out["target_date"] = (pd.Timestamp(FINAL_ORIGIN) + pd.to_timedelta(out["horizon_days"], unit="D")).dt.strftime("%Y-%m-%d")
    out["team_id"] = "ai-scientist"
    stale = (pd.Timestamp(FINAL_ORIGIN) - pd.to_datetime(out["series_id"].map(last["date"]))).dt.days
    out["method_note"] = f"{stage} champion iter {state['iteration']} sha {state['code_sha256'][:12]}" + \
        stale.map(lambda d: f"; last_value_stale_days={d}" if d > 0 else "")
    out = out.sort_values(["series_id", "horizon_days"]).reset_index(drop=True)
    dest = cfg.outputs_dir / f"submission_{FINAL_ORIGIN}.csv"
    out.to_csv(dest, index=False, float_format="%.6f")

    hist = cfg.series_cache_dir / f"history_{FINAL_ORIGIN}.csv"
    rep = cfg.outputs_dir / f"submission_{FINAL_ORIGIN}_scorer_report.json"
    cmd = [sys.executable, "-I", "-B", str(cfg.starter_dir / "scorer.py"), "--forecast", str(dest), "--truth", str(hist),
           "--history", str(hist), "--frozen-cutoff", FINAL_ORIGIN, "--catalog", str(sb.final_catalog_path),
           "--tier", "headline", "--horizons", ",".join(map(str, cfg.horizons)), "--out", str(rep)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    report = json.loads(rep.read_text())
    v = report["forecast_validation"]
    ok = r.returncode == 0 and report["status"] in ("scored", "scored_no_cells") and v["n_problems"] == 0
    print(json.dumps({"submission": str(dest), "stage": stage, "champion_iteration": state["iteration"], "rows": len(out),
                      "series": int(out["series_id"].nunique()), "filled_from_last_value": len(fill),
                      "lab_validation_problems": len(problems), "scorer_exit": r.returncode, "scorer_status": report["status"],
                      "scorer_rows_valid": v["rows_valid"], "scorer_problems": v["n_problems"], "valid": ok}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
