"""Information firewall and diagnostic trace generator (immutable to the agent).

Scores workspace/forecaster.py exactly like starter/scorer.py: cell loss = mean pinball over q10/q50/q90
against the truth dated origin + horizon; baseline = last value on or before the origin (flat); pooled
skill = 1 - mean cell loss / mean baseline cell loss; series skill excludes series whose baseline loss
is 0; missing or invalid forecast cells are filled with the baseline (zero skill for that cell).
Discovery runs expose full diagnostics; Holdout runs expose only the four gate aggregates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import traceback
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from lab.config import BENCHMARK_ORIGIN, SCORED_TIER, LabConfig, load_config
from lab.sandbox import ExaSignals, PointInTimeSandbox

FC_COLUMNS = ["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"]
HOLDOUT_KEYS = ("pooled_skill", "median_skill", "coverage_80", "skill_by_horizon")


def starter_modules(cfg: LabConfig):
    """Import the official starter helpers without writing __pycache__ into the read-only pack."""
    sys.dont_write_bytecode = True
    p = str(cfg.starter_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    import _common  # noqa: E402
    import baselines  # noqa: E402
    return _common, baselines


def load_forecaster(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"candidate_forecaster_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Forecaster()


class Evaluator:
    def __init__(self, cfg: LabConfig | None = None, sandbox: PointInTimeSandbox | None = None):
        self.cfg = cfg or load_config()
        self.sandbox = sandbox or PointInTimeSandbox(self.cfg)
        self._c, self._b = starter_modules(self.cfg)

    @cached_property
    def truth(self):
        df = self.sandbox.series_full[["series_id", "date", "value"]].copy()
        df["day"] = pd.to_datetime(df["date"], format="%Y-%m-%d").map(pd.Timestamp.toordinal).astype(np.int64)
        return self._c.SeriesStore(df)

    @cached_property
    def meta(self) -> pd.DataFrame:
        return self.sandbox.catalog.set_index("series_id")[["instrument_class", "pricing_tier", "source_id", "gpu_model"]]

    # ------------------------------------------------------------------------- forecasting ----
    def forecast(self, forecaster, origins, target_ids: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
        frames, timing = [], {}
        for o in origins:
            t0 = time.time()
            sl = self.sandbox.get_pack_slice(o, target_ids)
            sig = ExaSignals(self.sandbox, o) if self.cfg.enable_exa else None
            fc = forecaster.predict_origin(sl, o, tuple(self.cfg.horizons), sig)
            if not isinstance(fc, pd.DataFrame) or [c for c in FC_COLUMNS if c not in fc.columns]:
                raise ValueError(f"predict_origin({o}) must return a DataFrame with columns {FC_COLUMNS}")
            fc = fc[FC_COLUMNS].copy()
            fc["origin_date"] = fc["origin_date"].astype(str).str[:10]
            frames.append(fc)
            timing[o] = round(time.time() - t0, 3)
        return pd.concat(frames, ignore_index=True), {"seconds_by_origin": timing}

    # ----------------------------------------------------------------------------- scoring ----
    def score(self, fc: pd.DataFrame, origins, ids: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
        """One row per (series, origin, horizon) cell with truth, forecast (validated, baseline-filled) and losses."""
        c = self._c
        ids = ids or self.sandbox.scored_ids
        horizons = list(self.cfg.horizons)
        raw = fc.copy()
        raw["origin_date"] = raw["origin_date"].astype(str)
        clean, problems = c.validate_forecast(raw, horizons)
        clean = clean[clean["origin_date"].astype(str).isin(origins)]
        fmap = {(r.series_id, int(r.origin_day), int(r.horizon_days)): (float(r.q10), float(r.q50), float(r.q90))
                for r in clean.itertuples()}
        rows, n_filled, n_missing_truth = [], 0, 0
        for sid in ids:
            for o in origins:
                od = c.day_of(c.parse_date(o))
                lv = self.truth.last_value(sid, od)
                for h in horizons:
                    y = self.truth.value_at(sid, od + h)
                    if y is None or lv is None:
                        n_missing_truth += 1
                        continue
                    b = (lv[0],) * 3
                    f = fmap.get((sid, od, h))
                    if f is None:
                        f, n_filled = b, n_filled + 1
                    rows.append((sid, o, h, y, *f, lv[0], c.cell_pinball(y, *f), c.cell_pinball(y, *b)))
        cells = pd.DataFrame(rows, columns=["series_id", "origin_date", "horizon_days", "y", "q10", "q50", "q90",
                                            "last_value", "loss", "baseline_loss"])
        cells = cells.join(self.meta, on="series_id")
        return cells, {"n_rows_in": int(len(fc)), "n_invalid_rows": len(problems), "n_cells": int(len(cells)),
                       "n_cells_filled_from_baseline": n_filled, "n_cells_missing_truth": n_missing_truth,
                       "invalid_examples": problems[:5]}

    # ----------------------------------------------------------------------------- metrics ----
    @staticmethod
    def _skill(loss: float, base: float):
        return None if base == 0 else float(1.0 - loss / base)

    def aggregates(self, cells: pd.DataFrame) -> dict:
        ps = cells.groupby("series_id")[["loss", "baseline_loss"]].mean()
        ps = ps[ps["baseline_loss"] > 0]
        by_h = {str(h): self._skill(g["loss"].mean(), g["baseline_loss"].mean())
                for h, g in cells.groupby("horizon_days")}
        return {
            "pooled_skill": self._skill(cells["loss"].mean(), cells["baseline_loss"].mean()),
            "median_skill": float(np.median(1 - ps["loss"] / ps["baseline_loss"])) if len(ps) else None,
            "coverage_80": float(((cells["y"] >= cells["q10"]) & (cells["y"] <= cells["q90"])).mean()),
            "skill_by_horizon": by_h,
        }

    def diagnostics(self, cells: pd.DataFrame) -> dict:
        d = self.aggregates(cells)
        d["q10_breach_rate"] = float((cells["y"] < cells["q10"]).mean())
        d["q90_breach_rate"] = float((cells["y"] > cells["q90"]).mean())
        d["breach_targets"] = {"q10": 0.10, "q90": 0.10, "coverage_80": 0.80}
        d["mean_pinball"], d["mean_baseline_pinball"] = float(cells["loss"].mean()), float(cells["baseline_loss"].mean())
        d["skill_by_origin"] = {o: self._skill(g["loss"].mean(), g["baseline_loss"].mean())
                                for o, g in cells.groupby("origin_date")}
        cells = cells.assign(excess=cells["loss"] - cells["baseline_loss"],
                             inside=(cells["y"] >= cells["q10"]) & (cells["y"] <= cells["q90"]))
        sl = cells.groupby(["instrument_class", "pricing_tier", "horizon_days"]).agg(
            n_cells=("loss", "size"), loss=("loss", "mean"), baseline_loss=("baseline_loss", "mean"),
            excess_loss_sum=("excess", "sum"), coverage_80=("inside", "mean")).reset_index()
        sl["skill"] = 1 - sl["loss"] / sl["baseline_loss"].replace(0, np.nan)
        sl["share_of_total_loss"] = sl["loss"] * sl["n_cells"] / cells["loss"].sum()
        d["slices"] = _records(sl.sort_values("excess_loss_sum", ascending=False))
        d["worst_slices"] = d["slices"][:6]
        ser = cells.groupby("series_id").agg(
            instrument_class=("instrument_class", "first"), pricing_tier=("pricing_tier", "first"),
            n_cells=("loss", "size"), loss=("loss", "mean"), baseline_loss=("baseline_loss", "mean"),
            excess_loss_sum=("excess", "sum"), coverage_80=("inside", "mean"),
            mean_level=("last_value", "mean"), mean_q50_error=("q50", "mean"), mean_y=("y", "mean"),
            mean_width_rel=("q90", "mean")).reset_index()
        ser["mean_q50_error"] = ser["mean_q50_error"] - ser["mean_y"]
        ser["mean_width_rel"] = (cells.groupby("series_id").apply(
            lambda g: ((g["q90"] - g["q10"]) / g["last_value"]).mean(), include_groups=False)).to_numpy()
        ser["skill"] = 1 - ser["loss"] / ser["baseline_loss"].replace(0, np.nan)
        d["worst_series"] = _records(ser.sort_values("excess_loss_sum", ascending=False).head(5))
        d["best_series"] = _records(ser.sort_values("excess_loss_sum").head(3))
        return d

    # ---------------------------------------------------------------------------- top level ----
    def evaluate(self, forecaster, origins, expose_diagnostics: bool) -> dict:
        fc, info = self.forecast(forecaster, origins)
        cells, sinfo = self.score(fc, origins)
        if not expose_diagnostics:
            agg = self.aggregates(cells)
            return {k: agg[k] for k in HOLDOUT_KEYS}
        d = self.diagnostics(cells)
        d["scoring"] = sinfo
        d["runtime"] = info
        return d

    def benchmark_skill(self, forecaster, origin: str = BENCHMARK_ORIGIN) -> dict:
        fc, _ = self.forecast(forecaster, [origin])
        cells, _ = self.score(fc, [origin])
        return self.aggregates(cells)

    def full_run(self, forecaster_path: str | Path) -> dict:
        """Discovery (diagnostics) + Holdout (firewalled) + lab-only benchmark origin, one process."""
        f = load_forecaster(forecaster_path)
        t0 = time.time()
        disc = self.evaluate(f, list(self.cfg.discovery_origins), True)
        hold = self.evaluate(f, list(self.cfg.holdout_origins), False)
        bench = self.benchmark_skill(f)
        return {"discovery": disc, "holdout": hold, "benchmark_2026_10_24": {k: bench[k] for k in HOLDOUT_KEYS},
                "stage": self.cfg.stage, "seconds": round(time.time() - t0, 2)}

    # --------------------------------------------------------------- official scorer cross-check ----
    def official_score(self, fc: pd.DataFrame, origins: list[str]) -> dict:
        """Run starter/scorer.py on the same forecasts (frozen mode for one origin, backtest mode otherwise)."""
        hz = ",".join(map(str, self.cfg.horizons))
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fc.to_csv(td / "fc.csv", index=False, float_format="%.12g")
            cmd = [sys.executable, "-I", "-B", str(self.cfg.starter_dir / "scorer.py"), "--forecast", str(td / "fc.csv"),
                   "--truth", str(self.sandbox.series_path), "--catalog", str(self.sandbox.catalog_path),
                   "--tier", SCORED_TIER, "--horizons", hz, "--out", str(td / "report.json")]
            if len(origins) == 1:
                h = self.sandbox.series_full
                h[h["date"] <= origins[0]].to_csv(td / "hist.csv", index=False)
                cmd += ["--history", str(td / "hist.csv"), "--frozen-cutoff", origins[0]]
            else:
                cmd += ["--history", str(self.sandbox.series_path), "--missing-as-baseline"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return json.loads((td / "report.json").read_text())


def _records(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        out.append({k: (round(v, 6) if isinstance(v, float) and np.isfinite(v) else (None if isinstance(v, float) else
                    (int(v) if isinstance(v, (np.integer,)) else v))) for k, v in r.items()})
    return out


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    return str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a forecaster (Discovery diagnostics + Holdout gates)")
    ap.add_argument("--forecaster", required=True)
    ap.add_argument("--out", required=True, help="JSON result path")
    ap.add_argument("--enable-exa", action="store_true")
    ap.add_argument("--exa-cache-only", action="store_true")
    a = ap.parse_args()
    cfg = load_config(enable_exa=a.enable_exa, exa_cache_only=a.exa_cache_only)
    try:
        res = {"ok": True, **Evaluator(cfg).full_run(a.forecaster)}
    except Exception as e:  # reported to the controller as a RUNTIME_ERROR
        tb = traceback.format_exc().splitlines()
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000], "traceback_tail": "\n".join(tb[-25:])}
    Path(a.out).write_text(json.dumps(res, indent=2, default=_json_default, allow_nan=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
