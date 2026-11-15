"""Information firewall and diagnostic trace generator (immutable to the agent).

Scores workspace/forecaster.py exactly like starter/scorer.py: cell loss = mean pinball over q10/q50/q90
against the truth dated origin + horizon; baseline = last value on or before the origin (flat); pooled
skill = 1 - mean cell loss / mean baseline cell loss; series skill excludes series whose baseline loss
is 0; missing or invalid forecast cells are filled with the baseline (zero skill for that cell).
Discovery runs expose full diagnostics; Holdout runs expose only the four gate aggregates.
A lab-side sanity report (non-finite, crossing, non-positive and absurd quantiles versus the last value) is computed
on the raw forecasts; it never alters scoring, the controller uses it to reject a candidate before Holdout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
PHASES = ("full", "discovery", "holdout")
MAX_BACKTEST_SERIES = 5        # series whose per-cell residuals a backtest returns
BACKTEST_BOOTSTRAP = 2000


def sanity_report(fc: pd.DataFrame, last_value: pd.Series, max_abs_log_ratio: float) -> dict:
    """Lab-side output checks on raw forecast rows (scoring is unaffected).

    last_value is indexed by (series_id, origin_date) and holds the last observed value on or before the origin.
    Fatal: any non-finite quantile, or any quantile with |log(q / last value)| > max_abs_log_ratio. Crossing and
    non-positive rows are reported (the scorer fills them with the baseline) so the controller can bound their share.
    """
    q = fc[["q10", "q50", "q90"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = len(fc)
    finite = np.isfinite(q)
    row_finite = finite.all(axis=1)
    with np.errstate(invalid="ignore"):
        crossing = row_finite & ((q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2]))
        nonpos = row_finite & (q[:, 0] <= 0)
    keys = pd.MultiIndex.from_arrays([fc["series_id"].astype(str), fc["origin_date"].astype(str).str[:10]])
    lv = last_value.reindex(keys).to_numpy(dtype=float)
    ok_ref = row_finite & ~nonpos & np.isfinite(lv) & (lv > 0)
    ratio = np.full(n, np.nan)
    if ok_ref.any():
        ratio[ok_ref] = np.abs(np.log(q[ok_ref] / lv[ok_ref, None])).max(axis=1)
    extreme = np.nan_to_num(ratio, nan=0.0) > max_abs_log_ratio
    dup = int(fc.duplicated(["series_id", "origin_date", "horizon_days"]).sum()) if n else 0
    ex = fc.loc[(~row_finite) | extreme, ["series_id", "origin_date", "horizon_days", "q10", "q50", "q90"]].head(5)
    ex = ex.assign(last_value=lv[((~row_finite) | extreme)][:len(ex)])
    out = {
        "n_rows": int(n), "n_nonfinite_rows": int((~row_finite).sum()), "n_crossing_rows": int(crossing.sum()),
        "n_nonpositive_rows": int(nonpos.sum()), "n_invalid_rows": int((crossing | nonpos).sum()),
        "n_extreme_rows": int(extreme.sum()), "n_duplicate_rows": dup,
        "max_abs_log_ratio_vs_last": (round(float(np.nanmax(ratio)), 4) if np.isfinite(ratio).any() else None),
        "limit_abs_log_ratio": round(float(max_abs_log_ratio), 4),
        "examples": [{k: (v if not isinstance(v, float) or math.isfinite(v) else str(v)) for k, v in r.items()}
                     for r in ex.astype(object).to_dict("records")],
    }
    out["ok"] = out["n_nonfinite_rows"] == 0 and out["n_extreme_rows"] == 0
    return out


def paired_delta_stats(loss, loss_ref, base, block: int = 1, n_boot: int = BACKTEST_BOOTSTRAP, seed: int = 0) -> dict:
    """Paired pooled-skill delta of a candidate vs a reference from per-origin loss sums (in time order), with a
    circular moving-block bootstrap standard error over origins.

    delta = (sum loss_ref - sum loss) / sum baseline loss, i.e. pooled skill(candidate) - pooled skill(reference).
    `block` is the number of consecutive origins resampled together: origins closer than the longest horizon share
    target dates, so resampling them one by one (block=1) understates the error of dense or long-horizon runs."""
    arr = np.column_stack([np.asarray(x, dtype=float) for x in (loss, loss_ref, base)])
    n = len(arr)
    tot = arr[:, 2].sum() if n else 0.0
    delta = float((arr[:, 1].sum() - arr[:, 0].sum()) / tot) if tot else None
    block = max(1, min(int(block), n)) if n else 1
    se = None
    if n > 1 and tot:
        rng = np.random.default_rng(seed)
        n_blocks = -(-n // block)
        starts = rng.integers(0, n, size=(n_boot, n_blocks))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_boot, -1)[:, :n]
        den = arr[idx, 2].sum(1)
        boot = (arr[idx, 1].sum(1) - arr[idx, 0].sum(1)) / np.where(den == 0, np.nan, den)
        se = float(np.nanstd(boot, ddof=1))
    gain = arr[:, 1] - arr[:, 0] if n else np.array([])
    return {"pooled_skill_delta": delta, "delta_se": se, "delta_z": (delta / se) if (delta is not None and se) else None,
            "n_origins": int(n), "block_origins": int(block), "n_independent_blocks": round(n / block, 2) if n else 0,
            "share_origins_improved": float((gain > 0).mean()) if n else None}


def overlap_block(origins, max_horizon_days: int) -> int:
    """Number of consecutive origins whose target windows overlap (spacing shorter than the longest horizon)."""
    days = sorted(pd.Timestamp(str(o)[:10]).toordinal() for o in origins)
    if len(days) < 2:
        return 1
    gap = max(1, int(np.median(np.diff(days))))
    return max(1, -(-int(max_horizon_days) // gap))


def loss_by_origin(cells: pd.DataFrame) -> dict:
    g = cells.groupby("origin_date")[["loss", "baseline_loss"]].sum()
    return {str(o): {"loss_sum": float(r.loss), "baseline_sum": float(r.baseline_loss)} for o, r in g.iterrows()}


def finite_or_none(x):
    """JSON-safe copy: non-finite floats become None (json.dumps(allow_nan=False) would otherwise crash the run)."""
    if isinstance(x, dict):
        return {k: finite_or_none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite_or_none(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


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

    def sanity(self, fc: pd.DataFrame, origins) -> dict:
        """Sanity report; last values are only looked up for the evaluated origins (rows claiming any other origin
        get no reference value), so candidate-chosen dates can never read truth beyond the evaluated origins."""
        c = self._c
        allowed = {str(o)[:10] for o in origins}
        pairs = fc[["series_id", "origin_date"]].astype(str).drop_duplicates()
        vals = {}
        for sid, o in pairs.itertuples(index=False):
            if o[:10] not in allowed:
                vals[(sid, o[:10])] = np.nan
                continue
            lv = self.truth.last_value(sid, c.day_of(c.parse_date(o[:10])))
            vals[(sid, o[:10])] = np.nan if lv is None else float(lv[0])
        lvs = pd.Series(vals, dtype=float)
        if len(lvs):
            lvs.index = pd.MultiIndex.from_tuples(lvs.index)
        else:
            lvs.index = pd.MultiIndex.from_arrays([[], []])
        return sanity_report(fc, lvs, self.cfg.gates.sanity_max_abs_log_ratio)

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
        d["loss_by_origin"] = loss_by_origin(cells)   # weights of the paired per-origin error of a skill delta
        cells = _diag_columns(cells)
        total = cells["loss"].sum()
        d["loss_decomposition"] = _components(cells)
        hz = cells.groupby("horizon_days")
        bh = hz.agg(n_cells=("loss", "size"), loss=("loss", "mean"), baseline_loss=("baseline_loss", "mean"),
                    loss_sum=("loss", "sum"), coverage_80=("inside", "mean"), q10_breach_rate=("below", "mean"),
                    q90_breach_rate=("above", "mean"), mean_width_rel=("width_rel", "mean"),
                    median_width_rel=("width_rel", "median"), mean_bias_rel=("bias_rel", "mean"))
        bh["skill"] = 1 - bh["loss"] / bh["baseline_loss"].replace(0, np.nan)
        bh["share_of_total_loss"] = bh["loss_sum"] / total if total else np.nan
        d["by_horizon"] = {str(h): {**_records(bh.loc[[h]].drop(columns=["loss_sum"]))[0],
                                    "loss_components": _components(g)} for h, g in hz}
        sl = cells.groupby(["instrument_class", "pricing_tier", "horizon_days"]).agg(
            n_cells=("loss", "size"), loss=("loss", "mean"), baseline_loss=("baseline_loss", "mean"),
            excess_loss_sum=("excess", "sum"), coverage_80=("inside", "mean"), q10_breach_rate=("below", "mean"),
            q90_breach_rate=("above", "mean"), mean_width_rel=("width_rel", "mean"),
            mean_bias_rel=("bias_rel", "mean")).reset_index()
        sl["skill"] = 1 - sl["loss"] / sl["baseline_loss"].replace(0, np.nan)
        sl["share_of_total_loss"] = sl["loss"] * sl["n_cells"] / total
        d["slices"] = _records(sl.sort_values("excess_loss_sum", ascending=False))
        d["worst_slices"] = d["slices"][:6]
        ser = cells.groupby("series_id").agg(
            instrument_class=("instrument_class", "first"), pricing_tier=("pricing_tier", "first"),
            n_cells=("loss", "size"), loss=("loss", "mean"), baseline_loss=("baseline_loss", "mean"),
            loss_sum=("loss", "sum"), excess_loss_sum=("excess", "sum"), coverage_80=("inside", "mean"),
            mean_level=("last_value", "mean"), mean_q50_error=("q50", "mean"), mean_y=("y", "mean"),
            mean_width_rel=("width_rel", "mean")).reset_index()
        ser["mean_q50_error"] = ser["mean_q50_error"] - ser["mean_y"]
        ser["skill"] = 1 - ser["loss"] / ser["baseline_loss"].replace(0, np.nan)
        ser["share_of_total_loss"] = ser["loss_sum"] / total
        ser = ser.drop(columns=["loss_sum"])
        d["worst_series"] = _records(ser.sort_values("excess_loss_sum", ascending=False).head(5))
        d["best_series"] = _records(ser.sort_values("excess_loss_sum").head(3))
        d["largest_loss_series"] = _records(ser.sort_values("share_of_total_loss", ascending=False).head(5))
        return d

    # ---------------------------------------------------------------------------- top level ----
    def evaluate(self, forecaster, origins, expose_diagnostics: bool, with_sanity: bool = False):
        fc, info = self.forecast(forecaster, origins)
        sanity = self.sanity(fc, origins) if with_sanity or expose_diagnostics else None
        cells, sinfo = self.score(fc, origins)
        if not expose_diagnostics:
            agg = self.aggregates(cells)
            out = {k: agg[k] for k in HOLDOUT_KEYS}
            self.last_loss_by_origin = loss_by_origin(cells)   # lab-only (never part of the released aggregates)
            return (out, sanity) if with_sanity else out
        d = self.diagnostics(cells)
        d["scoring"] = sinfo
        d["runtime"] = info
        d["sanity"] = sanity
        return (d, sanity) if with_sanity else d

    def benchmark_skill(self, forecaster, origin: str = BENCHMARK_ORIGIN) -> dict:
        fc, _ = self.forecast(forecaster, [origin])
        cells, _ = self.score(fc, [origin])
        return self.aggregates(cells)

    def full_run(self, forecaster_path: str | Path, phase: str = "full") -> dict:
        """Discovery (diagnostics) and/or Holdout (firewalled) + lab-only benchmark origin, one process.

        phase="discovery" never touches Holdout; phase="holdout" never computes Discovery diagnostics.
        """
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        f = load_forecaster(forecaster_path)
        t0 = time.time()
        out = {"stage": self.cfg.stage, "phase": phase}
        if phase in ("full", "discovery"):
            out["discovery"] = self.evaluate(f, list(self.cfg.discovery_origins), True)
        if phase in ("full", "holdout"):
            if phase == "full":
                f = load_forecaster(forecaster_path)  # fresh instance: same Holdout numbers as the two-stage path
            hold, hs = self.evaluate(f, list(self.cfg.holdout_origins), False, with_sanity=True)
            by_origin = self.last_loss_by_origin
            bench = self.benchmark_skill(f)
            # holdout_by_origin is lab-only: the controller keeps it out of the ledger and of everything the agent sees
            out.update(holdout=hold, holdout_sanity_ok=bool(hs["ok"]), holdout_by_origin=by_origin,
                       benchmark_2026_10_24={k: bench[k] for k in HOLDOUT_KEYS})
        out["seconds"] = round(time.time() - t0, 2)
        return out

    # ------------------------------------------------------------------ dense paired backtest ----
    def paired_backtest(self, cand_path: str | Path, ref_path: str | Path, origins: list[str],
                        series_ids: list[str] | None = None) -> dict:
        """Candidate vs reference on dense pseudo-origins inside the Discovery window (see backtest_origins).

        Same scoring as Discovery, paired cell by cell: skill delta with an origin-block bootstrap standard error, the
        share of origins that improve, deltas by horizon / slice / series, and per-cell residuals of a few series."""
        bad = sorted(set(origins) - set(backtest_origins(self.cfg, 1)))
        if bad or not origins:
            raise PermissionError(f"backtest origins must lie in the Discovery window; bad: {bad[:5]}")
        out, cells = {}, {}
        for tag, path in (("candidate", cand_path), ("reference", ref_path)):
            fc, info = self.forecast(load_forecaster(path), origins)
            cells[tag], _ = self.score(fc, origins)
            secs = list(info["seconds_by_origin"].values())
            out[f"{tag}_seconds_per_origin"] = {"mean": round(float(np.mean(secs)), 3), "max": round(max(secs), 3)}
        k = ["series_id", "origin_date", "horizon_days"]
        c = _diag_columns(cells["candidate"])
        m = c.merge(cells["reference"][k + ["q10", "q50", "q90", "loss"]], on=k, suffixes=("", "_ref"))
        m["gain"] = m["loss_ref"] - m["loss"]
        base = m["baseline_loss"].sum()
        by_o = m.groupby("origin_date")[["loss", "loss_ref", "baseline_loss"]].sum()
        by_o = by_o.sort_index()
        # origins closer than the longest horizon share target dates: bootstrap them in blocks, not one by one
        block = overlap_block(by_o.index, max(self.cfg.horizons))
        st = paired_delta_stats(by_o["loss"], by_o["loss_ref"], by_o["baseline_loss"], block)
        iid = paired_delta_stats(by_o["loss"], by_o["loss_ref"], by_o["baseline_loss"], 1)
        delta, se = st["pooled_skill_delta"], st["delta_se"]
        out.update({
            "n_origins": len(origins), "n_cells": len(m), "origins": [origins[0], origins[-1]],
            "candidate": self.aggregates(cells["candidate"]), "reference": self.aggregates(cells["reference"]),
            "pooled_skill_delta": delta, "delta_se_origin_bootstrap": se, "delta_z": st["delta_z"],
            "bootstrap_block_origins": st["block_origins"], "n_independent_blocks": st["n_independent_blocks"],
            "delta_se_iid_origins": iid["delta_se"],   # ignores overlapping targets; too small on dense origins
            "share_origins_improved": float((by_o["loss_ref"] > by_o["loss"]).mean()),
            "share_cells_improved": float((m["gain"] > 0).mean()),
        })
        out["delta_by_horizon"] = {str(h): float(g["gain"].sum() / g["baseline_loss"].sum())
                                   for h, g in m.groupby("horizon_days")}
        sl = m.groupby(["instrument_class", "horizon_days"]).agg(
            n_cells=("loss", "size"), gain_sum=("gain", "sum"), baseline_sum=("baseline_loss", "sum"),
            coverage_80=("inside", "mean")).reset_index()
        sl["skill_delta_within_slice"] = sl["gain_sum"] / sl["baseline_sum"]
        sl["contribution_to_pooled_delta"] = sl["gain_sum"] / base
        out["delta_by_slice"] = _records(sl.drop(columns=["gain_sum", "baseline_sum"]))
        ser = m.groupby("series_id").agg(
            instrument_class=("instrument_class", "first"), loss=("loss", "sum"), loss_ref=("loss_ref", "sum"),
            baseline=("baseline_loss", "sum"), coverage_80=("inside", "mean"), mean_bias_rel=("bias_rel", "mean"),
            mean_width_rel=("width_rel", "mean")).reset_index()
        ser["skill"] = 1 - ser["loss"] / ser["baseline"].replace(0, np.nan)
        ser["skill_ref"] = 1 - ser["loss_ref"] / ser["baseline"].replace(0, np.nan)
        ser["contribution_to_pooled_delta"] = (ser["loss_ref"] - ser["loss"]) / base
        ser = ser.drop(columns=["loss", "loss_ref", "baseline"]).sort_values("contribution_to_pooled_delta")
        out["series_most_hurt"] = _records(ser.head(8))
        out["series_most_helped"] = _records(ser.tail(8).iloc[::-1])
        out["series_skill_quantiles"] = {q: float(ser["skill"].quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        sids = list(dict.fromkeys(series_ids or []))[:MAX_BACKTEST_SERIES]
        cols = k + ["last_value", "y", "q10", "q50", "q90", "q10_ref", "q50_ref", "q90_ref", "loss", "loss_ref",
                    "baseline_loss"]
        out["series_cells"] = {sid: _records(m.loc[m["series_id"] == sid, cols]) for sid in sids}
        return out

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


def backtest_origins(cfg: LabConfig, stride_days: int = 1, start: str | None = None,
                     end: str | None = None) -> list[str]:
    """Dense pseudo-origins (every `stride_days`, counted back from `end`) inside the Discovery window: on or after the
    first Discovery origin, and early enough that every target date precedes the first Holdout origin's targets."""
    import datetime as dt

    d0, d1 = dt.date.fromisoformat(cfg.discovery_origins[0]), dt.date.fromisoformat(cfg.discovery_origins[-1])
    last_ok = dt.date.fromisoformat(cfg.holdout_origins[0]) - dt.timedelta(days=max(cfg.horizons))
    lo = max(d0, dt.date.fromisoformat(start)) if start else d0
    hi = min(d1, last_ok, dt.date.fromisoformat(end)) if end else min(d1, last_ok)
    step = max(1, int(stride_days))
    out, d = [], hi
    while d >= lo:
        out.append(d.isoformat())
        d -= dt.timedelta(days=step)
    return out[::-1]


def _diag_columns(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-cell diagnostic columns: band position, relative width / bias and the five pinball components.

    cell loss = mean over q10/q50/q90 of pinball; each tail quantile's pinball splits into a breach part (truth outside
    that quantile, weight 0.9) and a width part (truth inside, weight 0.1), so the components sum to 3 x loss.
    """
    y, lv = cells["y"], cells["last_value"].where(cells["last_value"] != 0)
    below, above = y < cells["q10"], y > cells["q90"]
    e10, e90 = y - cells["q10"], y - cells["q90"]
    return cells.assign(
        excess=cells["loss"] - cells["baseline_loss"], inside=~below & ~above, below=below, above=above,
        width_rel=(cells["q90"] - cells["q10"]) / lv, bias_rel=(cells["q50"] - y) / lv,
        c_q10_breach=np.where(below, -0.9 * e10, 0.0), c_q10_width=np.where(below, 0.0, 0.1 * e10),
        c_q50=np.maximum(0.5 * (y - cells["q50"]), -0.5 * (y - cells["q50"])),
        c_q90_width=np.where(above, 0.0, -0.1 * e90), c_q90_breach=np.where(above, 0.9 * e90, 0.0))


def _components(cells: pd.DataFrame) -> dict:
    """Share of the pinball loss coming from each component (sums to 1)."""
    parts = {k: float(cells[f"c_{k}"].sum()) for k in ("q10_breach", "q10_width", "q50", "q90_width", "q90_breach")}
    tot = sum(parts.values())
    return {k: (round(v / tot, 4) if tot > 0 and math.isfinite(tot) else None) for k, v in parts.items()}


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
    ap.add_argument("--phase", choices=PHASES + ("backtest",), default="full")
    ap.add_argument("--reference", help="backtest: reference forecaster (default: the candidate is compared to it)")
    ap.add_argument("--origins", default="", help="backtest: comma-separated pseudo-origins")
    ap.add_argument("--series", default="", help="backtest: comma-separated series_ids to return per-cell residuals for")
    a = ap.parse_args()
    cfg = load_config(enable_exa=a.enable_exa, exa_cache_only=a.exa_cache_only)
    try:
        ev = Evaluator(cfg)
        if a.phase == "backtest":
            split = lambda x: [v.strip() for v in x.split(",") if v.strip()]
            run = ev.paired_backtest(a.forecaster, a.reference, split(a.origins), split(a.series))
        else:
            run = ev.full_run(a.forecaster, a.phase)
        res = {"ok": True, **finite_or_none(run)}
    except Exception as e:  # reported to the controller as a RUNTIME_ERROR
        tb = traceback.format_exc().splitlines()
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000], "traceback_tail": "\n".join(tb[-25:])}
    Path(a.out).write_text(json.dumps(res, indent=2, default=_json_default, allow_nan=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
