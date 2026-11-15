"""Point-in-time data slicer and Exa time-travel guard (immutable to the agent).

Every table handed to the forecaster is cut at the forecast origin. A row is visible at origin D only
when its *availability date* is on or before D:
  series_history       date                    (the starter's daily target series, build_series.py)
  daily_statistics     date
  aggregator_medians   week_start + 6 days     (a Monday-dated weekly median covers the whole week)
  index_prints         print_date (+6 days for the weekly gpueconomy prints); withheld rows dropped
  hardware_fundamentals release_date           (announced-but-unreleased parts are hidden)
The capability ladder/units are cross-sectional benchmark composites (not time-indexed) and are passed as is.
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from lab.config import CATALOG_AS_OF, FINAL_ORIGIN, SCORED_TIER, LabConfig, load_config

STATIC_CATALOG_COLUMNS = [
    "series_id", "table", "source_id", "instrument_class", "statistic", "gpu_model", "pricing_tier",
    "instance_type", "az_global_id", "members",
]
SERIES_COLUMNS = ["series_id", "date", "value", "n_obs", "n_providers", "is_filled", "instrument_class"]
TIME_INDEXED = ("series_history", "daily_statistics", "aggregator_medians", "index_prints", "hardware_fundamentals")


def _iso_plus(dates: pd.Series, days: int | pd.Series) -> pd.Series:
    return (pd.to_datetime(dates, format="%Y-%m-%d", errors="coerce") + pd.to_timedelta(days, unit="D")).dt.strftime("%Y-%m-%d")


def run_build_series(cfg: LabConfig, out: Path, catalog: Path, cutoff: str | None = None,
                     as_of: str | None = None, members_from: Path | None = None) -> None:
    """Run the official starter build_series.py (python -I -B so nothing is written into the pack)."""
    cmd = [sys.executable, "-I", "-B", str(cfg.starter_dir / "build_series.py"), "--pack", str(cfg.pack_dir),
           "--out", str(out), "--catalog", str(catalog), "--report", str(out.with_suffix(".report.json")),
           "--tmp-dir", str(out.parent / "_duck_tmp")]
    if cutoff:
        cmd += ["--cutoff", cutoff]
    if as_of:
        cmd += ["--as-of", as_of]
    if members_from:
        cmd += ["--members-from", str(members_from)]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True)


class PointInTimeSandbox:
    """Loads the pack once and serves origin-truncated slices; owns the Exa point-in-time guard."""

    def __init__(self, cfg: LabConfig | None = None):
        self.cfg = cfg or load_config()

    # ---------------------------------------------------------------- derived series cache ----
    @property
    def series_path(self) -> Path:
        return self.cfg.series_cache_dir / "series_daily.csv"

    @property
    def catalog_path(self) -> Path:
        return self.cfg.series_cache_dir / "series_catalog.csv"

    @property
    def final_catalog_path(self) -> Path:
        return self.cfg.series_cache_dir / f"catalog_{FINAL_ORIGIN}.csv"

    def ensure_series_cache(self) -> None:
        if not (self.series_path.exists() and self.catalog_path.exists()):
            run_build_series(self.cfg, self.series_path, self.catalog_path, as_of=CATALOG_AS_OF)
        if not self.final_catalog_path.exists():
            tmp = self.cfg.series_cache_dir / f"history_{FINAL_ORIGIN}.csv"
            run_build_series(self.cfg, tmp, self.final_catalog_path, cutoff=FINAL_ORIGIN, members_from=self.catalog_path)

    # ------------------------------------------------------------------------- base tables ----
    @cached_property
    def series_full(self) -> pd.DataFrame:
        """The full series file (truth AND history source). Never handed to a forecaster un-truncated."""
        self.ensure_series_cache()
        df = pd.read_csv(self.series_path, dtype={"series_id": str, "date": str}, low_memory=False)
        return df[SERIES_COLUMNS].sort_values(["series_id", "date"]).reset_index(drop=True)

    @cached_property
    def catalog(self) -> pd.DataFrame:
        self.ensure_series_cache()
        return pd.read_csv(self.catalog_path, dtype=str, keep_default_na=False)

    @cached_property
    def scored_ids(self) -> list[str]:
        c = self.catalog
        return sorted(c.loc[c["tier"] == SCORED_TIER, "series_id"].tolist())

    @cached_property
    def final_catalog(self) -> pd.DataFrame:
        self.ensure_series_cache()
        return pd.read_csv(self.final_catalog_path, dtype=str, keep_default_na=False)

    @cached_property
    def final_ids(self) -> list[str]:
        """Series forecast at the frozen origin: the 11-06 headline tier plus the 97 benchmark series."""
        c = self.final_catalog
        return sorted(set(c.loc[c["tier"] == SCORED_TIER, "series_id"]) | set(self.scored_ids))

    @cached_property
    def _tables(self) -> dict[str, pd.DataFrame]:
        p = self.cfg.pack_dir
        ds = pd.read_csv(p / "daily_statistics.csv", dtype={"date": str}, low_memory=False)
        ds = ds.drop(columns=["source_url", "licence_id"])
        ds["available_date"] = ds["date"]
        ag = pd.read_csv(p / "aggregator_medians.csv", dtype={"week_start": str}, low_memory=False)
        ag = ag[["week_start", "gpu_model", "pricing_mode", "median_usd_per_gpu_hour", "n_providers", "source_id",
                 "instrument_class", "gpu_slug", "reservation_months", "provider_median_usd_per_gpu_hour",
                 "min_usd_per_gpu_hour", "max_usd_per_gpu_hour", "n_offerings"]].copy()
        ag["available_date"] = _iso_plus(ag["week_start"], 6)
        ix = pd.read_csv(p / "index_prints.csv", dtype={"print_date": str}, low_memory=False)
        ix = ix[ix["status"] != "withheld"]
        ix = ix[["print_date", "index_id", "gpu_model", "value_usd_per_gpu_hour", "status", "source_id",
                 "instrument_class", "n_providers", "n_observations", "day_basis"]].copy()
        weekly = ix["index_id"].str.contains("weekly")
        ix["available_date"] = _iso_plus(ix["print_date"], np.where(weekly, 6, 0))
        hw = pd.read_csv(p / "hardware_fundamentals" / "hardware_fundamentals.csv", dtype={"release_date": str})
        hw = hw.drop(columns=["datasheet_url", "attribution", "source_url"], errors="ignore")
        hw = hw[hw["release_date"].notna()].copy()
        hw["release_date"] = hw["release_date"].str[:10]
        hw["available_date"] = hw["release_date"]
        cap = p / "capability"
        ladder = pd.read_csv(cap / "capability_ladder.csv", low_memory=False)
        units = pd.read_csv(cap / "capability_units.csv", low_memory=False)
        series = self.series_full.copy()
        series["available_date"] = series["date"]
        return {"series_history": series, "daily_statistics": ds, "aggregator_medians": ag, "index_prints": ix,
                "hardware_fundamentals": hw, "capability_ladder": ladder, "capability_units": units}

    # ------------------------------------------------------------------------------ slices ----
    def get_pack_slice(self, origin_date: str, target_ids: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Every pack table truncated at origin_date (rows available on or before it), plus the targets."""
        dt.date.fromisoformat(origin_date)
        t = self._tables
        out = {k: t[k].loc[t[k]["available_date"] <= origin_date].reset_index(drop=True) for k in TIME_INDEXED}
        out["capability_ladder"] = t["capability_ladder"].copy()
        out["capability_units"] = t["capability_units"].copy()
        cat = self.catalog[STATIC_CATALOG_COLUMNS]
        ids = target_ids if target_ids is not None else self.scored_ids
        out["catalog"] = cat.loc[cat["series_id"].isin(out["series_history"]["series_id"].unique())].reset_index(drop=True)
        tg = self.final_catalog if origin_date > CATALOG_AS_OF else self.catalog
        tg = pd.concat([cat, tg[STATIC_CATALOG_COLUMNS]]).drop_duplicates("series_id")
        out["targets"] = tg.loc[tg["series_id"].isin(ids)].sort_values("series_id").reset_index(drop=True)
        assert_point_in_time(out, origin_date)
        return out

    # ----------------------------------------------------------------- Exa time-travel guard ----
    def search_exa_point_in_time(self, query: str, origin_date: str, domains: list[str] | None = None,
                                 num_results: int = 8) -> list[dict]:
        if not self.cfg.enable_exa:
            raise PermissionError("Exa is disabled in this stage (enable_exa=False / PACK_ONLY)")
        dt.date.fromisoformat(origin_date)
        key = exa_cache_key(query, origin_date, domains, num_results)
        cache = self._exa_cache()
        if key in cache:
            return cache[key]
        api_key = os.environ.get("EXA_API_KEY")
        if self.cfg.exa_cache_only or not api_key:
            return []
        import httpx

        body = {"query": query, "type": "auto", "numResults": num_results,
                "endPublishedDate": f"{origin_date}T00:00:00.000Z", "contents": {"highlights": True}}
        if domains:
            body["includeDomains"] = list(domains)
        r = httpx.post("https://api.exa.ai/search", json=body, headers={"x-api-key": api_key}, timeout=60)
        r.raise_for_status()
        hits = filter_point_in_time(r.json().get("results", []), origin_date)
        rec = {"key": key, "query": query, "origin_date": origin_date, "domains": sorted(domains or []),
               "num_results": num_results, "results": hits,
               "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()}
        self.cfg.exa_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.exa_cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        cache[key] = hits
        return hits

    def _exa_cache(self) -> dict[str, list[dict]]:
        if not hasattr(self, "_exa_mem"):
            self._exa_mem = {}
            if self.cfg.exa_cache_path.exists():
                for line in self.cfg.exa_cache_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rec = json.loads(line)
                        self._exa_mem[rec["key"]] = filter_point_in_time(rec["results"], rec["origin_date"])
        return self._exa_mem


class ExaSignals:
    """The `external_signals` handle given to a forecaster in PACK_PLUS_EXA: searches are pinned to one origin."""

    def __init__(self, sandbox: PointInTimeSandbox, origin_date: str):
        self._sandbox, self.origin_date = sandbox, origin_date

    def search(self, query: str, domains: list[str] | None = None, num_results: int = 8) -> list[dict]:
        return self._sandbox.search_exa_point_in_time(query, self.origin_date, domains, num_results)


def exa_cache_key(query: str, origin_date: str, domains: list[str] | None, num_results: int) -> str:
    raw = json.dumps([query.strip(), origin_date, sorted(domains or []), int(num_results)])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def filter_point_in_time(results: list[dict], origin_date: str) -> list[dict]:
    """Keep only hits with a published date on or before the origin; normalise the record."""
    out = []
    for r in results:
        pub = r.get("publishedDate") or r.get("published_date")
        if not pub or str(pub)[:10] > origin_date:
            continue
        out.append({"title": r.get("title"), "url": r.get("url"), "published_date": str(pub),
                    "highlights": list(r.get("highlights") or [])})
    return out


def assert_point_in_time(slice_: dict[str, pd.DataFrame], origin_date: str) -> None:
    """Hard runtime check: no time-indexed table holds a row available or dated after the origin."""
    raw_cols = {"series_history": "date", "daily_statistics": "date", "aggregator_medians": "week_start",
                "index_prints": "print_date", "hardware_fundamentals": "release_date"}
    for name in TIME_INDEXED:
        df = slice_[name]
        if len(df):
            for col in ("available_date", raw_cols[name]):
                mx = str(df[col].max())
                if mx > origin_date:
                    raise AssertionError(f"leakage: {name}.{col} max {mx} > origin {origin_date}")


# ------------------------------------------------------------------- forecaster source audit ----
ALLOWED_IMPORTS = {"numpy", "pandas", "math", "statistics", "dataclasses", "typing", "collections", "functools",
                   "itertools", "re", "datetime", "warnings", "__future__", "bisect", "heapq", "operator", "enum"}
FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "locals",
                   "vars", "memoryview"}
FORBIDDEN_ATTR_PREFIXES = ("read_", "to_csv", "to_parquet", "to_pickle", "to_json", "to_sql", "to_excel",
                           "to_feather", "to_hdf", "to_stata", "loadtxt", "savetxt", "fromfile", "tofile", "genfromtxt")


def audit_forecaster_source(code: str) -> list[str]:
    """Information firewall for candidate code: no I/O, no network, no dynamic imports. Returns problems."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            mods = []
        problems += [f"import of '{m}' is not allowed (allowed: {sorted(ALLOWED_IMPORTS)})" for m in mods
                     if m not in ALLOWED_IMPORTS]
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"use of '{node.id}' is not allowed")
        if isinstance(node, ast.Attribute) and (node.attr.startswith(FORBIDDEN_ATTR_PREFIXES) or node.attr.startswith("__")
                                                and node.attr not in ("__init__", "__name__")):
            problems.append(f"attribute '.{node.attr}' is not allowed (no file I/O or dunder access)")
    has_cls = any(isinstance(n, ast.ClassDef) and n.name == "Forecaster" for n in tree.body)
    if not has_cls:
        problems.append("module must define a top-level class `Forecaster` with predict_origin(...)")
    return sorted(set(problems))
