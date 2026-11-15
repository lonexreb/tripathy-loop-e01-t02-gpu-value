"""Agent workbench (immutable to the agent): in-iteration tools that never touch Holdout.

* CandidateWorkbench: a draft copy of the champion that the agent edits with small exact-match str-replace
  edits (or rewrites), so a 120-250 line forecaster never has to be re-emitted as one giant tool argument.
* `python -m lab.workbench profile`: lab-side data-quality / table-join profile of one point-in-time slice.
* `python -m lab.workbench analyze`: runs agent analysis code `analyze(pack_slice, origin_date)` on point-in-time
  Discovery slices under the same source firewall as forecasters (no I/O, no network, allowed imports only).
The CLI runs in a child process started by RSIController with a timeout; only firewalled JSON comes back.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

from lab.sandbox import audit_forecaster_source

MAX_ANALYSIS_OUTPUT_CHARS = 20_000
_NEEDS_FORECASTER = "module must define a top-level class `Forecaster`"


class CandidateWorkbench:
    """Draft forecaster buffer for one iteration (starts as the champion)."""

    def __init__(self, base_code: str):
        self.base_code = base_code
        self.code = base_code
        self.n_edits = 0

    def view(self, start_line: int = 1, end_line: int | None = None) -> str:
        lines = self.code.splitlines()
        start = max(1, int(start_line or 1))
        end = len(lines) if not end_line else min(int(end_line), len(lines))
        return "\n".join(f"{i:4d}| {lines[i - 1]}" for i in range(start, end + 1))

    def replace(self, old_str: str, new_str: str) -> dict:
        if not old_str:
            return {"ok": False, "error": "old_str must be a non-empty exact snippet of the current candidate"}
        n = self.code.count(old_str)
        if n != 1:
            hint = ("copy the snippet exactly (indentation included) from view_candidate" if n == 0 else
                    "include more surrounding lines so the snippet is unique")
            return {"ok": False, "error": f"old_str must match exactly once; found {n} matches ({hint})"}
        self.code = self.code.replace(old_str, new_str, 1)
        self.n_edits += 1
        return self.status()

    def write(self, code: str) -> dict:
        self.code = code
        self.n_edits += 1
        return self.status()

    def reset(self) -> dict:
        self.code, self.n_edits = self.base_code, 0
        return self.status()

    def diff(self, max_chars: int = 12_000) -> str:
        d = "".join(difflib.unified_diff(self.base_code.splitlines(True), self.code.splitlines(True),
                                         "champion/forecaster.py", "candidate/forecaster.py", n=2))
        return d if len(d) <= max_chars else d[:max_chars] + "\n... [diff truncated]"

    def status(self) -> dict:
        return {"ok": True, "n_lines": len(self.code.splitlines()), "n_edits": self.n_edits,
                "identical_to_champion": self.code == self.base_code,
                "audit_problems": audit_forecaster_source(self.code)}


def audit_analysis_source(code: str) -> list[str]:
    """Same firewall as forecasters, but the module must define a top-level `analyze(pack_slice, origin_date)`."""
    problems = [p for p in audit_forecaster_source(code) if not p.startswith(_NEEDS_FORECASTER)]
    if problems and problems[0].startswith("SyntaxError"):
        return problems
    tree = ast.parse(code)
    if not any(isinstance(n, ast.FunctionDef) and n.name == "analyze" for n in tree.body):
        problems.append("module must define a top-level function `analyze(pack_slice, origin_date)`")
    return sorted(set(problems))


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"agent_module_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_jsonable(x, max_rows: int = 200):
    """DataFrames/Series/numpy values -> plain JSON values (DataFrames are capped at max_rows records)."""
    import numpy as np
    import pandas as pd

    if isinstance(x, pd.DataFrame):
        return {"columns": [str(c) for c in x.columns], "n_rows": int(len(x)),
                "records": to_jsonable(x.head(max_rows).to_dict("records"), max_rows)}
    if isinstance(x, pd.Series):
        return to_jsonable(x.head(max_rows).to_dict(), max_rows)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v, max_rows) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, max_rows) for v in list(x)[:max_rows]]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist()[:max_rows], max_rows)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return str(x)


def clip_json(obj, max_chars: int) -> str:
    """Serialise to JSON that is always valid: oversized payloads are replaced by a truncated string + note."""
    s = json.dumps(obj, default=str)
    if len(s) <= max_chars:
        return s
    return json.dumps({"truncated": True, "original_chars": len(s), "head": s[: max(0, max_chars - 200)]})


def run_analysis(code_path: Path, origins: list[str], cfg) -> dict:
    from lab.sandbox import PointInTimeSandbox

    bad = [o for o in origins if o not in cfg.discovery_origins]
    if bad:
        raise PermissionError(f"analysis origins must be Discovery origins; got {bad}")
    mod, sb, out = _load_module(code_path), PointInTimeSandbox(cfg), {}
    for o in origins:
        out[o] = to_jsonable(mod.analyze(sb.get_pack_slice(o), o))
    return {"results": out}


DATE_COLUMNS = ("available_date", "date", "week_start", "print_date", "release_date")


def profile_pack_slice(sl: dict, origin_date: str) -> dict:
    """Lab-side data-quality / table-usage profile of one point-in-time pack slice (what is there, what joins to
    the targets, which target series are stale, filled or gappy)."""
    import numpy as np
    import pandas as pd

    od = pd.Timestamp(origin_date)
    tables = {}
    for name, df in sl.items():
        info = {"rows": len(df), "columns": list(df.columns)}
        nulls = df.isna().mean() if len(df) else pd.Series(dtype=float)
        info["null_share"] = {c: round(float(v), 3) for c, v in nulls.items() if v > 0}
        dc = next((c for c in DATE_COLUMNS if c in df.columns), None)
        if dc and len(df):
            info["date_range"] = [str(df[dc].min()), str(df[dc].max())]
        tables[name] = info
    t, h = sl["targets"], sl["series_history"]
    fam = t.set_index("series_id")["gpu_family"] if "gpu_family" in t.columns else pd.Series(dtype=str)
    joins = {}
    for name, df in sl.items():
        if name in ("targets", "catalog") or "gpu_family" not in df.columns:
            continue
        have = set(df["gpu_family"])
        dc = next((c for c in DATE_COLUMNS if c in df.columns), None)
        recent = df[pd.to_datetime(df[dc], errors="coerce") > od - pd.Timedelta(days=28)] if dc else df
        joins[name] = {"targets_with_rows": int(fam.isin(have).sum()), "n_targets": len(fam),
                       "targets_with_rows_last_28d": int(fam.isin(set(recent["gpu_family"])).sum()),
                       "rows_last_28d_for_target_families": int(recent["gpu_family"].isin(set(fam)).sum())}
    ht = h[h["series_id"].isin(t["series_id"])].copy()
    ht["d"] = pd.to_datetime(ht["date"])
    rows = []
    for sid, g in ht.sort_values(["series_id", "d"]).groupby("series_id"):
        v, d = g["value"].to_numpy(float), g["d"]
        r90 = g[d > od - pd.Timedelta(days=90)]
        span = int((d.iloc[-1] - d.iloc[0]).days) + 1
        ch = np.flatnonzero(np.diff(v) != 0)
        lr = np.diff(np.log(r90["value"].to_numpy(float))) if len(r90) > 1 else np.array([])
        rows.append({"series_id": sid, "n_days": len(g), "first_date": str(d.iloc[0].date()),
                     "days_stale": int((od - d.iloc[-1]).days), "missing_days_in_span": span - len(g),
                     "filled_share_90d": float(r90["is_filled"].astype(float).mean()) if len(r90) else None,
                     "days_since_change": int((od - d.iloc[ch[-1] + 1]).days) if len(ch) else span,
                     "n_changes_90d": int((lr != 0).sum()), "max_abs_log_jump_90d": float(np.abs(lr).max()) if len(lr) else 0.0,
                     "median_n_obs_90d": float(r90["n_obs"].median()) if len(r90) else None})
    per = pd.DataFrame(rows)
    if len(per):
        per = per.merge(t[["series_id", "instrument_class"]], on="series_id", how="left")
        num = [c for c in per.columns if c not in ("series_id", "instrument_class", "first_date")]
        by_cls = per.groupby("instrument_class")[num].median().round(3)
        by_cls["n_series"] = per.groupby("instrument_class").size()
        per["flag"] = ((per["days_stale"] > 2).astype(int) + (per["filled_share_90d"].fillna(0) > 0.2).astype(int)
                       + (per["missing_days_in_span"] > 30).astype(int) + (per["n_changes_90d"] == 0).astype(int))
        flagged = per.sort_values(["flag", "days_stale", "filled_share_90d"], ascending=False).head(12)
    else:
        by_cls, flagged = pd.DataFrame(), per
    return {"origin_date": origin_date, "tables": tables, "target_joins_by_gpu_family": joins,
            "missing_targets": sorted(set(t["series_id"]) - set(ht["series_id"])),
            "targets_by_instrument_class": to_jsonable(by_cls.reset_index()),
            "flagged_target_series": to_jsonable(flagged)}


def run_profile(origin: str, cfg) -> dict:
    from lab.sandbox import PointInTimeSandbox

    if origin > cfg.discovery_origins[-1]:
        raise PermissionError(f"profile origin must be <= the last Discovery origin {cfg.discovery_origins[-1]}")
    return profile_pack_slice(PointInTimeSandbox(cfg).get_pack_slice(origin), origin)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--code", required=True)
    a.add_argument("--origins", required=True, help="comma-separated Discovery origins")
    pr = sub.add_parser("profile")
    pr.add_argument("--origin", required=True)
    for p in (a, pr):
        p.add_argument("--out", required=True)
        p.add_argument("--enable-exa", action="store_true")
        p.add_argument("--exa-cache-only", action="store_true")
    args = ap.parse_args()
    from lab.config import load_config
    from lab.evaluator import _json_default

    cfg = load_config(enable_exa=args.enable_exa, exa_cache_only=args.exa_cache_only)
    try:
        if args.cmd == "profile":
            res = {"ok": True, **run_profile(args.origin, cfg)}
        else:
            res = {"ok": True, **run_analysis(Path(args.code), [o.strip() for o in args.origins.split(",") if o.strip()],
                                              cfg)}
    except Exception as e:
        tb = traceback.format_exc().splitlines()
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:2000], "traceback_tail": "\n".join(tb[-25:])}
    Path(args.out).write_text(json.dumps(res, default=_json_default, allow_nan=False))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
