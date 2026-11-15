"""Run the local Phase 2 boundary/scorer acceptance against frozen repository artifacts."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from .audit import canonical_json, sha256_bytes, sha256_file
from .config import WORKSPACE_ROOT
from .data_boundary import build_actor_input_bundle, read_frozen_target_ids
from .official_scorer import OfficialScorerAdapter, ProtectedTruth

REPO_ROOT = WORKSPACE_ROOT.parent


def main() -> None:
    source_series = REPO_ROOT / "research" / "out" / "series_daily.csv"
    target_manifest = REPO_ROOT / "research" / "adapted_series.txt"
    forecast = REPO_ROOT / "research" / "out" / "bt_checkpoint_drift.csv"
    reference_report = REPO_ROOT / "research" / "out" / "report_checkpoint_drift.json"
    scorer = REPO_ROOT / "gpu-price-pack-v0.1.1-handover" / "starter" / "scorer.py"
    python_executable = REPO_ROOT / ".venv" / "bin" / "python"
    cutoff = date(2026, 10, 15)
    target_ids = read_frozen_target_ids(target_manifest)
    bundle = build_actor_input_bundle(
        source_series=source_series,
        bundle_root=WORKSPACE_ROOT / "artifacts" / "phase2" / "actor_bundles",
        cutoff_date=cutoff,
        target_ids=target_ids,
    )
    adapter_source = Path(__file__).with_name("official_scorer.py")
    identity = {
        "adapter_sha256": sha256_file(adapter_source),
        "bundle_digest": bundle.manifest.bundle_digest,
        "forecast_sha256": sha256_file(forecast),
        "history_sha256": bundle.manifest.artifacts[0].sha256,
        "reference_report_sha256": sha256_file(reference_report),
        "scorer_sha256": sha256_file(scorer),
    }
    acceptance_id = sha256_bytes(canonical_json(identity).encode("utf-8"))[:16]
    output_directory = WORKSPACE_ROOT / "artifacts" / "phase2" / "evaluations" / acceptance_id
    acceptance_path = output_directory / "acceptance.json"
    if acceptance_path.is_file():
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        print(json.dumps(acceptance, indent=2, sort_keys=True))
        if acceptance.get("status") != "passed":
            raise SystemExit(1)
        return

    result = OfficialScorerAdapter(
        scorer_path=scorer,
        python_executable=python_executable,
        expected_scorer_sha256=identity["scorer_sha256"],
    ).score(
        forecast_path=forecast,
        history_bundle=bundle,
        protected_truth=ProtectedTruth.from_path(source_series),
        output_directory=output_directory,
        mode="backtest",
    )
    actual = json.loads(result.report_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_report.read_text(encoding="utf-8"))
    comparisons = {
        "by_horizon": actual["by_horizon"] == reference["by_horizon"],
        "cells": actual["cells"] == reference["cells"],
        "overall": actual["overall"] == reference["overall"],
        "per_series": actual["per_series"] == reference["per_series"],
    }
    passed = (
        result.status == "scored"
        and all(comparisons.values())
        and bundle.manifest.max_date <= cutoff
        and bundle.manifest.target_ids == target_ids
    )
    acceptance = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "acceptance_id": acceptance_id,
        "bundle": {
            "bundle_id": bundle.manifest.bundle_id,
            "cutoff_date": bundle.manifest.cutoff_date.isoformat(),
            "max_date": bundle.manifest.max_date.isoformat(),
            "row_count": bundle.manifest.row_count,
            "target_count": len(bundle.manifest.target_ids),
            "allowlisted_files": sorted(artifact.path for artifact in bundle.manifest.artifacts),
        },
        "comparisons_to_frozen_reference": comparisons,
        "identity": identity,
        "score_receipt_sha256": sha256_file(result.receipt_path),
    }
    with acceptance_path.open("w", encoding="utf-8") as acceptance_handle:
        json.dump(acceptance, acceptance_handle, indent=2, sort_keys=True)
        acceptance_handle.write("\n")
        acceptance_handle.flush()
        os.fsync(acceptance_handle.fileno())
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
