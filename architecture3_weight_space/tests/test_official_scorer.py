import csv
import json
from datetime import date
from pathlib import Path

import pytest

from tripathy_arch3.config import WORKSPACE_ROOT
from tripathy_arch3.data_boundary import ACTOR_SERIES_COLUMNS, build_actor_input_bundle
from tripathy_arch3.official_scorer import (
    OfficialScorerAdapter,
    OfficialScorerError,
    ProtectedTruth,
)

REPO_ROOT = WORKSPACE_ROOT.parent


def write_series(path: Path) -> None:
    rows = []
    for day, value in (("2026-01-01", "1.0"), ("2026-01-02", "2.0"), ("2026-01-03", "3.0")):
        rows.append(
            {
                "series_id": "series-1",
                "date": day,
                "value": value,
                "statistic": "median",
                "n_obs": "1",
                "n_providers": "1",
                "is_filled": "0",
                "source_id": "source-1",
                "instrument_class": "marketplace_ask",
                "source_url": "https://example.com",
                "licence_id": "CC-BY-4.0",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTOR_SERIES_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_forecast(path: Path) -> None:
    path.write_text(
        "series_id,origin_date,horizon_days,q10,q50,q90,target_date\n"
        "series-1,2026-01-02,1,2.0,2.0,2.0,2026-01-03\n",
        encoding="utf-8",
    )


def scorer_adapter() -> OfficialScorerAdapter:
    return OfficialScorerAdapter(
        scorer_path=REPO_ROOT / "gpu-price-pack-v0.1.1-handover" / "starter" / "scorer.py",
        python_executable=REPO_ROOT / ".venv" / "bin" / "python",
    )


def test_official_scorer_uses_cutoff_history_and_protected_truth(tmp_path: Path) -> None:
    truth_path = tmp_path / "protected_truth.csv"
    forecast_path = tmp_path / "forecast.csv"
    write_series(truth_path)
    write_forecast(forecast_path)
    bundle = build_actor_input_bundle(
        source_series=truth_path,
        bundle_root=tmp_path / "bundles",
        cutoff_date=date(2026, 1, 2),
        target_ids=("series-1",),
    )
    result = scorer_adapter().score(
        forecast_path=forecast_path,
        history_bundle=bundle,
        protected_truth=ProtectedTruth.from_path(truth_path),
        output_directory=tmp_path / "evaluation",
        mode="frozen_cutoff",
        horizons=(1,),
        frozen_cutoff=date(2026, 1, 2),
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert result.return_code == 0
    assert report["status"] == "scored"
    assert report["cells"]["scored"] == 1
    assert report["inputs"]["history"]["sha256"] == bundle.manifest.artifacts[0].sha256
    assert truth_path.name not in bundle.read_bytes("series_history.csv").decode("utf-8")


def test_scorer_refuses_cutoff_mismatch_before_execution(tmp_path: Path) -> None:
    truth_path = tmp_path / "protected_truth.csv"
    forecast_path = tmp_path / "forecast.csv"
    write_series(truth_path)
    write_forecast(forecast_path)
    bundle = build_actor_input_bundle(
        source_series=truth_path,
        bundle_root=tmp_path / "bundles",
        cutoff_date=date(2026, 1, 2),
        target_ids=("series-1",),
    )
    with pytest.raises(OfficialScorerError, match="cutoff must match"):
        scorer_adapter().score(
            forecast_path=forecast_path,
            history_bundle=bundle,
            protected_truth=ProtectedTruth.from_path(truth_path),
            output_directory=tmp_path / "evaluation",
            mode="frozen_cutoff",
            horizons=(1,),
            frozen_cutoff=date(2026, 1, 1),
        )


def test_protected_truth_requires_evaluator_capability(tmp_path: Path) -> None:
    truth_path = tmp_path / "protected_truth.csv"
    write_series(truth_path)
    protected = ProtectedTruth.from_path(truth_path)
    with pytest.raises(PermissionError, match="only to the evaluator"):
        protected._path_for_evaluator(object())
