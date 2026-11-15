import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from tripathy_arch3.data_boundary import ACTOR_SERIES_COLUMNS, build_actor_input_bundle
from tripathy_arch3.forecaster import (
    ForecasterPatch,
    build_actor_prompt,
    compile_forecast,
    parse_actor_patch,
)


def bundle(tmp_path: Path):  # type: ignore[no-untyped-def]
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTOR_SERIES_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for day, value in [("2026-10-01", "1.0"), ("2026-10-02", "1.2")]:
            writer.writerow(
                {
                    "series_id": "target-a",
                    "date": day,
                    "value": value,
                    "statistic": "median",
                    "n_obs": "2",
                    "n_providers": "1",
                    "is_filled": "0",
                    "source_id": "source-a",
                    "instrument_class": "marketplace_ask",
                    "source_url": "https://example.com/a",
                    "licence_id": "CC-BY-4.0",
                }
            )
    return build_actor_input_bundle(
        source_series=source,
        bundle_root=tmp_path / "bundles",
        cutoff_date=date(2026, 10, 2),
        target_ids=("target-a",),
    )


def test_actor_patch_parser_rejects_surrounding_prose_and_drift_tuning() -> None:
    parsed = parse_actor_patch(
        '{"action":"propose_forecaster_patch","method":"repricing_interval",'
        '"window_days":30,"width_scale":1.2}'
    )
    assert parsed.window_days == 30
    with pytest.raises(ValueError, match="only one JSON"):
        parse_actor_patch("Here is the answer: " + parsed.model_dump_json())
    with pytest.raises(ValueError, match="canonical"):
        ForecasterPatch(
            action="propose_forecaster_patch",
            method="drift",
            window_days=30,
            width_scale=Decimal("1"),
        )


def test_compiler_uses_only_bundle_history_and_emits_ordered_rows(tmp_path: Path) -> None:
    actor_bundle = bundle(tmp_path)
    patch = ForecasterPatch(
        action="propose_forecaster_patch",
        method="repricing_interval",
        window_days=30,
        width_scale=Decimal("1"),
    )
    rows = compile_forecast(
        actor_bundle,
        patch=patch,
        origin_date=date(2026, 10, 2),
    )
    assert len(rows) == 3
    assert all(row.q10 <= row.q50 <= row.q90 for row in rows)
    assert rows[0].q50 == Decimal("1.2")
    assert "series_summaries" in build_actor_prompt(actor_bundle, origin_date=date(2026, 10, 2))
