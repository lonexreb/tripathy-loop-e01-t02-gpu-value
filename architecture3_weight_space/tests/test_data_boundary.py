import csv
from datetime import date
from pathlib import Path

import pytest

from tripathy_arch3.data_boundary import (
    ACTOR_SERIES_COLUMNS,
    ACTOR_SERIES_FILENAME,
    ActorInputBundle,
    DataBoundaryError,
    build_actor_input_bundle,
)


def write_source(path: Path) -> None:
    rows = [
        {
            "series_id": "target-a",
            "date": "2026-10-01",
            "value": "1.0",
            "statistic": "median",
            "n_obs": "2",
            "n_providers": "1",
            "is_filled": "0",
            "source_id": "source-a",
            "instrument_class": "marketplace_ask",
            "source_url": "https://example.com/a",
            "licence_id": "CC-BY-4.0",
        },
        {
            "series_id": "other-series",
            "date": "2026-10-01",
            "value": "9.0",
            "statistic": "median",
            "n_obs": "2",
            "n_providers": "1",
            "is_filled": "0",
            "source_id": "source-b",
            "instrument_class": "marketplace_ask",
            "source_url": "https://example.com/b",
            "licence_id": "CC-BY-4.0",
        },
        {
            "series_id": "target-a",
            "date": "2026-10-03",
            "value": "2.0",
            "statistic": "median",
            "n_obs": "2",
            "n_providers": "1",
            "is_filled": "0",
            "source_id": "source-a",
            "instrument_class": "marketplace_ask",
            "source_url": "https://example.com/a",
            "licence_id": "CC-BY-4.0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTOR_SERIES_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_actor_bundle_excludes_future_truth_and_out_of_scope_series(tmp_path: Path) -> None:
    source = tmp_path / "full_truth.csv"
    write_source(source)
    bundle = build_actor_input_bundle(
        source_series=source,
        bundle_root=tmp_path / "bundles",
        cutoff_date=date(2026, 10, 2),
        target_ids=("target-a",),
    )

    assert bundle.manifest.row_count == 1
    assert bundle.manifest.max_date == date(2026, 10, 1)
    assert bundle.manifest.excluded_after_cutoff == 1
    assert bundle.manifest.excluded_out_of_scope == 1
    history = bundle.read_bytes(ACTOR_SERIES_FILENAME).decode("utf-8")
    assert "2026-10-03" not in history
    assert "other-series" not in history
    assert str(source.resolve()) not in (bundle.root / "manifest.json").read_text(encoding="utf-8")

    reused = build_actor_input_bundle(
        source_series=source,
        bundle_root=tmp_path / "bundles",
        cutoff_date=date(2026, 10, 2),
        target_ids=("target-a",),
    )
    assert reused.root == bundle.root


def test_actor_bundle_denies_traversal_extra_files_and_tampering(tmp_path: Path) -> None:
    source = tmp_path / "full_truth.csv"
    write_source(source)
    bundle = build_actor_input_bundle(
        source_series=source,
        bundle_root=tmp_path / "bundles",
        cutoff_date=date(2026, 10, 2),
        target_ids=("target-a",),
    )

    with pytest.raises(DataBoundaryError, match="not allowlisted|relative"):
        bundle.read_bytes("../full_truth.csv")

    extra = bundle.root / "unexpected.txt"
    extra.write_text("not allowed", encoding="utf-8")
    with pytest.raises(DataBoundaryError, match="differs from manifest"):
        ActorInputBundle.load(bundle.root)
    extra.unlink()

    history = bundle.root / ACTOR_SERIES_FILENAME
    history.write_text(history.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(DataBoundaryError, match="hash mismatch"):
        ActorInputBundle.load(bundle.root)


def test_bundle_requires_every_frozen_target_to_have_history(tmp_path: Path) -> None:
    source = tmp_path / "full_truth.csv"
    write_source(source)
    with pytest.raises(DataBoundaryError, match="no history"):
        build_actor_input_bundle(
            source_series=source,
            bundle_root=tmp_path / "bundles",
            cutoff_date=date(2026, 10, 2),
            target_ids=("target-a", "missing-target"),
        )
