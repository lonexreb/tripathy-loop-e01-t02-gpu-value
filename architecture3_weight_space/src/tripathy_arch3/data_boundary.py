"""Build and enforce content-addressed, cutoff-safe actor input bundles."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .audit import canonical_json, sha256_bytes, sha256_file
from .contracts import ArtifactRef, NonEmpty, Sha256

ACTOR_SERIES_FILENAME = "series_history.csv"
BUNDLE_MANIFEST_FILENAME = "manifest.json"
ACTOR_SERIES_COLUMNS = (
    "series_id",
    "date",
    "value",
    "statistic",
    "n_obs",
    "n_providers",
    "is_filled",
    "source_id",
    "instrument_class",
    "source_url",
    "licence_id",
)


class DataBoundaryError(RuntimeError):
    """Raised when an as-of view or actor bundle violates its boundary."""


class ActorInputManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Annotated[int, Field(ge=1)] = 1
    bundle_id: NonEmpty
    created_at: datetime
    cutoff_date: date
    target_ids: tuple[NonEmpty, ...]
    source_name: NonEmpty
    source_sha256: Sha256
    builder_sha256: Sha256
    row_count: Annotated[int, Field(ge=1)]
    min_date: date
    max_date: date
    excluded_after_cutoff: Annotated[int, Field(ge=0)]
    excluded_out_of_scope: Annotated[int, Field(ge=0)]
    artifacts: tuple[ArtifactRef, ...]
    bundle_digest: Sha256

    @field_validator("created_at")
    @classmethod
    def timestamp_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if not self.target_ids or len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("target_ids must be non-empty and unique")
        if self.max_date > self.cutoff_date:
            raise ValueError("actor bundle contains a row after its cutoff")
        if self.min_date > self.max_date:
            raise ValueError("min_date may not be after max_date")
        paths = [artifact.path for artifact in self.artifacts]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("artifacts must be non-empty with unique paths")
        _, expected_digest = _bundle_identity(
            source_sha256=self.source_sha256,
            builder_sha256=self.builder_sha256,
            cutoff_date=self.cutoff_date,
            target_ids=self.target_ids,
        )
        if self.bundle_digest != expected_digest:
            raise ValueError("bundle_digest does not match the frozen bundle identity")
        return self


def read_frozen_target_ids(path: Path) -> tuple[str, ...]:
    target_ids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise DataBoundaryError("target manifest must contain unique target IDs")
    return target_ids


def _validate_relative_path(relative_path: str) -> PurePosixPath:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise DataBoundaryError("actor paths must be relative and may not contain '..'")
    return path


class ActorInputBundle:
    """Read-only capability for files explicitly listed in an actor bundle manifest."""

    def __init__(self, root: Path, manifest: ActorInputManifest) -> None:
        if root.is_symlink():
            raise DataBoundaryError("actor bundle root may not be a symlink")
        self.root = root.resolve()
        self.manifest = manifest
        self._artifacts = {artifact.path: artifact for artifact in manifest.artifacts}
        self.assert_isolated()

    @classmethod
    def load(cls, root: Path) -> Self:
        if root.is_symlink():
            raise DataBoundaryError("actor bundle root may not be a symlink")
        resolved_root = root.resolve()
        manifest_path = resolved_root / BUNDLE_MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise DataBoundaryError("actor bundle manifest is missing or is a symlink")
        manifest = ActorInputManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        return cls(resolved_root, manifest)

    def resolve_artifact(self, relative_path: str) -> Path:
        safe_path = _validate_relative_path(relative_path)
        artifact = self._artifacts.get(safe_path.as_posix())
        if artifact is None:
            raise DataBoundaryError(f"actor path is not allowlisted: {relative_path}")
        unresolved_candidate = self.root / safe_path
        if unresolved_candidate.is_symlink():
            raise DataBoundaryError("actor artifact may not be a symlink")
        candidate = unresolved_candidate.resolve()
        if not candidate.is_relative_to(self.root):
            raise DataBoundaryError("actor path escapes its bundle")
        if not candidate.is_file():
            raise DataBoundaryError("actor artifact is missing")
        if sha256_file(candidate) != artifact.sha256:
            raise DataBoundaryError(f"actor artifact hash mismatch: {relative_path}")
        return candidate

    def read_bytes(self, relative_path: str) -> bytes:
        return self.resolve_artifact(relative_path).read_bytes()

    def assert_isolated(self) -> None:
        allowed = {BUNDLE_MANIFEST_FILENAME, *self._artifacts}
        actual: set[str] = set()
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise DataBoundaryError(f"symlinks are forbidden in actor bundles: {path.name}")
            if path.is_file():
                relative = path.relative_to(self.root).as_posix()
                actual.add(relative)
        extras = actual - allowed
        missing = allowed - actual
        if extras or missing:
            raise DataBoundaryError(
                f"actor bundle file set differs from manifest; extras={sorted(extras)}, "
                f"missing={sorted(missing)}"
            )
        for relative_path in self._artifacts:
            self.resolve_artifact(relative_path)


def _bundle_identity(
    *,
    source_sha256: str,
    builder_sha256: str,
    cutoff_date: date,
    target_ids: tuple[str, ...],
) -> tuple[str, str]:
    identity = {
        "builder_sha256": builder_sha256,
        "columns": ACTOR_SERIES_COLUMNS,
        "cutoff_date": cutoff_date.isoformat(),
        "source_sha256": source_sha256,
        "target_ids": target_ids,
    }
    digest = sha256_bytes(canonical_json(identity).encode("utf-8"))
    return f"asof-{cutoff_date.isoformat()}-{digest[:16]}", digest


def build_actor_input_bundle(
    *,
    source_series: Path,
    bundle_root: Path,
    cutoff_date: date,
    target_ids: tuple[str, ...],
) -> ActorInputBundle:
    """Filter selected series through a cutoff and atomically publish the actor bundle."""
    if source_series.is_symlink():
        raise DataBoundaryError("source series may not be a symlink")
    source_series = source_series.resolve()
    if not source_series.is_file():
        raise DataBoundaryError("source series must be a regular, non-symlink file")
    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise DataBoundaryError("target_ids must be non-empty and unique")

    source_sha256 = sha256_file(source_series)
    builder_sha256 = sha256_file(Path(__file__))
    bundle_id, identity_digest = _bundle_identity(
        source_sha256=source_sha256,
        builder_sha256=builder_sha256,
        cutoff_date=cutoff_date,
        target_ids=target_ids,
    )
    bundle_root.mkdir(parents=True, exist_ok=True)
    final_root = bundle_root / bundle_id
    if final_root.exists():
        bundle = ActorInputBundle.load(final_root)
        if bundle.manifest.bundle_digest != identity_digest:
            raise DataBoundaryError("existing content-addressed bundle has an unexpected digest")
        return bundle

    temporary_root = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=bundle_root))
    try:
        output_path = temporary_root / ACTOR_SERIES_FILENAME
        selected_rows: list[dict[str, str]] = []
        selected_keys: set[tuple[str, date]] = set()
        selected_targets = set(target_ids)
        excluded_after_cutoff = 0
        excluded_out_of_scope = 0
        with source_series.open("r", encoding="utf-8", newline="") as source_handle:
            reader = csv.DictReader(source_handle)
            missing_columns = [
                column for column in ACTOR_SERIES_COLUMNS if column not in (reader.fieldnames or [])
            ]
            if missing_columns:
                raise DataBoundaryError(f"source series is missing columns: {missing_columns}")
            for row_number, row in enumerate(reader, start=2):
                series_id = row["series_id"].strip()
                try:
                    observation_date = date.fromisoformat(row["date"])
                except ValueError as error:
                    raise DataBoundaryError(f"invalid date at source row {row_number}") from error
                if observation_date > cutoff_date:
                    excluded_after_cutoff += 1
                    continue
                if series_id not in selected_targets:
                    excluded_out_of_scope += 1
                    continue
                try:
                    value = Decimal(row["value"])
                except InvalidOperation as error:
                    raise DataBoundaryError(f"invalid value at source row {row_number}") from error
                if not value.is_finite() or value <= 0:
                    raise DataBoundaryError(
                        f"non-positive or non-finite value at source row {row_number}"
                    )
                key = (series_id, observation_date)
                if key in selected_keys:
                    raise DataBoundaryError(
                        f"duplicate selected series/date at source row {row_number}"
                    )
                selected_keys.add(key)
                selected_rows.append({column: row[column] for column in ACTOR_SERIES_COLUMNS})

        represented_targets = {row["series_id"] for row in selected_rows}
        absent_targets = sorted(selected_targets - represented_targets)
        if absent_targets:
            raise DataBoundaryError(
                f"selected targets have no history through cutoff: {absent_targets}"
            )
        selected_rows.sort(key=lambda row: (row["series_id"], row["date"]))
        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=ACTOR_SERIES_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(selected_rows)
            output_handle.flush()
            os.fsync(output_handle.fileno())

        selected_dates = [date.fromisoformat(row["date"]) for row in selected_rows]
        artifact = ArtifactRef(
            path=ACTOR_SERIES_FILENAME,
            sha256=sha256_file(output_path),
            media_type="text/csv",
        )
        manifest = ActorInputManifest(
            bundle_id=bundle_id,
            created_at=datetime.now(UTC),
            cutoff_date=cutoff_date,
            target_ids=target_ids,
            source_name=source_series.name,
            source_sha256=source_sha256,
            builder_sha256=builder_sha256,
            row_count=len(selected_rows),
            min_date=min(selected_dates),
            max_date=max(selected_dates),
            excluded_after_cutoff=excluded_after_cutoff,
            excluded_out_of_scope=excluded_out_of_scope,
            artifacts=(artifact,),
            bundle_digest=identity_digest,
        )
        manifest_path = temporary_root / BUNDLE_MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_root, final_root)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    return ActorInputBundle.load(final_root)
