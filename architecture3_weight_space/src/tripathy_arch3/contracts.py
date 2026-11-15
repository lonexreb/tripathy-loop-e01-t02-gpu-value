"""Immutable, provider-independent contracts for Architecture 3 artifacts."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    field_validator,
    model_validator,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmpty = Annotated[str, Field(min_length=1)]
PositiveDecimal = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC)


def _optional_as_utc(value: datetime | None) -> datetime | None:
    return None if value is None else _as_utc(value)


def _require_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact paths must be relative and may not contain '..'")
    return value


class FrozenContract(BaseModel):
    """Strict base class used by records that become experiment artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class SplitRole(StrEnum):
    TRAIN = "train"
    SELECT = "select"
    AUDIT = "audit"
    FINAL = "final"


class ActionType(StrEnum):
    INSPECT_SERIES = "inspect_series"
    PROPOSE_HYPOTHESIS = "propose_hypothesis"
    SEARCH_EVIDENCE = "search_evidence"
    INSPECT_EVIDENCE = "inspect_evidence"
    RUN_EXPERIMENT = "run_experiment"
    PROPOSE_FORECASTER_PATCH = "propose_forecaster_patch"
    COMMIT_FORECAST = "commit_forecast"
    DIAGNOSE_RESEARCH_FAILURE = "diagnose_research_failure"
    PROPOSE_NEXT_TRAINING_PLAN = "propose_next_training_plan"


class CompletionStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class MemoryKind(StrEnum):
    FACTUAL_EVIDENCE = "factual_evidence"
    EPISODE_LESSON = "episode_lesson"
    CURRICULUM_PROPOSAL = "curriculum_proposal"


class ClaimStatus(StrEnum):
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    UNCERTAIN = "uncertain"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationCheck(StrEnum):
    SYNTACTIC = "syntactic"
    TEMPORAL = "temporal"
    SOURCE = "source"
    SCHEMA = "schema"
    EXECUTION = "execution"
    PARTITION = "partition"


class AdmissionStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ArtifactRef(FrozenContract):
    path: NonEmpty
    sha256: Sha256
    media_type: NonEmpty

    _relative_path = field_validator("path")(_require_relative_path)


class ResourceBudget(FrozenContract):
    max_actions: Annotated[int, Field(ge=1)]
    max_exa_queries: Annotated[int, Field(ge=0)] = 0
    max_teacher_calls: Annotated[int, Field(ge=0)] = 0
    max_wall_seconds: Annotated[int, Field(ge=1)]


class ResearchCase(FrozenContract):
    case_id: NonEmpty
    run_id: NonEmpty
    episode_id: NonEmpty
    origin_time: datetime
    information_cutoff: datetime
    split_role: SplitRole
    target_ids: tuple[NonEmpty, ...]
    horizons: tuple[Annotated[int, Field(gt=0)], ...]
    actor_checkpoint: NonEmpty
    memory_snapshot: Sha256 | None = None
    resource_budget: ResourceBudget

    _utc_timestamps = field_validator("origin_time", "information_cutoff")(_as_utc)

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.information_cutoff > self.origin_time:
            raise ValueError("information_cutoff may not be after origin_time")
        if not self.target_ids or len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("target_ids must be non-empty and unique")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be non-empty, unique, and sorted")
        return self


class ActorAction(FrozenContract):
    action_id: NonEmpty
    episode_id: NonEmpty
    step_id: Annotated[int, Field(ge=0)]
    tool: ActionType
    arguments: dict[str, JsonValue]
    proposed_at: datetime
    context_sha256: Sha256

    _utc_timestamp = field_validator("proposed_at")(_as_utc)


class ToolObservation(FrozenContract):
    action_id: NonEmpty
    status: CompletionStatus
    observed_at: datetime
    artifacts: tuple[ArtifactRef, ...] = ()
    payload_sha256: Sha256 | None = None
    error: str | None = None

    _utc_timestamp = field_validator("observed_at")(_as_utc)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.status is CompletionStatus.FAILED and not self.error:
            raise ValueError("failed observations require an error")
        if self.status is CompletionStatus.COMPLETED and self.error:
            raise ValueError("completed observations may not contain an error")
        return self


class EvidenceRecord(FrozenContract):
    evidence_id: NonEmpty
    source_id: NonEmpty
    episode_id: NonEmpty
    query: NonEmpty
    source_url: HttpUrl
    title: NonEmpty
    passage: NonEmpty
    license_id: NonEmpty
    license_read_at: datetime
    provider_request_id: NonEmpty
    snapshot_requested_at: datetime
    snapshot_confirmed: bool
    published_at: datetime | None = None
    event_time: datetime | None = None
    available_at: datetime | None = None
    retrieved_at: datetime
    forecast_cutoff: datetime
    content_sha256: Sha256
    response_sha256: Sha256
    decision_affected: NonEmpty
    admissible: bool
    rejection_reason: str | None = None

    _optional_utc_timestamps = field_validator(
        "published_at",
        "event_time",
        "available_at",
        "license_read_at",
        "snapshot_requested_at",
    )(_optional_as_utc)
    _utc_timestamps = field_validator(
        "retrieved_at",
        "forecast_cutoff",
    )(_as_utc)

    @model_validator(mode="after")
    def validate_admissibility(self) -> Self:
        if self.admissible:
            if not self.snapshot_confirmed:
                raise ValueError("admissible evidence requires a confirmed historical snapshot")
            if self.snapshot_requested_at > self.forecast_cutoff:
                raise ValueError("snapshot cutoff may not exceed the forecast cutoff")
            if self.published_at is None or self.event_time is None:
                raise ValueError("admissible evidence requires publication and event times")
            if self.available_at is None:
                raise ValueError("admissible evidence requires available_at")
            if max(self.published_at, self.event_time, self.available_at) > self.forecast_cutoff:
                raise ValueError("evidence available after the forecast cutoff is inadmissible")
            if self.rejection_reason:
                raise ValueError("admissible evidence may not have a rejection_reason")
        elif not self.rejection_reason:
            raise ValueError("rejected evidence requires a rejection_reason")
        return self


class MemoryRecord(FrozenContract):
    memory_id: NonEmpty
    kind: MemoryKind
    arm: NonEmpty
    generation: Annotated[int, Field(ge=0)]
    audience: tuple[NonEmpty, ...]
    source_ids: tuple[NonEmpty, ...]
    source_available_at: tuple[datetime, ...]
    effective_after: datetime
    ingested_at: datetime
    content_sha256: Sha256

    _utc_timestamps = field_validator("effective_after", "ingested_at")(_as_utc)

    @field_validator("source_available_at")
    @classmethod
    def normalize_source_timestamps(cls, values: tuple[datetime, ...]) -> tuple[datetime, ...]:
        return tuple(_as_utc(value) for value in values)

    @model_validator(mode="after")
    def validate_memory_lineage(self) -> Self:
        if not self.audience or len(set(self.audience)) != len(self.audience):
            raise ValueError("audience must be non-empty and unique")
        if not self.source_ids or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be non-empty and unique")
        if len(self.source_ids) != len(self.source_available_at):
            raise ValueError("each source_id needs a source_available_at timestamp")
        if self.effective_after < max(self.source_available_at):
            raise ValueError("memory may not become effective before all sources are available")
        if self.ingested_at < self.effective_after:
            raise ValueError("ingested_at may not precede effective_after")
        return self


class ForecastRow(FrozenContract):
    series_id: NonEmpty
    origin_date: date
    target_date: date
    horizon_days: Annotated[int, Field(gt=0)]
    q10: PositiveDecimal
    q50: PositiveDecimal
    q90: PositiveDecimal
    method_id: NonEmpty
    fallback: bool = False

    @model_validator(mode="after")
    def validate_quantiles(self) -> Self:
        if not self.q10 <= self.q50 <= self.q90:
            raise ValueError("forecast quantiles must satisfy q10 <= q50 <= q90")
        if self.target_date != self.origin_date + timedelta(days=self.horizon_days):
            raise ValueError("target_date must equal origin_date + horizon_days")
        return self


class ForecastCommit(FrozenContract):
    commit_id: NonEmpty
    case_id: NonEmpty
    created_at: datetime
    rows: tuple[ForecastRow, ...]
    code_sha256: Sha256
    inputs_sha256: Sha256

    _utc_timestamp = field_validator("created_at")(_as_utc)

    @model_validator(mode="after")
    def validate_rows(self) -> Self:
        if not self.rows:
            raise ValueError("a forecast commit requires at least one row")
        keys = {(row.series_id, row.origin_date, row.horizon_days) for row in self.rows}
        if len(keys) != len(self.rows):
            raise ValueError("forecast rows must have unique series/origin/horizon keys")
        if len({row.origin_date for row in self.rows}) != 1:
            raise ValueError("one forecast commit may contain only one origin_date")
        return self


class TeacherCritique(FrozenContract):
    review_id: NonEmpty
    episode_id: NonEmpty
    original_action_id: NonEmpty
    failure_type: NonEmpty
    evidence_ids: tuple[NonEmpty, ...]
    claim_status: ClaimStatus
    proposed_target_action: ActorAction
    verification_required: tuple[VerificationCheck, ...]
    confidence: Confidence
    created_at: datetime

    _utc_timestamp = field_validator("created_at")(_as_utc)

    @model_validator(mode="after")
    def validate_review(self) -> Self:
        if not self.verification_required:
            raise ValueError("teacher corrections require explicit verification checks")
        if len(set(self.verification_required)) != len(self.verification_required):
            raise ValueError("verification_required may not contain duplicates")
        return self


class SFTExample(FrozenContract):
    example_id: NonEmpty
    source_episode_id: NonEmpty
    split_role: SplitRole
    origin_cutoff: datetime
    original_context_sha256: Sha256
    target_action: ActorAction
    token_ids: tuple[int, ...]
    loss_mask: tuple[Annotated[int, Field(ge=0, le=1)], ...]
    assistant_target_start: Annotated[int, Field(ge=0)]
    verification: dict[VerificationCheck, bool]
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()
    verified_at: datetime

    _utc_timestamps = field_validator("origin_cutoff", "verified_at")(_as_utc)

    @model_validator(mode="after")
    def validate_training_mask(self) -> Self:
        if not self.token_ids or len(self.token_ids) != len(self.loss_mask):
            raise ValueError("token_ids and loss_mask must be non-empty and aligned")
        if self.assistant_target_start >= len(self.token_ids):
            raise ValueError("assistant_target_start must point inside token_ids")
        expected = (0,) * self.assistant_target_start + (1,) * (
            len(self.token_ids) - self.assistant_target_start
        )
        if self.loss_mask != expected:
            raise ValueError("only assistant target tokens may receive SFT loss weight")
        if self.accepted and (not self.verification or not all(self.verification.values())):
            raise ValueError("accepted examples require every recorded verification to pass")
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted examples may not have rejection reasons")
        if not self.accepted and not self.rejection_reasons:
            raise ValueError("rejected examples require a rejection reason")
        return self


class CheckpointRecord(FrozenContract):
    checkpoint_id: NonEmpty
    parent_checkpoint: str | None
    base_model_id: NonEmpty
    base_model_revision: NonEmpty
    base_weights_sha256: Sha256
    adapter_sha256: Sha256
    dataset_sha256: Sha256
    training_config_sha256: Sha256
    created_at: datetime
    effective_after: datetime

    _utc_timestamps = field_validator("created_at", "effective_after")(_as_utc)

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        if self.parent_checkpoint == self.checkpoint_id:
            raise ValueError("a checkpoint may not be its own parent")
        if self.effective_after > self.created_at:
            raise ValueError("effective_after may not be later than checkpoint creation")
        return self


class AdmissionDecision(FrozenContract):
    decision_id: NonEmpty
    candidate_checkpoint: NonEmpty
    parent_checkpoint: NonEmpty
    status: AdmissionStatus
    reason: NonEmpty
    protocol_sha256: Sha256
    metrics_artifact: ArtifactRef
    decided_at: datetime

    _utc_timestamp = field_validator("decided_at")(_as_utc)

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if self.candidate_checkpoint == self.parent_checkpoint:
            raise ValueError("candidate and parent checkpoints must differ")
        return self


class EventEnvelope(FrozenContract):
    schema_version: Annotated[int, Field(ge=1)] = 1
    event_id: UUID
    event_type: NonEmpty
    run_id: NonEmpty
    arm: NonEmpty
    seed: int
    generation: Annotated[int, Field(ge=0)]
    episode_id: str | None = None
    step_id: int | None = None
    trace_id: UUID
    parent_event_id: UUID | None = None
    created_at: datetime
    origin_date: date | None = None
    information_cutoff: datetime | None = None
    split_role: SplitRole | None = None
    actor_checkpoint: str | None = None
    parent_checkpoint: str | None = None
    curriculum_id: str | None = None
    dataset_sha256: Sha256 | None = None
    git_sha: Annotated[str, Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")]
    protocol_sha256: Sha256
    input_artifact_sha256: Sha256 | None = None
    output_artifact_sha256: Sha256 | None = None
    status: CompletionStatus
    duration_ms: Annotated[int, Field(ge=0)] | None = None
    usage: dict[str, JsonValue] = Field(default_factory=dict)
    payload_ref: ArtifactRef | None = None

    _created_at_utc = field_validator("created_at")(_as_utc)
    _optional_cutoff_utc = field_validator("information_cutoff")(_optional_as_utc)
