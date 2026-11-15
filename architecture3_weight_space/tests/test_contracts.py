from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError

from tripathy_arch3.contracts import (
    ActionType,
    ActorAction,
    EvidenceRecord,
    ForecastRow,
    MemoryKind,
    MemoryRecord,
    SFTExample,
    SplitRole,
    VerificationCheck,
)

SHA = "a" * 64
NOW = datetime(2026, 11, 15, 12, tzinfo=UTC)


def action() -> ActorAction:
    return ActorAction(
        action_id="action-1",
        episode_id="episode-1",
        step_id=1,
        tool=ActionType.SEARCH_EVIDENCE,
        arguments={"question": "Was capacity operational before the origin?"},
        proposed_at=NOW,
        context_sha256=SHA,
    )


def test_forecast_row_requires_positive_ordered_quantiles_and_exact_target_date() -> None:
    row = ForecastRow(
        series_id="series-1",
        origin_date=date(2026, 11, 6),
        target_date=date(2026, 11, 9),
        horizon_days=3,
        q10=Decimal("1.0"),
        q50=Decimal("1.2"),
        q90=Decimal("1.4"),
        method_id="frozen-method",
    )
    assert row.q10 <= row.q50 <= row.q90

    with pytest.raises(ValidationError, match="q10 <= q50 <= q90"):
        ForecastRow(
            series_id="series-1",
            origin_date=date(2026, 11, 6),
            target_date=date(2026, 11, 9),
            horizon_days=3,
            q10=Decimal("1.3"),
            q50=Decimal("1.2"),
            q90=Decimal("1.4"),
            method_id="bad-method",
        )

    with pytest.raises(ValidationError, match="target_date"):
        ForecastRow(
            series_id="series-1",
            origin_date=date(2026, 11, 6),
            target_date=date(2026, 11, 8),
            horizon_days=3,
            q10=Decimal("1.0"),
            q50=Decimal("1.2"),
            q90=Decimal("1.4"),
            method_id="bad-method",
        )


def test_admissible_evidence_cannot_be_available_after_cutoff() -> None:
    with pytest.raises(ValidationError, match="after the forecast cutoff"):
        EvidenceRecord(
            evidence_id="evidence-1",
            source_id="source-1",
            episode_id="episode-1",
            query="capacity",
            source_url="https://example.com/source",
            title="Source",
            passage="Historical source passage.",
            license_id="CC-BY-4.0",
            license_read_at=NOW,
            provider_request_id="request-1",
            snapshot_requested_at=datetime(2026, 11, 6, 23, 59, tzinfo=UTC),
            snapshot_confirmed=True,
            published_at=datetime(2026, 11, 7, tzinfo=UTC),
            event_time=datetime(2026, 11, 7, tzinfo=UTC),
            available_at=datetime(2026, 11, 7, tzinfo=UTC),
            retrieved_at=NOW,
            forecast_cutoff=datetime(2026, 11, 6, 23, 59, tzinfo=UTC),
            content_sha256=SHA,
            response_sha256=SHA,
            decision_affected="candidate feature eligibility",
            admissible=True,
        )


def test_memory_cannot_be_backdated_before_a_source_is_available() -> None:
    with pytest.raises(ValidationError, match="before all sources"):
        MemoryRecord(
            memory_id="memory-1",
            kind=MemoryKind.EPISODE_LESSON,
            arm="recursive_sft",
            generation=1,
            audience=("recursive_sft:g1",),
            source_ids=("source-1", "source-2"),
            source_available_at=(
                datetime(2026, 10, 1, tzinfo=UTC),
                datetime(2026, 10, 5, tzinfo=UTC),
            ),
            effective_after=datetime(2026, 10, 4, tzinfo=UTC),
            ingested_at=NOW,
            content_sha256=SHA,
        )


def test_sft_loss_mask_weights_only_assistant_target_tokens() -> None:
    example = SFTExample(
        example_id="example-1",
        source_episode_id="episode-1",
        split_role=SplitRole.TRAIN,
        origin_cutoff=datetime(2026, 10, 1, tzinfo=UTC),
        original_context_sha256=SHA,
        target_action=action(),
        token_ids=(101, 102, 103, 104),
        loss_mask=(0, 0, 1, 1),
        assistant_target_start=2,
        verification={VerificationCheck.SCHEMA: True, VerificationCheck.TEMPORAL: True},
        accepted=True,
        verified_at=NOW,
    )
    assert sum(example.loss_mask) == 2

    with pytest.raises(ValidationError, match="assistant target tokens"):
        SFTExample(
            **{
                **example.model_dump(),
                "loss_mask": (1, 0, 1, 1),
            }
        )


def test_contracts_forbid_unknown_fields_and_naive_timestamps() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        ActorAction(
            action_id="action-1",
            episode_id="episode-1",
            step_id=1,
            tool=ActionType.SEARCH_EVIDENCE,
            arguments={},
            proposed_at=datetime(2026, 11, 15),
            context_sha256=SHA,
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        ActorAction(
            **action().model_dump(),
            unplanned_field=str(UUID(int=0)),
        )
