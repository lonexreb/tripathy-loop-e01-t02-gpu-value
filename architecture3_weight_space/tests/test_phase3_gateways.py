from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from tripathy_arch3.budget import BudgetExceededError, RemoteBudgetLedger
from tripathy_arch3.contracts import MemoryKind
from tripathy_arch3.evidence_gateway import (
    EvidenceSearchRequest,
    ExaEvidenceGateway,
    SnapshotUnavailableError,
)
from tripathy_arch3.http_transport import HttpResponse
from tripathy_arch3.memory_gateway import (
    LocalMemoryRegistry,
    MemoryIngestRequest,
    MemoryRecallRequest,
    PastMemoryGateway,
)

NOW = datetime(2026, 11, 15, 12, tzinfo=UTC)
CUTOFF = datetime(2026, 10, 15, 23, 59, 59, tzinfo=UTC)


class ScriptedTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses: Iterator[HttpResponse] = iter(responses)
        self.calls: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
        timeout_seconds: float = 20.0,
    ) -> HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json_body": json_body,
                "timeout_seconds": timeout_seconds,
            }
        )
        return next(self._responses)


def response(status: int, payload: object, request_id: str = "req-test") -> HttpResponse:
    return HttpResponse(
        status_code=status,
        headers={"content-type": "application/json", "x-request-id": request_id},
        body=json.dumps(payload).encode("utf-8"),
    )


def budget(tmp_path: Path, **overrides: int | float) -> RemoteBudgetLedger:
    values: dict[str, int | float] = {
        "maximum_total_usd": 1.0,
        "maximum_modal_gpu_seconds": 1200,
        "maximum_exa_search_calls": 2,
        "maximum_past_write_calls": 3,
        "maximum_past_recall_calls": 3,
        "maximum_openai_teacher_calls": 2,
        "maximum_modal_function_calls": 8,
    }
    values.update(overrides)
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps({"authorized": True, "budget": values}), encoding="utf-8")
    return RemoteBudgetLedger(
        ledger_path=tmp_path / "ledger.json",
        authorization_path=authorization,
    )


def exa_request() -> EvidenceSearchRequest:
    return EvidenceSearchRequest(
        call_id="phase3-exa-test",
        episode_id="episode-g0",
        hypothesis_id="ppi-capacity-cost",
        information_need="Find the dated BLS producer-price release available by the origin.",
        query="BLS producer price index data center equipment October 2026",
        forecast_cutoff=CUTOFF,
        requested_at=NOW,
        include_domains=("bls.gov",),
        license_by_domain={"bls.gov": "US-PD"},
        license_read_at=NOW,
        decision_affected="whether a public inflation feature is eligible",
        max_results=1,
    )


def test_exa_gateway_requests_snapshot_and_admits_only_dated_content(tmp_path: Path) -> None:
    transport = ScriptedTransport(
        [
            response(
                200,
                {
                    "requestId": "exa-123",
                    "results": [
                        {
                            "id": "https://www.bls.gov/news.release/ppi.nr0.htm",
                            "url": "https://www.bls.gov/news.release/ppi.nr0.htm",
                            "title": "Producer Price Index News Release",
                            "publishedDate": "2026-10-14T12:30:00Z",
                            "text": "The Producer Price Index release was published on October 14.",
                        }
                    ],
                    "costDollars": {"total": 0.006},
                },
            )
        ]
    )
    gateway = ExaEvidenceGateway(
        api_key="secret-exa-key",
        transport=transport,
        budget=budget(tmp_path),
        capture_root=tmp_path / "artifacts" / "phase3" / "captures",
    )
    receipt = gateway.search(exa_request())

    body = transport.calls[0]["json_body"]
    assert isinstance(body, dict)
    assert body["endPublishedDate"] == "2026-10-15T23:59:59Z"
    assert body["contents"] == {
        "snapshotAsOf": "2026-10-15T23:59:59Z",
        "text": True,
    }
    assert receipt.records[0].admissible is True
    assert receipt.records[0].license_id == "US-PD"
    captures = list((tmp_path / "artifacts").rglob("*"))
    assert not any(path.is_file() and b"secret-exa-key" in path.read_bytes() for path in captures)


def test_exa_snapshot_entitlement_failure_is_captured_and_never_falls_back(
    tmp_path: Path,
) -> None:
    transport = ScriptedTransport(
        [response(403, {"requestId": "exa-403", "tag": "SNAPSHOT_NOT_ON_PLAN"})]
    )
    gateway = ExaEvidenceGateway(
        api_key="secret-exa-key",
        transport=transport,
        budget=budget(tmp_path),
        capture_root=tmp_path / "artifacts" / "phase3" / "captures",
    )
    with pytest.raises(SnapshotUnavailableError, match="SNAPSHOT_NOT_ON_PLAN"):
        gateway.search(exa_request())
    assert len(transport.calls) == 1
    assert list((tmp_path / "artifacts").rglob("response-body.bin"))


def test_evidence_request_rejects_prohibited_domain() -> None:
    with pytest.raises(ValidationError, match="prohibited evidence domain"):
        EvidenceSearchRequest(
            **{
                **exa_request().model_dump(),
                "include_domains": ("reddit.com",),
                "license_by_domain": {"reddit.com": "unknown"},
            }
        )


def test_past_memory_is_scoped_polled_and_revalidated_against_local_registry(
    tmp_path: Path,
) -> None:
    memory_id = "arch3-phase2-boundary-lesson"
    identity = "arch3-recursive-sft-g0-reader"
    audience = "arch3-recursive-sft-g0"
    transport = ScriptedTransport(
        [
            response(
                201,
                {
                    "slug": audience,
                    "kind": "fixed",
                    "identities": [identity],
                },
            ),
            response(
                202,
                {
                    "ingestionId": "7d9f2b6a-4c1e-4f8a-9b3d-2e5c8a1f6d40",
                    "sourceId": memory_id,
                    "status": "queued",
                },
            ),
            response(200, {"status": "completed", "settled": True}),
            response(
                200,
                {
                    "asOf": "2026-11-15T12:00:00Z",
                    "usedEvidenceTokens": 21,
                    "results": [
                        {
                            "id": "memory-result-1",
                            "content": "The actor data boundary reproduced the frozen score.",
                            "occurredAt": "2026-11-15T11:00:00Z",
                            "sources": [
                                {
                                    "sourceId": memory_id,
                                    "occurredAt": "2026-11-15T11:00:00Z",
                                }
                            ],
                        }
                    ],
                },
            ),
        ]
    )
    registry = LocalMemoryRegistry(tmp_path / "memory_registry.json")
    gateway = PastMemoryGateway(
        api_key="secret-past-key",
        transport=transport,
        budget=budget(tmp_path),
        registry=registry,
        capture_root=tmp_path / "artifacts" / "phase3" / "captures",
    )
    source_time = datetime(2026, 11, 15, 11, tzinfo=UTC)
    ingest = MemoryIngestRequest(
        call_id="phase3-past-ingest-test",
        memory_id=memory_id,
        kind=MemoryKind.EPISODE_LESSON,
        arm="recursive-sft",
        generation=0,
        identity=identity,
        audience_slug=audience,
        content="The actor data boundary reproduced the frozen score.",
        source_ids=("phase2-acceptance",),
        source_available_at=(source_time,),
        effective_after=source_time,
        idempotency_key="phase3-past-ingest-test-v1",
    )
    receipt = gateway.ingest(ingest, poll_interval_seconds=0)
    assert receipt.status == "completed"
    recall = gateway.recall(
        MemoryRecallRequest(
            call_id="phase3-past-recall-test",
            arm="recursive-sft",
            generation=0,
            identity=identity,
            audience_slug=audience,
            query="What did the actor data-boundary acceptance establish?",
            forecast_cutoff=NOW,
        )
    )
    assert recall.results[0].eligible is True
    recall_body = transport.calls[-1]["json_body"]
    assert isinstance(recall_body, dict)
    assert recall_body["queryTimestamp"] == "2026-11-15T12:00:00Z"
    assert recall_body["occurredTo"] == "2026-11-15T12:00:00Z"


def test_budget_fails_before_remote_call_ceiling(tmp_path: Path) -> None:
    ledger = budget(tmp_path, maximum_exa_search_calls=0)
    with pytest.raises(BudgetExceededError, match="ceiling"):
        ledger.reserve("exa_search", call_id="blocked", estimated_usd=Decimal("0.01"))


def test_gpu_seconds_are_cumulative_and_idempotent(tmp_path: Path) -> None:
    ledger = budget(tmp_path, maximum_modal_gpu_seconds=10)
    ledger.reserve_gpu_seconds(call_id="gpu-a", seconds=6)
    ledger.reserve_gpu_seconds(call_id="gpu-a", seconds=6)
    with pytest.raises(BudgetExceededError, match="GPU-seconds"):
        ledger.reserve_gpu_seconds(call_id="gpu-b", seconds=5)
