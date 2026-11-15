"""Scoped past.dev memory with local source registry and temporal validation."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Self
from urllib.parse import quote

from pydantic import Field, field_validator, model_validator

from .audit import canonical_json, sha256_bytes
from .budget import RemoteBudgetLedger
from .capture import capture_json, write_atomic
from .config import WORKSPACE_ROOT
from .contracts import ArtifactRef, FrozenContract, MemoryKind, MemoryRecord, NonEmpty, Sha256
from .http_transport import HttpTransport

PAST_API_ROOT = "https://api.past.dev/api/v1"
DEFAULT_CAPTURE_ROOT = WORKSPACE_ROOT / "artifacts" / "phase3" / "captures"
DEFAULT_REGISTRY_PATH = WORKSPACE_ROOT / "artifacts" / "phase3" / "memory_registry.json"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC)


def _provider_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return _utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


class MemoryGatewayError(RuntimeError):
    """past.dev provider or eligibility error with a frozen response capture."""


class MemoryIngestRequest(FrozenContract):
    call_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")]
    memory_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,199}$")]
    kind: MemoryKind
    arm: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,39}$")]
    generation: Annotated[int, Field(ge=0)]
    identity: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,199}$")]
    audience_slug: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,39}$")]
    content: NonEmpty
    source_ids: tuple[NonEmpty, ...]
    source_available_at: tuple[datetime, ...]
    effective_after: datetime
    idempotency_key: Annotated[str, Field(min_length=3, max_length=200)]

    _effective_after = field_validator("effective_after")(_utc)

    @field_validator("source_available_at")
    @classmethod
    def normalize_source_times(cls, values: tuple[datetime, ...]) -> tuple[datetime, ...]:
        return tuple(_utc(value) for value in values)

    @model_validator(mode="after")
    def validate_scope_and_time(self) -> Self:
        expected_audience = f"arch3-{self.arm}-g{self.generation}"
        expected_identity = expected_audience + "-reader"
        if self.audience_slug != expected_audience or self.identity != expected_identity:
            raise ValueError("memory audience and identity must match arm and generation")
        if not self.source_ids or len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("source_ids must be non-empty and unique")
        if len(self.source_ids) != len(self.source_available_at):
            raise ValueError("every source needs an availability timestamp")
        if self.effective_after < max(self.source_available_at):
            raise ValueError("memory may not become effective before all sources are available")
        return self


class MemoryIngestReceipt(FrozenContract):
    call_id: NonEmpty
    ingestion_id: NonEmpty
    source_id: NonEmpty
    status: NonEmpty
    record: MemoryRecord
    request_artifacts: tuple[ArtifactRef, ...]
    response_artifacts: tuple[ArtifactRef, ...]


class MemoryRecallRequest(FrozenContract):
    call_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")]
    arm: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,39}$")]
    generation: Annotated[int, Field(ge=0)]
    identity: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,199}$")]
    audience_slug: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{1,39}$")]
    query: NonEmpty
    forecast_cutoff: datetime
    max_tokens: Annotated[int, Field(ge=100, le=8000)] = 2000
    limit: Annotated[int, Field(ge=1, le=50)] = 20

    _forecast_cutoff = field_validator("forecast_cutoff")(_utc)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        expected_audience = f"arch3-{self.arm}-g{self.generation}"
        if self.audience_slug != expected_audience:
            raise ValueError("recall audience must match arm and generation")
        if self.identity != expected_audience + "-reader":
            raise ValueError("recall identity must match arm and generation")
        return self


class RecalledMemory(FrozenContract):
    result_id: NonEmpty
    content: NonEmpty
    occurred_at: datetime
    source_ids: tuple[NonEmpty, ...]
    eligible: bool
    rejection_reasons: tuple[NonEmpty, ...] = ()

    _occurred_at = field_validator("occurred_at")(_utc)

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.eligible and self.rejection_reasons:
            raise ValueError("eligible recall may not have rejection reasons")
        if not self.eligible and not self.rejection_reasons:
            raise ValueError("rejected recall requires a reason")
        return self


class MemoryRecallReceipt(FrozenContract):
    call_id: NonEmpty
    as_of: datetime
    used_evidence_tokens: Annotated[int, Field(ge=0)]
    request_artifact: ArtifactRef
    response_artifact: ArtifactRef
    response_sha256: Sha256
    results: tuple[RecalledMemory, ...]

    _as_of = field_validator("as_of")(_utc)


class LocalMemoryRegistry:
    """Authoritative map from past.dev source IDs to experiment scope and lineage."""

    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH) -> None:
        self.path = path

    def read(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records = payload.get("records", {}) if isinstance(payload, dict) else {}
        if not isinstance(records, dict):
            raise MemoryGatewayError("local memory registry is malformed")
        return {str(key): value for key, value in records.items() if isinstance(value, dict)}

    def register(self, request: MemoryIngestRequest, record: MemoryRecord) -> None:
        records = self.read()
        entry = {
            "arm": request.arm,
            "audience_slug": request.audience_slug,
            "generation": request.generation,
            "kind": request.kind,
            "memory_record": record.model_dump(mode="json"),
            "source_available_at": [value.isoformat() for value in request.source_available_at],
            "source_ids": list(request.source_ids),
        }
        existing = records.get(request.memory_id)
        if existing is not None and existing != entry:
            raise MemoryGatewayError("a memory source ID cannot be rebound to new lineage")
        records[request.memory_id] = entry
        payload = {
            "schema_version": 1,
            "records": records,
            "registry_sha256": sha256_bytes(canonical_json(records).encode("utf-8")),
        }
        write_atomic(self.path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


class PastMemoryGateway:
    def __init__(
        self,
        *,
        api_key: str,
        transport: HttpTransport,
        budget: RemoteBudgetLedger,
        registry: LocalMemoryRegistry,
        capture_root: Path = DEFAULT_CAPTURE_ROOT,
    ) -> None:
        if not api_key:
            raise ValueError("past.dev API key is required")
        self._api_key = api_key
        self._transport = transport
        self._budget = budget
        self._registry = registry
        self._capture_root = capture_root

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": "Bearer " + self._api_key,
            "Content-Type": "application/json",
        }

    def _call(
        self,
        *,
        method: str,
        path: str,
        call_id: str,
        body: object | None = None,
        timeout_seconds: float = 20.0,
    ) -> tuple[object, ArtifactRef, ArtifactRef]:
        response = self._transport.request(
            method,
            PAST_API_ROOT + path,
            headers=self._headers,
            json_body=body,
            timeout_seconds=timeout_seconds,
        )
        request_artifact, response_artifact = capture_json(
            self._capture_root,
            provider="past",
            call_id=call_id,
            request={"body": body, "method": method, "path": path},
            response_status=response.status_code,
            response_headers=response.headers,
            response_body=response.body,
        )
        try:
            payload = response.json() if response.body else {}
        except ValueError as error:
            raise MemoryGatewayError(
                f"past.dev returned non-JSON status {response.status_code}"
            ) from error
        if not 200 <= response.status_code < 300:
            code = payload.get("code", "unknown") if isinstance(payload, dict) else "unknown"
            raise MemoryGatewayError(
                f"past.dev request failed with status {response.status_code} ({code})"
            )
        return payload, request_artifact, response_artifact

    def ingest(
        self,
        request: MemoryIngestRequest,
        *,
        poll_timeout_seconds: float = 180.0,
        poll_interval_seconds: float = 2.0,
    ) -> MemoryIngestReceipt:
        audience_call_id = request.call_id + "-audience"
        self._budget.reserve("past_write", call_id=audience_call_id, estimated_usd=Decimal("0"))
        audience_payload, audience_request, audience_response = self._call(
            method="PUT",
            path="/audiences/" + quote(request.audience_slug, safe=""),
            call_id=audience_call_id,
            body={
                "kind": "fixed",
                "name": f"Architecture 3 {request.arm} generation {request.generation}",
                "identities": [request.identity],
            },
        )
        if not isinstance(audience_payload, dict):
            raise MemoryGatewayError("past.dev audience response must be an object")
        if request.identity not in audience_payload.get("identities", []):
            raise MemoryGatewayError("past.dev audience does not contain the scoped reader")

        self._budget.reserve("past_write", call_id=request.call_id, estimated_usd=Decimal("0.001"))
        ingest_body = {
            "id": request.memory_id,
            "content": request.content,
            "label": f"Architecture 3 {request.kind}",
            "timestamp": request.effective_after.isoformat().replace("+00:00", "Z"),
            "audience": request.audience_slug,
            "identity": request.identity,
            "idempotencyKey": request.idempotency_key,
            "metadata": {
                "arm": request.arm,
                "generation": request.generation,
                "kind": request.kind,
                "source_ids": list(request.source_ids),
                "source_available_at": [
                    value.isoformat().replace("+00:00", "Z")
                    for value in request.source_available_at
                ],
            },
        }
        payload, ingest_request, ingest_response = self._call(
            method="POST", path="/ingest", call_id=request.call_id, body=ingest_body
        )
        if not isinstance(payload, dict):
            raise MemoryGatewayError("past.dev ingest response must be an object")
        ingestion_id = str(payload.get("ingestionId") or "")
        source_id = str(payload.get("sourceId") or "")
        status = str(payload.get("status") or "")
        if not ingestion_id or not source_id:
            raise MemoryGatewayError("past.dev ingest response is missing IDs")

        response_artifacts = [audience_response, ingest_response]
        request_artifacts = [audience_request, ingest_request]
        deadline = time.monotonic() + poll_timeout_seconds
        poll_index = 0
        while status not in {"completed", "failed"} and time.monotonic() < deadline:
            if poll_index:
                time.sleep(poll_interval_seconds)
            status_payload, poll_request, poll_response = self._call(
                method="GET",
                path="/ingest/" + quote(ingestion_id, safe=""),
                call_id=f"{request.call_id}-poll-{poll_index:03d}",
            )
            request_artifacts.append(poll_request)
            response_artifacts.append(poll_response)
            if not isinstance(status_payload, dict):
                raise MemoryGatewayError("past.dev status response must be an object")
            status = str(status_payload.get("status") or "")
            if status not in {"queued", "processing", "completed", "failed"}:
                raise MemoryGatewayError("past.dev returned an unknown ingestion status")
            poll_index += 1
        if status != "completed":
            raise MemoryGatewayError(
                f"past.dev ingestion did not complete: {status or 'timeout'} ({ingestion_id})"
            )

        ingested_at = datetime.now(UTC)
        record = MemoryRecord(
            memory_id=request.memory_id,
            kind=request.kind,
            arm=request.arm,
            generation=request.generation,
            audience=(request.audience_slug,),
            source_ids=request.source_ids,
            source_available_at=request.source_available_at,
            effective_after=request.effective_after,
            ingested_at=ingested_at,
            content_sha256=sha256_bytes(request.content.encode("utf-8")),
        )
        self._registry.register(request, record)
        return MemoryIngestReceipt(
            call_id=request.call_id,
            ingestion_id=ingestion_id,
            source_id=source_id,
            status=status,
            record=record,
            request_artifacts=tuple(request_artifacts),
            response_artifacts=tuple(response_artifacts),
        )

    def recall(self, request: MemoryRecallRequest) -> MemoryRecallReceipt:
        self._budget.reserve("past_recall", call_id=request.call_id, estimated_usd=Decimal("0.001"))
        cutoff = request.forecast_cutoff.isoformat().replace("+00:00", "Z")
        payload, request_artifact, response_artifact = self._call(
            method="POST",
            path="/recall",
            call_id=request.call_id,
            body={
                "query": request.query,
                "identity": request.identity,
                "queryTimestamp": cutoff,
                "occurredTo": cutoff,
                "limit": request.limit,
                "maxTokens": request.max_tokens,
            },
            timeout_seconds=45.0,
        )
        if not isinstance(payload, dict):
            raise MemoryGatewayError("past.dev recall response must be an object")
        as_of = _provider_time(payload.get("asOf"))
        if as_of is None or as_of > request.forecast_cutoff:
            raise MemoryGatewayError("past.dev recall exceeded the requested time perspective")
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise MemoryGatewayError("past.dev recall results must be an array")
        registry = self._registry.read()
        results = tuple(
            self._validate_result(request, result, registry)
            for result in raw_results
            if isinstance(result, dict)
        )
        return MemoryRecallReceipt(
            call_id=request.call_id,
            as_of=as_of,
            used_evidence_tokens=int(payload.get("usedEvidenceTokens", 0)),
            request_artifact=request_artifact,
            response_artifact=response_artifact,
            response_sha256=response_artifact.sha256,
            results=results,
        )

    @staticmethod
    def _validate_result(
        request: MemoryRecallRequest,
        result: dict[str, object],
        registry: dict[str, dict[str, object]],
    ) -> RecalledMemory:
        sources = result.get("sources", [])
        source_ids = (
            tuple(
                str(source.get("sourceId"))
                for source in sources
                if isinstance(source, dict) and source.get("sourceId")
            )
            if isinstance(sources, list)
            else ()
        )
        reasons: list[str] = []
        occurred_at = _provider_time(result.get("occurredAt"))
        if occurred_at is None:
            occurred_at = request.forecast_cutoff
            reasons.append("result occurrence time missing")
        elif occurred_at > request.forecast_cutoff:
            reasons.append("result occurred after the forecast cutoff")
        if not source_ids:
            reasons.append("result has no source lineage")
        for source_id in source_ids:
            entry = registry.get(source_id)
            if entry is None:
                reasons.append(f"source {source_id} is absent from the local registry")
                continue
            if entry.get("arm") != request.arm or entry.get("generation") != request.generation:
                reasons.append(f"source {source_id} belongs to another arm or generation")
            if entry.get("audience_slug") != request.audience_slug:
                reasons.append(f"source {source_id} belongs to another audience")
            raw_times = entry.get("source_available_at", [])
            if isinstance(raw_times, list):
                available_times = [_provider_time(value) for value in raw_times]
                if any(
                    value is None or value > request.forecast_cutoff for value in available_times
                ):
                    reasons.append(f"source {source_id} was not available by the cutoff")
        content = str(result.get("content") or "[empty recalled content]")
        if content == "[empty recalled content]":
            reasons.append("result content is empty")
        return RecalledMemory(
            result_id=str(result.get("id") or "missing-result-id"),
            content=content,
            occurred_at=occurred_at,
            source_ids=source_ids or ("missing-source-id",),
            eligible=not reasons,
            rejection_reasons=tuple(reasons),
        )
