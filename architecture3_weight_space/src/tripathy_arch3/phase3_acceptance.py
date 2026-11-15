"""Run the bounded live Phase 3 Exa and past.dev acceptance checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from .audit import canonical_json, sha256_bytes, sha256_file
from .budget import RemoteBudgetLedger
from .capture import write_atomic
from .config import WORKSPACE_ROOT, load_settings
from .contracts import MemoryKind
from .evidence_gateway import (
    EvidenceGatewayError,
    EvidenceSearchRequest,
    ExaEvidenceGateway,
    SnapshotUnavailableError,
)
from .http_transport import HttpxTransport
from .memory_gateway import (
    LocalMemoryRegistry,
    MemoryIngestRequest,
    MemoryRecallRequest,
    PastMemoryGateway,
)

PHASE2_ACCEPTANCE = (
    WORKSPACE_ROOT / "artifacts" / "phase2" / "evaluations" / "7e8aa3bd6f9cffb6" / "acceptance.json"
)
AUTHORIZATION = WORKSPACE_ROOT / "config" / "execution_authorization.json"
OUTPUT = WORKSPACE_ROOT / "artifacts" / "phase3" / "acceptance.json"
HISTORICAL_CUTOFF = datetime(2026, 10, 15, 23, 59, 59, tzinfo=UTC)


def _authorization_time() -> datetime:
    payload = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    value = str(payload["authorized_at"])
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def run_acceptance() -> dict[str, object]:
    settings = load_settings()
    if not settings.exa_api_key or not settings.past_api_key:
        raise RuntimeError("Exa and past.dev credentials are required")
    if not PHASE2_ACCEPTANCE.is_file():
        raise RuntimeError("Phase 2 acceptance must exist before Phase 3")

    transport = HttpxTransport()
    budget = RemoteBudgetLedger()
    capture_root = WORKSPACE_ROOT / "artifacts" / "phase3" / "captures"
    source_available_at = _authorization_time()

    exa = ExaEvidenceGateway(
        api_key=settings.exa_api_key,
        transport=transport,
        budget=budget,
        capture_root=capture_root,
    )
    exa_request = EvidenceSearchRequest(
        call_id="phase3-exa-bls-snapshot-v1",
        episode_id="phase3-boundary-acceptance",
        hypothesis_id="public-cost-pressure-context",
        information_need=(
            "Find a dated primary BLS producer-price release that existed before the "
            "checkpoint cutoff; do not infer GPU rental prices from it."
        ),
        query="BLS Producer Price Index September 2026 news release",
        forecast_cutoff=HISTORICAL_CUTOFF,
        requested_at=datetime.now(UTC),
        include_domains=("bls.gov",),
        license_by_domain={"bls.gov": "US-PD-17-USC-105"},
        license_read_at=source_available_at,
        decision_affected="historical-snapshot gateway capability only",
        max_results=1,
    )
    exa_outcome: dict[str, object]
    exa_capture = capture_root / "exa" / exa_request.call_id / "response-body.bin"
    if exa_capture.is_file():
        captured_payload = json.loads(exa_capture.read_text(encoding="utf-8"))
        if captured_payload.get("tag") != "SNAPSHOT_NOT_ON_PLAN":
            raise RuntimeError("existing Exa capture is not the expected fail-closed response")
        exa_outcome = {
            "status": "snapshot_unavailable_fail_closed",
            "provider_request_id": captured_payload.get("requestId"),
            "response_sha256": sha256_file(exa_capture),
            "error_tag": captured_payload.get("tag"),
            "admissible_records": 0,
            "replay": True,
        }
    else:
        try:
            exa_receipt = exa.search(exa_request)
            exa_outcome = {
                "status": "snapshot_query_completed",
                "provider_request_id": exa_receipt.provider_request_id,
                "response_sha256": exa_receipt.response_sha256,
                "reported_cost_usd": str(exa_receipt.cost_usd),
                "records": [record.model_dump(mode="json") for record in exa_receipt.records],
                "admissible_records": sum(record.admissible for record in exa_receipt.records),
                "replay": False,
            }
        except SnapshotUnavailableError as error:
            exa_outcome = {
                "status": "snapshot_unavailable_fail_closed",
                "error": str(error),
                "admissible_records": 0,
                "replay": False,
            }
        except EvidenceGatewayError:
            raise

    memory_id = "arch3-phase2-boundary-lesson-v1"
    arm = "recursive-sft"
    generation = 0
    audience = f"arch3-{arm}-g{generation}"
    identity = audience + "-reader"
    registry = LocalMemoryRegistry()
    memory = PastMemoryGateway(
        api_key=settings.past_api_key,
        transport=transport,
        budget=budget,
        registry=registry,
        capture_root=capture_root,
    )
    phase2_sha = sha256_file(PHASE2_ACCEPTANCE)
    ingest_request = MemoryIngestRequest(
        call_id="phase3-past-boundary-ingest-v1",
        memory_id=memory_id,
        kind=MemoryKind.EPISODE_LESSON,
        arm=arm,
        generation=generation,
        identity=identity,
        audience_slug=audience,
        content=(
            "Local Phase 2 acceptance verified that the seven-target actor bundle ends "
            "at 2026-10-15 and that the unchanged official scorer exactly reproduces the "
            f"frozen checkpoint drift report. Acceptance artifact SHA-256: {phase2_sha}."
        ),
        source_ids=("phase2-acceptance-7e8aa3bd6f9cffb6",),
        source_available_at=(source_available_at,),
        effective_after=source_available_at,
        idempotency_key="phase3-past-boundary-ingest-v1",
    )
    registry_records = registry.read()
    if memory_id in registry_records:
        ingest_capture = capture_root / "past" / ingest_request.call_id / "response-body.bin"
        ingest_payload = json.loads(ingest_capture.read_text(encoding="utf-8"))
        ingestion_id = str(ingest_payload["ingestionId"])
    else:
        ingest_receipt = memory.ingest(ingest_request)
        ingestion_id = ingest_receipt.ingestion_id

    current_cutoff = datetime.now(UTC)
    current_recall = memory.recall(
        MemoryRecallRequest(
            call_id="phase3-past-current-recall-v2",
            arm=arm,
            generation=generation,
            identity=identity,
            audience_slug=audience,
            query="What did the local Phase 2 actor-boundary acceptance verify?",
            forecast_cutoff=current_cutoff,
            max_tokens=500,
            limit=5,
        )
    )
    historical_recall = memory.recall(
        MemoryRecallRequest(
            call_id="phase3-past-historical-recall-v2",
            arm=arm,
            generation=generation,
            identity=identity,
            audience_slug=audience,
            query="What did the local Phase 2 actor-boundary acceptance verify?",
            forecast_cutoff=HISTORICAL_CUTOFF,
            max_tokens=500,
            limit=5,
        )
    )
    current_eligible = [result for result in current_recall.results if result.eligible]
    historical_eligible = [result for result in historical_recall.results if result.eligible]
    if not current_eligible:
        raise RuntimeError("past.dev current recall returned no locally verified scoped memory")
    if historical_eligible:
        raise RuntimeError("future-derived memory became eligible at a historical cutoff")

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "completed_at": datetime.now(UTC).isoformat(),
        "authorization_sha256": sha256_file(AUTHORIZATION),
        "phase2_acceptance_sha256": phase2_sha,
        "exa": exa_outcome,
        "past": {
            "ingestion_id": ingestion_id,
            "memory_id": memory_id,
            "audience": audience,
            "identity": identity,
            "current_recall_sha256": current_recall.response_sha256,
            "current_eligible_results": len(current_eligible),
            "historical_recall_sha256": historical_recall.response_sha256,
            "historical_eligible_results": len(historical_eligible),
        },
        "budget_ledger": budget.read(),
    }
    result["acceptance_sha256"] = sha256_bytes(canonical_json(result).encode("utf-8"))
    write_atomic(OUTPUT, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def exa_outcome_status(result: dict[str, object]) -> str:
    exa = result.get("exa")
    if not isinstance(exa, dict):
        raise RuntimeError("Phase 3 result is missing its Exa outcome")
    return str(exa.get("status", "missing"))


def main() -> None:
    result = run_acceptance()
    print(
        json.dumps(
            {
                "acceptance_sha256": result["acceptance_sha256"],
                "exa_status": exa_outcome_status(result),
                "output": str(OUTPUT),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
