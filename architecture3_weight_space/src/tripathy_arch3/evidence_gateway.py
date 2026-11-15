"""Cutoff-safe Exa evidence discovery with immutable local replay captures."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Self
from urllib.parse import urlparse

from pydantic import Field, HttpUrl, field_validator, model_validator

from .audit import canonical_json, sha256_bytes
from .budget import RemoteBudgetLedger
from .capture import capture_json
from .config import WORKSPACE_ROOT
from .contracts import ArtifactRef, EvidenceRecord, FrozenContract, NonEmpty
from .http_transport import HttpTransport

EXA_SEARCH_URL = "https://api.exa.ai/search"
DEFAULT_CAPTURE_ROOT = WORKSPACE_ROOT / "artifacts" / "phase3" / "captures"
PROHIBITED_DOMAINS = (
    "vast.ai",
    "runpod.io",
    "ebay.com",
    "reddit.com",
    "gpulist.ai",
    "cloud-gpus.com",
    "skypilot.co",
    "dstack.ai",
    "silicondata.com",
    "semianalysis.com",
)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC)


def _parse_provider_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("Z", "+00:00")
    return _utc(datetime.fromisoformat(normalized))


def _domain(value: str) -> str:
    candidate = value.strip().lower()
    if "://" in candidate:
        candidate = urlparse(candidate).hostname or ""
    else:
        candidate = candidate.split("/", maxsplit=1)[0]
    return candidate.removeprefix("www.")


def _matches_domain(hostname: str, domain: str) -> bool:
    return hostname == domain or hostname.endswith("." + domain)


class EvidenceGatewayError(RuntimeError):
    """Provider or policy failure that leaves a captured diagnostic."""


class SnapshotUnavailableError(EvidenceGatewayError):
    """Historical snapshots are not enabled for the configured Exa account."""


class EvidenceSearchRequest(FrozenContract):
    call_id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]{2,127}$")]
    episode_id: NonEmpty
    hypothesis_id: NonEmpty
    information_need: NonEmpty
    query: NonEmpty
    forecast_cutoff: datetime
    requested_at: datetime
    include_domains: tuple[NonEmpty, ...]
    license_by_domain: dict[NonEmpty, NonEmpty]
    license_read_at: datetime
    decision_affected: NonEmpty
    max_results: Annotated[int, Field(ge=1, le=10)] = 3

    _timestamps = field_validator("forecast_cutoff", "requested_at", "license_read_at")(_utc)

    @model_validator(mode="after")
    def validate_policy_inputs(self) -> Self:
        if not self.include_domains:
            raise ValueError("include_domains may not be empty")
        domains = tuple(_domain(value) for value in self.include_domains)
        if not all(domains) or len(set(domains)) != len(domains):
            raise ValueError("include_domains must be valid and unique")
        prohibited = [
            domain
            for domain in domains
            if any(_matches_domain(domain, blocked) for blocked in PROHIBITED_DOMAINS)
        ]
        if prohibited:
            raise ValueError("prohibited evidence domain: " + ", ".join(prohibited))
        licensed = {_domain(value) for value in self.license_by_domain}
        if not set(domains).issubset(licensed):
            raise ValueError("every included domain needs a read license declaration")
        return self

    def provider_body(self) -> dict[str, object]:
        cutoff = self.forecast_cutoff.isoformat().replace("+00:00", "Z")
        return {
            "query": self.query,
            "type": "lite",
            "objective": (
                f"Episode {self.episode_id}; hypothesis {self.hypothesis_id}; "
                f"need: {self.information_need}. Rank public primary evidence "
                f"whose content and publication are available by {cutoff}."
            ),
            "includeDomains": list(self.include_domains),
            "excludeDomains": list(PROHIBITED_DOMAINS),
            "endPublishedDate": cutoff,
            "numResults": self.max_results,
            "moderation": True,
            "contents": {"snapshotAsOf": cutoff, "text": True},
            "userIdentifier": sha256_bytes(self.episode_id.encode("utf-8"))[:32],
        }


class EvidenceSearchReceipt(FrozenContract):
    call_id: NonEmpty
    provider_request_id: NonEmpty
    request_artifact: ArtifactRef
    response_artifact: ArtifactRef
    response_sha256: str
    cost_usd: Decimal
    retrieved_at: datetime
    records: tuple[EvidenceRecord, ...]

    _retrieved_at = field_validator("retrieved_at")(_utc)


class ExaEvidenceGateway:
    def __init__(
        self,
        *,
        api_key: str,
        transport: HttpTransport,
        budget: RemoteBudgetLedger,
        capture_root: Path = DEFAULT_CAPTURE_ROOT,
    ) -> None:
        if not api_key:
            raise ValueError("Exa API key is required")
        self._api_key = api_key
        self._transport = transport
        self._budget = budget
        self._capture_root = capture_root

    def search(self, request: EvidenceSearchRequest) -> EvidenceSearchReceipt:
        body = request.provider_body()
        self._budget.reserve(
            "exa_search",
            call_id=request.call_id,
            estimated_usd=Decimal("0.01"),
        )
        retrieved_at = datetime.now(UTC)
        response = self._transport.request(
            "POST",
            EXA_SEARCH_URL,
            headers={"x-api-key": self._api_key, "Content-Type": "application/json"},
            json_body=body,
        )
        request_artifact, response_artifact = capture_json(
            self._capture_root,
            provider="exa",
            call_id=request.call_id,
            request={
                "adapter_context": request.model_dump(mode="json"),
                "provider_body": body,
            },
            response_status=response.status_code,
            response_headers=response.headers,
            response_body=response.body,
        )
        response_sha = sha256_bytes(response.body)
        try:
            payload = response.json()
        except ValueError as error:
            raise EvidenceGatewayError(
                f"Exa returned non-JSON status {response.status_code}; response {response_sha}"
            ) from error
        if not isinstance(payload, dict):
            raise EvidenceGatewayError("Exa response must be a JSON object")
        if response.status_code != 200:
            tag = str(payload.get("tag", "unknown"))
            if tag.startswith("SNAPSHOT_"):
                raise SnapshotUnavailableError(
                    f"Exa historical snapshot unavailable ({tag}); response {response_sha}"
                )
            raise EvidenceGatewayError(
                f"Exa request failed with status {response.status_code} ({tag}); "
                f"response {response_sha}"
            )

        provider_request_id = str(payload.get("requestId") or "missing-request-id")
        warnings = payload.get("warnings", [])
        notices = payload.get("notices", [])
        snapshot_warning = any(
            isinstance(item, dict) and "snapshot" in canonical_json(item).lower()
            for item in (warnings if isinstance(warnings, list) else [])
        )
        restrictions_relaxed = any(
            isinstance(item, dict) and item.get("code") == "RESTRICTIONS_RELAXED"
            for item in (notices if isinstance(notices, list) else [])
        )
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise EvidenceGatewayError("Exa results must be an array")
        records = tuple(
            self._to_record(
                request=request,
                result=result,
                provider_request_id=provider_request_id,
                response_sha=response_sha,
                retrieved_at=retrieved_at,
                snapshot_warning=snapshot_warning,
                restrictions_relaxed=restrictions_relaxed,
            )
            for result in raw_results
            if isinstance(result, dict)
        )
        cost = payload.get("costDollars", {})
        cost_total = cost.get("total", 0) if isinstance(cost, dict) else 0
        return EvidenceSearchReceipt(
            call_id=request.call_id,
            provider_request_id=provider_request_id,
            request_artifact=request_artifact,
            response_artifact=response_artifact,
            response_sha256=response_sha,
            cost_usd=Decimal(str(cost_total)),
            retrieved_at=retrieved_at,
            records=records,
        )

    @staticmethod
    def _to_record(
        *,
        request: EvidenceSearchRequest,
        result: dict[str, object],
        provider_request_id: str,
        response_sha: str,
        retrieved_at: datetime,
        snapshot_warning: bool,
        restrictions_relaxed: bool,
    ) -> EvidenceRecord:
        url = str(result.get("url") or "https://invalid.local/missing-url")
        hostname = _domain(url)
        licensed_domain = next(
            (
                _domain(candidate)
                for candidate in request.license_by_domain
                if _matches_domain(hostname, _domain(candidate))
            ),
            None,
        )
        license_id = (
            request.license_by_domain[licensed_domain]
            if licensed_domain is not None
            else "unverified"
        )
        published_at = _parse_provider_datetime(result.get("publishedDate"))
        text = result.get("text")
        passage = text.strip() if isinstance(text, str) and text.strip() else "[no snapshot text]"
        reasons: list[str] = []
        if provider_request_id == "missing-request-id":
            reasons.append("provider request ID missing")
        if snapshot_warning:
            reasons.append("provider warned that the requested snapshot was not honored")
        if restrictions_relaxed:
            reasons.append("provider relaxed request restrictions")
        if any(_matches_domain(hostname, blocked) for blocked in PROHIBITED_DOMAINS):
            reasons.append("source domain is prohibited")
        if licensed_domain is None:
            reasons.append("source domain has no read license declaration")
        if published_at is None:
            reasons.append("publication time missing")
        elif published_at > request.forecast_cutoff:
            reasons.append("publication time is after the forecast cutoff")
        if passage == "[no snapshot text]":
            reasons.append("historical snapshot content missing")
        admissible = not reasons
        source_id = str(result.get("id") or url)
        evidence_id = (
            "exa-"
            + sha256_bytes(
                canonical_json(
                    {
                        "call_id": request.call_id,
                        "source_id": source_id,
                        "response_sha256": response_sha,
                    }
                ).encode("utf-8")
            )[:24]
        )
        return EvidenceRecord(
            evidence_id=evidence_id,
            source_id=source_id,
            episode_id=request.episode_id,
            query=request.query,
            source_url=HttpUrl(url),
            title=str(result.get("title") or "Untitled source"),
            passage=passage,
            license_id=license_id,
            license_read_at=request.license_read_at,
            provider_request_id=provider_request_id,
            snapshot_requested_at=request.forecast_cutoff,
            snapshot_confirmed=not snapshot_warning,
            published_at=published_at,
            event_time=published_at,
            available_at=published_at,
            retrieved_at=retrieved_at,
            forecast_cutoff=request.forecast_cutoff,
            content_sha256=sha256_bytes(passage.encode("utf-8")),
            response_sha256=response_sha,
            decision_affected=request.decision_affected,
            admissible=admissible,
            rejection_reason="; ".join(reasons) if reasons else None,
        )
