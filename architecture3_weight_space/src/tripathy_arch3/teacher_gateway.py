"""Fixed OpenAI teacher adapter using strict Responses API structured outputs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator

from .budget import RemoteBudgetLedger
from .capture import capture_json
from .config import WORKSPACE_ROOT
from .contracts import Confidence, FrozenContract, NonEmpty, Sha256
from .http_transport import HttpTransport

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_CAPTURE_ROOT = WORKSPACE_ROOT / "artifacts" / "phase4" / "captures"
TEACHER_MODEL: Literal["gpt-6.1-sol"] = "gpt-6.1-sol"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(UTC)


class ClaimAssessment(StrEnum):
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    UNKNOWN = "unknown"


class AllowedMethod(StrEnum):
    DRIFT = "drift"
    REPRICING_INTERVAL = "repricing_interval"


class TeacherDiagnosis(FrozenContract):
    failure_type: NonEmpty
    claim_status: ClaimAssessment
    confidence: Confidence
    evidence_ids: tuple[str, ...]
    diagnosis: NonEmpty
    recommended_method: AllowedMethod
    recommended_window_days: Literal[30, 60, 90]
    recommended_width_scale: Annotated[Decimal, Field(ge=Decimal("0.5"), le=Decimal("2"))]
    uncertainty: NonEmpty


class TeacherCorrection(FrozenContract):
    target_tool: Literal["propose_forecaster_patch"]
    method: AllowedMethod
    window_days: Literal[30, 60, 90]
    width_scale: Annotated[Decimal, Field(ge=Decimal("0.5"), le=Decimal("2"))]
    public_rationale: NonEmpty
    verification_required: tuple[Literal["schema", "temporal", "execution", "partition"], ...]


class TeacherCallReceipt(FrozenContract):
    call_id: NonEmpty
    model: Literal["gpt-6.1-sol"]
    response_id: NonEmpty
    response_sha256: Sha256
    request_sha256: Sha256
    created_at: datetime
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    estimated_cost_usd: Decimal
    output_json: dict[str, object]

    _created_at = field_validator("created_at")(_utc)


class TeacherGatewayError(RuntimeError):
    """OpenAI API or structured-output failure with locally captured evidence."""


def _strict_schema(model: type[FrozenContract]) -> dict[str, object]:
    schema = model.model_json_schema(mode="validation")
    schema["additionalProperties"] = False
    return schema


def _extract_output_text(payload: dict[str, object]) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        raise TeacherGatewayError("OpenAI response is missing output items")
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                raise TeacherGatewayError("OpenAI teacher refused the structured review")
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                return str(part["text"])
    raise TeacherGatewayError("OpenAI response contains no output_text")


class OpenAITeacherGateway:
    def __init__(
        self,
        *,
        api_key: str,
        transport: HttpTransport,
        budget: RemoteBudgetLedger,
        capture_root: Path = DEFAULT_CAPTURE_ROOT,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")
        self._api_key = api_key
        self._transport = transport
        self._budget = budget
        self._capture_root = capture_root

    def diagnose(
        self, *, call_id: str, experience: dict[str, object]
    ) -> tuple[TeacherDiagnosis, TeacherCallReceipt]:
        instructions = (
            "You are the fixed Architecture 3 diagnostic teacher. Review only the supplied "
            "committed actor action, frozen tool observations, and official scorer summary. "
            "Do not invent evidence, do not browse, do not alter the scorer, and do not claim "
            "that a method improved unless the supplied score proves it. Return a concise "
            "public diagnosis, not private reasoning."
        )
        output, receipt = self._structured_call(
            call_id=call_id,
            instructions=instructions,
            input_payload=experience,
            schema_name="architecture3_teacher_diagnosis",
            schema=_strict_schema(TeacherDiagnosis),
        )
        return TeacherDiagnosis.model_validate(output), receipt

    def construct_correction(
        self,
        *,
        call_id: str,
        original_context: dict[str, object],
        diagnosis: TeacherDiagnosis,
    ) -> tuple[TeacherCorrection, TeacherCallReceipt]:
        instructions = (
            "You are the fixed Architecture 3 correction constructor. Produce exactly one "
            "allowed forecaster-patch action for the original actor context. The action is a "
            "proposal only: local code will re-execute and verify it. Use no facts beyond the "
            "provided context and diagnosis. If method is drift, window_days must be 60 and "
            "width_scale must be 1."
        )
        output, receipt = self._structured_call(
            call_id=call_id,
            instructions=instructions,
            input_payload={
                "original_context": original_context,
                "diagnosis": diagnosis.model_dump(mode="json"),
            },
            schema_name="architecture3_teacher_correction",
            schema=_strict_schema(TeacherCorrection),
        )
        return TeacherCorrection.model_validate(output), receipt

    def _structured_call(
        self,
        *,
        call_id: str,
        instructions: str,
        input_payload: dict[str, object],
        schema_name: str,
        schema: dict[str, object],
    ) -> tuple[dict[str, object], TeacherCallReceipt]:
        self._budget.reserve("openai_teacher", call_id=call_id, estimated_usd=Decimal("0.03"))
        body = {
            "model": TEACHER_MODEL,
            "store": False,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 800,
            "instructions": instructions,
            "input": json.dumps(input_payload, sort_keys=True),
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        response = self._transport.request(
            "POST",
            OPENAI_RESPONSES_URL,
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
            },
            json_body=body,
            timeout_seconds=60.0,
        )
        request_artifact, response_artifact = capture_json(
            self._capture_root,
            provider="openai",
            call_id=call_id,
            request={"body": body, "method": "POST", "path": "/v1/responses"},
            response_status=response.status_code,
            response_headers=response.headers,
            response_body=response.body,
        )
        try:
            raw_payload = response.json()
        except ValueError as error:
            raise TeacherGatewayError("OpenAI returned a non-JSON response") from error
        if not isinstance(raw_payload, dict):
            raise TeacherGatewayError("OpenAI response must be an object")
        if response.status_code != 200:
            response_error = raw_payload.get("error")
            raise TeacherGatewayError(
                f"OpenAI request failed with status {response.status_code}: {response_error}"
            )
        if raw_payload.get("status") != "completed":
            raise TeacherGatewayError(
                f"OpenAI response did not complete: {raw_payload.get('status')}"
            )
        parsed = json.loads(_extract_output_text(raw_payload))
        if not isinstance(parsed, dict):
            raise TeacherGatewayError("OpenAI structured output must be an object")
        usage = raw_payload.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0
        output_tokens = int(usage.get("output_tokens", 0)) if isinstance(usage, dict) else 0
        estimated_cost = (
            Decimal(input_tokens) * Decimal("2") + Decimal(output_tokens) * Decimal("10")
        ) / Decimal(1_000_000)
        return parsed, TeacherCallReceipt(
            call_id=call_id,
            model=TEACHER_MODEL,
            response_id=str(raw_payload.get("id") or "missing-response-id"),
            response_sha256=response_artifact.sha256,
            request_sha256=request_artifact.sha256,
            created_at=datetime.now(UTC),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost,
            output_json=parsed,
        )
