from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tripathy_arch3.budget import RemoteBudgetLedger
from tripathy_arch3.http_transport import HttpResponse
from tripathy_arch3.teacher_gateway import OpenAITeacherGateway, TeacherDiagnosis


class OneResponseTransport:
    def __init__(self, output: dict[str, object]) -> None:
        self.body: dict[str, object] | None = None
        self.output = output

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json_body: object | None = None,
        timeout_seconds: float = 20.0,
    ) -> HttpResponse:
        assert method == "POST"
        assert url.endswith("/v1/responses")
        assert headers["Authorization"] == "Bearer test-secret"
        assert isinstance(json_body, dict)
        self.body = json_body
        payload = {
            "id": "resp_test",
            "status": "completed",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(self.output)}],
                }
            ],
        }
        return HttpResponse(
            status_code=200,
            headers={"content-type": "application/json", "x-request-id": "req-test"},
            body=json.dumps(payload).encode(),
        )


def make_budget(tmp_path: Path) -> RemoteBudgetLedger:
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "authorized": True,
                "budget": {
                    "maximum_total_usd": 1,
                    "maximum_modal_gpu_seconds": 1200,
                    "maximum_exa_search_calls": 2,
                    "maximum_past_write_calls": 3,
                    "maximum_past_recall_calls": 3,
                    "maximum_openai_teacher_calls": 2,
                    "maximum_modal_function_calls": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    return RemoteBudgetLedger(
        ledger_path=tmp_path / "ledger.json", authorization_path=authorization
    )


def test_teacher_uses_fixed_model_store_false_and_strict_schema(tmp_path: Path) -> None:
    transport = OneResponseTransport(
        {
            "failure_type": "interval_too_narrow",
            "claim_status": "fact",
            "confidence": "high",
            "evidence_ids": [],
            "diagnosis": "The committed interval missed the released target.",
            "recommended_method": "repricing_interval",
            "recommended_window_days": 60,
            "recommended_width_scale": "1.2",
            "uncertainty": "One origin is insufficient to establish improvement.",
        }
    )
    gateway = OpenAITeacherGateway(
        api_key="test-secret",
        transport=transport,
        budget=make_budget(tmp_path),
        capture_root=tmp_path / "artifacts" / "phase4" / "captures",
    )
    diagnosis, receipt = gateway.diagnose(
        call_id="teacher-diagnosis-test",
        experience={"actor_action": {"method": "drift"}, "official_score": 1.0},
    )

    assert diagnosis.recommended_method == "repricing_interval"
    assert receipt.model == "gpt-6.1-sol"
    assert transport.body is not None
    assert transport.body["model"] == "gpt-6.1-sol"
    assert transport.body["store"] is False
    text = transport.body["text"]
    assert isinstance(text, dict)
    output_format = text["format"]
    assert isinstance(output_format, dict)
    assert output_format["strict"] is True
    assert output_format["type"] == "json_schema"
    captures = list((tmp_path / "artifacts").rglob("*"))
    assert not any(path.is_file() and b"test-secret" in path.read_bytes() for path in captures)


def test_teacher_business_contract_rejects_out_of_range_correction() -> None:
    with pytest.raises(ValidationError, match="less than or equal to 2"):
        TeacherDiagnosis(
            failure_type="bad",
            claim_status="unknown",
            confidence="uncertain",
            evidence_ids=(),
            diagnosis="No supported claim.",
            recommended_method="repricing_interval",
            recommended_window_days=60,
            recommended_width_scale="3",
            uncertainty="Unknown.",
        )
