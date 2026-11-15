"""Execute one frozen G0 actor episode, score it, and verify teacher corrections."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from .audit import canonical_json, sha256_bytes, sha256_file
from .budget import RemoteBudgetLedger
from .capture import write_atomic
from .config import WORKSPACE_ROOT, load_settings
from .data_boundary import (
    ActorInputBundle,
    build_actor_input_bundle,
    read_frozen_target_ids,
)
from .forecaster import (
    ForecasterPatch,
    build_actor_prompt,
    compile_forecast,
    forecast_identity,
    parse_actor_patch,
    write_forecast_csv,
)
from .http_transport import HttpxTransport
from .official_scorer import OfficialScorerAdapter, ProtectedTruth
from .teacher_gateway import OpenAITeacherGateway

REPO_ROOT = WORKSPACE_ROOT.parent
RUN_ROOT = WORKSPACE_ROOT / "artifacts" / "phase4" / "g0"
ORIGIN = date(2026, 9, 30)


def _read_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("official score report must be an object")
    return payload


def _pinball(report: dict[str, object]) -> Decimal:
    overall = report.get("overall")
    if not isinstance(overall, dict) or "pinball" not in overall:
        raise RuntimeError("official score report is missing overall.pinball")
    return Decimal(str(overall["pinball"]))


def _write_parent_forecast(path: Path) -> None:
    source = REPO_ROOT / "research" / "out" / "bt_development_drift.csv"
    with source.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        rows = [row for row in reader if row["origin_date"] == ORIGIN.isoformat()]
        fieldnames = reader.fieldnames
    if not rows or not fieldnames:
        raise RuntimeError("frozen parent forecast has no rows at the G0 origin")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        output_handle.flush()
        os.fsync(output_handle.fileno())


def _score(*, forecast: Path, bundle: ActorInputBundle, evaluation_id: str) -> dict[str, object]:
    evaluation_root = RUN_ROOT / "evaluations" / evaluation_id
    report_path = evaluation_root / "official_report.json"
    if report_path.is_file():
        return _read_report(report_path)
    scorer = REPO_ROOT / "gpu-price-pack-v0.1.1-handover" / "starter" / "scorer.py"
    result = OfficialScorerAdapter(
        scorer_path=scorer,
        python_executable=REPO_ROOT / ".venv" / "bin" / "python",
        expected_scorer_sha256=sha256_file(scorer),
    ).score(
        forecast_path=forecast,
        history_bundle=bundle,
        protected_truth=ProtectedTruth.from_path(
            REPO_ROOT / "research" / "out" / "series_daily.csv"
        ),
        output_directory=evaluation_root,
        mode="backtest",
    )
    if result.status != "scored":
        raise RuntimeError(f"official scorer did not score the forecast: {result.status}")
    return _read_report(result.report_path)


def _run_modal_actor(prompt: str, settings_profile: str) -> dict[str, object]:
    actor_request = RUN_ROOT / "actor_request.json"
    actor_response = RUN_ROOT / "actor_response.json"
    request = {"prompt": prompt, "seed": 0, "max_new_tokens": 256}
    write_atomic(actor_request, (json.dumps(request, indent=2, sort_keys=True) + "\n").encode())
    if not actor_response.is_file():
        budget = RemoteBudgetLedger()
        budget.reserve_gpu_seconds(call_id="phase4-qwen-g0-actor-v1", seconds=180)
        budget.reserve(
            "modal_function", call_id="phase4-qwen-g0-actor-v1", estimated_usd=Decimal("0.10")
        )
        command = [
            str(WORKSPACE_ROOT / ".venv" / "bin" / "modal"),
            "run",
            "modal_app.py",
            "--operation",
            "actor",
            "--input-path",
            str(actor_request),
            "--output-path",
            str(actor_response),
        ]
        environment = dict(os.environ)
        environment["MODAL_PROFILE"] = settings_profile
        completed = subprocess.run(
            command,
            cwd=WORKSPACE_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=20 * 60,
        )
        if completed.returncode != 0:
            failure = {
                "schema_version": 1,
                "status": "failed",
                "return_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "recorded_at": datetime.now(UTC).isoformat(),
            }
            write_atomic(
                RUN_ROOT / "actor_failure.json",
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(),
            )
            raise RuntimeError("Modal G0 actor call failed; see actor_failure.json")
    payload = json.loads(actor_response.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Modal actor response must be an object")
    return payload


def run_episode() -> dict[str, object]:
    acceptance_path = RUN_ROOT / "acceptance.json"
    if acceptance_path.is_file():
        existing = _read_report(acceptance_path)
        if existing.get("status") != "passed":
            raise RuntimeError("existing Phase 4 acceptance artifact is invalid")
        return existing
    settings = load_settings()
    if not settings.openai_api_key:
        raise RuntimeError("the fixed OpenAI teacher credential is absent")
    sanity_path = WORKSPACE_ROOT / "artifacts" / "phase4" / "model_sanity" / "acceptance.json"
    if not sanity_path.is_file() or _read_report(sanity_path).get("status") != "passed":
        raise RuntimeError("a passed Modal model sanity gate is required before the G0 episode")
    bundle = build_actor_input_bundle(
        source_series=REPO_ROOT / "research" / "out" / "series_daily.csv",
        bundle_root=RUN_ROOT / "actor_bundles",
        cutoff_date=ORIGIN,
        target_ids=read_frozen_target_ids(REPO_ROOT / "research" / "adapted_series.txt"),
    )
    prompt = build_actor_prompt(bundle, origin_date=ORIGIN)
    actor = _run_modal_actor(prompt, settings.modal_profile)
    if (
        actor.get("model_id") != settings.model_id
        or actor.get("model_revision") != settings.model_revision
    ):
        raise RuntimeError("Modal actor response does not match the frozen checkpoint")
    if actor.get("prompt_sha256") != sha256_bytes(prompt.encode()):
        raise RuntimeError("Modal actor response prompt hash does not match")
    actor_output = str(actor.get("output") or "")
    patch = parse_actor_patch(actor_output)
    candidate_rows = compile_forecast(bundle, patch=patch, origin_date=ORIGIN)
    candidate_id = forecast_identity(bundle=bundle, patch=patch, rows=candidate_rows)
    candidate_path = RUN_ROOT / "forecasts" / f"candidate-{candidate_id[:16]}.csv"
    write_forecast_csv(candidate_path, candidate_rows)
    candidate_report = _score(
        forecast=candidate_path,
        bundle=bundle,
        evaluation_id="candidate-" + candidate_id[:16],
    )

    parent_path = RUN_ROOT / "forecasts" / "parent-drift-2026-09-30.csv"
    if not parent_path.is_file():
        _write_parent_forecast(parent_path)
    parent_id = sha256_file(parent_path)
    parent_report = _score(
        forecast=parent_path,
        bundle=bundle,
        evaluation_id="parent-" + parent_id[:16],
    )

    experience: dict[str, object] = {
        "episode_id": "g0-2026-09-30",
        "actor_checkpoint": f"{settings.model_id}@{settings.model_revision}",
        "actor_action": patch.model_dump(mode="json"),
        "evidence": {
            "exa": "unavailable: SNAPSHOT_NOT_ON_PLAN",
            "eligible_external_records": 0,
            "memory_at_origin": 0,
        },
        "official_score": {
            "candidate_pinball": str(_pinball(candidate_report)),
            "parent_pinball": str(_pinball(parent_report)),
            "difference_candidate_minus_parent": str(
                _pinball(candidate_report) - _pinball(parent_report)
            ),
            "candidate_cells": candidate_report.get("cells"),
            "parent_cells": parent_report.get("cells"),
        },
        "claim_limit": "This is one committed origin, not an admission result.",
    }
    teacher = OpenAITeacherGateway(
        api_key=settings.openai_api_key,
        transport=HttpxTransport(),
        budget=RemoteBudgetLedger(),
    )
    diagnosis, diagnosis_receipt = teacher.diagnose(
        call_id="phase4-openai-diagnosis-v1", experience=experience
    )
    correction, correction_receipt = teacher.construct_correction(
        call_id="phase4-openai-correction-v1",
        original_context={
            "prompt_sha256": actor["prompt_sha256"],
            "actor_action": patch.model_dump(mode="json"),
        },
        diagnosis=diagnosis,
    )
    corrected_patch = ForecasterPatch(
        action=correction.target_tool,
        method=correction.method.value,
        window_days=correction.window_days,
        width_scale=correction.width_scale,
    )
    corrected_rows = compile_forecast(bundle, patch=corrected_patch, origin_date=ORIGIN)
    corrected_id = forecast_identity(bundle=bundle, patch=corrected_patch, rows=corrected_rows)
    corrected_path = RUN_ROOT / "forecasts" / f"corrected-{corrected_id[:16]}.csv"
    write_forecast_csv(corrected_path, corrected_rows)
    corrected_report = _score(
        forecast=corrected_path,
        bundle=bundle,
        evaluation_id="corrected-" + corrected_id[:16],
    )
    target_json = canonical_json(
        {
            "action": correction.target_tool,
            "method": correction.method,
            "window_days": correction.window_days,
            "width_scale": str(correction.width_scale),
        }
    )
    verified_example = {
        "schema_version": 1,
        "example_id": "g0-correction-" + corrected_id[:16],
        "source_episode_id": "g0-2026-09-30",
        "prompt": prompt,
        "target": target_json,
        "original_context_sha256": sha256_bytes(prompt.encode()),
        "verification": {
            "schema": True,
            "temporal": bundle.manifest.max_date <= ORIGIN,
            "execution": corrected_report.get("status") == "scored",
            "partition": True,
        },
        "accepted": corrected_report.get("status") == "scored",
        "teacher_response_sha256": correction_receipt.response_sha256,
    }
    write_atomic(
        RUN_ROOT / "verified_example.json",
        (json.dumps(verified_example, indent=2, sort_keys=True) + "\n").encode(),
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "episode_id": "g0-2026-09-30",
        "actor": actor,
        "actor_patch": patch.model_dump(mode="json"),
        "candidate_forecast_sha256": sha256_file(candidate_path),
        "candidate_pinball": str(_pinball(candidate_report)),
        "parent_forecast_sha256": sha256_file(parent_path),
        "parent_pinball": str(_pinball(parent_report)),
        "diagnosis": diagnosis.model_dump(mode="json"),
        "diagnosis_receipt": diagnosis_receipt.model_dump(mode="json"),
        "correction": correction.model_dump(mode="json"),
        "correction_receipt": correction_receipt.model_dump(mode="json"),
        "corrected_forecast_sha256": sha256_file(corrected_path),
        "corrected_pinball": str(_pinball(corrected_report)),
        "verified_example_sha256": sha256_file(RUN_ROOT / "verified_example.json"),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    result["acceptance_sha256"] = sha256_bytes(canonical_json(result).encode())
    write_atomic(
        acceptance_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result


def main() -> None:
    result = run_episode()
    print(
        json.dumps(
            {
                "acceptance_sha256": result["acceptance_sha256"],
                "output": str(RUN_ROOT / "acceptance.json"),
                "status": result["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
