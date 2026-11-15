"""Evaluate one LoRA child under the frozen development/checkpoint admission gate."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

from .admission import GateOutcome, PairedGateMetrics, apply_frozen_admission_gate
from .audit import canonical_json, sha256_bytes, sha256_file
from .budget import RemoteBudgetLedger
from .capture import write_atomic
from .config import WORKSPACE_ROOT, load_settings
from .contracts import AdmissionDecision, AdmissionStatus, ArtifactRef, ForecastRow
from .data_boundary import ActorInputBundle, build_actor_input_bundle, read_frozen_target_ids
from .forecaster import (
    ForecasterPatch,
    build_actor_prompt,
    compile_forecast,
    parse_actor_patch,
    write_forecast_csv,
)
from .official_scorer import OfficialScorerAdapter, ProtectedTruth

PatchLookup = Callable[[date, str], ForecasterPatch]
REPO_ROOT = WORKSPACE_ROOT.parent
RUN_ROOT = WORKSPACE_ROOT / "artifacts" / "phase6"
PROPOSAL_ORIGIN = date(2026, 9, 30)
BLOCKS = {
    "development": (date(2026, 9, 1), date(2026, 9, 30)),
    "checkpoint": (date(2026, 10, 1), date(2026, 10, 15)),
    "protected": (date(2026, 10, 16), date(2026, 10, 30)),
}


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _write_immutable_json(path: Path, payload: object) -> None:
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.is_file():
        if path.read_bytes() != rendered:
            raise RuntimeError(f"immutable artifact already exists with different content: {path}")
        return
    write_atomic(path, rendered)


def _phase5_candidate() -> tuple[Path, dict[str, object]]:
    paths = sorted((WORKSPACE_ROOT / "artifacts" / "phase5").glob("g1-*/acceptance.json"))
    passed = [(path, _load_object(path)) for path in paths]
    passed = [(path, value) for path, value in passed if value.get("status") == "passed"]
    if len(passed) != 1:
        raise RuntimeError("exactly one passed Phase 5 candidate is required")
    return passed[0]


def _proposal_bundle() -> ActorInputBundle:
    return build_actor_input_bundle(
        source_series=REPO_ROOT / "research" / "out" / "series_daily.csv",
        bundle_root=RUN_ROOT / "proposal_bundle",
        cutoff_date=PROPOSAL_ORIGIN,
        target_ids=read_frozen_target_ids(REPO_ROOT / "research" / "adapted_series.txt"),
    )


def _run_adapter_actor(
    *, run_id: str, prompt: str, modal_profile: str
) -> dict[str, object]:
    candidate_root = RUN_ROOT / run_id
    request_path = candidate_root / "actor_request.json"
    response_path = candidate_root / "actor_response.json"
    request = {
        "run_id": run_id,
        "prompt": prompt,
        "seed": 0,
        "max_new_tokens": 256,
    }
    _write_immutable_json(request_path, request)
    if not response_path.is_file():
        call_id = "phase6-modal-adapter-actor-" + run_id
        budget = RemoteBudgetLedger()
        budget.reserve_gpu_seconds(call_id=call_id, seconds=180)
        budget.reserve("modal_function", call_id=call_id, estimated_usd=Decimal("0.10"))
        command = [
            str(WORKSPACE_ROOT / ".venv" / "bin" / "modal"),
            "run",
            "modal_app.py",
            "--operation",
            "actor_adapter",
            "--input-path",
            str(request_path),
            "--output-path",
            str(response_path),
        ]
        environment = dict(os.environ)
        environment["MODAL_PROFILE"] = modal_profile
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
            _write_immutable_json(candidate_root / "actor_failure.json", failure)
            raise RuntimeError("Modal adapter actor call failed; see actor_failure.json")
    return _load_object(response_path)


def _dates(start: date, end: date) -> tuple[date, ...]:
    if start > end:
        raise ValueError("block start may not follow its end")
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def _compile_block(
    *,
    block: str,
    patch: ForecasterPatch | PatchLookup,
    run_id: str,
    actor_generation: Literal["g1", "wsp1", "loop"] = "g1",
) -> tuple[Path, ActorInputBundle]:
    start, end = BLOCKS[block]
    target_ids = read_frozen_target_ids(REPO_ROOT / "research" / "adapted_series.txt")
    rows: list[ForecastRow] = []
    final_bundle: ActorInputBundle | None = None
    for origin in _dates(start, end):
        final_bundle = build_actor_input_bundle(
            source_series=REPO_ROOT / "research" / "out" / "series_daily.csv",
            bundle_root=RUN_ROOT / run_id / "actor_bundles" / block,
            cutoff_date=origin,
            target_ids=target_ids,
        )
        if isinstance(patch, ForecasterPatch):
            rows.extend(
                compile_forecast(
                    final_bundle, patch=patch, origin_date=origin, actor_generation=actor_generation
                )
            )
        else:  # one patch per series: compile each distinct patch, keep that series' rows
            chosen = {series_id: patch(origin, series_id) for series_id in target_ids}
            for distinct in sorted(set(chosen.values()), key=lambda item: item.width_scale):
                rows.extend(
                    row
                    for row in compile_forecast(
                        final_bundle,
                        patch=distinct,
                        origin_date=origin,
                        actor_generation=actor_generation,
                    )
                    if chosen[row.series_id] == distinct
                )
    if final_bundle is None:
        raise RuntimeError("forecast block unexpectedly contains no origins")
    forecast_path = RUN_ROOT / run_id / "forecasts" / f"{block}.csv"
    if forecast_path.is_file():
        forecast_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=forecast_path.parent, prefix=f".{block}.", suffix=".csv"
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            write_forecast_csv(temporary_path, tuple(rows))
            if temporary_path.read_bytes() != forecast_path.read_bytes():
                raise RuntimeError(f"immutable forecast differs on replay: {forecast_path}")
        finally:
            temporary_path.unlink(missing_ok=True)
    else:
        write_forecast_csv(forecast_path, tuple(rows))
    return forecast_path, final_bundle


def _score_block(
    *, block: str, forecast_path: Path, bundle: ActorInputBundle, run_id: str
) -> tuple[dict[str, object], Path]:
    evaluation_root = RUN_ROOT / run_id / "evaluations" / block
    report_path = evaluation_root / "official_report.json"
    cells_path = evaluation_root / "forecast_cells.csv"
    if report_path.is_file() and cells_path.is_file():
        return _load_object(report_path), cells_path
    scorer = REPO_ROOT / "gpu-price-pack-v0.1.1-handover" / "starter" / "scorer.py"
    result = OfficialScorerAdapter(
        scorer_path=scorer,
        python_executable=REPO_ROOT / ".venv" / "bin" / "python",
        expected_scorer_sha256=sha256_file(scorer),
    ).score(
        forecast_path=forecast_path,
        history_bundle=bundle,
        protected_truth=ProtectedTruth.from_path(
            REPO_ROOT / "research" / "out" / "series_daily.csv"
        ),
        output_directory=evaluation_root,
        mode="backtest",
    )
    if result.status != "scored" or result.cells_path is None:
        raise RuntimeError(f"official scorer did not produce scored cells for {block}")
    return _load_object(result.report_path), result.cells_path


def _run_paired(
    *,
    block: str,
    candidate_cells: Path,
    run_id: str,
    parent_cells: Path | None = None,
    label: str = "F0_drift",
) -> dict[str, object]:
    output_path = RUN_ROOT / run_id / "paired" / f"{block}_vs_{label}.json"
    if output_path.is_file():
        return _load_object(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parent_cells = parent_cells or REPO_ROOT / "research" / "out" / f"cells_{block}_drift.csv"
    completed = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-I",
            "-B",
            str(REPO_ROOT / "research" / "paired.py"),
            "--a",
            str(candidate_cells),
            "--b",
            str(parent_cells),
            "--out",
            str(output_path),
            "--reps",
            "2000",
            "--seed",
            "0",
        ],
        cwd=REPO_ROOT,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(f"paired comparison failed for {block}: {completed.stderr.strip()}")
    return _load_object(output_path)


def _summary(report: dict[str, object]) -> dict[str, object]:
    overall = report.get("overall")
    cells = report.get("cells")
    if not isinstance(overall, dict) or not isinstance(cells, dict):
        raise RuntimeError("official report is missing overall or cells")
    return {"status": report.get("status"), "overall": overall, "cells": cells}


def evaluate_patch(
    *,
    patch: ForecasterPatch,
    run_id: str,
    actor_generation: Literal["g1", "wsp1"] = "g1",
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], GateOutcome]:
    """Score one fixed patch once on development and checkpoint and apply the frozen gate."""
    reports: dict[str, dict[str, object]] = {}
    paired: dict[str, dict[str, object]] = {}
    for block in ("development", "checkpoint"):
        forecast_path, final_bundle = _compile_block(
            block=block, patch=patch, run_id=run_id, actor_generation=actor_generation
        )
        report, cells = _score_block(
            block=block,
            forecast_path=forecast_path,
            bundle=final_bundle,
            run_id=run_id,
        )
        reports[block] = report
        paired[block] = _run_paired(block=block, candidate_cells=cells, run_id=run_id)
    gate = apply_frozen_admission_gate(
        checkpoint=PairedGateMetrics.from_paired_payload(paired["checkpoint"]),
        development=PairedGateMetrics.from_paired_payload(paired["development"]),
    )
    return reports, paired, gate


def run_admission() -> dict[str, object]:
    settings = load_settings()
    _, phase5 = _phase5_candidate()
    checkpoint = phase5.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Phase 5 acceptance is missing its checkpoint record")
    run_id = str(phase5["run_id"])
    acceptance_path = RUN_ROOT / run_id / "acceptance.json"
    if acceptance_path.is_file():
        existing = _load_object(acceptance_path)
        if existing.get("status") != "passed":
            raise RuntimeError("existing Phase 6 acceptance artifact is invalid")
        return existing
    proposal_bundle = _proposal_bundle()
    prompt = build_actor_prompt(proposal_bundle, origin_date=PROPOSAL_ORIGIN)
    actor = _run_adapter_actor(
        run_id=run_id,
        prompt=prompt,
        modal_profile=settings.modal_profile,
    )
    if actor.get("model_id") != settings.model_id:
        raise RuntimeError("adapter actor response changed the frozen base model")
    if actor.get("model_revision") != settings.model_revision:
        raise RuntimeError("adapter actor response changed the frozen model revision")
    if actor.get("adapter_run_id") != run_id:
        raise RuntimeError("adapter actor response used the wrong child checkpoint")
    if actor.get("adapter_manifest_sha256") != checkpoint.get("adapter_sha256"):
        raise RuntimeError("adapter inference artifact differs from the trained checkpoint")
    if not any(name in str(actor.get("cuda_device")) for name in ("H100", "B200")):
        raise RuntimeError("adapter inference did not use an approved amended GPU")
    if actor.get("prompt_sha256") != sha256_bytes(prompt.encode()):
        raise RuntimeError("adapter actor response prompt hash does not match")
    patch = parse_actor_patch(str(actor.get("output") or ""))

    reports, paired, gate = evaluate_patch(patch=patch, run_id=run_id)
    metrics = {
        "schema_version": 1,
        "candidate_checkpoint": run_id,
        "parent_forecaster": "F_0",
        "proposal_origin": PROPOSAL_ORIGIN.isoformat(),
        "retrospective_not_causal_deployment": True,
        "patch": patch.model_dump(mode="json"),
        "reports": {name: _summary(report) for name, report in reports.items()},
        "paired_vs_parent": paired,
        "paired_script_sha256": sha256_file(REPO_ROOT / "research" / "paired.py"),
    }
    metrics_path = RUN_ROOT / run_id / "admission_metrics.json"
    _write_immutable_json(metrics_path, metrics)
    decision_path = RUN_ROOT / run_id / "admission_decision.json"
    if decision_path.is_file():
        existing_decision = AdmissionDecision.model_validate_json(
            decision_path.read_text(encoding="utf-8")
        )
        now = existing_decision.decided_at
    else:
        now = datetime.now(UTC)
    decision = AdmissionDecision(
        decision_id="admission-" + run_id,
        candidate_checkpoint=run_id,
        parent_checkpoint="F_0",
        status=gate.status,
        reason=gate.reason,
        protocol_sha256=sha256_file(REPO_ROOT / "research" / "PROTOCOL.md"),
        metrics_artifact=ArtifactRef(
            path=metrics_path.relative_to(WORKSPACE_ROOT).as_posix(),
            sha256=sha256_file(metrics_path),
            media_type="application/json",
        ),
        decided_at=now,
    )
    _write_immutable_json(decision_path, decision.model_dump(mode="json"))

    if gate.status is AdmissionStatus.ACCEPTED:
        forecast_path, final_bundle = _compile_block(
            block="protected", patch=patch, run_id=run_id
        )
        report, cells = _score_block(
            block="protected",
            forecast_path=forecast_path,
            bundle=final_bundle,
            run_id=run_id,
        )
        reports["protected"] = report
        paired["protected"] = _run_paired(
            block="protected", candidate_cells=cells, run_id=run_id
        )

    ledger = RemoteBudgetLedger().read()
    lineage: dict[str, object] = {
        "id": "F_arch3_" + run_id,
        "generation": 1,
        "parent": "F_0",
        "kind": "forecaster",
        "status": gate.status.value,
        "decision_reason": gate.reason,
        "improver": run_id,
        "actor_checkpoint": run_id,
        "base_model": f"{settings.model_id}@{settings.model_revision}",
        "method": "architecture3_weight_space.forecaster.compile_forecast",
        "params": patch.model_dump(mode="json"),
        "budget_used": {
            "estimated_total_usd_all_phases": ledger.get("estimated_total_usd"),
            "gpu_seconds_reserved_all_phases": ledger.get("gpu_seconds_reserved", 0),
            "evaluations": 2 + int(gate.status is AdmissionStatus.ACCEPTED),
        },
        "scores": {name: _summary(report) for name, report in reports.items()},
        "paired_vs_parent": paired,
        "evidence_ids": [],
        "checkpoint_effective_after": checkpoint.get("effective_after"),
        "evaluation_mode": "retrospective; not a claimed causal historical deployment",
        "created_utc": now.isoformat(),
    }
    lineage_path = REPO_ROOT / "research" / "lineage" / f"F_arch3_{run_id}.json"
    _write_immutable_json(lineage_path, lineage)

    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "candidate_checkpoint": run_id,
        "admission_status": gate.status.value,
        "admission_reason": gate.reason,
        "actor_response_sha256": sha256_file(RUN_ROOT / run_id / "actor_response.json"),
        "metrics_sha256": sha256_file(metrics_path),
        "decision_sha256": sha256_file(decision_path),
        "lineage_path": lineage_path.relative_to(REPO_ROOT).as_posix(),
        "lineage_sha256": sha256_file(lineage_path),
        "protected_scored": gate.status is AdmissionStatus.ACCEPTED,
        "rsi_claim_established": False,
        "claim_limit": (
            "One adapter admission result cannot establish recursive self-improvement; "
            "the frozen common-start I0/I1 descendant comparison was not executed."
        ),
        "completed_at": now.isoformat(),
    }
    result["acceptance_sha256"] = sha256_bytes(canonical_json(result).encode())
    _write_immutable_json(acceptance_path, result)
    return result


def main() -> None:
    result = run_admission()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
