"""Run and validate the one-example Modal LoRA sanity candidate."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from .audit import canonical_json, sha256_bytes, sha256_file
from .budget import RemoteBudgetLedger
from .capture import write_atomic
from .config import WORKSPACE_ROOT, load_settings
from .contracts import (
    ActionType,
    ActorAction,
    CheckpointRecord,
    SFTExample,
    SplitRole,
    VerificationCheck,
)

PHASE4_ROOT = WORKSPACE_ROOT / "artifacts" / "phase4" / "g0"
RUN_ROOT = WORKSPACE_ROOT / "artifacts" / "phase5"


def run_training() -> dict[str, object]:
    settings = load_settings()
    phase4_acceptance = PHASE4_ROOT / "acceptance.json"
    example_path = PHASE4_ROOT / "verified_example.json"
    if not phase4_acceptance.is_file() or not example_path.is_file():
        raise RuntimeError("a passed Phase 4 episode is required before LoRA training")
    phase4 = json.loads(phase4_acceptance.read_text(encoding="utf-8"))
    example = json.loads(example_path.read_text(encoding="utf-8"))
    if phase4.get("status") != "passed" or example.get("accepted") is not True:
        raise RuntimeError("Phase 4 did not produce an accepted verified example")

    training_config_core = {
        "learning_rate": 0.0001,
        "steps": 1,
        "seed": 0,
        "rank": 8,
        "alpha": 16,
        "target_modules": "all-linear",
        "base_model": settings.model_id,
        "base_model_revision": settings.model_revision,
    }
    training_config = {
        **training_config_core,
        "compute_gpu_priority": ["H100", "B200"],
        "compute_amendment": "config/hardware_amendment_2026-11-15.json",
    }
    identity = {
        "example_sha256": sha256_file(example_path),
        "training_config": training_config,
    }
    identity_sha = sha256_bytes(canonical_json(identity).encode())
    budget_identity_sha = sha256_bytes(
        canonical_json(
            {
                "example_sha256": sha256_file(example_path),
                "training_config": training_config_core,
            }
        ).encode()
    )
    run_id = "g1-" + identity_sha[:16]
    acceptance_path = RUN_ROOT / run_id / "acceptance.json"
    if acceptance_path.is_file():
        existing = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("status") != "passed":
            raise RuntimeError("existing Phase 5 acceptance artifact is invalid")
        return existing
    request_path = RUN_ROOT / run_id / "training_request.json"
    response_path = RUN_ROOT / run_id / "training_response.json"
    request = {
        "run_id": run_id,
        "prompt": example["prompt"],
        "target": example["target"],
        "seed": training_config["seed"],
        "steps": training_config["steps"],
        "learning_rate": training_config["learning_rate"],
    }
    write_atomic(request_path, (json.dumps(request, indent=2, sort_keys=True) + "\n").encode())
    if not response_path.is_file():
        budget = RemoteBudgetLedger()
        call_id = "phase5-modal-lora-" + budget_identity_sha[:16]
        budget.reserve_gpu_seconds(call_id=call_id, seconds=540)
        budget.reserve("modal_function", call_id=call_id, estimated_usd=Decimal("0.25"))
        command = [
            str(WORKSPACE_ROOT / ".venv" / "bin" / "modal"),
            "run",
            "modal_app.py",
            "--operation",
            "train_lora_sanity",
            "--input-path",
            str(request_path),
            "--output-path",
            str(response_path),
        ]
        environment = dict(os.environ)
        environment["MODAL_PROFILE"] = settings.modal_profile
        try:
            completed = subprocess.run(
                command,
                cwd=WORKSPACE_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30 * 60,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr
            failure = {
                "schema_version": 1,
                "status": "local_scheduling_timeout",
                "timeout_seconds": 30 * 60,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "recorded_at": datetime.now(UTC).isoformat(),
            }
            write_atomic(
                RUN_ROOT / run_id / "scheduling_failure.json",
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(),
            )
            raise RuntimeError("Modal LoRA training exceeded its local scheduling wait") from error
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
                RUN_ROOT / run_id / "training_failure.json",
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(),
            )
            raise RuntimeError("Modal LoRA training failed; see training_failure.json")

    remote = json.loads(response_path.read_text(encoding="utf-8"))
    if remote.get("status") != "passed":
        raise RuntimeError("Modal LoRA response is not passed")
    if remote.get("model_id") != settings.model_id:
        raise RuntimeError("LoRA response base model differs from the frozen actor")
    if remote.get("model_revision") != settings.model_revision:
        raise RuntimeError("LoRA response base revision differs from the frozen actor")
    remote_manifest_path = RUN_ROOT / run_id / "training_manifest_remote.json"
    remote_manifest = (
        json.loads(remote_manifest_path.read_text(encoding="utf-8"))
        if remote_manifest_path.is_file()
        else {}
    )
    cuda_device = str(remote.get("cuda_device") or remote_manifest.get("cuda_device") or "")
    if remote_manifest and remote_manifest.get("run_id") != run_id:
        raise RuntimeError("captured remote training manifest has the wrong run ID")
    if not any(name in cuda_device for name in ("H100", "B200")):
        raise RuntimeError("LoRA response did not use an approved amended GPU")
    losses = remote.get("losses")
    if not isinstance(losses, list) or len(losses) != 1:
        raise RuntimeError("LoRA sanity response must contain exactly one finite step loss")
    if remote.get("adapter_tensor_sha256_before") == remote.get("adapter_tensor_sha256_after"):
        raise RuntimeError("LoRA adapter weights did not change")
    if Decimal(str(remote.get("reload_max_abs_diff"))) > Decimal("0.00001"):
        raise RuntimeError("LoRA adapter reload equivalence failed")

    target_payload = json.loads(str(example["target"]))
    verified_at = datetime.now(UTC)
    target_action = ActorAction(
        action_id="teacher-correction-" + identity_sha[:16],
        episode_id=str(example["source_episode_id"]),
        step_id=1,
        tool=ActionType.PROPOSE_FORECASTER_PATCH,
        arguments=target_payload,
        proposed_at=verified_at,
        context_sha256=str(example["original_context_sha256"]),
    )
    verification = {
        VerificationCheck.SCHEMA: True,
        VerificationCheck.TEMPORAL: True,
        VerificationCheck.EXECUTION: True,
        VerificationCheck.PARTITION: True,
    }
    sft_example = SFTExample(
        example_id=str(example["example_id"]),
        source_episode_id=str(example["source_episode_id"]),
        split_role=SplitRole.TRAIN,
        origin_cutoff=datetime(2026, 9, 30, 23, 59, 59, tzinfo=UTC),
        original_context_sha256=str(example["original_context_sha256"]),
        target_action=target_action,
        token_ids=tuple(int(value) for value in remote["token_ids"]),
        loss_mask=tuple(int(value) for value in remote["loss_mask"]),
        assistant_target_start=int(remote["assistant_target_start"]),
        verification=verification,
        accepted=True,
        verified_at=verified_at,
    )
    dataset_path = RUN_ROOT / run_id / "sft_example.json"
    write_atomic(
        dataset_path,
        (json.dumps(sft_example.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    checkpoint = CheckpointRecord(
        checkpoint_id=run_id,
        parent_checkpoint=f"{settings.model_id}@{settings.model_revision}",
        base_model_id=settings.model_id,
        base_model_revision=settings.model_revision,
        base_weights_sha256=str(remote["base_tensor_sha256"]),
        adapter_sha256=str(remote["adapter_artifact_sha256"]),
        dataset_sha256=sha256_file(dataset_path),
        training_config_sha256=sha256_bytes(canonical_json(training_config).encode()),
        created_at=verified_at,
        effective_after=verified_at,
    )
    checkpoint_path = RUN_ROOT / run_id / "checkpoint.json"
    write_atomic(
        checkpoint_path,
        (json.dumps(checkpoint.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(),
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "run_id": run_id,
        "checkpoint": checkpoint.model_dump(mode="json"),
        "sft_example_sha256": sha256_file(dataset_path),
        "remote_training_response_sha256": sha256_file(response_path),
        "losses": losses,
        "reload_max_abs_diff": remote["reload_max_abs_diff"],
        "trainable_parameter_count": len(remote["trainable_parameter_names"]),
        "cuda_device": cuda_device,
        "budget_reservation_call_id": "phase5-modal-lora-" + budget_identity_sha[:16],
        "completed_at": verified_at.isoformat(),
        "trace_id": str(uuid5(NAMESPACE_URL, identity_sha)),
    }
    result["acceptance_sha256"] = sha256_bytes(canonical_json(result).encode())
    write_atomic(
        acceptance_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result


def main() -> None:
    result = run_training()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
