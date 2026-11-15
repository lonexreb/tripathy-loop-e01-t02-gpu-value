"""Run and validate the frozen Qwen snapshot sanity gate on Modal."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from decimal import Decimal

from .audit import canonical_json, sha256_bytes
from .budget import RemoteBudgetLedger
from .capture import write_atomic
from .config import WORKSPACE_ROOT, load_settings

RUN_ROOT = WORKSPACE_ROOT / "artifacts" / "phase4" / "model_sanity"


def run_modal_sanity() -> dict[str, object]:
    acceptance_path = RUN_ROOT / "acceptance.json"
    if acceptance_path.is_file():
        existing = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("status") != "passed":
            raise RuntimeError("existing Modal sanity acceptance artifact is invalid")
        return existing
    settings = load_settings()
    request_path = RUN_ROOT / "request.json"
    response_path = RUN_ROOT / "response.json"
    write_atomic(request_path, b"{}\n")
    if not response_path.is_file():
        call_id = "phase4-modal-model-sanity-v1"
        budget = RemoteBudgetLedger()
        budget.reserve_gpu_seconds(call_id=call_id, seconds=300)
        budget.reserve("modal_function", call_id=call_id, estimated_usd=Decimal("0.10"))
        command = [
            str(WORKSPACE_ROOT / ".venv" / "bin" / "modal"),
            "run",
            "modal_app.py",
            "--operation",
            "model_sanity",
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
                RUN_ROOT / "failure.json",
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(),
            )
            raise RuntimeError("Modal model sanity exceeded its local scheduling wait") from error
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
                RUN_ROOT / "failure.json",
                (json.dumps(failure, indent=2, sort_keys=True) + "\n").encode(),
            )
            raise RuntimeError("Modal model sanity failed; see artifacts/phase4/model_sanity")
    remote = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(remote, dict):
        raise RuntimeError("Modal model sanity response must be an object")
    if remote.get("model_id") != settings.model_id:
        raise RuntimeError("Modal sanity loaded a different model ID")
    if remote.get("model_revision") != settings.model_revision:
        raise RuntimeError("Modal sanity loaded a different model revision")
    if int(remote.get("parameters", 0)) <= 0:
        raise RuntimeError("Modal sanity reported no base model parameters")
    if remote.get("parameters") != remote.get("trainable_before_lora"):
        raise RuntimeError("base-model parameter accounting is inconsistent before LoRA")
    if int(remote.get("snapshot_file_count", 0)) <= 0:
        raise RuntimeError("the pinned model snapshot manifest is empty")
    if "L4" not in str(remote.get("cuda_device")):
        raise RuntimeError("the frozen Modal GPU must be an NVIDIA L4")
    now = datetime.now(UTC)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "model_id": remote["model_id"],
        "model_revision": remote["model_revision"],
        "model_class": remote["model_class"],
        "tokenizer_class": remote["tokenizer_class"],
        "parameters": remote["parameters"],
        "snapshot_file_count": remote["snapshot_file_count"],
        "snapshot_manifest_sha256": remote["snapshot_manifest_sha256"],
        "cuda_device": remote["cuda_device"],
        "dtype": remote["dtype"],
        "completed_at": now.isoformat(),
    }
    result["acceptance_sha256"] = sha256_bytes(canonical_json(result).encode())
    write_atomic(
        acceptance_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result


def main() -> None:
    print(json.dumps(run_modal_sanity(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
