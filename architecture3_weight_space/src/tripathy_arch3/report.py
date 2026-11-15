"""Generate a replay-oriented status report strictly from frozen local artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .audit import sha256_bytes, sha256_file
from .budget import RemoteBudgetLedger
from .capture import write_atomic
from .config import WORKSPACE_ROOT

REPORT_ROOT = WORKSPACE_ROOT / "artifacts" / "report"


def _record(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"status": "not_run", "path": path.relative_to(WORKSPACE_ROOT).as_posix()}
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = payload.get("status") if isinstance(payload, dict) else None
    return {
        "status": str(status or "recorded"),
        "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
        "sha256": sha256_file(path),
    }


def _single_acceptance(pattern: str) -> dict[str, object]:
    paths = sorted(WORKSPACE_ROOT.glob(pattern))
    if not paths:
        return {"status": "not_run", "pattern": pattern}
    if len(paths) > 1:
        return {
            "status": "ambiguous",
            "pattern": pattern,
            "paths": [path.relative_to(WORKSPACE_ROOT).as_posix() for path in paths],
        }
    return _record(paths[0])


def build_summary() -> dict[str, object]:
    manifest = json.loads(
        (WORKSPACE_ROOT / "experiment" / "manifest.json").read_text(encoding="utf-8")
    )
    phase2_relative = manifest.get("local_acceptance", {}).get("phase2", {}).get(
        "acceptance_artifact"
    )
    phase2 = (
        _record(WORKSPACE_ROOT / str(phase2_relative))
        if phase2_relative
        else {"status": "not_run", "reason": "no Phase 2 artifact selected in the manifest"}
    )
    phase3 = _record(WORKSPACE_ROOT / "artifacts" / "phase3" / "acceptance.json")
    phase4_sanity = _record(
        WORKSPACE_ROOT / "artifacts" / "phase4" / "model_sanity" / "acceptance.json"
    )
    phase4_sanity_failure = _record(
        WORKSPACE_ROOT / "artifacts" / "phase4" / "model_sanity" / "failure.json"
    )
    phase4 = _record(WORKSPACE_ROOT / "artifacts" / "phase4" / "g0" / "acceptance.json")
    phase4_failure = _record(
        WORKSPACE_ROOT / "artifacts" / "phase4" / "modal_environment_sanity_failure.json"
    )
    phase5 = _single_acceptance("artifacts/phase5/g1-*/acceptance.json")
    phase6 = _single_acceptance("artifacts/phase6/g1-*/acceptance.json")
    gpu_blocked = phase4.get("status") == "not_run" and phase4_failure.get("status") != "not_run"
    if phase6.get("status") == "passed":
        overall_status = "completed"
    elif gpu_blocked:
        overall_status = "blocked_external_modal_billing"
    else:
        overall_status = "in_progress"

    ledger = RemoteBudgetLedger().read()
    summary: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "overall_status": overall_status,
        "phases": {
            "phase2_information_boundary": phase2,
            "phase3_evidence_memory": phase3,
            "phase4_model_sanity": phase4_sanity,
            "phase4_model_sanity_failure": phase4_sanity_failure,
            "phase4_g0_episode": phase4,
            "phase4_modal_failure": phase4_failure,
            "phase5_lora_training": phase5,
            "phase6_admission": phase6,
        },
        "budget": {
            "estimated_total_usd": ledger.get("estimated_total_usd", "0"),
            "gpu_seconds_reserved": ledger.get("gpu_seconds_reserved", 0),
            "ledger_sha256": sha256_file(RemoteBudgetLedger().ledger_path),
        },
        "claims": {
            "qwen_model_downloaded": phase4.get("status") == "passed",
            "lora_weights_updated": phase5.get("status") == "passed",
            "admission_gate_executed": phase6.get("status") == "passed",
            "recursive_self_improvement_established": False,
        },
        "blocker": (
            "Modal requires a payment method before the frozen L4 function can start."
            if gpu_blocked
            else None
        ),
        "replay": {
            "local_acceptance": "uv run ruff check . && uv run mypy && uv run pytest",
            "preflight": "uv run arch3-preflight",
            "resume_remote_pipeline": (
                "uv run arch3-modal-sanity && uv run arch3-phase4-episode && "
                "uv run arch3-phase5-training && uv run arch3-phase6-admission && "
                "uv run arch3-report"
            ),
        },
    }
    summary["summary_sha256"] = sha256_bytes(
        json.dumps(summary, separators=(",", ":"), sort_keys=True).encode()
    )
    return summary


def _markdown(summary: dict[str, object]) -> str:
    phases = summary["phases"]
    claims = summary["claims"]
    budget = summary["budget"]
    assert isinstance(phases, dict)
    assert isinstance(claims, dict)
    assert isinstance(budget, dict)
    lines = [
        "# Architecture 3 execution report",
        "",
        f"Overall status: `{summary['overall_status']}`.",
        "",
        "## Phase status",
        "",
    ]
    for name, phase in phases.items():
        assert isinstance(phase, dict)
        lines.append(f"- `{name}`: `{phase.get('status', 'unknown')}`")
    lines.extend(
        [
            "",
            "## Budget ledger",
            "",
            f"- Estimated reserved cost: `${budget.get('estimated_total_usd')}`",
            f"- Reserved Modal GPU seconds: `{budget.get('gpu_seconds_reserved')}`",
            "",
            "## Claim boundary",
            "",
            f"- Qwen snapshot downloaded: `{claims.get('qwen_model_downloaded')}`",
            f"- LoRA weights updated and reloaded: `{claims.get('lora_weights_updated')}`",
            f"- Frozen admission gate executed: `{claims.get('admission_gate_executed')}`",
            "- Recursive self-improvement established: `False`",
            "",
        ]
    )
    blocker = summary.get("blocker")
    if blocker:
        lines.extend(["## Current blocker", "", str(blocker), ""])
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            "uv run arch3-preflight",
            "uv run ruff check .",
            "uv run mypy",
            "uv run pytest",
            "```",
            "",
            "After the Modal billing prerequisite is resolved:",
            "",
            "```bash",
            "uv run arch3-modal-sanity",
            "uv run arch3-phase4-episode",
            "uv run arch3-phase5-training",
            "uv run arch3-phase6-admission",
            "uv run arch3-report",
            "```",
            "",
            "Numerical replay uses frozen local artifacts; fresh provider calls are not needed.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report() -> dict[str, object]:
    summary = build_summary()
    write_atomic(
        REPORT_ROOT / "summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    write_atomic(REPORT_ROOT / "report.md", _markdown(summary).encode())
    return summary


def main() -> None:
    print(json.dumps(generate_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
