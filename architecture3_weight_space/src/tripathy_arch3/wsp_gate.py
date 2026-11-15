"""Repo side of the weight-space-programming arm: actor prompts in, frozen gate out.

    uv run python -m tripathy_arch3.wsp_gate prompts   # before the local kernel (wsp.py)
    uv run python -m tripathy_arch3.wsp_gate gate      # after `wsp.py actor`
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime

from .audit import sha256_file
from .config import WORKSPACE_ROOT
from .contracts import AdmissionStatus
from .data_boundary import build_actor_input_bundle, read_frozen_target_ids
from .forecaster import build_actor_prompt, parse_actor_patch
from .phase6_admission import (
    BLOCKS,
    REPO_ROOT,
    RUN_ROOT,
    _dates,
    _summary,
    _write_immutable_json,
    evaluate_patch,
)

WSP_ROOT = WORKSPACE_ROOT / "artifacts" / "wsp"
PREREGISTRATION = WORKSPACE_ROOT / "config" / "wsp_preregistration_2026-11-15.json"


def write_prompts() -> None:
    """One actor prompt per development origin, built through the cutoff-safe bundle."""
    targets = read_frozen_target_ids(REPO_ROOT / "research" / "adapted_series.txt")
    prompts: dict[str, str] = {}
    for origin in _dates(*BLOCKS["development"]):
        bundle = build_actor_input_bundle(
            source_series=REPO_ROOT / "research" / "out" / "series_daily.csv",
            bundle_root=WSP_ROOT / "actor_bundles",
            cutoff_date=origin,
            target_ids=targets,
        )
        prompts[origin.isoformat()] = build_actor_prompt(bundle, origin_date=origin)
    WSP_ROOT.mkdir(parents=True, exist_ok=True)
    (WSP_ROOT / "prompts.json").write_text(json.dumps(prompts, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(prompts)} prompts")


def _rows_without_method(run_id: str, block: str) -> list[list[str]]:
    with (RUN_ROOT / run_id / "forecasts" / f"{block}.csv").open(encoding="utf-8") as handle:
        return [row[:-1] for row in csv.reader(handle)]


def run_gate() -> None:
    actor = json.loads((WSP_ROOT / "actor_response.json").read_text(encoding="utf-8"))
    manifest_path = WSP_ROOT / "edit" / "manifest.json"
    if actor["edit_manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError("actor response was produced by a different edit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patch = parse_actor_patch(actor["output"])
    run_id = "wsp1-" + actor["edited_tensor_sha256"][:16]
    reports, paired, gate = evaluate_patch(patch=patch, run_id=run_id, actor_generation="wsp1")

    # Direct-patch control: the same patch with no model must give the same numbers.
    control_id = "control-direct-" + run_id
    evaluate_patch(patch=patch, run_id=control_id)
    control_matches = all(
        _rows_without_method(run_id, block) == _rows_without_method(control_id, block)
        for block in ("development", "checkpoint")
    )

    now = datetime.now(UTC).isoformat()
    lineage = {
        "id": "F_arch3_" + run_id,
        "generation": 1,
        "parent": "F_0",
        "kind": "forecaster",
        "status": gate.status.value,
        "decision_reason": gate.reason,
        "improver": "weight-space programming (direct edit, no gradient steps)",
        "base_model": manifest["base_model"],
        "method": "architecture3_weight_space.forecaster.compile_forecast",
        "params": patch.model_dump(mode="json"),
        "weight_edit": {
            key: manifest[key]
            for key in (
                "tensor",
                "changed_weights",
                "fraction_of_model",
                "relative_change_frobenius",
                "changed_columns",
                "decisions",
                "base_tensor_sha256",
                "edited_tensor_sha256",
                "gradient_steps",
            )
        },
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "budget_used": {"remote_usd": 0, "gpu_seconds": 0, "evaluations": 2},
        "scores": {name: _summary(report) for name, report in reports.items()},
        "paired_vs_parent": paired,
        "direct_patch_control_matches": control_matches,
        "evidence_ids": [],
        "evaluation_mode": "retrospective; not a claimed causal historical deployment",
        "created_utc": now,
    }
    _write_immutable_json(REPO_ROOT / "research" / "lineage" / f"F_arch3_{run_id}.json", lineage)
    result = {
        "schema_version": 1,
        "status": "passed",
        "candidate": run_id,
        "patch": patch.model_dump(mode="json"),
        "admission_status": gate.status.value,
        "admission_reason": gate.reason,
        "checkpoint": gate.checkpoint.model_dump(mode="json"),
        "development": gate.development.model_dump(mode="json"),
        "direct_patch_control_matches": control_matches,
        "protected_scored": False,
        "rsi_claim_established": False,
        "admitted": gate.status is AdmissionStatus.ACCEPTED,
        "completed_at": now,
    }
    _write_immutable_json(WSP_ROOT / "acceptance.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    {"prompts": write_prompts, "gate": run_gate}[sys.argv[1]]()
