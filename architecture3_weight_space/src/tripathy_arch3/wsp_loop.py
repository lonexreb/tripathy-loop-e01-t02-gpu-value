"""Autonomous weight-space-programming loop over the forecasting actor.

    .venv-wsp/bin/python -m tripathy_arch3.wsp_loop run --arm i0 --cycles 3
    .venv-wsp/bin/python -m tripathy_arch3.wsp_loop run --arm i1 --cycles 3
    .venv-wsp/bin/python -m tripathy_arch3.wsp_loop protected

Each cycle: an LLM editor (headless Claude, no tools) reads a development-only brief and
proposes one conditional edit; the kernel programs it in closed form (no optimizer, no
backward pass); the frozen gate accepts or rejects it against the current actor. Re-running
a finished arm replays the stored editor replies instead of calling the editor again.

All edits live in the last MLP's down_proj. Its input does not depend on its own weights and
its output is added to the final residual, so one cached pass of the base model gives every
edited actor's decisions exactly; accepted edits are re-checked on the real weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from . import wsp
from .admission import PairedGateMetrics, apply_frozen_admission_gate
from .audit import sha256_file
from .config import WORKSPACE_ROOT
from .contracts import AdmissionStatus
from .data_boundary import build_actor_input_bundle, read_frozen_target_ids
from .forecaster import ForecasterPatch, summarize_actor_history
from .phase6_admission import (
    BLOCKS,
    REPO_ROOT,
    RUN_ROOT,
    _compile_block,
    _dates,
    _run_paired,
    _score_block,
)

ROOT = WORKSPACE_ROOT / "artifacts" / "wsp_loop"
PREREGISTRATION = WORKSPACE_ROOT / "config" / "wsp_loop_preregistration_2026-11-15.json"
LAYER = "model.layers.23.mlp.down_proj"
BANDS = {"narrow": Decimal("0.5"), "medium": Decimal("1"), "wide": Decimal("2")}
NAMES = tuple(BANDS)
PATCHES = {
    name: ForecasterPatch(
        action="propose_forecaster_patch",
        method="repricing_interval",
        window_days=60,
        width_scale=width,
    )
    for name, width in BANDS.items()
}
ANSWER = '{"band":"'
CHOOSING_END = date(2026, 9, 24)
GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MAX_COLUMNS = 32
MIN_LOCALITY = 0.97
FEATURES = ("all", "instrument_class", "gpu_model", "unchanged_fraction", "relative_q80")
EDITOR_COMMAND = ("claude", "-p", "--model", "sonnet", "--strict-mcp-config", "--tools", "")


class EditSpec(BaseModel):
    """What the editor may ask for: promote one band for one class of (origin, series)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str
    feature: Literal["all", "instrument_class", "gpu_model", "unchanged_fraction", "relative_q80"]
    op: Literal["==", "<", ">="]
    value: str | float
    promote: Literal["narrow", "medium", "wide"]

    def matches(self, features: dict[str, Any]) -> bool:
        if self.feature == "all":
            return True
        observed = features[self.feature]
        if self.op == "==":
            return str(observed) == str(self.value)
        return (float(observed) < float(self.value)) == (self.op == "<")


EDITOR_SYSTEM = (
    "You are the editor in a weight-space-programming loop. A small language model (the actor) "
    "chooses, for each GPU rental price series and forecast origin, how wide an 80% forecast band "
    "to use: narrow, medium or wide. You improve it by proposing ONE direct weight edit per "
    "reply: pick a class of (origin, series) decisions by one feature test and the band the "
    "actor should choose for that class. A deterministic kernel then writes the edit into the "
    "weights and a frozen evaluator accepts it only if forecasts improve on unseen origins. "
    "Lower pinball loss is better. Use the development evidence in the brief, do not repeat a "
    "proposal that already failed, and prefer the change with the largest expected gain. "
    "Numeric features change from origin to origin, so a threshold applies per decision. "
    "Answer only with the structured output."
)


def _rows() -> list[dict[str, Any]]:
    """Every (block, origin, series) decision with its cutoff-safe prompt and features."""
    targets = read_frozen_target_ids(REPO_ROOT / "research" / "adapted_series.txt")
    rows: list[dict[str, Any]] = []
    for block in ("development", "checkpoint", "protected"):
        for origin in _dates(*BLOCKS[block]):
            bundle = build_actor_input_bundle(
                source_series=REPO_ROOT / "research" / "out" / "series_daily.csv",
                bundle_root=ROOT / "actor_bundles",
                cutoff_date=origin,
                target_ids=targets,
            )
            for summary in summarize_actor_history(bundle, origin_date=origin):
                series_id = str(summary["series_id"])
                prompt = json.dumps(
                    {
                        "task": "Choose the width of the 80% forecast band for this series.",
                        "origin_date": origin.isoformat(),
                        "horizons_days": [1, 3, 7],
                        "series": summary,
                        "allowed_bands": list(NAMES),
                        "band_meaning": {
                            "narrow": "0.5x the typical recent daily move",
                            "medium": "1x the typical recent daily move",
                            "wide": "2x the typical recent daily move",
                        },
                        "output_contract": {"band": "narrow, medium or wide"},
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                rows.append(
                    {
                        "block": block,
                        "origin": origin,
                        "series_id": series_id,
                        "prompt": prompt,
                        "features": {
                            "instrument_class": series_id.split(".")[1],
                            "gpu_model": series_id.split(".")[2],
                            "unchanged_fraction": round(
                                float(str(summary["unchanged_fraction"])), 4
                            ),
                            "relative_q80": round(
                                float(str(summary["absolute_change_q80"]))
                                / float(str(summary["last_value"])),
                                5,
                            ),
                        },
                    }
                )
    return rows


class Kernel:
    """Base-model cache plus the closed-form algebra for last-layer edits."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        import torch

        self.torch = torch
        self.rows = rows
        self.tokenizer, self.model = wsp._load()
        self.weight = self.model.get_submodule(LAYER).weight
        self.base_weight = self.weight.clone()
        prefix = self.tokenizer(ANSWER, add_special_tokens=False)["input_ids"]
        self.band_ids = []
        for name in NAMES:
            ids = self.tokenizer(ANSWER + name, add_special_tokens=False)["input_ids"]
            if ids[: len(prefix)] != prefix:
                raise RuntimeError("band token does not follow the fixed answer prefix")
            self.band_ids.append(ids[len(prefix)])
        if len(set(self.band_ids)) != len(NAMES):
            raise RuntimeError("band first tokens are not distinct")
        head = self.model.get_output_embeddings().weight
        norm = wsp._final_norm(self.model)
        self.band_unembed = head[self.band_ids].cpu()
        self.norm_gain = norm(head.new_ones(1, 1, head.shape[1]))[0, 0].cpu()
        self._norm = norm

        digest = hashlib.sha256("\n".join(row["prompt"] for row in rows).encode()).hexdigest()
        cache_path = ROOT / "cache.pt"
        if cache_path.is_file() and torch.load(cache_path)["digest"] == digest:
            cache = torch.load(cache_path)
        else:
            resid, keys, gaps = [], [], []
            for row in rows:
                record = wsp._record(
                    self.tokenizer,
                    self.model,
                    row["prompt"],
                    (prefix, self.band_ids[0], self.band_ids[2]),
                )
                resid.append(record["resid"].cpu())
                keys.append(record[LAYER][0][-1].cpu())
                gaps.append(record["gap"])
            cache = {
                "digest": digest,
                "resid": torch.stack(resid),
                "keys": torch.stack(keys),
                "gaps": torch.tensor(gaps),
            }
            torch.save(cache, cache_path)
        self.resid, self.keys = cache["resid"], cache["keys"]
        logits = self.logits(self.resid)
        error = float((logits[:, 2] - logits[:, 0] - cache["gaps"]).abs().max())
        if error > 1e-2:
            raise RuntimeError(f"cached read-out disagrees with the model by {error}")
        self.readout_error = error
        self.unrelated = [
            wsp._encode(self.tokenizer, text, system=None) for text in wsp.UNRELATED_PROMPTS
        ]
        self.base_unrelated = [lp.argmax(-1) for lp in wsp._log_probs(self.model, self.unrelated)]

    def logits(self, resid: Any) -> Any:
        with self.torch.no_grad():
            normed = self._norm(resid.to(self.model.device)[None])[0].cpu()
        return normed @ self.band_unembed.T

    def resid_with(self, edits: list[dict[str, Any]]) -> Any:
        resid = self.resid.clone()
        for edit in edits:
            response = self.keys[:, edit["columns"]] @ self.torch.tensor(edit["row"])
            resid += edit["strength"] * response[:, None] * self.torch.tensor(edit["unit"])[None]
        return resid

    def choices(self, edits: list[dict[str, Any]]) -> list[int]:
        return [int(c) for c in self.logits(self.resid_with(edits)).argmax(dim=1)]

    def program(
        self, spec: EditSpec, accepted: list[dict[str, Any]], *, random_seed: int | None = None
    ) -> dict[str, Any]:
        """Closed-form conditional edit and its dose-response on the choosing origins."""
        torch = self.torch
        member = torch.tensor([spec.matches(row["features"]) for row in self.rows])
        dev = torch.tensor([row["block"] == "development" for row in self.rows])
        choosing = dev & torch.tensor([row["origin"] <= CHOOSING_END for row in self.rows])
        held_out = dev & ~choosing
        inside, outside = member & choosing, ~member & choosing
        if not bool(inside.any()):
            return {"status": "invalid", "reason": "no choosing decision matches the class"}
        resid = self.resid_with(accepted)
        logits = self.logits(resid)
        parent = logits.argmax(dim=1)
        target = NAMES.index(spec.promote)
        rest = [i for i in range(len(NAMES)) if i != target]
        if float((parent[inside] == target).float().mean()) >= 0.9:
            return {"status": "already_satisfied", "reason": "class already takes that band"}

        key_in = self.keys[inside].mean(dim=0)
        if bool(outside.any()):
            key_out = self.keys[outside].mean(dim=0)
            columns = (key_in - key_out).abs().topk(MAX_COLUMNS).indices.sort().values
            pair = torch.stack([key_in[columns], key_out[columns]])
            gram = pair @ pair.T + 1e-8 * torch.eye(2)
            row = pair.T @ torch.linalg.solve(gram, torch.tensor([1.0, 0.0]))
        else:
            columns = key_in.abs().topk(MAX_COLUMNS).indices.sort().values
            row = key_in[columns] / (key_in[columns] @ key_in[columns])
        rms = resid[inside].pow(2).mean(dim=1).sqrt().mean()
        direction = (
            (self.band_unembed[target] - self.band_unembed[rest].mean(dim=0)) * self.norm_gain / rms
        )
        unit = direction / direction.norm()
        if random_seed is not None:
            noise = torch.randn(unit.shape, generator=torch.Generator().manual_seed(random_seed))
            noise -= (noise @ unit) * unit
            unit = noise / noise.norm()
        margin = float((logits[inside, target] - logits[inside][:, rest].max(dim=1).values).mean())
        unit_strength = max(-margin, 0.5) / float(direction.norm())
        response = (self.keys[:, columns] @ row)[:, None] * unit[None]

        def rates(strength: float, mask_in: Any, mask_out: Any) -> tuple[float, float]:
            new = self.logits(resid + strength * response).argmax(dim=1)
            hit = float((new[mask_in] == target).float().mean()) if bool(mask_in.any()) else 0.0
            moved = (
                float((new[mask_out] != parent[mask_out]).float().mean())
                if bool(mask_out.any())
                else 0.0
            )
            return hit, moved

        dose, chosen = [], None
        for multiple in GRID:
            strength = multiple * unit_strength
            hit, moved = rates(strength, inside, outside)
            dose.append(
                {
                    "multiple": multiple,
                    "class_on_target": round(hit, 3),
                    "others_moved": round(moved, 3),
                }
            )
            if hit >= 0.9 and moved <= 0.1:
                chosen = strength
                break
        result: dict[str, Any] = {
            "status": "programmed" if chosen is not None else "not_programmable",
            "class_size_choosing": int(inside.sum()),
            "other_size_choosing": int(outside.sum()),
            "class_margin_before": round(margin, 4),
            "dose_response": dose,
        }
        if chosen is not None:
            hit, moved = rates(chosen, member & held_out, ~member & held_out)
            result["held_out"] = {"class_on_target": round(hit, 3), "others_moved": round(moved, 3)}
            result["edit"] = {
                "columns": columns.tolist(),
                "row": row.tolist(),
                "unit": unit.tolist(),
                "strength": chosen,
            }
        return result

    def verify_on_weights(self, edits: list[dict[str, Any]]) -> dict[str, Any]:
        """Write the stack into the real tensor; compare with the algebra; measure locality."""
        torch = self.torch
        analytic = self.logits(self.resid_with(edits))
        with torch.no_grad():
            for edit in edits:
                delta = edit["strength"] * torch.outer(
                    torch.tensor(edit["unit"]), torch.tensor(edit["row"])
                )
                self.weight[:, edit["columns"]] += delta.to(self.weight.device)
            edited_hash = hashlib.sha256(self.weight.cpu().numpy().tobytes()).hexdigest()
            changed = int((self.weight != self.base_weight).sum())
            relative = float((self.weight - self.base_weight).norm() / self.base_weight.norm())
            prefix = self.tokenizer(ANSWER, add_special_tokens=False)["input_ids"]
            worst = 0.0
            for index in range(0, len(self.rows), 29):
                record = wsp._record(
                    self.tokenizer,
                    self.model,
                    self.rows[index]["prompt"],
                    (prefix, self.band_ids[0], self.band_ids[2]),
                )
                expected = float(analytic[index, 2] - analytic[index, 0])
                worst = max(worst, abs(record["gap"] - expected))
            now = [lp.argmax(-1) for lp in wsp._log_probs(self.model, self.unrelated)]
            agreement = sum(
                float((a == b).float().mean())
                for a, b in zip(self.base_unrelated, now, strict=True)
            ) / len(now)
            self.weight.copy_(self.base_weight)
        return {
            "edited_tensor_sha256": edited_hash,
            "changed_weights": changed,
            "relative_change_frobenius": relative,
            "algebra_vs_real_weights_max_gap_error": worst,
            "unrelated_top1_agreement": agreement,
        }


def _score(
    kernel: Kernel,
    run_id: str,
    choices: list[int],
    block: str,
    parent_cells: Path | None = None,
    label: str = "parent",
) -> tuple[dict[str, object], Path, dict[str, object] | None]:
    table = {
        (row["origin"], row["series_id"]): PATCHES[NAMES[choice]]
        for row, choice in zip(kernel.rows, choices, strict=True)
        if row["block"] == block
    }
    forecast, bundle = _compile_block(
        block=block,
        patch=lambda origin, series_id: table[(origin, series_id)],
        run_id=run_id,
        actor_generation="loop",
    )
    report, cells = _score_block(block=block, forecast_path=forecast, bundle=bundle, run_id=run_id)
    paired = (
        _run_paired(
            block=block,
            candidate_cells=cells,
            run_id=run_id,
            parent_cells=parent_cells,
            label=label,
        )
        if parent_cells
        else None
    )
    return report, cells, paired


def _loss_by_series(cells: Path) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    with cells.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "scored":
                totals[row["series_id"]].append(float(row["loss"]))
    return {series: round(sum(v) / len(v), 6) for series, v in sorted(totals.items())}


def _mean(values: dict[str, float]) -> float:
    return round(sum(values.values()) / len(values), 6)


def _brief(
    kernel: Kernel,
    choices: list[int],
    actor_dev_cells: Path,
    band_dev: dict[str, dict[str, float]],
    notebook: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    actor = _loss_by_series(actor_dev_cells)
    drift = _loss_by_series(REPO_ROOT / "research" / "out" / "cells_development_drift.csv")
    series: dict[str, dict[str, Any]] = {}
    for row, choice in zip(kernel.rows, choices, strict=True):
        if row["block"] != "development":
            continue
        entry = series.setdefault(
            row["series_id"],
            {"choices": Counter(), "unchanged_fraction": [], "relative_q80": [], "row": row},
        )
        entry["choices"][NAMES[choice]] += 1
        entry["unchanged_fraction"].append(row["features"]["unchanged_fraction"])
        entry["relative_q80"].append(row["features"]["relative_q80"])
    return {
        "bands": {name: f"width scale {width}" for name, width in BANDS.items()},
        "development_block": (
            "30 daily origins, 2026-09-01..2026-09-30; pinball loss, lower is better"
        ),
        "series": [
            {
                "series_id": series_id,
                "instrument_class": entry["row"]["features"]["instrument_class"],
                "gpu_model": entry["row"]["features"]["gpu_model"],
                "unchanged_fraction_range": [
                    min(entry["unchanged_fraction"]),
                    max(entry["unchanged_fraction"]),
                ],
                "relative_q80_range": [min(entry["relative_q80"]), max(entry["relative_q80"])],
                "actor_current_choices": dict(entry["choices"]),
                "actor_current_pinball": actor[series_id],
                "pinball_if_always": {band: band_dev[band][series_id] for band in NAMES},
                "drift_baseline_pinball": drift[series_id],
            }
            for series_id, entry in sorted(series.items())
        ],
        "features_you_may_test": {
            "all": "every decision (op and value ignored)",
            "instrument_class": "== one of the instrument_class strings",
            "gpu_model": "== one of the gpu_model strings",
            "unchanged_fraction": "< or >= a number in [0, 1]; share of unchanged days in 60",
            "relative_q80": "< or >= a number; 80th-percentile daily move / price",
        },
        "notebook_of_past_cycles": notebook,
        "failed_attempts_this_cycle": attempts,
    }


def _scripted(brief: dict[str, Any], tried: set[str]) -> EditSpec:
    """Fallback editor: the instrument class and band with the largest development gain."""
    best: tuple[float, EditSpec] | None = None
    classes = sorted({entry["instrument_class"] for entry in brief["series"]})
    for value in classes:
        for band in NAMES:
            gain = sum(
                entry["actor_current_pinball"] - entry["pinball_if_always"][band]
                for entry in brief["series"]
                if entry["instrument_class"] == value
            )
            spec = EditSpec(
                hypothesis=f"scripted: {band} lowers development loss most for {value}",
                feature="instrument_class",
                op="==",
                value=value,
                promote=band,  # type: ignore[arg-type]
            )
            if _spec_key(spec) not in tried and (best is None or gain > best[0]):
                best = (gain, spec)
    if best is None:
        raise RuntimeError("scripted editor has nothing left to propose")
    return best[1]


def _spec_key(spec: EditSpec) -> str:
    return f"{spec.feature}{spec.op}{spec.value}->{spec.promote}"


def _ask_editor(brief: dict[str, Any], folder: Path, tried: set[str]) -> tuple[EditSpec, str]:
    """One headless call; the stored reply is reused on a re-run (replay)."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "brief.json").write_text(json.dumps(brief, indent=1) + "\n", encoding="utf-8")
    reply_path = folder / "editor_reply.json"
    if reply_path.is_file():
        stored = json.loads(reply_path.read_text(encoding="utf-8"))
        return EditSpec.model_validate(stored["spec"]), stored["editor"]
    schema = EditSpec.model_json_schema()
    schema["additionalProperties"] = False
    prompt = "Propose the next weight edit.\n\nBRIEF:\n" + json.dumps(brief)
    environment = {
        k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")
    }
    editor, spec, raw = "headless-claude-sonnet", None, ""
    for _ in range(2):
        try:
            completed = subprocess.run(
                [
                    *EDITOR_COMMAND,
                    "--output-format",
                    "json",
                    "--no-session-persistence",
                    "--system-prompt",
                    EDITOR_SYSTEM,
                    "--json-schema",
                    json.dumps(schema),
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
                cwd=ROOT,
                env=environment,
            )
            raw = completed.stdout
            spec = EditSpec.model_validate(json.loads(raw)["structured_output"])
            break
        except Exception as error:  # noqa: BLE001 - any editor failure falls back, recorded
            raw = f"{raw}\n[editor call failed: {type(error).__name__}: {error}]"
    if spec is None:
        editor, spec = "scripted-fallback", _scripted(brief, tried)
    (folder / "editor_raw.txt").write_text(raw, encoding="utf-8")
    reply_path.write_text(
        json.dumps({"editor": editor, "spec": spec.model_dump(mode="json")}, indent=1) + "\n",
        encoding="utf-8",
    )
    return spec, editor


def _gate_numbers(paired: dict[str, object]) -> dict[str, Any]:
    overall = paired["overall"]
    assert isinstance(overall, dict)
    return {
        "diff": overall["mean_diff"],
        "ci90": paired["bootstrap_90ci_mean_diff"],
        "coverage": overall["coverage_a_q10_q90"],
    }


def run(arm: str, cycles: int, editor_mode: str, score_checkpoint: bool) -> None:
    if not PREREGISTRATION.is_file():
        raise SystemExit("refusing to run without the pre-registration file")
    out = ROOT / arm
    out.mkdir(parents=True, exist_ok=True)
    kernel = Kernel(_rows())

    base_choices = kernel.choices([])
    _, base_dev, _ = _score(kernel, "loop-base", base_choices, "development")
    band_dev = {}
    for index, name in enumerate(NAMES):
        _, cells, _ = _score(
            kernel, f"loop-const-{name}", [index] * len(kernel.rows), "development"
        )
        band_dev[name] = _loss_by_series(cells)

    notebook: list[dict[str, Any]] = []
    if arm == "i1":
        notebook = json.loads((ROOT / "i0" / "notebook.json").read_text(encoding="utf-8"))
        notebook = [{**entry, "from": "earlier run of this editor (I_0)"} for entry in notebook]
    inherited = len(notebook)
    accepted: list[dict[str, Any]] = []
    parent_id, parent_dev = "loop-base", base_dev
    tried: set[str] = set()
    stop_reason = "cycles completed"

    for cycle in range(1, cycles + 1):
        attempts: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"cycle": cycle, "attempts": attempts}
        parent_before = parent_id
        for attempt in (1, 2):
            folder = out / f"cycle{cycle}" / f"attempt{attempt}"
            choices_now = kernel.choices(accepted)
            brief = _brief(kernel, choices_now, parent_dev, band_dev, notebook, attempts)
            if editor_mode == "scripted":
                spec, editor = _scripted(brief, tried), "scripted"
                folder.mkdir(parents=True, exist_ok=True)
            else:
                spec, editor = _ask_editor(brief, folder, tried)
            tried.add(_spec_key(spec))
            programmed = kernel.program(spec, accepted)
            record: dict[str, Any] = {
                "spec": spec.model_dump(mode="json"),
                "editor": editor,
                "programming": {k: v for k, v in programmed.items() if k != "edit"},
            }
            attempts.append(record)
            if programmed["status"] != "programmed":
                record["outcome"] = programmed["status"]
                continue
            edit = programmed["edit"]
            control = kernel.program(spec, accepted, random_seed=cycle * 10 + attempt)
            record["random_direction_control_dose_response"] = control["dose_response"]
            run_id = f"loop-{arm}-c{cycle}a{attempt}"
            candidate = kernel.choices([*accepted, edit])
            _, dev_cells, dev_paired = _score(kernel, run_id, candidate, "development", parent_dev)
            assert dev_paired is not None
            record["development_vs_parent"] = _gate_numbers(dev_paired)
            if float(str(record["development_vs_parent"]["diff"])) > 0:
                record["outcome"] = "rejected_on_development_screen"
                continue
            if not score_checkpoint:
                record["outcome"] = "smoke_test_stopped_before_checkpoint"
                break
            parent_ckpt = RUN_ROOT / parent_id / "evaluations" / "checkpoint" / "forecast_cells.csv"
            if not parent_ckpt.is_file():
                _score(kernel, parent_id, kernel.choices(accepted), "checkpoint")
            _, _, ckpt_paired = _score(kernel, run_id, candidate, "checkpoint", parent_ckpt)
            assert ckpt_paired is not None
            gate = apply_frozen_admission_gate(
                checkpoint=PairedGateMetrics.from_paired_payload(ckpt_paired),
                development=PairedGateMetrics.from_paired_payload(dev_paired),
            )
            record["checkpoint_vs_parent"] = {
                key: value for key, value in _gate_numbers(ckpt_paired).items() if key != "coverage"
            }
            record["outcome"] = gate.status.value
            if gate.status is AdmissionStatus.ACCEPTED:
                check = kernel.verify_on_weights([*accepted, edit])
                record["weights"] = check
                if check["unrelated_top1_agreement"] < MIN_LOCALITY:
                    record["outcome"] = "accepted_by_gate_but_fails_locality"
                    stop_reason = "locality below the pre-registered floor"
                elif check["algebra_vs_real_weights_max_gap_error"] > 1e-2:
                    record["outcome"] = "accepted_by_gate_but_weights_disagree"
                    stop_reason = "real weights disagree with the algebra"
                else:
                    accepted.append(edit)
                    parent_id, parent_dev = run_id, dev_cells
            break
        entry["outcome"] = attempts[-1]["outcome"]
        notebook.append(entry)
        lineage = {
            "id": f"F_arch3_loop-{arm}-c{cycle}",
            "generation": cycle,
            "parent": parent_before,
            "kind": "forecaster",
            "status": entry["outcome"],
            "improver": f"I_wsp{arm[-1]}",
            "method": "band actor with stacked last-MLP edits (wsp_loop)",
            "attempts": attempts,
            "accepted_edits_so_far": len(accepted),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "created_utc": datetime.now(UTC).isoformat(),
        }
        lineage_root = out if arm == "smoke" else REPO_ROOT / "research" / "lineage"
        (lineage_root / f"F_arch3_loop-{arm}-c{cycle}.json").write_text(
            json.dumps(lineage, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"cycle": cycle, "outcome": entry["outcome"]}))
        if stop_reason != "cycles completed":
            break

    (out / "notebook.json").write_text(
        json.dumps(notebook[inherited:], indent=1) + "\n", encoding="utf-8"
    )
    head = {
        "arm": arm,
        "head_run_id": parent_id,
        "accepted_edits": accepted,
        "stop_reason": stop_reason,
        "readout_error": kernel.readout_error,
        "base_choice_counts": dict(Counter(NAMES[c] for c in base_choices)),
        "head_choice_counts": dict(Counter(NAMES[c] for c in kernel.choices(accepted))),
    }
    (out / "head.json").write_text(json.dumps(head, indent=1) + "\n", encoding="utf-8")
    improver = {
        "id": f"I_wsp{arm[-1]}",
        "kind": "improver",
        "status": "control" if arm == "i0" else "candidate",
        "method": "headless LLM editor proposing closed-form weight edits; "
        + ("empty notebook at start" if arm == "i0" else "starts with I_wsp0's notebook"),
        "cycles": len(notebook) - inherited,
        "accepted_edits": len(accepted),
        "head_run_id": parent_id,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    if arm != "smoke":
        (REPO_ROOT / "research" / "lineage" / f"I_wsp{arm[-1]}.json").write_text(
            json.dumps(improver, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({k: v for k, v in head.items() if k != "accepted_edits"}, indent=1))


def protected() -> None:
    """The recursion test: each arm's final actor, scored once on the protected block."""
    kernel = Kernel(_rows())
    heads = {
        arm: json.loads((ROOT / arm / "head.json").read_text(encoding="utf-8"))
        for arm in ("i0", "i1")
    }
    drift = REPO_ROOT / "research" / "out" / "cells_protected_drift.csv"
    result: dict[str, Any] = {
        "same_edits": heads["i0"]["accepted_edits"] == heads["i1"]["accepted_edits"]
    }
    cells: dict[str, Path] = {}
    for arm, head in heads.items():
        run_id = f"loop-{arm}-head"
        _, cells[arm], paired = _score(
            kernel, run_id, kernel.choices(head["accepted_edits"]), "protected", drift, "F0_drift"
        )
        assert paired is not None
        result[f"{arm}_head_vs_F0_drift"] = _gate_numbers(paired)
    paired = _run_paired(
        block="protected",
        candidate_cells=cells["i1"],
        run_id="loop-i1-head",
        parent_cells=cells["i0"],
        label="I0_head",
    )
    numbers = _gate_numbers(paired)
    result["i1_head_vs_i0_head"] = numbers
    result["recursion_claim_supported"] = bool(
        float(str(numbers["diff"])) < 0 and float(str(numbers["ci90"][1])) < 0
    )
    (ROOT / "protected_result.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--arm", choices=("i0", "i1", "smoke"), required=True)
    runner.add_argument("--cycles", type=int, default=3)
    runner.add_argument("--editor", choices=("claude", "scripted"), default="claude")
    runner.add_argument("--no-checkpoint", action="store_true")
    commands.add_parser("protected")
    args = parser.parse_args()
    if args.command == "run":
        run(args.arm, args.cycles, args.editor, not args.no_checkpoint)
    else:
        protected()


if __name__ == "__main__":
    main()
