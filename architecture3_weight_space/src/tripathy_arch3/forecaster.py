"""Deterministic compiler from bounded actor patches to forecast commitments."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .audit import canonical_json, sha256_bytes, sha256_file
from .contracts import ForecastRow, FrozenContract
from .data_boundary import ACTOR_SERIES_FILENAME, ActorInputBundle


class ForecasterPatch(FrozenContract):
    action: Literal["propose_forecaster_patch"]
    method: Literal["drift", "repricing_interval"]
    window_days: Literal[30, 60, 90]
    width_scale: Annotated[Decimal, Field(ge=Decimal("0.5"), le=Decimal("2"))]

    @model_validator(mode="after")
    def validate_drift_parameters(self) -> Self:
        if self.method == "drift" and (self.window_days != 60 or self.width_scale != Decimal("1")):
            raise ValueError("drift must use the canonical window 60 and width scale 1")
        return self


def parse_actor_patch(output: str) -> ForecasterPatch:
    """Extract one JSON object and reject prose after the object."""
    stripped = output.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("actor output does not contain a JSON object")
    prefix = stripped[:start].strip()
    suffix = stripped[end + 1 :].strip()
    if prefix or suffix:
        raise ValueError("actor output must contain only one JSON object")
    payload = json.loads(stripped[start : end + 1])
    return ForecasterPatch.model_validate(payload)


def _read_history(bundle: ActorInputBundle) -> dict[str, list[tuple[date, Decimal]]]:
    history: dict[str, list[tuple[date, Decimal]]] = defaultdict(list)
    path = bundle.resolve_artifact(ACTOR_SERIES_FILENAME)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            history[row["series_id"]].append(
                (date.fromisoformat(row["date"]), Decimal(row["value"]))
            )
    return dict(history)


def _quantile(values: list[Decimal], probability: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    ordered = sorted(values)
    position = int((Decimal(len(ordered) - 1) * probability).to_integral_value())
    return ordered[position]


def summarize_actor_history(
    bundle: ActorInputBundle, *, origin_date: date
) -> list[dict[str, object]]:
    if origin_date > bundle.manifest.cutoff_date:
        raise ValueError("origin_date exceeds actor bundle cutoff")
    summaries: list[dict[str, object]] = []
    for series_id, all_rows in sorted(_read_history(bundle).items()):
        rows = [(day, value) for day, value in all_rows if day <= origin_date]
        if not rows:
            raise ValueError(f"series has no history at origin: {series_id}")
        trailing = rows[-60:]
        deltas = [
            abs(right[1] - left[1]) for left, right in zip(trailing, trailing[1:], strict=False)
        ]
        nonzero = sum(delta > 0 for delta in deltas)
        summaries.append(
            {
                "series_id": series_id,
                "last_date": trailing[-1][0].isoformat(),
                "last_value": str(trailing[-1][1]),
                "trailing_observations": len(trailing),
                "nonzero_changes": nonzero,
                "unchanged_fraction": (
                    str(Decimal(len(deltas) - nonzero) / Decimal(len(deltas))) if deltas else "1"
                ),
                "absolute_change_q80": str(_quantile(deltas, Decimal("0.8"))),
            }
        )
    return summaries


def build_actor_prompt(bundle: ActorInputBundle, *, origin_date: date) -> str:
    episode = {
        "task": "Choose one frozen forecasting patch before any truth is released.",
        "origin_date": origin_date.isoformat(),
        "information_cutoff": bundle.manifest.cutoff_date.isoformat(),
        "horizons": [1, 3, 7],
        "allowed_actions": [
            {
                "action": "propose_forecaster_patch",
                "method": "drift",
                "window_days": 60,
                "width_scale": 1,
            },
            {
                "action": "propose_forecaster_patch",
                "method": "repricing_interval",
                "window_days": "one of 30, 60, 90",
                "width_scale": "number from 0.5 through 2",
            },
        ],
        "series_summaries": summarize_actor_history(bundle, origin_date=origin_date),
        "output_contract": {
            "action": "propose_forecaster_patch",
            "method": "drift or repricing_interval",
            "window_days": "30, 60, or 90",
            "width_scale": "0.5 to 2",
        },
    }
    return canonical_json(episode)


def compile_forecast(
    bundle: ActorInputBundle,
    *,
    patch: ForecasterPatch,
    origin_date: date,
    horizons: tuple[int, ...] = (1, 3, 7),
    actor_generation: Literal["g0", "g1", "wsp1"] = "g0",
) -> tuple[ForecastRow, ...]:
    if origin_date > bundle.manifest.cutoff_date:
        raise ValueError("origin_date exceeds actor bundle cutoff")
    rows: list[ForecastRow] = []
    for series_id, all_rows in sorted(_read_history(bundle).items()):
        history = [(day, value) for day, value in all_rows if day <= origin_date]
        if not history:
            raise ValueError(f"series has no eligible history: {series_id}")
        last_value = history[-1][1]
        trailing = history[-patch.window_days :]
        deltas = [
            abs(right[1] - left[1]) for left, right in zip(trailing, trailing[1:], strict=False)
        ]
        empirical_width = _quantile(deltas, Decimal("0.8"))
        floor_width = last_value * Decimal("0.01")
        for horizon in horizons:
            if patch.method == "drift":
                width = Decimal("0")
            else:
                width = (
                    max(empirical_width, floor_width) * patch.width_scale * Decimal(horizon).sqrt()
                )
            rows.append(
                ForecastRow(
                    series_id=series_id,
                    origin_date=origin_date,
                    target_date=origin_date.fromordinal(origin_date.toordinal() + horizon),
                    horizon_days=horizon,
                    q10=max(Decimal("0.000001"), last_value - width),
                    q50=last_value,
                    q90=last_value + width,
                    method_id=(
                        f"actor-{actor_generation}-drift"
                        if patch.method == "drift"
                        else (
                            f"actor-{actor_generation}-repricing-"
                            f"w{patch.window_days}-s{patch.width_scale}"
                        )
                    ),
                )
            )
    return tuple(rows)


def forecast_identity(
    *, bundle: ActorInputBundle, patch: ForecasterPatch, rows: tuple[ForecastRow, ...]
) -> str:
    payload = {
        "bundle_digest": bundle.manifest.bundle_digest,
        "compiler_sha256": sha256_file(Path(__file__)),
        "patch": patch.model_dump(mode="json"),
        "rows": [row.model_dump(mode="json") for row in rows],
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def write_forecast_csv(path: Path, rows: tuple[ForecastRow, ...]) -> None:
    if not rows:
        raise ValueError("forecast rows may not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "series_id",
                "origin_date",
                "horizon_days",
                "q10",
                "q50",
                "q90",
                "target_date",
                "method",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "series_id": row.series_id,
                    "origin_date": row.origin_date.isoformat(),
                    "horizon_days": row.horizon_days,
                    "q10": str(row.q10),
                    "q50": str(row.q50),
                    "q90": str(row.q90),
                    "target_date": row.target_date.isoformat(),
                    "method": row.method_id,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
