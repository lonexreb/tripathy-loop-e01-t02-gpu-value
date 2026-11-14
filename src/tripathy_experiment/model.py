from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ValidationError(ValueError):
    """Raised when an experiment has not passed its start gate."""


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    status: str
    question: str
    hypothesis: str
    primary_metric: str
    baseline: str
    budget_usd: float
    stop_condition: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentManifest:
        return cls(
            experiment_id=str(payload.get("experiment_id", "")),
            status=str(payload.get("status", "")),
            question=str(payload.get("question", "")),
            hypothesis=str(payload.get("hypothesis", "")),
            primary_metric=str(payload.get("primary_metric", "")),
            baseline=str(payload.get("baseline", "")),
            budget_usd=float(payload.get("budget_usd", 0)),
            stop_condition=str(payload.get("stop_condition", "")),
        )

    def validate(self) -> None:
        missing = [
            field
            for field in ("question", "hypothesis", "primary_metric", "baseline", "stop_condition")
            if not getattr(self, field)
        ]
        if missing:
            raise ValidationError(f"Start gate incomplete: {', '.join(missing)}")
        if self.budget_usd <= 0:
            raise ValidationError("Start gate incomplete: budget_usd must be positive")
