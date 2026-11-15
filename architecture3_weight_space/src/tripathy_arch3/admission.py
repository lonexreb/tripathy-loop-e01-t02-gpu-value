"""Pure implementation of the frozen checkpoint admission rule."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Self

from pydantic import Field, model_validator

from .contracts import AdmissionStatus, FrozenContract

FiniteDecimal = Annotated[Decimal, Field(allow_inf_nan=False)]


class PairedGateMetrics(FrozenContract):
    """The exact paired statistics consumed by the admission gate."""

    mean_diff: FiniteDecimal
    ci_lower: FiniteDecimal
    ci_upper: FiniteDecimal
    ci_excludes_zero: bool

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.ci_lower > self.ci_upper:
            raise ValueError("paired confidence interval bounds are reversed")
        excludes_zero = self.ci_upper < 0 or self.ci_lower > 0
        if excludes_zero != self.ci_excludes_zero:
            raise ValueError("ci_excludes_zero disagrees with the interval bounds")
        return self

    @classmethod
    def from_paired_payload(cls, payload: dict[str, object]) -> Self:
        overall = payload.get("overall")
        interval = payload.get("bootstrap_90ci_mean_diff")
        if not isinstance(overall, dict) or not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("paired result is missing its overall statistic or 90% interval")
        return cls(
            mean_diff=Decimal(str(overall["mean_diff"])),
            ci_lower=Decimal(str(interval[0])),
            ci_upper=Decimal(str(interval[1])),
            ci_excludes_zero=bool(payload.get("ci_excludes_zero")),
        )


class GateOutcome(FrozenContract):
    status: AdmissionStatus
    reason: str
    checkpoint: PairedGateMetrics
    development: PairedGateMetrics


def apply_frozen_admission_gate(
    *, checkpoint: PairedGateMetrics, development: PairedGateMetrics
) -> GateOutcome:
    """Apply PROTOCOL.md section 4 without fallbacks or discretionary thresholds."""
    checkpoint_passes = (
        checkpoint.mean_diff < 0
        and checkpoint.ci_excludes_zero
        and checkpoint.ci_upper < 0
    )
    development_passes = development.mean_diff <= 0
    if checkpoint_passes and development_passes:
        status = AdmissionStatus.ACCEPTED
        reason = (
            "checkpoint paired mean difference is negative with the 90% CI entirely below "
            "zero, and the development paired mean difference is non-positive"
        )
    else:
        failures: list[str] = []
        if not checkpoint_passes:
            failures.append(
                "checkpoint paired difference is not negative with its 90% CI entirely "
                "below zero"
            )
        if not development_passes:
            failures.append("development paired mean difference is positive")
        status = AdmissionStatus.REJECTED
        reason = "; ".join(failures)
    return GateOutcome(
        status=status,
        reason=reason,
        checkpoint=checkpoint,
        development=development,
    )
