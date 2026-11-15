from decimal import Decimal

import pytest

from tripathy_arch3.admission import (
    PairedGateMetrics,
    apply_frozen_admission_gate,
)
from tripathy_arch3.contracts import AdmissionStatus


def metrics(mean: str, lower: str, upper: str) -> PairedGateMetrics:
    return PairedGateMetrics(
        mean_diff=Decimal(mean),
        ci_lower=Decimal(lower),
        ci_upper=Decimal(upper),
        ci_excludes_zero=Decimal(upper) < 0 or Decimal(lower) > 0,
    )


def test_frozen_gate_requires_both_checkpoint_and_development_conditions() -> None:
    accepted = apply_frozen_admission_gate(
        checkpoint=metrics("-0.02", "-0.03", "-0.01"),
        development=metrics("0", "-0.01", "0.01"),
    )
    assert accepted.status is AdmissionStatus.ACCEPTED

    checkpoint_rejection = apply_frozen_admission_gate(
        checkpoint=metrics("-0.01", "-0.02", "0.001"),
        development=metrics("-0.01", "-0.02", "-0.001"),
    )
    assert checkpoint_rejection.status is AdmissionStatus.REJECTED

    development_rejection = apply_frozen_admission_gate(
        checkpoint=metrics("-0.02", "-0.03", "-0.01"),
        development=metrics("0.000001", "-0.01", "0.01"),
    )
    assert development_rejection.status is AdmissionStatus.REJECTED


def test_paired_metrics_reject_inconsistent_interval_flag() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        PairedGateMetrics(
            mean_diff=Decimal("-0.01"),
            ci_lower=Decimal("-0.02"),
            ci_upper=Decimal("-0.001"),
            ci_excludes_zero=False,
        )
