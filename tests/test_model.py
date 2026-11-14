import pytest

from tripathy_experiment.model import ExperimentManifest, ValidationError


def test_placeholder_manifest_is_blocked() -> None:
    manifest = ExperimentManifest.from_dict({"experiment_id": "01", "budget_usd": 0})
    with pytest.raises(ValidationError, match="Start gate incomplete"):
        manifest.validate()


def test_complete_manifest_passes() -> None:
    manifest = ExperimentManifest.from_dict(
        {
            "experiment_id": "01",
            "status": "planning",
            "question": "Can the candidate beat the baseline?",
            "hypothesis": "The candidate improves the primary metric.",
            "primary_metric": "score",
            "baseline": "fixed baseline",
            "budget_usd": 10,
            "stop_condition": "Stop after ten runs.",
        }
    )
    manifest.validate()
