from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import uuid4

from .model import ExperimentManifest


@dataclass(frozen=True)
class RunResult:
    run_id: str
    experiment_id: str
    status: str
    started_at: str
    dry_run: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ExperimentRunner:
    """Provider-neutral orchestration boundary for one experiment run."""

    def run(self, manifest: ExperimentManifest, *, dry_run: bool) -> RunResult:
        manifest.validate()
        return RunResult(
            run_id=str(uuid4()),
            experiment_id=manifest.experiment_id,
            status="validated" if dry_run else "completed",
            started_at=datetime.now(UTC).isoformat(),
            dry_run=dry_run,
        )
