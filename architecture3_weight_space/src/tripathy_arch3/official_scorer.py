"""Evaluator-only adapter around the unchanged Track 2 official scorer."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from .audit import sha256_file
from .data_boundary import ACTOR_SERIES_FILENAME, ActorInputBundle


class OfficialScorerError(RuntimeError):
    """Raised when the scorer cannot execute or produces an invalid artifact."""


_EVALUATOR_CAPABILITY = object()


@dataclass(frozen=True)
class ProtectedTruth:
    """Opaque truth handle; actor interfaces receive only ActorInputBundle."""

    _path: Path
    sha256: str

    @classmethod
    def from_path(cls, path: Path) -> ProtectedTruth:
        if path.is_symlink():
            raise OfficialScorerError("protected truth may not be a symlink")
        resolved = path.resolve()
        if not resolved.is_file():
            raise OfficialScorerError("protected truth must be a regular, non-symlink file")
        return cls(_path=resolved, sha256=sha256_file(resolved))

    def _path_for_evaluator(self, capability: object) -> Path:
        if capability is not _EVALUATOR_CAPABILITY:
            raise PermissionError("protected truth is available only to the evaluator adapter")
        if sha256_file(self._path) != self.sha256:
            raise OfficialScorerError("protected truth changed after its handle was created")
        return self._path


@dataclass(frozen=True)
class OfficialScoreResult:
    status: str
    return_code: int
    report_path: Path
    cells_path: Path | None
    receipt_path: Path
    stdout: str
    stderr: str


class OfficialScorerAdapter:
    """Execute the pinned scorer without exposing truth to the actor bundle."""

    def __init__(
        self,
        *,
        scorer_path: Path,
        python_executable: Path,
        expected_scorer_sha256: str | None = None,
    ) -> None:
        if scorer_path.is_symlink():
            raise OfficialScorerError("official scorer may not be a symlink")
        self.scorer_path = scorer_path.resolve()
        self.python_executable = Path(os.path.abspath(python_executable))
        if not self.scorer_path.is_file():
            raise OfficialScorerError("official scorer is missing or is a symlink")
        if not self.python_executable.is_file():
            raise OfficialScorerError("scorer Python executable is missing")
        self.scorer_sha256 = sha256_file(self.scorer_path)
        if expected_scorer_sha256 and self.scorer_sha256 != expected_scorer_sha256:
            raise OfficialScorerError("official scorer hash differs from the pinned hash")

    def score(
        self,
        *,
        forecast_path: Path,
        history_bundle: ActorInputBundle,
        protected_truth: ProtectedTruth,
        output_directory: Path,
        mode: Literal["backtest", "frozen_cutoff"],
        horizons: tuple[int, ...] = (1, 3, 7),
        frozen_cutoff: date | None = None,
        timeout_seconds: int = 300,
    ) -> OfficialScoreResult:
        if forecast_path.is_symlink():
            raise OfficialScorerError("forecast may not be a symlink")
        forecast_path = forecast_path.resolve()
        if not forecast_path.is_file():
            raise OfficialScorerError("forecast must be a regular, non-symlink file")
        if not horizons or tuple(sorted(set(horizons))) != horizons:
            raise OfficialScorerError("horizons must be non-empty, unique, and sorted")
        if mode == "frozen_cutoff" and frozen_cutoff is None:
            raise OfficialScorerError("frozen_cutoff mode requires a cutoff date")
        if mode == "backtest" and frozen_cutoff is not None:
            raise OfficialScorerError("backtest mode may not set frozen_cutoff")
        if frozen_cutoff and history_bundle.manifest.cutoff_date != frozen_cutoff:
            raise OfficialScorerError("history bundle cutoff must match frozen_cutoff")

        truth_path = protected_truth._path_for_evaluator(_EVALUATOR_CAPABILITY)
        history_path = history_bundle.resolve_artifact(ACTOR_SERIES_FILENAME)
        if truth_path.is_relative_to(history_bundle.root):
            raise OfficialScorerError("protected truth may not be mounted inside the actor bundle")

        output_directory = output_directory.resolve()
        if output_directory.is_relative_to(history_bundle.root):
            raise OfficialScorerError(
                "evaluator outputs may not be written inside the actor bundle"
            )
        output_directory.mkdir(parents=True, exist_ok=False)
        report_path = output_directory / "official_report.json"
        cells_path = output_directory / "forecast_cells.csv"
        targets_path = output_directory / "targets.txt"
        targets_path.write_text(
            "\n".join(history_bundle.manifest.target_ids) + "\n", encoding="utf-8"
        )

        command = [
            str(self.python_executable),
            "-I",
            "-B",
            str(self.scorer_path),
            "--forecast",
            str(forecast_path),
            "--truth",
            str(truth_path),
            "--history",
            str(history_path),
            "--horizons",
            ",".join(str(horizon) for horizon in horizons),
            "--series-ids",
            f"@{targets_path}",
            "--out",
            str(report_path),
            "--per-cell",
            str(cells_path),
        ]
        if frozen_cutoff:
            command.extend(["--frozen-cutoff", frozen_cutoff.isoformat()])

        started_at = datetime.now(UTC)
        completed = subprocess.run(
            command,
            cwd=output_directory,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.defpath,
                "PYTHONHASHSEED": "0",
                "TZ": "UTC",
            },
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        finished_at = datetime.now(UTC)
        if completed.returncode not in {0, 2}:
            raise OfficialScorerError(
                f"official scorer failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        if not report_path.is_file():
            raise OfficialScorerError("official scorer did not write its report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = str(report.get("status", "missing"))
        if completed.returncode == 0 and status not in {"scored", "scored_no_cells"}:
            raise OfficialScorerError(f"unexpected successful scorer status: {status}")
        if completed.returncode == 2 and status != "refused":
            raise OfficialScorerError(f"unexpected refusal scorer status: {status}")

        receipt = {
            "schema_version": 1,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "mode": mode,
            "frozen_cutoff": frozen_cutoff.isoformat() if frozen_cutoff else None,
            "horizons": horizons,
            "return_code": completed.returncode,
            "status": status,
            "scorer": {
                "name": self.scorer_path.name,
                "sha256": self.scorer_sha256,
            },
            "inputs": {
                "forecast_sha256": sha256_file(forecast_path),
                "history_bundle_id": history_bundle.manifest.bundle_id,
                "history_sha256": sha256_file(history_path),
                "truth_sha256": protected_truth.sha256,
            },
            "outputs": {
                "report_sha256": sha256_file(report_path),
                "cells_sha256": sha256_file(cells_path) if cells_path.is_file() else None,
            },
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        receipt_path = output_directory / "score_receipt.json"
        with receipt_path.open("w", encoding="utf-8") as receipt_handle:
            json.dump(receipt, receipt_handle, indent=2, sort_keys=True)
            receipt_handle.write("\n")
            receipt_handle.flush()
            os.fsync(receipt_handle.fileno())
        return OfficialScoreResult(
            status=status,
            return_code=completed.returncode,
            report_path=report_path,
            cells_path=cells_path if cells_path.is_file() else None,
            receipt_path=receipt_path,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
