"""Content hashing and append-only, hash-chained JSONL event journals."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from .contracts import EventEnvelope, Sha256


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashing and line-oriented storage."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class JournalIntegrityError(RuntimeError):
    """Raised when a journal is malformed, duplicated, or hash-chain-invalid."""


class JournalRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=0)
    previous_hash: Sha256 | None
    event: EventEnvelope
    record_hash: Sha256

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        previous_hash: str | None,
        event: EventEnvelope,
    ) -> Self:
        content = {
            "event": event.model_dump(mode="json"),
            "previous_hash": previous_hash,
            "sequence": sequence,
        }
        return cls(
            sequence=sequence,
            previous_hash=previous_hash,
            event=event,
            record_hash=sha256_bytes(canonical_json(content).encode("utf-8")),
        )

    def verify_hash(self) -> None:
        expected = JournalRecord.create(
            sequence=self.sequence,
            previous_hash=self.previous_hash,
            event=self.event,
        ).record_hash
        if self.record_hash != expected:
            raise JournalIntegrityError(f"record {self.sequence} hash mismatch")


class AppendOnlyEventJournal:
    """One append-only JSONL shard per worker with a local hash chain."""

    def __init__(self, run_directory: Path, worker_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", worker_id):
            raise ValueError("worker_id must be filesystem-safe")
        self.path = run_directory / "events" / f"{worker_id}.jsonl"

    @staticmethod
    def _parse_lines(lines: list[str]) -> list[JournalRecord]:
        records: list[JournalRecord] = []
        event_ids: set[str] = set()
        previous_hash: str | None = None
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise JournalIntegrityError(f"blank line at journal line {line_number}")
            try:
                record = JournalRecord.model_validate_json(line)
            except ValueError as error:
                raise JournalIntegrityError(
                    f"invalid record at journal line {line_number}: {error}"
                ) from error
            if record.sequence != len(records):
                raise JournalIntegrityError(f"unexpected sequence at journal line {line_number}")
            if record.previous_hash != previous_hash:
                raise JournalIntegrityError(f"broken hash chain at journal line {line_number}")
            record.verify_hash()
            event_id = str(record.event.event_id)
            if event_id in event_ids:
                raise JournalIntegrityError(f"duplicate event_id at journal line {line_number}")
            event_ids.add(event_id)
            previous_hash = record.record_hash
            records.append(record)
        return records

    def read_records(self) -> list[JournalRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return self._parse_lines(handle.read().splitlines())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append(self, event: EventEnvelope) -> JournalRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                records = self._parse_lines(handle.read().splitlines())
                if any(record.event.event_id == event.event_id for record in records):
                    raise JournalIntegrityError(f"duplicate event_id: {event.event_id}")
                previous_hash = records[-1].record_hash if records else None
                record = JournalRecord.create(
                    sequence=len(records),
                    previous_hash=previous_hash,
                    event=event,
                )
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(record.model_dump(mode="json")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return record
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
