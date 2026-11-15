import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from tripathy_arch3.audit import AppendOnlyEventJournal, JournalIntegrityError
from tripathy_arch3.contracts import CompletionStatus, EventEnvelope

SHA = "b" * 64
NOW = datetime(2026, 11, 15, 12, tzinfo=UTC)


def event(event_number: int) -> EventEnvelope:
    return EventEnvelope(
        event_id=UUID(int=event_number),
        event_type="run.started" if event_number == 1 else "run.completed",
        run_id="run-1",
        arm="frozen_actor",
        seed=0,
        generation=0,
        trace_id=UUID(int=100),
        created_at=NOW,
        git_sha="a" * 40,
        protocol_sha256=SHA,
        status=CompletionStatus.STARTED if event_number == 1 else CompletionStatus.COMPLETED,
    )


def test_journal_appends_and_verifies_hash_chain(tmp_path: Path) -> None:
    journal = AppendOnlyEventJournal(tmp_path / "run-1", "worker-0")
    first = journal.append(event(1))
    second = journal.append(event(2))
    records = journal.read_records()

    assert [record.sequence for record in records] == [0, 1]
    assert first.previous_hash is None
    assert second.previous_hash == first.record_hash

    with pytest.raises(JournalIntegrityError, match="duplicate event_id"):
        journal.append(event(2))


def test_journal_detects_tampering(tmp_path: Path) -> None:
    journal = AppendOnlyEventJournal(tmp_path / "run-1", "worker-0")
    journal.append(event(1))
    serialized = json.loads(journal.path.read_text(encoding="utf-8"))
    serialized["event"]["event_type"] = "tampered"
    journal.path.write_text(json.dumps(serialized) + "\n", encoding="utf-8")

    with pytest.raises(JournalIntegrityError, match="hash mismatch"):
        journal.read_records()


def test_worker_id_must_be_filesystem_safe(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filesystem-safe"):
        AppendOnlyEventJournal(tmp_path, "../other-worker")
