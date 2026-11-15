"""Durable, fail-closed accounting for authorized remote operations."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypedDict, cast

from .audit import canonical_json, sha256_bytes, sha256_file
from .config import WORKSPACE_ROOT

AUTHORIZATION_PATH = WORKSPACE_ROOT / "config" / "execution_authorization.json"
DEFAULT_LEDGER_PATH = WORKSPACE_ROOT / "artifacts" / "execution" / "ledger.json"

RemoteOperation = Literal[
    "exa_search",
    "past_write",
    "past_recall",
    "openai_teacher",
    "modal_function",
]

COUNTER_KEYS: dict[RemoteOperation, str] = {
    "exa_search": "maximum_exa_search_calls",
    "past_write": "maximum_past_write_calls",
    "past_recall": "maximum_past_recall_calls",
    "openai_teacher": "maximum_openai_teacher_calls",
    "modal_function": "maximum_modal_function_calls",
}


class BudgetExceededError(RuntimeError):
    """Raised before a remote call that would exceed an authorized ceiling."""


class LedgerEvent(TypedDict):
    call_id: str
    estimated_usd: str
    operation: RemoteOperation
    reserved_at: str


def _write_atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


class RemoteBudgetLedger:
    """Reserve call and dollar allowances before performing remote operations."""

    def __init__(
        self,
        ledger_path: Path = DEFAULT_LEDGER_PATH,
        authorization_path: Path = AUTHORIZATION_PATH,
    ) -> None:
        self.ledger_path = ledger_path
        self.authorization_path = authorization_path
        self.lock_path = ledger_path.with_suffix(".lock")

    def _authorization(self) -> dict[str, object]:
        authorization = cast(
            dict[str, object],
            json.loads(self.authorization_path.read_text(encoding="utf-8")),
        )
        if authorization.get("authorized") is not True:
            raise BudgetExceededError("remote execution is not authorized")
        return authorization

    def _empty_ledger(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "authorization_sha256": sha256_file(self.authorization_path),
            "estimated_total_usd": "0",
            "gpu_seconds_reserved": 0,
            "gpu_reservations": [],
            "counters": {operation: 0 for operation in COUNTER_KEYS},
            "events": [],
        }

    def read(self) -> dict[str, object]:
        if not self.ledger_path.exists():
            return self._empty_ledger()
        return cast(
            dict[str, object],
            json.loads(self.ledger_path.read_text(encoding="utf-8")),
        )

    def reserve(
        self,
        operation: RemoteOperation,
        *,
        call_id: str,
        estimated_usd: Decimal,
    ) -> dict[str, object]:
        if estimated_usd < 0:
            raise ValueError("estimated_usd may not be negative")
        authorization = self._authorization()
        expected_authorization_hash = sha256_file(self.authorization_path)
        budget = authorization["budget"]
        if not isinstance(budget, dict):
            raise BudgetExceededError("authorization budget is malformed")

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                ledger = self.read()
                if ledger.get("authorization_sha256") != expected_authorization_hash:
                    raise BudgetExceededError("authorization changed after the ledger was created")
                events = ledger.get("events")
                counters = ledger.get("counters")
                if not isinstance(events, list) or not isinstance(counters, dict):
                    raise BudgetExceededError("execution ledger is malformed")
                if any(
                    isinstance(event, dict) and event.get("call_id") == call_id for event in events
                ):
                    return ledger

                counter_key = COUNTER_KEYS[operation]
                current_count = int(counters.get(operation, 0))
                maximum_count = int(budget[counter_key])
                if current_count + 1 > maximum_count:
                    raise BudgetExceededError(f"{operation} call ceiling would be exceeded")

                current_total = Decimal(str(ledger.get("estimated_total_usd", "0")))
                next_total = current_total + estimated_usd
                maximum_total = Decimal(str(budget["maximum_total_usd"]))
                if next_total > maximum_total:
                    raise BudgetExceededError("authorized total dollar budget would be exceeded")

                event: LedgerEvent = {
                    "call_id": call_id,
                    "estimated_usd": str(estimated_usd),
                    "operation": operation,
                    "reserved_at": datetime.now(UTC).isoformat(),
                }
                events.append(event)
                counters[operation] = current_count + 1
                ledger["estimated_total_usd"] = str(next_total)
                ledger.pop("canonical_sha256", None)
                ledger_without_digest = {
                    key: value for key, value in ledger.items() if key != "ledger_sha256"
                }
                ledger["ledger_sha256"] = sha256_bytes(
                    canonical_json(ledger_without_digest).encode("utf-8")
                )
                _write_atomic_json(self.ledger_path, ledger)
                return ledger
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def reserve_gpu_seconds(self, *, call_id: str, seconds: int) -> None:
        if seconds < 0:
            raise ValueError("GPU seconds may not be negative")
        authorization = self._authorization()
        budget = authorization["budget"]
        if not isinstance(budget, dict):
            raise BudgetExceededError("authorization budget is malformed")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                ledger = self.read()
                reservations = ledger.setdefault("gpu_reservations", [])
                if not isinstance(reservations, list):
                    raise BudgetExceededError("GPU reservation ledger is malformed")
                if any(
                    isinstance(item, dict) and item.get("call_id") == call_id
                    for item in reservations
                ):
                    return
                current = int(str(ledger.get("gpu_seconds_reserved", 0)))
                maximum = int(budget["maximum_modal_gpu_seconds"])
                if current + seconds > maximum:
                    raise BudgetExceededError(
                        "authorized Modal GPU-seconds ceiling would be exceeded"
                    )
                reservations.append(
                    {
                        "call_id": call_id,
                        "seconds": seconds,
                        "reserved_at": datetime.now(UTC).isoformat(),
                    }
                )
                ledger["gpu_seconds_reserved"] = current + seconds
                ledger.pop("canonical_sha256", None)
                ledger_without_digest = {
                    key: value for key, value in ledger.items() if key != "ledger_sha256"
                }
                ledger["ledger_sha256"] = sha256_bytes(
                    canonical_json(ledger_without_digest).encode("utf-8")
                )
                _write_atomic_json(self.ledger_path, ledger)
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def ledger_digest(ledger: dict[str, object]) -> str:
    """Return a deterministic serialized representation for callers to hash."""
    return canonical_json(ledger)
