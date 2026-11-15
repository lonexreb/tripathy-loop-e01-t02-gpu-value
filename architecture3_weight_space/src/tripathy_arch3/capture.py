"""Content-addressed, secret-free request and response capture."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .audit import canonical_json, sha256_bytes
from .contracts import ArtifactRef


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def capture_json(
    root: Path,
    *,
    provider: str,
    call_id: str,
    request: object,
    response_status: int,
    response_headers: dict[str, str],
    response_body: bytes,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Freeze one sanitized request and exact response under a stable call id."""
    request_bytes = (canonical_json(request) + "\n").encode("utf-8")
    response_envelope = {
        "body_sha256": sha256_bytes(response_body),
        "headers": {
            key: value
            for key, value in response_headers.items()
            if key in {"content-type", "x-request-id", "request-id"}
        },
        "status_code": response_status,
    }
    response_bytes = (json.dumps(response_envelope, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    directory = root / provider / call_id
    request_path = directory / "request.json"
    response_path = directory / "response.json"
    raw_path = directory / "response-body.bin"
    write_atomic(request_path, request_bytes)
    write_atomic(response_path, response_bytes)
    write_atomic(raw_path, response_body)
    return (
        ArtifactRef(
            path=request_path.relative_to(root.parent.parent).as_posix(),
            sha256=sha256_bytes(request_bytes),
            media_type="application/json",
        ),
        ArtifactRef(
            path=raw_path.relative_to(root.parent.parent).as_posix(),
            sha256=sha256_bytes(response_body),
            media_type=response_headers.get("content-type", "application/octet-stream"),
        ),
    )
