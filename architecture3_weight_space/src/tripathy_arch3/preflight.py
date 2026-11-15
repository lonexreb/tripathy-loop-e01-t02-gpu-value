"""Local, credential-safe preflight and frozen-input integrity verifier."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict

from .audit import canonical_json, sha256_bytes, sha256_file
from .config import WORKSPACE_ROOT, load_settings

REPO_ROOT = WORKSPACE_ROOT.parent
EXPECTED_REMOTE = "https://github.com/lonexreb/tripathy-loop-e01-t02-gpu-value.git"
EXPECTED_BRANCH = "e01/track2-direction-a"
SCOPE_PATH = WORKSPACE_ROOT / "config" / "architecture3.json"
AUTHORIZATION_PATH = WORKSPACE_ROOT / "config" / "execution_authorization.json"


class CheckResult(TypedDict):
    name: str
    status: Literal["pass", "fail", "warning", "skipped"]
    detail: str


def _check(
    name: str,
    passed: bool,
    success_detail: str,
    failure_detail: str,
    *,
    warning: bool = False,
) -> CheckResult:
    if passed:
        return {"name": name, "status": "pass", "detail": success_detail}
    return {
        "name": name,
        "status": "warning" if warning else "fail",
        "detail": failure_detail,
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _parse_checksum_file(path: Path) -> tuple[str, str]:
    line = path.read_text(encoding="utf-8").strip()
    checksum, relative_path = line.split(maxsplit=1)
    return checksum, relative_path.lstrip("*")


def verify_pack_manifest(pack_root: Path) -> tuple[int, list[str]]:
    """Verify every path in the pack's authoritative MANIFEST.sha256."""
    manifest_path = pack_root / "MANIFEST.sha256"
    mismatches: list[str] = []
    count = 0
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative_path = line.split(maxsplit=1)
        except ValueError:
            mismatches.append(f"MANIFEST.sha256:{line_number}: malformed")
            continue
        relative_path = relative_path.lstrip("*")
        target = pack_root / relative_path
        count += 1
        if not target.is_file():
            mismatches.append(f"{relative_path}: missing")
        elif sha256_file(target) != expected:
            mismatches.append(f"{relative_path}: checksum mismatch")
    return count, mismatches


def _controlling_paths() -> list[Path]:
    fixed_paths = [
        WORKSPACE_ROOT / "IMPLEMENTATION_PLAN.md",
        WORKSPACE_ROOT / ".env.example",
        SCOPE_PATH,
        AUTHORIZATION_PATH,
        WORKSPACE_ROOT / "experiment" / "manifest.json",
        WORKSPACE_ROOT / "modal_app.py",
        WORKSPACE_ROOT / "pyproject.toml",
        WORKSPACE_ROOT / "uv.lock",
        REPO_ROOT / "experiment" / "manifest.json",
        REPO_ROOT / "research" / "PROTOCOL.md",
        REPO_ROOT / "research" / "origins.json",
        REPO_ROOT / "research" / "adapted_series.txt",
        REPO_ROOT / "research" / "paired.py",
        REPO_ROOT / "gpu-price-pack-v0.1.1-handover" / "starter" / "scorer.py",
        REPO_ROOT / "gpu-price-pack-v0.1.1-handover" / "MANIFEST.sha256",
        REPO_ROOT / "gpu-price-pack-v0.1.1-handover" / "DATA.md",
        REPO_ROOT / "gpu-price-pack-v0.1.1-handover" / "VERSION",
        REPO_ROOT / "source-downloads" / "ZIP.sha256",
    ]
    source_paths = sorted((WORKSPACE_ROOT / "src" / "tripathy_arch3").glob("*.py"))
    test_paths = sorted((WORKSPACE_ROOT / "tests").glob("test_*.py"))
    return [*fixed_paths, *source_paths, *test_paths]


def collect_preflight(*, verify_pack_files: bool) -> dict[str, object]:
    settings = load_settings()
    settings.validate(require_pinned_model=True, require_critic=True)
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    experiment = json.loads(
        (WORKSPACE_ROOT / "experiment" / "manifest.json").read_text(encoding="utf-8")
    )
    authorization = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    checks: list[CheckResult] = []

    paths = _controlling_paths()
    missing_paths = [str(path.relative_to(REPO_ROOT)) for path in paths if not path.is_file()]
    checks.append(
        _check(
            "controlling_files",
            not missing_paths,
            f"{len(paths)} required files present",
            "missing: " + ", ".join(missing_paths),
        )
    )

    remote = _git("remote", "get-url", "origin")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    dirty_lines = _git("status", "--short", "--untracked-files=all").splitlines()
    checks.extend(
        [
            _check("git_remote", remote == EXPECTED_REMOTE, remote, f"unexpected remote: {remote}"),
            _check("git_branch", branch == EXPECTED_BRANCH, branch, f"unexpected branch: {branch}"),
            _check(
                "git_worktree",
                not dirty_lines,
                "clean",
                f"{len(dirty_lines)} changed or untracked paths recorded",
                warning=True,
            ),
        ]
    )

    actor = scope["components"]["actor"]
    checks.append(
        _check(
            "pinned_actor",
            actor["model_id"] == settings.model_id
            and actor["revision"] == settings.model_revision
            and experiment["base_model"] == settings.model_id
            and experiment["base_model_revision"] == settings.model_revision,
            f"{settings.model_id}@{settings.model_revision}",
            "scope, manifest, and environment actor configuration differ",
        )
    )
    checks.append(
        _check(
            "service_credentials",
            bool(
                settings.openai_api_key
                and settings.exa_api_key
                and settings.past_api_key
                and (Path.home() / ".modal.toml").is_file()
            ),
            "OpenAI, Exa, past.dev, and Modal authentication are present",
            "one or more required service credentials are absent",
        )
    )
    remote_gates = scope["remote_gates"]
    gates_authorized = all(value is True for value in remote_gates.values())
    authorization_matches = (
        authorization.get("authorized") is True
        and authorization.get("scope", {}).get("provider_calls") is True
        and authorization.get("scope", {}).get("modal_gpu") is True
        and authorization.get("scope", {}).get("model_download") is True
        and authorization.get("scope", {}).get("lora_training") is True
        and authorization.get("scope", {}).get("git_commit") is False
        and authorization.get("scope", {}).get("git_push") is False
        and authorization.get("budget", {}).get("maximum_total_usd")
        == experiment.get("maximum_total_budget_usd")
        == 1.0
    )
    checks.append(
        _check(
            "remote_gates",
            gates_authorized and authorization_matches,
            "remote actions match the bounded execution authorization; Git writes remain closed",
            "remote gates and execution authorization do not match",
        )
    )

    zip_checksum_file = REPO_ROOT / "source-downloads" / "ZIP.sha256"
    expected_zip_hash, zip_name = _parse_checksum_file(zip_checksum_file)
    zip_path = REPO_ROOT / "source-downloads" / zip_name
    actual_zip_hash = sha256_file(zip_path) if zip_path.is_file() else None
    checks.append(
        _check(
            "pack_zip",
            actual_zip_hash == expected_zip_hash,
            f"published SHA-256 verified for {zip_name}",
            "pack ZIP is missing or does not match its published checksum",
        )
    )

    pack_root = REPO_ROOT / "gpu-price-pack-v0.1.1-handover"
    pack_file_count: int | None = None
    pack_mismatches: list[str] = []
    if verify_pack_files:
        pack_file_count, pack_mismatches = verify_pack_manifest(pack_root)
        checks.append(
            _check(
                "pack_manifest",
                not pack_mismatches,
                f"{pack_file_count} pack files match MANIFEST.sha256",
                "; ".join(pack_mismatches[:20]),
            )
        )
    else:
        checks.append(
            {
                "name": "pack_manifest",
                "status": "skipped",
                "detail": "full pack verification skipped by explicit CLI flag",
            }
        )

    file_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths if path.is_file()
    }
    stable_inputs = {
        "file_hashes": file_hashes,
        "model_id": settings.model_id,
        "model_revision": settings.model_revision,
        "pack_zip_sha256": actual_zip_hash,
        "scope": scope,
    }
    stable_input_digest = sha256_bytes(canonical_json(stable_inputs).encode("utf-8"))
    failures = sum(check["status"] == "fail" for check in checks)
    warnings = sum(check["status"] == "warning" for check in checks)
    status = "failed" if failures else "ready_with_warnings" if warnings else "ready"
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "experiment_id": scope["experiment_id"],
        "git": {
            "branch": branch,
            "dirty_paths": dirty_lines,
            "head": head,
            "remote": remote,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "configuration": settings.redacted_summary(),
        "checks": checks,
        "file_hashes": file_hashes,
        "pack": {
            "manifest_entries_verified": pack_file_count,
            "mismatches": pack_mismatches,
            "zip_sha256": actual_zip_hash,
        },
        "stable_input_digest": stable_input_digest,
        "summary": {"failures": failures, "warnings": warnings},
    }


def write_preflight(report: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/preflight/latest.json"),
        help="Output path relative to the Architecture 3 workspace unless absolute",
    )
    parser.add_argument(
        "--skip-pack-files",
        action="store_true",
        help="Skip the 672-file pack verification for a quick local check",
    )
    arguments = parser.parse_args()
    output_path = arguments.output
    if not output_path.is_absolute():
        output_path = WORKSPACE_ROOT / output_path
    report = collect_preflight(verify_pack_files=not arguments.skip_pack_files)
    write_preflight(report, output_path)
    summary = {
        "output": str(output_path),
        "stable_input_digest": report["stable_input_digest"],
        "status": report["status"],
        "summary": report["summary"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
