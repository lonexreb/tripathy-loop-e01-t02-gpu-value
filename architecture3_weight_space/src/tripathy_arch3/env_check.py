"""Print a credential-safe readiness summary."""

from __future__ import annotations

import json

from .config import DEFAULT_ENV_FILE, load_settings


def main() -> None:
    settings = load_settings()
    report = {
        "env_file": str(DEFAULT_ENV_FILE),
        "env_file_exists": DEFAULT_ENV_FILE.is_file(),
        **settings.redacted_summary(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
