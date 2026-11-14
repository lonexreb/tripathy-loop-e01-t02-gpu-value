from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import ExperimentManifest, ValidationError
from .runner import ExperimentRunner


def load_manifest(path: Path) -> ExperimentManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentManifest.from_dict(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tripathy Loop experiment runner")
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument("--manifest", type=Path, default=Path("experiment/manifest.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        if args.command == "validate":
            manifest.validate()
            print(json.dumps({"status": "valid", "experiment_id": manifest.experiment_id}))
            return 0

        result = ExperimentRunner().run(manifest, dry_run=args.dry_run)
        print(json.dumps(result.to_dict(), sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
