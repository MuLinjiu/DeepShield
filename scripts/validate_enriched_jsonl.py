#!/usr/bin/env python3
"""Validate enriched JSONL records against schema expectations."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

import jsonschema


def load_schema(schema_path: Path) -> dict:
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.Draft7Validator.check_schema(schema)
    return schema


def list_jsonl_targets(paths: Iterable[Path]) -> List[Path]:
    targets: List[Path] = []
    for path in paths:
        if path.is_dir():
            targets.extend(sorted(p for p in path.glob("*.jsonl") if p.is_file()))
        elif path.is_file():
            targets.append(path)
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    if not targets:
        raise FileNotFoundError("No JSONL files found in provided paths.")
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSONL files or directories containing enriched LLM input JSONL files",
    )
    parser.add_argument(
        "--schema",
        required=True,
        help="Path to JSON schema describing record structure",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of records to validate per file (default: 100)",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    schema = load_schema(schema_path)
    validator = jsonschema.Draft7Validator(schema)

    targets = list_jsonl_targets(Path(p) for p in args.paths)
    total_records = 0
    errors_found: List[str] = []

    for jsonl_path in targets:
        validated = 0
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    errors_found.append(f"{jsonl_path}:{line_no}: invalid JSON: {exc}")
                    continue

                for error in validator.iter_errors(record):
                    location = " -> ".join(str(x) for x in error.path) or "(root)"
                    errors_found.append(
                        f"{jsonl_path}:{line_no}: {location}: {error.message}"
                    )
                validated += 1
                total_records += 1
                if validated >= args.limit:
                    break

    if errors_found:
        for message in errors_found:
            print(message)
        print(f"\nValidation failed for {len(errors_found)} issue(s) across {len(targets)} file(s).")
        sys.exit(1)

    print(f"Validation succeeded: {total_records} records across {len(targets)} file(s).")


if __name__ == "__main__":
    main()
