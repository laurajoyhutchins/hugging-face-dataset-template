from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .config import DatasetConfig
from .errors import DatasetValidationError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_records_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        encoded = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def project_parquet_rows(
    records: list[dict[str, Any]], schema: dict[str, Any]
) -> list[dict[str, Any]]:
    """Project complete union columns without redefining source JSON identities.

    Parquet has no representation for a missing field distinct from a null
    value. Missing *optional* fields are represented by null only when that is
    unambiguous given the validated source rows; the source JSONL stays pinned
    separately in the manifest.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not isinstance(properties, dict):
        raise DatasetValidationError("Dataset schema requires declared object properties")

    columns = sorted({key for record in records for key in record})
    for column in columns:
        missing = any(column not in record for record in records)
        explicit_null = any(column in record and record[column] is None for record in records)
        if missing and column in required:
            raise DatasetValidationError(f"Required dataset field is absent: {column}")
        if missing and explicit_null:
            raise DatasetValidationError(
                f"Cannot losslessly project optional field {column!r}: "
                "missing and explicit null would be indistinguishable"
            )
    return [{column: row.get(column) for column in columns} for row in records]


def load_schema(config: DatasetConfig) -> dict[str, Any]:
    if not config.schema_path.is_file():
        raise DatasetValidationError(
            f"Schema file not found: {config.schema_path.relative_to(config.root)}"
        )
    try:
        schema = json.loads(config.schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Invalid JSON schema: {exc}") from exc
    if not isinstance(schema, dict):
        raise DatasetValidationError("Dataset schema must be a JSON object")
    Draft202012Validator.check_schema(schema)
    return schema


def load_source_records(
    config: DatasetConfig, schema: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    source_paths = sorted(path for path in config.root.glob(config.source_glob) if path.is_file())
    if not source_paths:
        raise DatasetValidationError(f"No source files match {config.source_glob!r}")

    validator = Draft202012Validator(schema)
    records: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []

    for source_path in source_paths:
        relative = source_path.relative_to(config.root).as_posix()
        sources.append({"path": relative, "sha256": sha256_file(source_path)})
        for line_number, raw_line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    f"{relative}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise DatasetValidationError(
                    f"{relative}:{line_number}: each JSONL record must be an object"
                )
            errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
            if errors:
                raise DatasetValidationError(f"{relative}:{line_number}: {errors[0].message}")
            records.append(record)

    if not records:
        raise DatasetValidationError("Source files contain no dataset records")
    return records, sources
