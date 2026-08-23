from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .config import load_config
from .errors import DatasetValidationError
from .io import canonical_records_sha256, load_schema, sha256_file


@dataclass(frozen=True)
class ValidationReport:
    output_path: Path
    row_count: int
    records_sha256: str


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetValidationError(f"Manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"Invalid manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DatasetValidationError("Manifest must be a JSON object")
    return manifest


def validate_dataset(config_path: str | Path = "dataset.toml") -> ValidationReport:
    config = load_config(config_path)
    manifest = _read_manifest(config.manifest_path)
    schema = load_schema(config)

    if manifest.get("schema_sha256") != sha256_file(config.schema_path):
        raise DatasetValidationError("Schema changed since build")

    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, list):
        raise DatasetValidationError("Manifest sources are invalid")
    for source in manifest_sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise DatasetValidationError("Manifest source entry is invalid")
        source_path = config.root / source["path"]
        if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
            raise DatasetValidationError(f"Source changed since build: {source['path']}")

    if not config.output_path.is_file():
        raise DatasetValidationError(f"Generated dataset not found: {config.output_path}")
    if manifest.get("output_sha256") != sha256_file(config.output_path):
        raise DatasetValidationError("Generated dataset bytes differ from the build manifest")

    import pyarrow.parquet as pq

    records = pq.read_table(config.output_path).to_pylist()
    validator = Draft202012Validator(schema)
    for index, record in enumerate(records, start=1):
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.path))
        if errors:
            raise DatasetValidationError(f"Generated row {index}: {errors[0].message}")

    records_sha256 = canonical_records_sha256(records)
    if records_sha256 != manifest.get("records_sha256"):
        raise DatasetValidationError("Generated dataset records differ from source build records")
    if len(records) != manifest.get("row_count"):
        raise DatasetValidationError("Generated dataset row count differs from the build manifest")

    return ValidationReport(config.output_path, len(records), records_sha256)
