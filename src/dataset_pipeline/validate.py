from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .errors import DatasetValidationError
from .io import (
    canonical_records_sha256,
    load_schema,
    load_source_records,
    project_parquet_rows,
    sha256_file,
)


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
    declared_paths = []
    for source in manifest_sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise DatasetValidationError("Manifest source entry is invalid")
        declared_paths.append(source["path"])

    # Membership is part of build identity, not just the hashes of listed files.
    # Compare sorted lists so duplicate manifest entries also fail closed.
    current_paths = sorted(
        path.relative_to(config.root).as_posix()
        for path in config.root.glob(config.source_glob)
        if path.is_file()
    )
    if sorted(declared_paths) != current_paths:
        raise DatasetValidationError(
            "Source inventory changed since build: added, removed, or renamed files"
        )

    for source in manifest_sources:
        source_path = config.root / source["path"]
        if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
            raise DatasetValidationError(f"Source changed since build: {source['path']}")

    if not config.output_path.is_file():
        raise DatasetValidationError(f"Generated dataset not found: {config.output_path}")
    if manifest.get("output_sha256") != sha256_file(config.output_path):
        raise DatasetValidationError("Generated dataset bytes differ from the build manifest")

    import pyarrow.parquet as pq

    # Source JSONL remains authoritative. Parquet null denotes an *absent*
    # optional field and is not a new JSON value to validate against source
    # field types or include in the source-record digest.
    source_records, _ = load_source_records(config, schema)
    records_sha256 = canonical_records_sha256(source_records)
    if records_sha256 != manifest.get("records_sha256"):
        raise DatasetValidationError("Source record identities differ from the build manifest")
    expected = project_parquet_rows(source_records, schema)
    table = pq.read_table(config.output_path)
    records = table.to_pylist()
    if len(records) != manifest.get("row_count"):
        raise DatasetValidationError("Generated dataset row count differs from the build manifest")
    if table.column_names != sorted({key for row in expected for key in row}):
        raise DatasetValidationError("Generated Parquet columns differ from source projection")
    if canonical_records_sha256(records) != canonical_records_sha256(expected):
        raise DatasetValidationError("Generated Parquet fields differ from source projection")
    return ValidationReport(config.output_path, len(records), records_sha256)
