from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
class BuildResult:
    output_path: Path
    manifest_path: Path
    row_count: int


def build_dataset(config_path: str | Path = "dataset.toml") -> BuildResult:
    config = load_config(config_path)
    schema = load_schema(config)
    records, sources = load_source_records(config, schema)

    import pyarrow as pa
    import pyarrow.parquet as pq

    # Arrow otherwise infers columns from the first row and can silently drop
    # optional fields that appear only in later source records.
    projected = project_parquet_rows(records, schema)
    try:
        table = pa.Table.from_pylist(projected)
        materialized = table.to_pylist()
    except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:
        raise DatasetValidationError(
            f"Cannot encode schema-valid records as Parquet: {exc}"
        ) from exc
    # Type inference may widen mixed JSON integers/floats into double and round
    # large exact integers. Do not write a valid-looking manifest for data whose
    # physical Arrow representation differs from the source projection.
    if canonical_records_sha256(materialized) != canonical_records_sha256(projected):
        raise DatasetValidationError(
            "Parquet encoding changes schema-valid source values"
        )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        config.output_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )

    manifest = {
        "format_version": 1,
        "output": config.output_path.relative_to(config.root).as_posix(),
        "output_sha256": sha256_file(config.output_path),
        "records_sha256": canonical_records_sha256(records),
        "row_count": len(records),
        "schema": config.schema_path.relative_to(config.root).as_posix(),
        "schema_sha256": sha256_file(config.schema_path),
        "sources": sources,
    }
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return BuildResult(config.output_path, config.manifest_path, len(records))
