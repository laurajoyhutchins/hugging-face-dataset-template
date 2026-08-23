from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .io import canonical_records_sha256, load_schema, load_source_records, sha256_file


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

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
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
