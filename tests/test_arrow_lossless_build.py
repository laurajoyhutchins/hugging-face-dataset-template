from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dataset_pipeline.build import build_dataset
from dataset_pipeline.errors import DatasetValidationError
from dataset_pipeline.io import canonical_records_sha256
from dataset_pipeline.validate import validate_dataset


def _set_readings(root: Path, first: int | float, second: int | float) -> list[dict]:
    rows = [
        {"id": "a", "text": "alpha", "reading": first},
        {"id": "b", "text": "beta", "reading": second},
    ]
    schema_path = root / "schema" / "dataset.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["properties"]["reading"] = {"type": "number"}
    schema["required"].append("reading")
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    source = root / "source" / "records.jsonl"
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return rows


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (2**53 + 1, 1.5),
        (1, 1.5),
    ],
)
def test_builder_fails_closed_before_manifest_for_mixed_numeric_arrow_coercion(
    dataset_project: Path, first: int, second: float
) -> None:
    rows = _set_readings(dataset_project, first, second)
    # Both rows are valid JSON Schema numbers. Arrow infers float64 when it
    # sees the float and cannot retain the original integer JSON representation.
    assert rows[0]["reading"] == first
    with pytest.raises(DatasetValidationError, match="Parquet encoding changes"):
        build_dataset(dataset_project / "dataset.toml")
    assert not (dataset_project / "data" / "train.parquet").exists()
    assert not (dataset_project / "data" / "manifest.json").exists()


@pytest.mark.parametrize(
    ("first", "second"),
    [(1, 2), (1.25, 2.5)],
)
def test_same_type_numeric_sources_keep_exact_manifest_identity(
    dataset_project: Path, first: int | float, second: int | float
) -> None:
    rows = _set_readings(dataset_project, first, second)
    result = build_dataset(dataset_project / "dataset.toml")
    assert pq.read_table(result.output_path).to_pylist() == rows
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["records_sha256"] == canonical_records_sha256(rows)
    assert validate_dataset(dataset_project / "dataset.toml").records_sha256 == (
        canonical_records_sha256(rows)
    )
