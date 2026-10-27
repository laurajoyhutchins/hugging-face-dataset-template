"""Sparse optional source fields must survive Parquet generation and readback."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dataset_pipeline.build import build_dataset
from dataset_pipeline.errors import DatasetValidationError
from dataset_pipeline.io import canonical_records_sha256
from dataset_pipeline.validate import validate_dataset


def _write_source(project: Path, rows: list[dict[str, object]], extra_properties=None) -> None:
    schema_path = project / "schema" / "dataset.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["properties"]["optional_note"] = {"type": "string"}
    schema["properties"].update(extra_properties or {})
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    (project / "source" / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


@pytest.mark.parametrize("note_on_first", [False, True])
def test_optional_field_in_either_row_survives_schema_and_manifest(
    dataset_project: Path, note_on_first: bool
) -> None:
    rows: list[dict[str, object]] = [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta"},
    ]
    rows[0 if note_on_first else 1]["optional_note"] = "retained"
    _write_source(dataset_project, rows)

    result = build_dataset(dataset_project / "dataset.toml")
    projected = pq.read_table(result.output_path)
    assert projected.column_names == ["id", "optional_note", "text"]
    expected = [
        {"id": row["id"], "optional_note": row.get("optional_note"), "text": row["text"]}
        for row in rows
    ]
    assert projected.to_pylist() == expected

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["records_sha256"] == canonical_records_sha256(rows)
    assert validate_dataset(dataset_project / "dataset.toml").records_sha256 == (
        manifest["records_sha256"]
    )


def test_multiple_sparse_fields_are_deterministically_projected(dataset_project: Path) -> None:
    rows: list[dict[str, object]] = [
        {"id": "a", "text": "alpha", "optional_number": 7},
        {"id": "b", "text": "beta", "optional_note": "later"},
    ]
    _write_source(dataset_project, rows, {"optional_number": {"type": "integer"}})
    first = build_dataset(dataset_project / "dataset.toml")
    assert pq.read_table(first.output_path).column_names == [
        "id", "optional_note", "optional_number", "text"
    ]
    assert validate_dataset(dataset_project / "dataset.toml").row_count == 2


def test_ambiguous_missing_and_explicit_null_fail_before_publication(
    dataset_project: Path,
) -> None:
    rows: list[dict[str, object]] = [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta", "optional_note": None},
    ]
    _write_source(dataset_project, rows, {"optional_note": {"type": ["string", "null"]}})
    with pytest.raises(DatasetValidationError, match="missing and explicit null"):
        build_dataset(dataset_project / "dataset.toml")
    assert not (dataset_project / "data" / "train.parquet").exists()
    assert not (dataset_project / "data" / "manifest.json").exists()


def test_parquet_tamper_with_rehashed_bytes_cannot_change_sparse_column(
    dataset_project: Path,
) -> None:
    rows: list[dict[str, object]] = [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta", "optional_note": "authoritative"},
    ]
    _write_source(dataset_project, rows)
    result = build_dataset(dataset_project / "dataset.toml")
    table = pq.read_table(result.output_path)
    import pyarrow as pa

    forged = table.set_column(
        table.column_names.index("optional_note"),
        "optional_note",
        pa.array([None, "forged"]),
    )
    pq.write_table(forged, result.output_path)
    # Even an attacker updating output_sha256 can't hide the semantic mismatch.
    from dataset_pipeline.io import sha256_file

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"] = sha256_file(result.output_path)
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Parquet fields differ"):
        validate_dataset(dataset_project / "dataset.toml")
