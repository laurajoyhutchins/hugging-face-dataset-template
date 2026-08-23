from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from datasets import load_dataset

from dataset_pipeline.build import build_dataset
from dataset_pipeline.errors import DatasetValidationError
from dataset_pipeline.publish import stage_hub_projection
from dataset_pipeline.validate import validate_dataset


def test_build_materializes_parquet_and_manifest(dataset_project: Path) -> None:
    result = build_dataset(dataset_project / "dataset.toml")

    assert result.row_count == 2
    assert pq.read_table(result.output_path).to_pylist() == [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta"},
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count"] == 2
    assert manifest["output"] == "data/train.parquet"
    assert len(manifest["records_sha256"]) == 64
    assert len(manifest["schema_sha256"]) == 64
    assert len(manifest["output_sha256"]) == 64


def test_validate_detects_source_drift(dataset_project: Path) -> None:
    build_dataset(dataset_project / "dataset.toml")
    (dataset_project / "source" / "records.jsonl").write_text(
        '{"id":"a","text":"changed"}\n{"id":"b","text":"beta"}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="Source changed since build"):
        validate_dataset(dataset_project / "dataset.toml")


def test_stage_is_hub_compatible_projection(dataset_project: Path, tmp_path: Path) -> None:
    build_dataset(dataset_project / "dataset.toml")
    staging = tmp_path / "hub"
    stage_hub_projection(dataset_project / "dataset.toml", staging)

    files = sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    )
    assert files == ["README.md", "data/manifest.json", "data/train.parquet"]

    dataset = load_dataset(
        "parquet",
        data_files={"train": str(staging / "data" / "train.parquet")},
        split="train",
    )
    assert dataset.num_rows == 2
    assert dataset.column_names == ["id", "text"]
    assert dataset[0] == {"id": "a", "text": "alpha"}
