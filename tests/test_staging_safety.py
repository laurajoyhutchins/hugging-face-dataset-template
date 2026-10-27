from __future__ import annotations

from pathlib import Path

import pytest

from dataset_pipeline.errors import DatasetValidationError
from dataset_pipeline.publish import stage_hub_projection


@pytest.mark.parametrize(
    "destination_kind",
    ["project", "ancestor", "source", "schema", "generated", "nested", "git"],
)
def test_staging_rejects_project_overlap_before_validation(
    dataset_project: Path, monkeypatch: pytest.MonkeyPatch, destination_kind: str
) -> None:
    protected = dataset_project / "source" / "records.jsonl"
    before = protected.read_bytes()
    targets = {
        "project": dataset_project,
        "ancestor": dataset_project.parent,
        "source": dataset_project / "source",
        "schema": dataset_project / "schema",
        "generated": dataset_project / "data",
        "nested": dataset_project / "source" / "nested",
        "git": dataset_project / ".git",
    }

    def unexpected_validation(_config: object) -> None:
        raise AssertionError("invalid staging destination reached validation")

    monkeypatch.setattr("dataset_pipeline.publish.validate_dataset", unexpected_validation)
    with pytest.raises(DatasetValidationError, match="outside the authoritative dataset project"):
        stage_hub_projection(dataset_project / "dataset.toml", targets[destination_kind])
    assert protected.read_bytes() == before


def test_symlink_alias_into_project_is_rejected_before_validation(
    dataset_project: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path_factory.mktemp("alias-project")
    alias = external / "alias"
    alias.symlink_to(dataset_project, target_is_directory=True)
    sentinel = dataset_project / "source" / "records.jsonl"
    original = sentinel.read_bytes()
    monkeypatch.setattr(
        "dataset_pipeline.publish.validate_dataset",
        lambda _config: pytest.fail("invalid path reached validation"),
    )
    with pytest.raises(DatasetValidationError, match="outside the authoritative dataset project"):
        stage_hub_projection(dataset_project / "dataset.toml", alias / "data")
    assert sentinel.read_bytes() == original


def test_external_symlink_destination_is_rejected(
    dataset_project: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path_factory.mktemp("alias-external") / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep", encoding="utf-8")
    alias = external.parent / "alias"
    alias.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        "dataset_pipeline.publish.validate_dataset",
        lambda _config: pytest.fail("symlink reached validation"),
    )
    with pytest.raises(DatasetValidationError, match="cannot be a symbolic link"):
        stage_hub_projection(dataset_project / "dataset.toml", alias)
    assert (external / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_external_staging_replaces_only_generated_projection(
    dataset_project: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dataset_pipeline.publish.validate_dataset", lambda _config: None)
    data = dataset_project / "data"
    (data / "train.parquet").write_bytes(b"parquet-fixture-bytes")
    (data / "manifest.json").write_text('{"row_count":2}', encoding="utf-8")
    destination = tmp_path_factory.mktemp("valid-staging") / "safe-external-staging"
    destination.mkdir()
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    staged = stage_hub_projection(dataset_project / "dataset.toml", destination)
    assert staged == destination
    assert sorted(
        p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()
    ) == ["README.md", "data/manifest.json", "data/train.parquet"]
    assert (staged / "data" / "train.parquet").read_bytes() == b"parquet-fixture-bytes"
    assert (data / "train.parquet").read_bytes() == b"parquet-fixture-bytes"
