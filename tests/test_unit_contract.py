from __future__ import annotations

from pathlib import Path

import pytest

from dataset_pipeline.build import build_dataset
from dataset_pipeline.config import load_config
from dataset_pipeline.errors import DatasetConfigurationError, DatasetValidationError
from dataset_pipeline.publish import publish_dataset


def test_config_resolves_paths_relative_to_config_file(dataset_project: Path) -> None:
    config = load_config(dataset_project / "dataset.toml")

    assert config.root == dataset_project
    assert config.schema_path == dataset_project / "schema" / "dataset.schema.json"
    assert config.output_path == dataset_project / "data" / "train.parquet"


def test_build_rejects_schema_violation_before_materialization(dataset_project: Path) -> None:
    (dataset_project / "source" / "records.jsonl").write_text(
        '{"id":"a","text":"alpha"}\n{"id":"b"}\n', encoding="utf-8"
    )

    with pytest.raises(DatasetValidationError, match="source/records.jsonl:2"):
        build_dataset(dataset_project / "dataset.toml")


def test_publish_rejects_placeholder_repo_id_before_network_call(dataset_project: Path) -> None:
    with pytest.raises(DatasetConfigurationError, match="huggingface.repo_id"):
        publish_dataset(dataset_project / "dataset.toml")


def test_publish_replaces_stale_remote_projection(
    dataset_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = dataset_project / "dataset.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("CHANGE_ME", "owner/dataset"),
        encoding="utf-8",
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    calls: dict[str, object] = {}

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            calls["token"] = token

        def create_repo(self, **kwargs: object) -> None:
            calls["create_repo"] = kwargs

        def upload_folder(self, **kwargs: object) -> str:
            calls["upload_folder"] = kwargs
            return "published"

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr(
        "dataset_pipeline.publish.stage_hub_projection",
        lambda config_path, destination: staging,
    )

    result = publish_dataset(config_path, token="test-token")

    assert result == "published"
    upload = calls["upload_folder"]
    assert isinstance(upload, dict)
    assert upload["delete_patterns"] == "**"
