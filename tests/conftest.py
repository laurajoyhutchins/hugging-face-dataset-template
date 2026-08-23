from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def dataset_project(tmp_path: Path) -> Path:
    (tmp_path / "source").mkdir()
    (tmp_path / "schema").mkdir()
    (tmp_path / "data").mkdir()

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["id", "text"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "text": {"type": "string"},
        },
        "additionalProperties": False,
    }
    (tmp_path / "schema" / "dataset.schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )
    (tmp_path / "source" / "records.jsonl").write_text(
        '{"id":"a","text":"alpha"}\n{"id":"b","text":"beta"}\n',
        encoding="utf-8",
    )
    card = (
        "---\nconfigs:\n- config_name: default\n  data_files:\n"
        "  - split: train\n    path: data/train.parquet\n---\n\n# Fixture\n"
    )
    (tmp_path / "README.md").write_text(card, encoding="utf-8")
    (tmp_path / "dataset.toml").write_text(
        '[dataset]\n'
        'source_glob = "source/*.jsonl"\n'
        'schema = "schema/dataset.schema.json"\n'
        'output = "data/train.parquet"\n'
        'manifest = "data/manifest.json"\n\n'
        '[huggingface]\n'
        'repo_id = "CHANGE_ME"\n'
        'private = false\n',
        encoding="utf-8",
    )
    return tmp_path
