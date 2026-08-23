from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .errors import DatasetConfigurationError


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    source_glob: str
    schema_path: Path
    output_path: Path
    manifest_path: Path
    huggingface_repo_id: str
    huggingface_private: bool


def _required_string(table: dict[str, object], key: str, section: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetConfigurationError(f"{section}.{key} must be a non-empty string")
    return value


def load_config(path: str | Path = "dataset.toml") -> DatasetConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise DatasetConfigurationError(f"Configuration file not found: {config_path}")

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    dataset = raw.get("dataset")
    huggingface = raw.get("huggingface")
    if not isinstance(dataset, dict):
        raise DatasetConfigurationError("dataset.toml must contain a [dataset] table")
    if not isinstance(huggingface, dict):
        raise DatasetConfigurationError("dataset.toml must contain a [huggingface] table")

    root = config_path.parent
    source_glob = _required_string(dataset, "source_glob", "dataset")
    schema = _required_string(dataset, "schema", "dataset")
    output = _required_string(dataset, "output", "dataset")
    manifest = _required_string(dataset, "manifest", "dataset")
    repo_id = _required_string(huggingface, "repo_id", "huggingface")
    private = huggingface.get("private", False)
    if not isinstance(private, bool):
        raise DatasetConfigurationError("huggingface.private must be true or false")

    return DatasetConfig(
        root=root,
        source_glob=source_glob,
        schema_path=root / schema,
        output_path=root / output,
        manifest_path=root / manifest,
        huggingface_repo_id=repo_id,
        huggingface_private=private,
    )
