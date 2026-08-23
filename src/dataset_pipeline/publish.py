from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .config import load_config
from .errors import DatasetConfigurationError, DatasetValidationError
from .validate import validate_dataset

_PLACEHOLDER_REPO_IDS = {"CHANGE_ME", "YOUR_USERNAME/YOUR_DATASET"}


def stage_hub_projection(
    config_path: str | Path = "dataset.toml", destination: str | Path = ".hub-staging"
) -> Path:
    config = load_config(config_path)
    validate_dataset(config_path)

    readme = config.root / "README.md"
    if not readme.is_file():
        raise DatasetValidationError("README.md dataset card is required for publication")

    destination_path = Path(destination).resolve()
    if destination_path.exists():
        shutil.rmtree(destination_path)
    destination_path.mkdir(parents=True)

    shutil.copy2(readme, destination_path / "README.md")
    for artifact in (config.output_path, config.manifest_path):
        relative = artifact.relative_to(config.root)
        target = destination_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, target)
    return destination_path


def publish_dataset(config_path: str | Path = "dataset.toml", token: str | None = None) -> str:
    config = load_config(config_path)
    if config.huggingface_repo_id in _PLACEHOLDER_REPO_IDS or "/" not in config.huggingface_repo_id:
        raise DatasetConfigurationError(
            "huggingface.repo_id must be set to an owner/dataset repository before publishing"
        )

    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory(prefix="dataset-hub-") as temporary_directory:
        staging = stage_hub_projection(config_path, temporary_directory)
        api = HfApi(token=token)
        api.create_repo(
            repo_id=config.huggingface_repo_id,
            repo_type="dataset",
            private=config.huggingface_private,
            exist_ok=True,
        )
        result = api.upload_folder(
            repo_id=config.huggingface_repo_id,
            repo_type="dataset",
            folder_path=staging,
            commit_message="Publish dataset projection",
        )
    return str(result)
