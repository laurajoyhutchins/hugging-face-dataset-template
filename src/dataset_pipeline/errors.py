class DatasetPipelineError(RuntimeError):
    """Base error for dataset pipeline failures."""


class DatasetConfigurationError(DatasetPipelineError):
    """Raised when dataset.toml is invalid or incomplete."""


class DatasetValidationError(DatasetPipelineError):
    """Raised when source or generated data violates the repository contract."""
