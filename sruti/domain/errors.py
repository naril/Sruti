class SrutiError(Exception):
    """Base exception for all domain/application errors."""


class ConfigurationError(SrutiError):
    """Invalid configuration or missing required runtime setup."""


class DependencyMissingError(SrutiError):
    """Required local executable/model was not available."""


class StageExecutionError(SrutiError):
    """Stage failed during execution."""


class ExistingOutputError(SrutiError):
    """Stage outputs already exist and policy does not allow overwrite."""


class NonInteractivePromptError(SrutiError):
    """Ask-mode requested while running non-interactively."""


class InvalidLlmJsonError(SrutiError):
    """LLM returned invalid JSON after retries."""
