"""
Configuration management for MLSysBook tools.

This module provides centralized configuration management with support for
environment variables, configuration files, and sensible defaults.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

from .exceptions import ConfigurationError


@dataclass
class Config:
    """Central configuration class for MLSysBook tools.

    This class manages all configuration settings with support for environment
    variables, configuration files, and programmatic overrides.
    """

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parents[3])
    quarto_root: Path = field(default_factory=lambda: Path(__file__).parents[3] / "quarto")
    tools_root: Path = field(default_factory=lambda: Path(__file__).parents[1])
    content_root: Path = field(default_factory=lambda: Path(__file__).parents[3] / "quarto" / "contents")

    # Logging configuration
    log_level: str = field(default="INFO")
    log_format: str = field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_to_file: bool = field(default=False)
    log_file_path: Optional[Path] = field(default=None)

    # API configurations
    openai_api_key: Optional[str] = field(default=None)
    ollama_base_url: str = field(default="http://localhost:11434")
    ollama_model: str = field(default="llama3.1:8b")

    # Processing settings
    max_workers: int = field(default=4)
    chunk_size: int = field(default=1000)
    enable_caching: bool = field(default=True)
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "mlsysbook")

    # Content processing settings
    backup_enabled: bool = field(default=True)
    backup_dir: Path = field(default_factory=lambda: Path(__file__).parents[3] / "backups")
    dry_run: bool = field(default=False)

    # Quality thresholds
    similarity_threshold: float = field(default=0.65)
    min_caption_length: int = field(default=10)
    max_caption_length: int = field(default=500)

    def __post_init__(self) -> None:
        """Initialize configuration after dataclass creation."""
        self._load_from_environment()
        self._load_from_config_file()
        self._validate_configuration()

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            "MLSYSBOOK_LOG_LEVEL": "log_level",
            "MLSYSBOOK_LOG_TO_FILE": "log_to_file",
            "MLSYSBOOK_DRY_RUN": "dry_run",
            "MLSYSBOOK_BACKUP_ENABLED": "backup_enabled",
            "MLSYSBOOK_MAX_WORKERS": "max_workers",
            "OPENAI_API_KEY": "openai_api_key",
            "OLLAMA_BASE_URL": "ollama_base_url",
            "OLLAMA_MODEL": "ollama_model",
        }

        for env_var, attr_name in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if hasattr(self, attr_name):
                    current_value = getattr(self, attr_name)
                    if isinstance(current_value, bool):
                        setattr(self, attr_name, env_value.lower() in ("true", "1", "yes", "on"))
                    elif isinstance(current_value, int):
                        try:
                            setattr(self, attr_name, int(env_value))
                        except ValueError:
                            raise ConfigurationError(
                                f"Invalid integer value for {env_var}: {env_value}"
                            )
                    elif isinstance(current_value, float):
                        try:
                            setattr(self, attr_name, float(env_value))
                        except ValueError:
                            raise ConfigurationError(
                                f"Invalid float value for {env_var}: {env_value}"
                            )
                    else:
                        setattr(self, attr_name, env_value)

    def _load_from_config_file(self) -> None:
        """Load configuration from config file if it exists."""
        config_paths = [
            self.project_root / "mlsysbook.yaml",
            self.project_root / "mlsysbook.yml",
            self.project_root / ".mlsysbook.yaml",
            self.project_root / ".mlsysbook.yml",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)

                    if config_data:
                        self._apply_config_dict(config_data)
                    break
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to load configuration from {config_path}: {e}"
                    )

    def _apply_config_dict(self, config_data: Dict[str, Any]) -> None:
        """Apply configuration from a dictionary."""
        for key, value in config_data.items():
            if hasattr(self, key):
                # Convert path strings to Path objects
                if key.endswith(('_root', '_dir', '_path')) and isinstance(value, str):
                    setattr(self, key, Path(value))
                else:
                    setattr(self, key, value)

    def _validate_configuration(self) -> None:
        """Validate configuration values."""
        # Ensure directories exist or can be created
        for attr_name in ['cache_dir', 'backup_dir']:
            dir_path = getattr(self, attr_name)
            if dir_path and not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ConfigurationError(
                        f"Cannot create directory {dir_path}: {e}"
                    )

        # Validate numeric ranges
        if self.max_workers < 1:
            raise ConfigurationError("max_workers must be at least 1")

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ConfigurationError("similarity_threshold must be between 0.0 and 1.0")

        if self.min_caption_length < 1:
            raise ConfigurationError("min_caption_length must be at least 1")

        if self.max_caption_length < self.min_caption_length:
            raise ConfigurationError("max_caption_length must be >= min_caption_length")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        config_dict = self.to_dict()

        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {path}: {e}")


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        The global Config instance, creating it if necessary.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance.

    This is primarily useful for testing.
    """
    global _config
    _config = None
