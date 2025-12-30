"""Configuration module for RagInProduction.

Loads settings from config.yaml for non-sensitive values and .env for API keys.
Fails fast with clear error messages if required configuration is missing.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""
    pass


def _get_project_root() -> Path:
    """Get the project root directory (where config.yaml lives)."""
    # Navigate from src/config/ up to project root
    return Path(__file__).parent.parent.parent


def _load_yaml_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = _get_project_root() / "config.yaml"
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_required_env(key: str) -> str:
    """Get required environment variable or raise ConfigurationError."""
    value = os.environ.get(key)
    if not value:
        raise ConfigurationError(
            f"Required environment variable '{key}' is not set. "
            f"Please add it to your .env file."
        )
    return value


def _get_optional_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get optional environment variable with default."""
    return os.environ.get(key, default)


@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str
    model: str
    embedding_model: str
    embedding_dimensions: int


@dataclass(frozen=True)
class AzureAISearchConfig:
    """Azure AI Search configuration."""
    api_key: str
    endpoint: str
    index_name: str


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""
    path: str
    sqlite_engine_url: str


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration container."""
    openai: OpenAIConfig
    azure_ai_search: AzureAISearchConfig
    database: DatabaseConfig
    logging: LoggingConfig


def load_config() -> AppConfig:
    """
    Load and validate all application configuration.

    Loads from config.yaml for non-sensitive settings and .env for API keys.
    Fails fast if required configuration is missing.

    Returns:
        AppConfig: Validated application configuration.

    Raises:
        ConfigurationError: If required configuration is missing.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Load YAML configuration
    yaml_config = _load_yaml_config()

    # Build OpenAI config
    openai_section = yaml_config.get("openai", {})
    embedding_section = openai_section.get("embedding_model", {})

    openai_config = OpenAIConfig(
        api_key=_get_required_env("OPENAI_API_KEY"),
        model=openai_section.get("model", "gpt-4o"),
        embedding_model=embedding_section.get("name", "text-embedding-3-small"),
        embedding_dimensions=embedding_section.get("dimensions", 1536),
    )

    # Build Azure AI Search config
    ai_search_section = yaml_config.get("ai_search", {})

    azure_ai_search_config = AzureAISearchConfig(
        api_key=_get_required_env("AI_SEARCH_KEY"),
        endpoint=ai_search_section.get("endpoint", _get_required_env("AI_SEARCH_ENDPOINT")),
        index_name=ai_search_section.get("index_name", _get_required_env("AI_SEARCH_NAME")),
    )

    # Build Database config
    db_section = yaml_config.get("database", {})
    sqlalchemy_section = yaml_config.get("sqlalchemy", {})

    database_config = DatabaseConfig(
        path=db_section.get("path", "products.db"),
        sqlite_engine_url=sqlalchemy_section.get("sqlite_engine_url", "sqlite:///products.db"),
    )

    # Build Logging config
    logging_section = yaml_config.get("logging", {})

    logging_config = LoggingConfig(
        level=logging_section.get("level", "INFO"),
    )

    return AppConfig(
        openai=openai_config,
        azure_ai_search=azure_ai_search_config,
        database=database_config,
        logging=logging_config,
    )


# Module-level singleton for convenience
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the application configuration singleton.

    Lazy-loads configuration on first access.

    Returns:
        AppConfig: Application configuration.

    Raises:
        ConfigurationError: If required configuration is missing.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config
