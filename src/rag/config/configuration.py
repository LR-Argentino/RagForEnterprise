"""Configuration module for RagInProduction.

Loads settings from environment-specific config files:
- APP_ENV=dev  → config_dev.yaml (SQLite backend, local development)
- APP_ENV=test → config_test.yaml (CosmosDB backend, production-like testing)
- Default      → config.yaml

API keys are loaded from .env file.
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


def _get_config_filename() -> str:
    """Get config filename based on APP_ENV environment variable.

    Returns:
        Config filename:
        - APP_ENV=dev  → config_dev.yaml
        - APP_ENV=test → config_test.yaml
        - Default      → config.yaml
    """
    app_env = os.environ.get("APP_ENV", "").lower()

    if app_env == "dev":
        return "config_dev.yaml"
    elif app_env == "test":
        return "config_test.yaml"
    else:
        return "config.yaml"


def _load_yaml_config() -> dict:
    """Load configuration from environment-specific config file."""
    config_filename = _get_config_filename()
    config_path = _get_project_root() / config_filename

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}. "
            f"Set APP_ENV to 'dev' or 'test', or create {config_filename}."
        )

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
    pdf_index_name: str


@dataclass(frozen=True)
class DocumentIntelligenceConfig:
    """Azure Document Intelligence configuration."""
    api_key: str
    endpoint: str
    model_id: str


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""
    path: str
    sqlite_engine_url: str
    document_tracking_path: str


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str


@dataclass(frozen=True)
class AuditLogConfig:
    """Audit log configuration with backend toggle."""
    backend: str  # "sqlite" or "cosmosdb"
    sqlite_path: str


@dataclass(frozen=True)
class CosmosDBConfig:
    """Azure Cosmos DB configuration for audit logs."""
    endpoint: str
    key: str
    database_name: str
    container_name: str
    partition_key_path: str


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration container."""
    openai: OpenAIConfig
    azure_ai_search: AzureAISearchConfig
    database: DatabaseConfig
    logging: LoggingConfig
    document_intelligence: DocumentIntelligenceConfig
    audit_log: AuditLogConfig
    cosmosdb: Optional[CosmosDBConfig]  # Only required when audit_log.backend == "cosmosdb"


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
        pdf_index_name=ai_search_section.get("pdf_index_name", "pdf-document-index"),
    )

    # Build Document Intelligence config
    doc_intel_section = yaml_config.get("document_intelligence", {})

    document_intelligence_config = DocumentIntelligenceConfig(
        api_key=_get_required_env("DOCUMENT_INTELLIGENCE_KEY"),
        endpoint=doc_intel_section.get("endpoint", ""),
        model_id=doc_intel_section.get("model_id", "prebuilt-layout"),
    )

    # Build Database config
    db_section = yaml_config.get("database", {})
    sqlalchemy_section = yaml_config.get("sqlalchemy", {})

    database_config = DatabaseConfig(
        path=db_section.get("path", "products.db"),
        sqlite_engine_url=sqlalchemy_section.get("sqlite_engine_url", "sqlite:///products.db"),
        document_tracking_path=db_section.get("document_tracking_path", "document_tracking.db"),
    )

    # Build Logging config
    logging_section = yaml_config.get("logging", {})

    logging_config = LoggingConfig(
        level=logging_section.get("level", "INFO"),
    )

    # Build AuditLog config
    audit_log_section = yaml_config.get("audit_log", {})
    audit_log_backend = audit_log_section.get("backend", "sqlite")

    audit_log_config = AuditLogConfig(
        backend=audit_log_backend,
        sqlite_path=audit_log_section.get("sqlite_path", "rag_logs.db"),
    )

    # Build CosmosDB config (only if backend is cosmosdb)
    cosmosdb_config: Optional[CosmosDBConfig] = None
    if audit_log_backend == "cosmosdb":
        cosmosdb_section = yaml_config.get("cosmosdb", {})
        cosmosdb_config = CosmosDBConfig(
            endpoint=_get_required_env("COSMOSDB_ENDPOINT"),
            key=_get_required_env("COSMOSDB_KEY"),
            database_name=cosmosdb_section.get("database_name", "ragchat_info"),
            container_name=cosmosdb_section.get("container_name", "ragchat_logs"),
            partition_key_path=cosmosdb_section.get("partition_key_path", "/id"),
        )

    return AppConfig(
        openai=openai_config,
        azure_ai_search=azure_ai_search_config,
        database=database_config,
        logging=logging_config,
        document_intelligence=document_intelligence_config,
        audit_log=audit_log_config,
        cosmosdb=cosmosdb_config,
    )


# Module-level singleton for convenience
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the application configuration singleton.

    Lazy-loads configuration on first access.
    Config file is selected based on APP_ENV environment variable.

    Returns:
        AppConfig: Application configuration.

    Raises:
        ConfigurationError: If required configuration is missing.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_environment() -> str:
    """Get current environment name.

    Returns:
        'dev', 'test', or 'default' based on APP_ENV.
    """
    app_env = os.environ.get("APP_ENV", "").lower()
    return app_env if app_env in ("dev", "test") else "default"


def reset_config() -> None:
    """Reset the config singleton. Useful for testing."""
    global _config
    _config = None
