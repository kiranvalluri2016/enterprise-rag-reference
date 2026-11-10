"""Runtime configuration validation for local and production modes."""
from __future__ import annotations

import os


class ConfigurationError(ValueError):
    pass


def is_production(environment: dict[str, str] | None = None) -> bool:
    environment = environment or os.environ
    return environment.get("RAG_ENV", "development").lower() == "production"


def validate_runtime_configuration(environment: dict[str, str] | None = None) -> None:
    """Fail before serving when production security dependencies are unconfigured."""
    environment = environment or os.environ
    if not is_production(environment):
        return
    required = (
        "RAG_AZURE_KEY_VAULT_URL",
        "RAG_JWT_SECRET_NAME",
        "RAG_ADMIN_TOKEN_SECRET_NAME",
        "RAG_AUDIT_POSTGRES_DSN_SECRET_NAME",
    )
    missing = [name for name in required if not environment.get(name)]
    if missing:
        raise ConfigurationError("Production configuration is incomplete")
