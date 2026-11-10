"""Secret-resolution boundary; values are never logged or returned from this module."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Protocol

LOGGER = logging.getLogger("rag_reference.secrets")


class SecretResolver(Protocol):
    def get(self, name: str) -> str | None: ...


class EnvironmentSecretResolver:
    def get(self, name: str) -> str | None:
        return os.environ.get(name)


class AzureKeyVaultSecretResolver:
    """Managed-identity Key Vault resolver with a bounded in-process value cache."""

    def __init__(self, vault_url: str, cache_ttl_seconds: int = 300) -> None:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
        except ImportError as exc:
            raise RuntimeError("Install the azure optional dependency to use Azure Key Vault") from exc
        self._client = SecretClient(vault_url=vault_url, credential=DefaultAzureCredential(), logging_enable=False)
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> str | None:
        with self._lock:
            cached = self._cache.get(name)
            if cached and cached[1] > time.monotonic():
                LOGGER.info("secret_cache_hit")
                return cached[0]
        try:
            value = self._client.get_secret(name).value
        except Exception as exc:
            LOGGER.warning("key_vault_secret_unavailable")
            raise RuntimeError("Required Azure Key Vault secret is unavailable") from exc
        if value is None:
            return None
        with self._lock:
            self._cache[name] = (value, time.monotonic() + self._cache_ttl_seconds)
        LOGGER.info("key_vault_secret_resolved")
        return value


def configured_secret_resolver() -> SecretResolver:
    vault_url = os.environ.get("RAG_AZURE_KEY_VAULT_URL")
    if vault_url:
        return AzureKeyVaultSecretResolver(vault_url)
    return EnvironmentSecretResolver()
