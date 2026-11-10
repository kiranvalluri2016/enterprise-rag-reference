"""Pluggable, server-side LLM provider adapters.

Adapters intentionally receive only a prepared grounded prompt. They never read
HTTP headers or client-supplied provider names, and they keep credentials out of
responses and logs.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .secrets import EnvironmentSecretResolver, SecretResolver

LOGGER = logging.getLogger("rag_reference.providers")


class ProviderError(RuntimeError):
    """A safe-to-classify provider failure; the underlying body is not exposed."""


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    request_id: str


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider_request_id: str | None = None


class GenerationProvider(Protocol):
    name: str

    def ready(self) -> bool: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class ProviderRegistry:
    """Thread-safe provider extension point and active-routing snapshot."""

    def __init__(self) -> None:
        self._providers: dict[str, GenerationProvider] = {}
        self._active: str | None = None
        self._revision = 0
        self._lock = threading.RLock()

    def register(self, provider: GenerationProvider) -> None:
        if not provider.name or provider.name in self._providers:
            raise ValueError("Provider name must be unique and nonempty")
        with self._lock:
            self._providers[provider.name] = provider

    def activate(self, name: str) -> int:
        with self._lock:
            provider = self._providers.get(name)
            if provider is None:
                raise ValueError("Unknown provider")
            if not provider.ready():
                raise ValueError("Provider is not configured")
            self._active = name
            self._revision += 1
            LOGGER.info("provider_activated provider=%s routing_revision=%s", name, self._revision)
            return self._revision

    def active(self) -> tuple[GenerationProvider, str, int]:
        with self._lock:
            if self._active is None:
                raise ProviderError("No active provider")
            provider = self._providers[self._active]
            return provider, self._active, self._revision

    def status(self) -> dict:
        with self._lock:
            return {
                "active_provider": self._active,
                "routing_revision": self._revision,
                "providers": {name: {"ready": provider.ready()} for name, provider in sorted(self._providers.items())},
            }


class CircuitBreakerProvider:
    """Stops repeatedly failing provider calls for a bounded cooldown period."""

    def __init__(self, provider: GenerationProvider, failure_threshold: int = 3, cooldown_seconds: int = 30) -> None:
        self.provider = provider
        self.name = provider.name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    def ready(self) -> bool:
        return self.provider.ready()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            if time.monotonic() < self._open_until:
                LOGGER.warning("provider_circuit_open provider=%s", self.name)
                raise ProviderError("Provider circuit is open")
        try:
            result = self.provider.generate(request)
        except ProviderError:
            with self._lock:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._open_until = time.monotonic() + self.cooldown_seconds
                    LOGGER.error("provider_circuit_tripped provider=%s", self.name)
            raise
        with self._lock:
            self._failures = 0
        LOGGER.info("provider_generation_succeeded provider=%s", self.name)
        return result


class HTTPJSONProvider:
    """A minimal adapter for a named provider's JSON completion endpoint."""

    def __init__(self, name: str, kind: str, endpoint: str, model: str, secret_name: str,
                 api_key_header: str = "Authorization", extra_headers: dict[str, str] | None = None,
                 secrets: SecretResolver | None = None):
        self.name = name
        self.kind = kind
        self.endpoint = endpoint
        self.model = model
        self.secret_name = secret_name
        self.api_key_header = api_key_header
        self.extra_headers = extra_headers or {}
        self.secrets = secrets or EnvironmentSecretResolver()

    def ready(self) -> bool:
        return bool(self.endpoint and self.model and self.secrets.get(self.secret_name))

    def generate(self, request: GenerationRequest) -> GenerationResult:
        key = self.secrets.get(self.secret_name)
        if not self.ready() or key is None:
            raise ProviderError("Provider is not configured")
        payload = self._payload(request.prompt)
        headers = {"Content-Type": "application/json", "X-Client-Request-Id": request.request_id}
        headers.update(self.extra_headers)
        headers[self.api_key_header] = key if self.api_key_header == "api-key" else f"Bearer {key}"
        outbound = Request(self.endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urlopen(outbound, timeout=float(os.environ.get("RAG_PROVIDER_TIMEOUT_SECONDS", "20"))) as response:
                data = json.loads(response.read())
                return GenerationResult(self._text(data), response.headers.get("x-request-id"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            LOGGER.warning("provider_generation_failed provider=%s", self.name)
            raise ProviderError("Provider request failed") from exc

    def _payload(self, prompt: str) -> dict:
        if self.kind == "openai-responses":
            return {"model": self.model, "input": prompt, "store": False}
        if self.kind == "anthropic-messages":
            return {"model": self.model, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]}
        return {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}

    def _text(self, data: dict) -> str:
        if self.kind == "openai-responses":
            return str(data.get("output_text", ""))
        if self.kind == "anthropic-messages":
            return "".join(str(item.get("text", "")) for item in data.get("content", []) if item.get("type") == "text")
        return str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))


def configured_registry(secrets: SecretResolver | None = None) -> ProviderRegistry:
    """Build the supported provider set; each is inactive until explicitly routed."""
    registry = ProviderRegistry()
    secrets = secrets or EnvironmentSecretResolver()
    registry.register(CircuitBreakerProvider(HTTPJSONProvider("openai", "openai-responses", os.environ.get("RAG_OPENAI_ENDPOINT", "https://api.openai.com/v1/responses"), os.environ.get("RAG_OPENAI_MODEL", ""), os.environ.get("RAG_OPENAI_API_KEY_SECRET_NAME", "OPENAI_API_KEY"), secrets=secrets)))
    registry.register(CircuitBreakerProvider(HTTPJSONProvider("azure-openai", "chat-completions", os.environ.get("RAG_AZURE_OPENAI_ENDPOINT", ""), os.environ.get("RAG_AZURE_OPENAI_MODEL", ""), os.environ.get("RAG_AZURE_OPENAI_API_KEY_SECRET_NAME", "AZURE_OPENAI_API_KEY"), "api-key", secrets=secrets)))
    registry.register(CircuitBreakerProvider(HTTPJSONProvider("anthropic", "anthropic-messages", os.environ.get("RAG_ANTHROPIC_ENDPOINT", "https://api.anthropic.com/v1/messages"), os.environ.get("RAG_ANTHROPIC_MODEL", ""), os.environ.get("RAG_ANTHROPIC_API_KEY_SECRET_NAME", "ANTHROPIC_API_KEY"), "x-api-key", {"anthropic-version": "2023-06-01"}, secrets)))
    registry.register(CircuitBreakerProvider(HTTPJSONProvider("openai-compatible", "chat-completions", os.environ.get("RAG_OPENAI_COMPATIBLE_ENDPOINT", ""), os.environ.get("RAG_OPENAI_COMPATIBLE_MODEL", ""), os.environ.get("RAG_OPENAI_COMPATIBLE_API_KEY_SECRET_NAME", "RAG_OPENAI_COMPATIBLE_API_KEY"), secrets=secrets)))
    return registry
