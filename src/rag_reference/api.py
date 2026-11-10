"""HTTP JSON API using Python's standard library (no cloud keys needed)."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .config import is_production, validate_runtime_configuration
from .core import Retriever, load_documents
from .orchestration import Orchestrator, ResponseCache
from .providers import ProviderError, configured_registry
from .secrets import configured_secret_resolver
from .security import AuditLedger, AuthenticationError, AuthorizationError, FixedWindowRateLimiter, JWTAuthenticator, Principal, postgres_audit_ledger
from .vectors import HashEmbeddingProvider, HybridRetriever, InMemoryVectorStore

DEFAULT_DOCS = Path("sample_docs")
MAX_BODY_BYTES = 16_384
DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 20
LOGGER = logging.getLogger("rag_reference")


class APIError(Exception):
    """A client-safe HTTP error."""

    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message


def configure_logging(level_name: str | None = None) -> None:
    """Configure application logging without adding duplicate handlers."""
    configured_level = (level_name or os.environ.get("RAG_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, configured_level, None)
    if not isinstance(level, int):
        configured_level = "INFO"
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    LOGGER.setLevel(level)
    if configured_level != (level_name or os.environ.get("RAG_LOG_LEVEL", "INFO")).upper():
        LOGGER.warning("Invalid RAG_LOG_LEVEL; using INFO")


def create_handler(orchestrator: Orchestrator, authenticator: JWTAuthenticator | None = None,
                   rate_limiter: FixedWindowRateLimiter | None = None, audit: AuditLedger | None = None,
                   admin_token: str | None = None, production: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EnterpriseRAGReference/0.1"

        def log_message(self, format: str, *args) -> None:
            """Suppress BaseHTTPRequestHandler's duplicate, unstructured logs."""

        def _begin_request(self) -> None:
            self.request_id = uuid.uuid4().hex
            self.response_status: int | None = None
            self.request_started = time.perf_counter()

        def _finish_request(self) -> None:
            duration_ms = (time.perf_counter() - self.request_started) * 1000
            LOGGER.info(
                "http_request method=%s route=%s status=%s duration_ms=%.2f request_id=%s",
                self.command, urlsplit(self.path).path,
                self.response_status if self.response_status is not None else "write_failed",
                duration_ms, self.request_id,
            )

        def _run(self, action) -> None:
            self._begin_request()
            try:
                action()
            except APIError as error:
                self._send_error_safely(error.status, error.code, error.message)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                LOGGER.warning("response_write_failed request_id=%s", self.request_id)
            except Exception as exc:
                if os.environ.get("RAG_LOG_TRACEBACKS", "false").lower() == "true":
                    LOGGER.exception("unexpected_server_exception request_id=%s exception_type=%s", self.request_id, type(exc).__name__)
                else:
                    LOGGER.error("unexpected_server_exception request_id=%s exception_type=%s", self.request_id, type(exc).__name__)
                self._send_error_safely(500, "INTERNAL_ERROR", "An internal server error occurred")
            finally:
                self._finish_request()

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.response_status = status
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-ID", self.request_id)
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, code: str, message: str) -> None:
            self._send(status, {"error": {"code": code, "message": message}, "request_id": self.request_id})

        def _send_error_safely(self, status: int, code: str, message: str) -> None:
            try:
                self._send_error(status, code, message)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                LOGGER.warning("response_write_failed request_id=%s", self.request_id)

        def _content_length(self) -> int:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise APIError(400, "INVALID_CONTENT_LENGTH", "Content-Length is required")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise APIError(400, "INVALID_CONTENT_LENGTH", "Content-Length must be an integer") from exc
            if length < 0:
                raise APIError(400, "INVALID_CONTENT_LENGTH", "Content-Length must not be negative")
            if length == 0:
                raise APIError(400, "EMPTY_REQUEST_BODY", "Request body must not be empty")
            if length > MAX_BODY_BYTES:
                raise APIError(413, "PAYLOAD_TOO_LARGE", "Request body exceeds the supported size")
            return length

        def _read_json(self) -> dict:
            length = self._content_length()
            raw_body = self.rfile.read(length)
            if len(raw_body) != length:
                raise APIError(400, "INVALID_REQUEST", "Request body is incomplete")
            try:
                data = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise APIError(400, "MALFORMED_JSON", "Request body must be valid JSON") from exc
            if not isinstance(data, dict):
                raise APIError(400, "INVALID_REQUEST", "Request body must be a JSON object")
            return data

        def _read_query(self) -> tuple[str, int]:
            data = self._read_json()
            if "question" not in data:
                raise APIError(400, "MISSING_QUESTION", "Question is required")
            question = data["question"]
            if not isinstance(question, str):
                raise APIError(400, "INVALID_QUESTION", "Question must be a string")
            if not question.strip():
                raise APIError(400, "INVALID_QUESTION", "Question must not be empty")
            top_k = data.get("top_k", DEFAULT_TOP_K)
            if type(top_k) is not int:
                raise APIError(400, "INVALID_TOP_K", "top_k must be an integer")
            if not MIN_TOP_K <= top_k <= MAX_TOP_K:
                raise APIError(400, "INVALID_TOP_K", "top_k must be between 1 and 20")
            return question, top_k

        def _principal(self, role: str | None = None) -> Principal:
            if authenticator is None:
                return Principal("anonymous", "public", frozenset({"query", "admin"}))
            try:
                principal = authenticator.authenticate(self.headers.get("Authorization"))
            except AuthenticationError as exc:
                raise APIError(401, "AUTHENTICATION_REQUIRED", "Valid authentication is required") from exc
            if role is not None and role not in principal.roles:
                raise APIError(403, "CAPABILITY_NOT_ENTITLED", "Required capability is not granted")
            return principal

        def _readiness(self) -> tuple[bool, dict]:
            provider_status = orchestrator.providers.status()
            active = provider_status["active_provider"]
            active_ready = bool(active and provider_status["providers"][active]["ready"])
            issues = []
            if not active_ready:
                issues.append("ACTIVE_PROVIDER_UNAVAILABLE")
            if production and authenticator is None:
                issues.append("AUTHENTICATION_UNAVAILABLE")
            if production and isinstance(audit, AuditLedger):
                issues.append("DURABLE_AUDIT_UNAVAILABLE")
            return not issues, {"status": "ready" if not issues else "not_ready", "issues": issues,
                                "providers": provider_status}

        def do_GET(self) -> None:
            def action() -> None:
                if urlsplit(self.path).path == "/health":
                    self._send(200, {"status": "ok", "documents": len(orchestrator.retriever.passages), "providers": orchestrator.providers.status()})
                elif urlsplit(self.path).path == "/live":
                    self._send(200, {"status": "live"})
                elif urlsplit(self.path).path == "/ready":
                    ready, payload = self._readiness()
                    self._send(200 if ready else 503, payload)
                else:
                    raise APIError(404, "NOT_FOUND", "Endpoint was not found")
            self._run(action)

        def do_POST(self) -> None:
            def action() -> None:
                if urlsplit(self.path).path != "/query":
                    raise APIError(404, "NOT_FOUND", "Endpoint was not found")
                principal = self._principal("query")
                if rate_limiter is not None and not rate_limiter.allow(principal.subject):
                    raise APIError(429, "RATE_LIMITED", "Request limit exceeded")
                question, top_k = self._read_query()
                try:
                    result = orchestrator.query(question, top_k, self.request_id, principal.tenant_id)
                except ProviderError:
                    raise APIError(503, "GENERATION_UNAVAILABLE", "The active generation provider is unavailable")
                except PermissionError:
                    raise APIError(403, "CAPABILITY_NOT_ENTITLED", "Tenant is not available")
                if audit is not None:
                    audit.append("query", self.request_id, principal, result["status"], {"provider": result.get("governance", {}).get("provider")})
                self._send(200, result)
            self._run(action)

        def do_PUT(self) -> None:
            def action() -> None:
                if urlsplit(self.path).path != "/admin/providers/active":
                    raise APIError(404, "NOT_FOUND", "Endpoint was not found")
                principal = self._principal("admin")
                token = admin_token if admin_token is not None else os.environ.get("RAG_ADMIN_TOKEN")
                supplied = self.headers.get("X-Admin-Token", "")
                if not token or not compare_digest(supplied, token):
                    raise APIError(403, "FORBIDDEN", "Administrative authorization is required")
                provider = self._read_json().get("provider")
                if not isinstance(provider, str):
                    raise APIError(400, "INVALID_PROVIDER", "Provider must be a string")
                try:
                    revision = orchestrator.providers.activate(provider)
                except ValueError:
                    raise APIError(400, "INVALID_PROVIDER", "Provider is unknown or not configured")
                if audit is not None:
                    audit.append("provider_routing_changed", self.request_id, principal, "success", {"provider": provider, "routing_revision": revision})
                self._send(200, {"active_provider": provider, "routing_revision": revision})

            self._run(action)

        def _method_not_allowed(self) -> None:
            def action() -> None:
                raise APIError(405, "METHOD_NOT_ALLOWED", "HTTP method is not supported")

            self._run(action)

        do_DELETE = _method_not_allowed
        do_PATCH = _method_not_allowed

    return Handler


def main() -> None:
    configure_logging()
    validate_runtime_configuration()
    docs = Path(os.environ.get("RAG_DOCS_DIR", str(DEFAULT_DOCS))).resolve()
    try:
        secrets = configured_secret_resolver()
        passages = load_documents(docs)
        retriever = HybridRetriever(passages, HashEmbeddingProvider(), InMemoryVectorStore())
        orchestrator = Orchestrator(retriever, configured_registry(secrets), ResponseCache())
    except Exception as exc:
        if os.environ.get("RAG_LOG_TRACEBACKS", "false").lower() == "true":
            LOGGER.exception("document_loading_failed directory=%s exception_type=%s", docs, type(exc).__name__)
        else:
            LOGGER.error("document_loading_failed directory=%s exception_type=%s", docs, type(exc).__name__)
        raise
    host = os.environ.get("RAG_HOST", "127.0.0.1")
    port = int(os.environ.get("RAG_PORT", "8000"))
    LOGGER.info("documents_loaded count=%s directory=%s", len(passages), docs)
    LOGGER.info("server_starting host=%s port=%s", host, port)
    secret = secrets.get(os.environ.get("RAG_JWT_SECRET_NAME", "RAG_JWT_SECRET"))
    production = is_production()
    authenticator = JWTAuthenticator(
        secret, os.environ.get("RAG_JWT_ISSUER"), os.environ.get("RAG_JWT_AUDIENCE"),
        require_expiration=production, require_issuer_audience=production,
    ) if secret else None
    limiter = FixedWindowRateLimiter(int(os.environ.get("RAG_RATE_LIMIT", "60")))
    audit_dsn = secrets.get(os.environ.get("RAG_AUDIT_POSTGRES_DSN_SECRET_NAME", "RAG_AUDIT_POSTGRES_DSN"))
    audit = postgres_audit_ledger(audit_dsn) if audit_dsn else AuditLedger()
    admin_token = secrets.get(os.environ.get("RAG_ADMIN_TOKEN_SECRET_NAME", "RAG_ADMIN_TOKEN"))
    ThreadingHTTPServer((host, port), create_handler(orchestrator, authenticator, limiter, audit, admin_token, production)).serve_forever()


if __name__ == "__main__":
    main()
