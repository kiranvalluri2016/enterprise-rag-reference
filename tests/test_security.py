import base64
import hashlib
import hmac
import json
import time
import unittest

from rag_reference.providers import CircuitBreakerProvider, GenerationRequest, HTTPJSONProvider, ProviderError
from rag_reference.security import AuditLedger, AuditVerificationError, AuthenticationError, FixedWindowRateLimiter, JWTAuthenticator, PostgresAuditLedger, Principal
from rag_reference.secrets import EnvironmentSecretResolver


def token(secret, claims, header=None):
    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=").decode()
    first, second = encode(header or {"alg": "HS256", "typ": "JWT"}), encode(claims)
    signature = base64.urlsafe_b64encode(hmac.new(secret.encode(), f"{first}.{second}".encode(), hashlib.sha256).digest()).rstrip(b"=").decode()
    return f"{first}.{second}.{signature}"


class SecurityTests(unittest.TestCase):
    secret = "a" * 32

    def test_jwt_validates_identity_and_roles(self):
        authenticator = JWTAuthenticator(self.secret, "issuer", "audience")
        signed = token(self.secret, {"sub": "user-1", "tenant_id": "public", "roles": ["query"], "exp": time.time() + 60, "iss": "issuer", "aud": "audience"})
        principal = authenticator.authenticate(f"Bearer {signed}")
        self.assertEqual((principal.subject, principal.tenant_id, principal.roles), ("user-1", "public", frozenset({"query"})))

    def test_jwt_rejects_expired_tampered_and_unsafe_algorithms(self):
        authenticator = JWTAuthenticator(self.secret)
        expired = token(self.secret, {"sub": "user", "tenant_id": "public", "roles": [], "exp": time.time() - 1})
        tampered = token(self.secret, {"sub": "user", "tenant_id": "public", "roles": []})[:-1] + "x"
        wrong_algorithm = token(self.secret, {"sub": "user", "tenant_id": "public", "roles": []}, {"alg": "none"})
        for value in (expired, tampered, wrong_algorithm):
            with self.subTest(value=value):
                with self.assertRaises(AuthenticationError):
                    authenticator.authenticate(f"Bearer {value}")

    def test_strict_jwt_requires_expiration_issuer_and_audience(self):
        with self.assertRaises(ValueError):
            JWTAuthenticator(self.secret, require_issuer_audience=True)
        authenticator = JWTAuthenticator(self.secret, "issuer", "audience", require_expiration=True, require_issuer_audience=True)
        missing_exp = token(self.secret, {"sub": "user", "tenant_id": "public", "roles": [], "iss": "issuer", "aud": "audience"})
        with self.assertRaises(AuthenticationError):
            authenticator.authenticate(f"Bearer {missing_exp}")

    def test_rate_limit_and_hash_chained_audit_events(self):
        limiter = FixedWindowRateLimiter(limit=2, window_seconds=60)
        self.assertTrue(limiter.allow("user"))
        self.assertTrue(limiter.allow("user"))
        self.assertFalse(limiter.allow("user"))
        ledger = AuditLedger()
        principal = Principal("user", "public", frozenset({"query"}))
        first = ledger.append("query", "one", principal, "success")
        second = ledger.append("query", "two", principal, "success")
        self.assertEqual(second["previous_hash"], first["event_hash"])
        ledger.verify()
        ledger.events[1]["outcome"] = "tampered"
        with self.assertRaises(AuditVerificationError):
            ledger.verify()

    def test_circuit_breaker_opens_after_failures(self):
        class FailingProvider:
            name = "failing"
            def ready(self): return True
            def generate(self, request): raise ProviderError("failed")
        provider = CircuitBreakerProvider(FailingProvider(), failure_threshold=2, cooldown_seconds=60)
        for _ in range(3):
            with self.assertRaises(ProviderError):
                provider.generate(GenerationRequest("prompt", "request"))

    def test_provider_uses_injected_secret_resolver(self):
        class Resolver:
            def get(self, name): return "key" if name == "provider-key" else None
        provider = HTTPJSONProvider("test", "chat-completions", "https://example.invalid", "model", "provider-key", secrets=Resolver())
        self.assertTrue(provider.ready())

    def test_openai_responses_parser_handles_top_level_and_nested_text(self):
        provider = HTTPJSONProvider("test", "openai-responses", "https://example.invalid", "model", "provider-key")
        self.assertEqual(provider._text({"output_text": "top-level"}), "top-level")
        nested = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "nested "}, {"type": "output_text", "text": "text"}]}]}
        self.assertEqual(provider._text(nested), "nested text")

    def test_postgres_audit_ledger_writes_a_hash_chained_event(self):
        class Cursor:
            def __init__(self): self.commands = []; self._fetch = None
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, sql, params=None):
                self.commands.append((sql, params))
                self._fetch = None
            def fetchone(self): return self._fetch
        class Connection:
            def __init__(self): self.cursor_value = Cursor()
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def cursor(self): return self.cursor_value
        connection = Connection()
        ledger = PostgresAuditLedger(lambda: connection)
        event = ledger.append("query", "request", Principal("user", "public", frozenset({"query"})), "success")
        self.assertEqual(event["sequence"], 1)
        self.assertEqual(len(event["event_hash"]), 64)
        self.assertTrue(any("INSERT INTO rag_audit_events" in sql for sql, _ in connection.cursor_value.commands))

    def test_environment_secret_resolver_returns_only_named_value(self):
        self.assertIsNone(EnvironmentSecretResolver().get("MISSING_TEST_SECRET"))

