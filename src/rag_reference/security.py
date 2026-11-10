"""Small, dependency-free security primitives for the reference service."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import math
import threading
import time
from dataclasses import dataclass

LOGGER = logging.getLogger("rag_reference.security")


class AuthenticationError(ValueError):
    pass


class AuthorizationError(ValueError):
    pass


class AuditVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: frozenset[str]


class JWTAuthenticator:
    """HS256 JWT verifier. Production callers should source the secret from a vault."""

    def __init__(self, secret: str, issuer: str | None = None, audience: str | None = None,
                 require_expiration: bool = False, require_issuer_audience: bool = False) -> None:
        if len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters")
        if require_issuer_audience and (not issuer or not audience):
            raise ValueError("Issuer and audience are required for strict JWT validation")
        self.secret = secret.encode()
        self.issuer = issuer
        self.audience = audience
        self.require_expiration = require_expiration

    def authenticate(self, authorization: str | None) -> Principal:
        if not authorization or not authorization.startswith("Bearer "):
            LOGGER.warning("authentication_rejected reason=missing_or_invalid_header")
            raise AuthenticationError("Bearer token is required")
        token = authorization[7:]
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthenticationError("Malformed bearer token")
        header = self._decode(parts[0])
        claims = self._decode(parts[1])
        if header.get("alg") != "HS256":
            raise AuthenticationError("Unsupported token algorithm")
        expected = hmac.new(self.secret, f"{parts[0]}.{parts[1]}".encode(), hashlib.sha256).digest()
        signature = self._decode_bytes(parts[2])
        if not hmac.compare_digest(expected, signature):
            LOGGER.warning("authentication_rejected reason=signature")
            raise AuthenticationError("Invalid bearer token")
        now = time.time()
        if self.require_expiration and "exp" not in claims:
            raise AuthenticationError("Bearer token expiration is required")
        for field, comparison in (("exp", lambda value: now >= value), ("nbf", lambda value: now < value)):
            value = claims.get(field)
            if value is not None:
                if type(value) not in (int, float) or not math.isfinite(value) or comparison(value):
                    raise AuthenticationError("Expired or inactive bearer token")
        if self.issuer is not None and claims.get("iss") != self.issuer:
            raise AuthenticationError("Invalid token issuer")
        if self.audience is not None:
            audiences = claims.get("aud")
            if isinstance(audiences, str):
                audiences = [audiences]
            if not isinstance(audiences, list) or self.audience not in audiences:
                raise AuthenticationError("Invalid token audience")
        subject, tenant_id, roles = claims.get("sub"), claims.get("tenant_id"), claims.get("roles", [])
        if not isinstance(subject, str) or not subject or not isinstance(tenant_id, str) or not tenant_id:
            raise AuthenticationError("Token identity claims are invalid")
        if not isinstance(roles, list) or not all(isinstance(role, str) and role for role in roles):
            raise AuthenticationError("Token role claims are invalid")
        LOGGER.info("authentication_accepted tenant_context=true role_count=%s", len(roles))
        return Principal(subject, tenant_id, frozenset(roles))

    @staticmethod
    def _decode(segment: str) -> dict:
        try:
            value = json.loads(JWTAuthenticator._decode_bytes(segment))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthenticationError("Malformed bearer token") from exc
        if not isinstance(value, dict):
            raise AuthenticationError("Malformed bearer token")
        return value

    @staticmethod
    def _decode_bytes(segment: str) -> bytes:
        try:
            return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
        except (binascii.Error, ValueError) as exc:
            raise AuthenticationError("Malformed bearer token") from exc


class FixedWindowRateLimiter:
    """Per-subject fixed-window limiter; replace with shared storage across replicas."""

    def __init__(self, limit: int = 60, window_seconds: int = 60) -> None:
        self.limit, self.window_seconds = limit, window_seconds
        self._entries: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = time.monotonic()
            started, used = self._entries.get(key, (now, 0))
            if now - started >= self.window_seconds:
                started, used = now, 0
            if used >= self.limit:
                self._entries[key] = (started, used)
                LOGGER.warning("rate_limit_rejected")
                return False
            self._entries[key] = (started, used + 1)
            LOGGER.info("rate_limit_accepted remaining=%s", self.limit - used - 1)
            return True


class AuditLedger:
    """Hash-chained, in-process audit witness; durable storage is a deployment concern."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._last_hash = "0" * 64
        self._lock = threading.Lock()

    def append(self, event_type: str, request_id: str, principal: Principal, outcome: str, metadata: dict | None = None) -> dict:
        with self._lock:
            event = {"sequence": len(self.events) + 1, "event_type": event_type, "request_id": request_id,
                     "subject": principal.subject, "tenant_id": principal.tenant_id, "outcome": outcome,
                     "metadata": metadata or {}, "previous_hash": self._last_hash}
            encoded = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
            event["event_hash"] = hashlib.sha256(encoded).hexdigest()
            self.events.append(event)
            self._last_hash = event["event_hash"]
            LOGGER.info("audit_event_appended event_type=%s sequence=%s outcome=%s", event_type, event["sequence"], outcome)
            return dict(event)

    def verify(self) -> None:
        verify_audit_chain(self.events)


class PostgresAuditLedger:
    """Append-only, hash-chained PostgreSQL audit ledger using a pooled connection factory."""

    def __init__(self, connect) -> None:
        self._connect = connect

    def append(self, event_type: str, request_id: str, principal: Principal, outcome: str, metadata: dict | None = None) -> dict:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext('rag_audit_events_chain'))")
            cursor.execute("SELECT sequence, event_hash FROM rag_audit_events ORDER BY sequence DESC LIMIT 1")
            previous = cursor.fetchone()
            sequence, previous_hash = (previous[0] + 1, previous[1]) if previous else (1, "0" * 64)
            event = {"sequence": sequence, "event_type": event_type, "request_id": request_id,
                     "subject": principal.subject, "tenant_id": principal.tenant_id, "outcome": outcome,
                     "metadata": metadata or {}, "previous_hash": previous_hash}
            event_hash = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            cursor.execute(
                """INSERT INTO rag_audit_events
                   (sequence, event_type, request_id, subject, tenant_id, outcome, metadata, previous_hash, event_hash)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)""",
                (sequence, event_type, request_id, principal.subject, principal.tenant_id, outcome,
                 json.dumps(metadata or {}, separators=(",", ":")), previous_hash, event_hash),
            )
        event["event_hash"] = event_hash
        LOGGER.info("postgres_audit_event_appended event_type=%s sequence=%s outcome=%s", event_type, sequence, outcome)
        return event

    def verify(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT sequence, event_type, request_id, subject, tenant_id, outcome, metadata, previous_hash, event_hash FROM rag_audit_events ORDER BY sequence")
            rows = cursor.fetchall()
        events = []
        for row in rows:
            metadata = json.loads(row[6]) if isinstance(row[6], str) else row[6]
            events.append({"sequence": row[0], "event_type": row[1], "request_id": row[2], "subject": row[3],
                           "tenant_id": row[4], "outcome": row[5], "metadata": metadata,
                           "previous_hash": row[7], "event_hash": row[8]})
        verify_audit_chain(events)


def verify_audit_chain(events: list[dict]) -> None:
    previous_hash = "0" * 64
    for expected_sequence, stored in enumerate(events, start=1):
        if stored.get("sequence") != expected_sequence or stored.get("previous_hash") != previous_hash:
            raise AuditVerificationError("Audit sequence or predecessor is invalid")
        event = {key: value for key, value in stored.items() if key != "event_hash"}
        expected_hash = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if not hmac.compare_digest(expected_hash, str(stored.get("event_hash", ""))):
            raise AuditVerificationError("Audit event hash is invalid")
        previous_hash = expected_hash
    LOGGER.info("audit_chain_verified event_count=%s", len(events))


def postgres_audit_ledger(dsn: str):
    """Build a pooled audit ledger lazily; requires the optional postgres extra."""
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as exc:
        raise RuntimeError("Install the postgres optional dependency to use PostgreSQL audit storage") from exc
    pool = ConnectionPool(conninfo=dsn, min_size=1, max_size=5, open=True)
    return PostgresAuditLedger(pool.connection)
