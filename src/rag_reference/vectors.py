"""Embedding and vector-store boundaries for governed hybrid retrieval."""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from dataclasses import dataclass
from typing import Callable, Protocol

from .core import Passage, Retriever, tokenize

LOGGER = logging.getLogger("rag_reference.vectors")


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class VectorRecord:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    embedding: tuple[float, ...]
    tenant_id: str
    corpus_version: str


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None: ...
    def search(self, embedding: list[float], tenant_id: str, corpus_version: str, top_k: int) -> list[dict]: ...


class HashEmbeddingProvider:
    """Deterministic local embedding used only for development and tests."""

    name = "deterministic-hash-v1"

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in tokenize(text):
            bucket = int.from_bytes(hashlib.blake2s(token.encode(), digest_size=4).digest(), "big") % self.dimensions
            values[bucket] += 1.0
        magnitude = math.sqrt(sum(value * value for value in values))
        return [value / magnitude for value in values] if magnitude else values


class EmbeddingCache:
    """Thread-safe content-addressed cache scoped to an embedding model version."""

    def __init__(self) -> None:
        self._entries: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def embed(self, provider: EmbeddingProvider, text: str) -> list[float]:
        key = hashlib.sha256(f"{provider.name}:{text}".encode()).hexdigest()
        with self._lock:
            cached = self._entries.get(key)
        if cached is not None:
            LOGGER.info("embedding_cache_hit provider=%s", provider.name)
            return list(cached)
        embedding = provider.embed(text)
        if len(embedding) != provider.dimensions:
            raise ValueError("Embedding provider returned an unexpected dimension")
        with self._lock:
            self._entries[key] = list(embedding)
        LOGGER.info("embedding_cache_stored provider=%s", provider.name)
        return embedding


class InMemoryVectorStore:
    """Deterministic development store; not a substitute for a durable vector DB."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], VectorRecord] = {}
        self._lock = threading.RLock()

    def upsert(self, records: list[VectorRecord]) -> None:
        with self._lock:
            self._records.update({(record.tenant_id, record.chunk_id): record for record in records})
        LOGGER.info("vector_records_upserted count=%s", len(records))

    def search(self, embedding: list[float], tenant_id: str, corpus_version: str, top_k: int) -> list[dict]:
        with self._lock:
            eligible = [record for record in self._records.values() if record.tenant_id == tenant_id and record.corpus_version == corpus_version]
        scored = []
        for record in eligible:
            score = sum(left * right for left, right in zip(embedding, record.embedding))
            if score > 0:
                scored.append({"doc_id": record.doc_id, "title": record.title, "text": record.text, "score": round(score, 6)})
        result = sorted(scored, key=lambda item: (-item["score"], item["doc_id"]))[:top_k]
        LOGGER.info("vector_search_completed result_count=%s", len(result))
        return result


class HybridRetriever:
    """Fuses lexical and semantic ranks after mandatory tenant/corpus filtering."""

    def __init__(self, passages: list[Passage], embeddings: EmbeddingProvider, store: VectorStore, tenant_id: str = "public") -> None:
        self.passages = passages
        self.lexical = Retriever(passages)
        self.embeddings = embeddings
        self.embedding_cache = EmbeddingCache()
        self.store = store
        self.tenant_id = tenant_id
        self.corpus_version = self._corpus_version(passages)
        self.store.upsert([
            VectorRecord(passage.doc_id, passage.doc_id, passage.title, passage.text,
                         tuple(self.embedding_cache.embed(embeddings, passage.text)), tenant_id, self.corpus_version)
            for passage in passages
        ])

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        lexical = self.lexical.search(question, top_k)
        semantic = self.store.search(self.embedding_cache.embed(self.embeddings, question), self.tenant_id, self.corpus_version, top_k)
        by_id: dict[str, dict] = {}
        for rank, item in enumerate(lexical, start=1):
            merged = by_id.setdefault(item["doc_id"], dict(item))
            merged["rrf_score"] = merged.get("rrf_score", 0) + 1 / (60 + rank)
        for rank, item in enumerate(semantic, start=1):
            merged = by_id.setdefault(item["doc_id"], dict(item))
            merged["rrf_score"] = merged.get("rrf_score", 0) + 1 / (60 + rank)
        result = [
            {key: value for key, value in item.items() if key != "rrf_score"} | {"score": round(item["rrf_score"], 6)}
            for item in sorted(by_id.values(), key=lambda item: (-item["rrf_score"], item["doc_id"]))[:top_k]
        ]
        LOGGER.info("hybrid_retrieval_completed lexical_count=%s semantic_count=%s result_count=%s", len(lexical), len(semantic), len(result))
        return result

    @staticmethod
    def _corpus_version(passages: list[Passage]) -> str:
        material = "\n".join(f"{passage.doc_id}\0{passage.text}" for passage in sorted(passages, key=lambda item: item.doc_id))
        return hashlib.sha256(material.encode()).hexdigest()[:16]


class PgVectorStore:
    """Optional pgvector adapter. Supply a pooled DB-API connection factory in production."""

    def __init__(self, connect: Callable[[], object]) -> None:
        self._connect = connect

    def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        tenant_ids = {record.tenant_id for record in records}
        if len(tenant_ids) != 1:
            raise ValueError("A pgvector upsert must contain one tenant")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT set_config('app.tenant_id', %s, true)", (records[0].tenant_id,))
            cursor.executemany(
                """INSERT INTO rag_chunks (chunk_id, doc_id, title, content, tenant_id, corpus_version, embedding_model, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET title = EXCLUDED.title, content = EXCLUDED.content,
                   corpus_version = EXCLUDED.corpus_version,
                   embedding_model = EXCLUDED.embedding_model, embedding = EXCLUDED.embedding""",
                [(record.chunk_id, record.doc_id, record.title, record.text, record.tenant_id, record.corpus_version,
                  "configured-at-deployment", "[" + ",".join(map(str, record.embedding)) + "]") for record in records],
            )
        LOGGER.info("pgvector_records_upserted count=%s", len(records))

    def search(self, embedding: list[float], tenant_id: str, corpus_version: str, top_k: int) -> list[dict]:
        vector = "[" + ",".join(map(str, embedding)) + "]"
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
            cursor.execute(
                """SELECT doc_id, title, content, 1 - (embedding <=> %s::vector) AS score
                   FROM rag_chunks WHERE tenant_id = %s AND corpus_version = %s AND deleted_at IS NULL
                   ORDER BY embedding <=> %s::vector LIMIT %s""",
                (vector, tenant_id, corpus_version, vector, top_k),
            )
            result = [{"doc_id": row[0], "title": row[1], "text": row[2], "score": round(float(row[3]), 6)} for row in cursor.fetchall()]
        LOGGER.info("pgvector_search_completed result_count=%s", len(result))
        return result

    def soft_delete_document(self, tenant_id: str, doc_id: str) -> int:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant_id,))
            cursor.execute("UPDATE rag_chunks SET deleted_at = now(), updated_at = now() WHERE doc_id = %s AND deleted_at IS NULL", (doc_id,))
            deleted = cursor.rowcount
        LOGGER.info("pgvector_document_soft_deleted row_count=%s", deleted)
        return deleted
