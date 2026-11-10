import unittest

from rag_reference.core import Passage
from rag_reference.vectors import EmbeddingCache, HashEmbeddingProvider, HybridRetriever, InMemoryVectorStore


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.passages = [
            Passage("retries", "Reliability", "Retries require idempotency for safe jobs."),
            Passage("search", "Retrieval", "Semantic retrieval finds conceptually related passages."),
        ]
        self.provider = HashEmbeddingProvider()
        self.store = InMemoryVectorStore()
        self.retriever = HybridRetriever(self.passages, self.provider, self.store, tenant_id="tenant-a")

    def test_hybrid_retrieval_preserves_citable_passages(self):
        result = self.retriever.search("idempotency retries", top_k=2)
        self.assertEqual(result[0]["doc_id"], "retries")
        self.assertIn("text", result[0])
        self.assertTrue(result[0]["score"] > 0)

    def test_store_enforces_tenant_and_corpus_filters(self):
        query = self.provider.embed("idempotency")
        self.assertEqual(self.store.search(query, "another-tenant", self.retriever.corpus_version, 3), [])
        self.assertEqual(self.store.search(query, "tenant-a", "old-corpus", 3), [])

    def test_embedding_cache_scopes_entries_by_provider_name(self):
        cache = EmbeddingCache()
        first = cache.embed(self.provider, "same text")
        second = cache.embed(self.provider, "same text")
        self.assertEqual(first, second)

    def test_corpus_version_changes_when_document_content_changes(self):
        changed = [Passage("retries", "Reliability", "Retries need a different policy.")]
        self.assertNotEqual(self.retriever.corpus_version, HybridRetriever(changed, self.provider, InMemoryVectorStore()).corpus_version)
