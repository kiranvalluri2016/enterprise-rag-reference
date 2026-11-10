import unittest

from rag_reference.core import Passage, Retriever
from rag_reference.orchestration import Orchestrator, ResponseCache
from rag_reference.providers import GenerationResult, ProviderError, ProviderRegistry


class Provider:
    def __init__(self, name, text, ready=True):
        self.name, self.text, self.is_ready, self.calls = name, text, ready, 0

    def ready(self):
        return self.is_ready

    def generate(self, request):
        self.calls += 1
        return GenerationResult(self.text, "provider-request-id")


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever([Passage("source-1", "Source", "Retries need idempotency.")])
        self.registry = ProviderRegistry()
        self.good = Provider("good", "Use idempotency for retries. [source-1]")
        self.registry.register(self.good)
        self.registry.activate("good")
        self.orchestrator = Orchestrator(self.retriever, self.registry, ResponseCache(ttl_seconds=60))

    def test_grounded_generation_and_cache(self):
        first = self.orchestrator.query("How should retries work?", 3, "one")
        second = self.orchestrator.query("How should retries work?", 3, "two")
        self.assertEqual(first["status"], "grounded_generation")
        self.assertFalse(first["governance"]["cache_hit"])
        self.assertTrue(second["governance"]["cache_hit"])
        self.assertEqual(self.good.calls, 1)

    def test_unverified_citations_abstain(self):
        self.registry.register(Provider("bad", "Unsupported claim. [other]"))
        self.registry.activate("bad")
        result = self.orchestrator.query("retries", 3, "one")
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["governance"]["reason"], "CITATION_VERIFICATION_FAILED")

    def test_active_provider_switch_is_atomic_for_new_requests(self):
        alternate = Provider("alternate", "Alternative answer. [source-1]")
        self.registry.register(alternate)
        revision = self.registry.activate("alternate")
        result = self.orchestrator.query("retries", 3, "one")
        self.assertEqual(result["governance"]["provider"], "alternate")
        self.assertEqual(result["governance"]["routing_revision"], revision)

    def test_unconfigured_provider_cannot_activate(self):
        self.registry.register(Provider("offline", "", ready=False))
        with self.assertRaisesRegex(ValueError, "not configured"):
            self.registry.activate("offline")

    def test_no_active_provider_fails_closed(self):
        registry = ProviderRegistry()
        registry.register(Provider("unused", "[source-1]"))
        with self.assertRaises(ProviderError):
            Orchestrator(self.retriever, registry).query("retries", 3, "one")
