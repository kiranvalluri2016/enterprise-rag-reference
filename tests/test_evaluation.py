import unittest

from rag_reference.core import Passage, Retriever
from rag_reference.evaluation import EvaluationCase, EvaluationThresholds, GroundedEvaluator, release_decision
from rag_reference.orchestration import Orchestrator
from rag_reference.providers import GenerationResult, ProviderRegistry


class Provider:
    name = "evaluation-provider"
    def ready(self): return True
    def generate(self, request): return GenerationResult("Retries must be idempotent. [retries]")


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        registry = ProviderRegistry()
        registry.register(Provider())
        registry.activate("evaluation-provider")
        self.evaluator = GroundedEvaluator(Orchestrator(Retriever([Passage("retries", "Retries", "Retries require idempotency.")]), registry))

    def test_evaluation_measures_grounded_citations(self):
        report = self.evaluator.evaluate([EvaluationCase("retries", "Why use idempotency for retries?", frozenset({"retries"}))])
        self.assertEqual((report["grounded_rate"], report["citation_precision"], report["citation_recall"]), (1.0, 1.0, 1.0))

    def test_release_requires_human_approval_after_metrics_pass(self):
        report = {"grounded_rate": 1.0, "citation_precision": 1.0, "citation_recall": 1.0}
        self.assertEqual(release_decision(report, EvaluationThresholds())["status"], "HUMAN_REVIEW_REQUIRED")
        self.assertEqual(release_decision(report, EvaluationThresholds(), human_approved=True)["status"], "ELIGIBLE_FOR_PROMOTION")
