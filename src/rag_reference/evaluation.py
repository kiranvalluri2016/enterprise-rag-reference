"""Deterministic evaluation and human-gated release decisions for grounded responses."""
from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger("rag_reference.evaluation")


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_citation_ids: frozenset[str]
    tenant_id: str = "public"


@dataclass(frozen=True)
class EvaluationThresholds:
    minimum_grounded_rate: float = 1.0
    minimum_citation_precision: float = 1.0
    minimum_citation_recall: float = 1.0


class GroundedEvaluator:
    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    def evaluate(self, cases: list[EvaluationCase]) -> dict:
        if not cases:
            raise ValueError("At least one evaluation case is required")
        results = []
        for case in cases:
            response = self.orchestrator.query(case.question, 3, f"evaluation-{case.case_id}", case.tenant_id)
            actual = {citation["doc_id"] for citation in response["citations"]}
            results.append({"case_id": case.case_id, "status": response["status"], "expected_citation_ids": sorted(case.expected_citation_ids),
                            "actual_citation_ids": sorted(actual)})
        grounded = sum(result["status"] == "grounded_generation" for result in results)
        correct = sum(len(set(result["actual_citation_ids"]) & set(result["expected_citation_ids"])) for result in results)
        actual_total = sum(len(result["actual_citation_ids"]) for result in results)
        expected_total = sum(len(result["expected_citation_ids"]) for result in results)
        report = {"case_count": len(results), "grounded_rate": grounded / len(results),
                  "citation_precision": correct / actual_total if actual_total else 0.0,
                  "citation_recall": correct / expected_total if expected_total else 0.0,
                  "cases": results}
        LOGGER.info("evaluation_completed case_count=%s grounded_rate=%.3f", len(results), report["grounded_rate"])
        return report


def release_decision(report: dict, thresholds: EvaluationThresholds, human_approved: bool = False) -> dict:
    metrics_pass = (report["grounded_rate"] >= thresholds.minimum_grounded_rate and
                    report["citation_precision"] >= thresholds.minimum_citation_precision and
                    report["citation_recall"] >= thresholds.minimum_citation_recall)
    if not metrics_pass:
        status = "BLOCKED_EVALUATION"
    elif not human_approved:
        status = "HUMAN_REVIEW_REQUIRED"
    else:
        status = "ELIGIBLE_FOR_PROMOTION"
    LOGGER.info("release_decision status=%s", status)
    return {"status": status, "metrics_pass": metrics_pass, "human_approved": human_approved}
