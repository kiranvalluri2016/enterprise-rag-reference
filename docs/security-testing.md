# Security testing baseline

The automated suite covers malformed requests, safe errors, request correlation, JWT signature/claim validation, role checks, tenant-bound retrieval, rate limiting, cache boundaries, circuit breaking, citation verification, and injection-resistant source framing.

Required CI additions before deployment:

1. Run the full test suite on every change.
2. Run dependency and container vulnerability scans against pinned build inputs.
3. Generate and retain an SBOM artifact.
4. Run prompt-injection, data-exfiltration, retrieval-poisoning, authorization, rate-limit, and provider-outage tests against a staging service.
5. Review false-positive/false-negative safety and grounding metrics before provider or corpus promotion.

The test suite cannot prove immunity to LLM prompt injection or model hallucination. The service mitigates those risks by source framing, tenant filtering, citation verification, abstention, and restricted tool surface; production requires adversarial evaluation with the chosen live models.
