# AI governance baseline

This document is a control register for the reference implementation. It is not a certification claim.

## Provider and model register

| Control | Enforced behavior |
|---|---|
| Provider allowlist | Only registered adapters can become active. |
| Change control | Runtime route changes require the administrative capability and `RAG_ADMIN_TOKEN`; the audit witness records the revision. |
| Credential custody | Credentials are read only from server-side environment integration points and are excluded from logs and responses. |
| Failure handling | Provider calls use a bounded timeout; repeated classified failures open a circuit breaker. |
| Model approval | Deployment owners must set an explicit provider model; no default model is selected. |

## Data and retention policy

Only synthetic local documents are shipped. A production deployment must classify sources before ingestion, define the legal basis and retention duration, and execute `soft_delete_document` followed by scheduled physical erasure and backup expiry. The application cache is process-local, bounded, and ephemeral.

## Quality and safety release gate

Before a provider/model/corpus release: record the provider/model version, corpus version, evaluation dataset, grounded-answer score, citation-verification failure rate, prompt-injection results, and human approval. Block promotion when any required gate lacks evidence.

`GroundedEvaluator` emits reproducible grounded-rate and citation precision/recall metrics. `release_decision` remains fail-closed: passing metrics yields `HUMAN_REVIEW_REQUIRED`, not automatic promotion.

## Incident response

Disable the provider route, preserve sanitized audit correlation IDs, rotate affected credentials through the secret manager, assess source/corpus exposure, and require a reviewed re-enable decision. Do not use application logs as a substitute for durable audit retention.
