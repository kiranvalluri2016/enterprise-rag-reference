# Deployment readiness contract

`RAG_ENV=production` enables fail-closed configuration validation. It requires these configuration names, whose values must be held in Azure Key Vault:

- `RAG_AZURE_KEY_VAULT_URL`
- `RAG_JWT_SECRET_NAME`
- `RAG_ADMIN_TOKEN_SECRET_NAME`
- `RAG_AUDIT_POSTGRES_DSN_SECRET_NAME`
- `RAG_JWT_ISSUER`
- `RAG_JWT_AUDIENCE`

The workload identity needs only Key Vault secret-read access. The PostgreSQL runtime identity needs the least privileges described in `audit_schema.sql` and the pgvector migration; use a transaction-mode connection pool.

Production JWT validation requires a signed HS256 token with `exp`, the configured issuer, and the configured audience. Tokens missing any of these constraints are rejected.

Endpoints:

- `/live`: process liveness only; suitable for container restart checks.
- `/ready`: returns 200 only with an active configured provider and—when production mode is active—JWT authentication plus PostgreSQL durable audit storage.
- `/health`: compatibility/status endpoint; it is not a deployment readiness assertion.

The provided Docker health check targets `/live`. Route external traffic only after the deployment platform confirms `/ready`.
