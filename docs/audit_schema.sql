-- Apply only through the deployment migration process with a dedicated migrator role.
CREATE TABLE rag_audit_events (
    sequence BIGINT PRIMARY KEY,
    event_type TEXT NOT NULL,
    request_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    previous_hash CHAR(64) NOT NULL,
    event_hash CHAR(64) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX rag_audit_events_tenant_created_idx ON rag_audit_events (tenant_id, created_at DESC);
CREATE INDEX rag_audit_events_request_id_idx ON rag_audit_events (request_id);

-- The runtime role receives INSERT only. An independent audit-review role receives SELECT.
REVOKE ALL ON rag_audit_events FROM PUBLIC;
-- Example deployment grants (replace role names in the migration environment):
-- GRANT INSERT ON rag_audit_events TO rag_audit_writer;
-- GRANT SELECT ON rag_audit_events TO rag_audit_reviewer;
-- Never grant UPDATE or DELETE to the runtime writer.
