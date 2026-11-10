-- Apply through the deployment migration system, not manually in production.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE rag_chunks (
    chunk_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding vector(64) NOT NULL,
    deleted_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, chunk_id)
);

CREATE INDEX rag_chunks_tenant_corpus_active_idx
    ON rag_chunks (tenant_id, corpus_version)
    WHERE deleted_at IS NULL;

CREATE INDEX rag_chunks_embedding_hnsw_idx
    ON rag_chunks USING hnsw (embedding vector_cosine_ops)
    WHERE deleted_at IS NULL;

-- The application must set app.tenant_id with set_config(..., true) for each transaction.
ALTER TABLE rag_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_chunks FORCE ROW LEVEL SECURITY;

CREATE POLICY rag_chunks_tenant_isolation ON rag_chunks
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
