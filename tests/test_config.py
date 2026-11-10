import unittest

from rag_reference.config import ConfigurationError, is_production, validate_runtime_configuration


class ConfigurationTests(unittest.TestCase):
    def test_development_mode_does_not_require_deployment_integrations(self):
        validate_runtime_configuration({})
        self.assertFalse(is_production({}))

    def test_production_mode_requires_external_security_configuration(self):
        with self.assertRaises(ConfigurationError):
            validate_runtime_configuration({"RAG_ENV": "production"})

    def test_production_mode_accepts_named_secret_contract(self):
        environment = {
            "RAG_ENV": "production",
            "RAG_AZURE_KEY_VAULT_URL": "https://example.vault.azure.net",
            "RAG_JWT_SECRET_NAME": "rag-jwt-secret",
            "RAG_ADMIN_TOKEN_SECRET_NAME": "rag-admin-token",
            "RAG_AUDIT_POSTGRES_DSN_SECRET_NAME": "rag-audit-postgres-dsn",
        }
        validate_runtime_configuration(environment)
        self.assertTrue(is_production(environment))
