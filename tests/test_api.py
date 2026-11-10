import http.client
import json
import socket
import threading
import unittest
import os
from unittest.mock import Mock
from http.server import ThreadingHTTPServer

from rag_reference.api import create_handler
from rag_reference.core import Passage, Retriever
from rag_reference.orchestration import Orchestrator
from rag_reference.providers import GenerationResult, ProviderRegistry


class FakeProvider:
    name = "fake"

    def ready(self):
        return True

    def generate(self, request):
        return GenerationResult("Retries should be idempotent. [test-1]")


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        retriever = Retriever([Passage("test-1", "Example", "Reliable jobs use retries and idempotency.")])
        registry = ProviderRegistry()
        registry.register(FakeProvider())
        registry.activate("fake")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(Orchestrator(retriever, registry)))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        conn.request(method, path, body=body, headers=request_headers)
        response = conn.getresponse()
        result = response.status, dict(response.getheaders()), json.loads(response.read())
        conn.close()
        return result

    def test_health(self):
        status, headers, result = self.request("GET", "/health")
        self.assertEqual((status, result["documents"]), (200, 1))
        self.assertTrue(headers["X-Request-ID"])

    def test_liveness_and_readiness(self):
        status, _, result = self.request("GET", "/live")
        self.assertEqual((status, result["status"]), (200, "live"))
        status, _, result = self.request("GET", "/ready")
        self.assertEqual((status, result["status"]), (200, "ready"))

    def test_query(self):
        status, headers, result = self.request("POST", "/query", json.dumps({"question": "idempotency"}))
        self.assertEqual(status, 200)
        self.assertEqual(result["citations"][0]["doc_id"], "test-1")
        self.assertTrue(headers["X-Request-ID"])

    def test_malformed_json(self):
        status, headers, result = self.request("POST", "/query", '{"question":')
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "MALFORMED_JSON")
        self.assertEqual(headers["X-Request-ID"], result["request_id"])

    def test_question_validation(self):
        cases = [
            ({}, "MISSING_QUESTION"),
            ({"question": 123}, "INVALID_QUESTION"),
            ({"question": "  "}, "INVALID_QUESTION"),
        ]
        for payload, code in cases:
            with self.subTest(payload=payload):
                status, _, result = self.request("POST", "/query", json.dumps(payload))
                self.assertEqual(status, 400)
                self.assertEqual(result["error"]["code"], code)

    def test_top_k_validation(self):
        for top_k in ("3", 0, 21, True):
            with self.subTest(top_k=top_k):
                status, _, result = self.request("POST", "/query", json.dumps({"question": "retries", "top_k": top_k}))
                self.assertEqual(status, 400)
                self.assertEqual(result["error"]["code"], "INVALID_TOP_K")

    def test_empty_and_excessive_body(self):
        status, _, result = self.request("POST", "/query", "")
        self.assertEqual((status, result["error"]["code"]), (400, "EMPTY_REQUEST_BODY"))
        status, _, result = self.request("POST", "/query", "x" * 16_385)
        self.assertEqual((status, result["error"]["code"]), (413, "PAYLOAD_TOO_LARGE"))

    def test_missing_and_invalid_content_length(self):
        for header in (None, "not-a-number", "-1"):
            with self.subTest(header=header):
                lines = ["POST /query HTTP/1.1", "Host: localhost", "Content-Type: application/json"]
                if header is not None:
                    lines.append(f"Content-Length: {header}")
                raw_request = ("\r\n".join(lines) + "\r\n\r\n").encode()
                with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3) as conn:
                    conn.sendall(raw_request)
                    chunks = []
                    while chunk := conn.recv(4096):
                        chunks.append(chunk)
                    response = b"".join(chunks).decode("iso-8859-1")
                self.assertIn("400", response)
                self.assertIn("INVALID_CONTENT_LENGTH", response)

    def test_unknown_route(self):
        status, _, result = self.request("GET", "/unknown")
        self.assertEqual(status, 404)
        self.assertEqual(result["error"]["code"], "NOT_FOUND")

    def test_unsupported_method(self):
        status, _, result = self.request("DELETE", "/query")
        self.assertEqual(status, 405)
        self.assertEqual(result["error"]["code"], "METHOD_NOT_ALLOWED")

    def test_active_provider_can_only_be_changed_with_admin_token(self):
        old_token = os.environ.get("RAG_ADMIN_TOKEN")
        os.environ["RAG_ADMIN_TOKEN"] = "admin-test-token"
        try:
            status, _, result = self.request("PUT", "/admin/providers/active", json.dumps({"provider": "fake"}))
            self.assertEqual((status, result["error"]["code"]), (403, "FORBIDDEN"))
            status, _, result = self.request("PUT", "/admin/providers/active", json.dumps({"provider": "fake"}), {"X-Admin-Token": "admin-test-token"})
            self.assertEqual(status, 200)
            self.assertEqual(result["active_provider"], "fake")
        finally:
            if old_token is None:
                del os.environ["RAG_ADMIN_TOKEN"]
            else:
                os.environ["RAG_ADMIN_TOKEN"] = old_token

    def test_internal_error_is_safe_and_has_consistent_request_id(self):
        # Replace the captured retriever's method for this request, then restore it.
        handler = self.server.RequestHandlerClass
        orchestrator = next(cell.cell_contents for cell in handler.do_POST.__closure__ if isinstance(cell.cell_contents, Orchestrator))
        search = orchestrator.retriever.search
        orchestrator.retriever.search = Mock(side_effect=RuntimeError("sensitive internal detail"))
        try:
            status, headers, result = self.request("POST", "/query", json.dumps({"question": "retries"}))
        finally:
            orchestrator.retriever.search = search
        self.assertEqual(status, 500)
        self.assertEqual(result["error"]["code"], "INTERNAL_ERROR")
        self.assertNotIn("sensitive internal detail", json.dumps(result))
        self.assertEqual(headers["X-Request-ID"], result["request_id"])

    def test_request_log_does_not_include_question(self):
        secret = "do-not-log-this-question"
        with self.assertLogs("rag_reference", level="INFO") as logs:
            status, _, _ = self.request("POST", "/query", json.dumps({"question": secret}))
        self.assertEqual(status, 200)
        self.assertNotIn(secret, "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
