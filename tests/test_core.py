import json
import tempfile
import unittest
from pathlib import Path

from rag_reference.core import Passage, Retriever, load_documents


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.r = Retriever([
            Passage("a", "Alpha", "Retries and idempotency prevent duplicate jobs."),
            Passage("b", "Beta", "Grounded retrieval should provide citations."),
        ])

    def test_relevant_document_first(self):
        self.assertEqual(self.r.search("idempotency retries")[0]["doc_id"], "a")

    def test_citations_and_grounded_extract(self):
        result = self.r.answer("retrieval citations")
        self.assertEqual(result["status"], "grounded_extract")
        self.assertEqual(result["citations"][0]["doc_id"], "b")
        self.assertIn("[b]", result["answer"])

    def test_abstain_on_no_match(self):
        result = self.r.answer("zebras")
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["answer"])

    def test_invalid_input(self):
        for question, top_k in [("", 3), ("hello", 0), ("hello", 21)]:
            with self.assertRaises(ValueError):
                self.r.search(question, top_k)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("one", "two"):
                (Path(directory) / f"{name}.json").write_text(json.dumps({"id": "same", "title": name, "text": "hello"}))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_documents(Path(directory))

    def test_missing_document_fields_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "bad.json").write_text('{"id":"x"}')
            with self.assertRaisesRegex(ValueError, "Invalid document"):
                load_documents(Path(directory))


if __name__ == "__main__":
    unittest.main()
