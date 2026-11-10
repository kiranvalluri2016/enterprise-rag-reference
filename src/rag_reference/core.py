"""Deterministic lexical retrieval primitives with source citations."""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")
LOGGER = logging.getLogger("rag_reference.core")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Passage:
    doc_id: str
    title: str
    text: str


class Retriever:
    def __init__(self, passages: list[Passage]):
        if not passages:
            raise ValueError("At least one passage is required")
        self.passages = passages
        self.counts = [Counter(tokenize(p.text)) for p in passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        self.df = Counter(term for counts in self.counts for term in counts)

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        if not question.strip():
            raise ValueError("Question must not be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        terms = set(tokenize(question))
        results = []
        n = len(self.passages)
        for passage, counts, length in zip(self.passages, self.counts, self.lengths):
            score = 0.0
            for term in terms:
                tf = counts[term]
                if not tf:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / self.avg_length))
            if score > 0:
                results.append({"doc_id": passage.doc_id, "title": passage.title,
                                "text": passage.text, "score": round(score, 6)})
        ranked = sorted(results, key=lambda r: (-r["score"], r["doc_id"]))[:top_k]
        LOGGER.info("lexical_search_completed result_count=%s", len(ranked))
        return ranked

    def answer(self, question: str, top_k: int = 3) -> dict:
        matches = self.search(question, top_k)
        if not matches:
            return {"question": question, "status": "abstained", "answer": None, "citations": []}
        # Extractive output only: never invent claims beyond retrieved text.
        return {"question": question, "status": "grounded_extract",
                "answer": "\n\n".join(f"[{m['doc_id']}] {m['text']}" for m in matches),
                "citations": [{"doc_id": m["doc_id"], "title": m["title"], "score": m["score"]} for m in matches]}


def load_documents(folder: Path) -> list[Passage]:
    passages = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(isinstance(data.get(k), str) and data[k].strip() for k in ("id", "title", "text")):
            raise ValueError(f"Invalid document: {path.name}")
        passages.append(Passage(data["id"], data["title"], data["text"]))
    if len({p.doc_id for p in passages}) != len(passages):
        raise ValueError("Duplicate document IDs")
    LOGGER.info("documents_loaded count=%s", len(passages))
    return passages
