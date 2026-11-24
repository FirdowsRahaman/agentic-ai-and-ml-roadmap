"""Minimal air-gapped Hybrid RAG (BM25 + Vector RRF) & Read-Only SQL Router Agent."""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DocumentChunk:
    doc_id: str
    text: str


class HybridRetriever:
    """Combines Okapi BM25 lexical search with dense cosine vectors via Reciprocal Rank Fusion."""

    def __init__(self, chunks: list[DocumentChunk], dim: int = 128):
        self.chunks = {c.doc_id: c for c in chunks}
        self.dim = dim
        self.doc_len: dict[str, int] = {}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.vectors: dict[str, np.ndarray] = {}

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.doc_len[chunk.doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term][chunk.doc_id] = tf
            self.vectors[chunk.doc_id] = self._embed(chunk.text)

        self.avgdl = sum(self.doc_len.values()) / max(len(chunks), 1)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_-]+", text.lower())

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        for tok in self._tokenize(text):
            vec[hash(tok) % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-12)

    def _bm25_rank(self, query: str) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        n_docs = len(self.chunks)
        for term in self._tokenize(query):
            if term not in self.postings:
                continue
            df = len(self.postings[term])
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in self.postings[term].items():
                dl = self.doc_len[doc_id]
                denom = tf + 1.5 * (1.0 - 0.75 + 0.75 * (dl / self.avgdl))
                scores[doc_id] += idf * ((tf * 2.5) / denom)
        return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def _dense_rank(self, query: str) -> list[str]:
        q_vec = self._embed(query)
        sims = {doc_id: float(np.dot(vec, q_vec)) for doc_id, vec in self.vectors.items()}
        return [d for d, _ in sorted(sims.items(), key=lambda x: x[1], reverse=True)]

    def search(self, query: str, top_k: int = 2) -> list[DocumentChunk]:
        bm25_list = self._bm25_rank(query)
        dense_list = self._dense_rank(query)
        rrf: dict[str, float] = defaultdict(float)
        for ranked in (bm25_list, dense_list):
            for rank, doc_id in enumerate(ranked, start=1):
                rrf[doc_id] += 1.0 / (60 + rank)
        top_ids = [d for d, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        return [self.chunks[d] for d in top_ids]


class ReadOnlySQLTool:
    """In-memory SQL analytics engine enforcing read-only SELECT privileges."""

    FORBIDDEN = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|ATTACH)\b", re.IGNORECASE)

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE orders (order_id TEXT, region TEXT, amount_usd REAL)"
        )
        self.conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [
                ("ORD-1", "APAC", 1200.0),
                ("ORD-2", "APAC", 800.0),
                ("ORD-3", "EMEA", 950.0),
            ],
        )

    def execute(self, sql: str) -> list[tuple]:
        if self.FORBIDDEN.search(sql) or not sql.strip().upper().startswith("SELECT"):
            raise PermissionError("Guardrail blocked non-SELECT SQL statement.")
        return self.conn.execute(sql).fetchall()


class LocalCopilotRouter:
    """Routes quantitative questions to ReadOnlySQLTool and policy questions to HybridRetriever."""

    def __init__(self, retriever: HybridRetriever, sql_tool: ReadOnlySQLTool):
        self.retriever = retriever
        self.sql_tool = sql_tool

    def ask(self, question: str) -> dict[str, str]:
        q_low = question.lower()
        if any(kw in q_low for kw in ("total", "sum", "revenue", "orders", "apac", "emea")):
            rows = self.sql_tool.execute(
                "SELECT region, SUM(amount_usd) FROM orders GROUP BY region ORDER BY region"
            )
            formatted = ", ".join(f"{region}: ${total:.2f}" for region, total in rows)
            return {"route": "sql_tool", "answer": f"Regional revenue -> {formatted}"}

        hits = self.retriever.search(question, top_k=1)
        best = hits[0]
        return {
            "route": "hybrid_rag",
            "answer": f"[{best.doc_id}] {best.text}",
        }


def build_default_copilot() -> LocalCopilotRouter:
    docs = [
        DocumentChunk("POL-101", "Enterprise customers receive a 30-day full refund guarantee on annual plans."),
        DocumentChunk("POL-204", "Error code ERR-409-SYNC indicates an expired OAuth 2.1 MCP token."),
        DocumentChunk("POL-309", "All PII columns must be redacted prior to external model inference."),
    ]
    return LocalCopilotRouter(HybridRetriever(docs), ReadOnlySQLTool())


if __name__ == "__main__":
    bot = build_default_copilot()
    print(bot.ask("What does error ERR-409-SYNC mean?"))
    print(bot.ask("What is total revenue across APAC and EMEA orders?"))
