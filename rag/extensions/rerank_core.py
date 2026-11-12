"""Reorder one already-selected candidate set.

The first-stage retriever still chooses the n records. These rerankers only
change that set's order, so Recall at n cannot change. Order-sensitive metrics
such as MRR and nDCG can.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter


_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reorder(record_ids: list[str], scores: list[float]) -> tuple[list[str], list[float]]:
    if len(record_ids) != len(scores):
        raise ValueError("record ids and scores must have the same length")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("candidate record ids must be unique")
    order = sorted(range(len(record_ids)), key=lambda i: (-scores[i], i))
    return (
        [record_ids[i] for i in order],
        [float(scores[i]) for i in order],
    )


class BM25Reranker:
    """Score the selected records with corpus-level Okapi BM25.

    Inverse document frequency is estimated once on the full corpus. Each call
    scores only the n records that retrieval already returned.
    """

    def __init__(self, records: dict[str, str], k1: float = 1.2, b: float = 0.75):
        if not records:
            raise ValueError("BM25 reranker needs the corpus records")
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and b in [0, 1]")
        self.method = "bm25"
        self.k1 = float(k1)
        self.b = float(b)

        start = time.perf_counter()
        term_freq: dict[str, dict[str, int]] = {}
        doc_len: dict[str, int] = {}
        document_freq: Counter[str] = Counter()
        total_len = 0
        for record_id, text in records.items():
            counts = Counter(tokenize(text))
            term_freq[str(record_id)] = dict(counts)
            length = sum(counts.values())
            doc_len[str(record_id)] = length
            total_len += length
            document_freq.update(counts.keys())

        self._term_freq = term_freq
        self._doc_len = doc_len
        self._n = len(records)
        self._avgdl = total_len / self._n
        self._idf = {
            term: math.log(1.0 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }
        self.index_build_ms = (time.perf_counter() - start) * 1000.0

    def score(self, query: str, record_id: str) -> float:
        counts = self._term_freq.get(str(record_id))
        if counts is None:
            raise KeyError(record_id)
        length = self._doc_len[str(record_id)]
        if self._avgdl == 0:
            return 0.0
        norm = self.k1 * (1.0 - self.b + self.b * length / self._avgdl)
        total = 0.0
        for term, query_tf in Counter(tokenize(query)).items():
            tf = counts.get(term, 0)
            if tf == 0 or query_tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            total += idf * (tf * (self.k1 + 1.0)) / (tf + norm) * query_tf
        return float(total)

    def rerank(self, query: str, record_ids: list[str], texts: list[str] | None = None) -> dict:
        if texts is not None and len(texts) != len(record_ids):
            raise ValueError("texts must align with record ids")
        start = time.perf_counter()
        raw = [self.score(query, record_id) for record_id in record_ids]
        ordered_ids, ordered_scores = reorder(list(record_ids), raw)
        return {
            "record_ids": ordered_ids,
            "scores": ordered_scores,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "method": self.method,
        }


class CrossEncoderReranker:
    """Score each selected query-document pair with a cross-encoder.

    Only the n selected texts are scored. Model loading is kept separate from
    the latency returned by ``rerank``.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 16,
    ):
        if max_length < 8 or batch_size < 1:
            raise ValueError("max_length must be >= 8 and batch_size must be positive")
        from sentence_transformers import CrossEncoder

        kwargs = {"max_length": max_length}
        if device is not None:
            kwargs["device"] = device
        start = time.perf_counter()
        self.model = CrossEncoder(model_name, **kwargs)
        self.load_ms = (time.perf_counter() - start) * 1000.0
        self.batch_size = batch_size
        self.method = "cross-encoder"
        self.model_name = model_name

    def rerank(self, query: str, record_ids: list[str], texts: list[str] | None = None) -> dict:
        if texts is None or len(texts) != len(record_ids):
            raise ValueError("cross-encoder reranking needs one text per record id")
        start = time.perf_counter()
        predicted = self.model.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        ordered_ids, ordered_scores = reorder(list(record_ids), [float(score) for score in predicted])
        return {
            "record_ids": ordered_ids,
            "scores": ordered_scores,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "method": self.method,
        }


def build_reranker(
    method: str,
    records: dict[str, str] | None = None,
    device: str | None = None,
    model_name: str | None = None,
    max_length: int = 256,
    batch_size: int = 16,
    k1: float = 1.2,
    b: float = 0.75,
):
    if method == "bm25":
        if records is None:
            raise ValueError("BM25 reranker needs the corpus records")
        return BM25Reranker(records, k1=k1, b=b)
    if method == "cross-encoder":
        kwargs = {
            "device": device,
            "max_length": max_length,
            "batch_size": batch_size,
        }
        if model_name is not None:
            kwargs["model_name"] = model_name
        return CrossEncoderReranker(**kwargs)
    raise ValueError("method must be 'bm25' or 'cross-encoder'")
