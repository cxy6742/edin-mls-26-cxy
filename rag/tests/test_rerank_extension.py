"""Reranking keeps the retrieved candidate set and can change its order."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extensions.rerank_core import BM25Reranker
from src.metrics import mrr_at_k, recall_at_k


class RerankExtensionTests(unittest.TestCase):
    def setUp(self):
        self.records = {
            "weather": "The weather is sunny and warm today.",
            "tax": "Sales tax is charged on items bought in a retail store.",
            "market": "The opening stock price depends on buyers and sellers.",
        }
        self.reranker = BM25Reranker(self.records)

    def test_lexical_match_moves_ahead_without_changing_the_set(self):
        original = ["weather", "market", "tax"]
        result = self.reranker.rerank("sales tax in a retail store", original)
        self.assertEqual(result["record_ids"][0], "tax")
        self.assertEqual(set(result["record_ids"]), set(original))
        self.assertEqual(len(result["record_ids"]), len(original))
        self.assertEqual(result["method"], "bm25")
        self.assertGreaterEqual(result["latency_ms"], 0.0)
        self.assertEqual(result["scores"], sorted(result["scores"], reverse=True))

    def test_equal_scores_keep_the_original_order(self):
        original = ["weather", "market", "tax"]
        result = self.reranker.rerank("zzzz-no-shared-token", original)
        self.assertEqual(result["record_ids"], original)
        self.assertEqual(result["scores"], [0.0, 0.0, 0.0])

    def test_recall_at_k_is_unchanged_when_only_order_changes(self):
        original = ["weather", "market", "tax"]
        relevance = {"tax": 1.0}
        result = self.reranker.rerank("sales tax in a retail store", original)
        self.assertEqual(
            recall_at_k(original, relevance, 3),
            recall_at_k(result["record_ids"], relevance, 3),
        )
        self.assertGreater(
            mrr_at_k(result["record_ids"], relevance, 3),
            mrr_at_k(original, relevance, 3),
        )

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaises(KeyError):
            self.reranker.rerank("sales tax", ["missing"])


if __name__ == "__main__":
    unittest.main()
