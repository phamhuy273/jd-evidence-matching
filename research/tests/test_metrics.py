"""tests/test_metrics.py - Unit tests for IR and Agreement metrics with hand-calculated examples.

Governed by Rule R26 of the Anti-Hardcoding Specification:
Verify all mathematical evaluation metrics against at least 3 known hand-calculated cases.
"""

import math
import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score


def dcg_at_k(relevance_scores, k=5):
    """Compute Discounted Cumulative Gain at k."""
    scores = np.asarray(relevance_scores, dtype=float)[:k]
    if len(scores) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(scores) + 2))
    return float(np.sum(scores / discounts))


def ndcg_at_k(relevance_scores, ideal_scores=None, k=5):
    """Compute Normalized Discounted Cumulative Gain at k against ideal ranking."""
    actual_dcg = dcg_at_k(relevance_scores, k)
    if ideal_scores is None:
        ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)
    if ideal_dcg == 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def mrr_at_k(relevance_scores, k=10, threshold=1.0):
    """Compute Reciprocal Rank at k (1 / rank of first item >= threshold)."""
    sub = relevance_scores[:k] if k is not None else relevance_scores
    for idx, score in enumerate(sub):
        if score >= threshold:
            return 1.0 / (idx + 1)
    return 0.0


def precision_recall_at_k(relevance_scores, total_relevant_in_pool, k=5, threshold=1.0):
    """Compute Precision@k, Recall@k, and F1@k against total relevant in pool."""
    sub = relevance_scores[:k]
    hits = sum(1 for s in sub if s >= threshold)
    p_at_k = hits / k if k > 0 else 0.0
    r_at_k = hits / total_relevant_in_pool if total_relevant_in_pool > 0 else 0.0
    f1_at_k = (2 * p_at_k * r_at_k / (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0
    return p_at_k, r_at_k, f1_at_k


def context_precision_at_k(relevance_scores, k=5, threshold=1.0):
    """Compute Context Precision@k (RAGAs / IR standard)."""
    sub = relevance_scores[:k]
    v = [1 if s >= threshold else 0 for s in sub]
    n_rel = sum(v)
    if n_rel == 0:
        return 0.0
    cum_hits = 0
    weighted_prec = 0.0
    for idx, is_rel in enumerate(v):
        if is_rel:
            cum_hits += 1
            weighted_prec += cum_hits / (idx + 1)
    return float(weighted_prec / n_rel)


class TestMetricsHandCalculations:
    """Test suite validating mathematical metrics against hand-calculated examples (R26)."""

    def test_dcg_hand_calculated_cases(self):
        # Case 1: [2, 0, 1, 2, 0]
        # DCG = 2/log2(2) + 0/log2(3) + 1/log2(4) + 2/log2(5) + 0/log2(6)
        #     = 2.0 + 0.0 + 0.5 + 2/2.32192809489 + 0.0
        #     = 2.0 + 0.0 + 0.5 + 0.861353116 + 0.0 = 3.361353
        rel1 = [2, 0, 1, 2, 0]
        expected_dcg1 = 2.0 / math.log2(2) + 0.0 + 1.0 / math.log2(4) + 2.0 / math.log2(5)
        assert pytest.approx(dcg_at_k(rel1, k=5), rel=1e-5) == expected_dcg1
        assert pytest.approx(dcg_at_k(rel1, k=5), abs=1e-3) == 3.361

        # Case 2: [0, 1, 0, 0, 0]
        # DCG = 1 / log2(3) = 0.63092975
        rel2 = [0, 1, 0, 0, 0]
        expected_dcg2 = 1.0 / math.log2(3)
        assert pytest.approx(dcg_at_k(rel2, k=5), rel=1e-5) == expected_dcg2

        # Case 3: all zeros [0, 0, 0, 0, 0]
        assert dcg_at_k([0, 0, 0, 0, 0], k=5) == 0.0

        # Case 4: ideal [2, 2, 1, 0, 0]
        # IDCG = 2/1 + 2/log2(3) + 1/2 = 2.0 + 1.2618595 + 0.5 = 3.7618595
        rel4 = [2, 2, 1, 0, 0]
        expected_dcg4 = 2.0 + 2.0 / math.log2(3) + 0.5
        assert pytest.approx(dcg_at_k(rel4, k=5), rel=1e-5) == expected_dcg4

    def test_ndcg_hand_calculated_cases(self):
        # Case 1: rel = [2, 0, 1, 2, 0], ideal = [2, 2, 1, 0, 0]
        # NDCG = 3.361353 / 3.761860 = 0.893535
        rel1 = [2, 0, 1, 2, 0]
        ideal1 = [2, 2, 1, 0, 0]
        expected_ndcg1 = (2.0 + 0.5 + 2.0 / math.log2(5)) / (2.0 + 2.0 / math.log2(3) + 0.5)
        assert pytest.approx(ndcg_at_k(rel1, ideal1, k=5), rel=1e-5) == expected_ndcg1
        assert pytest.approx(ndcg_at_k(rel1, ideal1, k=5), abs=1e-3) == 0.894

        # Case 2: all zeros -> NDCG = 0.0
        assert ndcg_at_k([0, 0, 0], [2, 1, 0], k=3) == 0.0

        # Case 3: perfect ranking -> NDCG = 1.0
        assert ndcg_at_k([2, 2, 1, 0, 0], [2, 2, 1, 0, 0], k=5) == 1.0

        # Case 4: single hit at rank 2: [0, 1, 0] vs ideal [1, 0, 0]
        # DCG = 1/log2(3) = 0.63093, IDCG = 1/log2(2) = 1.0 -> NDCG = 0.63093
        assert pytest.approx(ndcg_at_k([0, 1, 0], [1, 0, 0], k=3), rel=1e-5) == 1.0 / math.log2(3)

    def test_mrr_hand_calculated_cases(self):
        # Case 1: First relevant at rank 1 -> MRR = 1/1 = 1.0
        assert mrr_at_k([2, 0, 1, 0], k=4) == 1.0

        # Case 2: First relevant at rank 2 -> MRR = 1/2 = 0.5
        assert mrr_at_k([0, 1, 2, 0], k=4) == 0.5

        # Case 3: First relevant at rank 3 -> MRR = 1/3 ≈ 0.333333
        assert pytest.approx(mrr_at_k([0, 0, 1, 0], k=4), rel=1e-5) == 1.0 / 3.0

        # Case 4: No relevant items in top-k -> MRR = 0.0
        assert mrr_at_k([0, 0, 0, 0], k=4) == 0.0

    def test_precision_recall_f1_hand_calculated_cases(self):
        # Case 1: 3 relevant in top 5, 4 total relevant in pool
        # P@5 = 3/5 = 0.60
        # R@5 = 3/4 = 0.75
        # F1@5 = (2 * 0.60 * 0.75) / (0.60 + 0.75) = 0.90 / 1.35 ≈ 0.666667
        p, r, f1 = precision_recall_at_k([1, 0, 2, 1, 0], total_relevant_in_pool=4, k=5)
        assert pytest.approx(p, rel=1e-5) == 0.60
        assert pytest.approx(r, rel=1e-5) == 0.75
        assert pytest.approx(f1, rel=1e-5) == 2.0 * 0.60 * 0.75 / 1.35

        # Case 2: Zero hits
        p0, r0, f10 = precision_recall_at_k([0, 0, 0, 0, 0], total_relevant_in_pool=3, k=5)
        assert p0 == 0.0 and r0 == 0.0 and f10 == 0.0

        # Case 3: Empty ground truth pool (no relevant items exist)
        p_empty, r_empty, f1_empty = precision_recall_at_k([0, 0], total_relevant_in_pool=0, k=2)
        assert p_empty == 0.0 and r_empty == 0.0 and f1_empty == 0.0

    def test_context_precision_hand_calculated_cases(self):
        # Case 1: relevant at rank 1 and 3 in top-5
        # Precision@1 = 1/1 = 1.000
        # Precision@3 = 2/3 ≈ 0.666667
        # CP@5 = (1.000 + 2/3) / 2 = (5/3) / 2 = 5/6 ≈ 0.833333
        cp1 = context_precision_at_k([1, 0, 2, 0, 0], k=5)
        assert pytest.approx(cp1, rel=1e-5) == 5.0 / 6.0

        # Case 2: No relevant in top-k -> CP = 0.0
        assert context_precision_at_k([0, 0, 0], k=3) == 0.0

        # Case 3: All relevant -> CP = 1.0
        assert context_precision_at_k([2, 1, 2], k=3) == 1.0

    def test_quadratic_weighted_kappa(self):
        # Exact agreement
        a = [0, 1, 2, 1, 0]
        b = [0, 1, 2, 1, 0]
        assert cohen_kappa_score(a, b, weights="quadratic") == 1.0

        # Known slight disagreement
        a = [0, 1, 2, 1, 0]
        b = [0, 1, 1, 1, 0]
        k = cohen_kappa_score(a, b, weights="quadratic")
        assert 0.7 < k < 1.0
