"""tests/test_evaluation_phase5.py - Verifies Phase 5 2x2 Factorial Benchmark and Statistical Tests.

Governed by:
- Rule R22: Paired two-sided Wilcoxon signed-rank test.
- Rule R23: 95% Bootstrap Confidence Intervals with 10,000 resamples.
- Rule R24: Exact p-value reporting with Holm-Bonferroni correction and CI checking.
- Rule R26: Verifiable benchmark results across all 12 configurations.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT_DIR / "dataset" / "benchmark_results"
SUMMARY_CSV = BENCHMARK_DIR / "table_3_retrieval_benchmark.csv"
STATS_JSON = BENCHMARK_DIR / "statistical_tests_report.json"
PER_QUERY_CSV = BENCHMARK_DIR / "per_query_evaluation.csv"


class TestEvaluationPhase5:
    """Validates complete Phase 5 empirical evaluation and hypothesis testing artifacts."""

    def test_benchmark_summary_exists_and_contains_all_12_configs(self):
        assert SUMMARY_CSV.exists(), f"Missing Table 3 benchmark summary: {SUMMARY_CSV}"
        df = pd.read_csv(SUMMARY_CSV)
        # 12 configs + 1 random baseline = 13 rows
        assert len(df) >= 12, f"Expected at least 12 configurations, found {len(df)}"

        # Verify metric bounds
        for col in ["ndcg_10_mean", "ndcg_5_mean", "ndcg_3_mean", "ndcg_1_mean", "mrr_10_mean", "p5_mean", "recall_10_mean", "cp_5_mean"]:
            assert col in df.columns
            vals = df[col].values
            assert (vals >= 0.0).all() and (vals <= 1.0).all(), f"Metric {col} values out of bounds [0, 1]"

    def test_per_query_records_cover_all_25_jds(self):
        assert PER_QUERY_CSV.exists(), f"Missing per-query records: {PER_QUERY_CSV}"
        df = pd.read_csv(PER_QUERY_CSV)
        assert len(df) == 12 * 25, f"Expected 300 records (12 configs x 25 JDs), found {len(df)}"
        jds = set(df["jd_id"].unique())
        assert len(jds) == 25, f"Expected 25 unique JDs, found {len(jds)}"

    def test_family_of_4_statistical_tests(self):
        assert STATS_JSON.exists(), f"Missing statistical tests report: {STATS_JSON}"
        with open(STATS_JSON, "r", encoding="utf-8") as f:
            stats = json.load(f)

        assert len(stats) == 4, f"Expected Family of 4 comparisons, found {len(stats)}"

        comp_map = {item["comparison"]: item for item in stats}

        # Comp 1: Full Proposed vs Full Baseline (rerank_ast_with_header vs rerank_line_no_header)
        c1 = comp_map["Comp 1 (Full Proposed vs Full Baseline)"]
        assert c1["ci_95_low"] <= c1["mean_diff"] <= c1["ci_95_high"]
        assert c1["p_holm"] >= 0.05, "Comp 1 should be declared non-significant under Holm correction"
        assert not c1["statistically_significant"]

        # Comp 2: Pure Chunking Effect (rerank_ast_no_header vs rerank_line_no_header)
        c2 = comp_map["Comp 2 (Pure Chunking Effect)"]
        assert c2["mean_diff"] < 0, "AST No-Header should show degradation without headers"
        assert c2["ci_95_high"] < 0, "Comp 2 95% CI must exclude 0"
        assert c2["p_holm"] < 0.05, "Comp 2 must be statistically significant"
        assert c2["statistically_significant"]

        # Comp 3: Header Effect on AST (rerank_ast_with_header vs rerank_ast_no_header)
        c3 = comp_map["Comp 3 (Header Effect on AST)"]
        assert c3["mean_diff"] > 0, "Header must show positive improvement on AST"
        assert c3["ci_95_low"] > 0, "Comp 3 95% CI must be strictly above 0"
        assert c3["p_holm"] < 0.05, "Comp 3 must be statistically significant"
        assert c3["statistically_significant"]

        # Comp 4: Header Effect on Line (rerank_line_with_header vs rerank_line_no_header)
        c4 = comp_map["Comp 4 (Header Effect on Line)"]
        assert c4["ci_95_low"] <= c4["mean_diff"] <= c4["ci_95_high"]
        assert c4["p_holm"] >= 0.05, "Comp 4 should not be statistically significant after correction"
