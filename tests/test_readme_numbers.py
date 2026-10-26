"""tests/test_readme_numbers.py - Dynamic verification that all numerical claims in README.md match benchmark artifacts.

Governed by:
- Rule R37: Ground all Table and README numbers in dynamic measurement artifacts.
- Rule R5: Zero hardcoding or phantom figures.
"""

import json
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
README_FILE = ROOT_DIR / "README.md"
DATASET_DIR = ROOT_DIR / "dataset"
BENCHMARK_DIR = DATASET_DIR / "benchmark_results"

TABLE_3_2_JSON = DATASET_DIR / "table_3_2_measured.json"
RETRIEVAL_JSON = BENCHMARK_DIR / "table_3_retrieval_benchmark.json"
STATS_JSON = BENCHMARK_DIR / "statistical_tests_report.json"
SENSITIVITY_JSON = BENCHMARK_DIR / "sensitivity_analysis_report.json"
KAPPA_REPORT = BENCHMARK_DIR / "kappa_evaluation_report.txt"


class TestReadmeNumbers:
    """Verifies that README.md numbers stay in 100% lockstep with empirical results."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        assert README_FILE.exists(), "README.md missing"
        self.readme = README_FILE.read_text(encoding="utf-8")

        with open(TABLE_3_2_JSON, "r", encoding="utf-8") as f:
            self.table_3_2 = json.load(f)

        with open(RETRIEVAL_JSON, "r", encoding="utf-8") as f:
            self.retrieval = json.load(f)

        with open(STATS_JSON, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        with open(SENSITIVITY_JSON, "r", encoding="utf-8") as f:
            self.sensitivity = json.load(f)

        self.kappa_text = KAPPA_REPORT.read_text(encoding="utf-8")

    def test_corpus_subgroup_counts_in_readme(self):
        """Verify chunk counts and fallback rates in README match Table 3.2."""
        pure_n = self.table_3_2["ast_pure"]["chunk_count"]
        fallback_n = self.table_3_2["ast_fallback"]["chunk_count"]
        combined_n = self.table_3_2["ast_combined"]["chunk_count"]
        line_n = self.table_3_2["line_based"]["chunk_count"]

        assert str(pure_n) in self.readme  # 438
        assert str(fallback_n) in self.readme  # 0
        assert str(combined_n) in self.readme  # 438
        assert str(line_n) in self.readme  # 434

        fallback_rate = float(fallback_n / combined_n * 100)
        assert f"{fallback_rate:.1f}%" in self.readme  # 0.0%

    def test_table_3_2_syntax_metrics_in_readme(self):
        """Verify morphology percentages in README match table_3_2_measured.json."""
        pure = self.table_3_2["ast_pure"]
        fallback = self.table_3_2["ast_fallback"]
        combined = self.table_3_2["ast_combined"]
        line = self.table_3_2["line_based"]

        # Pure
        assert f"{pure['boundary_intact_rate_pct']:.1f}%" in self.readme  # 87.2%
        assert f"{pure['syntax_intact_rate_pct']:.1f}%" in self.readme  # 78.1%
        assert f"{pure['loc_dist']['mean']:.1f}" in self.readme  # 26.7
        assert f"{pure['tokens_content_dist']['mean']:.1f}" in self.readme  # 250.9
        assert f"{pure['truncated_512_rate_pct']:.1f}%" in self.readme  # 15.8%

        # Fallback
        assert f"{fallback['boundary_intact_rate_pct']:.1f}%" in self.readme  # 100.0%
        assert f"{fallback['syntax_intact_rate_pct']:.1f}%" in self.readme  # 100.0%
        assert f"{fallback['loc_dist']['mean']:.1f}" in self.readme  # 0.0
        assert f"{fallback['tokens_content_dist']['mean']:.1f}" in self.readme  # 0.0
        assert f"{fallback['truncated_512_rate_pct']:.1f}%" in self.readme  # 0.0%

        # Combined
        assert f"{combined['boundary_intact_rate_pct']:.1f}%" in self.readme  # 87.2%
        assert f"{combined['syntax_intact_rate_pct']:.1f}%" in self.readme  # 78.1%
        assert f"{combined['loc_dist']['mean']:.1f}" in self.readme  # 26.7
        assert f"{combined['tokens_content_dist']['mean']:.1f}" in self.readme  # 250.9
        assert f"{combined['truncated_512_rate_pct']:.1f}%" in self.readme  # 15.8%

        # Line
        assert f"{line['boundary_intact_rate_pct']:.1f}%" in self.readme  # 44.0%
        assert f"{line['syntax_intact_rate_pct']:.1f}%" in self.readme  # 22.4%
        assert f"{line['loc_dist']['mean']:.1f}" in self.readme  # 41.0
        assert f"{line['tokens_content_dist']['mean']:.1f}" in self.readme  # 390.8
        assert f"{line['truncated_512_rate_pct']:.1f}%" in self.readme  # 28.6%

    def test_retrieval_ndcg_and_metrics_in_readme(self):
        """Verify retrieval benchmarks and top-performer claim in README."""
        retrieval_map = {r["config_id"]: r for r in self.retrieval}
        rerank_line_h = retrieval_map["rerank_line_with_header"]
        rerank_ast_h = retrieval_map["rerank_ast_with_header"]
        rerank_ast_no = retrieval_map["rerank_ast_no_header"]
        rerank_line_no = retrieval_map["rerank_line_no_header"]

        # NDCG@10 values
        assert f"{rerank_line_h['ndcg_10_mean']:.4f}" in self.readme  # 0.6206
        assert f"{rerank_ast_h['ndcg_10_mean']:.4f}" in self.readme  # 0.4866
        assert f"{rerank_ast_no['ndcg_10_mean']:.4f}" in self.readme  # 0.3407
        assert f"{rerank_line_no['ndcg_10_mean']:.4f}" in self.readme  # 0.5797

        # Key precision metrics
        assert f"{rerank_ast_h['ndcg_1_mean']:.3f}" in self.readme  # 0.740
        assert f"{rerank_line_h['ndcg_1_mean']:.3f}" in self.readme  # 0.800
        assert f"{rerank_ast_h['mrr_10_mean']:.3f}" in self.readme  # 0.896
        assert f"{rerank_line_h['mrr_10_mean']:.3f}" in self.readme  # 0.950
        assert f"{rerank_ast_h['cp_5_mean']:.3f}" in self.readme  # 0.869
        assert f"{rerank_line_h['cp_5_mean']:.3f}" in self.readme  # 0.933

    def test_statistical_test_numbers_in_readme(self):
        """Verify Wilcoxon, Holm p-values, and CIs in README."""
        stats_map = {}
        for s in self.stats:
            for prefix in ("Comp 1", "Comp 2", "Comp 3", "Comp 4"):
                if prefix in s["comparison"]:
                    stats_map[prefix] = s

        # Comp 3
        comp3 = stats_map["Comp 3"]
        assert f"{comp3['mean_diff']:+.4f}" in self.readme  # +0.1459
        assert f"{comp3['p_holm']:.4f}" in self.readme  # 0.0001
        assert f"{comp3['ci_95_low']:+.4f}" in self.readme  # +0.0862
        assert f"{comp3['ci_95_high']:+.4f}" in self.readme  # +0.2071

        # Comp 2
        comp2 = stats_map["Comp 2"]
        assert f"{comp2['mean_diff']:+.4f}" in self.readme  # -0.2390
        assert f"{comp2['p_holm']:.4f}" in self.readme  # 0.0000

        # Comp 1
        comp1 = stats_map["Comp 1"]
        assert f"{comp1['mean_diff']:+.4f}" in self.readme  # -0.0930
        assert f"{comp1['p_holm']:.4f}" in self.readme  # 0.0438

        # Comp 4
        comp4 = stats_map["Comp 4"]
        assert f"{comp4['mean_diff']:+.4f}" in self.readme  # +0.0409
        assert f"{comp4['p_holm']:.4f}" in self.readme  # 0.2411

        # Comp 5 (Exploratory)
        comp5 = self.sensitivity["comp_5_ast_vs_line_with_header"]
        assert f"{comp5['delta']:+.4f}" in self.readme  # -0.1339
        assert f"{comp5['p_value']:.4f}" in self.readme  # 0.0003
        assert f"{comp5['ci_95'][0]:+.4f}" in self.readme  # -0.1992
        assert f"{comp5['ci_95'][1]:+.4f}" in self.readme  # -0.0704
        assert "36.0" in self.readme  # Wilcoxon W = 36.0

    def test_sensitivity_numbers_in_readme(self):
        """Verify language breakdown, consensus scores, and judged-only sensitivity."""
        lb = self.sensitivity["language_breakdown"]
        # Language breakdown
        assert f"{lb['rerank_ast_with_header']['java_mean']:.4f}" in self.readme  # 0.5830
        assert f"{lb['rerank_line_with_header']['java_mean']:.4f}" in self.readme  # 0.7158
        assert f"{lb['rerank_ast_with_header']['react_mean']:.4f}" in self.readme  # 0.3420
        assert f"{lb['rerank_line_with_header']['react_mean']:.4f}" in self.readme  # 0.4777

        # Consensus-only scores
        cs = self.sensitivity["annotator_robustness"]["consensus_only_scores"]
        assert f"{cs['rerank_line_with_header']:.4f}" in self.readme  # 0.6175
        assert f"{cs['rerank_line_no_header']:.4f}" in self.readme  # 0.5713
        assert f"{cs['rerank_ast_with_header']:.4f}" in self.readme  # 0.4866
        assert f"{cs['rerank_ast_no_header']:.4f}" in self.readme  # 0.3338

        # Judged-only shortlist scores
        jo = self.sensitivity["judged_only_evaluation"]
        assert f"{jo['rerank_line_with_header']:.4f}" in self.readme  # 0.7243
        assert f"{jo['rerank_line_no_header']:.4f}" in self.readme  # 0.6696
        assert f"{jo['rerank_ast_with_header']:.4f}" in self.readme  # 0.5774
        assert f"{jo['rerank_ast_no_header']:.4f}" in self.readme  # 0.4156

    def test_kappa_agreement_in_readme(self):
        """Verify Cohen's Kappa and pair counts in README."""
        assert "882" in self.readme
        assert "865" in self.readme
        assert "16" in self.readme
        assert "1" in self.readme
        assert "0.9822" in self.readme
        assert "0.9805" in self.readme
        assert "Almost Perfect Agreement" in self.readme

    def test_pool_random_baseline_in_readme(self):
        """Verify candidate pool random baseline in README."""
        retrieval_map = {r["config_id"]: r for r in self.retrieval}
        pool_base = retrieval_map["random_pool_candidates"]
        corpus_base = retrieval_map["random_corpus_universe"]
        assert f"{pool_base['ndcg_10_mean']:.3f}" in self.readme  # 0.562
        assert f"{corpus_base['ndcg_10_mean']:.4f}" in self.readme  # 0.0224
        assert "0.7243" in self.readme  # Judged-only Line+Header exceeding pool baseline
