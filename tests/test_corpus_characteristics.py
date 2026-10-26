"""tests/test_corpus_characteristics.py - Unit tests for 2x2 dual chunk corpus and Table 3.2.

Governed by:
- Rule R5: Both branches measured by identical functions without theoretical assignments.
- Rule R10: Sampling frame logging for AST and Line chunking.
- Rule R11: Uniqueness of chunk IDs across entire unified corpus.
- Rule R37: Cross-checking table_3_2.tex numbers against table_3_2_measured.json.
"""

import json
from pathlib import Path
import pytest

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
CORPUS_PARQUET = DATASET_DIR / "chunk_corpus.parquet"
SAMPLING_LOG = DATASET_DIR / "sampling_frame_log.json"
TABLE_3_2_JSON = DATASET_DIR / "table_3_2_measured.json"
TABLE_3_2_TEX = ROOT_DIR / "report" / "tables" / "table_T2.tex"


class TestCorpusCharacteristics:
    """Verifies Phase 2 corpus integrity, 4 text variants, and empirical Table 3.2 measurements."""

    def test_corpus_size_and_strategies(self):
        assert CORPUS_PARQUET.exists(), f"Corpus Parquet missing: {CORPUS_PARQUET}"
        df = pd.read_parquet(CORPUS_PARQUET).fillna("")
        chunks = df.to_dict(orient="records")

        assert len(chunks) == 872, f"Expected 872 unified chunks, got {len(chunks)}"

        ast_chunks = [c for c in chunks if c["strategy"] == "AST"]
        line_chunks = [c for c in chunks if c["strategy"] == "LINE"]
        assert len(ast_chunks) == 438, f"Expected 438 AST chunks, got {len(ast_chunks)}"
        assert len(line_chunks) == 434, f"Expected 434 Line chunks, got {len(line_chunks)}"

    def test_chunk_id_uniqueness(self):
        df = pd.read_parquet(CORPUS_PARQUET)
        chunk_ids = df["chunk_id"].tolist()
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found!"

    def test_four_text_variants_present(self):
        df = pd.read_parquet(CORPUS_PARQUET)
        chunks = df.to_dict(orient="records")

        for c in chunks:
            assert "text_with_header" in c and len(c["text_with_header"]) > 5
            assert "text_no_header" in c and len(c["text_no_header"]) > 5
            assert "raw_code" in c and len(c["raw_code"]) > 5
            assert "header_code" in c and len(c["header_code"]) > 5
            assert "// File:" in c["context_header"]

    def test_sampling_frame_log_completeness(self):
        assert SAMPLING_LOG.exists()
        with open(SAMPLING_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)

        assert log["universe"]["total_files"] == 189
        assert log["universe"]["total_repos"] == 40
        assert log["corpus_totals"]["ast_chunks_count"] == 438
        assert log["corpus_totals"]["line_chunks_count"] == 434

    def test_table_3_2_measured_values_and_no_zero_by_construction(self):
        assert TABLE_3_2_JSON.exists()
        with open(TABLE_3_2_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        ast = data["ast_progressive"]
        line = data["line_based"]

        # Rule R5: Measure both branches with identical functions
        assert line["boundary_cut_rate_pct"] > 0.0, "Line boundary cut rate must be measured"
        assert line["syntax_error_rate_pct"] > 0.0, "Line syntax error rate must be measured"

        # Rule B1: Verify all 3 AST subgroups
        assert "ast_pure" in data
        assert "ast_fallback" in data
        assert "ast_combined" in data
        pure = data["ast_pure"]
        fallback = data["ast_fallback"]

        assert pure["chunk_count"] == 438
        assert fallback["chunk_count"] == 0
        assert ast["chunk_count"] == 438
        assert line["chunk_count"] == 434
        assert data["metadata"]["fallback_rate_pct"] < 10.0

        # AST Pure header retention is 100%
        assert pure["header_retention_rate_pct"] == 100.0
        assert pure["loc_dist"]["mean"] == pytest.approx(26.7, abs=0.1)
        assert pure["boundary_intact_rate_pct"] == pytest.approx(87.2, abs=0.1)
        assert pure["syntax_intact_rate_pct"] == pytest.approx(78.1, abs=0.1)

        # Check LaTeX file contains the measured numbers (Rule R37)
        tex_text = TABLE_3_2_TEX.read_text(encoding="utf-8")
        assert "434" in tex_text
        assert "438" in tex_text
        assert f"{pure['loc_dist']['mean']:.1f}" in tex_text
        assert f"{line['loc_dist']['mean']:.1f}" in tex_text
        assert f"{pure['boundary_intact_rate_pct']:.1f}" in tex_text
        assert f"{ast['boundary_intact_rate_pct']:.1f}" in tex_text
        assert f"{line['boundary_intact_rate_pct']:.1f}" in tex_text

        # Cross-check against paper text
        approach_tex = (ROOT_DIR / "paper" / "section_approach.tex").read_text(encoding="utf-8")
        assert "87.2\\%" in approach_tex
        assert "434" in approach_tex
        assert "438" in approach_tex
        assert "872" in approach_tex
        assert "0.0\\%" in approach_tex

        table1_tex = (ROOT_DIR / "paper" / "tables" / "table_1_paper.tex").read_text(encoding="utf-8")
        assert "0.621" in table1_tex
        assert "0.580" in table1_tex
        assert "0.487" in table1_tex
        assert "0.341" in table1_tex
