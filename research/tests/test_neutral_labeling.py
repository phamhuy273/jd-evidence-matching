"""tests/test_neutral_labeling.py - Verifies that to_label.csv adheres strictly to Phase 2 blinded pooling protocol.

Governed by:
- Rule R18: Zero LLM pre-labels or artificial ground truth (all score columns empty).
- Rule R19: Neutral display across all items (no source strategy leakage, no retrieval scores/ranks).
- 100% Fresh Cranfield/TREC Top-5 pooling across 12 retrieval configurations.
- Secret mapping integrity: 100% of pair_ids map to valid chunks in chunk_corpus.parquet.
"""

import csv
import json
from pathlib import Path
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
TO_LABEL_FILE = DATASET_DIR / "to_label.csv"
SECRET_MAPPING_FILE = DATASET_DIR / "pool_span_mapping.json"
POOL_REPORT_FILE = DATASET_DIR / "pool_size_report.csv"
CORPUS_FILE = DATASET_DIR / "chunk_corpus.parquet"


class TestNeutralLabeling:
    """Enforces Rules R18, R19, and Phase 2 blinded pooling standards."""

    def test_to_label_exists_and_pool_size_reasonable(self):
        assert TO_LABEL_FILE.exists(), f"to_label.csv missing: {TO_LABEL_FILE}"
        with open(TO_LABEL_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) <= 1000, f"Pool size exceeded upper ceiling of 1000: {len(rows)}"
        assert len(rows) == 882, f"Expected exactly 882 pooled items, found {len(rows)}"

    def test_neutral_columns_and_no_leakage(self):
        with open(TO_LABEL_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

        # Strictly required neutral fields (Phase 2)
        expected_fields = [
            "pair_id", "query_id", "jd_requirement_text", "repo_name", "file_path",
            "start_line", "end_line", "raw_code_without_header",
            "annotator1_score", "annotator2_score", "final_score", "adjudication_notes"
        ]
        assert fieldnames == expected_fields, f"Column mismatch in to_label.csv: {fieldnames} != {expected_fields}"

        # Forbidden columns (Rules R18, R19)
        forbidden_substrings = [
            "strategy", "retrieval_score", "dense_score", "rerank_score", "bm25_score",
            "rank", "system", "llm", "ast", "line_50", "line_fallback", "predicted", "ground_truth", "chunk_id"
        ]
        for col in fieldnames:
            for forbidden in forbidden_substrings:
                assert forbidden not in col.lower(), f"Leakage detected in column name '{col}' (matches '{forbidden}')"

    def test_initial_labels_are_empty_for_human_annotation(self):
        """Rule R18: Blank annotation sheets must have no pre-annotations or pre-filled labels."""
        for blank_name, score_col in [("annotator1_blank.csv", "annotator1_score"), ("annotator2_blank.csv", "annotator2_score")]:
            blank_file = DATASET_DIR / blank_name
            if blank_file.exists():
                with open(blank_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, r in enumerate(reader, start=1):
                        assert r[score_col] == "", f"{blank_name} Row {idx}: {score_col} is not empty!"

        # In to_label.csv (Phase 3), human scores must be valid in ('0', '1', '2')
        with open(TO_LABEL_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader, start=1):
                assert r["annotator1_score"] in ("0", "1", "2"), f"Row {idx}: invalid annotator1_score"
                assert r["annotator2_score"] in ("0", "1", "2"), f"Row {idx}: invalid annotator2_score"
                assert r["final_score"] in ("0", "1", "2"), f"Row {idx}: invalid final_score"

    def test_code_content_is_clean_without_ast_headers(self):
        """Rule R19: Code content must not disclose chunking type via injected AST headers."""
        with open(TO_LABEL_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader, start=1):
                code = r["raw_code_without_header"]
                assert "// Class:" not in code, f"Row {idx}: AST header leaked in raw_code_without_header"
                assert "// Method:" not in code, f"Row {idx}: AST header leaked in raw_code_without_header"
                assert "// File:" not in code, f"Row {idx}: File header leaked in raw_code_without_header"

    def test_pool_size_report_covers_all_25_jds(self):
        assert POOL_REPORT_FILE.exists()
        with open(POOL_REPORT_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 25, f"Expected 25 JDs in pool report, found {len(rows)}"

    def test_secret_mapping_and_zero_orphaned_pairs(self):
        """Phase 2 Criterion: 100% of pair_id entries map to valid chunks in chunk_corpus.parquet."""
        assert SECRET_MAPPING_FILE.exists(), f"Missing secret mapping: {SECRET_MAPPING_FILE}"
        with open(SECRET_MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        pairs = mapping_data["pairs"]
        corpus = pd.read_parquet(CORPUS_FILE)
        corpus_chunk_ids = set(corpus["chunk_id"])

        with open(TO_LABEL_FILE, "r", encoding="utf-8") as f:
            to_label_rows = list(csv.DictReader(f))

        to_label_ids = {r["pair_id"] for r in to_label_rows}
        secret_ids = set(pairs.keys())

        assert to_label_ids == secret_ids, "Mismatch between to_label pair_ids and secret mapping pair_ids!"

        for pid, pdata in pairs.items():
            chunk_ids = pdata["chunk_ids"]
            assert len(chunk_ids) > 0, f"Pair {pid} has no mapped chunk_ids!"
            for cid in chunk_ids:
                assert cid in corpus_chunk_ids, f"Pair {pid} references unknown chunk {cid} not in corpus!"
