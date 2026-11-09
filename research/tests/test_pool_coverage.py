"""tests/test_pool_coverage.py - Verifies that ALL 12 configurations have 100% judged/pooled Top-5 coverage under Phase 2.

Governed by Phase 2:
- Unified Cranfield/TREC Top-5 pooling across ALL 12 systems (BM25, Dense, Rerank).
- 100% coverage at Top-5 for all 12 configurations in to_label.csv.
"""

import csv
import json
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
RUNS_DIR = DATASET_DIR / "retrieval_runs"
TO_LABEL_FILE = DATASET_DIR / "to_label.csv"
SECRET_MAPPING_FILE = DATASET_DIR / "pool_span_mapping.json"

ALL_12_CONFIGS = [
    "bm25_line_no_header",
    "bm25_line_with_header",
    "bm25_ast_no_header",
    "bm25_ast_with_header",
    "dense_line_no_header",
    "dense_line_with_header",
    "dense_ast_no_header",
    "dense_ast_with_header",
    "rerank_line_no_header",
    "rerank_line_with_header",
    "rerank_ast_no_header",
    "rerank_ast_with_header",
]


def test_all_12_configs_have_full_top5_pooled_coverage():
    """All 12 configurations (BM25, Dense, Rerank) must have 100% Top-5 coverage in to_label.csv."""
    assert TO_LABEL_FILE.exists(), f"Missing to_label.csv: {TO_LABEL_FILE}"
    with open(TO_LABEL_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        qid_col = "query_id" if "query_id" in (reader.fieldnames or []) else "jd_id"
        pool_set = {(r[qid_col], r["repo_name"], r["file_path"], int(r["start_line"]), int(r["end_line"])) for r in reader}

    for cfg in ALL_12_CONFIGS:
        run_file = RUNS_DIR / f"{cfg}.json"
        assert run_file.exists(), f"Missing run file: {run_file}"
        with open(run_file, "r", encoding="utf-8") as f:
            run_data = json.load(f)

        unpooled_top5 = 0
        total_top5 = 0
        for qid, items in run_data.items():
            for it in items[:5]:
                key = (qid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"]))
                total_top5 += 1
                if key not in pool_set:
                    unpooled_top5 += 1

        assert total_top5 == 125, f"Expected 125 top-5 items (25 queries * 5), found {total_top5}"
        assert unpooled_top5 == 0, f"Config {cfg} has {unpooled_top5} unpooled items in top-5!"


def test_pool_size_report_exact_concordance():
    """Verify pool_size_report.csv has 25 queries and matches to_label row count."""
    pool_report_file = DATASET_DIR / "pool_size_report.csv"
    assert pool_report_file.exists()
    with open(pool_report_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 25, f"Expected 25 queries in pool report, found {len(rows)}"
    total_unique = sum(int(r["new_to_label"]) for r in rows)

    with open(TO_LABEL_FILE, "r", encoding="utf-8") as f:
        to_label_rows = list(csv.DictReader(f))

    assert total_unique == len(to_label_rows), f"Report sum ({total_unique}) != to_label rows ({len(to_label_rows)})"


def test_all_12_configs_judged_at_3_and_5_are_100_percent():
    """Verify that pool_coverage_report.csv has exactly 1.0 Judged@3 and Judged@5 for all 12 systems."""
    coverage_file = DATASET_DIR / "benchmark_results" / "pool_coverage_report.csv"
    assert coverage_file.exists(), f"Missing coverage file: {coverage_file}"
    with open(coverage_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 12, f"Expected 12 configurations, found {len(rows)}"
    for r in rows:
        cfg = r["configuration"]
        assert float(r["judged_at_3"]) == 1.0, f"Config {cfg} Judged@3 is {r['judged_at_3']}, expected 1.0"
        assert float(r["judged_at_5"]) == 1.0, f"Config {cfg} Judged@5 is {r['judged_at_5']}, expected 1.0"
        assert int(r["top3_unjudged"]) == 0, f"Config {cfg} top3_unjudged is {r['top3_unjudged']}, expected 0"
        assert int(r["top5_unjudged"]) == 0, f"Config {cfg} top5_unjudged is {r['top5_unjudged']}, expected 0"

