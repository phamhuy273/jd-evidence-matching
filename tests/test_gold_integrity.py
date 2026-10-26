"""tests/test_gold_integrity.py - Verifies integrity and immutability of ground_truth_final.csv.

Governed by Rule R17 of the Anti-Hardcoding Specification:
Gold labels cannot be modified under any circumstances. Lock SHA-256 hash and verify before any run.
"""

import csv
import hashlib
from pathlib import Path
import pytest
from sklearn.metrics import cohen_kappa_score

LOCKED_GOLD_LABELS_SHA256 = "6530C8FEC0D1190A7DB108E647E324EBFF3F0E6FB0A64814726D0F6049D904B6"
LOCKED_NORMALIZED_MASTER_SHA256 = "0799ABAFA394B706457638EB75F3D26AF0B40E2D2673465285A5D1BDDFDD754C"
ROOT_DIR = Path(__file__).resolve().parent.parent
GOLD_FILE = ROOT_DIR / "dataset" / "ground_truth_final.csv"


class TestGoldIntegrity:
    """Enforces Rule R17: Gold Ground Truth Immutability & Structural Validity."""

    def test_gold_file_exists(self):
        assert GOLD_FILE.exists(), f"Ground truth file missing: {GOLD_FILE}"

    def test_gold_sha256_hash_unmodified(self):
        # 1. Verify complete 882-row file hash on normalized newlines (\r\n -> \n)
        # to guarantee cross-platform determinism across Linux, macOS, and Windows git checkouts
        with open(GOLD_FILE, "rb") as f:
            norm_content = f.read().replace(b"\r\n", b"\n")
            computed_hash = hashlib.sha256(norm_content).hexdigest().upper()
        assert computed_hash == LOCKED_NORMALIZED_MASTER_SHA256, (
            f"VIOLATION: ground_truth_final.csv has been modified!\n"
            f"Expected (normalized): {LOCKED_NORMALIZED_MASTER_SHA256}\n"
            f"Actual:                 {computed_hash}"
        )

        # 2. Verify Rule R17: Ground truth labels remain strictly immutable
        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        labels_str = ",".join(r["ground_truth_label"] for r in rows)
        gold_labels_hash = hashlib.sha256(labels_str.encode()).hexdigest().upper()
        assert gold_labels_hash == LOCKED_GOLD_LABELS_SHA256, (
            f"VIOLATION OF RULE R17: The ground truth labels have been modified!\n"
            f"Expected: {LOCKED_GOLD_LABELS_SHA256}\n"
            f"Actual:   {gold_labels_hash}"
        )

    def test_gold_row_count_and_structure(self):
        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 882, f"Expected exactly 882 ground truth pairs, found {len(rows)}"

        # Check required columns
        required_cols = {"jd_id", "repo_name", "file_path", "start_line", "end_line", "ground_truth_label"}
        assert required_cols.issubset(set(reader.fieldnames or [])), (
            f"Missing required columns in gold file: {required_cols - set(reader.fieldnames or [])}"
        )

        # Check exactly 25 JDs
        jds = [r["jd_id"] for r in rows]
        unique_jds = sorted(list(set(jds)))
        assert len(unique_jds) == 25, f"Expected 25 unique JDs, found {len(unique_jds)}"

        java_jds = [j for j in unique_jds if "JAVA" in j]
        react_jds = [j for j in unique_jds if "REACT" in j]
        assert len(java_jds) == 15, f"Expected 15 Java JDs, found {len(java_jds)}"
        assert len(react_jds) == 10, f"Expected 10 React JDs, found {len(react_jds)}"

    def test_gold_label_domain_and_types(self):
        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader, start=2):
                raw_lbl = r.get("ground_truth_label")
                assert raw_lbl is not None, f"Row {idx}: missing ground_truth_label"
                assert raw_lbl in ("0", "1", "2"), f"Row {idx}: invalid label '{raw_lbl}', must be 0, 1, or 2"
                start = int(r["start_line"])
                end = int(r["end_line"])
                assert start >= 1, f"Row {idx}: invalid start_line {start}"
                assert end >= start, f"Row {idx}: end_line {end} < start_line {start}"

    def test_annotator_kappa_reproducible(self):
        # Verify quadratic weighted kappa on annotator files if present
        a1_file = ROOT_DIR / "dataset" / "ground_truth_annotator1.csv"
        a2_file = ROOT_DIR / "dataset" / "ground_truth_annotator2.csv"
        if a1_file.exists() and a2_file.exists():
            with open(a1_file, "r", encoding="utf-8") as f1, open(a2_file, "r", encoding="utf-8") as f2:
                rows1 = list(csv.DictReader(f1))
                rows2 = list(csv.DictReader(f2))
            assert len(rows1) == len(rows2) == 882
            labels1 = [int(r["human_label"]) for r in rows1]
            labels2 = [int(r["human_label"]) for r in rows2]
            kappa_all = cohen_kappa_score(labels1, labels2, weights="quadratic")
            assert kappa_all == pytest.approx(0.9822, abs=0.001), f"Unexpected overall Kappa {kappa_all}, expected ~0.9822"

    def test_unmatched_gold_pairs_count(self):
        """Enforces Phase 3: Zero unretrievable ground-truth pairs (0 orphaned items)."""
        import pandas as pd
        corpus_path = ROOT_DIR / "dataset" / "chunk_corpus.parquet"
        assert corpus_path.exists(), "chunk_corpus.parquet missing"
        corpus = pd.read_parquet(corpus_path)
        coords = {(r.repo_name, r.file_path, int(r.start_line), int(r.end_line)) for r in corpus.itertuples()}

        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        unmatched = [r for r in rows if (r["repo_name"], r["file_path"], int(r["start_line"]), int(r["end_line"])) not in coords]
        assert len(unmatched) == 0, f"Expected 0 unmatched pairs, found {len(unmatched)}"
