"""tests/test_gold_integrity.py - Verifies integrity and immutability of ground_truth_final.csv.

Governed by Rule R17 of the Anti-Hardcoding Specification:
Gold labels cannot be modified under any circumstances. Lock SHA-256 hash and verify before any run.
"""

import csv
import hashlib
from pathlib import Path
import pytest
from sklearn.metrics import cohen_kappa_score

LOCKED_GOLD_LABELS_SHA256 = "F9E3011EFA003C6AAB7D6DFD46CE94E1FFDDC741F193C02E0748D39D67650531"
LOCKED_MASTER_SHA256 = "2260C21DEBBCE08EFD4ACB4F5ED2429120FDEC98D64954EAE7978FFE3647875B"
ROOT_DIR = Path(__file__).resolve().parent.parent
GOLD_FILE = ROOT_DIR / "dataset" / "ground_truth_final.csv"


class TestGoldIntegrity:
    """Enforces Rule R17: Gold Ground Truth Immutability & Structural Validity."""

    def test_gold_file_exists(self):
        assert GOLD_FILE.exists(), f"Ground truth file missing: {GOLD_FILE}"

    def test_gold_sha256_hash_unmodified(self):
        # 1. Verify complete 603-row file hash
        with open(GOLD_FILE, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest().upper()
        assert computed_hash == LOCKED_MASTER_SHA256, (
            f"VIOLATION: ground_truth_final.csv has been modified!\n"
            f"Expected: {LOCKED_MASTER_SHA256}\n"
            f"Actual:   {computed_hash}"
        )

        # 2. Verify Rule R17: 250 gold labels remain strictly immutable
        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        labels_str = ",".join(r["ground_truth_label"] for r in rows[:250])
        gold_labels_hash = hashlib.sha256(labels_str.encode()).hexdigest().upper()
        assert gold_labels_hash == LOCKED_GOLD_LABELS_SHA256, (
            f"VIOLATION OF RULE R17: The 250 gold labels have been modified!\n"
            f"Expected: {LOCKED_GOLD_LABELS_SHA256}\n"
            f"Actual:   {gold_labels_hash}"
        )

    def test_gold_row_count_and_structure(self):
        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 603, f"Expected exactly 603 ground truth pairs, found {len(rows)}"

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

        # The first 250 gold rows must have exactly 10 per JD
        gold_jds = [r["jd_id"] for r in rows[:250]]
        from collections import Counter
        counts = Counter(gold_jds)
        for jd, count in counts.items():
            assert count == 10, f"JD {jd} has {count} gold candidates, expected exactly 10"

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
            assert len(rows1) == len(rows2) == 603
            labels1 = [int(r["human_label"]) for r in rows1]
            labels2 = [int(r["human_label"]) for r in rows2]
            kappa_all = cohen_kappa_score(labels1, labels2, weights="quadratic")
            assert pytest.approx(kappa_all, abs=0.01) == 0.78, f"Unexpected overall Kappa {kappa_all}, expected ~0.78"

            # Gold subset (first 250)
            kappa_gold = cohen_kappa_score(labels1[:250], labels2[:250], weights="quadratic")
            assert pytest.approx(kappa_gold, abs=0.01) == 0.88, f"Unexpected Gold Kappa {kappa_gold}, expected ~0.88"

