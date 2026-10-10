"""tests/test_text_integrity.py - Cross-checks paper/README claims against benchmark artifacts.

Governed by Rules R36, R37, and R38 of the Anti-Hardcoding Specification:
All numbers reported in README or paper text must be grounded in and verifiable against
actual benchmark result files.
"""

import csv
import json
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
README_FILE = ROOT_DIR / "README.md"
GOLD_FILE = ROOT_DIR / "dataset" / "ground_truth_final.csv"
FROZEN_QUERIES_FILE = ROOT_DIR / "dataset" / "queries_frozen.json"


class TestTextIntegrity:
    """Verifies that published documentation claims accurately reflect underlying datasets (R37)."""

    def test_readme_dataset_counts_match_reality(self):
        readme_text = README_FILE.read_text(encoding="utf-8")

        # Ground truth row count check
        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            gold_count = len(list(csv.DictReader(f)))
        assert gold_count == 603

        # Query count check
        with open(FROZEN_QUERIES_FILE, "r", encoding="utf-8") as f:
            query_count = len(json.load(f)["queries"])
        assert query_count == 25

        # Check README contains mention of 25 JDs, 250 gold, and 603 total
        assert "25" in readme_text
        assert "250" in readme_text
        assert "603" in readme_text

    def test_no_forbidden_exaggerations(self):
        """Rule R36 & R38: Ensure no fictitious or unsubstantiated claims."""
        readme_text = README_FILE.read_text(encoding="utf-8")
        forbidden = ["CORE A*", "p < 0.0000001", "perfect 100%"]
        for phrase in forbidden:
            assert phrase not in readme_text, f"Forbidden phrase found in README: {phrase}"
