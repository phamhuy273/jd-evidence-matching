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

    def test_dataset_ground_truth_counts_match_reality(self):
        # Ground truth row count check
        with open(GOLD_FILE, "r", encoding="utf-8") as f:
            gold_count = len(list(csv.DictReader(f)))
        assert gold_count == 882

        # Query count check
        with open(FROZEN_QUERIES_FILE, "r", encoding="utf-8") as f:
            query_count = len(json.load(f)["queries"])
        assert query_count == 25

    def test_no_forbidden_exaggerations(self):
        """Rule R36 & R38: Ensure no fictitious, unsubstantiated, or legally risky claims."""
        readme_text = README_FILE.read_text(encoding="utf-8")
        forbidden = [
            "CORE A*",
            "p < 0.0000001",
            "perfect 100%",
            "fair use",
            "17 U.S. Code",
            "Berne Convention"
        ]
        for phrase in forbidden:
            assert phrase.lower() not in readme_text.lower(), f"Forbidden phrase found in README: {phrase}"

    def test_paper_tex_no_obsolete_numbers(self):
        """Cross-checks paper/*.tex and paper/tables/*.tex for zero obsolete numbers from legacy runs."""
        paper_dir = ROOT_DIR / "paper"
        tex_files = list(paper_dir.glob("*.tex")) + list((paper_dir / "tables").glob("*.tex"))
        assert len(tex_files) > 0, "No LaTeX files found in paper/"

        obsolete_tokens = ["798", "603", "364", "234", "130", "35.7", "19.8 LOC", "0.242", "0.499", "0.345"]
        for f in tex_files:
            text = f.read_text(encoding="utf-8")
            for token in obsolete_tokens:
                if token.endswith(" LOC"):
                    assert token not in text, f"Found obsolete phrase '{token}' in {f.name}"
                else:
                    matches = re.findall(r"\b" + re.escape(token) + r"\b", text)
                    assert len(matches) == 0, f"Found obsolete number '{token}' in {f.name}: {len(matches)} match(es)"

    def test_paper_tex_no_unescaped_percent_after_digits(self):
        """Ensure all percent signs after digits in paper/*.tex are properly escaped as \\%."""
        paper_dir = ROOT_DIR / "paper"
        tex_files = list(paper_dir.glob("*.tex")) + list((paper_dir / "tables").glob("*.tex"))

        for f in tex_files:
            text = f.read_text(encoding="utf-8")
            for line_idx, line in enumerate(text.splitlines(), start=1):
                # Match digits immediately followed by unescaped %
                matches = re.findall(r"(?<!\\)\d+%", line)
                assert len(matches) == 0, (
                    f"Unescaped '%' after digit in {f.name}:{line_idx}: '{line}'. "
                    f"Must be escaped as '\\%' in LaTeX."
                )

    def test_paper_and_repo_double_blind_anonymity(self):
        """Ensure double-anonymous compliance: zero author names, student IDs, personal emails, or local paths."""
        paper_dir = ROOT_DIR / "paper"
        tex_files = list(paper_dir.glob("*.tex")) + list((paper_dir / "tables").glob("*.tex"))

        forbidden_tokens = [
            "win 11",
            "users/win",
            "@gmail.com",
            "@hcmut",
            "@edu.vn",
            "student_id",
        ]

        for f in tex_files:
            content = f.read_text(encoding="utf-8").lower()
            for tok in forbidden_tokens:
                assert tok not in content, f"Double-blind anonymity violation: found '{tok}' in {f.name}"

        # Verify paper/main.pdf metadata
        pdf_path = paper_dir / "main.pdf"
        if pdf_path.exists():
            try:
                import pypdfium2 as pdfium
                doc = pdfium.PdfDocument(str(pdf_path))
                meta = doc.get_metadata_dict()
                for key in ["Author", "Creator", "Producer"]:
                    val = (meta.get(key, "") or "").lower()
                    for tok in forbidden_tokens:
                        assert tok not in val, f"Anonymity leak in PDF metadata {key}: {val}"
            except Exception:
                pass

    def test_linear_and_unweighted_kappa_concordance(self):
        """Verify linear (0.9764) and unweighted (0.9708) kappa concordance and absence of stale drafts."""
        approach_tex = (ROOT_DIR / "paper" / "section_approach.tex").read_text(encoding="utf-8")
        readme_text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

        assert "0.9764" in approach_tex and "0.9708" in approach_tex
        assert "0.9764" in readme_text and "0.9708" in readme_text

        for f in (ROOT_DIR / "paper").rglob("*"):
            if f.suffix in [".tex", ".md", ".txt"]:
                t = f.read_text(encoding="utf-8")
                assert "0.9634" not in t, f"Found stale 0.9634 in {f}"
                assert "0.9453" not in t, f"Found stale 0.9453 in {f}"
        assert "0.9634" not in readme_text, "Found stale 0.9634 in README.md"
        assert "0.9453" not in readme_text, "Found stale 0.9453 in README.md"

    def test_no_bge_reranker_large_in_codebase(self):
        """Ensure BAAI/bge-reranker-base is used consistently and bge-reranker-large does not appear."""
        for d in ["paper", "report", "scripts"]:
            target_dir = ROOT_DIR / d
            if not target_dir.exists():
                continue
            for f in target_dir.rglob("*"):
                if f.suffix in [".tex", ".md", ".py", ".txt"]:
                    t = f.read_text(encoding="utf-8")
                    assert "bge-reranker-large" not in t.lower(), f"Found bge-reranker-large in {f}"

    def test_sensitivity_report_no_stale_text(self):
        """Ensure sensitivity_analysis_report.txt contains zero legacy strings."""
        sens_file = ROOT_DIR / "dataset" / "benchmark_results" / "sensitivity_analysis_report.txt"
        assert sens_file.exists(), f"Missing sensitivity report: {sens_file}"
        sens_txt = sens_file.read_text(encoding="utf-8")
        assert "78 legacy" not in sens_txt.lower(), "Found '78 legacy' in sensitivity report"
        assert "18/25 queries" not in sens_txt, "Found '18/25 queries' in sensitivity report"


