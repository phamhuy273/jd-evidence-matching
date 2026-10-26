"""tests/test_doc_hashes.py - Verifies that all 64-character SHA-256 hashes in documentation match expected locks.

Governed by Rule R17 and documentation integrity audit.
"""

import re
from pathlib import Path
import pytest
from tests.test_gold_integrity import LOCKED_GOLD_LABELS_SHA256, LOCKED_NORMALIZED_MASTER_SHA256

ROOT_DIR = Path(__file__).resolve().parent.parent
README_FILE = ROOT_DIR / "README.md"
PROVENANCE_FILE = ROOT_DIR / "PROVENANCE.md"

VALID_HASHES = {
    LOCKED_GOLD_LABELS_SHA256.upper(),
    LOCKED_NORMALIZED_MASTER_SHA256.upper(),
}


def test_doc_sha256_hashes_are_valid():
    """Extract all 64-character hex strings in documentation and verify they match locked hashes."""
    hex_pattern = re.compile(r"\b[0-9a-fA-F]{64}\b")

    for doc_path in [README_FILE, PROVENANCE_FILE]:
        assert doc_path.exists(), f"Missing file: {doc_path}"
        content = doc_path.read_text(encoding="utf-8")
        found_hashes = hex_pattern.findall(content)
        assert len(found_hashes) > 0, f"No 64-char hashes found in {doc_path.name}"

        for h in found_hashes:
            h_upper = h.upper()
            assert h_upper in VALID_HASHES, (
                f"Invalid or stale 64-char SHA-256 hash in {doc_path.name}: {h}\n"
                f"Allowed valid hashes: {VALID_HASHES}"
            )
