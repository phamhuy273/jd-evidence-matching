"""tests/test_reproducibility.py - Determinism and seed consistency tests.

Governed by Rule R33:
Ensure statistical operations (bootstrap CI with fixed seed 42) produce strictly identical
numerical results when run repeatedly.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "scripts"))
from evaluate_retrieval import bootstrap_ci_paired, holm_bonferroni_correction


class TestReproducibility:
    """Verifies that randomized statistical operations in evaluate_retrieval are 100% deterministic under fixed seeds."""

    def test_bootstrap_ci_paired_determinism(self):
        a = np.array([0.80, 0.75, 0.90, 0.85, 0.70, 0.95, 0.88, 0.82, 0.79, 0.91] * 2)
        b = np.array([0.70, 0.65, 0.80, 0.75, 0.68, 0.85, 0.78, 0.72, 0.71, 0.80] * 2)

        mean1, low1, high1 = bootstrap_ci_paired(a, b, n_boot=1000, seed=42)
        mean2, low2, high2 = bootstrap_ci_paired(a, b, n_boot=1000, seed=42)

        assert mean1 == mean2
        assert low1 == low2
        assert high1 == high2

    def test_holm_bonferroni_determinism_and_properties(self):
        p_vals = [0.001, 0.04, 0.02, 0.80]
        adj1 = holm_bonferroni_correction(p_vals)
        adj2 = holm_bonferroni_correction(p_vals)

        assert adj1 == adj2
        # Adjusted p-values must be >= raw p-values and <= 1.0
        for raw, adj in zip(p_vals, adj1):
            assert adj >= raw
            assert adj <= 1.0
