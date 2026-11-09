"""tests/test_power_analysis.py - Unit test for statistical power analysis and MDE calculations.

Governed by Rule R26 of the Anti-Hardcoding Specification:
Verify MDE computation across 5 paired comparisons under two-sided t-test approximation.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT_DIR / "scripts" / "power_analysis.py"
OUTPUT_JSON = ROOT_DIR / "dataset" / "benchmark_results" / "power_analysis.json"
OUTPUT_TXT = ROOT_DIR / "dataset" / "benchmark_results" / "power_analysis.txt"


def test_power_analysis_execution_and_determinism():
    """Verify scripts/power_analysis.py runs cleanly and produces deterministic MDE bounds."""
    # Run script
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(ROOT_DIR)
    )
    assert result.returncode == 0
    assert OUTPUT_JSON.exists(), "power_analysis.json missing"
    assert OUTPUT_TXT.exists(), "power_analysis.txt missing"

    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "Comp 1" in data
    assert "Comp 5" in data
    assert "_summary" in data

    # Check MDE ranges
    mde_min = data["_summary"]["min_mde_0.05"]
    mde_max = data["_summary"]["max_mde_0.0125"]
    assert pytest.approx(mde_min, abs=1e-3) == 0.080
    assert pytest.approx(mde_max, abs=1e-3) == 0.130

    # Check Comp 1 MDE
    comp1 = data["Comp 1"]
    assert pytest.approx(comp1["mde_alpha_0.05"], abs=1e-3) == 0.107
    assert pytest.approx(comp1["mde_alpha_0.0125"], abs=1e-3) == 0.130
