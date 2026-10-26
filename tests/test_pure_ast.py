"""Smoke tests for the pure-AST pipeline (stub backend only; NO model results are produced here)."""
import subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"


def _py(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, check=True)


def test_universe_sizes(tmp_path):
    sys.path.insert(0, str(S))
    import pure_ast_run as r
    c, q = r.load(ROOT); U = r.universes(c)
    assert len(U["pure_ast"]) == 438 and len(U["ast_full"]) == 438 and len(U["line_full"]) == 434
    assert (U["pure_ast"].chunk_type != "LINE_FALLBACK").all()
    assert len(q) == 25


def test_stub_run_and_neutral_pool(tmp_path):
    out = tmp_path / "pa_stub"
    _py(S / "pure_ast_run.py", "--root", ROOT, "--backend", "stub", "--outdir", out)
    assert (out / "runs" / "rerank_pure_ast_with_header.json").exists()
    _py(S / "pure_ast_pool.py", "--root", ROOT, "--rundir", out / "runs", "--outdir", out / "pool")
    d = pd.read_csv(out / "pool" / "to_label_pure_ast.csv")
    assert not ({"chunk_type", "context_header", "systems", "score", "rank", "tier", "strategy"} & set(d.columns))


def test_stub_cannot_write_to_official_dir(tmp_path):
    p = subprocess.run([sys.executable, S / "pure_ast_run.py", "--root", ROOT, "--backend", "stub", "--outdir", tmp_path / "official"], capture_output=True, text=True)
    assert p.returncode != 0


def test_bpref_known_values(tmp_path):
    _py(S / "compute_bpref.py", "--root", ROOT, "--outdir", tmp_path)
    s = pd.read_csv(tmp_path / "bpref_summary.csv").set_index("config")
    assert abs(s.loc["bm25_line_no_header", "bpref"] - 0.246) < 0.005
    assert abs(s.loc["rerank_line_with_header", "bpref"] - 0.353) < 0.005
    assert s["bpref"].between(0, 1).all()


def test_header_ablation_stub_runs_and_skips_unmatched(tmp_path):
    out = tmp_path / "ha_stub"
    _py(S / "header_ablation.py", "--root", ROOT, "--backend", "stub", "--outdir", out, "--boot", "50")
    import json
    r = json.load(open(out / "header_ablation_results.json"))
    assert r["n_pairs_used"] + r["n_pairs_skipped"] == 882 and r["n_pairs_skipped"] == 0
    assert set(r["subsets"]) == {"all", "pure_ast"}
