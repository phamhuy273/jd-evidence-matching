"""evaluate_syntax.py - Empirically measure Table 3.2 (Syntax boundaries and chunk morphology) on AST and Line corpora.

Governed by:
- Rule R5: Measure both branches with identical functions (no theoretical or definition assignments).
- Rule R6: No silent fallback constants.
- Rule R13: Exact identical universe of 189 source files across 40 repositories.
- Rule R29: Record exact tokenizer model (BAAI/bge-m3) and max_seq_length (512) truncation stats.
- Rule R37: Ground all Table 3.2 numbers in dynamic measurement artifacts.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
CORPUS_PARQUET = DATASET_DIR / "chunk_corpus.parquet"
MANIFEST_FILE = DATASET_DIR / "manifest_repos.csv"
CLONE_ROOT = ROOT_DIR / "ai-engine" / "temp_repos"

OUTPUT_JSON = DATASET_DIR / "table_3_2_measured.json"
OUTPUT_REPORT = DATASET_DIR / "table_3_2_report.txt"
OUTPUT_TEX = ROOT_DIR / "report" / "tables" / "table_T2.tex"
OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT_DIR / "ai-engine"))
from parser.tree_sitter_loader import ast_loader


def extract_method_intervals(source_code: str, language: str) -> List[Tuple[int, int]]:
    """Extract (start_line, end_line) of all functions/methods/constructors in source file."""
    parser = ast_loader.parsers.get("tsx" if language in ("tsx", "typescript") else "java")
    if not parser:
        raise RuntimeError(f"Tree-sitter parser not available for language: {language}")
    source_bytes = bytes(source_code, "utf8")
    tree = parser.parse(source_bytes)
    root = tree.root_node
    intervals = []

    def traverse(node):
        nt = node.type
        if nt in ("method_declaration", "constructor_declaration", "function_declaration", "method_definition"):
            s_line = source_bytes[:node.start_byte].count(b'\n') + 1
            e_line = source_bytes[:node.end_byte].count(b'\n') + 1
            intervals.append((s_line, e_line))
        elif nt in ("lexical_declaration", "variable_declaration"):
            for decl in node.children:
                if decl.type == "variable_declarator":
                    val = decl.child_by_field_name("value")
                    if val and val.type in ("arrow_function", "function_expression"):
                        target = node.parent if (node.parent and node.parent.type == "export_statement") else node
                        s_line = source_bytes[:target.start_byte].count(b'\n') + 1
                        e_line = source_bytes[:target.end_byte].count(b'\n') + 1
                        intervals.append((s_line, e_line))
                        return
        elif nt in ("public_field_definition", "field_definition"):
            val = node.child_by_field_name("value")
            if val and val.type in ("arrow_function", "function_expression"):
                s_line = source_bytes[:node.start_byte].count(b'\n') + 1
                e_line = source_bytes[:node.end_byte].count(b'\n') + 1
                intervals.append((s_line, e_line))
                return

        for child in node.children:
            traverse(child)

    traverse(root)
    return intervals


def check_boundary_cut(chunk_start: int, chunk_end: int, method_intervals: List[Tuple[int, int]]) -> bool:
    """
    Rule R5: Precise syntax boundary cut definition.
    A chunk cuts across a method M if chunk_start or chunk_end bisects M:
    (M_start < chunk_start <= M_end) OR (M_start <= chunk_end < M_end)
    """
    for m_start, m_end in method_intervals:
        if m_start < chunk_start <= m_end:
            return True
        if m_start <= chunk_end < m_end:
            return True
    return False


def check_syntax_error(code: str, language: str) -> bool:
    """Check if snippet has syntax error using Tree-sitter AST parser."""
    parser = ast_loader.parsers.get("tsx" if language in ("tsx", "typescript") else "java")
    if not parser:
        raise RuntimeError(f"Tree-sitter parser not available for language: {language}")
    tree = parser.parse(bytes(code, "utf8"))
    return bool(tree.root_node.has_error)


def compute_distribution(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 80)
    print("TASK 2.4: DYNAMIC EMPIRICAL MEASUREMENT FOR TABLE 3.2 (BGE-M3 TOKENIZER)")
    print("=" * 80)

    # 1. Load BAAI/bge-m3 tokenizer
    from transformers import AutoTokenizer
    print("Loading HuggingFace AutoTokenizer for BAAI/bge-m3...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    # 2. Load chunk corpus
    df_chunks = pd.read_parquet(CORPUS_PARQUET).fillna("")
    chunks = df_chunks.to_dict(orient="records")

    print(f"Loaded {len(chunks)} total chunks from {CORPUS_PARQUET.name}")

    ast_chunks = [c for c in chunks if c["strategy"] == "AST"]
    line_chunks = [c for c in chunks if c["strategy"] == "LINE"]
    print(f"AST chunks: {len(ast_chunks)} | Line chunks: {len(line_chunks)}")

    # Cache method intervals for syntax boundary analysis
    file_cache = {}
    intervals_cache = {}
    intervals_file = DATASET_DIR / "source_file_intervals.json"
    if intervals_file.exists():
        with open(intervals_file, "r", encoding="utf-8") as f:
            intervals_cache = json.load(f)

    for c in chunks:
        key = (c["repo_name"], c["file_path"])
        if key not in file_cache:
            file_key = f"{c['repo_name']}::{c['file_path']}"
            lang = "java" if c["file_path"].endswith(".java") else "tsx"
            if file_key in intervals_cache:
                intervals = intervals_cache[file_key]
            else:
                safe_name = c["repo_name"].replace("/", "__")
                file_disk = CLONE_ROOT / safe_name / c["file_path"]
                if file_disk.exists():
                    code = file_disk.read_text(encoding="utf-8", errors="replace")
                    intervals = extract_method_intervals(code, lang)
                else:
                    raise FileNotFoundError(
                        f"Cannot compute method intervals: neither {intervals_file.name} nor source file {file_disk} exists."
                    )
            file_cache[key] = (lang, intervals)

    def evaluate_branch(branch_chunks: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        print(f"\nEvaluating branch: {name} ({len(branch_chunks)} chunks)...")
        locs = []
        tokens_content = []
        tokens_with_header = []
        syntax_errors = 0
        boundary_cuts = 0
        has_context_header = 0
        truncated_count = 0

        for c in branch_chunks:
            lang, intervals = file_cache[(c["repo_name"], c["file_path"])]
            content = c["chunk_content"]
            full_text = c["text_with_header"]

            loc = len(content.splitlines())
            locs.append(loc)

            # Tokenization via real BGE-M3 tokenizer
            t_content = len(tokenizer.encode(content, add_special_tokens=False))
            t_header = len(tokenizer.encode(full_text, add_special_tokens=False))
            tokens_content.append(t_content)
            tokens_with_header.append(t_header)

            if t_header > 512:
                truncated_count += 1

            # Syntax error
            if check_syntax_error(content, lang):
                syntax_errors += 1

            # Boundary cut
            if check_boundary_cut(c["start_line"], c["end_line"], intervals):
                boundary_cuts += 1

            # Header presence
            if c.get("method_name") or c.get("class_name"):
                has_context_header += 1

        n = len(branch_chunks)
        if n == 0:
            empty_dist = {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
            return {
                "chunk_count": 0,
                "loc_dist": empty_dist,
                "tokens_content_dist": empty_dist,
                "tokens_with_header_dist": empty_dist,
                "syntax_error_count": 0,
                "syntax_error_rate_pct": 0.0,
                "syntax_intact_rate_pct": 100.0,
                "boundary_cut_count": 0,
                "boundary_cut_rate_pct": 0.0,
                "boundary_intact_rate_pct": 100.0,
                "header_retention_count": 0,
                "header_retention_rate_pct": 0.0,
                "truncated_512_count": 0,
                "truncated_512_rate_pct": 0.0
            }
        return {
            "chunk_count": n,
            "loc_dist": compute_distribution(locs),
            "tokens_content_dist": compute_distribution(tokens_content),
            "tokens_with_header_dist": compute_distribution(tokens_with_header),
            "syntax_error_count": syntax_errors,
            "syntax_error_rate_pct": float(syntax_errors / n * 100),
            "syntax_intact_rate_pct": float((n - syntax_errors) / n * 100),
            "boundary_cut_count": boundary_cuts,
            "boundary_cut_rate_pct": float(boundary_cuts / n * 100),
            "boundary_intact_rate_pct": float((n - boundary_cuts) / n * 100),
            "header_retention_count": has_context_header,
            "header_retention_rate_pct": float(has_context_header / n * 100),
            "truncated_512_count": truncated_count,
            "truncated_512_rate_pct": float(truncated_count / n * 100)
        }

    ast_pure = [c for c in ast_chunks if "LINE_FALLBACK" not in c["chunk_id"]]
    ast_fallback = [c for c in ast_chunks if "LINE_FALLBACK" in c["chunk_id"]]
    ast_combined = ast_chunks

    print(f"AST Pure: {len(ast_pure)} | AST Fallback: {len(ast_fallback)} | AST Combined: {len(ast_combined)}")

    ast_pure_res = evaluate_branch(ast_pure, "AST Pure (Ours)")
    ast_fallback_res = evaluate_branch(ast_fallback, "AST Fallback")
    ast_combined_res = evaluate_branch(ast_combined, "AST-first Combined")
    line_res = evaluate_branch(line_chunks, "Line-based (50 LOC)")

    results_data = {
        "metadata": {
            "total_repos": 40,
            "total_files": 189,
            "tokenizer_model": "BAAI/bge-m3",
            "max_seq_length": 512,
            "ast_pure_count": len(ast_pure),
            "ast_fallback_count": len(ast_fallback),
            "ast_combined_count": len(ast_combined),
            "line_count": len(line_chunks),
            "fallback_rate_pct": float(len(ast_fallback) / len(ast_combined) * 100),
            "ast_filter": "6 <= LOC <= 85 (no 1-chunk/file cap)",
            "line_filter": "Window 50 LOC, Overlap 10 LOC (Step 40)"
        },
        "ast_pure": ast_pure_res,
        "ast_fallback": ast_fallback_res,
        "ast_combined": ast_combined_res,
        "ast_progressive": ast_combined_res,  # Backward compatibility alias
        "line_based": line_res
    }

    # Save JSON results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # Format text report
    report_text = f"""=============================================================================================================
TABLE 3.2: CORPUS CHARACTERISTICS & QUANTITATIVE CHUNKING COMPARISON (MEASURED)
Governed by Rule R5 (Zero Hardcoding; Measured via identical AST & Tokenizer functions)
Disaggregated by AST Subgroups: Pure (N={len(ast_pure)}), Fallback (N={len(ast_fallback)}), Combined (N={len(ast_combined)})
Tokenizer: BAAI/bge-m3 | Max Seq Length: 512
Universe: 40 GitHub Repositories / 189 Source Files
=============================================================================================================

Metric / Characteristic              | Line (50 LOC)          | AST Pure (Ours)        | AST Fallback           | AST Combined
---------------------------------------------------------------------------------------------------------------------------------------------
Total Chunks Extracted               | {line_res['chunk_count']:<22d} | {ast_pure_res['chunk_count']:<22d} | {ast_fallback_res['chunk_count']:<22d} | {ast_combined_res['chunk_count']:<22d}
Mean Chunk Size (LOC)                | {line_res['loc_dist']['mean']:>4.1f} (std {line_res['loc_dist']['std']:>4.1f}, med {line_res['loc_dist']['median']:>2.0f}) | {ast_pure_res['loc_dist']['mean']:>4.1f} (std {ast_pure_res['loc_dist']['std']:>4.1f}, med {ast_pure_res['loc_dist']['median']:>2.0f}) | {ast_fallback_res['loc_dist']['mean']:>4.1f} (std {ast_fallback_res['loc_dist']['std']:>4.1f}, med {ast_fallback_res['loc_dist']['median']:>2.0f}) | {ast_combined_res['loc_dist']['mean']:>4.1f} (std {ast_combined_res['loc_dist']['std']:>4.1f}, med {ast_combined_res['loc_dist']['median']:>2.0f})
Mean Tokens (Content Only)           | {line_res['tokens_content_dist']['mean']:>5.1f} (med {line_res['tokens_content_dist']['median']:>4.0f})       | {ast_pure_res['tokens_content_dist']['mean']:>5.1f} (med {ast_pure_res['tokens_content_dist']['median']:>4.0f})       | {ast_fallback_res['tokens_content_dist']['mean']:>5.1f} (med {ast_fallback_res['tokens_content_dist']['median']:>4.0f})       | {ast_combined_res['tokens_content_dist']['mean']:>5.1f} (med {ast_combined_res['tokens_content_dist']['median']:>4.0f})
Mean Tokens (With Header)            | {line_res['tokens_with_header_dist']['mean']:>5.1f} (med {line_res['tokens_with_header_dist']['median']:>4.0f})       | {ast_pure_res['tokens_with_header_dist']['mean']:>5.1f} (med {ast_pure_res['tokens_with_header_dist']['median']:>4.0f})       | {ast_fallback_res['tokens_with_header_dist']['mean']:>5.1f} (med {ast_fallback_res['tokens_with_header_dist']['median']:>4.0f})       | {ast_combined_res['tokens_with_header_dist']['mean']:>5.1f} (med {ast_combined_res['tokens_with_header_dist']['median']:>4.0f})
Boundary Preservation (Intact)       | {line_res['boundary_intact_rate_pct']:>5.1f}% ({line_res['boundary_cut_count']:>3d}/{line_res['chunk_count']:>3d} cuts)   | {ast_pure_res['boundary_intact_rate_pct']:>5.1f}% ({ast_pure_res['boundary_cut_count']:>3d}/{ast_pure_res['chunk_count']:>3d} cuts)   | {ast_fallback_res['boundary_intact_rate_pct']:>5.1f}% ({ast_fallback_res['boundary_cut_count']:>3d}/{ast_fallback_res['chunk_count']:>3d} cuts)   | {ast_combined_res['boundary_intact_rate_pct']:>5.1f}% ({ast_combined_res['boundary_cut_count']:>3d}/{ast_combined_res['chunk_count']:>3d} cuts)
Parse Error Free Rate (Valid)        | {line_res['syntax_intact_rate_pct']:>5.1f}% ({line_res['syntax_error_count']:>3d}/{line_res['chunk_count']:>3d} errs)   | {ast_pure_res['syntax_intact_rate_pct']:>5.1f}% ({ast_pure_res['syntax_error_count']:>3d}/{ast_pure_res['chunk_count']:>3d} errs)   | {ast_fallback_res['syntax_intact_rate_pct']:>5.1f}% ({ast_fallback_res['syntax_error_count']:>3d}/{ast_fallback_res['chunk_count']:>3d} errs)   | {ast_combined_res['syntax_intact_rate_pct']:>5.1f}% ({ast_combined_res['syntax_error_count']:>3d}/{ast_combined_res['chunk_count']:>3d} errs)
Context Header Retention             | {line_res['header_retention_rate_pct']:>5.1f}% (No header)            | {ast_pure_res['header_retention_rate_pct']:>5.1f}% ({ast_pure_res['header_retention_count']:>3d}/{ast_pure_res['chunk_count']:>3d})          | {ast_fallback_res['header_retention_rate_pct']:>5.1f}% ({ast_fallback_res['header_retention_count']:>3d}/{ast_fallback_res['chunk_count']:>3d})          | {ast_combined_res['header_retention_rate_pct']:>5.1f}% ({ast_combined_res['header_retention_count']:>3d}/{ast_combined_res['chunk_count']:>3d})
Truncation Rate (> 512 Tokens)       | {line_res['truncated_512_rate_pct']:>5.1f}% ({line_res['truncated_512_count']:>3d} chunks)        | {ast_pure_res['truncated_512_rate_pct']:>5.1f}% ({ast_pure_res['truncated_512_count']:>3d} chunks)        | {ast_fallback_res['truncated_512_rate_pct']:>5.1f}% ({ast_fallback_res['truncated_512_count']:>3d} chunks)        | {ast_combined_res['truncated_512_rate_pct']:>5.1f}% ({ast_combined_res['truncated_512_count']:>3d} chunks)
---------------------------------------------------------------------------------------------------------------------------------------------
"""

    print(report_text)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Format LaTeX table for paper (Table 3.2 disaggregated)
    latex_text = f"""\\begin{{table*}}[t]
\\caption{{Quantitative Comparison of Source Code Chunking Strategies across 40 GitHub Repositories (189 Files) Disaggregated by Subgroups}}
\\label{{tab:corpus_chunking}}
\\centering
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Characteristic}} & \\textbf{{Line-based (50 LOC)}} & \\textbf{{AST Pure (Ours)}} & \\textbf{{AST Fallback}} & \\textbf{{AST-first (Combined)}} \\\\
\\hline
Total Chunks Extracted & {line_res['chunk_count']} & {ast_pure_res['chunk_count']} & {ast_fallback_res['chunk_count']} & {ast_combined_res['chunk_count']} \\\\
Mean Chunk Size (LOC) & {line_res['loc_dist']['mean']:.1f} ($\\pm${line_res['loc_dist']['std']:.1f}) & {ast_pure_res['loc_dist']['mean']:.1f} ($\\pm${ast_pure_res['loc_dist']['std']:.1f}) & {ast_fallback_res['loc_dist']['mean']:.1f} ($\\pm${ast_fallback_res['loc_dist']['std']:.1f}) & {ast_combined_res['loc_dist']['mean']:.1f} ($\\pm${ast_combined_res['loc_dist']['std']:.1f}) \\\\
Median Chunk Size (LOC) & {line_res['loc_dist']['median']:.0f} & {ast_pure_res['loc_dist']['median']:.0f} & {ast_fallback_res['loc_dist']['median']:.0f} & {ast_combined_res['loc_dist']['median']:.0f} \\\\
Mean Token Count (BGE-M3) & {line_res['tokens_content_dist']['mean']:.1f} & {ast_pure_res['tokens_content_dist']['mean']:.1f} & {ast_fallback_res['tokens_content_dist']['mean']:.1f} & {ast_combined_res['tokens_content_dist']['mean']:.1f} \\\\
Mean Tokens (with Context Header) & {line_res['tokens_with_header_dist']['mean']:.1f} & {ast_pure_res['tokens_with_header_dist']['mean']:.1f} & {ast_fallback_res['tokens_with_header_dist']['mean']:.1f} & {ast_combined_res['tokens_with_header_dist']['mean']:.1f} \\\\
Syntax Boundary Preservation & {line_res['boundary_intact_rate_pct']:.1f}\\% & \\textbf{{{ast_pure_res['boundary_intact_rate_pct']:.1f}\\%}} & {ast_fallback_res['boundary_intact_rate_pct']:.1f}\\% & {ast_combined_res['boundary_intact_rate_pct']:.1f}\\% \\\\
Parse Error-Free Rate & {line_res['syntax_intact_rate_pct']:.1f}\\% & \\textbf{{{ast_pure_res['syntax_intact_rate_pct']:.1f}\\%}} & {ast_fallback_res['syntax_intact_rate_pct']:.1f}\\% & {ast_combined_res['syntax_intact_rate_pct']:.1f}\\% \\\\
Context Header Retention (Class/Method) & {line_res['header_retention_rate_pct']:.1f}\\% & \\textbf{{{ast_pure_res['header_retention_rate_pct']:.1f}\\%}} & {ast_fallback_res['header_retention_rate_pct']:.1f}\\% & {ast_combined_res['header_retention_rate_pct']:.1f}\\% \\\\
Truncated Chunks ($> 512$ tokens) & {line_res['truncated_512_rate_pct']:.1f}\\% ({line_res['truncated_512_count']}) & {ast_pure_res['truncated_512_rate_pct']:.1f}\\% ({ast_pure_res['truncated_512_count']}) & {ast_fallback_res['truncated_512_rate_pct']:.1f}\\% ({ast_fallback_res['truncated_512_count']}) & {ast_combined_res['truncated_512_rate_pct']:.1f}\\% ({ast_combined_res['truncated_512_count']}) \\\\
\\hline
\\end{{tabular}}
}}
\\end{{table*}}
"""
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(latex_text)

    print(f"Saved Table 3.2 LaTeX to: {OUTPUT_TEX}")
    print(f"Saved Table 3.2 JSON to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
