"""
scripts/generate_clean_corpus.py - Regenerate chunk_corpus.parquet and fallback_audit.csv
using upgraded Tree-sitter AST parser.
"""

import csv
import json
import sys
from pathlib import Path
from collections import Counter
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "ai-engine"))
from parser.tree_sitter_loader import ast_loader

DATASET_DIR = ROOT_DIR / "dataset"
MANIFEST_FILE = DATASET_DIR / "manifest_repos.csv"
CORPUS_PARQUET = DATASET_DIR / "chunk_corpus.parquet"
FALLBACK_AUDIT_CSV = DATASET_DIR / "fallback_audit.csv"

def get_file_language(file_path: str):
    ext = file_path.split(".")[-1].lower()
    if ext == "java":
        return "java", "JAVA"
    elif ext in ("tsx", "jsx"):
        return "tsx", "REACT_TS"
    elif ext in ("ts", "mts"):
        return "typescript", "REACT_TS"
    else:
        return "javascript", "REACT_TS"

def line_based_chunking(
    file_path: str,
    source_code: str,
    window_size: int = 50,
    overlap: int = 10,
    min_loc: int = 6
):
    lines = source_code.splitlines()
    total_lines = len(lines)
    chunks = []
    if total_lines == 0:
        return chunks

    step = max(1, window_size - overlap)
    for start_idx in range(0, total_lines, step):
        end_idx = min(start_idx + window_size, total_lines)
        chunk_lines = lines[start_idx:end_idx]
        loc = len(chunk_lines)

        if loc < min_loc and total_lines >= min_loc:
            if end_idx == total_lines:
                break
            continue

        start_line = start_idx + 1
        end_line = end_idx
        chunk_text = "\n".join(chunk_lines)

        context_header = f"// File: {file_path}\n// Lines: {start_line}-{end_line}"
        text_with_header = f"{context_header}\n\n{chunk_text}"
        text_no_header = chunk_text

        chunks.append({
            "strategy": "LINE",
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "loc": loc,
            "chunk_type": "LINE_50",
            "class_name": None,
            "method_name": None,
            "context_header": context_header,
            "chunk_content": chunk_text,
            "text_with_header": text_with_header,
            "text_no_header": text_no_header,
            "raw_code": text_no_header,
            "header_code": text_with_header
        })

        if end_idx == total_lines:
            break

    return chunks

def ast_based_chunking(
    file_path: str,
    source_code: str,
    language: str,
    min_loc: int = 6,
    max_loc: int = 85
):
    raw_chunks = ast_loader.parse_and_chunk(file_path, source_code, language=language, include_export_prefix=True)
    filtered_chunks = []

    for c in raw_chunks:
        start_line = c["start_line"]
        end_line = c["end_line"]
        loc = end_line - start_line + 1

        if loc < min_loc and len(raw_chunks) > 1:
            continue
        if loc > max_loc:
            continue

        chunk_text = c["chunk_content"]
        header = c["context_header"]

        filtered_chunks.append({
            "strategy": "AST",
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "loc": loc,
            "chunk_type": c["chunk_type"],
            "class_name": c.get("class_name"),
            "method_name": c.get("method_name"),
            "context_header": header,
            "chunk_content": chunk_text,
            "text_with_header": f"{header}\n\n{chunk_text}",
            "text_no_header": chunk_text,
            "raw_code": chunk_text,
            "header_code": f"{header}\n\n{chunk_text}"
        })

    return filtered_chunks

def main():
    print("=" * 80)
    print("REGENERATING CLEAN CHUNK CORPUS & AUDIT (PHASE 1)")
    print("=" * 80)

    # 1. Reconstruct all 189 source files from existing corpus line chunks
    existing_corpus = pd.read_parquet(CORPUS_PARQUET)
    line_df = existing_corpus[existing_corpus["strategy"] == "LINE"]

    reconstructed = {}
    for (repo, fp), g in line_df.groupby(["repo_name", "file_path"]):
        g = g.sort_values("start_line")
        file_lines = {}
        for _, row in g.iterrows():
            s = row["start_line"]
            for idx, line in enumerate(row["chunk_content"].splitlines()):
                file_lines[s + idx] = line
        max_line = max(file_lines.keys())
        cat = g["category"].iloc[0]
        reconstructed[(repo, fp, cat)] = "\n".join([file_lines[l] for l in range(1, max_line + 1)])

    print(f"Loaded {len(reconstructed)} source files from universe.")
    assert len(reconstructed) == 189, f"Expected 189 files, found {len(reconstructed)}"

    all_ast_chunks = []
    all_line_chunks = []
    fallback_records = []

    for (repo, fp, cat), code in reconstructed.items():
        parser_lang, _ = get_file_language(fp)

        # Generate AST chunks
        ast_chunks = ast_based_chunking(fp, code, parser_lang, min_loc=6, max_loc=85)
        for c in ast_chunks:
            c["repo_name"] = repo
            c["category"] = cat
            c["chunk_id"] = f"ast::{repo}::{fp}::{c['start_line']}_{c['end_line']}::{c['chunk_type']}"
            all_ast_chunks.append(c)

            if "LINE_FALLBACK" in c["chunk_type"]:
                fallback_records.append({
                    "chunk_id": c["chunk_id"],
                    "repo_name": repo,
                    "file_path": fp,
                    "category": cat,
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                    "loc": c["loc"],
                    "diagnosed_fallback_reason": "Zero syntactic AST nodes extracted"
                })

        # Generate Line chunks
        line_chunks = line_based_chunking(fp, code, window_size=50, overlap=10, min_loc=6)
        for c in line_chunks:
            c["repo_name"] = repo
            c["category"] = cat
            c["chunk_id"] = f"line::{repo}::{fp}::{c['start_line']}_{c['end_line']}::{c['chunk_type']}"
            all_line_chunks.append(c)

    print(f"Generated AST chunks: {len(all_ast_chunks)}")
    print(f"Generated Line chunks: {len(all_line_chunks)}")
    print(f"Total fallback chunks: {len(fallback_records)} / {len(all_ast_chunks)}")

    fallback_rate = len(fallback_records) / len(all_ast_chunks) if all_ast_chunks else 0
    print(f"Overall Fallback Rate: {fallback_rate:.2%}")

    # Verify ID uniqueness
    all_chunks = all_ast_chunks + all_line_chunks
    ids = [c["chunk_id"] for c in all_chunks]
    assert len(ids) == len(set(ids)), f"Duplicate chunk IDs found: {len(ids)} vs {len(set(ids))}"

    # Save to Parquet
    df_new = pd.DataFrame(all_chunks)
    df_new.to_parquet(CORPUS_PARQUET, index=False)
    print(f"Saved regenerated corpus to {CORPUS_PARQUET} (Shape: {df_new.shape})")

    # Save fallback audit
    df_audit = pd.DataFrame(fallback_records, columns=[
        "chunk_id", "repo_name", "file_path", "category",
        "start_line", "end_line", "loc", "diagnosed_fallback_reason"
    ])
    df_audit.to_csv(FALLBACK_AUDIT_CSV, index=False)
    print(f"Saved fallback audit to {FALLBACK_AUDIT_CSV} ({len(df_audit)} rows)")

    # Save sampling frame log (Rule R10)
    SAMPLING_LOG = DATASET_DIR / "sampling_frame_log.json"
    sampling_log_data = {
        "universe": {
            "total_repos": 40,
            "total_files": 189,
            "manifest_file": "manifest_repos.csv"
        },
        "filters": {
            "ast_filter": "6 <= LOC <= 85 (No 1-chunk/file cap)",
            "line_filter": "Window 50 LOC, Overlap 10 LOC (Step 40), Min 6 LOC"
        },
        "corpus_totals": {
            "ast_chunks_count": len(all_ast_chunks),
            "line_chunks_count": len(all_line_chunks),
            "total_corpus_chunks": len(all_chunks)
        },
        "ast_chunk_types_breakdown": dict(Counter(c["chunk_type"] for c in all_ast_chunks))
    }
    with open(SAMPLING_LOG, "w", encoding="utf-8") as f:
        json.dump(sampling_log_data, f, indent=2, ensure_ascii=False)
    print(f"Saved sampling frame log to {SAMPLING_LOG}")

    # Print chunk breakdown
    print("\nAST Chunk Type Distribution:")
    for ctype, count in Counter(c["chunk_type"] for c in all_ast_chunks).most_common():
        print(f"  {ctype:30s}: {count:4d}")

if __name__ == "__main__":
    main()
