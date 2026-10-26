#!/usr/bin/env python3
"""scripts/execute_phase2_retrieval_and_pool.py - Phase 2: 12-Run Retrieval & Blinded Pooling.

Governed by:
- 100% fresh, single-wave, blinded Cranfield/TREC pooling protocol (Rule R18, R19, R44).
- 12 retrieval configurations: (2 chunking strategies x 2 header conditions x 3 tiers).
- Unified Top-5 candidate pooling across ALL 12 configurations.
- Span-level deduplication: (query_id, repo_name, file_path, start_line, end_line).
- Strict blinding: Neutral sequential pair_id (P0001, P0002, ...), no strategy/header/score leakage.
- Separate secret mapping: dataset/pool_span_mapping.json.
- Separate blank annotation sheets: dataset/annotator1_blank.csv, dataset/annotator2_blank.csv.
- Ergonomic sorting by (query_id, repo_name, file_path, start_line, end_line).
"""

import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
CORPUS_PARQUET = DATASET_DIR / "chunk_corpus.parquet"
QUERIES_FROZEN_FILE = DATASET_DIR / "queries_frozen.json"

CACHE_EMB_DIR = DATASET_DIR / "cache_embeddings"
RETRIEVAL_RUNS_DIR = DATASET_DIR / "retrieval_runs"
CACHE_EMB_DIR.mkdir(parents=True, exist_ok=True)
RETRIEVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

TO_LABEL_FILE = DATASET_DIR / "to_label.csv"
ANNOTATOR1_BLANK_FILE = DATASET_DIR / "annotator1_blank.csv"
ANNOTATOR2_BLANK_FILE = DATASET_DIR / "annotator2_blank.csv"
SECRET_MAPPING_FILE = DATASET_DIR / "pool_span_mapping.json"
POOL_REPORT_FILE = DATASET_DIR / "pool_size_report.csv"

SEED = 42
MAX_SEQ_LENGTH = 512
TOP_RETRIEVAL_DEPTH = 20
POOL_TOP_K = 5

CONFIGS = [
    "bm25_ast_no_header",
    "bm25_ast_with_header",
    "bm25_line_no_header",
    "bm25_line_with_header",
    "dense_ast_no_header",
    "dense_ast_with_header",
    "dense_line_no_header",
    "dense_line_with_header",
    "rerank_ast_no_header",
    "rerank_ast_with_header",
    "rerank_line_no_header",
    "rerank_line_with_header",
]


def tokenize_code(text: str) -> List[str]:
    """Tokenize code text for BM25 retrieval."""
    return [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)]


def compute_hash(texts: List[str], model_name: str, max_seq_length: int) -> str:
    """Deterministic composite SHA-256 hash of inputs for embedding caching."""
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(str(max_seq_length).encode("utf-8"))
    for t in texts:
        hasher.update(t.encode("utf-8"))
    return hasher.hexdigest()


def get_cached_or_compute_embeddings(
    cache_name: str,
    texts: List[str],
    model: SentenceTransformer,
    model_name: str = "BAAI/bge-m3"
) -> np.ndarray:
    """Retrieve cached embeddings if SHA-256 hash matches; otherwise compute and cache."""
    current_hash = compute_hash(texts, model_name, MAX_SEQ_LENGTH)
    npy_path = CACHE_EMB_DIR / f"{cache_name}.npy"
    meta_path = CACHE_EMB_DIR / f"{cache_name}_meta.json"

    if npy_path.exists() and meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("input_hash") == current_hash and meta.get("count") == len(texts):
                print(f"  [CACHE HIT] Loaded {cache_name} ({len(texts)} embeddings)")
                return np.load(npy_path)
            else:
                print(f"  [CACHE STALE] Hash mismatch for {cache_name}, recomputing...")
        except Exception as e:
            print(f"  [CACHE ERROR] Reading {cache_name} failed ({e}), recomputing...")

    print(f"  [COMPUTE] Encoding {len(texts)} items for {cache_name} with {model_name}...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    np.save(npy_path, embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"name": cache_name, "model": model_name, "count": len(texts), "input_hash": current_hash}, f, indent=2)
    return embeddings


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 80)
    print("PHASE 2: 12-RUN RETRIEVAL & 100% FRESH BLINDED POOL GENERATION")
    print("=" * 80)

    # 1. Load frozen queries
    with open(QUERIES_FROZEN_FILE, "r", encoding="utf-8-sig") as f:
        frozen_data = json.load(f)
    queries_dict = frozen_data["queries"]
    jd_ids = sorted(list(queries_dict.keys()))
    query_texts = [queries_dict[jid]["primary_query"] for jid in jd_ids]
    print(f"Loaded {len(jd_ids)} frozen queries.")

    # 2. Load chunk corpus
    corpus = pd.read_parquet(CORPUS_PARQUET)
    print(f"Loaded {len(corpus)} total chunks from {CORPUS_PARQUET.name}.")

    ast_df = corpus[corpus["strategy"] == "AST"].reset_index(drop=True)
    line_df = corpus[corpus["strategy"] == "LINE"].reset_index(drop=True)
    print(f"  AST chunks: {len(ast_df)} (Pure: {(ast_df['chunk_type'] != 'LINE_FALLBACK').sum()})")
    print(f"  Line chunks: {len(line_df)}")

    # 3. Setup conditions
    conditions = {
        "ast_with_header": {
            "df": ast_df,
            "texts": ast_df["header_code"].tolist()
        },
        "ast_no_header": {
            "df": ast_df,
            "texts": ast_df["raw_code"].tolist()
        },
        "line_with_header": {
            "df": line_df,
            "texts": line_df["header_code"].tolist()
        },
        "line_no_header": {
            "df": line_df,
            "texts": line_df["raw_code"].tolist()
        }
    }

    # 4. Dense retrieval (BAAI/bge-m3)
    print("\n--- Initializing Dense Model (BAAI/bge-m3) ---")
    dense_model_name = "BAAI/bge-m3"
    dense_model = SentenceTransformer(dense_model_name)
    dense_model.max_seq_length = MAX_SEQ_LENGTH

    query_embs = get_cached_or_compute_embeddings("queries_frozen_bgem3", query_texts, dense_model, dense_model_name)

    chunk_embs = {}
    for cond_name, c_data in conditions.items():
        chunk_embs[cond_name] = get_cached_or_compute_embeddings(f"emb_{cond_name}_bgem3", c_data["texts"], dense_model, dense_model_name)

    # 5. BM25 Indices
    print("\n--- Initializing BM25 Indices ---")
    bm25_indices = {}
    for cond_name, c_data in conditions.items():
        tokenized_corpus = [tokenize_code(t) for t in c_data["texts"]]
        bm25_indices[cond_name] = BM25Okapi(tokenized_corpus)
        print(f"  BM25 index built for {cond_name} ({len(tokenized_corpus)} docs)")

    # 6. CrossEncoder Reranker (BAAI/bge-reranker-base)
    print("\n--- Initializing CrossEncoder (BAAI/bge-reranker-base) ---")
    reranker_model_name = "BAAI/bge-reranker-base"
    reranker = CrossEncoder(reranker_model_name)

    # 7. Execute 12 Retrieval Runs
    print("\n--- Executing 12 Retrieval Runs (Depth=20) ---")
    retrieval_results = {cfg: {} for cfg in CONFIGS}

    for cond_name, c_data in conditions.items():
        df = c_data["df"]
        texts = c_data["texts"]
        embs = chunk_embs[cond_name]
        bm25 = bm25_indices[cond_name]

        print(f"\nProcessing condition: {cond_name} ({len(df)} chunks)...")

        for q_idx, jid in enumerate(jd_ids):
            q_text = query_texts[q_idx]
            q_emb = query_embs[q_idx]

            # Tier 1: BM25
            bm25_scores = np.asarray(bm25.get_scores(tokenize_code(q_text)), dtype=float)
            bm25_order = np.argsort(-bm25_scores)[:TOP_RETRIEVAL_DEPTH]
            bm25_ranked = []
            for rank, idx in enumerate(bm25_order, start=1):
                row = df.iloc[idx]
                bm25_ranked.append({
                    "rank": int(rank),
                    "score": float(bm25_scores[idx]),
                    "chunk_id": str(row["chunk_id"]),
                    "repo_name": str(row["repo_name"]),
                    "file_path": str(row["file_path"]),
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"])
                })
            retrieval_results[f"bm25_{cond_name}"][jid] = bm25_ranked

            # Tier 2: Dense (BAAI/bge-m3 cosine similarity via dot product of normalized embeddings)
            dense_scores = np.dot(embs, q_emb)
            dense_order = np.argsort(-dense_scores)[:TOP_RETRIEVAL_DEPTH]
            dense_ranked = []
            for rank, idx in enumerate(dense_order, start=1):
                row = df.iloc[idx]
                dense_ranked.append({
                    "rank": int(rank),
                    "score": float(dense_scores[idx]),
                    "chunk_id": str(row["chunk_id"]),
                    "repo_name": str(row["repo_name"]),
                    "file_path": str(row["file_path"]),
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"]),
                    "_chunk_idx": int(idx)
                })
            # Save dense run without internal _chunk_idx
            retrieval_results[f"dense_{cond_name}"][jid] = [
                {k: v for k, v in item.items() if k != "_chunk_idx"}
                for item in dense_ranked
            ]

            # Tier 3: CrossEncoder Re-ranker (Re-ranks Top-20 of Dense)
            rerank_pairs = [(q_text, texts[item["_chunk_idx"]]) for item in dense_ranked]
            rerank_scores = np.asarray(reranker.predict(rerank_pairs, batch_size=16), dtype=float)
            rerank_sorted_order = np.argsort(-rerank_scores)
            rerank_ranked = []
            for new_rank, orig_idx in enumerate(rerank_sorted_order, start=1):
                orig_item = dense_ranked[orig_idx]
                rerank_ranked.append({
                    "rank": int(new_rank),
                    "score": float(rerank_scores[orig_idx]),
                    "dense_rank": int(orig_item["rank"]),
                    "dense_score": float(orig_item["score"]),
                    "chunk_id": orig_item["chunk_id"],
                    "repo_name": orig_item["repo_name"],
                    "file_path": orig_item["file_path"],
                    "start_line": int(orig_item["start_line"]),
                    "end_line": int(orig_item["end_line"])
                })
            retrieval_results[f"rerank_{cond_name}"][jid] = rerank_ranked

    # Overwrite all 12 JSON files in dataset/retrieval_runs/
    print("\n--- Saving 12 Retrieval Run JSON Files ---")
    for config_id, run_data in retrieval_results.items():
        run_file = RETRIEVAL_RUNS_DIR / f"{config_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2)
        print(f"  Saved {run_file.name} ({len(run_data)} queries)")

    # 8. Unified Top-5 Pooling Across ALL 12 Systems
    print("\n" + "=" * 80)
    print(f"UNIFIED TOP-{POOL_TOP_K} POOLING ACROSS ALL 12 CONFIGURATIONS (RULE R18, R19)")
    print("=" * 80)

    # Build corpus coordinate lookup for clean raw_code
    corpus_by_coord = {}
    for r in corpus.itertuples():
        key = (r.repo_name, r.file_path, int(r.start_line), int(r.end_line))
        corpus_by_coord.setdefault(key, []).append(r)

    # Pool candidates per query
    pooled_spans = {}  # span_key -> data
    pool_report_rows = []

    for jid in jd_ids:
        q_info = queries_dict[jid]
        cat = q_info["category"]
        jd_req_text = q_info["primary_query"]

        query_spans = set()
        system_contributions = 0

        for sys_id in CONFIGS:
            top_k_items = retrieval_results[sys_id][jid][:POOL_TOP_K]
            for it in top_k_items:
                system_contributions += 1
                r_name = it["repo_name"]
                f_path = it["file_path"]
                s_line = int(it["start_line"])
                e_line = int(it["end_line"])
                cid = it["chunk_id"]

                span_key = (jid, r_name, f_path, s_line, e_line)
                query_spans.add(span_key)

                if span_key not in pooled_spans:
                    # Find raw code from corpus
                    matching_chunks = corpus_by_coord.get((r_name, f_path, s_line, e_line), [])
                    assert len(matching_chunks) > 0, f"Missing chunk in corpus for coord {span_key}"
                    raw_code = matching_chunks[0].raw_code

                    pooled_spans[span_key] = {
                        "query_id": jid,
                        "jd_requirement_text": jd_req_text,
                        "repo_name": r_name,
                        "file_path": f_path,
                        "start_line": s_line,
                        "end_line": e_line,
                        "raw_code_without_header": raw_code,
                        "chunk_ids": set(),
                        "retrieved_by": []
                    }

                pooled_spans[span_key]["chunk_ids"].add(cid)
                pooled_spans[span_key]["retrieved_by"].append({
                    "system": sys_id,
                    "rank": it["rank"],
                    "score": it["score"]
                })

        pool_report_rows.append({
            "jd_id": jid,
            "category": cat,
            "total_pooled": system_contributions,
            "already_gold": 0,
            "new_to_label": len(query_spans)
        })

    print(f"Total candidate instances pooled: {sum(r['total_pooled'] for r in pool_report_rows)}")
    print(f"Unique candidate spans across 25 JDs: {len(pooled_spans)}")

    # 9. Deterministic Ergonomic Sorting: (query_id, repo_name, file_path, start_line, end_line)
    sorted_spans = sorted(
        pooled_spans.values(),
        key=lambda x: (x["query_id"], x["repo_name"], x["file_path"], x["start_line"], x["end_line"])
    )

    # 10. Assign Neutral Sequential IDs: P0001, P0002, ...
    to_label_rows = []
    annotator1_rows = []
    annotator2_rows = []
    secret_pairs = {}
    chunk_lookup = {}

    for idx, span in enumerate(sorted_spans, start=1):
        pair_id = f"P{idx:04d}"
        q_id = span["query_id"]

        # Clean code check (zero header leakage)
        code = span["raw_code_without_header"]
        assert "// Class:" not in code, f"{pair_id}: // Class: leaked in raw code"
        assert "// Method:" not in code, f"{pair_id}: // Method: leaked in raw code"
        assert "// File:" not in code, f"{pair_id}: // File: leaked in raw code"

        # to_label.csv record
        to_label_rows.append({
            "pair_id": pair_id,
            "query_id": q_id,
            "jd_requirement_text": span["jd_requirement_text"],
            "repo_name": span["repo_name"],
            "file_path": span["file_path"],
            "start_line": span["start_line"],
            "end_line": span["end_line"],
            "raw_code_without_header": code,
            "annotator1_score": "",
            "annotator2_score": "",
            "final_score": "",
            "adjudication_notes": ""
        })

        # annotator1_blank.csv record
        annotator1_rows.append({
            "pair_id": pair_id,
            "query_id": q_id,
            "jd_requirement_text": span["jd_requirement_text"],
            "repo_name": span["repo_name"],
            "file_path": span["file_path"],
            "start_line": span["start_line"],
            "end_line": span["end_line"],
            "raw_code_without_header": code,
            "annotator1_score": "",
            "notes": ""
        })

        # annotator2_blank.csv record
        annotator2_rows.append({
            "pair_id": pair_id,
            "query_id": q_id,
            "jd_requirement_text": span["jd_requirement_text"],
            "repo_name": span["repo_name"],
            "file_path": span["file_path"],
            "start_line": span["start_line"],
            "end_line": span["end_line"],
            "raw_code_without_header": code,
            "annotator2_score": "",
            "notes": ""
        })

        # Secret mapping
        chunk_ids_list = sorted(list(span["chunk_ids"]))
        secret_pairs[pair_id] = {
            "pair_id": pair_id,
            "query_id": q_id,
            "repo_name": span["repo_name"],
            "file_path": span["file_path"],
            "start_line": span["start_line"],
            "end_line": span["end_line"],
            "chunk_ids": chunk_ids_list,
            "systems_retrieved": [r["system"] for r in span["retrieved_by"]],
            "system_details": span["retrieved_by"]
        }

        # Index chunk lookup for evaluation
        for cid in chunk_ids_list:
            chunk_lookup[f"{q_id}::{cid}"] = pair_id

    # 11. Export dataset/to_label.csv
    to_label_fieldnames = [
        "pair_id", "query_id", "jd_requirement_text", "repo_name", "file_path",
        "start_line", "end_line", "raw_code_without_header",
        "annotator1_score", "annotator2_score", "final_score", "adjudication_notes"
    ]
    with open(TO_LABEL_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=to_label_fieldnames)
        writer.writeheader()
        writer.writerows(to_label_rows)
    print(f"Exported {TO_LABEL_FILE.name} ({len(to_label_rows)} blinded pairs).")

    # 12. Export separate blank sheets
    with open(ANNOTATOR1_BLANK_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "query_id", "jd_requirement_text", "repo_name", "file_path",
            "start_line", "end_line", "raw_code_without_header", "annotator1_score", "notes"
        ])
        writer.writeheader()
        writer.writerows(annotator1_rows)
    print(f"Exported {ANNOTATOR1_BLANK_FILE.name}.")

    with open(ANNOTATOR2_BLANK_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "query_id", "jd_requirement_text", "repo_name", "file_path",
            "start_line", "end_line", "raw_code_without_header", "annotator2_score", "notes"
        ])
        writer.writeheader()
        writer.writerows(annotator2_rows)
    print(f"Exported {ANNOTATOR2_BLANK_FILE.name}.")

    # 13. Export dataset/pool_span_mapping.json
    secret_payload = {
        "metadata": {
            "total_blinded_pairs": len(to_label_rows),
            "total_queries": len(jd_ids),
            "systems_pooled_count": len(CONFIGS),
            "systems_pooled": CONFIGS,
            "pool_k_per_system": POOL_TOP_K,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "pairs": secret_pairs,
        "chunk_lookup": chunk_lookup
    }
    with open(SECRET_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(secret_payload, f, indent=2)
    print(f"Exported {SECRET_MAPPING_FILE.name} ({len(secret_pairs)} entries, {len(chunk_lookup)} chunk lookups).")

    # 14. Export dataset/pool_size_report.csv
    with open(POOL_REPORT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["jd_id", "category", "total_pooled", "already_gold", "new_to_label"])
        writer.writeheader()
        writer.writerows(pool_report_rows)
    print(f"Exported {POOL_REPORT_FILE.name} (25 queries covered).")

    # 15. Verification: Check that 100% of pair_id map to valid chunks in chunk_corpus.parquet
    corpus_chunk_ids = set(corpus["chunk_id"])
    orphans = []
    for pid, pdata in secret_pairs.items():
        for cid in pdata["chunk_ids"]:
            if cid not in corpus_chunk_ids:
                orphans.append((pid, cid))
    assert len(orphans) == 0, f"Found {len(orphans)} orphaned chunk mappings! {orphans[:5]}"
    print("\n[VERIFICATION PASS] 100% of pair_ids map to valid chunks in chunk_corpus.parquet (0 orphaned pairs).")
    print("=" * 80)
    print("PHASE 2 COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
