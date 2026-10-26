#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Task: 2x2 FACTORIAL BENCHMARK EVALUATION (PHASE 3 / GATE G5)
Governed by: Rules R20, R22, R23, R24, R25, R26, R44, R45, R46
Target: IEEE SANER 2027 (ERA Track) & Double-Anonymous Peer Review
=============================================================================
"""

import os
import sys
import json
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Configure Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RUNS_DIR = DATASET_DIR / "retrieval_runs"
RESULTS_DIR = DATASET_DIR / "benchmark_results"
PAPER_DIR = PROJECT_ROOT / "paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)

GT_FILE = DATASET_DIR / "ground_truth_final.csv"
QUERIES_FILE = DATASET_DIR / "queries_frozen.json"

CONFIGS = [
    # Tier 1: Lexical Sparse
    "bm25_line_no_header",
    "bm25_line_with_header",
    "bm25_ast_no_header",
    "bm25_ast_with_header",
    # Tier 2: Dense Bi-Encoder
    "dense_line_no_header",
    "dense_line_with_header",
    "dense_ast_no_header",
    "dense_ast_with_header",
    # Tier 3: Two-Stage Re-ranking
    "rerank_line_no_header",
    "rerank_line_with_header",
    "rerank_ast_no_header",
    "rerank_ast_with_header"
]

CONFIG_LABELS = {
    "bm25_line_no_header": "BM25 | Line (50 LOC) | No-Header",
    "bm25_line_with_header": "BM25 | Line (50 LOC) | With-Header",
    "bm25_ast_no_header": "BM25 | AST Syntactic | No-Header",
    "bm25_ast_with_header": "BM25 | AST Syntactic | With-Header",
    "dense_line_no_header": "Dense | Line (50 LOC) | No-Header",
    "dense_line_with_header": "Dense | Line (50 LOC) | With-Header",
    "dense_ast_no_header": "Dense | AST Syntactic | No-Header",
    "dense_ast_with_header": "Dense | AST Syntactic | With-Header",
    "rerank_line_no_header": "Re-rank | Line (50 LOC) | No-Header (Full Baseline)",
    "rerank_line_with_header": "Re-rank | Line (50 LOC) | With-Header",
    "rerank_ast_no_header": "Re-rank | AST Syntactic | No-Header",
    "rerank_ast_with_header": "Re-rank | AST Syntactic | With-Header (Full Proposed)"
}


# =============================================================================
# METRIC FUNCTIONS (Verified against test_metrics.py & Rule R26)
# =============================================================================

def dcg_at_k(scores: List[float], k: int = 10) -> float:
    """Compute Discounted Cumulative Gain at k."""
    sub = scores[:k]
    if len(sub) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(sub) + 2))
    return float(np.sum(np.asarray(sub, dtype=float) / discounts))


def ndcg_at_k(rel_scores: List[float], ideal_scores: List[float], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at k against ideal ranking."""
    actual = dcg_at_k(rel_scores, k)
    ideal = dcg_at_k(ideal_scores, k)
    if ideal == 0.0:
        return 0.0
    return float(actual / ideal)


def mrr_at_k(rel_scores: List[float], k: int = 10, threshold: float = 1.0) -> float:
    """Compute Reciprocal Rank at k."""
    sub = rel_scores[:k]
    for idx, score in enumerate(sub):
        if score >= threshold:
            return 1.0 / (idx + 1)
    return 0.0


def precision_recall_at_k(rel_scores: List[float], total_rel_in_pool: int, k: int = 5, threshold: float = 1.0) -> Tuple[float, float, float]:
    """Compute Precision@k, Recall@k, and F1@k."""
    sub = rel_scores[:k]
    hits = sum(1 for s in sub if s >= threshold)
    p_k = hits / k if k > 0 else 0.0
    r_k = hits / total_rel_in_pool if total_rel_in_pool > 0 else 0.0
    f1_k = (2.0 * p_k * r_k / (p_k + r_k)) if (p_k + r_k) > 0 else 0.0
    return float(p_k), float(r_k), float(f1_k)


def context_precision_at_k(rel_scores: List[float], k: int = 5, threshold: float = 1.0) -> float:
    """Compute Context Precision@k (RAGAs / IR benchmark standard)."""
    sub = rel_scores[:k]
    v = [1 if s >= threshold else 0 for s in sub]
    n_rel = sum(v)
    if n_rel == 0:
        return 0.0
    cum_hits = 0
    weighted_prec = 0.0
    for idx, is_rel in enumerate(v):
        if is_rel:
            cum_hits += 1
            weighted_prec += cum_hits / (idx + 1)
    return float(weighted_prec / n_rel)


def judged_at_k(is_judged_flags: List[bool], k: int = 10) -> float:
    """Compute coverage / proportion of judged items at top-k (Rule R20)."""
    sub = is_judged_flags[:k]
    if len(sub) == 0:
        return 0.0
    return float(sum(1 for j in sub if j) / len(sub))


def compute_bpref_single(judged_labels: List[Any], R: int, N: int) -> float:
    """Binary Preference (bpref) calculation for single query."""
    if R == 0:
        return 0.0
    nrel = 0
    s = 0.0
    for l in judged_labels:
        if l is None:
            continue
        if l >= 1:
            denom = min(R, N)
            s += 1.0 - (min(nrel, R) / denom if denom > 0 else 0.0)
        else:
            nrel += 1
    return float(s / R)


# =============================================================================
# STATISTICAL PROTOCOL (Rules R22, R23, R24, R25)
# =============================================================================

def bootstrap_ci_paired(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """95% Bootstrap Confidence Interval on paired mean difference (Delta = mean(A) - mean(B))."""
    diff = a - b
    n = len(diff)
    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(float(np.mean(diff[idx])))
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    return float(np.mean(diff)), float(ci_low), float(ci_high)


def bootstrap_ci_single(vals: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """95% Bootstrap Confidence Interval on single mean."""
    n = len(vals)
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(float(np.mean(vals[idx])))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Step-down Holm-Bonferroni correction over a family of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = p_val * (m - rank)
        adj = max(running_max, adj)
        adj = min(1.0, adj)
        running_max = adj
        adjusted[orig_idx] = float(adj)
    return adjusted


# =============================================================================
# MAIN BENCHMARK EVALUATOR
# =============================================================================

def main():
    print("=" * 80)
    print("  PHASE 3: 2x2 FACTORIAL BENCHMARK EVALUATION (GATE G5)")
    print("  Governed by Rules R20 - R26 & Registered Analysis Plan")
    print("=" * 80)

    # 1. Load Ground Truth
    df_gt = pd.read_csv(GT_FILE)
    print(f"Loaded Unified Ground Truth: {len(df_gt)} pairs from {GT_FILE.name}")

    gt_map = {}
    gt_query_rel = {}
    gt_R = {}
    gt_N = {}

    for _, r in df_gt.iterrows():
        jid = r["jd_id"] if "jd_id" in r else r["query_id"]
        key = (jid, r["repo_name"], r["file_path"], int(r["start_line"]), int(r["end_line"]))
        lbl = int(r["ground_truth_label"])
        gt_map[key] = lbl
        if jid not in gt_query_rel:
            gt_query_rel[jid] = []
        if lbl > 0:
            gt_query_rel[jid].append(lbl)
            gt_R[jid] = gt_R.get(jid, 0) + 1
        else:
            gt_N[jid] = gt_N.get(jid, 0) + 1

    # Sort ideal relevances descending per query
    for jid in gt_query_rel:
        gt_query_rel[jid].sort(reverse=True)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)["queries"]
    if isinstance(queries_raw, dict):
        query_ids = list(queries_raw.keys())
    else:
        query_ids = [q["jd_id"] for q in queries_raw]
    print(f"Loaded {len(query_ids)} Frozen Queries (15 Java, 10 React/TypeScript)")

    # 2. Evaluate all 12 configurations
    system_metrics = {}
    per_query_records = []

    for cfg in CONFIGS:
        run_file = RUNS_DIR / f"{cfg}.json"
        if not run_file.exists():
            print(f"Error: Missing run file {run_file}")
            continue

        with open(run_file, "r", encoding="utf-8") as f:
            run_data = json.load(f)

        cfg_records = []
        for jid in query_ids:
            items = run_data.get(jid, [])
            rel_scores = []
            judged_flags = []
            bpref_labels = []

            for it in items:
                k = (jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"]))
                if k in gt_map:
                    lbl = gt_map[k]
                    rel_scores.append(float(lbl))
                    judged_flags.append(True)
                    bpref_labels.append(lbl)
                else:
                    rel_scores.append(0.0)  # Rule R20: Unjudged assigned relevance = 0
                    judged_flags.append(False)
                    bpref_labels.append(None)

            ideal = gt_query_rel.get(jid, [])
            total_rel = len(ideal)

            # Metrics
            n1 = ndcg_at_k(rel_scores, ideal, k=1)
            n3 = ndcg_at_k(rel_scores, ideal, k=3)
            n5 = ndcg_at_k(rel_scores, ideal, k=5)
            n10 = ndcg_at_k(rel_scores, ideal, k=10)
            mrr = mrr_at_k(rel_scores, k=10, threshold=1.0)
            p5, r10, f1_5 = precision_recall_at_k(rel_scores, total_rel, k=5, threshold=1.0)
            p10, r10_full, _ = precision_recall_at_k(rel_scores, total_rel, k=10, threshold=1.0)
            cp5 = context_precision_at_k(rel_scores, k=5, threshold=1.0)
            cp10 = context_precision_at_k(rel_scores, k=10, threshold=1.0)
            j3 = judged_at_k(judged_flags, k=3)
            j5 = judged_at_k(judged_flags, k=5)
            j10 = judged_at_k(judged_flags, k=10)

            # Bpref
            R_val = gt_R.get(jid, 0)
            N_val = gt_N.get(jid, 0)
            bp = compute_bpref_single(bpref_labels, R_val, N_val)

            rec = {
                "config": cfg,
                "jd_id": jid,
                "ndcg_1": n1,
                "ndcg_3": n3,
                "ndcg_5": n5,
                "ndcg_10": n10,
                "mrr_10": mrr,
                "bpref": bp,
                "precision_5": p5,
                "recall_10": r10_full,
                "f1_5": f1_5,
                "context_precision_5": cp5,
                "context_precision_10": cp10,
                "judged_3": j3,
                "judged_5": j5,
                "judged_10": j10
            }
            cfg_records.append(rec)
            per_query_records.append(rec)

        system_metrics[cfg] = cfg_records

    # Save per-query breakdown
    df_per_query = pd.DataFrame(per_query_records)
    df_per_query.to_csv(RESULTS_DIR / "per_query_evaluation.csv", index=False, encoding="utf-8")
    print(f"Saved per-query breakdown: {len(df_per_query)} records")

    # 3. Aggregate Table 3: Summary across 12 configurations
    summary_rows = []
    for cfg in CONFIGS:
        recs = system_metrics[cfg]
        n10_vals = np.array([r["ndcg_10"] for r in recs])
        n5_vals = np.array([r["ndcg_5"] for r in recs])
        n3_vals = np.array([r["ndcg_3"] for r in recs])
        n1_vals = np.array([r["ndcg_1"] for r in recs])
        mrr_vals = np.array([r["mrr_10"] for r in recs])
        bp_vals = np.array([r["bpref"] for r in recs])
        p5_vals = np.array([r["precision_5"] for r in recs])
        r10_vals = np.array([r["recall_10"] for r in recs])
        f1_vals = np.array([r["f1_5"] for r in recs])
        cp5_vals = np.array([r["context_precision_5"] for r in recs])
        cp10_vals = np.array([r["context_precision_10"] for r in recs])
        j5_vals = np.array([r["judged_5"] for r in recs])
        j10_vals = np.array([r["judged_10"] for r in recs])

        # 95% bootstrap CI on mean NDCG@10
        ci_low, ci_high = bootstrap_ci_single(n10_vals, n_boot=10000, seed=42)

        # Tier breakdown
        tier = "BM25" if cfg.startswith("bm25") else ("Dense (BGE-M3)" if cfg.startswith("dense") else "Re-rank (Cross-Encoder)")
        chunking = "AST Syntactic" if "ast" in cfg else "Line (50 LOC)"
        header = "With-Header" if "with_header" in cfg else "No-Header"

        summary_rows.append({
            "config_id": cfg,
            "tier": tier,
            "chunking": chunking,
            "header": header,
            "ndcg_10_mean": float(np.mean(n10_vals)),
            "ndcg_10_std": float(np.std(n10_vals)),
            "ndcg_10_ci_95_low": float(ci_low),
            "ndcg_10_ci_95_high": float(ci_high),
            "ndcg_5_mean": float(np.mean(n5_vals)),
            "ndcg_5_std": float(np.std(n5_vals)),
            "ndcg_3_mean": float(np.mean(n3_vals)),
            "ndcg_3_std": float(np.std(n3_vals)),
            "ndcg_1_mean": float(np.mean(n1_vals)),
            "ndcg_1_std": float(np.std(n1_vals)),
            "mrr_10_mean": float(np.mean(mrr_vals)),
            "mrr_10_std": float(np.std(mrr_vals)),
            "bpref_mean": float(np.mean(bp_vals)),
            "p5_mean": float(np.mean(p5_vals)),
            "p5_std": float(np.std(p5_vals)),
            "recall_10_mean": float(np.mean(r10_vals)),
            "recall_10_std": float(np.std(r10_vals)),
            "f1_5_mean": float(np.mean(f1_vals)),
            "f1_5_std": float(np.std(f1_vals)),
            "cp_5_mean": float(np.mean(cp5_vals)),
            "cp_10_mean": float(np.mean(cp10_vals)),
            "judged_5_mean": float(np.mean(j5_vals)),
            "judged_10_mean": float(np.mean(j10_vals))
        })

    # Add Pre-registered Random Baselines
    corpus_parquet = DATASET_DIR / "chunk_corpus.parquet"
    if corpus_parquet.exists():
        df_corpus = pd.read_parquet(corpus_parquet)
        c_items = df_corpus[["repo_name", "file_path", "start_line", "end_line"]].to_dict("records")
        n_c = len(c_items)
        rng_c = np.random.default_rng(42)
        c_rand_n10 = []
        for jid in query_ids:
            ideal = gt_query_rel.get(jid, [])
            p_n10 = []
            for _ in range(1000):
                idx = rng_c.choice(n_c, size=10, replace=False)
                sc = [gt_map.get((jid, c_items[i]["repo_name"], c_items[i]["file_path"], int(c_items[i]["start_line"]), int(c_items[i]["end_line"])), 0.0) for i in idx]
                p_n10.append(ndcg_at_k(sc, ideal, k=10))
            c_rand_n10.append(np.mean(p_n10))

        c_ci_low, c_ci_high = bootstrap_ci_single(np.array(c_rand_n10), n_boot=10000, seed=42)
        summary_rows.append({
            "config_id": "random_corpus_universe",
            "tier": "Random Baseline",
            "chunking": "Random Selection",
            "header": f"Full Corpus ({n_c} Chunks)",
            "ndcg_10_mean": float(np.mean(c_rand_n10)),
            "ndcg_10_std": float(np.std(c_rand_n10)),
            "ndcg_10_ci_95_low": float(c_ci_low),
            "ndcg_10_ci_95_high": float(c_ci_high),
            "ndcg_5_mean": 0.0, "ndcg_5_std": 0.0,
            "ndcg_3_mean": 0.0, "ndcg_3_std": 0.0,
            "ndcg_1_mean": 0.0, "ndcg_1_std": 0.0,
            "mrr_10_mean": 0.0, "mrr_10_std": 0.0,
            "bpref_mean": 0.0,
            "p5_mean": 0.0, "p5_std": 0.0,
            "recall_10_mean": 0.0, "recall_10_std": 0.0,
            "f1_5_mean": 0.0, "f1_5_std": 0.0,
            "cp_5_mean": 0.0, "cp_10_mean": 0.0,
            "judged_5_mean": 0.0,
            "judged_10_mean": 0.0
        })

        # Pre-registered Baseline 2: Random Selection from Evaluated Pool
        rng_p = np.random.default_rng(42)
        pool_rand_n10 = []
        for jid in query_ids:
            ideal = gt_query_rel.get(jid, [])
            pool_scores = [float(val) for val in df_gt[df_gt["jd_id"] == jid]["ground_truth_label"].tolist()]
            if len(pool_scores) >= 10:
                p_sim = []
                for _ in range(1000):
                    idx = rng_p.choice(len(pool_scores), size=10, replace=False)
                    sc = [pool_scores[i] for i in idx]
                    p_sim.append(ndcg_at_k(sc, ideal, k=10))
                pool_rand_n10.append(np.mean(p_sim))
            else:
                pool_rand_n10.append(0.0)

        p_ci_low, p_ci_high = bootstrap_ci_single(np.array(pool_rand_n10), n_boot=10000, seed=42)
        summary_rows.append({
            "config_id": "random_pool_candidates",
            "tier": "Random Baseline",
            "chunking": "Random Selection",
            "header": "Evaluated Pool Candidates",
            "ndcg_10_mean": float(np.mean(pool_rand_n10)),
            "ndcg_10_std": float(np.std(pool_rand_n10)),
            "ndcg_10_ci_95_low": float(p_ci_low),
            "ndcg_10_ci_95_high": float(p_ci_high),
            "ndcg_5_mean": 0.0, "ndcg_5_std": 0.0,
            "ndcg_3_mean": 0.0, "ndcg_3_std": 0.0,
            "ndcg_1_mean": 0.0, "ndcg_1_std": 0.0,
            "mrr_10_mean": 0.0, "mrr_10_std": 0.0,
            "bpref_mean": 0.0,
            "p5_mean": 0.0, "p5_std": 0.0,
            "recall_10_mean": 0.0, "recall_10_std": 0.0,
            "f1_5_mean": 0.0, "f1_5_std": 0.0,
            "cp_5_mean": 0.0, "cp_10_mean": 0.0,
            "judged_5_mean": 1.0,
            "judged_10_mean": 1.0
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(RESULTS_DIR / "table_3_retrieval_benchmark.csv", index=False, encoding="utf-8")
    with open(RESULTS_DIR / "table_3_retrieval_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print("Saved Table 3 benchmark results to CSV and JSON.")

    # 4. Statistical Testing Protocol
    # Confirmatory Family of 4 comparisons (Rerank tier)
    print("\n" + "=" * 80)
    print("  PRIMARY CONFIRMATORY HYPOTHESIS TESTING (FAMILY OF 4)")
    print("  Metric: NDCG@10 (Two-Stage Cross-Encoder Tier) | Alpha = 0.05")
    print("=" * 80)

    confirmatory_defs = [
        ("Comp 1 (Full Proposed vs Full Baseline)", "rerank_ast_with_header", "rerank_line_no_header"),
        ("Comp 2 (Pure Chunking Effect)", "rerank_ast_no_header", "rerank_line_no_header"),
        ("Comp 3 (Header Effect on AST)", "rerank_ast_with_header", "rerank_ast_no_header"),
        ("Comp 4 (Header Effect on Line)", "rerank_line_with_header", "rerank_line_no_header")
    ]

    def run_tests_on_family(comparisons_list, apply_holm=True):
        res_list = []
        raw_p = []
        for name, cfg_a, cfg_b in comparisons_list:
            a_vals = np.array([r["ndcg_10"] for r in system_metrics[cfg_a]])
            b_vals = np.array([r["ndcg_10"] for r in system_metrics[cfg_b]])
            diff = a_vals - b_vals

            mean_diff, ci_low, ci_high = bootstrap_ci_paired(a_vals, b_vals, n_boot=10000, seed=42)
            nz_diff = diff[diff != 0.0]
            if len(nz_diff) == 0:
                w_stat, p_val = 0.0, 1.0
            else:
                w_res = wilcoxon(a_vals, b_vals, alternative="two-sided")
                w_stat = float(w_res.statistic)
                p_val = float(w_res.pvalue)

            wins = int(np.sum(diff > 0.0))
            losses = int(np.sum(diff < 0.0))
            ties = int(np.sum(diff == 0.0))

            raw_p.append(p_val)
            res_list.append({
                "comparison": name,
                "system_A": cfg_a,
                "system_B": cfg_b,
                "mean_A": float(np.mean(a_vals)),
                "mean_B": float(np.mean(b_vals)),
                "mean_diff": float(mean_diff),
                "ci_95_low": float(ci_low),
                "ci_95_high": float(ci_high),
                "wilcoxon_stat": w_stat,
                "p_raw": p_val,
                "wins": wins,
                "losses": losses,
                "ties": ties
            })

        if apply_holm:
            holm_p = holm_bonferroni_correction(raw_p)
            for idx, res in enumerate(res_list):
                res["p_holm"] = holm_p[idx]
                is_sig = (res["p_holm"] < 0.05) and (res["ci_95_low"] > 0 or res["ci_95_high"] < 0)
                res["statistically_significant"] = is_sig
                if is_sig:
                    winner = res["system_A"] if res["mean_diff"] > 0 else res["system_B"]
                    res["conclusion"] = f"Statistically Significant (p_holm = {res['p_holm']:.4f}, winner: {winner})"
                else:
                    res["conclusion"] = f"Not Statistically Significant (p_holm = {res['p_holm']:.4f} >= 0.05 or CI spans 0)"
        else:
            for res in res_list:
                res["p_holm"] = res["p_raw"]
                is_sig = (res["p_raw"] < 0.05) and (res["ci_95_low"] > 0 or res["ci_95_high"] < 0)
                res["statistically_significant"] = is_sig
                if is_sig:
                    winner = res["system_A"] if res["mean_diff"] > 0 else res["system_B"]
                    res["conclusion"] = f"Statistically Significant (p_raw = {res['p_raw']:.4f}, winner: {winner})"
                else:
                    res["conclusion"] = f"Not Statistically Significant (p_raw = {res['p_raw']:.4f} >= 0.05 or CI spans 0)"

        return res_list

    confirmatory_results = run_tests_on_family(confirmatory_defs, apply_holm=True)

    # Exploratory Contrasts
    exploratory_defs = [
        ("Comp 5 (AST+Header vs Line+Header, Rerank Tier)", "rerank_ast_with_header", "rerank_line_with_header"),
        ("Dense Header Effect on AST", "dense_ast_with_header", "dense_ast_no_header"),
        ("Dense Header Effect on Line", "dense_line_with_header", "dense_line_no_header"),
        ("BM25 Header Effect on AST", "bm25_ast_with_header", "bm25_ast_no_header"),
        ("BM25 Header Effect on Line", "bm25_line_with_header", "bm25_line_no_header")
    ]
    exploratory_results = run_tests_on_family(exploratory_defs, apply_holm=False)

    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("\nFAMILY OF 4 CONFIRMATORY TESTING REPORT (PRIMARY METRIC: NDCG@10):")
    log("-" * 90)
    log(f"{'Comparison':<42} | {'Delta':^8} | {'95% Bootstrap CI':^18} | {'p_raw':^8} | {'p_holm':^8} | {'W/L/T':^7}")
    log("-" * 90)
    for r in confirmatory_results:
        ci_str = f"[{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}]"
        wlt_str = f"{r['wins']}/{r['losses']}/{r['ties']}"
        log(f"{r['comparison']:<42} | {r['mean_diff']:^+8.4f} | {ci_str:^18} | {r['p_raw']:^8.4f} | {r['p_holm']:^8.4f} | {wlt_str:^7}")
    log("-" * 90)

    log("\nEXPLORATORY CONTRASTS (OUTSIDE HOLM FAMILY):")
    log("-" * 90)
    log(f"{'Comparison':<42} | {'Delta':^8} | {'95% Bootstrap CI':^18} | {'p_raw':^8} | {'Sig?':^8} | {'W/L/T':^7}")
    log("-" * 90)
    for r in exploratory_results:
        ci_str = f"[{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}]"
        wlt_str = f"{r['wins']}/{r['losses']}/{r['ties']}"
        sig_str = "YES" if r["statistically_significant"] else "NO"
        log(f"{r['comparison']:<42} | {r['mean_diff']:^+8.4f} | {ci_str:^18} | {r['p_raw']:^8.4f} | {sig_str:^8} | {wlt_str:^7}")
    log("-" * 90)

    # Save outputs
    with open(RESULTS_DIR / "statistical_tests_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Keep confirmatory_results as primary list for test_evaluation_phase5.py backwards compatibility,
    # and attach full structured dict as well.
    all_tests_payload = confirmatory_results + exploratory_results
    with open(RESULTS_DIR / "statistical_tests_report.json", "w", encoding="utf-8") as f:
        json.dump(confirmatory_results, f, indent=2)

    with open(RESULTS_DIR / "statistical_tests_comprehensive.json", "w", encoding="utf-8") as f:
        json.dump({
            "confirmatory_family_4": confirmatory_results,
            "exploratory_contrasts": exploratory_results
        }, f, indent=2)
    print("\nSaved statistical tests reports to TXT and JSON.")

    # 5. Generate LaTeX Table for IEEE Paper (Table 3.3)
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Empirical Retrieval and Re-ranking Performance Across 2$\times$2 Factorial Pipeline Configurations ($N=25$ Job Descriptions, Graded Ground Truth $N=882$).}",
        r"\label{tab:main_retrieval_benchmark}\label{tab:retrieval_2x2}",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"\textbf{Retrieval Tier} & \textbf{Configuration} & \textbf{NDCG@1} & \textbf{NDCG@3} & \textbf{NDCG@5} & \textbf{NDCG@10} & \textbf{95\% CI} & \textbf{MRR@10} & \textbf{bpref} & \textbf{P@5} & \textbf{Judged@5} \\",
        r"\midrule"
    ]

    current_tier = ""
    for r in summary_rows:
        if r["tier"] != current_tier:
            if current_tier != "":
                latex_lines.append(r"\midrule")
            current_tier = r["tier"]
            latex_lines.append(f"\\multicolumn{{11}}{{l}}{{\\textit{{{current_tier}}}}} \\\\")

        cfg_name = f"{r['chunking']} + {r['header']}"
        ci_str = f"[{r['ndcg_10_ci_95_low']:.2f}, {r['ndcg_10_ci_95_high']:.2f}]"
        row_str = (
            f"  & {cfg_name:<30} & "
            f"{r['ndcg_1_mean']:.3f} & {r['ndcg_3_mean']:.3f} & {r['ndcg_5_mean']:.3f} & "
            f"\\textbf{{{r['ndcg_10_mean']:.3f}}} & {ci_str} & {r['mrr_10_mean']:.3f} & "
            f"{r['bpref_mean']:.3f} & {r['p5_mean']:.3f} & {r['judged_5_mean']*100:.1f}\\% \\\\"
        )
        latex_lines.append(row_str)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}\\",
        r"\footnotesize{\textit{Note:} All 12 configurations across lexical (BM25), dense bi-encoder (BGE-M3), and two-stage cross-encoder (BGE-Reranker-Base) tiers were pooled uniformly at Top-5 ($N=25$ JDs, $N=882$ unique graded candidate spans, Judged@5 = 100.0\%). All metrics are evaluated against the adjudicated unified ground truth.}",
        r"\vspace{-2mm}",
        r"\end{table*}"
    ])

    latex_path = PROJECT_ROOT / "report" / "tables" / "table_T3.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"Generated LaTeX Table: {latex_path.name}")

    print("\n" + "=" * 80)
    print("  PHASE 3 BENCHMARK EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
