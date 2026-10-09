#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Task: TWO-STAGE RETRIEVAL BENCHMARK WITH CROSS-ENCODER RE-RANKING
Stage 1 (Bi-Encoder Dense): BAAI/bge-m3 (1024-dim)
Stage 2 (Cross-Encoder Re-ranker): BAAI/bge-reranker-base (or bge-reranker-large)
Objective: Evaluate Top-1 and Top-3 Ranking Optimization via Cross-Attention
Target Publication: IEEE SANER 2027 (ERA Track - CORE A) & Double-Anonymous Peer Review
=============================================================================
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Configure UTF-8 encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BENCHMARK_RESULTS_DIR = DATASET_DIR / "benchmark_results"
BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JD_SKILLS_FILE = DATASET_DIR / "extracted_jd_skills.json"
FINAL_GT_FILE = DATASET_DIR / "ground_truth_final.csv"

OUTPUT_REPORT = BENCHMARK_RESULTS_DIR / "reranker_evaluation_report.txt"
PER_QUERY_CSV = BENCHMARK_RESULTS_DIR / "reranker_per_query_results.csv"

# Stage 1 (Bi-Encoder BGE-M3) vector caches
EMB_JD_CACHE = DATASET_DIR / "embeddings_jd_bgem3.npy"
EMB_AST_CACHE = DATASET_DIR / "embeddings_ast_bgem3.npy"
EMB_LINE_CACHE = DATASET_DIR / "embeddings_line_bgem3.npy"

# Stage 2 (Cross-Encoder Re-ranker) score cache
RERANK_SCORES_CACHE = DATASET_DIR / "reranker_scores_cache.npy"


def dcg_at_k(relevance_scores, k=5):
    """Compute Discounted Cumulative Gain at k (Linear Gain formulation)."""
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    if not len(relevance_scores):
        return 0.0
    gains = relevance_scores
    discounts = np.log2(np.arange(2, len(relevance_scores) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevance_scores, k=5):
    """Compute Normalized Discounted Cumulative Gain at k (NDCG@k = DCG@k / IDCG@k)."""
    actual_dcg = dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)
    if ideal_dcg == 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def context_precision_at_k(relevance_scores, k=5, threshold=1.0):
    """Compute Context Precision@k following RAGAs standard."""
    sub_scores = relevance_scores[:k]
    v = [1 if s >= threshold else 0 for s in sub_scores]
    n_relevant = sum(v)
    if n_relevant == 0:
        return 0.0

    cumulative_hits = 0
    weighted_precisions = 0.0
    for idx, is_rel in enumerate(v):
        if is_rel:
            cumulative_hits += 1
            precision_at_idx = cumulative_hits / (idx + 1)
            weighted_precisions += precision_at_idx

    return float(weighted_precisions / n_relevant)


def precision_recall_at_k(relevance_scores, k=5, threshold=1.0):
    """Compute Precision@k, Recall@k, and F1@k."""
    sub_scores = relevance_scores[:k]
    binary_hits = [1 if s >= threshold else 0 for s in sub_scores]
    total_relevant = sum([1 if s >= threshold else 0 for s in relevance_scores])
    
    p_at_k = sum(binary_hits) / k if k > 0 else 0.0
    r_at_k = sum(binary_hits) / total_relevant if total_relevant > 0 else 0.0
    f1_at_k = (2 * p_at_k * r_at_k / (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0
    return p_at_k, r_at_k, f1_at_k


def mrr_score(relevance_scores, k=5, threshold=1.0):
    """Compute Mean Reciprocal Rank at Top-k."""
    sub_scores = relevance_scores[:k] if k is not None else relevance_scores
    for idx, score in enumerate(sub_scores):
        if score >= threshold:
            return 1.0 / (idx + 1)
    return 0.0


def count_win_loss_tie(scores_a, scores_b, tol=1e-5):
    """Count pairwise Wins, Losses, and Ties between configurations."""
    wins = sum(1 for a, b in zip(scores_a, scores_b) if a - b > tol)
    losses = sum(1 for a, b in zip(scores_a, scores_b) if b - a > tol)
    ties = sum(1 for a, b in zip(scores_a, scores_b) if abs(a - b) <= tol)
    return wins, losses, ties


def safe_wilcoxon(x, y, alternative='greater'):
    """Safe Wilcoxon signed-rank test avoiding zero-difference exceptions."""
    diff = np.array(x) - np.array(y)
    if np.all(np.isclose(diff, 0, atol=1e-7)):
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(x, y, alternative=alternative)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def evaluate_ranking_dict(ranked_labels_list):
    """Compute average metrics dictionary for a ranked candidate list."""
    metrics = {'p5': [], 'r5': [], 'f15': [], 'ndcg1': [], 'ndcg3': [], 'ndcg5': [], 'mrr': [], 'ctx_p5': []}
    for lbls in ranked_labels_list:
        p5, r5, f15 = precision_recall_at_k(lbls, k=5)
        metrics['p5'].append(p5)
        metrics['r5'].append(r5)
        metrics['f15'].append(f15)
        metrics['ndcg1'].append(ndcg_at_k(lbls, k=1))
        metrics['ndcg3'].append(ndcg_at_k(lbls, k=3))
        metrics['ndcg5'].append(ndcg_at_k(lbls, k=5))
        metrics['mrr'].append(mrr_score(lbls, k=5))
        metrics['ctx_p5'].append(context_precision_at_k(lbls, k=5))
    return metrics


def run_reranker_benchmark(model_name="BAAI/bge-reranker-base", force_recompute=False):
    print("=" * 80)
    print("TWO-STAGE RETRIEVAL BENCHMARK WITH CROSS-ENCODER RE-RANKING")
    print(f"Stage 1 (Bi-Encoder): BAAI/bge-m3 (Dense 1024-dim)")
    print(f"Stage 2 (Cross-Encoder): {model_name}")
    print("=" * 80)

    # 1. Load Ground Truth
    if not FINAL_GT_FILE.exists():
        print(f"Error: Missing {FINAL_GT_FILE.name}")
        return

    df = pd.read_csv(FINAL_GT_FILE)
    label_col = 'ground_truth_label' if 'ground_truth_label' in df.columns else 'human_label'
    print(f"Loaded {len(df)} ground-truth evidence samples from {FINAL_GT_FILE.name}")

    # 2. Load 25 JDs
    if not JD_SKILLS_FILE.exists():
        print(f"Error: Missing {JD_SKILLS_FILE.name}")
        return

    with open(JD_SKILLS_FILE, 'r', encoding='utf-8') as f:
        jd_skills_data = json.load(f)

    jd_queries = {}
    for jd_id, data in jd_skills_data.items():
        title = data.get("title", jd_id)
        domain = data.get("domain", "")
        raw_mand = data.get("mandatory_skills", [])
        mand_skills = [s["skill_name"] if isinstance(s, dict) else str(s) for s in raw_mand]
        skills_str = ", ".join(mand_skills)
        query_text = f"Job Title: {title}. Domain: {domain}. Mandatory Technical Skills: {skills_str}."
        jd_queries[jd_id] = {
            "title": title,
            "domain": domain,
            "skills": mand_skills,
            "query_text": query_text
        }

    unique_jds = sorted(df['jd_id'].unique().tolist())
    print(f"Loaded query descriptions for {len(unique_jds)} Job Descriptions.")

    # 3. Prepare Code Text for Chunks
    ast_texts = []
    line_texts = []
    for _, row in df.iterrows():
        ctx = str(row.get('context_header', '')).strip()
        code = str(row.get('chunk_content', '')).strip()
        ast_texts.append(f"{ctx}\n\n{code}" if ctx else code)
        line_texts.append(code)

    # 4. Load Stage 1 Embedding Matrices
    print("Loading Stage 1 (BGE-M3) embedding matrices...")
    if not (EMB_JD_CACHE.exists() and EMB_AST_CACHE.exists() and EMB_LINE_CACHE.exists()):
        print("Missing BGE-M3 cache files! Run evaluate_retrieval_benchmarks.py first.")
        return

    jd_embs_dict = np.load(EMB_JD_CACHE, allow_pickle=True).item()
    emb_ast = np.load(EMB_AST_CACHE)
    emb_line = np.load(EMB_LINE_CACHE)

    # 5. Build (Query, Code) Pairs for Stage 2 Cross-Encoder
    all_pairs_ast = []
    all_pairs_line = []
    jd_slice_map = {}

    current_idx = 0
    for jd_id in unique_jds:
        jd_indices = df.index[df['jd_id'] == jd_id].tolist()
        q_text = jd_queries[jd_id]["query_text"]
        count = len(jd_indices)
        jd_slice_map[jd_id] = (current_idx, current_idx + count, jd_indices)
        for idx in jd_indices:
            all_pairs_ast.append((q_text, ast_texts[idx]))
            all_pairs_line.append((q_text, line_texts[idx]))
        current_idx += count

    print(f"Prepared {len(all_pairs_ast)} candidate pairs across {len(unique_jds)} JDs.")

    # 6. Cross-Encoder Re-ranker Scoring
    cache_valid = (
        not force_recompute
        and RERANK_SCORES_CACHE.exists()
    )

    if cache_valid:
        print("Loading Cross-Encoder re-ranking predictions from cache...")
        cached_data = np.load(RERANK_SCORES_CACHE, allow_pickle=True).item()
        scores_rerank_ast_all = cached_data.get("ast")
        scores_rerank_line_all = cached_data.get("line")
    else:
        print(f"Loading Cross-Encoder Re-ranker model: {model_name}...")
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(model_name)
        print("Loaded Re-ranker model.")

        print(f"Scoring {len(all_pairs_ast)} AST candidate pairs...")
        scores_rerank_ast_all = reranker.predict(all_pairs_ast, batch_size=16, show_progress_bar=True)

        print(f"Scoring {len(all_pairs_line)} Line-based candidate pairs...")
        scores_rerank_line_all = reranker.predict(all_pairs_line, batch_size=16, show_progress_bar=True)

        np.save(RERANK_SCORES_CACHE, {
            "model": model_name,
            "ast": scores_rerank_ast_all,
            "line": scores_rerank_line_all
        })
        print("Saved Re-ranker predictions to cache.")

    # 7. Evaluate 4 Competing Configurations
    # Config 1: Line-based Stage 1 (Bi-Encoder BGE-M3)
    # Config 2: AST Progressive Stage 1 (Bi-Encoder BGE-M3)
    # Config 3: Line-based Stage 2 (+ Re-ranker)
    # Config 4: AST Progressive Stage 2 (+ Re-ranker - Proposed Pipeline)
    ranked_labels_ast_s1 = []
    ranked_labels_line_s1 = []
    ranked_labels_line_s2 = []
    ranked_labels_ast_s2 = []

    per_query_rows = []

    for jd_id in unique_jds:
        start_pos, end_pos, jd_indices = jd_slice_map[jd_id]
        gt_labels = df.loc[jd_indices, label_col].values.astype(float)
        query_vec = jd_embs_dict[jd_id]

        # a) AST Stage 1
        scores_ast_s1 = np.dot(emb_ast[jd_indices], query_vec)
        rank_ast_s1 = np.argsort(-scores_ast_s1)
        ranked_labels_ast_s1.append(gt_labels[rank_ast_s1])

        # b) Line Stage 1
        scores_line_s1 = np.dot(emb_line[jd_indices], query_vec)
        rank_line_s1 = np.argsort(-scores_line_s1)
        ranked_labels_line_s1.append(gt_labels[rank_line_s1])

        # c) Line Stage 2
        scores_line_s2 = scores_rerank_line_all[start_pos:end_pos]
        rank_line_s2 = np.argsort(-scores_line_s2)
        ranked_labels_line_s2.append(gt_labels[rank_line_s2])

        # d) AST Stage 2
        scores_ast_s2 = scores_rerank_ast_all[start_pos:end_pos]
        rank_ast_s2 = np.argsort(-scores_ast_s2)
        ranked_labels_ast_s2.append(gt_labels[rank_ast_s2])

        per_query_rows.append({
            "jd_id": jd_id,
            "title": jd_queries[jd_id]["title"],
            "ast_s1_ndcg1": ndcg_at_k(gt_labels[rank_ast_s1], 1),
            "ast_s2_ndcg1": ndcg_at_k(gt_labels[rank_ast_s2], 1),
            "line_s1_ndcg1": ndcg_at_k(gt_labels[rank_line_s1], 1),
            "line_s2_ndcg1": ndcg_at_k(gt_labels[rank_line_s2], 1),
            "ast_s1_ndcg5": ndcg_at_k(gt_labels[rank_ast_s1], 5),
            "ast_s2_ndcg5": ndcg_at_k(gt_labels[rank_ast_s2], 5),
            "line_s1_ndcg5": ndcg_at_k(gt_labels[rank_line_s1], 5),
            "line_s2_ndcg5": ndcg_at_k(gt_labels[rank_line_s2], 5),
        })

    # Save per-query CSV
    pd.DataFrame(per_query_rows).to_csv(PER_QUERY_CSV, index=False, encoding='utf-8-sig')

    # Compute full metrics
    m_ast_s1 = evaluate_ranking_dict(ranked_labels_ast_s1)
    m_line_s1 = evaluate_ranking_dict(ranked_labels_line_s1)
    m_line_s2 = evaluate_ranking_dict(ranked_labels_line_s2)
    m_ast_s2 = evaluate_ranking_dict(ranked_labels_ast_s2)

    avg_ast_s1 = {k: np.mean(v) for k, v in m_ast_s1.items()}
    avg_line_s1 = {k: np.mean(v) for k, v in m_line_s1.items()}
    avg_line_s2 = {k: np.mean(v) for k, v in m_line_s2.items()}
    avg_ast_s2 = {k: np.mean(v) for k, v in m_ast_s2.items()}

    # 8. Statistical Significance Testing (Wilcoxon Signed-Rank Test)
    w_ast_s2_vs_s1_n1, p_ast_s2_vs_s1_n1 = safe_wilcoxon(m_ast_s2['ndcg1'], m_ast_s1['ndcg1'], alternative='greater')
    w_ast_s2_vs_s1_n5, p_ast_s2_vs_s1_n5 = safe_wilcoxon(m_ast_s2['ndcg5'], m_ast_s1['ndcg5'], alternative='greater')
    wins_s2_vs_s1_n1, losses_s2_vs_s1_n1, ties_s2_vs_s1_n1 = count_win_loss_tie(m_ast_s2['ndcg1'], m_ast_s1['ndcg1'])

    w_ast_s2_vs_line_s1_n1, p_ast_s2_vs_line_s1_n1 = safe_wilcoxon(m_ast_s2['ndcg1'], m_line_s1['ndcg1'], alternative='greater')
    w_ast_s2_vs_line_s1_n5, p_ast_s2_vs_line_s1_n5 = safe_wilcoxon(m_ast_s2['ndcg5'], m_line_s1['ndcg5'], alternative='greater')
    wins_s2_vs_line_n1, losses_s2_vs_line_n1, ties_s2_vs_line_n1 = count_win_loss_tie(m_ast_s2['ndcg1'], m_line_s1['ndcg1'])
    wins_s2_vs_line_n5, losses_s2_vs_line_n5, ties_s2_vs_line_n5 = count_win_loss_tie(m_ast_s2['ndcg5'], m_line_s1['ndcg5'])

    p_s2_vs_s1_n1_str = "p < 0.001 (***)" if p_ast_s2_vs_s1_n1 < 0.001 else f"p = {p_ast_s2_vs_s1_n1:.4f}"
    p_s2_vs_line_n1_str = "p < 0.001 (***)" if p_ast_s2_vs_line_s1_n1 < 0.001 else f"p = {p_ast_s2_vs_line_s1_n1:.4f}"
    p_s2_vs_line_n5_str = "p < 0.01 (**)" if p_ast_s2_vs_line_s1_n5 < 0.01 else f"p = {p_ast_s2_vs_line_s1_n5:.4f}"

    report_text = f"""=============================================================================
TWO-STAGE RETRIEVAL BENCHMARK REPORT: CROSS-ENCODER RE-RANKING
STAGE 1 MODEL: BAAI/bge-m3 (Dense Bi-Encoder 1024-dim)
STAGE 2 MODEL: {model_name} (Cross-Encoder Re-ranker)
EVALUATION: RANKING OPTIMIZATION AT TOP-1 AND TOP-3
TARGET PUBLICATION: IEEE SANER 2027 (ERA TRACK - CORE A)
=============================================================================

1. RETRIEVAL & RE-RANKING PERFORMANCE COMPARISON (AVERAGED ACROSS 25 JDs):
--------------------------------------------------------------------------------------------------------------------
Configuration                                | P@5    | R@5    | F1@5   | NDCG@1 | NDCG@3 | NDCG@5 | MRR    | Ctx-P@5
--------------------------------------------------------------------------------------------------------------------
Baseline 2 (Stage 1): Line-based + BGE-M3    | {avg_line_s1['p5']:.4f} | {avg_line_s1['r5']:.4f} | {avg_line_s1['f15']:.4f} | {avg_line_s1['ndcg1']:.4f} | {avg_line_s1['ndcg3']:.4f} | {avg_line_s1['ndcg5']:.4f} | {avg_line_s1['mrr']:.4f} | {avg_line_s1['ctx_p5']:.4f}
Proposed (Stage 1): AST Progressive + BGE-M3 | {avg_ast_s1['p5']:.4f} | {avg_ast_s1['r5']:.4f} | {avg_ast_s1['f15']:.4f} | {avg_ast_s1['ndcg1']:.4f} | {avg_ast_s1['ndcg3']:.4f} | {avg_ast_s1['ndcg5']:.4f} | {avg_ast_s1['mrr']:.4f} | {avg_ast_s1['ctx_p5']:.4f}
--------------------------------------------------------------------------------------------------------------------
Baseline 2 (Stage 2): Line-based + Re-ranker | {avg_line_s2['p5']:.4f} | {avg_line_s2['r5']:.4f} | {avg_line_s2['f15']:.4f} | {avg_line_s2['ndcg1']:.4f} | {avg_line_s2['ndcg3']:.4f} | {avg_line_s2['ndcg5']:.4f} | {avg_line_s2['mrr']:.4f} | {avg_line_s2['ctx_p5']:.4f}
Proposed (Stage 2): AST + Re-ranker (Ours)   | {avg_ast_s2['p5']:.4f} | {avg_ast_s2['r5']:.4f} | {avg_ast_s2['f15']:.4f} | {avg_ast_s2['ndcg1']:.4f} | {avg_ast_s2['ndcg3']:.4f} | {avg_ast_s2['ndcg5']:.4f} | {avg_ast_s2['mrr']:.4f} | {avg_ast_s2['ctx_p5']:.4f}
--------------------------------------------------------------------------------------------------------------------
AST Performance Gain via Re-ranking (Delta Stage 2 vs Stage 1):
  • NDCG@1: {avg_ast_s1['ndcg1']:.4f} -> {avg_ast_s2['ndcg1']:.4f} ({(avg_ast_s2['ndcg1']-avg_ast_s1['ndcg1'])/avg_ast_s1['ndcg1']*100:+.1f}%)
  • NDCG@3: {avg_ast_s1['ndcg3']:.4f} -> {avg_ast_s2['ndcg3']:.4f} ({(avg_ast_s2['ndcg3']-avg_ast_s1['ndcg3'])/avg_ast_s1['ndcg3']*100:+.1f}%)
  • NDCG@5: {avg_ast_s1['ndcg5']:.4f} -> {avg_ast_s2['ndcg5']:.4f} ({(avg_ast_s2['ndcg5']-avg_ast_s1['ndcg5'])/avg_ast_s1['ndcg5']*100:+.1f}%)

2. STATISTICAL SIGNIFICANCE TESTS (PAIRED WILCOXON SIGNED-RANK TEST):
--------------------------------------------------------------------------------------------------------------------
(a) Re-ranking Impact on Proposed AST Pipeline (AST Stage 2 vs. AST Stage 1):
  • On NDCG@1:
    - Wilcoxon W = {w_ast_s2_vs_s1_n1:.1f}, p-value = {p_s2_vs_s1_n1_str}
    - Query Win/Loss/Tie: Wins = {wins_s2_vs_s1_n1} | Losses = {losses_s2_vs_s1_n1} | Ties = {ties_s2_vs_s1_n1}
  • On NDCG@5:
    - Wilcoxon W = {w_ast_s2_vs_s1_n5:.1f}, p-value = {p_ast_s2_vs_s1_n5:.6f}

(b) AST + Re-ranker vs. Line-based Baseline (AST Stage 2 vs. Line Stage 1):
  • On NDCG@1:
    - Wilcoxon W = {w_ast_s2_vs_line_s1_n1:.1f}, p-value = {p_s2_vs_line_n1_str}
    - Query Win/Loss/Tie: Wins = {wins_s2_vs_line_n1} | Losses = {losses_s2_vs_line_n1} | Ties = {ties_s2_vs_line_n1}
  • On NDCG@5:
    - Wilcoxon W = {w_ast_s2_vs_line_s1_n5:.1f}, p-value = {p_s2_vs_line_n5_str}
    - Query Win/Loss/Tie: Wins = {wins_s2_vs_line_n5} | Losses = {losses_s2_vs_line_n5} | Ties = {ties_s2_vs_line_n5}

3. LATEX ABLATION TABLE FORMATTING FOR IEEE SANER 2027:
--------------------------------------------------------------------------------------------------------------------
\\begin{{table*}}[t]
\\caption{{Two-Stage Retrieval Performance with Cross-Encoder Re-ranking across 25 JDs}}
\\label{{tab:reranker_performance}}
\\centering
\\begin{{tabular}}{{lccccccc}}
\\hline
\\textbf{{Configuration}} & \\textbf{{P@5}} & \\textbf{{R@5}} & \\textbf{{F1@5}} & \\textbf{{NDCG@1}} & \\textbf{{NDCG@3}} & \\textbf{{NDCG@5}} & \\textbf{{MRR}} \\\\
\\hline
Line-based + BGE-M3 (Stage 1) & {avg_line_s1['p5']:.3f} & {avg_line_s1['r5']:.3f} & {avg_line_s1['f15']:.3f} & {avg_line_s1['ndcg1']:.3f} & {avg_line_s1['ndcg3']:.3f} & {avg_line_s1['ndcg5']:.3f} & {avg_line_s1['mrr']:.3f} \\\\
AST Progressive + BGE-M3 (Stage 1) & {avg_ast_s1['p5']:.3f} & {avg_ast_s1['r5']:.3f} & {avg_ast_s1['f15']:.3f} & {avg_ast_s1['ndcg1']:.3f} & {avg_ast_s1['ndcg3']:.3f} & {avg_ast_s1['ndcg5']:.3f} & {avg_ast_s1['mrr']:.3f} \\\\
\\hline
Line-based + BGE-Reranker (Stage 2) & {avg_line_s2['p5']:.3f} & {avg_line_s2['r5']:.3f} & {avg_line_s2['f15']:.3f} & {avg_line_s2['ndcg1']:.3f} & {avg_line_s2['ndcg3']:.3f} & {avg_line_s2['ndcg5']:.3f} & {avg_line_s2['mrr']:.3f} \\\\
\\textbf{{Proposed: AST + BGE-Reranker (Stage 2)}} & \\textbf{{{avg_ast_s2['p5']:.3f}}} & \\textbf{{{avg_ast_s2['r5']:.3f}}} & \\textbf{{{avg_ast_s2['f15']:.3f}}} & \\textbf{{{avg_ast_s2['ndcg1']:.3f}}} & \\textbf{{{avg_ast_s2['ndcg3']:.3f}}} & \\textbf{{{avg_ast_s2['ndcg5']:.3f}}} & \\textbf{{{avg_ast_s2['mrr']:.3f}}} \\\\
\\hline
\\end{{tabular}}
\\end{{table*}}
--------------------------------------------------------------------------------------------------------------------
"""

    print(report_text)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Saved Re-ranker Report to: {OUTPUT_REPORT.name}")
    print(f"Saved Per-Query CSV to: {PER_QUERY_CSV.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Two-Stage Retrieval with Cross-Encoder Re-ranking")
    parser.add_argument("--model", type=str, default="BAAI/bge-reranker-base", help="HuggingFace Cross-Encoder model name")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of Re-ranker scores")
    args = parser.parse_args()

    run_reranker_benchmark(model_name=args.model, force_recompute=args.recompute)
