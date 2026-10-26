#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Phase 3: Experiments and Common-Support Analysis
Governed by: 04_EXPERIMENTS_AND_COMMON_SUPPORT.md & Rules R20-R26
=============================================================================
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Windows console encoding safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RUNS_DIR = DATASET_DIR / "retrieval_runs"
RESULTS_DIR = DATASET_DIR / "benchmark_results"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GT_FILE = DATASET_DIR / "ground_truth_final.csv"
QUERIES_FILE = DATASET_DIR / "queries_frozen.json"
MAPPING_FILE = RESULTS_DIR / "evidence_chunk_mapping.csv"

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

# --- Metric Functions ---
def dcg_at_k(scores: List[float], k: int = 10) -> float:
    sub = scores[:k]
    if len(sub) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(sub) + 2))
    return float(np.sum(np.asarray(sub, dtype=float) / discounts))

def ndcg_at_k(rel_scores: List[float], ideal_scores: List[float], k: int = 10) -> float:
    actual = dcg_at_k(rel_scores, k)
    ideal = dcg_at_k(ideal_scores, k)
    if ideal == 0.0:
        return 0.0
    return float(actual / ideal)

def mrr_at_k(rel_scores: List[float], k: int = 10, threshold: float = 1.0) -> float:
    sub = rel_scores[:k]
    for idx, score in enumerate(sub):
        if score >= threshold:
            return 1.0 / (idx + 1)
    return 0.0

def precision_recall_at_k(rel_scores: List[float], total_rel_in_pool: int, k: int = 5, threshold: float = 1.0) -> Tuple[float, float, float]:
    sub = rel_scores[:k]
    hits = sum(1 for s in sub if s >= threshold)
    p_k = hits / k if k > 0 else 0.0
    r_k = hits / total_rel_in_pool if total_rel_in_pool > 0 else 0.0
    f1_k = (2.0 * p_k * r_k / (p_k + r_k)) if (p_k + r_k) > 0 else 0.0
    return float(p_k), float(r_k), float(f1_k)

def context_precision_at_k(rel_scores: List[float], k: int = 5, threshold: float = 1.0) -> float:
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
    sub = is_judged_flags[:k]
    if len(sub) == 0:
        return 0.0
    return float(sum(1 for j in sub if j) / len(sub))

def compute_bpref_single(judged_labels: List[Any], R: int, N: int) -> float:
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

def bootstrap_ci_single(vals: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float]:
    n = len(vals)
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(float(np.mean(vals[idx])))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return float(ci_low), float(ci_high)

def bootstrap_ci_paired(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    diff = a - b
    n = len(diff)
    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(float(np.mean(diff[idx])))
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    return float(np.mean(diff)), float(ci_low), float(ci_high)


def evaluate_dataset(df_gt_subset: pd.DataFrame, run_data_all: Dict[str, Any], query_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates all 12 configs on a specific ground-truth subset (End-to-End or Common Support).
    Returns (summary_df, per_query_df).
    """
    gt_map = {}
    gt_query_rel = {}
    gt_R = {}
    gt_N = {}

    for _, r in df_gt_subset.iterrows():
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

    for jid in gt_query_rel:
        gt_query_rel[jid].sort(reverse=True)

    summary_records = []
    per_query_records = []

    for cfg in CONFIGS:
        run_data = run_data_all[cfg]
        cfg_q_records = []

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
                    rel_scores.append(0.0)
                    judged_flags.append(False)
                    bpref_labels.append(None)

            ideal = gt_query_rel.get(jid, [])
            total_rel = len(ideal)

            n1 = ndcg_at_k(rel_scores, ideal, k=1)
            n3 = ndcg_at_k(rel_scores, ideal, k=3)
            n5 = ndcg_at_k(rel_scores, ideal, k=5)
            n10 = ndcg_at_k(rel_scores, ideal, k=10)
            mrr = mrr_at_k(rel_scores, k=10, threshold=1.0)
            p5, r5, f1_5 = precision_recall_at_k(rel_scores, total_rel, k=5, threshold=1.0)
            p10, r10, _ = precision_recall_at_k(rel_scores, total_rel, k=10, threshold=1.0)
            cp10 = context_precision_at_k(rel_scores, k=10, threshold=1.0)
            j10 = judged_at_k(judged_flags, k=10)

            R_val = gt_R.get(jid, 0)
            N_val = gt_N.get(jid, 0)
            bp = compute_bpref_single(bpref_labels, R_val, N_val)

            rec = {
                "config": cfg,
                "jd_id": jid,
                "ndcg@1": n1,
                "ndcg@3": n3,
                "ndcg@5": n5,
                "ndcg@10": n10,
                "p@5": p5,
                "r@10": r10,
                "f1@5": f1_5,
                "p@10": p10,
                "mrr": mrr,
                "context_precision@10": cp10,
                "bpref": bp,
                "judged@10": j10,
                "total_rel_in_pool": total_rel
            }
            cfg_q_records.append(rec)
            per_query_records.append(rec)

        # Aggregate across queries
        df_cfg = pd.DataFrame(cfg_q_records)
        ndcg10_vals = df_cfg["ndcg@10"].values
        ci_low, ci_high = bootstrap_ci_single(ndcg10_vals)

        sum_rec = {
            "config": cfg,
            "label": CONFIG_LABELS[cfg],
            "ndcg@1": float(df_cfg["ndcg@1"].mean()),
            "ndcg@3": float(df_cfg["ndcg@3"].mean()),
            "ndcg@5": float(df_cfg["ndcg@5"].mean()),
            "ndcg@10": float(df_cfg["ndcg@10"].mean()),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "p@5": float(df_cfg["p@5"].mean()),
            "r@10": float(df_cfg["r@10"].mean()),
            "p@10": float(df_cfg["p@10"].mean()),
            "mrr": float(df_cfg["mrr"].mean()),
            "context_precision@10": float(df_cfg["context_precision@10"].mean()),
            "bpref": float(df_cfg["bpref"].mean()),
            "judged@10": float(df_cfg["judged@10"].mean())
        }
        summary_records.append(sum_rec)

    return pd.DataFrame(summary_records), pd.DataFrame(per_query_records)


def main():
    print("=" * 80)
    print("  PHASE 3: EXPERIMENTS AND COMMON-SUPPORT ANALYSIS")
    print("  Evaluating End-to-End vs Common Support across 12 Configurations")
    print("=" * 80)

    # 1. Load inputs
    df_gt = pd.read_csv(GT_FILE)
    print(f"Loaded Unified Ground Truth: {len(df_gt)} pairs from {GT_FILE.name}")

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)["queries"]
    if isinstance(queries_raw, dict):
        query_ids = list(queries_raw.keys())
    else:
        query_ids = [q["jd_id"] for q in queries_raw]
    print(f"Loaded {len(query_ids)} Frozen Queries (15 Java, 10 React/TypeScript)")

    df_map = pd.read_csv(MAPPING_FILE)
    print(f"Loaded Evidence Mapping: {len(df_map)} pairs from {MAPPING_FILE.name}")

    # Load all 12 retrieval runs
    run_data_all = {}
    for cfg in CONFIGS:
        run_file = RUNS_DIR / f"{cfg}.json"
        with open(run_file, "r", encoding="utf-8") as f:
            run_data_all[cfg] = json.load(f)
    print("Loaded all 12 retrieval runs.")

    # 2. Define Common Support Membership
    # Task 3: Common-support query/evidence must have evidence represented in both strategy pools
    # Under any-overlap definition (Primary): common_support_overlap == True
    membership_records = []
    for _, r in df_map.iterrows():
        in_cs = bool(r["common_support_overlap"])
        if in_cs:
            reason = "Represented in both AST and Line pools (>=1 overlapping chunk)"
        elif r["ast_cov_type"] == "no_file":
            reason = "Excluded: file not present in AST pool (empty/no classes/functions)"
        else:
            reason = "Excluded: evidence lines outside AST extracted spans in file"

        membership_records.append({
            "pair_id": r["pair_id"],
            "jd_id": r["jd_id"],
            "repo_name": r["repo_name"],
            "file_path": r["file_path"],
            "start_line": r["start_line"],
            "end_line": r["end_line"],
            "ground_truth_label": r["ground_truth_label"],
            "in_common_support": in_cs,
            "exclusion_reason": reason,
            "in_strict_containment_support": bool(r["common_support_contained"])
        })
    df_membership = pd.DataFrame(membership_records)
    membership_out = RESULTS_DIR / "common_support_membership.csv"
    df_membership.to_csv(membership_out, index=False)
    print(f"Exported Common Support Membership: {membership_out.name} ({len(df_membership)} items)")

    # 3. Evaluate End-to-End ($N=882$)
    print("\n--- Computing End-to-End Metrics (N=882 Universe) ---")
    df_e2e_summary, df_e2e_per_query = evaluate_dataset(df_gt, run_data_all, query_ids)
    e2e_summary_out = RESULTS_DIR / "end_to_end_metrics.csv"
    df_e2e_summary.to_csv(e2e_summary_out, index=False)
    print(f"Saved End-to-End Metrics: {e2e_summary_out.name}")

    # 4. Evaluate Primary Common Support ($N=826$ Overlap Universe)
    print("\n--- Computing Common Support Metrics (N=826 Overlap Universe) ---")
    cs_pair_ids = set(df_membership[df_membership["in_common_support"]]["pair_id"])
    df_gt_cs = df_gt[df_map["pair_id"].isin(cs_pair_ids)].copy()
    print(f"Common Support Ground Truth size: {len(df_gt_cs)} pairs ({len(df_gt_cs[df_gt_cs['ground_truth_label'] > 0])} relevant)")

    df_cs_summary, df_cs_per_query = evaluate_dataset(df_gt_cs, run_data_all, query_ids)
    cs_summary_out = RESULTS_DIR / "common_support_metrics.csv"
    df_cs_summary.to_csv(cs_summary_out, index=False)
    print(f"Saved Common Support Metrics: {cs_summary_out.name}")

    cs_pq_out = RESULTS_DIR / "common_support_per_query.csv"
    df_cs_per_query.to_csv(cs_pq_out, index=False)
    print(f"Saved Common Support Per-Query: {cs_pq_out.name}")

    # 5. Evaluate Strict Containment Common Support ($N=330$ Universe) for Sensitivity
    print("\n--- Computing Strict Containment Support Metrics (N=330 Universe) ---")
    sc_pair_ids = set(df_membership[df_membership["in_strict_containment_support"]]["pair_id"])
    df_gt_sc = df_gt[df_map["pair_id"].isin(sc_pair_ids)].copy()
    df_sc_summary, df_sc_per_query = evaluate_dataset(df_gt_sc, run_data_all, query_ids)
    sc_summary_out = RESULTS_DIR / "common_support_strict_containment_metrics.csv"
    df_sc_summary.to_csv(sc_summary_out, index=False)

    # 6. Contrast End-to-End vs Common Support (Task 4 & 5)
    print("\n" + "=" * 80)
    print("  COMPARISON: END-TO-END vs COMMON SUPPORT (NDCG@10)")
    print("=" * 80)
    header = f"{'Configuration':<45} | {'E2E NDCG@10':<12} | {'CS NDCG@10':<12} | {'Delta (CS-E2E)':<14} | {'95% CI (CS)':<18}"
    print(header)
    print("-" * len(header))

    contrast_records = []
    for cfg in CONFIGS:
        e2e_row = df_e2e_summary[df_e2e_summary["config"] == cfg].iloc[0]
        cs_row = df_cs_summary[df_cs_summary["config"] == cfg].iloc[0]
        delta_cs_e2e = cs_row["ndcg@10"] - e2e_row["ndcg@10"]

        print(f"{CONFIG_LABELS[cfg]:<45} | {e2e_row['ndcg@10']:<12.4f} | {cs_row['ndcg@10']:<12.4f} | {delta_cs_e2e:<+14.4f} | [{cs_row['ci_95_low']:.4f}, {cs_row['ci_95_high']:.4f}]")

        contrast_records.append({
            "config": cfg,
            "label": CONFIG_LABELS[cfg],
            "e2e_ndcg10": float(e2e_row["ndcg@10"]),
            "e2e_bpref": float(e2e_row["bpref"]),
            "e2e_mrr": float(e2e_row["mrr"]),
            "e2e_p5": float(e2e_row["p@5"]),
            "e2e_r10": float(e2e_row["r@10"]),
            "cs_ndcg10": float(cs_row["ndcg@10"]),
            "cs_bpref": float(cs_row["bpref"]),
            "cs_mrr": float(cs_row["mrr"]),
            "cs_p5": float(cs_row["p@5"]),
            "cs_r10": float(cs_row["r@10"]),
            "delta_cs_minus_e2e_ndcg10": float(delta_cs_e2e),
            "cs_ci_95_low": float(cs_row["ci_95_low"]),
            "cs_ci_95_high": float(cs_row["ci_95_high"])
        })

    # Paired comparisons on Common Support
    print("\n--- Paired Head-to-Head Tests on Common Support (N=25 Queries) ---")
    comparisons = [
        ("Line+Header vs AST+Header (Re-rank)", "rerank_line_with_header", "rerank_ast_with_header"),
        ("Line No-Header vs AST No-Header (Re-rank)", "rerank_line_no_header", "rerank_ast_no_header"),
        ("AST Header Effect: AST+Header vs AST No-Header", "rerank_ast_with_header", "rerank_ast_no_header"),
        ("Line Header Effect: Line+Header vs Line No-Header", "rerank_line_with_header", "rerank_line_no_header"),
        ("Line+Header vs AST No-Header (Ablation Contrast)", "rerank_line_with_header", "rerank_ast_no_header")
    ]

    paired_test_results = []
    for comp_name, cfg_a, cfg_b in comparisons:
        # E2E paired
        a_e2e = df_e2e_per_query[df_e2e_per_query["config"] == cfg_a]["ndcg@10"].values
        b_e2e = df_e2e_per_query[df_e2e_per_query["config"] == cfg_b]["ndcg@10"].values
        diff_e2e = a_e2e - b_e2e
        stat_e2e, p_e2e = wilcoxon(diff_e2e, zero_method="pratt") if np.any(diff_e2e != 0) else (0.0, 1.0)
        mean_d_e2e, ci_l_e2e, ci_h_e2e = bootstrap_ci_paired(a_e2e, b_e2e)

        # CS paired
        a_cs = df_cs_per_query[df_cs_per_query["config"] == cfg_a]["ndcg@10"].values
        b_cs = df_cs_per_query[df_cs_per_query["config"] == cfg_b]["ndcg@10"].values
        diff_cs = a_cs - b_cs
        stat_cs, p_cs = wilcoxon(diff_cs, zero_method="pratt") if np.any(diff_cs != 0) else (0.0, 1.0)
        mean_d_cs, ci_l_cs, ci_h_cs = bootstrap_ci_paired(a_cs, b_cs)

        print(f"\n{comp_name}:")
        print(f"  End-to-End:     Delta = {mean_d_e2e:+.4f} (95% CI [{ci_l_e2e:.4f}, {ci_h_e2e:.4f}]), Wilcoxon W = {stat_e2e:.1f}, p = {p_e2e:.4f}")
        print(f"  Common Support: Delta = {mean_d_cs:+.4f} (95% CI [{ci_l_cs:.4f}, {ci_h_cs:.4f}]), Wilcoxon W = {stat_cs:.1f}, p = {p_cs:.4f}")

        paired_test_results.append({
            "comparison": comp_name,
            "cfg_a": cfg_a,
            "cfg_b": cfg_b,
            "e2e": {
                "mean_delta": float(mean_d_e2e),
                "ci_95_low": float(ci_l_e2e),
                "ci_95_high": float(ci_h_e2e),
                "wilcoxon_w": float(stat_e2e),
                "p_raw": float(p_e2e)
            },
            "common_support": {
                "mean_delta": float(mean_d_cs),
                "ci_95_low": float(ci_l_cs),
                "ci_95_high": float(ci_h_cs),
                "wilcoxon_w": float(stat_cs),
                "p_raw": float(p_cs)
            }
        })

    # Summary Report JSON
    report_json = {
        "phase": "Phase 3: Experiments and Common-Support Analysis",
        "date": "2026-10-19",
        "evidence_counts": {
            "total_pairs_universe": len(df_gt),
            "total_relevant_pairs": int((df_gt["ground_truth_label"] > 0).sum()),
            "common_support_pairs": len(df_gt_cs),
            "common_support_relevant_pairs": int((df_gt_cs["ground_truth_label"] > 0).sum()),
            "common_support_retention_pct": float(len(df_gt_cs) / len(df_gt) * 100),
            "common_support_relevant_retention_pct": float((df_gt_cs["ground_truth_label"] > 0).sum() / (df_gt["ground_truth_label"] > 0).sum() * 100),
            "strict_containment_pairs": len(df_gt_sc),
            "strict_containment_relevant_pairs": int((df_gt_sc["ground_truth_label"] > 0).sum()),
            "excluded_pairs": int(len(df_gt) - len(df_gt_cs)),
            "excluded_by_in_file_gap": int(sum(1 for r in membership_records if "outside AST" in r["exclusion_reason"])),
            "excluded_by_empty_file": int(sum(1 for r in membership_records if "empty" in r["exclusion_reason"]))
        },
        "query_retention": {
            "total_queries": len(query_ids),
            "queries_in_common_support": len(query_ids),
            "query_retention_pct": 100.0
        },
        "contrast_e2e_vs_common_support": contrast_records,
        "paired_comparisons": paired_test_results
    }

    report_json_path = RESULTS_DIR / "common_support_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    print(f"\nSaved structured report JSON: {report_json_path.name}")
    print("=" * 80)
    print("  PHASE 3 EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
