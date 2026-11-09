#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Phase 4: Statistical Analysis and Sensitivity Checks
Governed by: 05_STATISTICAL_ANALYSIS.md & Rules R22, R23, R24, R25
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
RESULTS_DIR = DATASET_DIR / "benchmark_results"
REPORTS_DIR = PROJECT_ROOT / "reports"

GT_FILE = DATASET_DIR / "ground_truth_final.csv"
QUERIES_FILE = DATASET_DIR / "queries_frozen.json"
E2E_PQ_FILE = RESULTS_DIR / "per_query_evaluation.csv"
CS_PQ_FILE = RESULTS_DIR / "common_support_per_query.csv"
CS_MEMBERSHIP_FILE = RESULTS_DIR / "common_support_membership.csv"


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


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Step-down Holm-Bonferroni correction over a family of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = p_val * (m - rank)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def compute_cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Cliff's delta effect size between two paired arrays."""
    diff = a - b
    n = len(diff)
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    return float((n_pos - n_neg) / n) if n > 0 else 0.0


def rank_biserial_correlation(diff: np.ndarray, w_stat: float) -> float:
    """Computes matched-pairs rank biserial correlation from Wilcoxon W."""
    nonzero_diff = diff[diff != 0]
    n = len(nonzero_diff)
    if n == 0:
        return 0.0
    t_total = n * (n + 1) / 2.0
    # r = 1 - 2*W_min / t_total, or r_rb = 4 * |W - t_total/2| / t_total
    return float(1.0 - (2.0 * w_stat / t_total))


def main():
    print("=" * 80)
    print("  PHASE 4: STATISTICAL ANALYSIS AND SENSITIVITY CHECKS")
    print("  Governed by 05_STATISTICAL_ANALYSIS.md")
    print("=" * 80)

    # 1. Load Data
    df_e2e_pq = pd.read_csv(E2E_PQ_FILE)
    df_cs_pq = pd.read_csv(CS_PQ_FILE)

    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)["queries"]
    if isinstance(queries_raw, dict):
        query_meta = {qid: qdata for qid, qdata in queries_raw.items()}
    else:
        query_meta = {q["jd_id"]: q for q in queries_raw}

    java_qids = [qid for qid, meta in query_meta.items() if meta.get("category", "").upper() == "JAVA" or "JAVA" in qid]
    react_qids = [qid for qid, meta in query_meta.items() if meta.get("category", "").upper() in ("REACT", "TYPESCRIPT", "REACT/TYPESCRIPT") or "REACT" in qid]

    print(f"Loaded {len(df_e2e_pq)} E2E query rows and {len(df_cs_pq)} Common-Support query rows.")
    print(f"Query counts: Java = {len(java_qids)}, React/TypeScript = {len(react_qids)}.")

    # 2. Define Primary Hypothesis Family (Family A: Common Support Re-rank Comparisons)
    # 4 registered hypothesis tests under Two-Stage Re-ranking on Common Support:
    # CS-Comp 1: Line+Header vs AST+Header (Primary Architecture Contrast on Shared Evidence)
    # CS-Comp 2: Line No-Header vs AST No-Header (Pure Architecture Contrast on Shared Evidence)
    # CS-Comp 3: AST+Header vs AST No-Header (AST Header Dependency on Shared Evidence)
    # CS-Comp 4: Line+Header vs Line No-Header (Line Header Dependency on Shared Evidence)
    family_a_defs = [
        {
            "id": "CS-Comp 1",
            "name": "Line+Header vs AST+Header (Common Support)",
            "cfg_a": "rerank_line_with_header",
            "cfg_b": "rerank_ast_with_header",
            "hyp": "Line+Header != AST+Header",
            "tier": "Two-Stage Re-ranking",
            "type": "Primary Architecture Contrast"
        },
        {
            "id": "CS-Comp 2",
            "name": "Line No-Header vs AST No-Header (Common Support)",
            "cfg_a": "rerank_line_no_header",
            "cfg_b": "rerank_ast_no_header",
            "hyp": "Line No-Header != AST No-Header",
            "tier": "Two-Stage Re-ranking",
            "type": "Pure Architecture Contrast"
        },
        {
            "id": "CS-Comp 3",
            "name": "AST+Header vs AST No-Header (Common Support)",
            "cfg_a": "rerank_ast_with_header",
            "cfg_b": "rerank_ast_no_header",
            "hyp": "AST+Header != AST No-Header",
            "tier": "Two-Stage Re-ranking",
            "type": "Metadata Ablation (AST)"
        },
        {
            "id": "CS-Comp 4",
            "name": "Line+Header vs Line No-Header (Common Support)",
            "cfg_a": "rerank_line_with_header",
            "cfg_b": "rerank_line_no_header",
            "hyp": "Line+Header != Line No-Header",
            "tier": "Two-Stage Re-ranking",
            "type": "Metadata Ablation (Line)"
        }
    ]

    # Run Wilcoxon and Bootstrap for Family A
    p_raw_a = []
    family_a_results = []

    for test in family_a_defs:
        a_vals = df_cs_pq[df_cs_pq["config"] == test["cfg_a"]].sort_values("jd_id")["ndcg@10"].values
        b_vals = df_cs_pq[df_cs_pq["config"] == test["cfg_b"]].sort_values("jd_id")["ndcg@10"].values
        diff = a_vals - b_vals

        mean_delta, ci_low, ci_high = bootstrap_ci_paired(a_vals, b_vals)
        res = wilcoxon(diff, zero_method="pratt")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
        p_raw_a.append(p_val)

        cliffs_d = compute_cliffs_delta(a_vals, b_vals)
        r_rb = rank_biserial_correlation(diff, stat)

        family_a_results.append({
            **test,
            "mean_a": float(np.mean(a_vals)),
            "mean_b": float(np.mean(b_vals)),
            "delta": float(mean_delta),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "wilcoxon_w": stat,
            "p_raw": p_val,
            "cliffs_delta": cliffs_d,
            "rank_biserial_r": r_rb,
            "n_queries": len(diff),
            "n_positive": int(np.sum(diff > 0)),
            "n_negative": int(np.sum(diff < 0)),
            "n_zero": int(np.sum(diff == 0))
        })

    # Apply Holm-Bonferroni on Family A
    p_adj_a = holm_bonferroni_correction(p_raw_a)
    for res, p_adj in zip(family_a_results, p_adj_a):
        res["p_holm"] = float(p_adj)
        res["significant_at_05"] = bool(p_adj < 0.05)

    print("\n--- Family A: Primary Common Support Comparisons (Re-rank, N=25) ---")
    for r in family_a_results:
        sig = "REJECT H0 (Sig)" if r["significant_at_05"] else "FAIL TO REJECT (Not Sig)"
        print(f"[{r['id']}] {r['name']}:")
        print(f"   Delta = {r['delta']:+.4f} (95% CI [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}])")
        print(f"   Wilcoxon W = {r['wilcoxon_w']:.1f}, p_raw = {r['p_raw']:.4f}, p_Holm = {r['p_holm']:.4f} -> {sig}")
        print(f"   Effect sizes: Cliff's delta = {r['cliffs_delta']:+.3f}, Rank-biserial r = {r['rank_biserial_r']:+.3f}")
        print(f"   Wins / Losses / Ties: {r['n_positive']} / {r['n_negative']} / {r['n_zero']}")

    # 3. Family B: End-to-End vs Common Support Contrast (Testing Coverage Impact)
    # Does the advantage of Line over AST change between End-to-End and Common Support?
    print("\n--- Family B: End-to-End vs Common Support Contrast Analysis ---")
    # For each query: delta_E2E = Line_E2E - AST_E2E, delta_CS = Line_CS - AST_CS
    # Difference in delta: gap_change = delta_E2E - delta_CS
    col_e2e = "ndcg_10" if "ndcg_10" in df_e2e_pq.columns else "ndcg@10"
    line_e2e = df_e2e_pq[df_e2e_pq["config"] == "rerank_line_with_header"].sort_values("jd_id")[col_e2e].values
    ast_e2e = df_e2e_pq[df_e2e_pq["config"] == "rerank_ast_with_header"].sort_values("jd_id")[col_e2e].values
    delta_e2e = line_e2e - ast_e2e

    col_cs = "ndcg@10" if "ndcg@10" in df_cs_pq.columns else "ndcg_10"
    line_cs = df_cs_pq[df_cs_pq["config"] == "rerank_line_with_header"].sort_values("jd_id")[col_cs].values
    ast_cs = df_cs_pq[df_cs_pq["config"] == "rerank_ast_with_header"].sort_values("jd_id")[col_cs].values
    delta_cs = line_cs - ast_cs

    contrast_diff = delta_e2e - delta_cs
    mean_gap_change, ci_gap_low, ci_gap_high = bootstrap_ci_paired(delta_e2e, delta_cs)
    res_contrast = wilcoxon(contrast_diff, zero_method="pratt")

    print(f"Line Advantage in E2E: Mean Delta = {np.mean(delta_e2e):+.4f}")
    print(f"Line Advantage in CS:  Mean Delta = {np.mean(delta_cs):+.4f}")
    print(f"Coverage Contribution to Gap: Mean = {mean_gap_change:+.4f} (95% CI [{ci_gap_low:.4f}, {ci_gap_high:.4f}])")
    print(f"Wilcoxon W = {res_contrast.statistic:.1f}, p = {res_contrast.pvalue:.4f}")

    family_b_result = {
        "id": "Contrast-Comp B1",
        "name": "Coverage Confound Impact on Line vs AST Advantage (With-Header)",
        "mean_delta_e2e": float(np.mean(delta_e2e)),
        "mean_delta_cs": float(np.mean(delta_cs)),
        "coverage_contribution_delta": float(mean_gap_change),
        "ci_95_low": float(ci_gap_low),
        "ci_95_high": float(ci_gap_high),
        "coverage_share_of_gap_pct": float(mean_gap_change / np.mean(delta_e2e) * 100),
        "wilcoxon_w": float(res_contrast.statistic),
        "p_raw": float(res_contrast.pvalue)
    }

    # 4. Family C: Language Subgroup Sensitivity (Java vs React/TypeScript)
    print("\n--- Family C: Language Subgroup Breakdown on Common Support ---")
    subgroups = [
        ("Java Subgroup (N=15 JDs)", java_qids),
        ("React/TypeScript Subgroup (N=10 JDs)", react_qids)
    ]

    subgroup_results = []
    for sub_name, qids in subgroups:
        df_sub = df_cs_pq[df_cs_pq["jd_id"].isin(qids)]
        l_hdr = df_sub[df_sub["config"] == "rerank_line_with_header"].sort_values("jd_id")["ndcg@10"].values
        a_hdr = df_sub[df_sub["config"] == "rerank_ast_with_header"].sort_values("jd_id")["ndcg@10"].values
        l_nohdr = df_sub[df_sub["config"] == "rerank_line_no_header"].sort_values("jd_id")["ndcg@10"].values
        a_nohdr = df_sub[df_sub["config"] == "rerank_ast_no_header"].sort_values("jd_id")["ndcg@10"].values

        d_hdr = l_hdr - a_hdr
        mean_d_hdr, ci_l_hdr, ci_h_hdr = bootstrap_ci_paired(l_hdr, a_hdr)
        w_hdr = wilcoxon(d_hdr, zero_method="pratt")

        d_nohdr = l_nohdr - a_nohdr
        mean_d_nohdr, ci_l_nohdr, ci_h_nohdr = bootstrap_ci_paired(l_nohdr, a_nohdr)
        w_nohdr = wilcoxon(d_nohdr, zero_method="pratt")

        print(f"\n{sub_name}:")
        print(f"  Line+Header vs AST+Header: Mean Delta = {mean_d_hdr:+.4f} (95% CI [{ci_l_hdr:.4f}, {ci_h_hdr:.4f}]), W = {w_hdr.statistic:.1f}, p = {w_hdr.pvalue:.4f}")
        print(f"  Line No-Header vs AST No-Header: Mean Delta = {mean_d_nohdr:+.4f} (95% CI [{ci_l_nohdr:.4f}, {ci_h_nohdr:.4f}]), W = {w_nohdr.statistic:.1f}, p = {w_nohdr.pvalue:.4f}")

        subgroup_results.append({
            "subgroup": sub_name,
            "n_queries": len(qids),
            "line_with_header_mean": float(np.mean(l_hdr)),
            "ast_with_header_mean": float(np.mean(a_hdr)),
            "with_header_delta": float(mean_d_hdr),
            "with_header_ci_low": float(ci_l_hdr),
            "with_header_ci_high": float(ci_h_hdr),
            "with_header_w": float(w_hdr.statistic),
            "with_header_p": float(w_hdr.pvalue),
            "line_no_header_mean": float(np.mean(l_nohdr)),
            "ast_no_header_mean": float(np.mean(a_nohdr)),
            "no_header_delta": float(mean_d_nohdr),
            "no_header_ci_low": float(ci_l_nohdr),
            "no_header_ci_high": float(ci_h_nohdr),
            "no_header_w": float(w_nohdr.statistic),
            "no_header_p": float(w_nohdr.pvalue)
        })

    # 5. Exploratory Analysis: Strict Containment Support ($N=330$)
    df_sc_summary = pd.read_csv(RESULTS_DIR / "common_support_strict_containment_metrics.csv")
    sc_l_hdr = df_sc_summary[df_sc_summary["config"] == "rerank_line_with_header"]["ndcg@10"].values[0]
    sc_a_hdr = df_sc_summary[df_sc_summary["config"] == "rerank_ast_with_header"]["ndcg@10"].values[0]
    sc_l_nohdr = df_sc_summary[df_sc_summary["config"] == "rerank_line_no_header"]["ndcg@10"].values[0]
    sc_a_nohdr = df_sc_summary[df_sc_summary["config"] == "rerank_ast_no_header"]["ndcg@10"].values[0]

    strict_containment_results = {
        "n_pairs": 330,
        "line_with_header_ndcg10": float(sc_l_hdr),
        "ast_with_header_ndcg10": float(sc_a_hdr),
        "with_header_delta": float(sc_l_hdr - sc_a_hdr),
        "line_no_header_ndcg10": float(sc_l_nohdr),
        "ast_no_header_ndcg10": float(sc_a_nohdr),
        "no_header_delta": float(sc_l_nohdr - sc_a_nohdr)
    }
    print("\n--- Exploratory: Strict Containment Common Support (N=330 Pairs) ---")
    print(f"  Line+Header: {sc_l_hdr:.4f} vs AST+Header: {sc_a_hdr:.4f} (Delta = {sc_l_hdr - sc_a_hdr:+.4f})")
    print(f"  Line No-Header: {sc_l_nohdr:.4f} vs AST No-Header: {sc_a_nohdr:.4f} (Delta = {sc_l_nohdr - sc_a_nohdr:+.4f})")

    # 6. Save comprehensive JSON output
    revision_stats_json = {
        "phase": "Phase 4: Statistical Analysis and Sensitivity Checks",
        "date": "2026-10-19",
        "seed": 42,
        "n_boot": 10000,
        "family_a_primary_tests": family_a_results,
        "family_b_contrast_tests": family_b_result,
        "family_c_language_subgroups": subgroup_results,
        "exploratory_strict_containment": strict_containment_results
    }

    out_file = RESULTS_DIR / "revision_statistical_tests.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(revision_stats_json, f, indent=2)
    print(f"\nSaved structured statistical results: {out_file.name}")
    print("=" * 80)
    print("  PHASE 4 STATISTICAL RUN COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
