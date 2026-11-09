#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Task: INTER-ANNOTATOR AGREEMENT EVALUATION & GROUND TRUTH CURATION
Metrics: Quadratic Weighted Cohen's Kappa (kappa_w) & 3x3 Confusion Matrix
Target: IEEE SANER 2027 (ERA Track) & Double-Anonymous Peer Review
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Configure UTF-8 encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BENCHMARK_RESULTS_DIR = DATASET_DIR / "benchmark_results"
BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_default_files():
    """Locate default labeled files in dataset directory."""
    a1_candidates = [
        DATASET_DIR / "ground_truth_annotator1.csv",
        DATASET_DIR / "ground_truth_final.csv"
    ]
    a1_file = next((f for f in a1_candidates if f.exists()), None)

    a2_candidates = [
        DATASET_DIR / "ground_truth_annotator2.csv",
        DATASET_DIR / "ground_truth_final.csv"
    ]
    a2_file = next((f for f in a2_candidates if f.exists()), None)

    return a1_file, a2_file


def interpret_kappa(kappa_val):
    """Interpret kappa score according to Landis & Koch (1977) benchmark standards."""
    if kappa_val < 0.0:
        return "Poor Agreement", "POOR"
    elif kappa_val <= 0.20:
        return "Slight Agreement", "SLIGHT"
    elif kappa_val <= 0.40:
        return "Fair Agreement", "FAIR"
    elif kappa_val <= 0.60:
        return "Moderate Agreement", "MODERATE"
    elif kappa_val <= 0.80:
        return "Substantial Agreement", "SUBSTANTIAL"
    else:
        return "Almost Perfect Agreement", "ALMOST_PERFECT"


def plot_confusion_matrix(cm, labels, output_path):
    """Plot confusion matrix heatmap and save to file."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Inter-Annotator Confusion Matrix (3-Tier Scale)", fontsize=13, fontweight='bold', pad=15)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        tick_marks = np.arange(len(labels))
        class_names = [f"Level {l}" for l in labels]
        plt.xticks(tick_marks, class_names, fontsize=11)
        plt.yticks(tick_marks, class_names, fontsize=11)
        plt.xlabel("Annotator 2 Labels", fontsize=11, labelpad=10, fontweight='bold')
        plt.ylabel("Annotator 1 Labels", fontsize=11, labelpad=10, fontweight='bold')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                plt.text(j, i, f"{cm[i, j]:d}\n({cm[i, j]/np.sum(cm)*100:.1f}%)",
                         horizontalalignment="center", verticalalignment="center",
                         color=color, fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return True
    except Exception as e:
        print(f"Warning: Could not plot confusion matrix: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate Inter-Annotator Agreement (Quadratic Weighted Cohen's Kappa)")
    parser.add_argument("--file-a1", type=str, default=None, help="Path to Annotator 1 labeled CSV")
    parser.add_argument("--file-a2", type=str, default=None, help="Path to Annotator 2 labeled CSV")
    parser.add_argument("--col-a1", type=str, default=None, help="Column name for Annotator 1 label")
    parser.add_argument("--col-a2", type=str, default=None, help="Column name for Annotator 2 label")
    parser.add_argument("--output-final", type=str, default=str(DATASET_DIR / "ground_truth_final.csv"))
    parser.add_argument("--output-report", type=str, default=str(BENCHMARK_RESULTS_DIR / "kappa_evaluation_report.txt"))
    parser.add_argument("--output-plot", type=str, default=str(BENCHMARK_RESULTS_DIR / "confusion_matrix_kappa.png"))
    parser.add_argument("--disagreements", type=str, default=str(DATASET_DIR / "disagreements_adjudication.csv"))
    args = parser.parse_args()

    # Resolve default files
    default_a1, default_a2 = find_default_files()
    path_a1 = Path(args.file_a1) if args.file_a1 else default_a1
    path_a2 = Path(args.file_a2) if args.file_a2 else default_a2

    if not path_a1 or not path_a1.exists():
        print(f"Error: Missing labeled data file for Annotator 1 at {path_a1}")
        return
    if not path_a2 or not path_a2.exists():
        print(f"Error: Missing labeled data file for Annotator 2 at {path_a2}")
        return

    df_a1 = pd.read_csv(path_a1)
    df_a2 = pd.read_csv(path_a2)

    # Resolve columns
    col_a1 = args.col_a1
    if not col_a1:
        for c in ['human_label_annotator1', 'human_label', 'relevance_level']:
            if c in df_a1.columns:
                col_a1 = c
                break

    col_a2 = args.col_a2
    if not col_a2:
        for c in ['human_label_annotator2', 'human_label', 'relevance_level']:
            if c in df_a2.columns:
                col_a2 = c
                break

    # Merge on pair_id
    if 'pair_id' in df_a1.columns and 'pair_id' in df_a2.columns and path_a1 != path_a2:
        cols_to_merge_a1 = ['pair_id', col_a1]
        for extra_col in ['jd_id', 'query_id']:
            if extra_col in df_a1.columns:
                cols_to_merge_a1.append(extra_col)
                break
        merged = pd.merge(df_a1[cols_to_merge_a1], df_a2[['pair_id', col_a2]], on='pair_id', suffixes=('_a1', '_a2'))
        y_a1 = merged[f"{col_a1}_a1"].values
        y_a2 = merged[f"{col_a2}_a2"].values
        q_ids = merged['jd_id'].values if 'jd_id' in merged.columns else (merged['query_id'].values if 'query_id' in merged.columns else None)
    else:
        y_a1 = df_a1[col_a1].values
        y_a2 = df_a1[col_a2].values
        q_ids = df_a1['jd_id'].values if 'jd_id' in df_a1.columns else (df_a1['query_id'].values if 'query_id' in df_a1.columns else None)

    # Clean valid integer labels (0, 1, 2)
    valid_mask = pd.notna(y_a1) & pd.notna(y_a2)
    y_a1 = y_a1[valid_mask].astype(int)
    y_a2 = y_a2[valid_mask].astype(int)
    if q_ids is not None:
        q_ids = q_ids[valid_mask]

    n_pairs = len(y_a1)
    labels = [0, 1, 2]

    def compute_partition_stats(ya, yb, name):
        n = len(ya)
        kw = float(cohen_kappa_score(ya, yb, weights='quadratic', labels=labels))
        kl = float(cohen_kappa_score(ya, yb, weights='linear', labels=labels))
        ku = float(cohen_kappa_score(ya, yb, labels=labels))
        matches = int(np.sum(ya == yb))
        po = float(matches / n)
        diffs = np.abs(ya - yb)
        dev1 = int(np.sum(diffs == 1))
        dev2 = int(np.sum(diffs == 2))
        level, icon = interpret_kappa(kw)
        return {
            "partition": name,
            "n": n,
            "exact_matches": matches,
            "exact_agreement_pct": round(po * 100.0, 2),
            "disagreement_delta_1": dev1,
            "disagreement_delta_1_pct": round(dev1 / n * 100.0, 2),
            "disagreement_delta_2": dev2,
            "disagreement_delta_2_pct": round(dev2 / n * 100.0, 2),
            "unweighted_kappa": round(ku, 4),
            "linear_weighted_kappa": round(kl, 4),
            "quadratic_weighted_kappa": round(kw, 4),
            "agreement_level": level,
            "status_icon": icon,
        }

    total_stats = compute_partition_stats(y_a1, y_a2, "Complete Unified Ground Truth (Total)")

    subsets = {}
    if q_ids is not None:
        java_mask = np.array([str(qid).startswith("JD_JAVA") for qid in q_ids])
        react_mask = np.array([str(qid).startswith("JD_REACT") for qid in q_ids])
        if np.sum(java_mask) > 0 and np.sum(react_mask) > 0:
            subsets["java_subgroup"] = compute_partition_stats(y_a1[java_mask], y_a2[java_mask], "Java Spring Boot Ecosystem")
            subsets["react_subgroup"] = compute_partition_stats(y_a1[react_mask], y_a2[react_mask], "React / TypeScript Ecosystem")

    cm = confusion_matrix(y_a1, y_a2, labels=labels)

    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("\n" + "=" * 80)
    log("  INTER-ANNOTATOR AGREEMENT EVALUATION REPORT (TABLE 3.1)")
    log("=" * 80)
    log(f"• Total Evaluated Candidate Evidence Pairs (N): {total_stats['n']}")
    log(f"• Exact Consensus Matches (po)               : {total_stats['exact_matches']} / {total_stats['n']} ({total_stats['exact_agreement_pct']:.2f}%)")
    log(f"• Minor Disagreements (|delta| = 1)          : {total_stats['disagreement_delta_1']} / {total_stats['n']} ({total_stats['disagreement_delta_1_pct']:.2f}%)")
    log(f"• Severe Disagreements (|delta| = 2)         : {total_stats['disagreement_delta_2']} / {total_stats['n']} ({total_stats['disagreement_delta_2_pct']:.2f}%)")
    log("-" * 80)
    log(f"• Unweighted Cohen's Kappa (kappa)            : {total_stats['unweighted_kappa']:.4f}")
    log(f"• Linear Weighted Kappa (kappa_linear)        : {total_stats['linear_weighted_kappa']:.4f}")
    log(f"• QUADRATIC WEIGHTED KAPPA (kappa_w)          : {total_stats['quadratic_weighted_kappa']:.4f}  <-- [PRIMARY METRIC]")
    log(f"• Agreement Level (Landis & Koch, 1977)       : {total_stats['agreement_level']} [{total_stats['status_icon']}]")
    log("=" * 80)

    if subsets:
        log("\nPARTITION BREAKDOWN:")
        log("   " + "-" * 75)
        log(f"   {'Partition':<35} | {'N':^5} | {'Exact (%)':^12} | {'kappa_w':^8} | {'Landis & Koch':^15}")
        log("   " + "-" * 75)
        for k, sub in subsets.items():
            log(f"   {sub['partition']:<35} | {sub['n']:^5} | {sub['exact_matches']} ({sub['exact_agreement_pct']:.1f}%) | {sub['quadratic_weighted_kappa']:.4f} | {sub['agreement_level']:<15}")
        log(f"   {total_stats['partition']:<35} | {total_stats['n']:^5} | {total_stats['exact_matches']} ({total_stats['exact_agreement_pct']:.1f}%) | {total_stats['quadratic_weighted_kappa']:.4f} | {total_stats['agreement_level']:<15}")
        log("   " + "-" * 75)

    log("\nCONFUSION MATRIX (3x3):")
    log("   (Rows: Annotator 1  x  Columns: Annotator 2)")
    log("   " + "-" * 65)
    log(f"   {'Annotator 1 \\ Annotator 2':<26} | {'Level 0':^10} | {'Level 1':^10} | {'Level 2':^10} | {'Row Sum':^9}")
    log("   " + "-" * 65)
    for i, l in enumerate(labels):
        row_sum = np.sum(cm[i, :])
        log(f"   Level {l:<20} | {cm[i, 0]:^10} | {cm[i, 1]:^10} | {cm[i, 2]:^10} | {row_sum:^9}")
    log("   " + "-" * 65)
    col_sums = np.sum(cm, axis=0)
    log(f"   {'Column Sum (Annotator 2)':<26} | {col_sums[0]:^10} | {col_sums[1]:^10} | {col_sums[2]:^10} | {n_pairs:^9}")
    log("   " + "-" * 65)

    # Plot confusion matrix
    plot_path = Path(args.output_plot)
    if plot_confusion_matrix(cm, labels, plot_path):
        log(f"\nSaved confusion matrix plot: {plot_path.name}")

    # Write report file
    report_path = Path(args.output_report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"Saved evaluation report: {report_path.name}")

    # Write json report file
    json_path = report_path.with_suffix(".json")
    json_data = {
        "total": total_stats,
        "subsets": subsets,
        "confusion_matrix": cm.tolist()
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    log(f"Saved evaluation JSON report: {json_path.name}")


if __name__ == "__main__":
    main()
