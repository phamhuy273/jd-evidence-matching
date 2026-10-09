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
        DATASET_DIR / "ground_truth_annotator1_labeled.csv",
        DATASET_DIR / "ground_truth_final.csv"
    ]
    a1_file = next((f for f in a1_candidates if f.exists()), None)

    a2_candidates = [
        DATASET_DIR / "ground_truth_annotator2_labeled.csv",
        DATASET_DIR / "ground_truth_final.csv"
    ]
    a2_file = next((f for f in a2_candidates if f.exists()), None)

    return a1_file, a2_file


def interpret_kappa(kappa_val):
    """Interpret kappa score according to Landis & Koch (1977) benchmark standards."""
    if kappa_val < 0.0:
        return "Poor Agreement", "POOR"
    elif kappa_val < 0.40:
        return "Slight / Fair Agreement", "FAIR"
    elif kappa_val < 0.60:
        return "Moderate Agreement", "MODERATE"
    elif kappa_val < 0.75:
        return "Substantial Agreement", "SUBSTANTIAL"
    else:
        return "Almost Perfect / Excellent Agreement", "EXCELLENT"


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
        merged = pd.merge(df_a1[['pair_id', col_a1]], df_a2[['pair_id', col_a2]], on='pair_id', suffixes=('_a1', '_a2'))
        y_a1 = merged[f"{col_a1}_a1"].values
        y_a2 = merged[f"{col_a2}_a2"].values
    else:
        y_a1 = df_a1[col_a1].values
        y_a2 = df_a1[col_a2].values

    # Clean valid integer labels (0, 1, 2)
    valid_mask = pd.notna(y_a1) & pd.notna(y_a2)
    y_a1 = y_a1[valid_mask].astype(int)
    y_a2 = y_a2[valid_mask].astype(int)

    n_pairs = len(y_a1)
    labels = [0, 1, 2]

    # Metrics
    kappa_quadratic = cohen_kappa_score(y_a1, y_a2, weights='quadratic', labels=labels)
    kappa_linear = cohen_kappa_score(y_a1, y_a2, weights='linear', labels=labels)
    kappa_unweighted = cohen_kappa_score(y_a1, y_a2, labels=labels)

    raw_matches = int(np.sum(y_a1 == y_a2))
    po = raw_matches / n_pairs
    diffs = np.abs(y_a1 - y_a2)
    minor_disagreements = int(np.sum(diffs == 1))
    severe_disagreements = int(np.sum(diffs == 2))

    cm = confusion_matrix(y_a1, y_a2, labels=labels)
    level_desc, status_icon = interpret_kappa(kappa_quadratic)

    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("\n" + "=" * 80)
    log("  INTER-ANNOTATOR AGREEMENT EVALUATION REPORT (TABLE 3.1)")
    log("=" * 80)
    log(f"• Total Evaluated Candidate Evidence Pairs (N): {n_pairs}")
    log(f"• Exact Consensus Matches (po)               : {raw_matches} / {n_pairs} ({po*100:.2f}%)")
    log(f"• Minor Disagreements (|delta| = 1)          : {minor_disagreements} / {n_pairs} ({(minor_disagreements/n_pairs)*100:.2f}%)")
    log(f"• Severe Disagreements (|delta| = 2)         : {severe_disagreements} / {n_pairs} ({(severe_disagreements/n_pairs)*100:.2f}%)")
    log("-" * 80)
    log(f"• Unweighted Cohen's Kappa (kappa)            : {kappa_unweighted:.4f}")
    log(f"• Linear Weighted Kappa (kappa_linear)        : {kappa_linear:.4f}")
    log(f"• QUADRATIC WEIGHTED KAPPA (kappa_w)          : {kappa_quadratic:.4f}  <-- [PRIMARY METRIC]")
    log(f"• Agreement Level (Landis & Koch, 1977)       : {level_desc} [{status_icon}]")
    log("=" * 80)

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


if __name__ == "__main__":
    main()
