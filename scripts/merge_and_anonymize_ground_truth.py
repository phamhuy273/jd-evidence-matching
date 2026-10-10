#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Task: MERGE, ADJUDICATE, ANONYMIZE & FINALIZE GROUND TRUTH DATASETS
Target: IEEE SANER 2027 (ERA Track) & Double-Anonymous Peer Review
Governed by: Rules R17, R18, R19, R21, R44
=============================================================================
"""

import os
import sys
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Windows UTF-8 stdout configuration
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
SCRATCH_DIR = PROJECT_ROOT / "scratch"
BACKUP_DIR = SCRATCH_DIR / "backup_raw_annotations"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("  PHASE 4: GROUND TRUTH MERGE, ADJUDICATION & ANONYMIZATION")
    print("=" * 80)

    # 1. Paths
    gold_final_path = DATASET_DIR / "ground_truth_final.csv"
    gold_a1_path = DATASET_DIR / "ground_truth_annotator1_labeled.csv"
    gold_a2_path = DATASET_DIR / "ground_truth_annotator2_labeled.csv"
    old_disag_path = DATASET_DIR / "disagreements_adjudication.csv"

    p2_huy_path = DATASET_DIR / "ground_truth_huy_raw.csv"
    p2_an_path = DATASET_DIR / "ground_truth_an_raw.csv"
    p2_adj_json = SCRATCH_DIR / "adjudicated_items_phase4.json"

    # Backup raw files before modifying
    for src in [gold_final_path, gold_a1_path, gold_a2_path, old_disag_path, p2_huy_path, p2_an_path]:
        if src.exists():
            shutil.copy2(src, BACKUP_DIR / src.name)
    print(f"Backed up raw files to {BACKUP_DIR}")

    # 2. Load Part 1 (250 Gold items)
    df_gold = pd.read_csv(gold_final_path)
    df_gold_a1 = pd.read_csv(gold_a1_path)
    df_gold_a2 = pd.read_csv(gold_a2_path)
    df_gold_disag = pd.read_csv(old_disag_path)
    print(f"Loaded Gold 250 items: {len(df_gold)} rows, {len(df_gold_disag)} disagreements")

    # 3. Load Part 2 (353 Pooled items)
    df_p2_huy = pd.read_csv(p2_huy_path)
    df_p2_an = pd.read_csv(p2_an_path)
    with open(p2_adj_json, "r", encoding="utf-8") as f:
        p2_adj_list = json.load(f)
    p2_adj_map = {item["pair_id"]: item for item in p2_adj_list}
    print(f"Loaded Pooled 353 items: {len(df_p2_huy)} rows, {len(p2_adj_map)} adjudicated disagreements")

    # 4. Standardize and Merge Annotator 1
    # Part 1: PAIR_001 to PAIR_250
    rows_a1 = []
    for idx, r in df_gold_a1.iterrows():
        rows_a1.append({
            "pair_id": r["pair_id"],
            "jd_id": r["jd_id"],
            "jd_title": r["jd_title"],
            "jd_level": r["jd_level"],
            "jd_domain": r["jd_domain"],
            "jd_mandatory_skills": r["jd_mandatory_skills"],
            "repo_name": r["repo_name"],
            "file_path": r["file_path"],
            "class_name": r.get("class_name", ""),
            "method_name": r.get("method_name", ""),
            "start_line": int(r["start_line"]),
            "end_line": int(r["end_line"]),
            "context_header": r.get("context_header", ""),
            "chunk_content": r["chunk_content"],
            "human_label": int(r["human_label"]),
            "human_note": r.get("human_note", "")
        })

    # Part 2: PAIR_251 to PAIR_603
    for idx, r in df_p2_huy.iterrows():
        new_pair_id = f"PAIR_{251 + idx:03d}"
        rows_a1.append({
            "pair_id": new_pair_id,
            "jd_id": r["jd_id"],
            "jd_title": r["jd_title"],
            "jd_level": r["jd_level"],
            "jd_domain": r["jd_domain"],
            "jd_mandatory_skills": r["jd_mandatory_skills"],
            "repo_name": r["repo_name"],
            "file_path": r["file_path"],
            "class_name": "",
            "method_name": "",
            "start_line": int(r["start_line"]),
            "end_line": int(r["end_line"]),
            "context_header": r.get("context_header", ""),
            "chunk_content": r["chunk_content"],
            "human_label": int(r["human_label"]),
            "human_note": r.get("human_note", "") if pd.notna(r.get("human_note")) else ""
        })

    df_annotator1 = pd.DataFrame(rows_a1)

    # 5. Standardize and Merge Annotator 2
    rows_a2 = []
    for idx, r in df_gold_a2.iterrows():
        rows_a2.append({
            "pair_id": r["pair_id"],
            "jd_id": r["jd_id"],
            "jd_title": r["jd_title"],
            "jd_level": r["jd_level"],
            "jd_domain": r["jd_domain"],
            "jd_mandatory_skills": r["jd_mandatory_skills"],
            "repo_name": r["repo_name"],
            "file_path": r["file_path"],
            "class_name": r.get("class_name", ""),
            "method_name": r.get("method_name", ""),
            "start_line": int(r["start_line"]),
            "end_line": int(r["end_line"]),
            "context_header": r.get("context_header", ""),
            "chunk_content": r["chunk_content"],
            "human_label": int(r["human_label"]),
            "human_note": r.get("human_note", "")
        })

    for idx, r in df_p2_an.iterrows():
        new_pair_id = f"PAIR_{251 + idx:03d}"
        rows_a2.append({
            "pair_id": new_pair_id,
            "jd_id": r["jd_id"],
            "jd_title": r["jd_title"],
            "jd_level": r["jd_level"],
            "jd_domain": r["jd_domain"],
            "jd_mandatory_skills": r["jd_mandatory_skills"],
            "repo_name": r["repo_name"],
            "file_path": r["file_path"],
            "class_name": "",
            "method_name": "",
            "start_line": int(r["start_line"]),
            "end_line": int(r["end_line"]),
            "context_header": r.get("context_header", ""),
            "chunk_content": r["chunk_content"],
            "human_label": int(r["human_label"]),
            "human_note": r.get("human_note", "") if pd.notna(r.get("human_note")) else ""
        })

    df_annotator2 = pd.DataFrame(rows_a2)

    # 6. Build Master Disagreements File (122 rows)
    rows_disag = []
    # Old disagreements (29 rows)
    gold_map = {r["pair_id"]: r for _, r in df_gold.iterrows()}
    for idx, r in df_gold_disag.iterrows():
        p_id = r["pair_id"]
        g_item = gold_map.get(p_id, {})
        rows_disag.append({
            "pair_id": p_id,
            "jd_id": g_item.get("jd_id", ""),
            "jd_title": r["jd_title"],
            "repo_name": r["repo_name"],
            "file_path": r["file_path"],
            "start_line": int(g_item.get("start_line", 0)),
            "end_line": int(g_item.get("end_line", 0)),
            "class_name": g_item.get("class_name", ""),
            "method_name": r.get("method_name", ""),
            "annotator_1_label": int(r["human_label_annotator1"]),
            "annotator_2_label": int(r["human_label_annotator2"]),
            "final_resolved_label": int(r["final_resolved_label"]),
            "resolution_reason": r["resolution_reason"],
            "adjudicated_by": "Lead Adjudicator"
        })

    # New disagreements (93 rows)
    for idx, r in df_p2_huy.iterrows():
        old_item_id = r["pair_id"]
        if old_item_id in p2_adj_map:
            adj = p2_adj_map[old_item_id]
            new_pair_id = f"PAIR_{251 + idx:03d}"
            rows_disag.append({
                "pair_id": new_pair_id,
                "jd_id": r["jd_id"],
                "jd_title": r["jd_title"],
                "repo_name": r["repo_name"],
                "file_path": r["file_path"],
                "start_line": int(r["start_line"]),
                "end_line": int(r["end_line"]),
                "class_name": "",
                "method_name": "",
                "annotator_1_label": int(adj["annotator_1_label"]),
                "annotator_2_label": int(adj["annotator_2_label"]),
                "final_resolved_label": int(adj["final_resolved_label"]),
                "resolution_reason": adj["resolution_reason"],
                "adjudicated_by": "Lead Adjudicator"
            })

    df_disagreements = pd.DataFrame(rows_disag)

    # 7. Build Ground Truth Final (603 rows)
    rows_final = []
    # First 250 Gold items
    for idx, r in df_gold.iterrows():
        rows_final.append({
            "pair_id": r["pair_id"],
            "jd_id": r["jd_id"],
            "jd_title": r["jd_title"],
            "jd_level": r["jd_level"],
            "jd_domain": r["jd_domain"],
            "jd_mandatory_skills": r["jd_mandatory_skills"],
            "repo_name": r["repo_name"],
            "file_path": r["file_path"],
            "class_name": r.get("class_name", ""),
            "method_name": r.get("method_name", ""),
            "start_line": int(r["start_line"]),
            "end_line": int(r["end_line"]),
            "context_header": r.get("context_header", ""),
            "chunk_content": r["chunk_content"],
            "annotator_1_label": int(r["human_label_annotator1"]),
            "annotator_2_label": int(r["human_label_annotator2"]),
            "ground_truth_label": int(r["ground_truth_label"]),
            "adjudication_note": r.get("adjudication_note", "")
        })

    # 353 Pooled items
    for idx, r in df_p2_huy.iterrows():
        old_item_id = r["pair_id"]
        new_pair_id = f"PAIR_{251 + idx:03d}"
        l1 = int(r["human_label"])
        l2 = int(df_p2_an.iloc[idx]["human_label"])

        if old_item_id in p2_adj_map:
            adj = p2_adj_map[old_item_id]
            gt_label = int(adj["final_resolved_label"])
            adj_note = adj["resolution_reason"]
        else:
            gt_label = l1
            adj_note = "Consensus agreement"

        rows_final.append({
            "pair_id": new_pair_id,
            "jd_id": r["jd_id"],
            "jd_title": r["jd_title"],
            "jd_level": r["jd_level"],
            "jd_domain": r["jd_domain"],
            "jd_mandatory_skills": r["jd_mandatory_skills"],
            "repo_name": r["repo_name"],
            "file_path": r["file_path"],
            "class_name": "",
            "method_name": "",
            "start_line": int(r["start_line"]),
            "end_line": int(r["end_line"]),
            "context_header": r.get("context_header", ""),
            "chunk_content": r["chunk_content"],
            "annotator_1_label": l1,
            "annotator_2_label": l2,
            "ground_truth_label": gt_label,
            "adjudication_note": adj_note
        })

    df_final = pd.DataFrame(rows_final)

    # 8. Save unified files
    print("\nWriting unified datasets...")
    df_annotator1.to_csv(DATASET_DIR / "ground_truth_annotator1.csv", index=False, encoding="utf-8")
    df_annotator1.to_csv(DATASET_DIR / "ground_truth_annotator1_labeled.csv", index=False, encoding="utf-8")

    df_annotator2.to_csv(DATASET_DIR / "ground_truth_annotator2.csv", index=False, encoding="utf-8")
    df_annotator2.to_csv(DATASET_DIR / "ground_truth_annotator2_labeled.csv", index=False, encoding="utf-8")

    df_disagreements.to_csv(DATASET_DIR / "disagreements_adjudication.csv", index=False, encoding="utf-8")
    df_final.to_csv(DATASET_DIR / "ground_truth_final.csv", index=False, encoding="utf-8")

    # Also mirror to desktop replication package
    desktop_repl_dataset = Path("C:/Users/win 11/Desktop/ast-rag-replication-package/dataset")
    if desktop_repl_dataset.exists():
        df_annotator1.to_csv(desktop_repl_dataset / "ground_truth_annotator1.csv", index=False, encoding="utf-8")
        df_annotator1.to_csv(desktop_repl_dataset / "ground_truth_annotator1_labeled.csv", index=False, encoding="utf-8")
        df_annotator2.to_csv(desktop_repl_dataset / "ground_truth_annotator2.csv", index=False, encoding="utf-8")
        df_annotator2.to_csv(desktop_repl_dataset / "ground_truth_annotator2_labeled.csv", index=False, encoding="utf-8")
        df_disagreements.to_csv(desktop_repl_dataset / "disagreements_adjudication.csv", index=False, encoding="utf-8")
        df_final.to_csv(desktop_repl_dataset / "ground_truth_final.csv", index=False, encoding="utf-8")
        print("Mirrored unified datasets to Desktop replication package.")

    # 9. Clean up raw files and AI caches from dataset/
    for cleanup_file in [p2_huy_path, p2_an_path, DATASET_DIR / "gemini_annotations_cache.json"]:
        if cleanup_file.exists():
            cleanup_file.unlink()
            print(f"Cleaned up intermediate file: {cleanup_file.name}")

    # 10. Summary verification
    print("\n" + "=" * 80)
    print("  VERIFICATION & DATASET SUMMARY")
    print("=" * 80)
    print(f"• ground_truth_annotator1.csv: {len(df_annotator1)} rows, Columns: {list(df_annotator1.columns)}")
    print(f"• ground_truth_annotator2.csv: {len(df_annotator2)} rows, Columns: {list(df_annotator2.columns)}")
    print(f"• disagreements_adjudication.csv: {len(df_disagreements)} rows, Columns: {list(df_disagreements.columns)}")
    print(f"• ground_truth_final.csv: {len(df_final)} rows, Columns: {list(df_final.columns)}")
    print("-" * 80)
    print(f"Ground Truth Label Distribution (Overall {len(df_final)} pairs):")
    print(df_final["ground_truth_label"].value_counts().to_string())
    print("-" * 80)
    
    # Check Kappa across all 603
    k_all = cohen_kappa_score(df_final["annotator_1_label"], df_final["annotator_2_label"], weights="quadratic", labels=[0, 1, 2])
    print(f"Overall Quadratic Weighted Kappa (N=603): {k_all:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
