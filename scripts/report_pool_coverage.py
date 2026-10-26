"""scripts/report_pool_coverage.py - Analyzes pool coverage and judged@k rates across all 12 configurations.

Calculates exact top-3 and top-10 judged coverage from retrieval_runs/ and ground_truth_final.csv.
Outputs results to dataset/benchmark_results/pool_coverage_report.csv.
"""

import json
import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
RUNS_DIR = DATASET_DIR / "retrieval_runs"
GT_FILE = DATASET_DIR / "ground_truth_final.csv"
OUTPUT_CSV = DATASET_DIR / "benchmark_results" / "pool_coverage_report.csv"

CONFIGS = [
    "bm25_line_no_header",
    "bm25_line_with_header",
    "bm25_ast_no_header",
    "bm25_ast_with_header",
    "dense_line_no_header",
    "dense_line_with_header",
    "dense_ast_no_header",
    "dense_ast_with_header",
    "rerank_line_no_header",
    "rerank_line_with_header",
    "rerank_ast_no_header",
    "rerank_ast_with_header",
]


def load_ground_truth_keys():
    gt_set = set()
    with open(GT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["jd_id"], r["repo_name"], r["file_path"], int(r["start_line"]), int(r["end_line"]))
            gt_set.add(key)
    return gt_set


def compute_pool_coverage():
    gt_set = load_ground_truth_keys()
    results = []

    for cfg in CONFIGS:
        run_file = RUNS_DIR / f"{cfg}.json"
        with open(run_file, "r", encoding="utf-8") as f:
            run_data = json.load(f)

        top3_total = 0
        top3_unjudged = 0
        top5_total = 0
        top5_unjudged = 0
        top10_total = 0
        top10_unjudged = 0

        for qid, items in run_data.items():
            for i, it in enumerate(items[:10]):
                key = (qid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"]))
                is_judged = key in gt_set

                top10_total += 1
                if not is_judged:
                    top10_unjudged += 1

                if i < 5:
                    top5_total += 1
                    if not is_judged:
                        top5_unjudged += 1

                if i < 3:
                    top3_total += 1
                    if not is_judged:
                        top3_unjudged += 1

        j3 = (top3_total - top3_unjudged) / top3_total if top3_total > 0 else 0.0
        j5 = (top5_total - top5_unjudged) / top5_total if top5_total > 0 else 0.0
        j10 = (top10_total - top10_unjudged) / top10_total if top10_total > 0 else 0.0

        results.append({
            "configuration": cfg,
            "tier": "BM25" if "bm25" in cfg else ("Dense" if "dense" in cfg else "Rerank"),
            "top3_total": top3_total,
            "top3_unjudged": top3_unjudged,
            "judged_at_3": round(j3, 4),
            "top5_total": top5_total,
            "top5_unjudged": top5_unjudged,
            "judged_at_5": round(j5, 4),
            "top10_total": top10_total,
            "top10_unjudged": top10_unjudged,
            "judged_at_10": round(j10, 4),
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "configuration", "tier", "top3_total", "top3_unjudged", "judged_at_3",
        "top5_total", "top5_unjudged", "judged_at_5",
        "top10_total", "top10_unjudged", "judged_at_10"
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Generated pool coverage report at: {OUTPUT_CSV}")
    for r in results:
        print(f"{r['configuration']:25s} | Top-3 Judged: {r['judged_at_3']*100:5.1f}% ({r['top3_unjudged']:2d} unjudged) | Top-10 Judged: {r['judged_at_10']*100:5.1f}% ({r['top10_unjudged']:3d} unjudged)")


if __name__ == "__main__":
    compute_pool_coverage()
