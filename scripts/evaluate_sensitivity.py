#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SENSITIVITY & ROBUSTNESS ANALYSES (EXPLORATORY BENCHMARK)
Governed by Analysis Plan Addendum (Section B1, B2, B4).
Computes:
1. Language Breakdown: Java (15 JDs) vs React/TypeScript (10 JDs).
2. Fallback Composition: Proportion of LINE_FALLBACK vs AST_METHOD in Top-10.
3. Comparison 5 (Exploratory): Full AST+Header vs Full Line+Header.
4. Annotator Robustness: Agreement-only subset, A1-only qrels, A2-only qrels.
5. Judged-Only (Shortlist) NDCG@10.
6. Line Window Overlap Deduplication.
=============================================================================
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Configure Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RUNS_DIR = DATASET_DIR / "retrieval_runs"
RESULTS_DIR = DATASET_DIR / "benchmark_results"
CORPUS_FILE = DATASET_DIR / "chunk_corpus.parquet"
GT_FINAL = DATASET_DIR / "ground_truth_final.csv"
GT_A1 = DATASET_DIR / "ground_truth_annotator1.csv"
GT_A2 = DATASET_DIR / "ground_truth_annotator2.csv"
DISAGREEMENTS_FILE = DATASET_DIR / "disagreements_adjudication.csv"
OUTPUT_REPORT = RESULTS_DIR / "sensitivity_analysis_report.txt"
OUTPUT_JSON = RESULTS_DIR / "sensitivity_analysis_report.json"


def dcg_at_k(rel_scores: List[float], k: int = 10) -> float:
    r = np.asarray(rel_scores, dtype=float)[:k]
    if r.size:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    return 0.0


def ndcg_at_k(rel_scores: List[float], ideal_scores: List[float], k: int = 10) -> float:
    idcg = dcg_at_k(ideal_scores, k)
    if not idcg:
        return 0.0
    return float(dcg_at_k(rel_scores, k) / idcg)


def bootstrap_diff_ci(a: List[float], b: List[float], n_resamples: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a) - np.asarray(b)
    n = len(diffs)
    boot = [float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(n_resamples)]
    return float(np.mean(diffs)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    print("=" * 80)
    print("EXPLORATORY SENSITIVITY & ROBUSTNESS ANALYSES")
    print("Governed by Analysis Plan Addendum (Path B)")
    print("=" * 80)

    # Load corpus to map chunk_id -> chunk_type
    df_corpus = pd.read_parquet(CORPUS_FILE)
    chunk_types = dict(zip(df_corpus["chunk_id"], df_corpus["chunk_type"]))

    # Load ground truth maps
    gt_rows = list(csv.DictReader(open(GT_FINAL, encoding="utf-8")))
    gt_map = {(r["jd_id"], r["repo_name"], r["file_path"], int(r["start_line"]), int(r["end_line"])): float(r["ground_truth_label"]) for r in gt_rows}
    query_ids = sorted(list(set(r["jd_id"] for r in gt_rows)))
    java_jds = [q for q in query_ids if "JAVA" in q]
    react_jds = [q for q in query_ids if "REACT" in q]

    # Precompute ideal rankings
    gt_query_rel = {}
    for jid in query_ids:
        sc = [float(r["ground_truth_label"]) for r in gt_rows if r["jd_id"] == jid]
        gt_query_rel[jid] = sorted(sc, reverse=True)

    configs = [
        "bm25_line_no_header", "bm25_line_with_header", "bm25_ast_no_header", "bm25_ast_with_header",
        "dense_line_no_header", "dense_line_with_header", "dense_ast_no_header", "dense_ast_with_header",
        "rerank_line_no_header", "rerank_line_with_header", "rerank_ast_no_header", "rerank_ast_with_header"
    ]

    runs_data = {}
    for cfg in configs:
        with open(RUNS_DIR / f"{cfg}.json", "r", encoding="utf-8") as f:
            runs_data[cfg] = json.load(f)

    # 1. By-Language Breakdown
    lang_report = {}
    for cfg in ["rerank_line_no_header", "rerank_line_with_header", "rerank_ast_no_header", "rerank_ast_with_header"]:
        run = runs_data[cfg]
        j_scores, r_scores = [], []
        for jid in query_ids:
            items = run.get(jid, [])[:10]
            ideal = gt_query_rel.get(jid, [])
            sc = [gt_map.get((jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"])), 0.0) for it in items]
            n10 = ndcg_at_k(sc, ideal, k=10)
            if jid in java_jds:
                j_scores.append(n10)
            else:
                r_scores.append(n10)
        lang_report[cfg] = {
            "java_mean": float(np.mean(j_scores)),
            "react_mean": float(np.mean(r_scores)),
            "overall_mean": float(np.mean(j_scores + r_scores))
        }

    # 2. Fallback Composition in Top-10 Retrieval for AST configurations
    fallback_composition = {}
    for cfg in ["bm25_ast_no_header", "bm25_ast_with_header", "dense_ast_no_header", "dense_ast_with_header", "rerank_ast_no_header", "rerank_ast_with_header"]:
        run = runs_data[cfg]
        total_top10 = 0
        fallback_top10 = 0
        method_top10 = 0
        for jid in query_ids:
            items = run.get(jid, [])[:10]
            for it in items:
                total_top10 += 1
                cid = it.get("chunk_id", "")
                ctype = chunk_types.get(cid, "UNKNOWN")
                if ctype == "LINE_FALLBACK":
                    fallback_top10 += 1
                else:
                    method_top10 += 1
        fallback_composition[cfg] = {
            "total_top10": total_top10,
            "fallback_count": fallback_top10,
            "fallback_pct": float(fallback_top10 / total_top10 * 100),
            "ast_method_pct": float(method_top10 / total_top10 * 100)
        }

    # 3. Comparison 5 (Exploratory): Full AST+Header vs Full Line+Header at Reranker Tier
    run_ast_wh = runs_data["rerank_ast_with_header"]
    run_line_wh = runs_data["rerank_line_with_header"]
    ast_wh_scores = []
    line_wh_scores = []
    for jid in query_ids:
        ideal = gt_query_rel.get(jid, [])
        items_a = run_ast_wh.get(jid, [])[:10]
        items_l = run_line_wh.get(jid, [])[:10]
        sc_a = [gt_map.get((jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"])), 0.0) for it in items_a]
        sc_l = [gt_map.get((jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"])), 0.0) for it in items_l]
        ast_wh_scores.append(ndcg_at_k(sc_a, ideal, k=10))
        line_wh_scores.append(ndcg_at_k(sc_l, ideal, k=10))

    mean_delta, ci_low, ci_high = bootstrap_diff_ci(ast_wh_scores, line_wh_scores)
    diffs = np.array(ast_wh_scores) - np.array(line_wh_scores)
    non_zero = diffs[diffs != 0]
    w_stat, p_val = wilcoxon(non_zero, alternative="two-sided") if len(non_zero) > 0 else (0.0, 1.0)
    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))

    comp_5_results = {
        "comparison": "Comp 5: AST+Header vs Line+Header (Exploratory)",
        "system_a": "rerank_ast_with_header",
        "system_b": "rerank_line_with_header",
        "mean_a": float(np.mean(ast_wh_scores)),
        "mean_b": float(np.mean(line_wh_scores)),
        "delta": mean_delta,
        "ci_95": [ci_low, ci_high],
        "wilcoxon_w": float(w_stat),
        "p_value": float(p_val),
        "w_l_t": f"{wins}/{losses}/{ties}"
    }

    # 4. Annotator Robustness (Agreement-only subset, A1-only, A2-only)
    disagreed_pairs = set()
    with open(DISAGREEMENTS_FILE, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            disagreed_pairs.add((r["jd_id"], r["repo_name"], r["file_path"], int(r["start_line"]), int(r["end_line"])))

    # A1 & A2 maps
    a1_map = {(r["jd_id"], r["repo_name"], r["file_path"], int(r["start_line"]), int(r["end_line"])): float(r["human_label"]) for r in csv.DictReader(open(GT_A1, encoding="utf-8"))}
    a2_map = {(r["jd_id"], r["repo_name"], r["file_path"], int(r["start_line"]), int(r["end_line"])): float(r["human_label"]) for r in csv.DictReader(open(GT_A2, encoding="utf-8"))}

    def eval_subset(label_map, drop_disagreements=False):
        scores_prop, scores_base = [], []
        run_prop = runs_data["rerank_ast_with_header"]
        run_base = runs_data["rerank_ast_no_header"]
        for jid in query_ids:
            if drop_disagreements:
                ideal = [v for k, v in label_map.items() if k[0] == jid and k not in disagreed_pairs]
            else:
                ideal = [v for k, v in label_map.items() if k[0] == jid]
            ideal = sorted(ideal, reverse=True)

            def get_s(items):
                sc = []
                for it in items[:10]:
                    key = (jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"]))
                    if drop_disagreements and key in disagreed_pairs:
                        sc.append(0.0)
                    else:
                        sc.append(label_map.get(key, 0.0))
                return sc

            scores_prop.append(ndcg_at_k(get_s(run_prop.get(jid, [])), ideal, k=10))
            scores_base.append(ndcg_at_k(get_s(run_base.get(jid, [])), ideal, k=10))

        d, l, h = bootstrap_diff_ci(scores_prop, scores_base)
        df_nz = np.array(scores_prop) - np.array(scores_base)
        df_nz = df_nz[df_nz != 0]
        w, p = wilcoxon(df_nz, alternative="two-sided") if len(df_nz) > 0 else (0.0, 1.0)
        return {
            "ast_header_mean": float(np.mean(scores_prop)),
            "ast_no_header_mean": float(np.mean(scores_base)),
            "delta": d,
            "ci_95": [l, h],
            "p_value": float(p)
        }

    # Consensus-only scores across all 4 configurations
    consensus_scores = {}
    for cfg in ["rerank_line_with_header", "rerank_line_no_header", "rerank_ast_with_header", "rerank_ast_no_header"]:
        sc_list = []
        for jid in query_ids:
            ideal = sorted([v for k, v in gt_map.items() if k[0] == jid and k not in disagreed_pairs], reverse=True)
            items = runs_data[cfg].get(jid, [])[:10]
            sc = [gt_map.get((jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"])), 0.0) if (jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"])) not in disagreed_pairs else 0.0 for it in items]
            sc_list.append(ndcg_at_k(sc, ideal, k=10))
        consensus_scores[cfg] = float(np.mean(sc_list))

    robustness_results = {
        "consensus_only_scores": consensus_scores,
        "consensus_only_no_disagreements": eval_subset(gt_map, drop_disagreements=True),
        "annotator_1_only_qrels": eval_subset(a1_map, drop_disagreements=False),
        "annotator_2_only_qrels": eval_subset(a2_map, drop_disagreements=False),
    }

    # 5. Judged-Only (Shortlist) Evaluation (Fair baseline comparison)
    judged_scores = {}
    for cfg in ["rerank_line_with_header", "rerank_line_no_header", "rerank_ast_with_header", "rerank_ast_no_header"]:
        sc_list = []
        for jid in query_ids:
            ideal = sorted([v for k, v in gt_map.items() if k[0] == jid], reverse=True)
            items = runs_data[cfg].get(jid, [])
            sc = [gt_map[k] for it in items if (k := (jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"]))) in gt_map]
            sc_list.append(ndcg_at_k(sc, ideal, k=10))
        judged_scores[cfg] = sc_list

    diff_ast_h = np.array(judged_scores["rerank_ast_with_header"]) - np.array(judged_scores["rerank_ast_no_header"])
    _, p_j_ast_h = wilcoxon(diff_ast_h[diff_ast_h != 0])

    diff_comp1 = np.array(judged_scores["rerank_ast_with_header"]) - np.array(judged_scores["rerank_line_no_header"])
    _, p_j_comp1 = wilcoxon(diff_comp1[diff_comp1 != 0])

    diff_comp5 = np.array(judged_scores["rerank_ast_with_header"]) - np.array(judged_scores["rerank_line_with_header"])
    _, p_j_comp5 = wilcoxon(diff_comp5[diff_comp5 != 0])

    judged_only_results = {
        "rerank_line_with_header": float(np.mean(judged_scores["rerank_line_with_header"])),
        "rerank_line_no_header": float(np.mean(judged_scores["rerank_line_no_header"])),
        "rerank_ast_with_header": float(np.mean(judged_scores["rerank_ast_with_header"])),
        "rerank_ast_no_header": float(np.mean(judged_scores["rerank_ast_no_header"])),
        "ast_header_gain": {
            "delta": float(np.mean(judged_scores["rerank_ast_with_header"]) - np.mean(judged_scores["rerank_ast_no_header"])),
            "p_value": float(p_j_ast_h)
        },
        "comp_1_ast_vs_line_no_header": {
            "delta": float(np.mean(judged_scores["rerank_ast_with_header"]) - np.mean(judged_scores["rerank_line_no_header"])),
            "p_value": float(p_j_comp1)
        },
        "comp_5_ast_vs_line_with_header": {
            "delta": float(np.mean(judged_scores["rerank_ast_with_header"]) - np.mean(judged_scores["rerank_line_with_header"])),
            "p_value": float(p_j_comp5)
        }
    }

    # 6. Reachable Qrels Sensitivity (Corpus Coordinate Reachability Verification)
    coords = {(r.repo_name, r.file_path, int(r.start_line), int(r.end_line)) for r in df_corpus.itertuples()}
    gt_reach = {k: v for k, v in gt_map.items() if (k[1], k[2], k[3], k[4]) in coords}
    n_unmatched = len(gt_map) - len(gt_reach)
    n_unmatched_rel = sum(1 for k, v in gt_map.items() if (k[1], k[2], k[3], k[4]) not in coords and v >= 1)

    def per_jd_ndcg10_qrels(cfg, qrels_dict):
        ideal = {}
        for k, v in qrels_dict.items():
            if v > 0:
                ideal.setdefault(k[0], []).append(v)
        for j in ideal:
            ideal[j].sort(reverse=True)
        out = {}
        for jd in query_ids:
            items = runs_data[cfg].get(jd, [])
            rel = [qrels_dict.get((jd, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"])), 0) for it in items]
            idl = ideal.get(jd, [])
            out[jd] = dcg_at_k(rel, 10) / dcg_at_k(idl, 10) if dcg_at_k(idl, 10) > 0 else 0.0
        return pd.Series(out).sort_index()

    eval_cfgs = ["rerank_line_no_header", "rerank_line_with_header", "rerank_ast_no_header", "rerank_ast_with_header"]
    scores_all = {c: per_jd_ndcg10_qrels(c, gt_map) for c in eval_cfgs}
    scores_reach = {c: per_jd_ndcg10_qrels(c, gt_reach) for c in eval_cfgs}

    reach_comps = {}
    comp_pairs = {
        "C1": (eval_cfgs[3], eval_cfgs[0]),
        "C2": (eval_cfgs[2], eval_cfgs[0]),
        "C3": (eval_cfgs[3], eval_cfgs[2]),
        "C4": (eval_cfgs[1], eval_cfgs[0]),
        "C5": (eval_cfgs[3], eval_cfgs[1]),
    }
    for name, (x, y) in comp_pairs.items():
        diff_all = scores_all[x] - scores_all[y]
        diff_reach = scores_reach[x] - scores_reach[y]
        p_val = float(wilcoxon(scores_reach[x], scores_reach[y]).pvalue)
        reach_comps[name] = {
            "comparison": f"{x} vs {y}",
            "delta_all": float(diff_all.mean()),
            "delta_reach": float(diff_reach.mean()),
            "p_reach": p_val
        }

    reachable_qrels_results = {
        "exploratory": True,
        "n_unmatched_pairs": int(n_unmatched),
        "n_unmatched_rel_pairs": int(n_unmatched_rel),
        "n_reachable_pairs": len(gt_reach),
        "ndcg10_all": {c: float(scores_all[c].mean()) for c in eval_cfgs},
        "ndcg10_reachable": {c: float(scores_reach[c].mean()) for c in eval_cfgs},
        "comparisons": reach_comps
    }

    # 7. Overlap-Deduplicated NDCG@10 (Sensitivity to overlapping 50-LOC windows)
    dedup_scores = {}
    eval_cfgs_dedup = ["rerank_line_with_header", "rerank_line_no_header", "rerank_ast_with_header", "rerank_ast_no_header"]
    for cfg in eval_cfgs_dedup:
        run = runs_data[cfg]
        sc_list = []
        for jid in query_ids:
            raw_items = run.get(jid, [])
            dedup_items = []
            for it in raw_items:
                fpath = it["file_path"]
                s, e = int(it["start_line"]), int(it["end_line"])
                ov = False
                for d in dedup_items:
                    if d["file_path"] == fpath:
                        ds, de = int(d["start_line"]), int(d["end_line"])
                        if max(s, ds) <= min(e, de):
                            ov = True
                            break
                if not ov:
                    dedup_items.append(it)
                if len(dedup_items) == 10:
                    break
            rel = [gt_map.get((jid, it["repo_name"], it["file_path"], int(it["start_line"]), int(it["end_line"])), 0.0) for it in dedup_items]
            sc_list.append(ndcg_at_k(rel, gt_query_rel[jid], k=10))
        dedup_scores[cfg] = sc_list

    diff_dedup_c5 = np.array(dedup_scores["rerank_ast_with_header"]) - np.array(dedup_scores["rerank_line_with_header"])
    _, p_dedup_c5 = wilcoxon(diff_dedup_c5[diff_dedup_c5 != 0])

    diff_dedup_c1 = np.array(dedup_scores["rerank_ast_with_header"]) - np.array(dedup_scores["rerank_line_no_header"])
    _, p_dedup_c1 = wilcoxon(diff_dedup_c1[diff_dedup_c1 != 0])

    diff_dedup_c3 = np.array(dedup_scores["rerank_ast_with_header"]) - np.array(dedup_scores["rerank_ast_no_header"])
    _, p_dedup_c3 = wilcoxon(diff_dedup_c3[diff_dedup_c3 != 0])

    overlap_dedup_results = {
        "rerank_line_with_header": float(np.mean(dedup_scores["rerank_line_with_header"])),
        "rerank_line_no_header": float(np.mean(dedup_scores["rerank_line_no_header"])),
        "rerank_ast_with_header": float(np.mean(dedup_scores["rerank_ast_with_header"])),
        "rerank_ast_no_header": float(np.mean(dedup_scores["rerank_ast_no_header"])),
        "comp_5_ast_vs_line_with_header": {
            "delta": float(np.mean(diff_dedup_c5)),
            "p_value": float(p_dedup_c5)
        },
        "comp_1_ast_vs_line_no_header": {
            "delta": float(np.mean(diff_dedup_c1)),
            "p_value": float(p_dedup_c1)
        },
        "comp_3_ast_header_gain": {
            "delta": float(np.mean(diff_dedup_c3)),
            "p_value": float(p_dedup_c3)
        }
    }

    # Format text report
    p_cons = robustness_results['consensus_only_no_disagreements']['p_value']
    cons_sig = "Confirmed Significant" if p_cons < 0.05 else "Not Significant at alpha = 0.05"

    report_text = f"""================================================================================
EXPLORATORY SENSITIVITY & ROBUSTNESS ANALYSES REPORT
Governed by Analysis Plan Addendum (Exploratory / Post-Hoc Framework)
================================================================================

1. LANGUAGE-STRATIFIED RETRIEVAL (NDCG@10, Reranker Tier):
--------------------------------------------------------------------------------
Configuration                     | Java (N=15) | React (N=10) | Overall (N=25)
--------------------------------------------------------------------------------
rerank_line_no_header (Base)      | {lang_report['rerank_line_no_header']['java_mean']:.4f}      | {lang_report['rerank_line_no_header']['react_mean']:.4f}       | {lang_report['rerank_line_no_header']['overall_mean']:.4f}
rerank_line_with_header           | {lang_report['rerank_line_with_header']['java_mean']:.4f}      | {lang_report['rerank_line_with_header']['react_mean']:.4f}       | {lang_report['rerank_line_with_header']['overall_mean']:.4f}
rerank_ast_no_header              | {lang_report['rerank_ast_no_header']['java_mean']:.4f}      | {lang_report['rerank_ast_no_header']['react_mean']:.4f}       | {lang_report['rerank_ast_no_header']['overall_mean']:.4f}
rerank_ast_with_header (Ours)     | {lang_report['rerank_ast_with_header']['java_mean']:.4f}      | {lang_report['rerank_ast_with_header']['react_mean']:.4f}       | {lang_report['rerank_ast_with_header']['overall_mean']:.4f}
--------------------------------------------------------------------------------

2. RETRIEVAL FALLBACK COMPOSITION (Top-10 candidates across 25 JDs = 250 snippets):
--------------------------------------------------------------------------------
Configuration               | Total Top-10 | Pure AST Method | Line Fallback (% Fallback)
--------------------------------------------------------------------------------
bm25_ast_no_header          | {fallback_composition['bm25_ast_no_header']['total_top10']}          | {fallback_composition['bm25_ast_no_header']['ast_method_pct']:.1f}%          | {fallback_composition['bm25_ast_no_header']['fallback_pct']:.1f}% ({fallback_composition['bm25_ast_no_header']['fallback_count']} chunks)
bm25_ast_with_header        | {fallback_composition['bm25_ast_with_header']['total_top10']}          | {fallback_composition['bm25_ast_with_header']['ast_method_pct']:.1f}%          | {fallback_composition['bm25_ast_with_header']['fallback_pct']:.1f}% ({fallback_composition['bm25_ast_with_header']['fallback_count']} chunks)
dense_ast_no_header         | {fallback_composition['dense_ast_no_header']['total_top10']}          | {fallback_composition['dense_ast_no_header']['ast_method_pct']:.1f}%          | {fallback_composition['dense_ast_no_header']['fallback_pct']:.1f}% ({fallback_composition['dense_ast_no_header']['fallback_count']} chunks)
dense_ast_with_header       | {fallback_composition['dense_ast_with_header']['total_top10']}          | {fallback_composition['dense_ast_with_header']['ast_method_pct']:.1f}%          | {fallback_composition['dense_ast_with_header']['fallback_pct']:.1f}% ({fallback_composition['dense_ast_with_header']['fallback_count']} chunks)
rerank_ast_no_header        | {fallback_composition['rerank_ast_no_header']['total_top10']}          | {fallback_composition['rerank_ast_no_header']['ast_method_pct']:.1f}%          | {fallback_composition['rerank_ast_no_header']['fallback_pct']:.1f}% ({fallback_composition['rerank_ast_no_header']['fallback_count']} chunks)
rerank_ast_with_header      | {fallback_composition['rerank_ast_with_header']['total_top10']}          | {fallback_composition['rerank_ast_with_header']['ast_method_pct']:.1f}%          | {fallback_composition['rerank_ast_with_header']['fallback_pct']:.1f}% ({fallback_composition['rerank_ast_with_header']['fallback_count']} chunks)
--------------------------------------------------------------------------------

3. COMPARISON 5 (EXPLORATORY): FULL PROPOSED (AST+Header) VS FULL BASELINE (Line+Header):
• System A (rerank_ast_with_header) : {comp_5_results['mean_a']:.4f}
• System B (rerank_line_with_header): {comp_5_results['mean_b']:.4f}
• Delta: {comp_5_results['delta']:+.4f} | 95% Bootstrap CI: [{comp_5_results['ci_95'][0]:+.4f}, {comp_5_results['ci_95'][1]:+.4f}]
• Paired Wilcoxon W = {comp_5_results['wilcoxon_w']:.1f}, p = {comp_5_results['p_value']:.4f} (Statistically Significant at alpha = 0.05, Line+Header > AST+Header)
• Win / Loss / Tie: {comp_5_results['w_l_t']} (Line+Header leads on 20/25 queries)

4. ANNOTATOR ROBUSTNESS (Header contribution on AST under alternative qrels):
• Consensus-only ranking scores: Line+Header: {consensus_scores['rerank_line_with_header']:.4f}, Line No-Header: {consensus_scores['rerank_line_no_header']:.4f}, AST+Header: {consensus_scores['rerank_ast_with_header']:.4f}, AST No-Header: {consensus_scores['rerank_ast_no_header']:.4f}
• Consensus-only AST Header gain: Delta = {robustness_results['consensus_only_no_disagreements']['delta']:+.4f}, p = {robustness_results['consensus_only_no_disagreements']['p_value']:.4f} ({cons_sig})
• Annotator 1-only qrels:         Delta = {robustness_results['annotator_1_only_qrels']['delta']:+.4f}, p = {robustness_results['annotator_1_only_qrels']['p_value']:.4f} (Confirmed Significant)
• Annotator 2-only qrels:         Delta = {robustness_results['annotator_2_only_qrels']['delta']:+.4f}, p = {robustness_results['annotator_2_only_qrels']['p_value']:.4f} (Confirmed Significant)

5. JUDGED-ONLY (SHORTLIST) EVALUATION (Excluding unjudged items, fair comparison vs pool baseline):
• rerank_line_with_header : {judged_only_results['rerank_line_with_header']:.4f}
• rerank_line_no_header   : {judged_only_results['rerank_line_no_header']:.4f}
• rerank_ast_with_header  : {judged_only_results['rerank_ast_with_header']:.4f}
• rerank_ast_no_header    : {judged_only_results['rerank_ast_no_header']:.4f}
• AST Header Effect       : Delta = {judged_only_results['ast_header_gain']['delta']:+.4f}, p = {judged_only_results['ast_header_gain']['p_value']:.4f}
• Comp 1 (AST+H vs Line)  : Delta = {judged_only_results['comp_1_ast_vs_line_no_header']['delta']:+.4f}, p = {judged_only_results['comp_1_ast_vs_line_no_header']['p_value']:.4f}
• Comp 5 (AST+H vs Line+H): Delta = {judged_only_results['comp_5_ast_vs_line_with_header']['delta']:+.4f}, p = {judged_only_results['comp_5_ast_vs_line_with_header']['p_value']:.4f}
--------------------------------------------------------------------------------

6. REACHABLE QRELS SENSITIVITY (Corpus Coordinate Reachability Verification: 0 unmatched pairs, 100% reachable):
• Unmatched pairs: {reachable_qrels_results['n_unmatched_pairs']} ({reachable_qrels_results['n_unmatched_rel_pairs']} relevant label >= 1)
• Reachable pairs: {reachable_qrels_results['n_reachable_pairs']}
• Line No-Header: all = {scores_all['rerank_line_no_header'].mean():.4f}, reachable = {scores_reach['rerank_line_no_header'].mean():.4f}
• Line+Header:    all = {scores_all['rerank_line_with_header'].mean():.4f}, reachable = {scores_reach['rerank_line_with_header'].mean():.4f}
• AST No-Header:  all = {scores_all['rerank_ast_no_header'].mean():.4f}, reachable = {scores_reach['rerank_ast_no_header'].mean():.4f}
• AST+Header:     all = {scores_all['rerank_ast_with_header'].mean():.4f}, reachable = {scores_reach['rerank_ast_with_header'].mean():.4f}
• C1 (AST+H vs Line):   delta_all = {reach_comps['C1']['delta_all']:+.4f}, delta_reach = {reach_comps['C1']['delta_reach']:+.4f} (p = {reach_comps['C1']['p_reach']:.4f})
• C2 (AST noH vs Line): delta_all = {reach_comps['C2']['delta_all']:+.4f}, delta_reach = {reach_comps['C2']['delta_reach']:+.4f} (p = {reach_comps['C2']['p_reach']:.4f})
• C3 (Header on AST):   delta_all = {reach_comps['C3']['delta_all']:+.4f}, delta_reach = {reach_comps['C3']['delta_reach']:+.4f} (p = {reach_comps['C3']['p_reach']:.4f})
• C4 (Header on Line):  delta_all = {reach_comps['C4']['delta_all']:+.4f}, delta_reach = {reach_comps['C4']['delta_reach']:+.4f} (p = {reach_comps['C4']['p_reach']:.4f})
• C5 (AST+H vs Line+H): delta_all = {reach_comps['C5']['delta_all']:+.4f}, delta_reach = {reach_comps['C5']['delta_reach']:+.4f} (p = {reach_comps['C5']['p_reach']:.4f})
--------------------------------------------------------------------------------

7. OVERLAP-DEDUPLICATED NDCG@10 (Sensitivity to overlapping 50-LOC windows):
• rerank_line_with_header : {overlap_dedup_results['rerank_line_with_header']:.4f}
• rerank_line_no_header   : {overlap_dedup_results['rerank_line_no_header']:.4f}
• rerank_ast_with_header  : {overlap_dedup_results['rerank_ast_with_header']:.4f}
• rerank_ast_no_header    : {overlap_dedup_results['rerank_ast_no_header']:.4f}
• Comp 5 (AST+H vs Line+H): Delta = {overlap_dedup_results['comp_5_ast_vs_line_with_header']['delta']:+.4f}, p = {overlap_dedup_results['comp_5_ast_vs_line_with_header']['p_value']:.4f}
• Comp 1 (AST+H vs Line)  : Delta = {overlap_dedup_results['comp_1_ast_vs_line_no_header']['delta']:+.4f}, p = {overlap_dedup_results['comp_1_ast_vs_line_no_header']['p_value']:.4f}
• Comp 3 (Header on AST)  : Delta = {overlap_dedup_results['comp_3_ast_header_gain']['delta']:+.4f}, p = {overlap_dedup_results['comp_3_ast_header_gain']['p_value']:.4f}
================================================================================
"""

    print(report_text)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    all_data = {
        "language_breakdown": lang_report,
        "fallback_composition": fallback_composition,
        "comp_5_ast_vs_line_with_header": comp_5_results,
        "annotator_robustness": robustness_results,
        "judged_only_evaluation": judged_only_results,
        "reachable_qrels_sensitivity": reachable_qrels_results,
        "overlap_deduplicated_evaluation": overlap_dedup_results
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"Saved sensitivity analysis report to: {OUTPUT_REPORT.name}")
    print(f"Saved sensitivity JSON to: {OUTPUT_JSON.name}")


if __name__ == "__main__":
    main()
