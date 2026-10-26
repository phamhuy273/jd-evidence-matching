"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Module: STATISTICAL POWER & MINIMUM DETECTABLE EFFECT (MDE) ANALYSIS
Governed by: Rules R1-R46 (Exploratory Addendum for SANER 2027 ERA Review)
=============================================================================
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def calculate_mde(sd_diff: float, n: int = 25, alpha: float = 0.05, power: float = 0.80) -> float:
    """
    Compute Minimum Detectable Effect size for a two-sided paired t-test
    using non-central t-distribution approximation.
    """
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # Approximate normal quantile for power beta
    z_beta = stats.norm.ppf(power)
    # Standard formula for paired MDE: (t_crit + z_beta) * (sd_diff / sqrt(n))
    mde = (t_crit + z_beta) * (sd_diff / np.sqrt(n))
    return float(mde)


def run_power_analysis():
    pq_path = Path("dataset/benchmark_results/per_query_evaluation.csv")
    if not pq_path.exists():
        raise FileNotFoundError(f"Missing per_query_evaluation.csv at {pq_path}")

    df = pd.read_csv(pq_path)
    piv = df.pivot(index="jd_id", columns="config", values="ndcg_10")

    comparisons = [
        ("Comp 1", "rerank_ast_with_header", "rerank_line_no_header"),
        ("Comp 2", "rerank_ast_no_header", "rerank_line_no_header"),
        ("Comp 3", "rerank_ast_with_header", "rerank_ast_no_header"),
        ("Comp 4", "rerank_line_with_header", "rerank_line_no_header"),
        ("Comp 5", "rerank_ast_with_header", "rerank_line_with_header"),
    ]

    results = {}
    lines = []
    lines.append("=" * 80)
    lines.append("  STATISTICAL POWER & MINIMUM DETECTABLE EFFECT (MDE) REPORT")
    lines.append("  Sample size N = 25 JDs, Paired t-test approximation, Power = 80%")
    lines.append("=" * 80)

    mde_05_list = []
    mde_0125_list = []

    for comp_id, sys_a, sys_b in comparisons:
        diff = piv[sys_a] - piv[sys_b]
        mean_diff = float(diff.mean())
        sd_diff = float(diff.std(ddof=1))
        dz = float(mean_diff / sd_diff) if sd_diff > 0 else 0.0

        mde_alpha_05 = calculate_mde(sd_diff, n=len(diff), alpha=0.05, power=0.80)
        mde_alpha_0125 = calculate_mde(sd_diff, n=len(diff), alpha=0.0125, power=0.80)

        mde_05_list.append(mde_alpha_05)
        mde_0125_list.append(mde_alpha_0125)

        results[comp_id] = {
            "comparison": f"{sys_a} vs. {sys_b}",
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "cohens_d_z": dz,
            "mde_alpha_0.05": mde_alpha_05,
            "mde_alpha_0.0125": mde_alpha_0125,
        }

        lines.append(f"{comp_id}: {sys_a} vs. {sys_b}")
        lines.append(f"  Mean diff     : {mean_diff:+.4f}")
        lines.append(f"  SD diff       : {sd_diff:.4f}")
        lines.append(f"  Cohen's d_z   : {dz:+.4f}")
        lines.append(f"  MDE (a=0.0500): {mde_alpha_05:.4f}")
        lines.append(f"  MDE (a=0.0125): {mde_alpha_0125:.4f}")
        lines.append("-" * 80)

    overall_summary = {
        "min_mde_0.05": float(min(mde_05_list)),
        "max_mde_0.0125": float(max(mde_0125_list)),
        "mde_range_str": f"{min(mde_05_list):.3f} to {max(mde_0125_list):.3f}",
    }
    results["_summary"] = overall_summary

    lines.append(f"OVERALL MDE RANGE: {overall_summary['mde_range_str']} NDCG@10")
    lines.append("=" * 80)

    out_json = Path("dataset/benchmark_results/power_analysis.json")
    out_txt = Path("dataset/benchmark_results/power_analysis.txt")
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {out_json} and {out_txt}")
    print(f"MDE Range: {overall_summary['mde_range_str']}")
    return results


if __name__ == "__main__":
    run_power_analysis()
