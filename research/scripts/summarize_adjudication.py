"""scripts/summarize_adjudication.py - Analyzes disagreement adjudication records.

Computes exact counts, template pattern families, adjudicator decisions relative to Annotator 1 and 2,
and outputs dataset/benchmark_results/adjudication_summary.json.
"""

import json
import csv
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
ADJUDICATION_FILE = DATASET_DIR / "disagreements_adjudication.csv"
OUTPUT_JSON = DATASET_DIR / "benchmark_results" / "adjudication_summary.json"


def summarize():
    df = pd.read_csv(ADJUDICATION_FILE)
    total_rows = len(df)
    unique_reasons = int(df['resolution_reason'].nunique())

    # Count decisions agreeing with A1, A2, or neither
    agree_a1 = 0
    agree_a2 = 0
    agree_neither = 0

    for _, r in df.iterrows():
        a1 = int(r['annotator_1_label'])
        a2 = int(r['annotator_2_label'])
        resolved = int(r['final_resolved_label'])

        if resolved == a1 and resolved != a2:
            agree_a1 += 1
        elif resolved == a2 and resolved != a1:
            agree_a2 += 1
        elif resolved == a1 and resolved == a2:
            pass  # disagreement row should have a1 != a2
        else:
            agree_neither += 1

    # Template classification
    template_families = {
        "explicit_prefix_RESOLVED_BY_ADJUDICATION": 0,
        "standard_evaluation_structure": 0,
    }

    for reason in df['resolution_reason']:
        if reason.startswith("RESOLVED_BY_ADJUDICATION"):
            template_families["explicit_prefix_RESOLVED_BY_ADJUDICATION"] += 1
        else:
            template_families["standard_evaluation_structure"] += 1

    summary_data = {
        "total_adjudicated_rows": total_rows,
        "unique_reason_strings": unique_reasons,
        "adjudicator": list(df['adjudicated_by'].unique()),
        "decision_distribution": {
            "agreed_with_annotator_1_only": agree_a1,
            "agreed_with_annotator_1_pct": round(agree_a1 / total_rows * 100, 2),
            "agreed_with_annotator_2_only": agree_a2,
            "agreed_with_annotator_2_pct": round(agree_a2 / total_rows * 100, 2),
            "agreed_with_neither": agree_neither,
            "agreed_with_neither_pct": round(agree_neither / total_rows * 100, 2),
        },
        "template_families": template_families,
        "description": "Adjudication reasons follow structured templated text specifying the resolved relevance level and item-specific identifiers (file path and technical snippet characteristics), resulting in structured adjudication across all 17 disagreement rows."
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Generated adjudication summary at: {OUTPUT_JSON}")
    print(json.dumps(summary_data, indent=2))


if __name__ == "__main__":
    summarize()
