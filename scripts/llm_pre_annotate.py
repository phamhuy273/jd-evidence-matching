#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/llm_pre_annotate.py - Batch LLM Pre-Annotation & Streamlit Generator

Applies Gemini 3.5 Flash Lite in batches (10 items/batch) to pre-grade candidate code chunks
strictly following annotation_guidelines.md (Levels 0, 1, 2).
Exports datasets in the format required by C:\\Users\\win 11\\Desktop\\streamlit-labeling\\app_label.py
for human review and adjudication.
"""

import os
import sys
import json
import time
import csv
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Reconfigure stdout for UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
TO_LABEL_FILE = DATASET_DIR / "to_label.csv"
CACHE_FILE = DATASET_DIR / "gemini_annotations_cache.json"

STREAMLIT_DIR = Path(r"C:\Users\win 11\Desktop\streamlit-labeling")
STREAMLIT_HUY = STREAMLIT_DIR / "ground_truth_huy_raw.csv"
STREAMLIT_AN = STREAMLIT_DIR / "ground_truth_an_raw.csv"

# Load Gemini API Key
load_dotenv(ROOT_DIR / ".env")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()

import google.generativeai as genai
genai.configure(api_key=GEMINI_KEY)
MODEL_NAME = "models/gemini-3.5-flash-lite"
model = genai.GenerativeModel(
    MODEL_NAME,
    generation_config={"response_mime_type": "application/json"}
)

BATCH_SIZE = 10

SYSTEM_PROMPT = """You are a Senior Software Engineering Technical Auditor evaluating candidate source code against Job Descriptions (JDs) for an IEEE SANER 2027 empirical study.

Evaluate whether each code snippet provides valid, observable evidence that the candidate possesses the mandatory technical skills required by the JD.

Relevance Scale (Strictly follow these operational criteria):
- Level 0 (Irrelevant / Boilerplate / Insufficient):
  * Code belongs to an unrelated technology or language.
  * Trivial boilerplate with zero business logic: empty class/interface, standard imports, simple getters/setters, default UI toasts (e.g. WelcomeToast, static Carousel wrapper).
  * 50-line window cutting across only imports or class headers without any method logic.

- Level 1 (Partially Relevant / Peripheral Evidence):
  * Peripheral or indirect evidence: basic configuration bean (e.g. basic Swagger/OpenAPI setup), simple CRUD delegation without business logic (e.g. return repo.findById(id)), standard auditing listener (e.g. getCurrentAuditor).
  * React component rendering basic JSX without complex state, custom hooks, or TypeScript typing.
  * Code snippet was cut halfway so that the core logic is severed/missing, leaving only peripheral traces.

- Level 2 (Highly Relevant / Strong Evidence):
  * Direct, non-trivial evidence of mandatory competencies:
    - Spring Boot: In-depth SecurityFilterChain with JWT lambda DSL, authorization filters, role-based endpoint access, @RestController with @PreAuthorize and DTO validation, complex transactional services (@Transactional).
    - React/TypeScript: Custom state hooks (useAuth, useFetch, useDebounce), state management reducers (Redux Toolkit createSlice/asyncThunk, Zustand), typed forms with Zod/Yup validation, API consumers with error boundaries.
  * CRITICAL: If a code snippet was cut at line 50, but the visible portion ALREADY contains complete core evidence of the skill, award Level 2! Grade based on the observable evidence in the snippet.

Return a valid JSON array of objects, one for each item in the batch:
[
  {
    "item_id": "<ITEM_XXX>",
    "label": 0 or 1 or 2,
    "rationale": "<Concise 1-2 sentence technical explanation explaining why the code satisfies or fails the requirement>"
  },
  ...
]
"""


def evaluate_batch(batch_items: list) -> list:
    """Invokes Gemini 3.5 Flash Lite on a batch of items with retry logic."""
    payload = []
    for it in batch_items:
        payload.append({
            "item_id": it["item_id"],
            "jd_title": it["jd_title"],
            "jd_level": it["jd_level"],
            "jd_domain": it["jd_domain"],
            "mandatory_skills": it["jd_mandatory_skills"],
            "file_path": it["file_path"],
            "lines": f"{it['start_line']}..{it['end_line']}",
            "code": it["code_content"]
        })

    prompt = SYSTEM_PROMPT + "\n\nItems to evaluate:\n" + json.dumps(payload, ensure_ascii=False)

    for attempt in range(5):
        try:
            response = model.generate_content(prompt)
            data = json.loads(response.text.strip())
            if isinstance(data, list) and len(data) > 0:
                result_map = {}
                for obj in data:
                    iid = obj.get("item_id")
                    lbl = int(obj.get("label", 0))
                    if lbl not in [0, 1, 2]:
                        lbl = 0
                    rat = str(obj.get("rationale", "")).strip()
                    result_map[iid] = {"label": lbl, "rationale": rat}
                return result_map
            else:
                raise ValueError(f"Expected JSON list, got: {type(data)}")
        except Exception as e:
            wait_time = 3.0 * (attempt + 1)
            print(f"    [Retry {attempt+1}/5] API error: {e}. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

    # Fallback if all 5 retries fail
    return {it["item_id"]: {"label": 0, "rationale": "API Batch Timeout"} for it in batch_items}


def main():
    print("=" * 80)
    print("PHASE 4: BATCH LLM PRE-ANNOTATION (GEMINI 3.5 FLASH LITE) & STREAMLIT EXPORT")
    print("=" * 80)

    # 1. Load to_label.csv
    if not TO_LABEL_FILE.exists():
        print(f"Error: {TO_LABEL_FILE} does not exist!")
        sys.exit(1)

    with open(TO_LABEL_FILE, "r", encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    print(f"Loaded {len(items)} items from {TO_LABEL_FILE.name}")

    # 2. Load or clean cache
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                raw_cache = json.load(f)
            # Retain only valid evaluations without errors
            for k, v in raw_cache.items():
                if "API Evaluation Error" not in v.get("rationale", "") and "API Batch Timeout" not in v.get("rationale", ""):
                    cache[k] = v
            print(f"Loaded {len(cache)} valid evaluations from cache.")
        except Exception:
            cache = {}

    # 3. Create batches of items needing evaluation
    to_process = [it for it in items if it["item_id"] not in cache]
    print(f"Items to evaluate with Gemini 3.5 Flash Lite: {len(to_process)}")

    if to_process:
        batches = [to_process[i:i + BATCH_SIZE] for i in range(0, len(to_process), BATCH_SIZE)]
        print(f"Split into {len(batches)} batches (batch size {BATCH_SIZE})")

        for b_idx, batch in enumerate(batches, start=1):
            print(f"  Evaluating Batch {b_idx}/{len(batches)} ({len(batch)} items: {batch[0]['item_id']}..{batch[-1]['item_id']})...")
            batch_results = evaluate_batch(batch)
            for it in batch:
                iid = it["item_id"]
                if iid in batch_results:
                    cache[iid] = batch_results[iid]
                else:
                    cache[iid] = {"label": 0, "rationale": "Missing from batch response"}

            # Save progress after each batch
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)

            time.sleep(2.0)  # Safe delay between batches

        print(f"Completed all batches! Total cached evaluations: {len(cache)}")

    # 4. Construct Streamlit-ready datasets
    label_distribution = {0: 0, 1: 0, 2: 0}
    streamlit_rows = []

    for it in items:
        eval_res = cache.get(it["item_id"], {"label": 0, "rationale": ""})
        lbl = eval_res["label"]
        rsn = eval_res["rationale"]
        label_distribution[lbl] = label_distribution.get(lbl, 0) + 1

        row = {
            "pair_id": it["item_id"],
            "jd_id": it["jd_id"],
            "jd_title": it["jd_title"],
            "jd_level": it["jd_level"],
            "jd_domain": it["jd_domain"],
            "jd_mandatory_skills": it["jd_mandatory_skills"],
            "repo_name": it["repo_name"],
            "file_path": it["file_path"],
            "start_line": it["start_line"],
            "end_line": it["end_line"],
            "context_header": f"File: {it['file_path']} | Lines {it['start_line']}..{it['end_line']}",
            "chunk_content": it["code_content"],
            "gemini_suggested_label": lbl,
            "gemini_reason": rsn,
            "human_label": lbl,          # Pre-filled with LLM suggestion for human review
            "human_note": rsn            # Pre-filled with LLM rationale for human review
        }
        streamlit_rows.append(row)

    print("\nPre-annotation Label Distribution across 353 items:")
    for l in [0, 1, 2]:
        print(f"  Level {l}: {label_distribution[l]} items ({label_distribution[l]/len(items)*100:.1f}%)")

    # 5. Export to Streamlit and dataset locations
    fieldnames = list(streamlit_rows[0].keys())

    # Export to dataset/ directory
    master_huy_dataset = DATASET_DIR / "ground_truth_huy_raw.csv"
    master_an_dataset = DATASET_DIR / "ground_truth_an_raw.csv"

    for dest in [master_huy_dataset, master_an_dataset]:
        with open(dest, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(streamlit_rows)
        print(f"Exported: {dest}")

    # Export to C:\Users\win 11\Desktop\streamlit-labeling
    if STREAMLIT_DIR.exists():
        for dest in [STREAMLIT_HUY, STREAMLIT_AN]:
            with open(dest, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(streamlit_rows)
            print(f"Exported to Streamlit App: {dest}")
    else:
        print(f"Warning: Streamlit directory {STREAMLIT_DIR} not found.")

    print("\n" + "=" * 80)
    print("LLM Pre-Annotation & Streamlit Dataset Export Complete!")
    print(f"Total items exported: {len(streamlit_rows)}")
    print("You can now run 'streamlit run app_label.py' on Desktop to review/adjust labels.")
    print("=" * 80)


if __name__ == "__main__":
    main()
