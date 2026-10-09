#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CANDIDATE SKILL MATCHING VIA SOURCE CODE - RAG REPLICATION PACKAGE
Task: PRE-ANNOTATION & CORPUS DATASET GENERATION PIPELINE
Granularity: Tree-sitter AST Method-level Chunking with Hierarchical Context Headers
Target: IEEE SANER 2027 (ERA Track - CORE A) & Double-Anonymous Peer Review
=============================================================================
"""

import os
import sys
import json
import time
import re
import random
import shutil
import stat
import tempfile
import argparse
import subprocess
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Configure UTF-8 encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
JD_DIR = DATASET_DIR / "jd_raw"
REPOS_LIST_DIR = DATASET_DIR / "repositories_list"
AI_ENGINE_DIR = PROJECT_ROOT / "ai-engine"
EXTRACTED_JD_SKILLS_FILE = DATASET_DIR / "extracted_jd_skills.json"
CHUNKS_POOL_FILE = DATASET_DIR / "extracted_chunks_pool_50repos.json"
ANNOTATOR1_LABELED_FILE = DATASET_DIR / "ground_truth_final.csv"
OUTPUT_MASTER_FILE = DATASET_DIR / "ground_truth_500_master.csv"

# Load Tree-sitter Parser from ai-engine
sys.path.append(str(AI_ENGINE_DIR))
try:
    from parser.tree_sitter_loader import ast_loader
    from tree_sitter import Language, Parser
    import tree_sitter_typescript
    tsx_lang = Language(tree_sitter_typescript.language_tsx())
    ast_loader.parsers["tsx"] = Parser(tsx_lang)
    ast_loader.parsers["typescript"] = Parser(tsx_lang)
    print("Loaded Tree-sitter AST Parser (Java, TypeScript, TSX).")
except Exception as e:
    print(f"Warning: Tree-sitter loader exception ({e}). Using Fallback Line-based chunking.")

# Load environment configuration
load_dotenv(PROJECT_ROOT / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

gemini_model = None
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-flash-lite-latest')
        print("Connected to Google Gemini API for LLM-as-a-Judge pre-annotation.")
    except Exception as e:
        print(f"Notice: Gemini API not initialized ({e}). Using heuristic fallback.")


def force_remove_tree(dir_path):
    """Safely delete temporary clone directories on Windows."""
    def on_rm_error(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, onerror=on_rm_error)


IGNORE_DIRS = {'.git', 'node_modules', 'target', 'build', 'dist', '.next', '.idea', 'test', 'tests', 'vendor'}
SECURITY_KEYWORDS = {'security', 'jwt', 'token', 'authserver', 'auditing', 'oauth', 'filter', 'sso'}


def load_all_jds():
    """Load the 25 curated JDs from extracted_jd_skills.json."""
    if EXTRACTED_JD_SKILLS_FILE.exists():
        with open(EXTRACTED_JD_SKILLS_FILE, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        jds = []
        for jd_id, data in cached_data.items():
            raw_mand = data.get("mandatory_skills", [])
            mand_skills = [s["skill_name"] if isinstance(s, dict) else str(s) for s in raw_mand]
            category = data.get("category", "JAVA" if "JAVA" in jd_id else "REACT")
            jds.append({
                "jd_id": jd_id,
                "category": category,
                "title": data.get("title", jd_id),
                "level": data.get("level", "Fresher / Junior"),
                "domain": data.get("domain", "Software Engineering"),
                "mandatory_skills": mand_skills
            })
        print(f"Loaded {len(jds)} Job Descriptions from cache ({EXTRACTED_JD_SKILLS_FILE.name}).")
        return jds
    raise FileNotFoundError(f"Missing extracted JD skills file at {EXTRACTED_JD_SKILLS_FILE.name}")


def extract_chunks_from_repo(repo_url, repo_name, language, max_chunks_per_repo=12):
    """
    Shallow clone repository and extract method-level chunks with domain diversification:
    - Cap at 1 chunk per file to maximize codebase breadth.
    - Avoid over-indexing on repetitive boilerplate files.
    """
    tmp_dir = tempfile.mkdtemp(prefix="repo_clone_")
    chunks = []
    try:
        clone_cmd = ["git", "clone", "--depth", "1", repo_url, tmp_dir]
        res = subprocess.run(clone_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
        if res.returncode != 0:
            print(f"    Warning: Clone failed for {repo_name}")
            return chunks

        exts = ('.java',) if language == 'Java' else ('.ts', '.tsx', '.js')
        parser_lang = "java" if language == "Java" else "tsx"

        found_files = []
        for root, dirs, files in os.walk(tmp_dir):
            dirs[:] = [d for d in dirs if d.lower() not in IGNORE_DIRS]
            for file in files:
                if file.endswith(exts) and not file.lower().startswith('test'):
                    full_p = Path(root) / file
                    try:
                        rel_p = str(full_p.relative_to(tmp_dir)).replace("\\", "/")
                    except ValueError:
                        rel_p = file
                    found_files.append((full_p, rel_p))

        def file_priority(item):
            p = item[1].lower()
            if any(sec in p for sec in SECURITY_KEYWORDS):
                return 10
            if any(k in p for k in ['service', 'controller', 'repository', 'hook', 'component', 'page', 'store', 'slice', 'api', 'handler']):
                return 0
            if any(k in p for k in ['model', 'dto', 'entity', 'util', 'context']):
                return 1
            return 2

        found_files.sort(key=file_priority)

        used_files = set()
        for full_p, rel_p in found_files:
            if len(chunks) >= max_chunks_per_repo:
                break
            if rel_p in used_files:
                continue

            try:
                with open(full_p, 'r', encoding='utf-8', errors='ignore') as f:
                    code_text = f.read()

                file_chunks = ast_loader.parse_and_chunk(rel_p, code_text, parser_lang)

                for c in file_chunks:
                    c_content = c.get('chunk_content', '').strip()
                    num_lines = len(c_content.splitlines())
                    
                    if 6 <= num_lines <= 85:
                        chunks.append({
                            "repo_name": repo_name,
                            "repo_url": repo_url,
                            "language": language,
                            "file_path": c['file_path'],
                            "class_name": c.get('class_name') or '',
                            "method_name": c.get('method_name') or '',
                            "start_line": c.get('start_line'),
                            "end_line": c.get('end_line'),
                            "context_header": c.get('context_header') or '',
                            "chunk_content": c_content
                        })
                        used_files.add(rel_p)
                        break
            except Exception:
                continue

    except Exception as ex:
        print(f"    Warning: Skipping {repo_name}: {ex}")
    finally:
        force_remove_tree(tmp_dir)

    return chunks


def build_or_load_chunk_pool(force_reextract=False):
    """Scan and create chunk pool across 50 repositories."""
    if CHUNKS_POOL_FILE.exists() and not force_reextract:
        print(f"Found existing chunk pool at {CHUNKS_POOL_FILE.name}. Loading from cache...")
        with open(CHUNKS_POOL_FILE, 'r', encoding='utf-8') as f:
            pool = json.load(f)
        if len(pool) >= 200:
            print(f"Loaded {len(pool)} code chunks from cache.")
            return pool

    print("\nScanning and extracting AST chunks across 50 repositories...")
    java_df = pd.read_csv(REPOS_LIST_DIR / "java_repos_cleaned.csv", encoding="utf-8-sig")
    react_df = pd.read_csv(REPOS_LIST_DIR / "react_ts_repos_cleaned.csv", encoding="utf-8-sig")

    all_chunks = []

    print("\n--- Extracting Java Repositories ---")
    for idx, row in java_df.iterrows():
        cks = extract_chunks_from_repo(row['url'], row['repo_name'], "Java", max_chunks_per_repo=10)
        all_chunks.extend(cks)

    print("\n--- Extracting React/TypeScript Repositories ---")
    for idx, row in react_df.iterrows():
        cks = extract_chunks_from_repo(row['url'], row['repo_name'], "React/TS", max_chunks_per_repo=10)
        all_chunks.extend(cks)

    with open(CHUNKS_POOL_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Completed extraction! Saved {len(all_chunks)} chunks to {CHUNKS_POOL_FILE.name}.\n")
    return all_chunks


def heuristic_evaluate_single(pair):
    """Rule-based heuristic evaluator fallback."""
    code_text = (pair['chunk_content'] + " " + pair['class_name'] + " " + pair['method_name']).lower()
    is_jd_java = "java" in pair['jd_title'].lower() or "spring" in pair['jd_title'].lower()
    is_code_java = pair.get('language') == 'Java' or pair['file_path'].endswith('.java')
    if is_jd_java != is_code_java:
        return 0, f"Cross-language code snippet unrelated to position {pair['jd_title']}."

    strong_patterns = [
        '@restcontroller', '@postmapping', '@getmapping', '@putmapping', '@deletemapping',
        '@service', '@transactional', 'findby', 'save', 'delete', 'useeffect', 'usestate',
        'usecallback', 'usememo', 'usecontext', 'redux', 'dispatch', 'createasyncthunk', 'axios'
    ]
    if any(p in code_text for p in strong_patterns) and any(w in code_text for w in ['user', 'order', 'product', 'auth', 'cart', 'post', 'item', 'fetch', 'save', 'update', 'blog', 'store']):
        return 2, f"Direct implementation evidence for mandatory competencies in JD ({pair['method_name'] or pair['class_name']})."

    partial_patterns = ['@entity', '@table', '@configuration', '@bean', '@data', 'interface', 'export const', 'props', 'styled', 'css']
    if any(p in code_text for p in partial_patterns) or len(pair['chunk_content'].splitlines()) < 8:
        return 1, "Indirect or peripheral configuration/boilerplate code."

    return 0, "Code snippet does not evidence required technical competencies."


def main():
    parser = argparse.ArgumentParser(description="Corpus Chunk Extraction and Dataset Curation Pipeline.")
    parser.add_argument("--force-reextract", action="store_true", help="Force re-extraction from clone repositories")
    parser.add_argument("--output", type=str, default=str(OUTPUT_MASTER_FILE), help="Output master dataset file path")
    args = parser.parse_args()

    print("=" * 80)
    print("CORPUS EXTRACTION & CANDIDATE EVIDENCE CHUNKING PIPELINE")
    print("=" * 80)

    # 1. Load 25 JDs
    jds = load_all_jds()

    # 2. Check existing master file
    out_file = Path(args.output)
    if out_file.exists() and not args.force_reextract:
        df_existing = pd.read_csv(out_file)
        if len(df_existing) == 500 and df_existing['human_label'].notna().all():
            print(f"\nMaster 500-sample dataset already exists at: {out_file.name}")
            print(f"Total samples: 500")
            print("\nLabel distribution:")
            print(df_existing['human_label'].value_counts().to_string())
            return

    # 3. Extract chunks pool from 50 repos
    chunks_pool = build_or_load_chunk_pool(force_reextract=args.force_reextract)
    print(f"\nAll corpus chunks synchronized at: {out_file.name}")


if __name__ == "__main__":
    main()
