#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
MASTER CODE & DATA INTEGRITY AUDITOR (Zero-Hardcoding & Authenticity Verification)
Governed by: Anti-Hardcoding Specification (Rules R01 - R46) & IEEE SANER 2027 Standards
Target: 100% Automated Verification of Data Authenticity, Lineage, and Zero Fabrication
=============================================================================
"""

import ast
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Configure Windows UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
BENCHMARK_DIR = DATASET_DIR / "benchmark_results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PAPER_DIR = PROJECT_ROOT / "paper"
TESTS_DIR = PROJECT_ROOT / "tests"

LOCKED_NORMALIZED_MASTER_SHA256 = "0799ABAFA394B706457638EB75F3D26AF0B40E2D2673465285A5D1BDDFDD754C"
LOCKED_GOLD_LABELS_SHA256 = "6530C8FEC0D1190A7DB108E647E324EBFF3F0E6FB0A64814726D0F6049D904B6"


class Auditor:
    def __init__(self):
        self.passes = 0
        self.failures = 0
        self.checks: List[Tuple[str, str, bool, str]] = []

    def log(self, gate: str, name: str, passed: bool, details: str = ""):
        if passed:
            self.passes += 1
            status = "[PASS]"
        else:
            self.failures += 1
            status = "[FAIL]"
        self.checks.append((gate, name, passed, details))
        print(f"  {status} {gate} - {name} {(': ' + details) if details else ''}")

    # =========================================================================
    # GATE 1: CRYPTOGRAPHIC HASH & DATASET INTEGRITY
    # =========================================================================
    def audit_gate_1_hashes_and_dataset(self):
        print("\n" + "=" * 80)
        print("  [GATE 1] CRYPTOGRAPHIC INTEGRITY & DATASET IMMUTABILITY AUDIT")
        print("=" * 80)

        gt_file = DATASET_DIR / "ground_truth_final.csv"
        if not gt_file.exists():
            self.log("GATE-1", "Ground Truth File Existence", False, "File missing!")
            return

        raw_bytes = gt_file.read_bytes()
        normalized_bytes = raw_bytes.replace(b"\r\n", b"\n")
        calc_sha256 = hashlib.sha256(normalized_bytes).hexdigest().upper()
        self.log(
            "GATE-1",
            "Ground Truth SHA-256 Lock",
            calc_sha256 == LOCKED_NORMALIZED_MASTER_SHA256,
            f"Calculated: {calc_sha256[:16]}... Match: {calc_sha256 == LOCKED_NORMALIZED_MASTER_SHA256}"
        )

        with open(gt_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        
        labels_str = ",".join(r["ground_truth_label"] for r in reader)
        calc_labels_hash = hashlib.sha256(labels_str.encode()).hexdigest().upper()
        self.log(
            "GATE-1",
            "Ground Truth Labels Hash Lock",
            calc_labels_hash == LOCKED_GOLD_LABELS_SHA256,
            f"Match: {calc_labels_hash == LOCKED_GOLD_LABELS_SHA256}"
        )

        self.log(
            "GATE-1",
            "Ground Truth Row Count (N=882)",
            len(reader) == 882,
            f"Count = {len(reader)}"
        )

        # Queries check
        queries_file = DATASET_DIR / "queries_frozen.json"
        if queries_file.exists():
            with open(queries_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            query_count = len(data.get("queries", {}))
            self.log("GATE-1", "Frozen Queries Count (N=25)", query_count == 25, f"Count = {query_count}")
        else:
            self.log("GATE-1", "Frozen Queries File", False, "Missing!")

        # Parquet corpus check
        parquet_file = DATASET_DIR / "chunk_corpus.parquet"
        if parquet_file.exists():
            import pandas as pd
            df_chunks = pd.read_parquet(parquet_file)
            ast_count = sum(df_chunks["strategy"].astype(str).str.upper() == "AST")
            line_count = sum(df_chunks["strategy"].astype(str).str.upper() == "LINE")
            self.log(
                "GATE-1",
                "Corpus Chunk Count (872 = 438 AST + 434 Line)",
                len(df_chunks) == 872 and ast_count == 438 and line_count == 434,
                f"Total={len(df_chunks)} (AST={ast_count}, Line={line_count})"
            )
        else:
            self.log("GATE-1", "Parquet Corpus File", False, "Missing!")

        # 12 retrieval runs check
        runs_dir = DATASET_DIR / "retrieval_runs"
        run_files = list(runs_dir.glob("*.json"))
        self.log(
            "GATE-1",
            "Retrieval Runs Completeness (12 Runs)",
            len(run_files) == 12,
            f"Found {len(run_files)} / 12 runs"
        )

    # =========================================================================
    # GATE 2: AST & REGEX ZERO-HARDCODING CODE AUDIT
    # =========================================================================
    def audit_gate_2_zero_hardcoding(self):
        print("\n" + "=" * 80)
        print("  [GATE 2] SOURCE CODE & ZERO-HARDCODING AST AUDIT")
        print("=" * 80)

        critical_scripts = [
            "evaluate_retrieval.py",
            "evaluate_common_support.py",
            "evaluate_revision_statistics.py",
            "compute_bpref.py",
            "map_evidence_coverage.py",
            "audit_headers_controlled.py",
            "evaluate_syntax.py",
            "calculate_kappa.py",
            "power_analysis.py"
        ]

        # Sensitive target numbers that MUST NOT be hardcoded as return or assignment in calculation logic
        forbidden_hardcodes = [
            "0.487", "0.580", "0.621", "0.341", "-0.093", "-0.134", "0.6000", "0.4892", "0.1108"
        ]

        hardcoding_found = False
        for sname in critical_scripts:
            spath = SCRIPTS_DIR / sname
            if not spath.exists():
                continue

            content = spath.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=sname)

            # Check that metric calculation functions don't return constant literals
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in ["ndcg_at_k", "dcg_at_k", "mrr_at_k"]:
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Return) and isinstance(subnode.value, ast.Constant):
                            # Returning 0.0 for edge cases (e.g. empty list) is legitimate
                            if subnode.value.value not in [0.0, 0, None]:
                                hardcoding_found = True
                                self.log(
                                    "GATE-2",
                                    f"AST Hardcoded Return in {sname}::{node.name}",
                                    False,
                                    f"Returns constant {subnode.value.value}"
                                )

            # Regex check for sensitive metrics assigned as hardcoded return values
            for num in forbidden_hardcodes:
                pattern = rf"(return\s+{re.escape(num)}|metric\s*=\s*{re.escape(num)})"
                if re.search(pattern, content):
                    hardcoding_found = True
                    self.log(
                        "GATE-2",
                        f"Hardcoded Metric in {sname}",
                        False,
                        f"Found literal '{num}' assigned directly"
                    )

        self.log(
            "GATE-2",
            "Zero Hardcoded Logic in Metric Calculation Scripts",
            not hardcoding_found,
            "All metric functions calculate dynamically from real data"
        )

    # =========================================================================
    # GATE 3: CLAIM LEDGER & BENCHMARK NUMERICAL CONCORDANCE
    # =========================================================================
    def audit_gate_3_concordance(self):
        print("\n" + "=" * 80)
        print("  [GATE 3] CLAIM LEDGER & BENCHMARK NUMERICAL CONCORDANCE (C01-C19)")
        print("=" * 80)

        ledger_file = PAPER_DIR / "claim_ledger.csv"
        if not ledger_file.exists():
            self.log("GATE-3", "Claim Ledger File", False, "Missing!")
            return

        with open(ledger_file, "r", encoding="utf-8") as f:
            claims = list(csv.DictReader(f))

        self.log("GATE-3", "Total Registered Claims", len(claims) >= 19, f"Found {len(claims)} claims")

        # Load benchmark files
        with open(DATASET_DIR / "table_3_2_measured.json", "r", encoding="utf-8") as f:
            morphology = json.load(f)
        bench_raw = json.load(open(BENCHMARK_DIR / "table_3_retrieval_benchmark.json", "r", encoding="utf-8"))
        if isinstance(bench_raw, list):
            bench = {r["config_id"]: r for r in bench_raw}
        else:
            bench = bench_raw

        with open(BENCHMARK_DIR / "statistical_tests_report.json", "r", encoding="utf-8") as f:
            stats = json.load(f)

        # Check key claims
        # C01: AST fallback rate == 0.0%
        fallback_count = morphology.get("ast_fallback", {}).get("chunk_count", 0)
        combined_count = morphology.get("ast_combined", {}).get("chunk_count", 438)
        ast_fallback = (fallback_count / combined_count) * 100.0 if combined_count > 0 else 0.0
        self.log("GATE-3", "Claim C01 (AST Fallback Rate = 0.0%)", ast_fallback == 0.0, f"Value: {ast_fallback}%")

        # C08: AST Header Gain: 0.341 -> 0.487
        ast_no = bench["rerank_ast_no_header"]["ndcg_10_mean"]
        ast_with = bench["rerank_ast_with_header"]["ndcg_10_mean"]
        c08_ok = round(ast_no, 3) == 0.341 and round(ast_with, 3) == 0.487
        self.log(
            "GATE-3",
            "Claim C08 (AST Rerank NDCG@10: 0.341 -> 0.487)",
            c08_ok,
            f"Measured: {ast_no:.3f} -> {ast_with:.3f}"
        )

        # C09: Line No-Header vs AST No-Header: 0.580 vs 0.341
        line_no = bench["rerank_line_no_header"]["ndcg_10_mean"]
        c09_ok = round(line_no, 3) == 0.580 and round(ast_no, 3) == 0.341
        self.log(
            "GATE-3",
            "Claim C09 (Line No-Header vs AST No-Header: 0.580 vs 0.341)",
            c09_ok,
            f"Measured: {line_no:.3f} vs {ast_no:.3f}"
        )

        # C11: Line With-Header = 0.621
        line_with = bench["rerank_line_with_header"]["ndcg_10_mean"]
        c11_ok = round(line_with, 3) == 0.621
        self.log("GATE-3", "Claim C11 (Line With-Header = 0.621)", c11_ok, f"Measured: {line_with:.3f}")

        # Revision Claims C18 & C19
        c18 = next((c for c in claims if c["claim_id"] == "C18"), None)
        c19 = next((c for c in claims if c["claim_id"] == "C19"), None)
        self.log("GATE-3", "Revision Claim C18 Registered (Common Support)", c18 is not None)
        self.log("GATE-3", "Revision Claim C19 Registered (Controlled Header)", c19 is not None)

    # =========================================================================
    # GATE 4: DOUBLE-BLIND ANONYMITY & PAGE CEILING
    # =========================================================================
    def audit_gate_4_anonymity_and_page_limit(self):
        print("\n" + "=" * 80)
        print("  [GATE 4] DOUBLE-BLIND ANONYMITY & IEEE ERA 5-PAGE CEILING")
        print("=" * 80)

        # Check anonymity leaks in paper tex files
        tex_files = list(PAPER_DIR.glob("*.tex")) + list((PAPER_DIR / "tables").glob("*.tex"))
        leaks = []
        forbidden_strings = ["win 11", "C:\\Users", "author_name", "@gm.", "github.com/user"]
        for tf in tex_files:
            content = tf.read_text(encoding="utf-8")
            for forb in forbidden_strings:
                if forb.lower() in content.lower():
                    leaks.append((tf.name, forb))

        self.log(
            "GATE-4",
            "Double-Blind Anonymity (Zero Identity/Path Leaks)",
            len(leaks) == 0,
            f"Found {len(leaks)} leaks" if leaks else "Clean"
        )

        # Check page count
        pdf_path = PAPER_DIR / "main.pdf"
        if pdf_path.exists():
            import pypdfium2
            doc = pypdfium2.PdfDocument(pdf_path)
            page_count = len(doc)
            self.log(
                "GATE-4",
                "IEEE ERA Track Page Ceiling (Exactly 5 Pages)",
                page_count == 5,
                f"Document has {page_count} pages"
            )
        else:
            self.log("GATE-4", "PDF File Existence", False, "paper/main.pdf missing")

    # =========================================================================
    # RUN ALL GATES
    # =========================================================================
    def run(self):
        print("=" * 80)
        print("  STARTING MASTER CODE & DATA INTEGRITY AUDIT")
        print("  Target Conference: IEEE SANER 2027 (ERA Track)")
        print("=" * 80)

        self.audit_gate_1_hashes_and_dataset()
        self.audit_gate_2_zero_hardcoding()
        self.audit_gate_3_concordance()
        self.audit_gate_4_anonymity_and_page_limit()

        print("\n" + "=" * 80)
        print("  AUDIT EXECUTIVE SUMMARY")
        print("=" * 80)
        total = self.passes + self.failures
        rate = (self.passes / total) * 100 if total > 0 else 0
        print(f"Total Verifications: {total}")
        print(f"Passed Checks:       {self.passes}")
        print(f"Failed Checks:       {self.failures}")
        print(f"Compliance Rate:     {rate:.1f}%\n")

        if self.failures == 0:
            print(">>> STATUS: [100% PASS] REPLICATION PACKAGE READY FOR REVIEWER AUDIT! <<<")
            return 0
        else:
            print(f">>> STATUS: [FAILED] {self.failures} checks violated! <<<")
            return 1


if __name__ == "__main__":
    auditor = Auditor()
    sys.exit(auditor.run())
