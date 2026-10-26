# Replication Package: An Empirical Evaluation of Source Code Chunking and Context Headers for Requirement Evidence Retrieval

[![Target Conference](https://img.shields.io/badge/IEEE_SANER_2027-ERA_Track-blue.svg)](https://conf.researchr.org/home/saner-2027)
[![Review Policy](https://img.shields.io/badge/Review_Policy-Double--Anonymous-orange.svg)]()
[![Python Version](https://img.shields.io/badge/Python-3.10_%7C_3.11_%7C_3.12-blue.svg)]()
[![Automated Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/Code_License-MIT-yellow.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-lightgrey.svg)](LICENSE)

> **Target Venue:** 34th IEEE International Conference on Software Analysis, Evolution and Reengineering (IEEE SANER 2027)  
> **Track:** Early Research Achievements (ERA Track)  
> **Review Policy:** Double-Anonymous Peer Review  
> **Experimental Protocol:** Pre-specified Protocol (Internal Analysis Plan) strictly governed by Anti-Hardcoding Rules R1–R46 and `analysis_plan_addendum.md`  

---

## 📌 Executive Summary & Research Questions

This replication package provides the curated code chunk corpora, precomputed retrieval runs, human ground truth annotations, Tree-sitter chunking engine, and reproduction scripts necessary to independently verify and reproduce all empirical tables, figures, and statistical tests reported in our paper.

Our chunking strategy is **AST-first with progressive context disclosure**: it extracts semantic syntax units (methods, records, interfaces, functional components, hooks, and wrappers) via Tree-sitter AST parsers with prepended context headers (`// Class: ...`, `// Method: ...`). In our corpus of 438 AST chunks across 40 repositories (189 source files), the upgraded AST parser achieves **100.0% pure AST extraction** (0 fallback chunks, 0.0% fallback rate, fully audited in `dataset/fallback_audit.csv`).

### Empirical Findings:

* **RQ1: Syntactic & Structural Integrity (Table 3.2):** How does method-level AST Progressive Disclosure chunking compare against standard fixed-window line slicing (50 LOC, 10 overlap) in preserving syntax boundaries and minimizing token bloat?
  * *AST Pure (N=438):* Preserves complete syntactic boundaries in **87.2%** of chunks (56 cuts across outer module wrappers), achieves a **78.1%** parse error-free rate, provides **100.0%** context header retention (class and method hierarchy), reduces mean LOC to **26.7 ± 23.3** (median: 16.0), reduces mean content tokens to **250.8 ± 211.7**, and limits 512-token truncation to **15.8%** (69 chunks).
  * *AST Fallback (N=0):* **0.0%** fallback rate (0/438 chunks, 100.0% boundary intact, 100.0% valid syntax, mean LOC 0.0, mean tokens 0.0, truncation rate 0.0%).
  * *AST Combined (N=438):* Identical to pure AST (87.2% boundary intact, 78.1% valid syntax, 100.0% context header retention, mean LOC 26.7, mean tokens 250.8, 15.8% 512-token truncation rate).
  * *Line Baseline (N=434):* Line-based slicing cuts across syntactic boundaries in **56.0%** of chunks (**44.0% intact**, 243/434 cuts), causes parse errors in **77.6%** of snippets (**22.4% valid syntax**), provides **0.0%** context header retention, mean LOC **41.0 ± 13.1**, mean tokens **390.8**, and suffers a **28.6%** 512-token truncation rate (124 chunks).

* **RQ2: 2×2 Factorial Retrieval Quality & Re-ranking (Table 3.3):** Across a $2 \times 2$ Factorial Design ($\{\text{AST-first Method}, \text{Line-based}\} \times \{\text{Without Header}, \text{With Header}\}$) evaluated over 3 retrieval tiers (BM25, BGE-M3 Dense, BGE-Reranker-Base):
  * *Top Performer:* **Line+Header** is the empirical top performer on aggregate reranking metrics ($\text{NDCG@10} = \mathbf{0.6206}$, $\text{NDCG@1} = \mathbf{0.800}$, $\text{MRR@10} = \mathbf{0.950}$, $\text{CP@5} = \mathbf{0.933}$).
  * *AST Performance:* Isolated AST method chunks without headers suffer severe context deprivation ($\text{NDCG@10} = \mathbf{0.3407}$, $\text{NDCG@1} = \mathbf{0.340}$, $\text{MRR@10} = \mathbf{0.699}$, $\text{CP@5} = \mathbf{0.669}$). Adding progressive context headers significantly boosts AST re-ranking performance to $\mathbf{0.4866}$ and Top-1 ranking to $\text{NDCG@1} = \mathbf{0.740}$ (**Comp 3:** $\Delta = \mathbf{+0.1459}$, paired Wilcoxon $W = 24.0$, $p_{\text{raw}} < 0.0001$, $p_{\text{Holm}} = \mathbf{0.0001} < 0.001$, 95% Bootstrap CI $[+0.0862, +0.2071]$). While AST+Header achieves higher precision metrics than AST No-Header, the line baseline without headers remains stronger on aggregate retrieval ($\text{NDCG@10} = \mathbf{0.5797}$, **Comp 1:** $\Delta = -0.0930, p_{\text{Holm}} = 0.0438$).
  * *Exploratory Comp 5 (AST+Header vs. Line+Header):* $\Delta = \mathbf{-0.1339}$, 95% Bootstrap CI $[-0.1992, -0.0704]$, Wilcoxon $W = 36.0, p = 0.0003 < 0.05$ (Line+Header wins 20 queries, AST+Header wins 5 queries, 0 ties).
  * *Candidate Pool Random Baseline:* $\text{NDCG@10} = \mathbf{0.562} \pm 0.146$ across 1,000 permutations within pooled candidates. Whole-corpus random baseline scores $\mathbf{0.0224}$, and when systems are evaluated on judged candidates (Shortlist evaluation), Line+Header achieves $\mathbf{0.7243}$, surpassing the pool random baseline.

* **Inter-Annotator Reliability (Table 3.1 & Figure 3.1):**
  * Evaluated on all 882 candidate pairs between Annotator 1 and Annotator 2 across all 12 systems pooled at Top-5: Exact Agreement = **98.07%** (865/882), Minor Disagreements = **1.81%** (16/882), Severe Disagreements = **0.11%** (1/882).
  * **Quadratic Weighted Cohen's Kappa ($\kappa_w$):** **`0.9822`** (*Almost Perfect Agreement*, Landis & Koch 1977 [>0.80]).
  * **Linear Weighted Kappa:** **`0.9764`** | **Unweighted Kappa:** **`0.9708`**.
  * **Subgroup Kappa:** Java $\kappa_w = \mathbf{0.9805}$ ($N=543$, 98.16% exact agreement), React/TS $\kappa_w = \mathbf{0.9789}$ ($N=339$, 97.94% exact agreement).
  * Adjudication of 17 disagreements was conducted by a single Lead Adjudicator based on the technical rubric (detailed in `dataset/disagreements_adjudication.csv` and summarized in `dataset/benchmark_results/adjudication_summary.json`). Annotator 1 scored pairs via the 3-point rubric directly, while Annotator 2 recorded diagnostic technical notes.

---

## 📁 Repository Structure

```text
.
├── ai-engine/                                  # Core parsing & chunking engine
│   └── parser/
│       ├── tree_sitter_loader.py               # Tree-sitter AST Progressive Disclosure & line fallback chunker
│       └── __init__.py
├── dataset/                                    # Curated datasets, queries, and benchmark runs
│   ├── queries_frozen.json                     # 25 pre-registered frozen queries with metadata
│   ├── manifest_repos.csv                      # 40 GitHub repositories with commit SHAs & licenses
│   ├── source_file_intervals.json              # Pinned method intervals across 189 files for boundary analysis
│   ├── fallback_audit.csv                      # Diagnostic audit of line-fallback chunks (0 chunks, 0.0% fallback)
│   ├── chunk_corpus.parquet                    # Dual corpus: 872 chunks (438 AST + 434 Line-based)
│   ├── ground_truth_final.csv                  # 882 unified labeled pairs (exhaustive Top-5 pool across 12 runs)
│   ├── ground_truth_annotator1.csv             # Anonymized annotations from Annotator 1 (882 pairs)
│   ├── ground_truth_annotator2.csv             # Anonymized annotations from Annotator 2 (882 pairs)
│   ├── disagreements_adjudication.csv          # 17 adjudicated disagreements with technical rationales
│   ├── to_label.csv                            # Neutral pooling candidate pairs (Top-5 across all 12 runs, 882 pairs)
│   ├── pool_size_report.csv                    # Adaptive pool size report across 25 JDs
│   ├── table_3_2_measured.json                 # Real corpus morphology metrics across 4 subgroups
│   ├── table_3_2_report.txt                    # Formatted text report for Table 3.2
│   ├── sampling_frame_log.json                 # Sampling log for AST and Line chunking (Rule R10)
│   ├── annotation_guidelines.md                # 3-tier relevance labeling guideline for annotators
│   ├── PROTOCOL_DEVIATIONS.md                  # Comprehensive empirical audit of protocol adjustments
│   ├── retrieval_runs/                         # 12 precomputed retrieval run JSON files (3 tiers x 4 conditions)
│   │   ├── bm25_ast_no_header.json
│   │   ├── bm25_ast_with_header.json
│   │   ├── bm25_line_no_header.json
│   │   ├── bm25_line_with_header.json
│   │   ├── dense_ast_no_header.json
│   │   ├── dense_ast_with_header.json
│   │   ├── dense_line_no_header.json
│   │   ├── dense_line_with_header.json
│   │   ├── rerank_ast_no_header.json
│   │   ├── rerank_ast_with_header.json
│   │   ├── rerank_line_no_header.json
│   │   └── rerank_line_with_header.json
│   └── benchmark_results/                      # Evaluation outputs, statistical reports, and figures
│       ├── table_3_retrieval_benchmark.csv     # Complete metrics for all 12 configurations + Random Baseline
│       ├── table_3_retrieval_benchmark.json
│       ├── statistical_tests_report.txt        # Family of 4 Wilcoxon tests + Holm-Bonferroni + Bootstrap CIs
│       ├── statistical_tests_report.json
│       ├── sensitivity_analysis_report.txt     # Exploratory sensitivity and subgroup analysis report
│       ├── sensitivity_analysis_results.json   # Sensitivity metrics in JSON format
│       ├── per_query_evaluation.csv            # Per-query metrics (300 records: 12 configs x 25 JDs)
│       ├── kappa_evaluation_report.txt         # Inter-annotator agreement evaluation report (Table 3.1)
│       ├── confusion_matrix_kappa.png          # Heatmap visualization of annotator confusion matrix
│       ├── pool_coverage_report.csv            # Audit of candidate pool coverage across all 12 runs
│       ├── pool_reconciliation_report.txt      # Mathematical reconciliation of candidate pool derivations
│       ├── adjudication_summary.json           # Template-level taxonomy of consensus adjudication
│       ├── controlled_header_audit.json        # Header field audit & BGE-M3 token impact
│       ├── evidence_chunk_mapping.csv          # Pair-level evidence-to-chunk mapping (882 pairs)
│       ├── evidence_coverage_report.json       # Evidence and line coverage metrics
│       ├── common_support_membership.csv       # Inclusion/exclusion flags across 882 pairs
│       ├── common_support_metrics.csv          # Benchmark metrics on shared evidence universe
│       ├── common_support_per_query.csv        # Per-query metrics on Common Support universe
│       └── revision_statistical_tests.json     # Hypothesis testing on Common Support & confound analysis
├── paper/                                      # SANER ERA manuscript, tables, and figures
│   ├── main.tex                                # Main LaTeX conference paper (IEEEtran, 5-page ERA limit)
│   ├── main.pdf                                # Compiled camera-ready conference PDF
│   ├── figs/                                   # Vector graphics (fig1.pdf interaction plot)
│   └── tables/                                 # Audited LaTeX tables (table_1_paper, table_2_paper, table_4_paper)
├── reports/                                    # Systematic Phase Audit Reports (Phases 0–6)
│   ├── phase0_baseline_audit.md                # Phase 0: Baseline & repository audit
│   ├── phase1_header_control.md                # Phase 1: Header audit & common schema control
│   ├── phase2_coverage_evidence.md             # Phase 2: Evidence mapping & reachability audit
│   ├── phase3_common_support_experiments.md    # Phase 3: Common Support retrieval benchmark
│   ├── phase4_statistics.md                    # Phase 4: Hypothesis tests & confound decomposition
│   ├── phase5_manuscript_revision.md           # Phase 5: Manuscript change log & verification
│   └── phase6_final_audit.md                   # Phase 6: Final pre-submission compliance audit
├── scripts/                                    # Standalone reproduction & evaluation runners
│   ├── reproduce_all.py                        # Master one-click runner executing all 7 steps end-to-end
│   ├── calculate_kappa.py                      # Reproduces Table 3.1 & Figure 3.1 (Inter-annotator agreement)
│   ├── evaluate_syntax.py                      # Reproduces Table 3.2 (AST pure, fallback, combined vs Line)
│   ├── evaluate_retrieval.py                   # Reproduces Table 3.3 (2x2 retrieval metrics & Wilcoxon tests)
│   ├── evaluate_sensitivity.py                 # Evaluates exploratory subgroup & sensitivity analyses
│   ├── report_pool_coverage.py                 # Generates candidate pool coverage analysis
│   ├── reconcile_pool.py                       # Validates coordinate-level candidate pool reconciliation
│   ├── summarize_adjudication.py               # Analyzes structured template structure of adjudication
│   ├── audit_headers_controlled.py             # Audits context headers and BGE-M3 token lengths
│   ├── map_evidence_coverage.py                # Maps 882 candidate pairs to chunk pools
│   ├── evaluate_common_support.py              # Evaluates retrieval on Common Support universe
│   └── evaluate_revision_statistics.py         # Runs Wilcoxon-Holm tests on Common Support
├── tests/                                      # Complete automated test suite (100% passing)
│   ├── test_corpus_characteristics.py          # Verifies Table 3.2 measurements and subgroups
│   ├── test_doc_hashes.py                      # Verifies documentation SHA-256 hashes against locked keys
│   ├── test_evaluation_phase5.py               # 2x2 Factorial benchmark verification
│   ├── test_experimental_controls.py           # Verifies experimental controls across conditions
│   ├── test_gold_integrity.py                  # Cryptographic SHA-256 integrity of ground truth
│   ├── test_metrics.py                         # Unit tests for IR and agreement metrics
│   ├── test_neutral_labeling.py                # Pooling budget tests
│   ├── test_parser.py                          # Unit tests for Tree-sitter parsers and extractors
│   ├── test_pool_coverage.py                   # Verifies candidate pool coverage distribution
│   ├── test_readme_numbers.py                  # Asserts README numerical claims match benchmark files
│   ├── test_reproducibility.py                 # Determinism & seed consistency verification
│   └── test_text_integrity.py                  # Dataset count cross-checks & claim safeguards
├── analysis_plan.md                            # Pre-registered experimental analysis plan (frozen at Phase 0)
├── analysis_plan_addendum.md                   # Exploratory analysis protocol addendum (Path B)
├── PROVENANCE.md                               # Comprehensive audit trail of data sources, licensing, and annotations
├── requirements.txt                            # Core package dependencies
├── requirements.lock                           # Fully locked, pinned transitive dependencies
├── run_all.bat                                 # One-click Windows master execution script
├── run_all.sh                                  # One-click Unix / macOS master execution script
├── .gitattributes                              # Cross-platform newline normalization rules
├── LICENSE                                     # MIT (Code) and CC-BY-4.0 (Dataset/Docs)
└── README.md                                   # Comprehensive replication guide
```

---

## ⚡ Quick Start & Environment Setup

### 1. Prerequisites
* **Python Version:** Python 3.10, 3.11, 3.12, 3.13, or 3.14.
* **Operating System:** Platform-independent standard Python environment (tested on Windows 11).
* **Hardware:** Minimal requirements (< 4 GB RAM, runs entirely on CPU).

### 2. Environment Setup
Create and activate an isolated virtual environment:

```bash
# Clone the anonymous replication package
git clone <anonymous-repository-url>
cd ast-rag-replication-package

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows (cmd):
venv\Scripts\activate.bat
# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# Install locked dependencies
pip install -r requirements.lock
# (Or pip install -r requirements.txt)
```

---

## 🚀 One-Click Master Reproduction

To execute the entire empirical validation pipeline end-to-end in a single command, run:

```bash
python scripts/reproduce_all.py
```
*(Alternatively, on Windows run `run_all.bat`, or on Linux/macOS run `bash run_all.sh`).*

### Typical Output (~30 seconds on commodity CPU):
```text
===============================================================================================
  MASTER REPRODUCTION SUMMARY
===============================================================================================
Step     | Target Paper Item                              | Status   | Duration | Key Empirical Verification
-----------------------------------------------------------------------------------------------
STEP_1   | Unit Tests & Experimental Controls             | PASS     |    2.5s  | All tests passed
STEP_2   | Table 3.1 & Figure 3.1 (Confusion Matrix)     | PASS     |    3.3s  | Quadratic Kappa = 0.9822, Agreement = 98.1%
STEP_3   | Table 3.2 (AST vs Line Chunking Morphology)   | PASS     |   14.1s  | AST Syntax Intact: 78.1% vs Line: 22.4%
STEP_4   | Table 3.3 & Confirmatory Hypothesis Tests     | PASS     |    3.3s  | Comp 3 AST Header Gain Statistically Significant
STEP_5   | Language, Fallback, Comp 5 & Reachable Qrels   | PASS     |    2.3s  | Comp 5 Delta = -0.1339, 0.0% AST Fallback
STEP_6   | Bpref Metric across BM25, Dense, and Rerank    | PASS     |    1.5s  | BM25 Bpref: 0.129-0.255, Rerank Bpref: 0.170-0.353
STEP_7   | MDE Analysis across 5 Comparisons              | PASS     |    0.8s  | MDE Range: 0.080 to 0.130 NDCG@10 (80% power)
===============================================================================================
Total Reproduction Time: 27.8s
RESULT: ALL 7 EMPIRICAL REPRODUCTION STEPS COMPLETED AND VERIFIED SUCCESSFULLY [PASS]
===============================================================================================
```

---

## 🔬 Detailed Step-by-Step Reproduction Guide

Each empirical claim, table, and statistical hypothesis test can be verified independently:

### Step 1: Automated Test Suite & Controls
Verifies hash immutability (Rule R17), ground truth integrity, metric implementations against hand-calculated proofs, parser edge cases, and numerical consistency of README claims:
```bash
python -m pytest tests/ -v
```

---

### Step 2: Reproduce Table 3.1 & Figure 3.1 — Inter-Annotator Reliability
Evaluates independent human annotations from Annotator 1 and Annotator 2 across all 882 candidate pairs, computing Quadratic Weighted Cohen's Kappa ($\kappa_w$), Linear Kappa, exact agreement rate, and generating the confusion matrix heatmap:
```bash
python scripts/calculate_kappa.py
```
* **Key Verified Results:**
  * Total Candidate Evidence Pairs ($N$): **882**
  * Exact Consensus Matches ($p_o$): **865 / 882 (98.07%)**
  * Minor Disagreements ($|\Delta| = 1$): **16 / 882 (1.81%)**
  * Severe Disagreements ($|\Delta| = 2$): **1 / 882 (0.11%)**
  * **Quadratic Weighted Kappa ($\kappa_w$):** **`0.9822`** (*Almost Perfect Agreement*, Landis & Koch 1977 [>0.80])
  * **Java Subgroup Kappa:** **`0.9805`** ($N=543$, 98.16% exact agreement)
  * **React/TS Subgroup Kappa:** **`0.9789`** ($N=339$, 97.94% exact agreement)
* **Generated Artifacts:**
  * Report: `dataset/benchmark_results/kappa_evaluation_report.txt`
  * Figure: `dataset/benchmark_results/confusion_matrix_kappa.png`

---

### Step 3: Reproduce Table 3.2 — Real Corpus Morphology & Syntax Boundaries
Executes live Tree-sitter syntax parsing and HuggingFace AutoTokenizer (`BAAI/bge-m3`) across all 872 chunks from 189 source files:
```bash
python scripts/evaluate_syntax.py
```
* **Key Verified Results:**
  * **AST Pure (N=438):** 87.2% boundary intact (56 cuts), 78.1% valid syntax, 100.0% context header retention, mean LOC 26.7, mean tokens 250.8, truncation >512 tokens: 15.8% (69 chunks).
  * **AST Fallback (N=0):** 0.0% fallback rate (0/438 chunks, 100.0% boundary intact, 100.0% valid syntax, mean LOC 0.0, mean tokens 0.0, truncation >512 tokens: 0.0% (0 chunks)).
  * **AST Combined (N=438):** 87.2% boundary intact (56 cuts), 78.1% valid syntax, 100.0% context header retention, mean LOC 26.7, mean tokens 250.8, truncation >512 tokens: 15.8% (69 chunks).
  * **Line Baseline (N=434):** 44.0% boundary intact (243 cuts), 22.4% valid syntax, 0.0% header retention, mean LOC 41.0, mean tokens 390.8, truncation >512 tokens: 28.6% (124 chunks).
* **Generated Artifacts:**
  * LaTeX Table: `report/tables/table_T2.tex` (and `paper/tables/table_2_paper.tex`)
  * JSON Metrics: `dataset/table_3_2_measured.json`
  * Report: `dataset/table_3_2_report.txt`

---

### Step 4: Reproduce Table 3.3 — 2×2 Factorial Retrieval Benchmark & Hypothesis Testing
Computes NDCG@{1, 3, 5, 10}, MRR@10, Precision@5, Recall@10, and Context Precision@5 across all 12 configurations (3 tiers $\times$ 4 factorial conditions) + Random Baseline. Executes the pre-registered Family of 4 Wilcoxon signed-rank tests with Holm-Bonferroni correction and 95% Bootstrap CIs (10,000 resamples, seed=42):
```bash
python scripts/evaluate_retrieval.py
```
* **Key Verified Results:**
  * **Comp 1 (Full Proposed AST+Header vs. Baseline Line No-Header):** $\Delta = \mathbf{-0.0930}$, 95% Bootstrap CI $[-0.1636, -0.0228]$, Wilcoxon $W = 78.0$, $p_{\text{raw}} = 0.0219$, $p_{\text{Holm}} = \mathbf{0.0438} < 0.05$ (Statistically significant).
  * **Comp 2 (Pure Chunking Effect without Headers):** $\Delta = \mathbf{-0.2390}$, 95% Bootstrap CI $[-0.2919, -0.1847]$, $W = 3.0$, $p_{\text{raw}} < 0.0001$, $p_{\text{Holm}} = \mathbf{0.0000} < 0.001$.
  * **Comp 3 (Header Effect on AST Chunks):** $\Delta = \mathbf{+0.1459}$, 95% Bootstrap CI $[+0.0862, +0.2071]$, $W = 24.0$, $p_{\text{raw}} < 0.0001$, $p_{\text{Holm}} = \mathbf{0.0001} < 0.001$ (Statistically significant improvement of progressive context headers).
  * **Comp 4 (Header Effect on Line Chunks):** $\Delta = \mathbf{+0.0409}$, 95% Bootstrap CI $[-0.0122, +0.0928]$, $W = 118.0$, $p_{\text{raw}} = 0.2411$, $p_{\text{Holm}} = \mathbf{0.2411} \ge 0.05$ (Not statistically significant).
  * **Re-ranking Tier Summary:**
    * Line With-Header: $\text{NDCG@10} = \mathbf{0.6206}$, $\text{NDCG@1} = \mathbf{0.800}$, $\text{MRR@10} = \mathbf{0.950}$, $\text{CP@5} = \mathbf{0.933}$.
    * Line No-Header: $\text{NDCG@10} = \mathbf{0.5797}$, $\text{NDCG@1} = \mathbf{0.800}$, $\text{MRR@10} = \mathbf{0.953}$, $\text{CP@5} = \mathbf{0.903}$.
    * AST With-Header: $\text{NDCG@10} = \mathbf{0.4866}$, $\text{NDCG@1} = \mathbf{0.740}$, $\text{MRR@10} = \mathbf{0.896}$, $\text{CP@5} = \mathbf{0.869}$.
    * AST No-Header: $\text{NDCG@10} = \mathbf{0.3407}$, $\text{NDCG@1} = \mathbf{0.340}$, $\text{MRR@10} = \mathbf{0.699}$, $\text{CP@5} = \mathbf{0.669}$.
  * **Random Baselines:** Full corpus universe random baseline $\text{NDCG@10} = \mathbf{0.0224}$; candidate pool random baseline $\text{NDCG@10} = \mathbf{0.562}$.
  * **Methodological Note on Exhaustive Top-5 Pooling:** All 12 retrieval configurations (BM25, Dense, and Re-ranking) contributed their Top-5 candidates to the unified pool. Judged@5 is strictly 100.0% across all 12 systems. Complete audit is in `dataset/benchmark_results/pool_coverage_report.csv`.
* **Generated Artifacts:**
  * LaTeX Table: `paper/tables/table_1_paper.tex`
  * Complete Metrics: `dataset/benchmark_results/table_3_retrieval_benchmark.csv`
  * Statistical Report: `dataset/benchmark_results/statistical_tests_report.txt`
  * Per-Query Breakdown: `dataset/benchmark_results/per_query_evaluation.csv`
  * Pool Coverage Report: `dataset/benchmark_results/pool_coverage_report.csv`

---

### Step 5: Reproduce Exploratory Sensitivity & Subgroup Analysis
Executes subgroup analyses evaluating language effects (Java vs. React/TS), fallback composition, Comp 5 (AST+Header vs Line+Header), and annotator sensitivity:
```bash
python scripts/evaluate_sensitivity.py
```
* **Key Verified Results:**
  * **Comp 5 (Exploratory AST+Header vs Line+Header):** $\Delta = \mathbf{-0.1339}$, 95% Bootstrap CI $[-0.1992, -0.0704]$, Wilcoxon $W = 36.0, p = 0.0003$ (Line+Header leads AST+Header significantly across 20 of 25 JDs, with 5 wins for AST+Header and 0 ties).
  * **Language Stratification (NDCG@10):**
    * *Java (N=15):* AST+Header = **0.5830**, Line+Header = **0.7158** ($\Delta = -0.1328$).
    * *React/TS (N=10):* AST+Header = **0.3420**, Line+Header = **0.4777** ($\Delta = -0.1357$).
  * **Fallback Composition:** With upgraded Tree-sitter parsers, line-fallback chunk count in Top-10 retrieval is **0 (0.0%)** across all 12 retrieval configurations.
  * **Annotator Robustness ($N=865$ Consensus Pairs):** Ranking order is invariant: Line+Header (**0.6175**) > Line No-Header (**0.5713**) > AST+Header (**0.4866**) > AST No-Header (**0.3338**). Header effect on AST: $\Delta = +0.1528, p < 0.001$.
  * **Judged-Only (Shortlist) Evaluation:** Evaluated strictly within pre-screened judged candidates: Line+Header (**0.7243**) > Line No-Header (**0.6696**) > AST+Header (**0.5774**) > AST No-Header (**0.4156**), with Line+Header easily exceeding the 0.562 pool random baseline.
* **Generated Artifacts:**
  * Report: `dataset/benchmark_results/sensitivity_analysis_report.txt`
  * JSON: `dataset/benchmark_results/sensitivity_analysis_report.json`

---

### Step 6: Binary Preference (Bpref) Analysis for Unjudged BM25 Items
Evaluates the binary preference metric (Buckley & Voorhees, SIGIR 2004) to assess retrieval effectiveness independently of unjudged pooling penalties:
```bash
python scripts/compute_bpref.py
```
* **Key Verified Results:**
  * BM25 tier: bpref ranges from **0.129** to **0.255** across all four configurations.
  * Dense tier: bpref ranges from **0.179** to **0.347**.
  * Rerank tier: bpref ranges from **0.170** to **0.353** (Line+Header 0.353 > Line No-Header 0.300 > AST+Header 0.241 > AST No-Header 0.170).
* **Generated Artifacts:**
  * Summary: `dataset/benchmark_results/bpref_summary.csv`
  * Report: `dataset/benchmark_results/bpref_report.txt`
  * JSON: `dataset/benchmark_results/bpref_report.json`

---

### Step 7: Statistical Power & Minimum Detectable Effect (MDE)
Calculates the minimum detectable effect size at 80% statistical power across all 5 pairwise comparisons:
```bash
python scripts/power_analysis.py
```
* **Key Verified Results:**
  * Detectable effect size ranges from **0.080** to **0.130** NDCG@10 across the 5 comparisons ($\alpha = 0.05$ to $0.0125$).
* **Generated Artifacts:**
  * Report: `dataset/benchmark_results/power_analysis.txt`
  * JSON: `dataset/benchmark_results/power_analysis.json`

---

### Step 8: Pure AST & Closed-Pool Header Ablation Pipelines
Verification suite and execution harnesses for the exploratory pure-AST re-run and closed-pool header component ablation:
```bash
python -m pytest tests/test_pure_ast.py -v
```
* **Pipeline Scripts:**
  * `scripts/pure_ast_run.py`: Runs retrieval on pure AST chunks vs. line windows from the same files.
  * `scripts/pure_ast_pool.py`: Pools unjudged top-$k$ items into neutral format for annotation.
  * `scripts/pure_ast_eval.py`: Evaluates pure-AST runs under shared and arm ideals.
  * `scripts/header_ablation.py`: Closed-pool re-scoring across header field variants.

---

### Step 9: Controlled Header & Common-Support Empirical Replication
Replicate the complete end-to-end controlled header audit and common-support analysis:
```bash
python scripts/audit_headers_controlled.py
python scripts/map_evidence_coverage.py
python scripts/evaluate_common_support.py
python scripts/evaluate_revision_statistics.py
```
* **Pipeline Scripts & Verifications:**
  * `scripts/audit_headers_controlled.py`: Audits all 872 chunks for metadata field presence and BGE-M3 token lengths (`dataset/benchmark_results/controlled_header_audit.json`).
  * `scripts/map_evidence_coverage.py`: Maps all 882 ground-truth candidate pairs to AST and Line chunk pools (`dataset/benchmark_results/evidence_coverage_report.json`).
  * `scripts/evaluate_common_support.py`: Evaluates retrieval benchmarks on the shared reachable evidence universe ($N=826$ pairs, `dataset/benchmark_results/common_support_metrics.csv`).
  * `scripts/evaluate_revision_statistics.py`: Runs paired Wilcoxon signed-rank tests with step-down Holm correction and bootstrap CIs on Common Support (`dataset/benchmark_results/revision_statistical_tests.json`).
  * `scripts/audit_code_and_data_integrity.py`: Master automated audit script verifying 16 cryptographic, zero-hardcoding, concordance, and double-blind gates.
  * `scripts/generate_clean_corpus.py`: Source corpus chunk generator extracting AST nodes and 50-LOC windows across repositories.
  * `scripts/fetch_upstream_repos.py`: Provenance utility to fetch and verify the 40 upstream repositories at their pinned commit SHAs.
  * `scripts/execute_phase2_retrieval_and_pool.py`: Blinded candidate pooling and multi-stage retrieval runner.


## 📊 Dataset Provenance & Integrity Specifications

| Dataset File | Records | Format | Description | Integrity Verification |
|:---|:---:|:---:|:---|:---|
| `queries_frozen.json` | 25 | JSON | Pre-registered industry Job Descriptions (15 Java, 10 TS/React) | Frozen at Phase 0 |
| `manifest_repos.csv` | 40 | CSV | GitHub repositories with pinned commit SHAs, licenses, file lists | Audited licenses |
| `source_file_intervals.json` | 189 | JSON | Pinned method line intervals across 189 source files | Verified via Tree-sitter |
| `fallback_audit.csv` | 0 | CSV | Detailed diagnostic audit of line-fallback chunks (0% fallback) | Audited |
| `chunk_corpus.parquet` | 872 | Parquet | Unified dual corpus: 438 AST + 434 Line chunks | Unique `chunk_id` enforced |
| `ground_truth_final.csv` | 882 | CSV | Unified ground truth (exhaustive Top-5 pool across 12 runs) | Normalized SHA-256 Hash Locked |
| `disagreements_adjudication.csv`| 17 | CSV | 17 resolved disagreements with technical rationales | Structured rubric templates |
| `retrieval_runs/*.json` | 12 files | JSON | Top-20 retrieval runs for 3 retrieval tiers $\times$ 4 conditions | Deterministic scoring |

### Ground Truth Normalized SHA-256 Hashes:
In accordance with Rule R17 and cross-platform verification (handling Windows `\r\n` vs. Linux `\n`), the ground truth dataset is cryptographically locked on normalized newlines:
* `ground_truth_final.csv` (Entire file): `0799ABAFA394B706457638EB75F3D26AF0B40E2D2673465285A5D1BDDFDD754C`
* Ground truth labels (882 labels): `6530C8FEC0D1190A7DB108E647E324EBFF3F0E6FB0A64814726D0F6049D904B6`
*(Automatically verified by `tests/test_gold_integrity.py`).*

---

## ⚖️ Ethical Considerations & Anonymity

* **Double-Anonymous Review Compliance:** All personal names, institutional affiliations, and author emails have been removed from repository files and docstrings. Annotators and adjudicators are referenced solely as `annotator_1`, `annotator_2`, and `Lead Adjudicator`.
* **Protocol Deviations & Methodological Transparency:** All procedural adjustments relative to the initial pre-registration protocol (including upgrading pooling to Top-5 across all 12 systems, achieving $\kappa_w = \mathbf{0.9822}$, 100% Judged@5 coverage, and 0 orphaned pairs) are systematically documented with empirical reconciliation logs in [`dataset/PROTOCOL_DEVIATIONS.md`](dataset/PROTOCOL_DEVIATIONS.md).
* **Repository Provenance & Third-Party Code:** The chunk corpora (`dataset/chunk_corpus.*`) contain code snippets extracted from 40 public GitHub repositories (20 Java, 20 TypeScript/React), fully listed with commit SHAs and repository URLs in [`dataset/manifest_repos.csv`](dataset/manifest_repos.csv) and documented in [`PROVENANCE.md`](PROVENANCE.md).
  * *Licensing Breakdown:* 19 repositories are licensed under MIT; 4 under Apache-2.0; 1 under AGPL-3.0; and 16 repositories are publicly hosted without an explicit root license file.
  * *Third-Party Code Usage:* These code snippets are included solely for academic benchmarking and replication in this peer-reviewed study. Original copyrights remain with their respective upstream authors.
* **Data Availability:** All raw and processed artifacts are provided in open, non-proprietary formats (`.csv`, `.json`, `.parquet`, `.tex`).

---

## 📜 License

* **Replication Package Scripts & Infrastructure:** The evaluation and analysis scripts authored for this study are released under the [MIT License](LICENSE).
* **Curated Annotations & Research Metadata:** The job descriptions, human relevance judgments, and evaluation logs authored for this study are made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).
* **Third-Party Code Snippets:** Mined source code snippets retained in `dataset/chunk_corpus.*` remain subject to the original licenses and terms of their respective upstream repositories (see `dataset/manifest_repos.csv` and `PROVENANCE.md`).
