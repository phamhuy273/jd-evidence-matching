# Data & Artifact Provenance

**Target Study:** IEEE SANER 2027 ERA Track — *An Empirical Evaluation of Source Code Chunking and Context Headers for Requirement Evidence Retrieval*  
**Governing Documents:** `analysis_plan.md` (Frozen at Phase 0) & `analysis_plan_addendum.md`  
**Review Mode:** Double-Anonymous Peer Review  

---

## 1. Overview and Scope

This document provides a transparent and rigorous audit trail for all data sources, curation steps, licensing classifications, human annotations, adjudication procedures, and benchmark artifacts included in this replication package.

To guarantee permanent, deterministic reproducibility and eliminate external network dependencies, this package provides the complete, pre-curated corpora, frozen queries, precomputed retrieval runs, and cryptographically verified ground truth files.

---

## 2. Source Repositories Universe

The candidate code universe is constructed from **40 public GitHub repositories** across two major software engineering ecosystems:
* **Java Backend Ecosystem (20 repositories):** Enterprise frameworks including Spring Boot, Spring Security, Hibernate/JPA, Kafka integrations, and microservice architectures.
* **TypeScript / React Frontend Ecosystem (20 repositories):** Modern web applications featuring functional React components, custom hooks, Redux/Context state management, Next.js, and TypeScript typing contracts.

All repositories are pinned to specific Git commit SHAs in [`dataset/manifest_repos.csv`](dataset/manifest_repos.csv) to guarantee temporal immutability:
* **Total files mined:** 189 source files (68 Java files, 121 TypeScript/React files).
* **Interval registry:** Pinned line-level intervals for all 189 files are stored in [`dataset/source_file_intervals.json`](dataset/source_file_intervals.json).

### Licensing Audit and Research Usage

A granular audit of repository licenses in `manifest_repos.csv` reveals the following distribution:
* **MIT License (19 repositories):** Permissive open-source license.
* **Apache-2.0 License (4 repositories):** Permissive open-source license with patent grant.
* **AGPL-3.0 License (1 repository):** Strong copyleft license (`anoma/namada-interface`).
* **Not Specified / Public GitHub (16 repositories):** Publicly accessible repositories on GitHub without an explicit root license file.

**Repository Distribution & Research Usage:**  
Source code snippets mined from these 40 repositories are included in `dataset/chunk_corpus.*` solely for academic research, empirical software analysis, and benchmark replication in this peer-reviewed study. Original copyrights remain with their respective upstream authors.

---

## 3. Query Universe (Job Descriptions)

The query set consists of **25 frozen industry Job Descriptions (JDs)** documented in [`dataset/queries_frozen.json`](dataset/queries_frozen.json):
* **15 Java Backend JDs (`JD_JAVA_01` – `JD_JAVA_15`):** Cover competencies such as Spring Boot MVC/WebFlux, Spring Security (JWT, OAuth2, RBAC), JPA/Hibernate transactions, distributed messaging (Kafka/RabbitMQ), and microservice resilience.
* **10 React/TypeScript JDs (`JD_REACT_01` – `JD_REACT_10`):** Cover competencies including React 18 functional components, custom hooks (`useAuth`, `useFetch`), asynchronous state management, TypeScript type narrowing, and UI performance optimization.
* **Seniority Stratification:** Balanced across Junior, Mid-Level, and Senior roles.

---

## 4. Chunk Corpora Architecture

The study evaluates two chunking paradigms across identical source files, comprising a unified corpus of **872 chunks** stored in [`dataset/chunk_corpus.parquet`](dataset/chunk_corpus.parquet):

### 4.1 AST-First Progressive Disclosure Corpus (438 Chunks)
* **Strategy:** Extracts semantic function/method/constructor nodes via Tree-sitter AST parsers with progressive context headers (`// Class: ...`, `// Method: ...`).
* **Pure AST Chunks (N=438, 100.0%):**
  * Java: 152 chunks.
  * React/TS: 286 chunks.
  * Syntax intact rate: **78.1%** | Boundary intact rate: **87.2%** (56 cuts) | Header retention: **100.0%**.
  * Truncation rate (> 512 tokens): **15.8%** (69 chunks).
  * Mean LOC: **26.7 ± 23.3** (median: 16.0) | Mean content tokens: **250.8 ± 211.7**.
* **Line-Fallback Chunks (N=0, 0.0%):**
  * With upgraded Tree-sitter grammars capturing top-level interfaces, records, React functional components, custom hooks, and wrappers, the fallback rate is **0.0% (0/438 chunks)**.
  * Fully audited in [`dataset/fallback_audit.csv`](dataset/fallback_audit.csv).

### 4.2 Fixed-Window Line Baseline Corpus (434 Chunks)
* **Strategy:** Standard 50-LOC sliding window with 10-LOC overlap (step 40 LOC, minimum 6 LOC).
* **Characteristics:** 120 Java chunks, 314 React/TS chunks.
* **Syntax degradation:** Boundary cuts in **56.0%** (243 cuts, 44.0% intact), syntax errors in **77.6%** (337 chunks, 22.4% valid syntax), 0% context header retention, mean LOC **41.0 ± 13.1**, mean tokens **390.8**, truncation rate (> 512 tokens): **28.6%** (124 chunks).

---

## 5. Candidate Pooling and Ground Truth Annotations

### 5.1 Candidate Pooling Strategy & Reconciliation
To construct an unbiased ground truth without favoring any single retrieval paradigm:
* **Candidate Pool Composition:** Candidates were pooled using a single-wave Cranfield/TREC protocol across **all 12 retrieval configurations** (3 tiers: BM25, Dense with BAAI/bge-m3, and Re-ranking with BAAI/bge-reranker-base $\times$ 4 factorial chunking conditions).
* **Exhaustive Top-5 Pooling:** Top-5 candidates per query from all 12 systems ($25 \text{ JDs} \times 12 \text{ configurations} \times 5 = 1,500 \text{ slots}$) were pooled and deduplicated by coordinate intervals `(jd_id, file_path, start_line, end_line)`.
* **Pool Derivation:**
  * Deduplication yields exactly **882 unique candidate pairs** (543 Java pairs, 339 React/TS pairs).
  * Top-3 Judged Coverage: **100.0%** across all 12 configurations.
  * Top-5 Judged Coverage: **100.0%** across all 12 configurations (zero unjudged items in Top-5 for any system).
  * Complete reconciliation is documented in `dataset/benchmark_results/pool_reconciliation_report.txt` and verified by `scripts/reconcile_pool.py`.

### 5.2 Blinding & Annotation Protocol
* **Blinding (Rules R18, R19):** Human annotators reviewed candidate pairs via [`dataset/to_label.csv`](dataset/to_label.csv). The annotation sheet presented only neutral metadata and raw code content; all system identifiers, similarity scores, ranks, and chunking strategy tags were completely removed.
* **Independent Dual Annotation:**
  * **Annotator 1:** Evaluated all 882 pairs using the 3-point ordinal scale (`0`: Irrelevant, `1`: Partially Relevant, `2`: Highly Relevant) following rubric criteria directly without free-form text notes (recorded in `dataset/ground_truth_annotator1.csv`).
  * **Annotator 2:** Evaluated all 882 pairs independently, assigning ordinal scores and contributing diagnostic technical notes (recorded in `dataset/ground_truth_annotator2.csv`).
* **Inter-Annotator Agreement (Table 3.1 & Figure 3.1):**
  * Exact agreement: **98.07%** (865/882 pairs).
  * Minor 1-point disagreements: **1.81%** (16/882 pairs).
  * Severe 2-point disagreements: **0.11%** (1/882 pairs).
  * **Quadratic Weighted Cohen's Kappa:** $\kappa_w = \mathbf{0.9822}$ (*Almost Perfect Agreement*, Landis & Koch 1977 [>0.80]).
  * Linear Weighted Kappa: $\kappa = \mathbf{0.9764}$.
  * Unweighted Cohen's Kappa: $\kappa = \mathbf{0.9708}$.
  * Java Subgroup Kappa: $\kappa_w = \mathbf{0.9805}$ ($N=543$, 98.16% exact agreement).
  * React/TS Subgroup Kappa: $\kappa_w = \mathbf{0.9789}$ ($N=339$, 97.94% exact agreement).

### 5.3 Consensus Adjudication
* Disagreements occurred on **17 pairs** ($label_1 \neq label_2$).
* An adjudication session was conducted by a single **Lead Adjudicator**, resolving every disagreement based on the technical rubric.
* **Standardized Diagnostic Structure:** As documented in `dataset/benchmark_results/adjudication_summary.json`, adjudication rationales follow a structured template across formatting conventions. All 17 adjudicated items are preserved in `dataset/disagreements_adjudication.csv`.
* The adjudicated labels were combined with unanimous agreement pairs into [`dataset/ground_truth_final.csv`](dataset/ground_truth_final.csv).

### 5.4 Protocol Deviations
* All procedural adjustments relative to the initial experimental plan are systematically cataloged and audited in [`dataset/PROTOCOL_DEVIATIONS.md`](dataset/PROTOCOL_DEVIATIONS.md).

### 5.5 Cryptographic Hash Locking
To prevent post-hoc label drift, ground truth files are locked on normalized newlines (Rule R17):
* `ground_truth_final.csv` (SHA-256): `0799ABAFA394B706457638EB75F3D26AF0B40E2D2673465285A5D1BDDFDD754C`
* Ground truth labels (882 labels): `6530C8FEC0D1190A7DB108E647E324EBFF3F0E6FB0A64814726D0F6049D904B6`
*(Verified dynamically by `tests/test_gold_integrity.py` and `tests/test_doc_hashes.py`).*

### 5.6 Retrievability & Reachable Qrels Audit
When mapping the 882 ground-truth pairs to the active `chunk_corpus.parquet` by line coordinates `(repo_name, file_path, start_line, end_line)`:
* All 882 pairs (**100.0%**) match current corpus chunks.
* Exactly **0 orphaned pairs** exist in the benchmark.
* Enforced by unit test in `tests/test_gold_integrity.py`.

---

## 6. Precomputed Retrieval Runs
 
* **Retrieval Runs (`dataset/retrieval_runs/*.json`):** 12 deterministic JSON ranking files containing Top-20 retrieved candidate lists across 3 retrieval tiers (BM25 with Rank-BM25, Dense with BAAI/bge-m3, and Re-ranking with BAAI/bge-reranker-base) $\times$ 4 factorial conditions.

---

## 7. Reproduction Pipeline Boundaries

This replication package is architected to reproduce and verify all empirical findings, tables, figures, sensitivity analyses, and statistical hypothesis tests from the frozen curated artifacts:
* **Included:** Live Tree-sitter AST parsing, syntax morphology analysis, full 2×2 factorial retrieval evaluation, bootstrap confidence intervals, Wilcoxon signed-rank tests, Holm-Bonferroni corrections, subgroup sensitivity analysis, and automated pytest integrity test suite.
* **Excluded by Design:** The initial raw web scraping of 40 GitHub repositories and the iterative interactive pooling workflow are excluded because they rely on external network services and live GitHub APIs that are subject to upstream repository changes, rate limits, and network volatility. Providing the frozen curated corpora and ground truth ensures permanent, deterministic, offline reproducibility for conference reviewers.
