# Analysis Plan Addendum: Exploratory Analyses

**Target Venue:** 34th IEEE International Conference on Software Analysis, Evolution and Reengineering (IEEE SANER 2027)  
**Track:** Early Research Achievements (ERA Track)  
**Governing Document:** `analysis_plan.md` (Frozen at Phase 0)  
**Protocol Status:** Pre-specified Confirmatory (Internal Analysis Plan) + Transparent Exploratory Addendum  

---

## 1. Confirmatory vs. Exploratory Boundary

To prevent post-hoc hypothesis generation (HARKing) and p-hacking, all analyses conducted in this replication package are strictly demarcated into **Confirmatory** (governed by the frozen `analysis_plan.md`) and **Exploratory** (governed by this Addendum):

| Component | Status | Hypothesis / Purpose | Pre-specified Criteria |
|:---|:---:|:---|:---|
| **Family of 4 Retrieval Comparisons (Comp 1–4)** | **Confirmatory** | Evaluate primary 2x2 factorial retrieval effects on NDCG@10 | Paired two-sided Wilcoxon signed-rank test; Holm-Bonferroni correction ($\alpha = 0.05$); 10,000 bootstrap resamples. |
| **Inter-Annotator Agreement (Table 3.1)** | **Confirmatory** | Measure annotation reliability across 882 candidate pairs | Quadratic Weighted Cohen's $\kappa_w$, Linear $\kappa$, Landis & Koch (1977) interpretation. |
| **Corpus Syntax & Morphology (Table 3.2)** | **Confirmatory** | Compare boundary cuts, syntax errors, LOC, and token truncation | Live Tree-sitter parsing and BAAI/bge-m3 tokenizer on identical 189 source files. |
| **Comp 5 (AST+Header vs. Line+Header)** | **Exploratory** | Directly compare top-performing AST against top-performing Line | Paired Wilcoxon signed-rank test; 95% bootstrap CI. Marked strictly exploratory. |
| **AST Subgroup Disaggregation (Pure vs. Fallback)** | **Exploratory** | Unpack morphology differences between AST method chunks and line fallbacks | Transparent audit of 0 fallback chunks (0.0% of AST corpus). |
| **Language Heterogeneity (Java vs. React/TS)** | **Exploratory** | Investigate language-specific retrieval performance | Sub-cohort evaluation across 15 Java JDs and 10 React/TS JDs. |
| **Candidate Pool Random Baseline & Shortlist** | **Exploratory** | Establish permutation baseline within pooled candidates and judged-only sensitivity | 1,000 permutations over pooled candidates ($k=10$, seed=42) and judged-only ranking. |
| **Annotator Robustness Analysis** | **Exploratory** | Assess metric stability across consensus-only and individual annotators | Re-scoring 12 retrieval runs on consensus pairs ($N=865$), Annotator 1, and Annotator 2. |
| **Downstream Generation Quality (RAGAs)** | **Exploratory** | Evaluate faithfulness and relevance on Top-3 retrieved contexts | LLM-as-a-judge with deterministic offline cache. |

---

## 2. Confirmatory Protocol Summary (Frozen)

As specified in `analysis_plan.md`:
1. **Primary Metric:** Normalized Discounted Cumulative Gain at rank 10 ($\text{NDCG@10}$).
2. **Confirmatory Family of 4 Tests:**
   - **Comp 1 (Full Proposed vs. Baseline Line without Header):** `rerank_ast_with_header` vs. `rerank_line_no_header` ($\Delta = -0.0930$, 95% Bootstrap CI $[-0.1636, -0.0228]$, Wilcoxon $W = 78.0$, $p_{\text{raw}} = 0.0219$, $p_{\text{Holm}} = 0.0438 < 0.05$, Statistically significant).
   - **Comp 2 (Pure Chunking Effect without Headers):** `rerank_ast_no_header` vs. `rerank_line_no_header` ($\Delta = -0.2390$, 95% Bootstrap CI $[-0.2919, -0.1847]$, Wilcoxon $W = 3.0$, $p_{\text{raw}} < 0.0001$, $p_{\text{Holm}} = 0.0000 < 0.001$, Statistically significant penalty for isolated AST snippets).
   - **Comp 3 (Header Effect on AST):** `rerank_ast_with_header` vs. `rerank_ast_no_header` ($\Delta = +0.1459$, 95% Bootstrap CI $[+0.0862, +0.2071]$, Wilcoxon $W = 24.0$, $p_{\text{raw}} < 0.0001$, $p_{\text{Holm}} = 0.0001 < 0.001$, Statistically significant gain of progressive context headers).
   - **Comp 4 (Header Effect on Line):** `rerank_line_with_header` vs. `rerank_line_no_header` ($\Delta = +0.0409$, 95% Bootstrap CI $[-0.0122, +0.0928]$, Wilcoxon $W = 118.0$, $p_{\text{raw}} = 0.2411$, $p_{\text{Holm}} = 0.2411 \ge 0.05$, Not statistically significant).

---

## 3. Exploratory Analysis Details & Results

### 3.1 Comparison 5: AST+Header vs. Line+Header (Exploratory Top-Performers)
- **Empirical Numbers:**
  - $\text{NDCG@10}(\text{AST+Header}) = 0.4866$
  - $\text{NDCG@10}(\text{Line+Header}) = 0.6206$
  - $\Delta = -0.1339$
  - 95% Bootstrap Confidence Interval (10,000 resamples): $[-0.1992, -0.0704]$ (Strictly negative)
  - Wilcoxon signed-rank test: $W = 36.0, p_{\text{raw}} = 0.0003 < 0.05$ (Statistically significant)
  - Query-level wins: Line+Header wins on 20 queries, AST+Header wins on 5 queries, 0 ties.
- **Scientific Takeaway:** The line baseline with progressive headers is the strongest overall reranking retrieval configuration ($\text{NDCG@10} = 0.6206, \text{NDCG@1} = 0.800, \text{MRR@10} = 0.950, \text{CP@5} = 0.933$). AST+Header ($\text{NDCG@10} = 0.4866, \text{NDCG@1} = 0.740, \text{MRR@10} = 0.896, \text{CP@5} = 0.869$) does not outperform Line+Header on aggregate retrieval metrics. However, progressive headers rescue AST from severe context deprivation ($\text{NDCG@10}$ jumps from $0.3407$ to $0.4866, \Delta = +0.1459, p = 0.0001$), while producing chunks with 87.2% boundary intactness and $78.1\%$ valid syntax in pure AST mode.

### 3.2 AST Subgroup Disaggregation
- **AST Pure (N=438, 100.0%):**
  - Boundary intact: **87.2%** (56 cuts across outer wrappers)
  - Parse error-free rate: **78.1%** (96 syntax errors)
  - Header retention: **100.0%** (438/438 include class and method context)
  - Truncation rate (> 512 tokens): **15.8%** (69 chunks)
  - Mean LOC: **26.7 ± 23.3** (median: 16.0) | Mean content tokens: **250.8 ± 211.7**
- **AST Fallback (N=0, 0.0%):**
  - Trigger: With upgraded Tree-sitter parsers supporting interfaces, records, React functional components, and hooks, the fallback rate is strictly **0.0% (0/438)**.
- **Full Fallback Audit:** Documented in `dataset/fallback_audit.csv`.

### 3.3 Language Heterogeneity (Java vs. React/TypeScript)
- **Java Cohort (15 JDs):**
  - Re-ranking NDCG@10: AST+Header = **0.5830** | Line+Header = **0.7158** ($\Delta = -0.1328$).
- **React/TypeScript Cohort (10 JDs):**
  - Re-ranking NDCG@10: AST+Header = **0.3420** | Line+Header = **0.4777** ($\Delta = -0.1357$).
- **Takeaway:** Line+Header consistently leads AST+Header across both language cohorts by approximately $\Delta \approx -0.133$.

### 3.4 Candidate Pool Random Baseline vs. Judged-Only Shortlist Evaluation
- **Candidate Pool Random Baseline:**
  - Methodology: 1,000 permutations randomly shuffling candidate items within each JD's pool, computing NDCG@10 against final ground truth.
  - Result: Mean $\text{NDCG@10} = \mathbf{0.562} \pm 0.146$.
  - Interpretation: The pool baseline evaluates strictly within the pre-screened candidate set where 100% of items were judged (0% unjudged). On the entire 872-chunk corpus, a random baseline achieves only $\text{NDCG@10} = \mathbf{0.0224}$.
- **Judged-Only (Shortlist) Sensitivity Analysis:**
  - Evaluates retrieval models strictly on judged candidate pool items:
    - `rerank_line_with_header`: **0.7243** (exceeds the 0.562 pool baseline)
    - `rerank_line_no_header`: **0.6696**
    - `rerank_ast_with_header`: **0.5774**
    - `rerank_ast_no_header`: **0.4156**
  - Takeaway: Ranking orders remain identical between full-corpus evaluation and judged-only shortlist evaluation.

### 3.5 Annotator Sensitivity
- **Consensus-only Subset ($N=865$ non-disputed pairs):**
  - Re-ranking NDCG@10 scores: Line+Header (**0.6175**) > Line No-Header (**0.5713**) > AST+Header (**0.4866**) > AST No-Header (**0.3338**).
  - Header effect on AST under consensus-only qrels: $\Delta = +0.1528$, paired Wilcoxon $p < 0.001$ (Confirmed Significant).
- **Annotator 1 Only ($N=882$):**
  - AST+Header = **0.4860** | AST No-Header = **0.3414** ($\Delta = +0.1446$, 95\% CI $[0.0863, 0.2040]$, $p < 0.001$).
- **Annotator 2 Only ($N=882$):**
  - AST+Header = **0.4861** | AST No-Header = **0.3383** ($\Delta = +0.1477$, 95\% CI $[0.0876, 0.2093]$, $p < 0.001$).

### 3.6 Statistical Power Analysis & Minimum Detectable Effect (MDE)
- **Status:** Exploratory (Paired $t$-test approximation, $N=25$ JDs, 80\% statistical power).
- **Rationale:** With $N=25$ queries, small paired differences may not achieve statistical significance. MDE establishes the minimum true difference required to achieve 80\% power under two-sided paired testing at $\alpha = 0.05$ (unadjusted) and $\alpha = 0.0125$ (Bonferroni-adjusted for 4 confirmatory tests).
- **Results across 5 comparisons:**
  - Comp 1 (AST+Header vs. Line No-Header): $\text{SD}_{\Delta} = 0.1838$, $\text{MDE}(\alpha=0.05) = 0.1068$, $\text{MDE}(\alpha=0.0125) = 0.1302$
  - Comp 2 (AST No-Header vs. Line No-Header): $\text{SD}_{\Delta} = 0.1395$, $\text{MDE}(\alpha=0.05) = 0.0811$, $\text{MDE}(\alpha=0.0125) = 0.0988$
  - Comp 3 (AST+Header vs. AST No-Header): $\text{SD}_{\Delta} = 0.1571$, $\text{MDE}(\alpha=0.05) = 0.0913$, $\text{MDE}(\alpha=0.0125) = 0.1113$
  - Comp 4 (Line+Header vs. Line No-Header): $\text{SD}_{\Delta} = 0.1368$, $\text{MDE}(\alpha=0.05) = 0.0795$, $\text{MDE}(\alpha=0.0125) = 0.0969$
  - Comp 5 (AST+Header vs. Line+Header): $\text{SD}_{\Delta} = 0.1689$, $\text{MDE}(\alpha=0.05) = 0.0981$, $\text{MDE}(\alpha=0.0125) = 0.1196$
- **Overall Range:** 0.080 to 0.130 NDCG@10.

### 3.7 Retrievability & Reachable Qrels Audit
- **Status:** Verified.
- **Results:** All 882 candidate pairs (100.0%) map directly to coordinates in `dataset/chunk_corpus.parquet`. There are exactly 0 orphaned or unretrievable pairs in the benchmark.

### 3.8 Binary Preference (Bpref) Analysis
- **Status:** Exploratory secondary evaluation metric (Buckley & Voorhees, SIGIR 2004).
- **Rationale:** Evaluates retrieval effectiveness independently of unjudged pooling penalties.
- **Results:**
  - BM25 tier: bpref ranges from **0.129** to **0.255** across all four configurations.
  - Dense tier: bpref ranges from **0.179** to **0.347**.
  - Rerank tier: bpref ranges from **0.170** to **0.353** (Line+Header 0.353 > Line No-Header 0.300 > AST+Header 0.241 > AST No-Header 0.170).
- **Execution:** `python scripts/compute_bpref.py` saves results to `dataset/benchmark_results/bpref_summary.csv` and `bpref_report.json`.

### 3.9 Pure AST Analysis
- **Status:** Confirmed with 0.0% fallback rate in the active 438-chunk AST corpus.
- **Takeaway:** Upgraded Tree-sitter parsers successfully capture 100% of chunks as syntactic units without needing line-window fallback.

### 3.10 Closed-Pool Header Component Ablation Framework
- **Status:** Exploratory ablation for RQ3.
- **Implementation:** `scripts/header_ablation.py` evaluates header field contributions across all 882 judged candidate pairs.

---

## 4. Reproducibility Instructions

All exploratory analyses can be dynamically executed and verified via:
```bash
python scripts/evaluate_sensitivity.py
python scripts/power_analysis.py
python scripts/compute_bpref.py
python -m pytest tests/test_pure_ast.py
```
Outputs are saved to `dataset/benchmark_results/sensitivity_analysis_report.txt`, `dataset/benchmark_results/power_analysis.json`, and `dataset/benchmark_results/bpref_report.txt`.
