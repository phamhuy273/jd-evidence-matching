# Pre-Registration Analysis Plan: AST-RAG (Path B) for IEEE SANER 2027 ERA

> **Document Version:** 1.0  
> **Date:** October 9, 2026  
> **Status:** FROZEN (Pre-registered before downstream retrieval & evaluation)  
> **Governing Standards:** R1–R46 Anti-Hardcoding & Experimental Integrity Rules  

---

## 1. Research Objectives & 2×2 Factorial Design

This empirical study investigates the impact of source code chunking strategies and contextual enrichment within Retrieval-Augmented Generation (RAG) pipelines for software developer skill evidence matching.

We employ a **2×2 Factorial Design**:
$$\text{Chunking Strategy } \{\text{AST Method-level}, \text{Line-based (50 LOC)}\} \times \text{Context Header } \{\text{Without Header}, \text{With Header}\}$$

Evaluated across three retrieval tiers:
1. **Lexical Sparse:** BM25Okapi
2. **Dense Bi-Encoder:** BAAI/bge-m3 (1024-dim, `max_seq_length = 512`)
3. **Two-Stage Cross-Encoder:** BAAI/bge-reranker-base (re-ranking top-20 candidates from Dense)

---

## 2. Hypotheses & Primary Comparison Family (Family of 4)

All confirmatory statistical tests are restricted to the primary metric at the Two-Stage Re-ranking tier. To strictly control the Family-Wise Error Rate (FWER), we apply the **Holm-Bonferroni step-down correction** across this family of 4 pre-specified orthogonal/confirmatory comparisons:

1. **Comparison 1 (Full Proposed vs. Full Baseline):**
   $$\text{AST + Header + Rerank} \quad \text{vs.} \quad \text{Line + Rerank}$$
   *Research Question:* Does syntax-aware progressive disclosure outperform traditional line slicing in end-to-end evidence re-ranking?

2. **Comparison 2 (Pure Chunking Effect):**
   $$\text{AST No-Header + Rerank} \quad \text{vs.} \quad \text{Line No-Header + Rerank}$$
   *Research Question:* When context headers are omitted from both, does AST syntactic boundary preservation improve re-ranking quality over fixed-window line slicing?

3. **Comparison 3 (Header Effect on AST):**
   $$\text{AST + Header + Rerank} \quad \text{vs.} \quad \text{AST No-Header + Rerank}$$
   *Research Question:* What is the isolated marginal contribution of hierarchical progressive context headers when added to AST method bodies?

4. **Comparison 4 (Header Effect on Line):**
   $$\text{Line + Header + Rerank} \quad \text{vs.} \quad \text{Line No-Header + Rerank}$$
   *Research Question:* Does adding file and line coordinate headers to line chunks improve re-ranking quality?

---

## 3. Metric Hierarchy

### 3.1. Primary Confirmation Metric (Pre-registered)
* **Primary Metric:** $\mathbf{NDCG@10}$ at the **Stage 2 Cross-Encoder Re-ranking tier** (dense top-20 re-ranked by `BAAI/bge-reranker-base`).
* *Rationale:* In real-world candidate auditing, technical recruiters inspect a pool of evidence (top-10 items). Graded relevance ($\text{rel} \in \{0, 1, 2\}$) at rank 10 comprehensively measures ranking precision and relevance discrimination.
* Any claim of statistical superiority or "outperformance" in the paper must be supported by a Holm-adjusted $p < 0.05$ on this primary metric with a 95% bootstrap confidence interval excluding zero (R24).

### 3.2. Secondary & Exploratory Metrics (Pre-registered)
All other metrics and tiers are explicitly designated as **secondary / exploratory**:
* **Secondary Ranking Metrics:** $NDCG@1$, $NDCG@3$, $NDCG@5$, $MRR@10$.
* **Set-Retrieval Metrics:** $P@5$, $Recall@10$ (measured against known relevant items in the pooled ground truth).
* **Coverage Metric:** $judged@k$ (the proportion of retrieved items in top-$k$ that received human annotation, R20).
* **Random Baseline:** Empirical average of 1,000 permutations over:
  1. Random ranking over the full corpus universe.
  2. Random ranking within the pooled annotated set.

---

## 4. Statistical Testing Protocol (R22, R23, R24)

1. **Test Type:** Two-sided Paired Wilcoxon Signed-Rank Test (`scipy.stats.wilcoxon(alternative='two-sided')`).
2. **Confidence Intervals:** 95% Bootstrap Confidence Interval on mean paired differences across the 25 Job Descriptions (10,000 resamples, fixed random seed `42`).
3. **Reporting Standards:**
   * Report exact mean difference: $\Delta = \text{Mean}(A) - \text{Mean}(B)$.
   * Report 95% Bootstrap CI: $[\text{CI}_{\text{lower}}, \text{CI}_{\text{upper}}]$.
   * Report exact unadjusted $p$-value ($p_{\text{raw}}$) and Holm-adjusted $p$-value ($p_{\text{Holm}}$).
   * Report Win / Loss / Tie counts across all 25 queries.
   * If $p_{\text{Holm}} \ge 0.05$ or CI spans 0, the result must be explicitly declared **not statistically significant** (R24).
   * If ties dominate (e.g. ceiling effect), Wilcoxon cannot claim positive significance; it must be reported as a ceiling limitation (R25).

---

## 5. Unjudged Items & Sensitivity Analyses (Pre-specified)

1. **Unjudged Items Handling (R20):**
   * Primary Analysis: Unjudged items are assigned relevance score = 0.
   * Reporting: The proportion of judged candidates at top-k ($judged@k$) is reported for every system.
2. **Sensitivity Analyses (Exploratory):**
   * **Sensitivity 1 (Overlapping Line Windows):** Primary analysis retains all retrieved line windows. Sensitivity analysis de-duplicates overlapping line windows by retaining only the highest-ranked window per file region.
   * **Sensitivity 2 (Query Formulation):** Primary query uses structured `"Title / Domain / Mandatory Skills"`. Sensitivity analysis evaluates raw Job Description text.
   * **Sensitivity 3 (Pool Bias Check):** Sequentially leave out contributions unique to each system and recompute rankings.
   * **Sensitivity 4 (Chunk Size Confounder):** Analyze subset of chunks matched on token length to isolate boundary effects from chunk length effects.
   * **Sensitivity 5 (Legacy Comparison):** Re-evaluate the original 250 Gold pairs strictly as a "Header Ablation on 10 Pre-selected Candidates" with correct two-sided statistical tests.
