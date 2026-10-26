# Annotation Guidelines: JD-to-Code Evidence Matching

**Target Study:** IEEE SANER 2027 ERA Track — *Code-to-JD Evidence Retrieval Benchmark*  
**Evaluation Protocol:** 2×2 Factorial Design (AST vs. Line × With Header vs. Without Header)  
**Governance:** Compliance with Anti-Hardcoding & Experimental Integrity Rules (R17, R18, R19, R44).

---

## 1. Objective and Scope

The objective of this annotation task is to evaluate whether a retrieved source code snippet provides genuine, observable evidence of the technical competencies requested by a Job Description (JD). 

Candidates frequently list keywords (e.g., "Spring Boot", "TypeScript", "React hooks") on resumes without verified proof. In our automated evaluation pipeline, a retrieval system surfaces candidate code snippets from candidate GitHub repositories. Human annotators determine whether each code snippet serves as valid evidence for the specific job profile.

---

## 2. Neutral Presentation and Blinding Protocol

To eliminate bias, the annotation interface conforms strictly to **Rules R18 and R19**:
1. **Blind Evaluation:** Annotators are never shown:
   - The retrieval system or condition that proposed the snippet (e.g., AST vs. Line-based window).
   - Any retrieval score, similarity measure, or rank.
   - Any LLM-generated summary, synthetic header, or automated prediction.
2. **Neutral Format:** Every snippet is presented identically:
   - **JD ID & Job Title & Domain**
   - **Mandatory Skills**
   - **Repository Name & File Path**
   - **Line Boundaries (Start Line – End Line)**
   - **Raw Source Code Snippet** (exactly as found in the repository file).

---

## 3. Three-Point Relevance Scale (0, 1, 2)

Each (JD, Code Snippet) pair is assigned an integer label in `{0, 1, 2}`.

### Level 0: Irrelevant / No Evidence / Insufficient Context
* **Definition:** The code snippet does not demonstrate the required technical skills, is completely off-topic, or consists solely of uninformative boilerplate that any template generates without demonstrating developer competency.
* **Operational Criteria:**
  - Code belongs to an unrelated tech stack or domain (e.g., standard SQL script or plain HTML when evaluating React component state).
  - Trivial boilerplate with zero business logic: empty class declarations, default constructors, Lombok `@Data` models without custom logic, trivial single-line getters/setters.
  - Basic presentation wrappers or UI mockups that do not demonstrate actual logic (e.g., a static `WelcomeToast` or simple UI string formatting).
* **Examples from Corpus & Disagreement Adjudication:**
  - `PAIR_155`: `WelcomeToast` component returning static toast message without state, event handling, or TypeScript typing $\rightarrow$ **Label 0**.
  - `PAIR_151`: Static `Carousel` wrapper without dynamic data binding or interaction handling $\rightarrow$ **Label 0**.
  - A 50-line window cutting across import statements and an interface definition with no methods $\rightarrow$ **Label 0**.

---

### Level 1: Partially Relevant / Weak or Peripheral Evidence
* **Definition:** The snippet demonstrates use of the target technology or language, but represents peripheral, superficial, or incomplete implementation evidence.
* **Operational Criteria:**
  - Standard CRUD controller delegation that contains no validation, exception handling, or business logic (e.g., single-line pass-through: `return repository.findById(id)`).
  - Incidental configuration beans that require only standard library invocations without demonstrating domain mastery (e.g., basic `defineOpenApi` Swagger bean).
  - Auditing/listener helpers (e.g., `getCurrentAuditor` in Spring Data JPA) that provide indirect proof of Spring usage but do not demonstrate core backend architecture.
  - A React component rendering UI with basic `useState` but without typing, custom hooks, or asynchronous state management when the JD demands Senior/Full-stack competence.
* **Examples from Corpus & Disagreement Adjudication:**
  - `PAIR_078`: `getCurrentAuditor()` returning `SecurityContextHolder.getContext().getAuthentication().getName()` $\rightarrow$ **Label 1** (demonstrates Spring Security awareness, but only 3 lines of peripheral boilerplate).
  - `PAIR_131`: `defineOpenApi()` configuration bean setting title and description $\rightarrow$ **Label 1** (valid OpenAPI setup, but lacks domain implementation depth).
  - Line window capturing a standard React component signature and partial JSX without showing the state management hook $\rightarrow$ **Label 1**.

---

### Level 2: Highly Relevant / Strong, Direct Evidence
* **Definition:** The snippet provides concrete, substantial, and non-trivial evidence demonstrating direct competence in the core technical skills specified by the JD.
* **Operational Criteria:**
  - Production-grade backend endpoints with authentication, validation, exception handling, and service interaction (e.g., `@RestController` with `@PreAuthorize`, DTO validation, and error envelopes).
  - Complex configuration demonstrating architectural expertise (e.g., complete `SecurityFilterChain` configuring JWT filters, CORS/CSRF policies, session management, and method-level security).
  - Non-trivial business logic implementation: custom service algorithms, transactional boundaries (`@Transactional`), caching strategies, complex database querying with JPA Criteria or custom queries.
  - In React/TypeScript: custom hooks (`useAuth`, `useFetch`, `useDebounce`), state management reducers, typed API consumers with error boundaries and loading states.
* **Examples from Corpus & Disagreement Adjudication:**
  - `PAIR_010` & `PAIR_132`: `securityFilterChain(HttpSecurity http)` configuring JWT authentication filters, stateless session creation, and RBAC endpoint rules $\rightarrow$ **Label 2** (directly proves Spring Security / JWT requirement).
  - `PAIR_005`: `AccountController.me()` endpoint utilizing `@AuthenticationPrincipal`, returning current user DTO with explicit HTTP status and documentation $\rightarrow$ **Label 2**.
  - `PAIR_086`: `register()` auth endpoint validating registration payload, hashing password via BCrypt, and returning JWT token $\rightarrow$ **Label 2**.

---

## 4. Annotation Workflow and Quality Assurance

1. **Independent Dual Annotation:**
   - Two qualified annotators (Annotator 1 and Annotator 2) independently evaluate each row in `to_label.csv` without consulting each other.
   - Annotators record their integer score (`0`, `1`, or `2`) in their designated column.
2. **Inter-Annotator Agreement Calculation:**
   - Agreement is evaluated using Cohen's Quadratic Weighted Kappa ($\kappa_w$):
     $$\kappa_w = 1 - \frac{\sum w_{ij} O_{ij}}{\sum w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(k - 1)^2}$$
   - The target threshold is $\kappa_w \ge 0.80$ (demonstrating strong inter-rater reliability).
3. **Consensus Adjudication of Disagreements:**
   - For all items where $label_1 \neq label_2$, an adjudication session is conducted.
   - Every resolved item MUST have:
     - The identity of the adjudicator (e.g., `Adjudicator: Lead Adjudicator`).
     - A concrete, factual technical explanation detailing *why* the snippet satisfies or fails the requirement. Template/boilerplate explanations are strictly prohibited.
4. **Header Bias Verification (Neutral Re-annotation of Gold):**
   - A random subset of 50 Gold items from `ground_truth_final.csv` is blinded and presented in the neutral format.
   - The concordance rate between the blinded annotation and original gold labels is computed to detect and report any header-induced scoring bias.
