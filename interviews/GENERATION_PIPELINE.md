# StaffML: Autonomous Curriculum Generation Engine

This document outlines the architectural and theoretical blueprint for the StaffML automated question generation pipeline. It moves the platform from heuristic "prompt engineering" to a deterministic, programmatic graph grounded in psychometrics and modern MLOps.

---

## 1. Theoretical Foundations (The Literature)

To ensure the generated content is scientifically rigorous and pedagogically sound, the pipeline is modeled on the following frameworks:

### A. Item Response Theory (IRT) & The Rasch Model
*   **Literature:** Lord, F. M. (1980). *Applications of item response theory to practical testing problems.*
*   **Implementation:** Questions are not statically bound to a difficulty level. We implement a dynamic ELO rating system. If a user with a "Staff" (L6) rating consistently fails an "L3" question, the system flags the question for auto-deprecation or semantic review, assuming it is either poorly worded or misclassified.

### B. Cognitive Apprenticeship & Faded Scaffolding
*   **Literature:** Collins, Brown, & Newman (1989). *Cognitive Apprenticeship.*
*   **Implementation:** The UI and question generation strictly follow a scaffolding fade.
    *   **L1/L2:** Maximum scaffolding (Multiple Choice Engine). Limits search space, teaches vocabulary.
    *   **L3:** Partial scaffolding (Architecture Debugger). Tests spatial reasoning.
    *   **L4-L6:** Zero scaffolding (Blank Terminal + Chaos Monkey). Tests synthesis and applied physics.

### C. Bayesian Knowledge Tracing (BKT)
*   **Literature:** Corbett & Anderson (1994). *Knowledge tracing: Modeling the acquisition of procedural knowledge.*
*   **Implementation:** The platform uses a Hidden Markov Model to calculate the probability that a user has mastered a specific node in the semantic graph (e.g., `KV-Cache Management`), rather than relying on raw completion counts.

---

## 2. Pipeline Architecture (The Directed Graph)

The generation pipeline is a 4-stage automated Directed Acyclic Graph (DAG), executing autonomously via cron jobs.

### Stage 1: Knowledge Extraction & Gap Detection
1. **Ingestion:** Parse `.qmd` files (Volume 1 & 2) and `NUMBERS.md` into semantic chunks.
2. **Density Clustering:** Embed all existing questions into a vector space (e.g., `all-MiniLM-L6-v2`).
3. **Targeting:** Identify semantic voids. If a concept exists in the text but has low vector density in the question pool, it is flagged as a generation target.

### Stage 2: The Cognitive Blueprint (Bloom's Taxonomy)
The target concept is assigned a cognitive verb based on the needed Level:
*   **L1/L2 (Remember/Understand):** Generate Declarative/Comparative questions.
*   **L3 (Apply):** Generate Visual/Structural bottleneck questions.
*   **L4/L5 (Analyze/Evaluate):** Generate Diagnostic scenarios with "noise" variables.
*   **L6 (Create):** Generate Blank-slate architecture requirements + Chaos events.

### Stage 3: Plausible Distractor Generation
The LLM is explicitly prompted to generate "Near-Miss Distractors" based on common engineering misconceptions.
*   *Wrong Answer 1:* Mathematically inverted logic.
*   *Wrong Answer 2:* True for software engineering, false for ML systems (e.g., relying on OS swap space instead of managing VRAM).
*   *Wrong Answer 3:* True for a different ML concept.

### Stage 4: Adversarial Validation (The Red Team)
A dual-agent loop prevents hallucinations and ambiguity.
1. **Solver Agent (Temp 0.0):** Attempts to answer the generated question using *only* the textbook chunks.
2. **Critique Agent:** Analyzes the Solver's logic. If the question is unanswerable, ambiguous, or lacks strict physical grounding, it is rejected and sent back to Stage 2.
3. **Formatter:** If passed, the question is mapped into the strict StaffML Markdown schema and committed via a GitHub Action.

---

## 3. The Tech Stack

We leverage the following tools to construct the deterministic pipeline:

*   **Orchestration & Optimization:** [DSPy](https://github.com/stanfordnlp/dspy) (Stanford). Used to program the LLM agents using `Signatures` rather than string prompts, allowing for mathematical prompt compilation and optimization.
*   **Structured Output:** [Instructor](https://github.com/jxnl/instructor) (Python). Enforces strict Pydantic models on the LLM output to guarantee the Markdown schema (e.g., `<details>`, `> **Options:**`) never breaks.
*   **Vector Database:** [LanceDB](https://lancedb.github.io/lancedb/) or [Chroma](https://www.trychroma.com/). Used for continuous density clustering to programmatically prevent repetitive question generation.
*   **Telemetry & Observability:** [LangSmith](https://smith.langchain.com/). Used to monitor the nightly cron job, logging every generation trace and running programmatic evaluators against the "Napkin Math" outputs.
