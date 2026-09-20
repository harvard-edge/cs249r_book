# Seminal Papers Compendium: Volume III
## Foundational Papers Anchoring *Agentic Machine Learning Systems*

**Book Title:** *Agentic Machine Learning Systems: The Systems Engineering of Inference-Time Compute and Autonomous Control Loops*
**Core Organizing Paradigm:** *The Stochastic Computer*
**Bibliography Source:** `books/references-vol3.bib`

---

## Purpose and Organization

This compendium establishes the intellectual lineage of Volume III. A graduate textbook in systems engineering must not rely on transient industry hype or superficial analogies. It must stand upon enduring principles of computer systems design and rigorous empirical machine learning research.

Each entry below is structured around four definitive dimensions:
1. **The Paper:** Title, authors, publication venue, year, and repository BibTeX key.
2. **The Key Insight:** The fundamental theoretical, physical, or algorithmic breakthrough.
3. **Why It's Important:** Why the work has stood (or will stand) the test of time in computer science.
4. **Why It Relates to This Book:** The precise subsystem, chapter, and section in Volume III that relies on this result, and how it governs the engineering of the Stochastic Computer.

---

```
Volume III Seminal Papers Map
├── Introduction: The Stochastic Computer
│   ├── Saltzer, Reed & Clark (1984) — End-to-End Arguments
│   ├── Wilkes, Wheeler & Gill (1951) — The Subroutine Library
│   ├── Amdahl (1967) — The Law of Diminishing Returns in Compute
│   ├── Karpathy (2017) — Software 2.0
│   └── Jimenez et al. (2024) — SWE-bench
├── Part I: The Stochastic Processor (Chapters 02–03)
│   ├── Williams, Waterman & Patterson (2009) — The Roofline Model
│   ├── Willard & Louf (2023) — Grammar-Constrained Decoding (Outlines)
│   ├── Pope et al. (2023) — Scaling Transformer Inference
│   ├── Kahneman (2011) — Dual-Process Deliberation (System 1 vs. System 2)
│   ├── Snell, Lee, Xu & Kumar (2024) — Scaling Test-Time Compute
│   ├── Lightman et al. (2023) — Process Reward Models (PRMs)
│   ├── Yao et al. (2023) — Tree of Thoughts
│   ├── Shinn et al. (2023) — Reflexion
│   └── Wang et al. (2022) — Self-Consistency
├── Part II: Context Memory and Storage (Chapters 04–06)
│   ├── Denning (1968) — The Working Set Model for Program Behavior
│   ├── Liu et al. (2024) — Lost in the Middle
│   ├── Kwon et al. (2023) — PagedAttention & vLLM
│   ├── Zheng et al. (2024) — SGLang & RadixAttention
│   ├── Agrawal et al. (2024) — Sarathi-Serve & Chunked Prefill
│   ├── Smith (1982) — Cache Memories & Write-Invalidation
│   ├── Robertson & Zaragoza (2009) — The BM25 Probabilistic Framework
│   ├── Karpukhin et al. (2020) — Dense Passage Retrieval (DPR)
│   └── Edge et al. (2024) — GraphRAG
├── Part III: Tool Actuation and I/O Peripherals (Chapters 07–08)
│   ├── Ritchie & Thompson (1974) — The UNIX Time-Sharing System
│   ├── Fielding (2000) — REST & Network Architectural Idempotency
│   ├── Schick et al. (2023) — Toolformer
│   ├── Patil et al. (2023) — Gorilla
│   ├── Saltzer & Schroeder (1975) — The Protection of Information in Computer Systems
│   ├── Agache et al. (2020) — Firecracker MicroVMs
│   └── Haas et al. (2017) — WebAssembly (Wasm)
├── Part IV: The Agent Operating System (Chapters 09–11)
│   ├── Saltzer & Kaashoek (2009) — Principles of Computer System Design
│   ├── Gray & Reuter (1992) — Transaction Processing & Write-Ahead Logging
│   ├── Rosenblum & Ousterhout (1992) — Log-Structured File Systems
│   ├── Garcia-Molina & Salem (1987) — Sagas
│   ├── Lamport, Shostak & Pease (1982) — The Byzantine Generals Problem
│   └── Schlichting & Schneider (1983) — Fail-Stop Distributed Systems
├── Part V: The Policy Compiler (Chapters 12–14)
│   ├── Ouyang et al. (2022) — InstructGPT
│   ├── Schulman, Wolski, Dhariwal, Radford & Klimov (2017) — PPO
│   ├── Shao et al. (2024) — DeepSeekMath & Group Relative Policy Optimization (GRPO)
│   └── Guo et al. (2025) — DeepSeek-R1
├── Part VI: Distributed Fleets and Operations (Chapters 15–17)
│   ├── Hewitt (1973) / Agha (1986) — Actors: Concurrent Computation
│   ├── Hoare (1978) — Communicating Sequential Processes (CSP)
│   ├── Lamport (1978) — Logical Time and Event Ordering
│   ├── Sambasivan et al. (2016) — Distributed Systems Tracing Provenance
│   ├── Dean & Barroso (2013) — The Tail at Scale
│   ├── Chen et al. (2023) — FrugalGPT
│   └── Leviathan, Kalman & Matias (2023) — Speculative Decoding
└── Part VII: System Synthesis (Chapter 18)
    ├── Brooks (1986) — No Silver Bullet: Essence and Accident
    └── Amodei et al. (2016) — Concrete Problems in AI Safety
```

---

## Introduction: The Stochastic Computer

### 1. End-to-End Arguments in System Design
- **The Paper:** Jerome H. Saltzer, David P. Reed, and David D. Clark. *End-to-End Arguments in System Design*. ACM Transactions on Computer Systems (TOCS), Vol. 2, No. 4, pp. 277–288, 1984.
  `BibTeX: saltzer1984endtoend`
- **The Key Insight:** Functions placed at low levels of a system (such as intermediate communication links) may be redundant or of little value when compared to the cost of providing them at that low level. A function can only be completely and correctly implemented with the knowledge and help of the application standing at the endpoints of the communication system.
- **Why It's Important:** This paper is the foundational architectural doctrine of the Internet and distributed systems, establishing the boundary between lower-tier transport and application-level correctness.
- **Why It Relates to This Book (Chapters 01, 07, 11):** In Volume III, we study the **Invariant Closure Principle**, which inverts the classical end-to-end argument for stochastic agents. Because intermediate neural outputs are non-deterministic, an external runtime cannot rely on the endpoint (the model) to verify its own invariants. Invariant verification must be enforced at the runtime boundary (tools, sandboxes, sagas), while task goal satisfaction remains an end-to-end evaluation.

### 2. The Preparation of Programs for an Electronic Digital Computer
- **The Paper:** Maurice V. Wilkes, David J. Wheeler, and Stanley Gill. *The Preparation of Programs for an Electronic Digital Computer (with special reference to the EDSAC and the use of a library of subroutines)*. Addison-Wesley, 1951.
  `BibTeX: wilkes1951`
- **The Key Insight:** Software engineering became viable only when computers transitioned from monolithic, hard-wired calculation runs to stored-program machines equipped with relocatable **subroutine libraries** with explicit parameter-passing contracts.
- **Why It's Important:** Wilkes invented microprogramming and the subroutine, establishing that complex software must be built from reusable, compositional units with stable calling conventions.
- **Why It Relates to This Book (Chapters 01, 07):** Agentic systems today stand at the exact transition point Wilkes faced in 1949: moving from monolithic prompt strings ("one-shot prompting") to structured, reusable tool subroutines and trajectory execution frames.

### 3. Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities
- **The Paper:** Gene M. Amdahl. *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities*. AFIPS Conference Proceedings, Vol. 30, pp. 483–485, 1967.
  `BibTeX: amdahl1967`
- **The Key Insight:** The overall speedup of a system gained from an improvement is limited by the fraction of time that the improved component is actually used: $\text{Speedup} = \frac{1}{(1-f) + f/s}$.
- **Why It's Important:** Amdahl’s Law is the universal physical limit on parallelization and component optimization in computer architecture.
- **Why It Relates to This Book (Chapters 01, 02, 17):** In an agentic trajectory, total elapsed time is $T_{\text{task}} = T_{\text{model}} + T_{\text{tool}} + T_{\text{wait}} + T_{\text{runtime}}$. When tool calls (compilers, test runners, network scrapers) account for 75% of the trajectory duration, doubling neural model throughput ($s=2$ on $f=0.25$) only yields a negligible 12.5% whole-system speedup.

### 4. Software 2.0
- **The Paper:** Andrej Karpathy. *Software 2.0*. Medium / Stanford AI Blog, 2017.
  `BibTeX: karpathy2017software2`
- **The Key Insight:** Programs can be written not by human engineers specifying explicit control logic in C++ or Python (Software 1.0), but by optimization algorithms searching parameter space conditioned on datasets and loss functions (Software 2.0).
- **Why It's Important:** Popularized the paradigm shift from manual algorithmic design to learned neural representations.
- **Why It Relates to This Book (Chapter 01):** Chapter 1 establishes the **Tripartite Software Evolution**: Software 1.0 (deterministic code), Software 2.0 (statistical neural weights), and **Software 3.0 (agentic systems)**. Software 3.0 uses Software 1.0 runtimes to govern and sandbox Software 2.0 neural cores executing multi-step goals.

### 5. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
- **The Paper:** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexun Zhang, Ofir Press, and Karthik Narasimhan. *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
  `BibTeX: jimenez2024swebench`
- **The Key Insight:** Static NLP benchmarks (MMLU, GSM8K) fail to evaluate agentic systems. Rigorous evaluation requires multi-file repository navigation, bug localization, test-driven reproduction, patch generation, and execution against hidden unit test suites.
- **Why It's Important:** Shifted the entire industry from measuring single-token cross-entropy to evaluating multi-turn software engineering task completion in real execution environments.
- **Why It Relates to This Book (Chapters 01, 16):** Serves as the primary operational case study throughout the volume, illustrating how raw language capabilities translate into verifiable trajectory execution.

---

## Part I: The Stochastic Processor (Chapters 02–03)

### 6. Roofline: An Insightful Visual Performance Model for Multicore Architectures
- **The Paper:** Samuel Williams, Andrew Waterman, and David Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. Communications of the ACM (CACM), Vol. 52, No. 4, pp. 65–76, 2009.
  `BibTeX: williams2009roofline`
- **The Key Insight:** Computational performance is bounded either by the processor's peak arithmetic throughput (FLOP/s) or by the memory subsystem's memory bandwidth (GB/s), governed by arithmetic intensity (FLOPs/byte).
- **Why It's Important:** The definitive visual and mathematical framework for diagnosing whether a workload is compute-bound or memory-bound.
- **Why It Relates to This Book (Chapter 02, Section 2.5):** Explains the fundamental latency and resource asymmetry of the Stochastic Processor: **Prefill is compute-bound (dense GEMM)** operating at peak FLOP/s, whereas **Decode is memory-bandwidth bound (GEMV)** operating at $\approx 1\text{ FLOP/byte}$ because every generated token must stream the entire model parameter tensor from HBM into the compute cores.

### 7. Efficient Guided Generation for Large Language Models
- **The Paper:** Brandon T. Willard and Rémi Louf. *Efficient Guided Generation for Large Language Models*. arXiv:2307.09702, 2023 (Implemented as the *Outlines* library).
  `BibTeX: willard2023outlines`
- **The Key Insight:** Regular expressions and Context-Free Grammars (CFGs) can be compiled into Finite State Machines (FSMs) that dynamically compute a binary mask over the vocabulary tensor before the softmax step, guaranteeing valid syntax without fine-tuning or rejection sampling.
- **Why It's Important:** Eliminated JSON syntax errors in model generation by enforcing structural invariants directly on the logit simplex.
- **Why It Relates to This Book (Chapter 02, Section 2.4):** Establishes the bounded guarantee of the Stochastic Processor: grammar masking guarantees *syntactic validity by construction*, but explicitly does not guarantee semantic correctness, environment preconditions, or authorization.

### 8. Efficiently Scaling Transformer Inference on TPU v4
- **The Paper:** Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, et al. *Efficiently Scaling Transformer Inference on TPU v4*. MLSys 2023.
  `BibTeX: pope2023scaling`
- **The Key Insight:** Detailed the hardware trade-offs of tensor parallelism, pipeline parallelism, and the extreme memory footprint of the Key-Value (KV) cache during long-context generation.
- **Why It's Important:** First comprehensive systems accounting of inference hardware limits across memory capacity, memory bandwidth, and inter-chip interconnects.
- **Why It Relates to This Book (Chapters 02, 05):** Grounds the physical resource cost of prompt tokens vs. generation tokens and the hardware reality of KV cache growth.

### 9. Thinking, Fast and Slow
- **The Paper / Book:** Daniel Kahneman. *Thinking, Fast and Slow*. Farrar, Straus and Giroux, 2011.
  `BibTeX: kahneman2011`
- **The Key Insight:** Cognitive processes operate across two modes: System 1 (fast, instinctive, associative, constant-time pattern recognition) and System 2 (slow, deliberate, analytical, variable-time search and verification).
- **Why It's Important:** Foundational cognitive psychology work demonstrating that non-trivial problem solving requires deliberate multi-step reasoning and error correction.
- **Why It Relates to This Book (Chapter 03, Section 3.1):** Direct theoretical analogy for inference compute: a single forward pass is pure System 1 (statistical next-token sampling), whereas inference-time deliberation (Ch 3) is the software engineering realization of System 2 (spending test-time compute to verify and search alternatives).

### 10. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
- **The Paper:** Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*. arXiv:2408.03314, 2024.
  `BibTeX: snell2024scaling`
- **The Key Insight:** On complex tasks, allocating additional test-time compute (via candidate search, revisions, and verifier scoring) can outperform a model with $14\times$ more parameters run greedily. Furthermore, the optimal balance between search breadth and revision depth depends strictly on task difficulty.
- **Why It's Important:** Formalized the "inference compute scaling law", demonstrating that inference compute is a first-class scaling axis alongside pretraining compute.
- **Why It Relates to This Book (Chapter 03, Sections 3.2, 3.6):** Provides the mathematical backbone for Chapter 3's budget accounting, calculation designs, and stopping rules.

### 11. Let's Verify Step by Step
- **The Paper:** Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. *Let's Verify Step by Step*. arXiv:2305.20050, 2023.
  `BibTeX: lightman2023let`
- **The Key Insight:** Process Reward Models (PRMs), which evaluate and score the correctness of each intermediate step of reasoning, drastically outperform Outcome Reward Models (ORMs) that only score the terminal answer.
- **Why It's Important:** Enabled granular credit assignment during multi-step reasoning, allowing search algorithms to prune invalid trajectories at the exact step where an error occurs.
- **Why It Relates to This Book (Chapter 03, Section 3.3):** Anchors the discussion of candidate selection and step-level verifiers in deliberate search.

### 12. Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- **The Paper:** Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*. NeurIPS 2023.
  `BibTeX: yao2023tree`
- **The Key Insight:** Generalized Chain-of-Thought prompting to tree structures, allowing models to explore multiple reasoning paths, evaluate intermediate choices, and backtrack when a path hits a dead end.
- **Why It's Important:** Established search (BFS, DFS, heuristic frontiers) as a principled mechanism for LLM reasoning over non-trivial combinatorial problem spaces.
- **Why It Relates to This Book (Chapter 03, Section 3.5):** Forms the basis of bounded search over alternative proposals.

### 13. Reflexion: Language Agents with Verbal Reinforcement Learning
- **The Paper:** Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R. Narasimhan, and Shunyu Yao. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023.
  `BibTeX: shinn2023reflexion`
- **The Key Insight:** Agents can self-improve across task iterations not by updating neural weights, but by translating environmental feedback and execution failures into natural language self-reflections staged in working memory.
- **Why It's Important:** Proved that episodic memory and linguistic self-critique provide a lightweight, immediate form of policy correction.
- **Why It Relates to This Book (Chapter 03, Section 3.4; Chapter 04):** Illustrates iterative revision and how feedback loops interact with working memory.

### 14. Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **The Paper:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. *Self-Consistency Improves Chain of Thought Reasoning in Language Models*. ICLR 2023.
  `BibTeX: wang2022selfconsistency`
- **The Key Insight:** Instead of greedy decoding, sampling multiple diverse reasoning paths from a model and taking the majority vote over final answers significantly boosts accuracy on complex tasks.
- **Why It's Important:** The simplest and most robust form of test-time compute allocation (Best-of-$N$ with voting).
- **Why It Relates to This Book (Chapter 03, Section 3.2):** Serves as the baseline candidate sampling mechanism before introducing learned step-verifiers and tree search.

---

## Part II: Context Memory and Storage (Chapters 04–06)

### 15. The Working Set Model for Program Behavior
- **The Paper:** Peter J. Denning. *The Working Set Model for Program Behavior*. Communications of the ACM (CACM), Vol. 11, No. 11, pp. 323–333, 1968.
  `BibTeX: denning1968`
- **The Key Insight:** Programs exhibit temporal and spatial locality; at any point in execution, a program requires only a subset of its total address space (its "working set") in physical memory to execute efficiently without thrashing.
- **Why It's Important:** Foundational principle of operating systems virtual memory, caching, and page replacement policies.
- **Why It Relates to This Book (Chapter 04):** Chapter 4 establishes that **the Context Window is the L1 Working Set of the Stochastic Computer**. The runtime cannot dump all project history into prompt space; it must maintain a strictly bounded working set via compaction, eviction, and summarization to prevent "context thrashing" (performance degradation and attention dispersion).

### 16. Lost in the Middle: How Language Models Use Long Contexts
- **The Paper:** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts*. TACL 2024.
  `BibTeX: liu2024lost`
- **The Key Insight:** Transformer attention is not uniform across long contexts. Retrieval and reasoning performance degrades severely when relevant information is placed in the middle of long prompts compared to the beginning or end.
- **Why It's Important:** Debunked the naive assumption that massive context windows (100k+ tokens) solve memory retrieval without active curation.
- **Why It Relates to This Book (Chapter 04, Section 4.2):** Provides the empirical evidence for context dispersion and why active context compaction and positional staging are mandatory systems requirements.

### 17. Efficient Memory Management for Large Language Model Serving with PagedAttention
- **The Paper:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023 (vLLM).
  `BibTeX: kwon2023vllm`
- **The Key Insight:** Key-Value (KV) cache memory in LLM serving suffers from up to 80% internal and external fragmentation due to dynamic sequence lengths. Paging the KV cache into fixed-size physical memory blocks eliminates fragmentation and enables near-zero-waste memory sharing across prompts.
- **Why It's Important:** Revolutionized LLM serving infrastructure, multiplying serving throughput by $2\text{--}4\times$ across the entire industry.
- **Why It Relates to This Book (Chapter 05):** Core systems anchor for Chapter 5 (*The KV-Cache Hierarchy*), showing how virtual memory concepts apply to accelerator HBM.

### 18. SGLang: Efficient Execution of Structured Language Model Programs
- **The Paper:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Hao Zhang. *SGLang: Efficient Execution of Structured Language Model Programs*. arXiv:2312.07104, 2024.
  `BibTeX: zheng2024sglang`
- **The Key Insight:** Multi-turn agent trajectories share significant prompt prefixes (system prompts, tool definitions, historical traces). Maintaining a **Radix Tree** over KV-cache blocks enables automatic, dynamic prefix caching and reuse across arbitrary multi-turn requests.
- **Why It's Important:** Made multi-turn agent execution economically viable by turning KV-cache reuse from a static manual trick into a dynamic, pageable tree cache.
- **Why It Relates to This Book (Chapter 05, Section 5.3):** Serves as the primary mechanism for prefix caching and shared execution trees in agent runtimes.

### 19. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
- **The Paper:** Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*. OSDI 2024.
  `BibTeX: agrawal2024sarathi`
- **The Key Insight:** Long prefill requests cause massive queue delays and bubble stalls for ongoing decode requests. Splitting large prefills into chunks (**Chunked Prefill**) and piggbacking them onto decode batches balances compute and memory bandwidth, eliminating latency spikes.
- **Why It's Important:** Solved the primary scheduling interference problem in shared LLM serving clusters.
- **Why It Relates to This Book (Chapter 05, Section 5.4; Chapter 17):** Crucial for understanding how long agent observation payloads impact cluster serving latency.

### 20. Cache Memories
- **The Paper:** Alan Jay Smith. *Cache Memories*. ACM Computing Surveys, Vol. 14, No. 3, pp. 473–530, 1982.
  `BibTeX: smith1982cache`
- **The Key Insight:** The definitive survey establishing the mechanics of cache coherence, line placement, write-through vs. write-back, and **write-invalidation protocols** across hierarchical memory systems.
- **Why It's Important:** The canonical text on cache design in computer architecture for over four decades.
- **Why It Relates to This Book (Chapter 06, Section 6.5):** Directly governs Section 6.5's **Write-Invalidation Protocols**: when an agent modifies a file on disk, pre-computed vector embeddings, inverted indices, and graph nodes become stale and must be invalidated or re-indexed.

### 21. The Probabilistic Relevance Framework: BM25 and Beyond
- **The Paper:** Stephen Robertson and Hugo Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, Vol. 3, No. 4, pp. 333–412, 2009.
  `BibTeX: robertson2009bm25`
- **The Key Insight:** Formalized BM25 lexical ranking, balancing term frequency saturation ($k_1$) and document length normalization ($b$) under probabilistic retrieval theory.
- **Why It's Important:** Remains the unshakeable, computationally lightweight baseline for exact-match keyword information retrieval.
- **Why It Relates to This Book (Chapter 06, Section 6.2):** Dense vector retrieval fails on exact identifiers, function names, and commit hashes. Volume III establishes that durable agent storage requires **hybrid lexical-dense retrieval** combining BM25 with embedding models via Reciprocal Rank Fusion (RRF).

### 22. Dense Passage Retrieval for Open-Domain Question Answering
- **The Paper:** Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.
  `BibTeX: karpukhin2020dpr`
- **The Key Insight:** Dual-encoder architectures trained with in-batch negatives project questions and passages into a shared continuous vector space, outperforming traditional lexical retrieval on semantic queries.
- **Why It's Important:** Established the modern vector database and RAG (Retrieval-Augmented Generation) paradigm.
- **Why It Relates to This Book (Chapter 06, Section 6.2):** Represents the dense vector search tier of persistent external memory.

### 23. From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- **The Paper:** Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apratim Muku, Shanon Morrison, and Jonathan Larson. *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*. arXiv:2404.16130, 2024.
  `BibTeX: edge2024graphrag`
- **The Key Insight:** Standard vector RAG fails on global questions ("What are the main architectural themes of this system?") because it retrieves isolated fragments. Extracting knowledge graphs and clustering communities hierarchically enables multi-scale global summarization.
- **Why It's Important:** Extended external memory from unstructured text chunks to structured semantic graphs.
- **Why It Relates to This Book (Chapter 06, Section 6.4):** Models the knowledge graph tier of external storage for complex codebases and enterprise knowledge.

---

## Part III: Tool Actuation and I/O Peripherals (Chapters 07–08)

### 24. The UNIX Time-Sharing System
- **The Paper:** Dennis M. Ritchie and Ken Thompson. *The UNIX Time-Sharing System*. Communications of the ACM (CACM), Vol. 17, No. 7, pp. 365–375, 1974.
  `BibTeX: ritchie1974unix`
- **The Key Insight:** Everything is a file: abstracting peripheral devices, disk storage, and inter-process communication into a uniform byte-stream interface accessed via `read()`, `write()`, and `close()`.
- **Why It's Important:** Established the modern operating system abstraction for I/O and peripheral interoperability.
- **Why It Relates to This Book (Chapter 07, Section 7.1):** Ritchie and Thompson’s unified I/O model is the direct ancestor of modern tool interfaces: mapping heterogeneous external systems into a standardized JSON request/response schema.

### 25. Architectural Styles and the Design of Network-based Software Architectures (REST)
- **The Paper:** Roy Thomas Fielding. *Architectural Styles and the Design of Network-based Software Architectures*. PhD Dissertation, UC Irvine, 2000.
  `BibTeX: fielding2000rest`
- **The Key Insight:** Distributed systems require stateless interaction, uniform resource identifiers, and explicit **idempotency guarantees** ($PUT$, $DELETE$) to operate reliably across unreliable networks.
- **Why It's Important:** The architectural blueprint of the modern World Wide Web and cloud APIs.
- **Why It Relates to This Book (Chapter 07, Section 7.3):** Agents execute over noisy environments and retry failed tool calls. Section 7.3 establishes **Idempotency Keys** as a mandatory peripheral protocol: retrying a non-idempotent tool (e.g., charge credit card, send email) without an idempotency key causes catastrophic duplicate real-world effects.

### 26. Toolformer: Language Models Can Teach Themselves to Use Tools
- **The Paper:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS 2023.
  `BibTeX: schick2023toolformer`
- **The Key Insight:** Language models can learn via self-supervised rejection sampling to inject functional API call tokens into their text stream, execute them, and condition subsequent generation on the API outputs.
- **Why It's Important:** Shifted AI models from passive statistical text generators to active systems capable of calling external peripherals.
- **Why It Relates to This Book (Chapter 07, Section 7.1; Chapter 13):** Foundation of neural tool invocation and tool-calling training.

### 27. Gorilla: Large Language Model Connected with Massive APIs
- **The Paper:** Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. *Gorilla: Large Language Model Connected with Massive APIs*. arXiv:2305.15334, 2023.
  `BibTeX: patil2023gorilla`
- **The Key Insight:** Fine-tuning models with retriever-aware API documentation dramatically reduces tool hallucination and handles evolving API schemas.
- **Why It's Important:** Demonstrated how models can reliably select from thousands of heterogeneous API specifications.
- **Why It Relates to This Book (Chapter 07, Section 7.2):** Informs tool registry architectures and schema discovery.

### 28. The Protection of Information in Computer Systems
- **The Paper:** Jerome H. Saltzer and Michael D. Schroeder. *The Protection of Information in Computer Systems*. Proceedings of the IEEE, Vol. 63, No. 9, pp. 1278–1308, 1975.
  `BibTeX: saltzer1975`
- **The Key Insight:** Established the foundational design principles for secure computer systems: **Least Privilege**, **Economy of Mechanism**, **Complete Mediation**, **Open Design**, and **Fail-Safe Defaults**.
- **Why It's Important:** The bedrock of cybersecurity and access control theory for 50 years.
- **Why It Relates to This Book (Chapter 08):** Chapter 8 (*Virtualization and Isolation*) translates Saltzer & Schroeder directly into agent architectures: models cannot be granted ambient host authority; tool execution requires capability tokens, complete mediation, and sandboxed isolation.

### 29. Firecracker: Lightweight Virtualization for Serverless Applications
- **The Paper:** Alexandru Agache, Marc Brooker, Andreea Florescu, Alexandra Iordache, Anthony Liguori, Rolf Neugebauer, Phil Piwonka, and Diana-Maria Popa. *Firecracker: Lightweight Virtualization for Serverless Applications*. NSDI 2020.
  `BibTeX: agache2020firecracker`
- **The Key Insight:** Traditional VMs are too heavy (seconds to boot, hundreds of MB overhead); containers share the host kernel and leak. A minimal Linux KVM-based Virtual Machine Monitor written in Rust boots microVMs in $<5\text{ ms}$ with a $5\text{ MB}$ memory footprint, achieving multi-tenant security at container speed.
- **Why It's Important:** Powers AWS Lambda and Fargate; defined modern multi-tenant workload isolation.
- **Why It Relates to This Book (Chapter 08, Section 8.2):** Serves as the premier hardware-isolated execution sandbox for running untrusted agent-generated code (e.g., executing arbitrary Python scripts in SWE-bench).

### 30. Bringing the Web up to Speed with WebAssembly
- **The Paper:** Andreas Haas, Andreas Rossberg, Derek L. Schuff, Ben L. Titzer, Michael Holman, Dan Gohman, Luke Wagner, Alon Zakai, and JF Bastien. *Bringing the Web up to Speed with WebAssembly*. PLDI 2017.
  `BibTeX: haas2017wasm`
- **The Key Insight:** A safe, portable, compact binary format running near-native speed in a capability-based, memory-safe sandbox without access to host resources unless explicitly imported.
- **Why It's Important:** Created a universal, fast, lightweight sandbox boundary across browsers, servers, and edge runtimes.
- **Why It Relates to This Book (Chapter 08, Section 8.3):** WebAssembly (Wasm/WASI) is evaluated as the ideal lightweight sandbox for non-POSIX agent tool execution and deterministic replay.

---

## Part IV: The Agent Operating System (Chapters 09–11)

### 31. Principles of Computer System Design: An Introduction
- **The Paper / Book:** Jerome H. Saltzer and M. Frans Kaashoek. *Principles of Computer System Design: An Introduction*. Morgan Kaufmann, 2009.
  `BibTeX: saltzer2009`
- **The Key Insight:** Comprehensive pedagogical framework for managing systems complexity: modularity, virtualization, names, concurrency, and fault tolerance via atomic transactions.
- **Why It's Important:** The canonical MIT graduate systems textbook.
- **Why It Relates to This Book (Whole Book, especially Part IV):** Provides the systems philosophy for Volume III: treating the agent runtime as an Operating System responsible for coordinating untrusted, concurrent, and fallible components.

### 32. Transaction Processing: Concepts and Techniques
- **The Paper / Book:** Jim Gray and Andreas Reuter. *Transaction Processing: Concepts and Techniques*. Morgan Kaufmann, 1992.
  `BibTeX: gray1992transaction`
- **The Key Insight:** Defined ACID semantics (Atomicity, Consistency, Isolation, Durability) and established **Write-Ahead Logging (WAL)**: never write an updated data page to non-volatile disk until the log record describing the update is safely flushed to persistent storage.
- **Why It's Important:** Jim Gray received the Turing Award for this work; it underlies every relational database, transactional filesystem, and distributed ledger in existence.
- **Why It Relates to This Book (Chapter 10, Section 10.2):** Trajectory event sourcing in agent runtimes must adhere to WAL discipline: an agent cannot dispatch a side-effecting action to a peripheral before the event record is appended to the durable trajectory log.

### 33. The Design and Implementation of a Log-Structured File System
- **The Paper:** Mendel Rosenblum and John K. Ousterhout. *The Design and Implementation of a Log-Structured File System*. ACM Transactions on Computer Systems (TOCS), Vol. 10, No. 1, pp. 26–52, 1992.
  `BibTeX: rosenblum1992lfs`
- **The Key Insight:** In a system where memory caches absorb reads, disk writes dominate. Writing all modifications sequentially to an append-only log maximizes write bandwidth, simplifies crash recovery, and eliminates random write overhead.
- **Why It's Important:** Pioneered append-only storage, inspiring modern event sourcing, Apache Kafka, and LSM-trees (RocksDB).
- **Why It Relates to This Book (Chapter 10, Section 10.1):** Justifies append-only event sourcing for trajectory storage, crash recovery, and deterministic execution replay.

### 34. Sagas
- **The Paper:** Hector Garcia-Molina and Kenneth Salem. *Sagas*. ACM SIGMOD Record, Vol. 16, No. 3, pp. 249–259, 1987.
  `BibTeX: garciamolina1987sagas`
- **The Key Insight:** Long-Lived Transactions (LLTs) cannot hold database locks for hours or days without paralyzing the system. A Saga breaks an LLT into a sequence of small, independent transactions $T_1, T_2, \dots, T_n$, each with an associated **compensating transaction** $C_1, C_2, \dots, C_{n-1}$ that semantically undoes its partial effects if a downstream step fails.
- **Why It's Important:** The definitive failure recovery pattern for distributed systems, microservices, and workflows where two-phase commit ($2\text{PC}$) is impractical.
- **Why It Relates to This Book (Chapter 11, Sections 11.2, 11.3):** Classical database rollback ($2\text{PC}$) is physically impossible in agent systems (an agent cannot "un-send" a Slack message or "un-drop" an AWS database). Agents must execute as Sagas: compensating forward or backward via explicit semantic repair actions.

### 35. The Byzantine Generals Problem
- **The Paper:** Leslie Lamport, Robert Shostak, and Marshall Pease. *The Byzantine Generals Problem*. ACM Transactions on Programming Languages and Systems (TOPLAS), Vol. 4, No. 3, pp. 382–401, 1982.
  `BibTeX: lamport1982byzantine`
- **The Key Insight:** Reliable computation in the presence of components that can fail arbitrarily, withhold information, or transmit conflicting messages (Byzantine faults) requires strict consensus bounds ($3m+1$ nodes to tolerate $m$ traitors).
- **Why It's Important:** Foundational text on adversarial fault tolerance and distributed consensus.
- **Why It Relates to This Book (Chapters 01, 11, 15):** Neural models are not malicious traitors, but their non-deterministic "Fail-Plausible" outputs (confidently hallucinating facts, subtly corrupting schemas) share mathematical characteristics with Byzantine components, requiring cross-verification before committing state.

### 36. Fail-Stop Distributed Systems
- **The Paper:** Richard D. Schlichting and Fred B. Schneider. *Fail-Stop Distributed Systems: An Approach to Halting and Recovering*. ACM Transactions on Computer Systems (TOCS), Vol. 1, No. 3, pp. 222–238, 1983.
  `BibTeX: schlichting1983failstop`
- **The Key Insight:** Formulated the **Fail-Stop** fault model: processors fail by halting, and other non-faulty processors can detect that failure immediately.
- **Why It's Important:** Simplified fault-tolerant algorithm design by establishing a predictable baseline failure mode.
- **Why It Relates to This Book (Chapters 01, 11):** Chapter 1 uses this paper to highlight the central hazard of AI systems: agents do **not** obey the Fail-Stop model. They do not crash when they are confused; they emit plausible-sounding incorrect code (Fail-Plausible).

---

## Part V: The Policy Compiler (Chapters 12–14)

### 37. Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
- **The Paper:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. *Training Language Models to Follow Instructions with Human Feedback*. NeurIPS 2022.
  `BibTeX: ouyang2022instructgpt`
- **The Key Insight:** Pure unsupervised pretraining optimizes next-token probability on internet text, not human intent. Fine-tuning with Supervised Fine-Tuning (SFT) followed by Reinforcement Learning from Human Feedback (RLHF) aligns the model’s outputs with human instructions.
- **Why It's Important:** The breakthrough behind ChatGPT; transformed LLMs from autocomplete engines into instruction-following agents.
- **Why It Relates to This Book (Chapters 12, 13, 14):** Serves as the blueprint for compiling raw trajectory data into aligned procedural models.

### 38. Proximal Policy Optimization Algorithms (PPO)
- **The Paper:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017.
  `BibTeX: schulman2017ppo`
- **The Key Insight:** Standard policy gradient methods suffer from destructively large policy updates. Clipping the objective function ($r_t(\theta) \hat{A}_t$) constrains policy changes within a trust region, ensuring stable and sample-efficient reinforcement learning.
- **Why It's Important:** Became the industry workhorse algorithm for deep RL and LLM alignment.
- **Why It Relates to This Book (Chapter 14, Section 14.3):** The foundation of RLVR (Reinforcement Learning with Verifiable Rewards) before analyzing memory-efficient alternatives like GRPO.

### 39. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)
- **The Paper:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, and Daya Guo. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300, 2024.
  `BibTeX: shao2024deepseekmath`
- **The Key Insight:** Standard PPO requires a separate Critic (Value) neural network roughly equal in size to the Actor model, consuming massive accelerator memory. **Group Relative Policy Optimization (GRPO)** eliminates the Critic entirely by sampling a group of outputs for each prompt and normalizing the reward across the group.
- **Why It's Important:** Reduced the GPU memory requirement for RL post-training by $\approx 50\%$, democratizing large-scale mathematical and reasoning reinforcement learning.
- **Why It Relates to This Book (Chapter 14, Section 14.4):** Provides the concrete accelerator memory accounting calculation in Chapter 14, showing how memory footprint dictates training cluster architecture.

### 40. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **The Paper:** Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, et al. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948, 2025.
  `BibTeX: deepseek2025r1`
- **The Key Insight:** Complex reasoning behaviors (long Chain-of-Thought, self-verification, reflection, alternative exploration) can emerge purely through large-scale RL with verifiable rewards (math, code execution) without prior human SFT data.
- **Why It's Important:** Proved that reasoning deliberation can be directly synthesized and incentivized into neural weights via verifiable reinforcement learning.
- **Why It Relates to This Book (Chapters 03, 14):** Bridges the test-time search mechanisms of Chapter 3 with the post-training optimization of Chapter 14.

---

## Part VI: Distributed Fleets and Operations (Chapters 15–17)

### 41. Actors: A Model of Concurrent Computation in Distributed Systems
- **The Paper / Book:** Carl Hewitt, Peter Bishop, and Richard Steiger. *A Universal Modular ACTOR Formalism for Artificial Intelligence*. IJCAI 1973; Gul Agha. *Actors: A Model of Concurrent Computation in Distributed Systems*. MIT Press, 1986.
  `BibTeX: agha1986actors`
- **The Key Insight:** Encapsulates state and behavior into independent concurrent entities ("Actors") that communicate exclusively through asynchronous message passing, eliminating shared-memory race conditions and locks.
- **Why It's Important:** The theoretical foundation of distributed, highly concurrent systems (Erlang, Akka, Ray).
- **Why It Relates to This Book (Chapter 15, Section 15.2):** Serves as the premier architectural paradigm for multi-agent fleets: treating individual agents as stateful Actors exchanging typed messages over communication channels.

### 42. Communicating Sequential Processes (CSP)
- **The Paper:** C. A. R. Hoare. *Communicating Sequential Processes*. Communications of the ACM (CACM), Vol. 21, No. 8, pp. 666–677, 1978.
  `BibTeX: hoare1978csp`
- **The Key Insight:** Concurrency is modeled as autonomous sequential processes that synchronize and exchange data over unbuffered or buffered channels, establishing formal algebraic reasoning for communication.
- **Why It's Important:** Inspired the Go programming language concurrency model and formal verification of concurrent software.
- **Why It Relates to This Book (Chapter 15, Section 15.3):** Governs synchronous pipelines and coordinator-worker topologies in multi-agent fleets.

### 43. Time, Clocks, and the Ordering of Events in a Distributed System
- **The Paper:** Leslie Lamport. *Time, Clocks, and the Ordering of Events in a Distributed System*. Communications of the ACM (CACM), Vol. 21, No. 7, pp. 558–565, 1978.
  `BibTeX: lamport1978time`
- **The Key Insight:** Physical time cannot be trusted across distributed nodes; event causality can only be tracked via logical clocks that define an unambiguous partial ordering of events ($a \to b$, the "happened-before" relation).
- **Why It's Important:** The single most cited and influential paper in distributed systems history; Lamport received the Turing Award in part for this work.
- **Why It Relates to This Book (Chapters 15, 16):** In a distributed fleet of multiple communicating agents executing concurrent tool calls, causality cannot be reconstructed from wall-clock timestamps. Trajectory tracing requires logical clocks and causal graph IDs.

### 44. So, you want to trace your distributed system? Key design insights from years of provenance research
- **The Paper:** Raja R. Sambasivan, Rodrigo Fonseca, Ilari Shafer, and Gregory R. Ganger. *So, you want to trace your distributed system? Key design insights from years of provenance research*. CMU-PDL-14-102, 2016.
  `BibTeX: sambasivan2016tracing`
- **The Key Insight:** Synthesized two decades of distributed tracing research, formalizing how trace context propagation, causal metadata graphs, and span hierarchies diagnose latency anomalies and concurrency bugs.
- **Why It's Important:** Grounded the architecture that evolved into Google Dapper, Zipkin, Jaeger, and the CNCF **OpenTelemetry** standard.
- **Why It Relates to This Book (Chapter 16, Section 16.1):** Direct foundation for Section 16.1: constructing distributed trajectory observability via OpenTelemetry spans across model calls, tool executions, and multi-agent message passing.

### 45. The Tail at Scale
- **The Paper:** Jeffrey Dean and Luiz André Barroso. *The Tail at Scale*. Communications of the ACM (CACM), Vol. 56, No. 2, pp. 74–80, 2013.
  `BibTeX: dean2013tail`
- **The Key Insight:** In large distributed systems executing fan-out requests across thousands of servers, tail latency ($P_{99}, P_{99.9}$) dominates user experience. Systems must use hedged requests, speculative retries, and proactive resource isolation to tolerate high tail variance.
- **Why It's Important:** The definitive systems guide for building responsive hyperscale cloud services out of unpredictable components.
- **Why It Relates to This Book (Chapter 16, Section 16.3; Chapter 17):** Critical for multi-agent fleet operations: when an agent fans out 10 sub-tasks across 10 LLM invocations, the slowest invocation dictates the overall trajectory step latency.

### 46. FrugalGPT: How to Use Large Language Models More Cheaply and Efficiently
- **The Paper:** Lingjiao Chen, Matei Zaharia, and James Zou. *FrugalGPT: How to Use Large Language Models More Cheaply and Efficiently*. arXiv:2305.05176, 2023.
  `BibTeX: chen2023frugalgpt`
- **The Key Insight:** Employing a cascading pipeline of models (querying a fast, cheap small model first, evaluating answer confidence, and escalating to expensive frontier models only when confidence is low) cuts inference costs by up to $98\%$ with zero loss in task accuracy.
- **Why It's Important:** Formalized model cascading and cost-performance optimization for production LLM deployments.
- **Why It Relates to This Book (Chapter 17, Section 17.2):** Forms the basis of Model Cascading and Tiered Routing in the Operations and Economics chapter.

### 47. Fast Inference from Transformers via Speculative Decoding
- **The Paper:** Yaniv Leviathan, Matan Kalman, and Yossi Matias. *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.
  `BibTeX: leviathan2023speculative`
- **The Key Insight:** A small draft model generates $K$ candidate tokens quickly (memory bandwidth bound on small weights); the large target model verifies all $K$ tokens in a single parallel forward pass (compute bound GEMM). By using a modified rejection sampling scheme, the output distribution is **mathematically identical** to sampling from the target model alone, yielding a $2\text{--}3\times$ speedup.
- **Why It's Important:** Proved that autoregressive generation latency can be broken without altering the output distribution.
- **Why It Relates to This Book (Chapter 17, Section 17.3):** Relocated from Chapter 2 to Chapter 17 as an advanced generation acceleration technique.

---

## Part VII: System Synthesis (Chapter 18)

### 48. No Silver Bullet—Essence and Accident in Software Engineering
- **The Paper:** Frederick P. Brooks, Jr. *No Silver Bullet—Essence and Accident in Software Engineering*. IEEE Computer, Vol. 20, No. 4, pp. 10–19, 1987.
  `BibTeX: brooks1987nosilverbullet`
- **The Key Insight:** Software development difficulties divide into **accidental complexity** (arising from our tools and language syntax, which can be solved) and **essential complexity** (arising from the inherent conceptual structure of software: complexity, conformity, changeability, and invisibility). No single technological breakthrough will ever yield a $10\times$ productivity jump across the board because essential complexity remains intractable.
- **Why It's Important:** Fred Brooks’ timeless warning against technology silver bullets; foundational to software engineering philosophy.
- **Why It Relates to This Book (Chapter 18, Section 18.2):** Grounds the synthesis chapter: Autonomous agents solve immense accidental complexity (boilerplate code, test script running, log scraping), but they do not eliminate the essential complexity of software specification, architectural coherence, and business requirement alignment.

---

### Summary Takeaway

This compendium demonstrates that Volume III is anchored in:
1. **Classical Systems Theory:** Saltzer, Wilkes, Amdahl, Gray, Garcia-Molina, Lamport, Ritchie, Brooks.
2. **Modern Systems & Infrastructure:** Kwon (vLLM), Zheng (SGLang), Agrawal (Sarathi), Agache (Firecracker), Williams (Roofline).
3. **Foundational Agentic & Deliberative Machine Learning:** Snell, Lightman, Yao, Shinn, Jimenez (SWE-bench), Karpathy, Shao & Guo (DeepSeek).

Every claim, calculation, and mechanism taught in *Agentic Machine Learning Systems* descends directly from this literature.
