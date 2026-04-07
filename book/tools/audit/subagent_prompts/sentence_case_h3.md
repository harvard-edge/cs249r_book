# Subagent prompt: H3+ heading sentence-case review

This is the prompt template the orchestrator dispatches to an Explore
subagent for each chapter that has open `h3-titlecase` issues.

The `{CHAPTER_FILE_PATH}` placeholder is replaced with the actual file
path before dispatch. Subagents are dispatched in parallel — N agents
in one tool message, one per chapter — so a vol2 sweep takes roughly
the wall time of one chapter, not N times that.

---

## Prompt template (verbatim)

You are reviewing a chapter of an MIT Press systems textbook for heading style compliance. The rule is: **H3 and below headings (`###`, `####`, `#####`) use sentence case** — only the first word and proper nouns are capitalized. **H1 (`#`) and H2 (`##`) use headline case** (capitalize principal words) and should NOT be touched.

**Proper nouns to preserve in their original capitalization:**

- **Product/framework names:** PyTorch, TensorFlow, JAX, CUDA, cuDNN, ImageNet, BERT, GPT-2, GPT-3, GPT-4, ResNet, AlexNet, MobileNet, EfficientNet, FlashAttention, PagedAttention, TinyML, AllReduce, Tensor Cores, Transformer Engine, MLPerf, MLCommons, ONNX, TensorRT, Stable Diffusion, DALL-E, MATLAB, Llama, Mistral, Mixtral, Gemini, Claude, Weights & Biases, Kubernetes, Slurm, Ray, Horovod, NCCL, Megatron, DeepSpeed, ZeRO, Adam, AdamW, SGD.
- **Hardware product names:** NVIDIA, AMD, Intel, Apple, A100, H100, V100, TPU, NPU, DPU, IPU, Snapdragon, Cortex, Neoverse, Xeon, EPYC, Ryzen, M1, M2, M3, Jetson, Orin, Xavier, Coral.
- **Acronyms (always uppercase):** GPU, CPU, TPU, NPU, FPGA, ASIC, FLOPs, FLOPS, TFLOPS, BLAS, LAPACK, GEMM, LLM, MLP, CNN, RNN, MoE, DAG, JIT, AOT, AST, IR, BPTT, DLRM, ETL, ELT, MLOps, NaN, ReLU, GELU, ROC, AUC, FFT, JSON, SSA, RBAC, GDPR, HIPAA, IOPS, NVMe, PTX, SIMD, MAC, RNN, RISC, MIPS, NAS, OTA, SoC, SLA, TCO, UAT, ViT, KWS, ICR, ILSVRC, CUDA.
- **Lowercase concept terms** that the merged prose file (§10.3) deliberately keeps lowercase: iron law, degradation equation, verification gap, bitter lesson (the concept), ML node, data wall, compute wall, memory wall, power wall, energy corollary, transformer (the architecture, generic use), four pillars framework, machine learning operations (generic). When you encounter these in an H3+ heading, lowercase them per §10.3.

**Read this file in full:** `{CHAPTER_FILE_PATH}`

**Read the rule context:** `/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude/rules/book-prose-merged.md` sections 10.3 and 10.9.

**Find every H3, H4, H5, H6 heading** in the chapter. For each one that violates sentence case, return an edit. For each one that already complies, do not return an edit. For each one where you are uncertain (proper-noun ambiguity, coined term), return an edit with `confidence: "low"` and explain in the `reason` field.

**Important rules:**
- H1 (`#`) and H2 (`##`) headings use headline case — DO NOT touch them.
- Never modify the section anchor ID `{#sec-...}` — those are permanent.
- Preserve all index markers, code spans, and inline math inside the heading text.
- Preserve `(GPT-2)` and other parenthetical acronyms exactly as written.
- The first word of the heading is ALWAYS capitalized in sentence case (even articles and prepositions). For example, `### A New Approach` becomes `### A new approach`, not `### a new approach`.
- "Bitter Lesson" stays capitalized when it refers specifically to Sutton's essay title (e.g. `### The Bitter Lesson`); lowercase when used as a general concept.

**Output format (return ONLY this JSON, no preamble, no explanation):**

```json
[
  {
    "line": 1532,
    "before": "### Hardware Balance: The Paradigm Partition",
    "after": "### Hardware balance: The paradigm partition",
    "confidence": "high",
    "reason": "No proper nouns; standard sentence case fix"
  },
  {
    "line": 1645,
    "before": "### ResNet Architecture",
    "after": "### ResNet architecture",
    "confidence": "high",
    "reason": "ResNet preserved as product name; 'Architecture' lowercased"
  },
  {
    "line": 2014,
    "before": "### The Iron Law of Training Performance",
    "after": "### The iron law of training performance",
    "confidence": "high",
    "reason": "iron law lowercased per §10.3"
  }
]
```

**Do not edit the file.** Only return the JSON. The orchestrator will apply the edits with the same five safety checks (no null bytes, no leftover sentinels, byte delta matches, quarto structural delta zero, no new issues introduced) that the script lane uses.

**If you find no violations, return `[]` (empty JSON array).**
