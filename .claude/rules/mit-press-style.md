---
paths:
  - "book/**"
---

# MIT Press House Style -- Copy Editor's Style Sheet

Job Number: 837-149678
Copy Editor: Pamela Hunt
Reference: The Chicago Manual of Style, 17th ed.; Webster's New Collegiate Dictionary, 11th ed.

## 1. Percent

- Spell out "percent" in body prose: `94 percent accuracy`
- `%` is OK inside tables, equations (`$60\%$`), code blocks, and figure captions where space is constrained
- When inline Python produces a number followed by `%`, change to: `` `{python} val_str` percent ``
- `\%` in LaTeX stays as-is

## 2. Em Dashes

- Closed em dashes only: `word---word` (no spaces before or after)
- Wrong: `word --- word`
- Maximum one em dash per paragraph; prefer colons for introducing consequences/explanations
- Ellipses in interviews showing pauses should become em dashes

## 3. Capitalization

Concept terms that are NOT proper nouns must be **lowercase** in body prose.

### Lowercase these terms

- iron law (not Iron Law)
- degradation equation (not Degradation Equation)
- verification gap (not Verification Gap)
- ML node (not ML Node; "ML" stays caps)
- bitter lesson (the concept; but "The Bitter Lesson" when citing Sutton's essay title)
- chapter 12, section 3.2, figure 1.1, table 1.1 (lowercase in prose references)
- machine learning operations (generic usage)
- transformer (neural network) -- lowercase per style sheet
- four pillars framework

### Keep capitalized

- D-A-M (acronym)
- TinyML (brand)
- AllReduce, FlashAttention, PagedAttention (API/product names)
- CUDA, cuDNN, PyTorch, TensorFlow, MATLAB (product names)
- ImageNet, BERT, GPT-4 (proper nouns / model names)
- Stable Diffusion, DALL-E (product names)
- Tensor Core, Tensor Processing Unit (TPU) (product names)
- Weights & Biases (company name)

### Exceptions -- keep capitals even for lowercase terms

1. Start of sentence
2. Inside `**bold**` at first definition of a term
3. Inside `***triple bold***` in definition callouts
4. In H1/H2 section headers (headline style)
5. In `\index{}` entries
6. In callout `title=` attributes
7. In bold table headers

### Heading style

- H1 and H2: headline style (capitalize principal words)
- H3 and below: sentence style (capitalize first word only, plus proper nouns)
- Figure captions and table titles: sentence style; no colon after figure/table number

## 4. Abbreviations

Expand every abbreviation on **first use per chapter**. After that, abbreviation alone is fine. Expansion resets at each chapter boundary.

Pattern: `convolutional neural network (CNN)` on first use, then `CNN`.

### Canonical forms (alphabetical)

- AST: abstract syntax tree (AST)
- AOT: ahead-of-time (AOT)
- AUC: area under the [ROC] curve (AUC)
- BPTT: backpropagation through time (BPTT)
- BLAS: Basic Linear Algebra Subprograms (BLAS)
- CI/CD: continuous integration/continuous deployment (CI/CD)
- CNN: convolutional neural network (CNN)
- CTM: continuous therapeutic monitoring (CTM)
- DAG: directed acyclic graph (DAG)
- DCE: dead-code elimination (DCE)
- ELT: extract, load, transform (ELT)
- ETL: extract, transform, load (ETL)
- FFT: fast Fourier transform (FFT)
- GDPR: General Data Protection Regulation (GDPR)
- GELU: Gaussian Error Linear Unit (GELU)
- GEMM: general matrix multiply (GEMM)
- HIPAA: Health Insurance Portability and Accountability Act (HIPAA)
- HOG: histogram of oriented gradients (HOG)
- i.i.d.: independent and identically distributed (i.i.d.) -- with periods
- ICR: information-compute ratio (ICR)
- ILSVRC: ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
- IOPS: input/output operations per second (IOPS)
- IR: intermediate representation (IR)
- JIT: just-in-time (JIT)
- JSON: JavaScript Object Notation (JSON)
- KWS: keyword spotting (KWS)
- LLMs: large language models (LLMs)
- MAC: multiply-accumulate (MAC)
- MIPS: microprocessor without interlocked pipelined stages (MIPS)
- MLPs: multilayer perceptrons (MLPs)
- MoE: mixture-of-experts (MoE)
- NAS: neural architecture search (NAS)
- NaN: not a number (NaN)
- NVMe: Non-Volatile Memory Express (NVMe)
- ONNX: Open Neural Network Exchange (ONNX)
- OTA: over-the-air (OTA) update
- PTX: Parallel Thread Execution (PTX)
- RBAC: role-based access control (RBAC)
- ReLU: rectified linear unit (ReLU)
- RISC: reduced instruction set computer (RISC)
- RNN: recurrent neural network (RNN)
- ROC: receiver operating characteristic (ROC) curve
- SIFT: scale-invariant feature transform (SIFT)
- SIMD: single instruction, multiple data (SIMD)
- SLA: service level agreement (SLA)
- SoC: system on chip (SoC)
- SSA: static single-assignment (SSA)
- TCO: total cost of ownership (TCO)
- TFDV: TensorFlow Data Validation (TFDV)
- TPU: Tensor Processing Unit (TPU)
- UAT: universal approximation theorem (UAT)
- ViT: vision transformer (ViT)

### Special cases

- CUDA, cuDNN: no expansion needed (well-known)
- i.i.d.: always with periods; expand in every chapter on first use
- vs.: always abbreviated with period; never write "versus"
- Adam: Adaptive Moment Estimation (Adam)

## 5. Bibliography

- Every `@inproceedings` must have a `publisher` field
- Every `@article` must have a `journal` field
- Include `pages` and `doi` when available
- Do NOT use em dashes for repeat author names
- Letter-by-letter alphabetical order
- Confirm all URLs are live
- Publisher locations (cities) removed for consistency

## 6. References in Prose

- "chapter 12" not "Chapter 12"
- "section 3.2" not "Section 3.2"
- "figure 1.1" not "Figure 1.1" (Quarto `@fig-` handles this)
- "table 1.1" not "Table 1.1" (Quarto `@tbl-` handles this)
- "fig. 1.1" acceptable in technical context

## 7. Slashes

- No spaces around slashes: `training/inference` not `training / inference`

## 8. Numbers

- Spell out one through nine in body prose
- Use digits for 10 and above
- Always use digits with units: "3 GB", "7 ms"
- All page/year ranges with full digits: 1992--1993, 175--185
- Space between date/bracketed date in citations: 1978 [1964]

## 9. Spelling Dictionary

Canonical forms (first spelling in Webster's):

- AllReduce
- backpropagation through time (BPTT)
- bitter lesson (lowercase)
- break-even (adj)
- checkpointing (n)
- ClinAIOps
- cloud ML
- coin-cell battery
- coreset (n)
- D-A-M taxonomy
- data center (two words)
- dataset (one word)
- decision-making (n)
- dtype (n)
- earbud
- e-commerce
- edge ML
- endpoint (one word)
- engineer-month (n)
- FlashAttention
- four pillars framework
- front end (n)
- f-string
- healthcare (n, adj)
- im2col
- iron law (lowercase)
- k-nearest neighbors
- lifecycle (n, adj)
- multi-chip (adj)
- nonzero (one word)
- one-hot encoding
- open-source (adj)
- PagedAttention
- perceptron
- pre- (closed up: pretrained, preprocessing)
- pytree (n)
- real time (n), real-time (adj)
- round trip (n), round-trip (adj)
- scatterplot (one word)
- smartphone (one word)
- smart speaker (two words)
- smartwatch (one word)
- softmax (function)
- space-time (n)
- speedup (n)
- time-series (adj)
- TinyML
- total cost of ownership (TCO)
- trade-off (n)
- training-serving skew
- transformer (neural network, lowercase)
- traveler (not traveller)
- vehicle-to-everything (V2X) communication
- wake-word detector
- x-axis, y-axis
- ZIP code

## 10. Punctuation

- Serial (Oxford) comma: always
- Comma after e.g. and i.e.
- Double quotation marks; period/comma inside, colon/semicolon outside
- Single quotation marks for quote within a quote
- Spaced periods between ellipsis points; four points for end of sentence
- Periods after table and figure titles if complete sentences
- No ellipses at beginning or end of quotes
- Contractions OK unless excessive

## 11. Abbreviations in Running Text

- Spell out "for example" instead of e.g. in running text (e.g. OK inside parentheses)
- Spell out "that is" instead of i.e. in running text (i.e. OK inside parentheses)
- Replace "etc." with "and so on" in running text (etc. OK inside parentheses and notes)
- Space between initials: B. F. Skinner
- US/UK OK as nouns (consistent usage)
- Abbreviate units of measure

## 12. Other

- Close up compounds formed with common prefixes (non-, re-, pre-; see Webster's)
- Italic only for emphasis (not bold or underlining)
- Glossary term capitalization: initial cap only if proper noun
- Remove space following hyphens and dashes
- Labels must be consistent
- Epigraph sources set flush left, preceded by em dash
- TOC is titled "Contents"
- Chapter numbers only on chapter opening pages and in TOC
