# Milestone 05: The Transformer Era (2017-2022)

## Historical Context

In 2017, Vaswani et al. published **"Attention Is All You Need,"** showing that attention mechanisms alone (no recurrence, no convolutions!) could achieve state-of-the-art results on sequence tasks. In 2020, Brown et al. (OpenAI) published **"Language Models are Few-Shot Learners"** (GPT-3), proving that autoregressive next-token prediction at scale produces emergent, general-purpose reasoning. In late 2022, OpenAI launched **ChatGPT**, demonstrating to the entire world that this generative transformer foundation could interact seamlessly with human thought.

Behind modern LLMs sits this exact mathematical and systems engine:
1. **Autoregressive Next-Token Prediction:** Teacher forcing with CrossEntropyLoss.
2. **Causal Self-Attention:** Masked attention preventing future token leakage.
3. **Pre-LayerNorm Residual Highway:** Clean gradient propagation through deep blocks.
4. **Systems Serving Efficiency:** The KV-cache (Module 18) and quantization (Module 15) that make generative sampling fast and interactive in production.

Now it's your turn to train TinyGPT from scratch on Shakespeare using YOUR Tiny🔥Torch!

## What You're Building

**Part 1 (`01_tinygpt_shakespeare.py` - Default):**
Train **TinyGPT** from scratch on Shakespeare. Brings together your tokenization, embeddings, causal multi-head self-attention, stacked transformer decoder blocks, cross-entropy loss, and autoregressive generation loop with temperature and top-k sampling.

**Part 2 (`02_vaswani_attention.py` - Attention Proof):**
Prove your attention mechanism on three synthetic sequence challenges (reversal, copying, prefix-controlled mixed tasks) without external data dependencies.

**Part 3 (`03_tinygpt_code_completion.py` - TinyCopilot):**
Train **TinyGPT** on Python code using the TinyPy dataset to complete functions, preserve indentation, and generate syntactically valid code verified with Abstract Syntax Tree parsing.

**Part 4 (`04_tinygpt_chat.py` - Conversational Q&A & Overfitting Detective):**
Train **TinyGPT** as a conversational assistant explaining TinyTorch concepts ("TinyTorch Teaching TinyTorch"). Run the Overfitting Detective experiment to observe the divergence between training loss (rote memorization) and held-out test loss.

## Required Modules

**Run after Module 13** (Complete transformer stack)

<table width="100%">
  <thead>
<tr>
<th width="25%"><b>Module</b></th>
<th width="25%">Component</th>
<th width="50%">What It Provides</th>
</tr>
</thead>
<tbody>
<tr><td><b>Module 01</b></td><td>Tensor</td><td>YOUR data structure with autograd</td></tr>
<tr><td><b>Module 02</b></td><td>Activations</td><td>YOUR ReLU/GELU activations</td></tr>
<tr><td><b>Module 03</b></td><td>Layers</td><td>YOUR Linear layers</td></tr>
<tr><td><b>Module 04</b></td><td>Losses</td><td>YOUR CrossEntropyLoss (sequence-shaped)</td></tr>
<tr><td><b>Module 05</b></td><td>DataLoader</td><td>YOUR Dataset/DataLoader batching</td></tr>
<tr><td><b>Module 06</b></td><td>Autograd</td><td>YOUR automatic differentiation</td></tr>
<tr><td><b>Module 07</b></td><td>Optimizers</td><td>YOUR AdamW optimizer</td></tr>
<tr><td><b>Module 08</b></td><td>Training</td><td>YOUR Trainer training loop</td></tr>
<tr><td><b>Module 10</b></td><td>Tokenization</td><td>YOUR Tokenizer</td></tr>
<tr><td><b>Module 11</b></td><td>Embeddings</td><td>YOUR token + positional embeddings</td></tr>
<tr><td><b>Module 12</b></td><td>Attention</td><td>YOUR multi-head self-attention</td></tr>
<tr><td><b>Module 13</b></td><td>Transformers</td><td>YOUR LayerNorm + TransformerBlock + TinyGPT</td></tr>
</tbody>
</table>

## Running the Milestone

**Run via TITO (recommended):**

```bash
# Runs Part 1 (TinyGPT on Shakespeare) by default
tito milestone run 05

# Or use convenience aliases:
tito milestone run transformer
tito milestone run tinygpt
```

**Run Part 2 (Attention Sequence Routing):**

```bash
tito milestone run 05 --part 2
```

**Run Part 3 (TinyCopilot Code Generation):**

```bash
tito milestone run 05 --part 3
```

**Run Part 4 (Conversational Q&A & Overfitting Detective):**

```bash
tito milestone run 05 --part 4
```

**Or run directly:**

```bash
# Part 1: TinyGPT
python3 milestones/05_2017_transformer/01_tinygpt_shakespeare.py --quick

# Part 2: Sequence tasks
python3 milestones/05_2017_transformer/02_vaswani_attention.py

# Part 3: TinyCopilot code generation (Python)
python3 milestones/05_2017_transformer/03_tinygpt_code_completion.py --quick

# Part 4: Conversational Q&A and Overfitting Detective
python3 milestones/05_2017_transformer/04_tinygpt_chat.py --quick
```

## Part 3: TinyCopilot with TinyPy (`03_tinygpt_code_completion.py`)

### Why TinyCopilot is the Natural Evolution from Shakespeare

Training an autoregressive model on natural prose like Shakespeare demonstrates statistical language structure: meter, vocabulary, and dialogue cadence. However, natural language is forgiving. A typo, an extra space, or an unexpected comma rarely prevents comprehension.

Code completion represents the natural evolution in sequence modeling for three foundational reasons:

1. **Rigid Syntactic Constraints:**
   Programming languages possess strict context-free grammars. Every opening parenthesis, bracket, and quotation mark must balance. In Python, indentation defines block scoping, and keywords such as `def`, `class`, `return`, and `if` demand exact downstream syntax. If an attention head fails to track an open delimiter, the generated code fails to parse.

2. **Long-Range Semantic Dependencies:**
   In code, tokens defined dozens or hundreds of positions earlier, such as function parameters or imported module identifiers, must be referenced with exact character fidelity later in the sequence. Causal self-attention provides the direct constant-distance path needed to bind variable names across statements without the gradient attenuation of recurrent networks.

3. **Bridging to Modern Developer Assistants:**
   The exact architecture trained here reflects the foundation of modern coding assistants, such as OpenAI Codex and GitHub Copilot. Autoregressive transformers trained with next-token cross-entropy loss generalize naturally from human conversational prose to algorithmic source code.

### Commands to Run

**Quick verification run (recommended for fast turnaround):**

```bash
python3 03_tinygpt_code_completion.py --quick
```

Or from the TinyTorch repository root:

```bash
python3 milestones/05_2017_transformer/03_tinygpt_code_completion.py --quick
```

**Run via TITO CLI:**

```bash
tito milestone run 05 --part 3
```

**Full training run:**

```bash
python3 03_tinygpt_code_completion.py --epochs 10
```

## Part 4: Conversational Concepts & Overfitting Detective (`04_tinygpt_chat.py`)

### TinyTorch Teaching TinyTorch and the Generalization Boundary

In Part 4, you train TinyGPT on curated question and answer pairs covering TinyTorch systems concepts (`tinytalks_tinytorch.txt`). This milestone part introduces two essential concepts:

1. **Conversational Dialogue Structuring:**
   The model learns the statistical structure of dialogue formatted with `Q: ` and `A: `. Autoregressive sampling completes the answer when primed with a conceptual question.

2. **The Overfitting Detective Experiment:**
   The TinyGPT model has approximately 119,000 learnable parameters, while the TinyTalks concept corpus is approximately 13.7 KB (about 13,700 characters). Because the model has more parameter capacity than the dataset has characters, it can easily achieve near-zero training loss through rote memorization.
   By splitting the corpus into an explicit training split (64 concept pairs) and a held-out test split (17 unseen concept pairs), students observe the classic machine learning phenomenon:
   - Early Epochs (Underfitting): Both train and test losses decrease as the model learns basic character statistics and formatting.
   - Middle Epochs (Sweet Spot): Train loss drops further, and test loss achieves its minimum.
   - Late Epochs (Overfitting): Train loss drops below 0.50 as the model memorizes the training questions, while test loss rises as the generalization gap widens.

### Commands to Run

```bash
# Fast verification run
python3 milestones/05_2017_transformer/04_tinygpt_chat.py --quick

# Interactive chat mode
python3 milestones/05_2017_transformer/04_tinygpt_chat.py --interactive
```

## Expected Results

| Part | Script | Task | Success Criteria |
|------|--------|------|------------------|
| 1 (Default) | `01_tinygpt_shakespeare.py` | Shakespeare Next-Token Prediction | Loss drops from ~3.0 to < 1.0 (PPL < 3.0); generates coherent verse |
| 2 (Optional) | `02_vaswani_attention.py` | Synthetic Reversal / Copy / Mixed | High accuracy (>95% reversal, >95% copy, >90% mixed) |
| 3 (TinyCopilot) | `03_tinygpt_code_completion.py` | TinyCopilot Code Generation | Loss drops steadily; generates valid syntax and completes functions |
| 4 (Chat) | `04_tinygpt_chat.py` | Conversational Q&A & Overfitting | Train loss drops to < 0.60; detects generalization gap on held-out split |

## Achievement Unlocked

After completing this milestone, you'll understand:
- How causal self-attention computes context-aware representations without future leakage
- Why teacher forcing trains all sequence positions in parallel
- How temperature and top-k sampling turn raw logits into generative prose
- Why autoregressive decode creates the prefix recomputation bottleneck that Part III will optimize
- How autoregressive transformers generalize from literary prose to formal code completion
- How micro-models transition from representation learning to rote memorization on small datasets

**You've validated the architecture powering modern AI!**

---

**Note for Next Milestone:** You can now BUILD generative transformers, but can you OPTIMIZE them for production? Milestone 06 (MLPerf Benchmarks) teaches systematic optimization: profiling → compression → KV-cache acceleration on TinyGPT!
