# Milestone 05: The Transformer Era (2017–2022)

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

**Or run directly:**

```bash
# Part 1: TinyGPT
python3 milestones/05_2017_transformer/01_tinygpt_shakespeare.py --quick

# Part 2: Sequence tasks
python3 milestones/05_2017_transformer/02_vaswani_attention.py
```

## Expected Results

| Part | Script | Task | Success Criteria |
|------|--------|------|------------------|
| 1 (Default) | `01_tinygpt_shakespeare.py` | Shakespeare Next-Token Prediction | Loss drops from ~3.0 to < 1.0 (PPL < 3.0); generates coherent verse |
| 2 (Optional) | `02_vaswani_attention.py` | Synthetic Reversal / Copy / Mixed | High accuracy (>95% reversal, >95% copy, >90% mixed) |

## Achievement Unlocked

After completing this milestone, you'll understand:
- How causal self-attention computes context-aware representations without future leakage
- Why teacher forcing trains all sequence positions in parallel
- How temperature and top-k sampling turn raw logits into generative prose
- Why autoregressive decode creates the prefix recomputation bottleneck that Part III will optimize

**You've validated the architecture powering modern AI!**

---

**Note for Next Milestone:** You can now BUILD generative transformers, but can you OPTIMIZE them for production? Milestone 06 (MLPerf Benchmarks) teaches systematic optimization: profiling → compression → KV-cache acceleration on TinyGPT!
