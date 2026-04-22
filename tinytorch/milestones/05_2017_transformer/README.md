# Milestone 05: The Transformer Era (2017)

## Historical Context

In 2017, Vaswani et al. published **"Attention Is All You Need,"** showing that attention mechanisms alone (no RNNs, no convolutions!) could achieve state-of-the-art results on sequence tasks. This breakthrough:

- Replaced RNNs/LSTMs for sequence modeling
- Enabled parallel training (unlike sequential RNNs)
- Scaled to massive datasets and model sizes
- Launched the era of GPT, BERT, and modern LLMs

Transformers didn't just improve NLP - they unified vision, language, and multimodal AI. Now it's your turn to build one from scratch using YOUR Tiny🔥Torch!

## What You're Building

**Primary milestone (what `tito milestone run 05` executes):** prove YOUR attention and transformer stack on **three synthetic sequence challenges** in a single script—reversal, copying, and prefix-controlled mixed tasks (see script docstring for details).

**Shipped language data:** the **TinyTalks** conversational Q&A corpus lives under `datasets/tinytalks/` (see that README) for character-level / transformer experiments and teaching materials. The checked-in milestone script uses in-script synthetic data so you can validate attention without extra file dependencies.

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
<tr><td><b>Module 04</b></td><td>Losses</td><td>YOUR CrossEntropyLoss</td></tr>
<tr><td><b>Module 05</b></td><td>DataLoader</td><td>YOUR data batching</td></tr>
<tr><td><b>Module 06</b></td><td>Autograd</td><td>YOUR automatic differentiation</td></tr>
<tr><td><b>Module 07</b></td><td>Optimizers</td><td>YOUR Adam optimizer</td></tr>
<tr><td><b>Module 10</b></td><td>Tokenization</td><td>YOUR CharTokenizer</td></tr>
<tr><td><b>Module 11</b></td><td>Embeddings</td><td>YOUR token + positional embeddings</td></tr>
<tr><td><b>Module 12</b></td><td>Attention</td><td>YOUR multi-head self-attention</td></tr>
<tr><td><b>Module 13</b></td><td>Transformers</td><td>YOUR LayerNorm + TransformerBlock + GPT</td></tr>
</tbody>
</table>

## Milestone Script (canonical)

### `01_vaswani_attention.py`

**Purpose:** PROVE your attention mechanism works on structured sequence tasks (reversal, copy, mixed).

- **Dataset:** Synthetic sequences generated in-script (no separate download)
- **Success criteria:** See script header (typically ~95% / ~95% / ~90% on the three challenges)

**Run via TITO (recommended):**

```bash
tito milestone run 05
```

**Or directly:**

```bash
cd milestones/05_2017_transformer
python 01_vaswani_attention.py
```

`python 01_vaswani_attention.py --help` lists optional flags (embedding dim, layers, heads, etc.).

### TinyTalks (`datasets/tinytalks/`)

Conversational Q&A text for transformer teaching and extensions—**not** required for `tito milestone run 05` as configured today. Documentation: `datasets/tinytalks/README.md`.

## Expected Results

| Phase | Task | What “good” looks like |
|-------|------|-------------------------|
| 1 | Reversal | High accuracy; anti-diagonal attention pattern |
| 2 | Copy | High accuracy; identity-style pattern |
| 3 | Mixed | Respects `[R]` vs `[C]` prefix |

## Key Learning: Why Attention Revolutionized AI

Transformers solve fundamental RNN limitations: sequential bottlenecks, vanishing signal, fixed hidden-state capacity. Attention lets every position attend to every other in parallel—scaled up, that is the core of GPT-style models.

## Running the Milestone

```bash
tito milestone run 05
```

## Further Reading

- **The Paper**: Vaswani et al. (2017). "Attention Is All You Need"
- **Illustrated Transformer**: http://jalammar.github.io/illustrated-transformer/
- **GPT Evolution**: Radford et al. (2018, 2019, 2020). GPT-1/2/3 papers
- **BERT**: Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"

## Achievement Unlocked

After completing this milestone, you'll understand:
- How self-attention computes context-aware representations
- Why transformers parallelize better than RNNs
- What positional embeddings do (give position information)
- How GPT-style autoregressive generation builds on these ideas

**You've validated the architecture powering modern AI!**

---

**Note for Next Milestone:** You can now BUILD transformers, but can you OPTIMIZE them for production? Milestone 06 (MLPerf) teaches systematic optimization: profiling → compression → acceleration!
