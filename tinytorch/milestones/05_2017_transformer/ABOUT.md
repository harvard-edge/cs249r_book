# Milestone 05: The Transformer Era (2017)

**ARCHITECTURE TIER** | Difficulty: 4/4 | Time: 60-120 min | Prerequisites: Modules 01-13

## Overview

In 2017, Vaswani et al.'s **"Attention Is All You Need"** showed that attention mechanisms alone could achieve state-of-the-art results - no RNNs, no convolutions. This launched the era of GPT, BERT, ChatGPT, and modern LLMs.

This milestone builds a character-level transformer using YOUR implementations.

## What You'll Build

Transformer models for text generation:
1. **Attention Proof**: Verify your attention works on sequence reversal
2. **Q&A Generation**: Train on TinyTalks conversational dataset
3. **Dialogue**: Multi-turn conversation generation

```
Tokens --> Embeddings --> [Attention --> FFN] x N --> Output
```

## Prerequisites

| Module | Component | What It Provides |
|--------|-----------|------------------|
| 01-08 | Foundation + Training | Complete training pipeline |
| 10 | Tokenization | YOUR CharTokenizer |
| 11 | Embeddings | YOUR token + positional embeddings |
| 12 | Attention | YOUR multi-head self-attention |
| 13 | Transformers | YOUR LayerNorm + TransformerBlock |

## Running the Milestone

```bash
cd milestones/05_2017_transformer

# Step 0: Prove attention works (~30 seconds)
python 00_vaswani_attention_proof.py
# Expected: 95%+ on sequence reversal

# Step 1: Q&A generation (3-5 min)
python 01_vaswani_generation.py --epochs 5
# Expected: Loss < 1.5, coherent responses

# Step 2: Dialogue generation (3-5 min)
python 02_vaswani_dialogue.py --epochs 5
# Expected: Context-aware responses
```

## Expected Results

| Script | Task | Success Criteria | Time |
|--------|------|------------------|------|
| 00 (Attention Proof) | Reverse sequences | 95%+ accuracy | 30 sec |
| 01 (Q&A) | Answer questions | Loss < 1.5, sensible words | 3-5 min |
| 02 (Dialogue) | Multi-turn chat | Topic coherence | 3-5 min |

## Key Learning

**Attention enables direct access to any position.** Unlike RNNs (sequential, bottlenecked), transformers can attend to ANY part of the input with learned relevance weights.

```
RNN:       h[t] = f(h[t-1], x[t])     # Sequential, lossy
Attention: out[i] = sum(attn[i,j] * v[j])  # Parallel, direct
```

This solves the fundamental RNN problems:
- Sequential processing --> Parallel (faster training)
- Vanishing gradients --> Direct connections (better long-range)
- Fixed hidden state --> Dynamic attention (no bottleneck)

## Systems Insights

- **Memory**: O(n^2) for attention matrix (sequence length squared)
- **Compute**: Highly parallelizable (unlike RNNs)
- **Architecture**: Position info via embeddings (no inherent ordering)

## Why Start with Attention Proof?

Script 00 is the **critical validation**. Sequence reversal is:
- **Impossible** without working attention
- **Instant** feedback (30 seconds)
- **Binary** pass/fail (95%+ or broken)

If 00 fails, debug your attention. If 00 passes, 01-02 will work.

## Historical Context

The original "Attention Is All You Need" paper used sequence-to-sequence tasks like translation. TinyTalks provides a similar Q&A format at character level.

This architecture (with scaling) powers:
- GPT-2, GPT-3, GPT-4, ChatGPT
- BERT, RoBERTa, T5
- Vision Transformers (ViT)
- Multimodal models (CLIP, Flamingo)

## What's Next

You can BUILD transformers, but can you OPTIMIZE them? Milestone 06 (MLPerf) teaches systematic optimization: profiling, compression, and acceleration for production deployment.

## Further Reading

- **The Paper**: Vaswani et al. (2017). "Attention Is All You Need"
- **Illustrated Guide**: http://jalammar.github.io/illustrated-transformer/
- **GPT Papers**: Radford et al. (2018, 2019, 2020). GPT-1/2/3
