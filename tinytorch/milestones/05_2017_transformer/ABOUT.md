# Milestone 05: The Transformer Era (2017)

```{tip} What You'll Learn
- Why attention beats sequential processing (direct access, no bottleneck)
- How transformers handle long-range dependencies that defeat RNNs
- The architecture that powers GPT, BERT, and every modern LLM
```

## Overview

**Imagine compressing an entire book into a single number.**

That's essentially what RNNs do. No matter how long the input—a sentence, a paragraph, an entire novel—it all gets squeezed through a fixed-size hidden state. Information from the beginning gradually fades as new tokens arrive. By the time you process token 1000, token 1 is nearly forgotten.

In 2017, Vaswani et al. asked: *what if we just... didn't?* Their paper "Attention Is All You Need" proved that attention mechanisms alone could achieve state-of-the-art results. No sequential bottleneck. No information loss. Any token can directly attend to any other token.

This launched the era of GPT, BERT, ChatGPT, and every modern LLM. This milestone builds a character-level transformer using YOUR TinyTorch implementations.

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
| 01-08 | Foundation + Training | Tensor, Layers, DataLoader, Training |
| 10 | Tokenization | YOUR CharTokenizer |
| 11 | Embeddings | YOUR token + positional embeddings |
| 12 | Attention | YOUR multi-head self-attention |
| 13 | Transformers | YOUR LayerNorm + TransformerBlock |

## Running the Milestone

Before running, ensure you have completed Modules 01-13. You can check your progress:

```bash
tito module status
```

```bash
cd milestones/05_2017_transformer

# Step 0: Prove attention works
python 00_vaswani_attention_proof.py
# Expected: 95%+ on sequence reversal

# Step 1: Q&A generation
python 01_vaswani_generation.py --epochs 5
# Expected: Loss < 1.5, coherent responses

# Step 2: Dialogue generation
python 02_vaswani_dialogue.py --epochs 5
# Expected: Context-aware responses
```

:::{admonition} Why Start with Script 00?
:class: note
Script 00 is the critical validation. Sequence reversal is practically impossible without working attention, gives instant feedback (30 seconds), and is binary pass/fail (95%+ or broken). If 00 fails, debug your attention. If 00 passes, 01-02 will work.
:::

## Expected Results

| Script | Task | Success Criteria | Time |
|--------|------|------------------|------|
| 00 (Attention Proof) | Reverse sequences | 95%+ accuracy | 30 sec |
| 01 (Q&A) | Answer questions | Loss < 1.5, sensible words | 3-5 min |
| 02 (Dialogue) | Multi-turn chat | Topic coherence | 3-5 min |

## The Aha Moment: Direct Access Everywhere

**Watch Script 00 prove attention works—in 30 seconds.**

Sequence reversal is practically impossible for simple sequential models. To output the first character, you need to remember the last character of input. To output the second character, you need the second-to-last. The entire sequence must be held in memory simultaneously.

RNNs struggle with this. Attention makes it trivial:

```
RNN:       h[t] = f(h[t-1], x[t])          # Sequential, lossy
Attention: out[i] = sum(attn[i,j] * v[j])  # Parallel, direct
```

**The moment you realize**: YOUR attention mechanism lets any position directly access any other position. No bottleneck. No forgetting. This is why transformers can handle entire documents, codebases, even books.

This solves the fundamental RNN problems:
- Sequential processing → Parallel (faster training)
- Vanishing gradients → Direct connections (better long-range)
- Fixed hidden state → Dynamic attention (no bottleneck)

## YOUR Code Powers This

This is the capstone of the Architecture Tier. Every component comes from YOUR implementations:

| Component | Your Module | What It Does |
|-----------|-------------|--------------|
| `CharTokenizer` | Module 10 | YOUR character-level tokenization |
| `TokenEmbedding` | Module 11 | YOUR learned token representations |
| `PositionalEmbedding` | Module 11 | YOUR position encodings |
| `MultiHeadAttention` | Module 12 | YOUR self-attention mechanism |
| `TransformerBlock` | Module 13 | YOUR attention + feedforward blocks |
| `LayerNorm` | Module 13 | YOUR normalization layers |

**No PyTorch. No HuggingFace. Just YOUR code generating coherent text.**

## Historical Context

The original "Attention Is All You Need" paper used sequence-to-sequence tasks like translation. TinyTalks provides a similar Q&A format at character level.

This architecture (with scaling) powers:
- GPT-2, GPT-3, GPT-4, ChatGPT
- BERT, RoBERTa, T5
- Vision Transformers (ViT)
- Multimodal models (CLIP, Flamingo)

## Systems Insights

- **Memory**: O(n²) for attention matrix (sequence length squared)
- **Compute**: Highly parallelizable (unlike RNNs)
- **Architecture**: Position info via embeddings (no inherent ordering)

## What's Next

You can BUILD transformers, but can you OPTIMIZE them? Milestone 06 (MLPerf) teaches systematic optimization: profiling, compression, and acceleration for production deployment.

## Further Reading

- **The Paper**: Vaswani et al. (2017). "Attention Is All You Need"
- **Illustrated Guide**: http://jalammar.github.io/illustrated-transformer/
- **GPT-2**: Radford, A. et al. (2019). "Language Models are Unsupervised Multitask Learners"
