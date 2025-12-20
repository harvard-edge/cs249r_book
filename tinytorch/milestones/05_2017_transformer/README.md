# Milestone 05: The Transformer Era (2017)

## Historical Context

In 2017, Vaswani et al. published **"Attention Is All You Need,"** showing that attention mechanisms alone (no RNNs, no convolutions!) could achieve state-of-the-art results on sequence tasks. This breakthrough:

- Replaced RNNs/LSTMs for sequence modeling
- Enabled parallel training (unlike sequential RNNs)
- Scaled to massive datasets and model sizes
- Launched the era of GPT, BERT, and modern LLMs

Transformers didn't just improve NLP - they unified vision, language, and multimodal AI. Now it's your turn to build one from scratch using YOUR Tiny🔥Torch!

## What You're Building

Character-level transformer models for text generation:
1. **Question Answering** - Train on TinyTalks Q&A dataset
2. **Dialogue Generation** - Generate coherent conversational responses

## Required Modules

**Run after Module 13** (Complete transformer stack)

| Module | Component | What It Provides |
|--------|-----------|------------------|
| Module 01 | Tensor | YOUR data structure with autograd |
| Module 02 | Activations | YOUR ReLU/GELU activations |
| Module 03 | Layers | YOUR Linear layers |
| Module 04 | Losses | YOUR CrossEntropyLoss |
| Module 05 | DataLoader | YOUR data batching |
| Module 06 | Autograd | YOUR automatic differentiation |
| Module 07 | Optimizers | YOUR Adam optimizer |
| **Module 10** | **Tokenization** | **YOUR CharTokenizer** |
| **Module 11** | **Embeddings** | **YOUR token + positional embeddings** |
| **Module 12** | **Attention** | **YOUR multi-head self-attention** |
| **Module 13** | **Transformers** | **YOUR LayerNorm + TransformerBlock + GPT** |

## Milestone Structure

This milestone uses **progressive difficulty** with 3 scripts:

### ⭐ 00_vaswani_attention_proof.py (START HERE!)
**Purpose:** PROVE your attention mechanism works

- **Dataset:** Auto-generated sequences (no files needed!)
- **Task:** Reverse sequences `[1,2,3,4] → [4,3,2,1]`
- **From Paper:** "Attention is All You Need" validation task
- **Training Time:** ~30 seconds
- **Expected:** 95%+ accuracy
- **Key Learning:** "My attention is computing relationships!"

**Why This Is THE Test:**
- IMPOSSIBLE without attention working
- Trains in 30 seconds (instant gratification!)
- Binary pass/fail (95%+ or broken)
- Proves Q·K·V computation works

**🎯 Run this FIRST to verify your attention before complex tasks!**

### 01_vaswani_generation.py
**Purpose:** Apply attention to real language (Q&A)

- **Dataset:** TinyTalks (17.5 KB, 5 difficulty levels)
- **Task:** Learn to answer questions (Q: ... A: ...)
- **Architecture:** Character-level GPT with attention
- **Expected:** Coherent responses in 3-5 minutes
- **Key Learning:** "Attention learns long-range dependencies!"

**Why TinyTalks?**
- Fast training = instant feedback
- Clear Q&A format = easy to verify learning
- Progressive difficulty = see capability growth
- Ships with TinyTorch = no downloads

### 02_vaswani_dialogue.py
**Purpose:** Generate natural conversational text

- **Dataset:** Same TinyTalks, different framing
- **Task:** Multi-turn dialogue generation
- **Expected:** Context-aware responses
- **Key Learning:** "Transformers capture conversation flow!"

**What Makes This Special:**
- Same model architecture as GPT/ChatGPT (scaled down)
- YOUR implementation from scratch (no magic!)
- Proves attention mechanism works

## Expected Results

| Script | Task | Context Length | Success Criteria | Training Time |
|--------|------|----------------|------------------|---------------|
| 01 (Q&A) | Answer questions | 128 chars | Loss < 1.5, sensible word choices | 3-5 min |
| 02 (Dialogue) | Multi-turn chat | 128 chars | Maintains topic coherence, loss < 1.5 | 3-5 min |

## Key Learning: Why Attention Revolutionized AI

Transformers solve the fundamental problems of RNNs:

### Problem with RNNs:
- **Sequential processing** → Can't parallelize (slow training)
- **Vanishing gradients** → Struggles with long sequences
- **Fixed hidden state** → Information bottleneck

### Transformer Solution:
- **Attention mechanism** → "Look at ANY position, weighted by relevance"
- **Parallel processing** → Process entire sequence at once
- **Direct connections** → Every position can attend to every other position

**The Key Insight:**
```
RNN:  Hidden state carries ALL information (bottleneck!)
      h[t] = f(h[t-1], x[t])  ← Sequential, lossy

Attention: Directly access ANY past position (no bottleneck!)
          output[i] = Σ attention[i,j] × value[j]  ← Parallel, lossless
```

This is why GPT, BERT, T5, and modern LLMs all use transformers!

## Running the Milestone

```bash
cd milestones/05_2017_transformer

# Step 1: Q&A generation (run after Module 13)
python 01_vaswani_generation.py --epochs 5 --batch-size 4

# Step 2: Dialogue generation (run after Module 13)
python 02_vaswani_dialogue.py --epochs 5 --batch-size 4
```

**Optional flags:**
- `--levels N` - Use first N difficulty levels (1-5)
- `--embed-dim D` - Embedding dimension (default: 64)
- `--num-layers L` - Number of transformer blocks (default: 3)
- `--num-heads H` - Attention heads (default: 4)

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
- How GPT-style autoregressive generation works

**You've built the architecture powering modern AI!**

---

**Note for Next Milestone:** You can now BUILD transformers, but can you OPTIMIZE them for production? Milestone 06 (MLPerf) teaches systematic optimization: profiling → compression → acceleration!
