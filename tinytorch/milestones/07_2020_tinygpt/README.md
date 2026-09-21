# Milestone 07: The Generative LLM Revolution (2020)

## Historical Context

In 2020, Brown et al. (OpenAI) published **"Language Models are Few-Shot Learners,"** introducing GPT-3. Rather than training models for specific classification tasks, GPT proved that scaling autoregressive next-token prediction produces emergent, general-purpose language capabilities.

- **Generative Pre-training**: Unified diverse language tasks into autoregressive sequence completion.
- **Causal Self-Attention**: Prevents future token leakage via lower-triangular causal masks.
- **End-to-End Autograd**: Hundreds of thousands of parameters receive analytical gradients through stacked Transformer blocks.
- **Emergent Cadences**: Synthesizes structured theatrical dialogue from simple character-level statistics.

Now, bring together all 20 modules of TinyTorch to train `TinyGPT` from scratch and generate Shakespearean text!

## What You're Building

**Primary milestone (what `tito milestone run 07` executes):**
1. **Tokenization**: Encode raw Shakespeare text with YOUR `CharTokenizer`.
2. **Dataset & Batching**: Slice text streams into autoregressive (input, target) shifted pairs with YOUR `TextWindowDataset` and `DataLoader`.
3. **Architecture**: Build a Pre-LN Causal Transformer Decoder with Token + Learned Positional Embeddings, Causal `MultiHeadAttention`, and GELU MLP blocks.
4. **Optimization**: Train with `CrossEntropyLoss` and `AdamW` using reverse-mode autograd.
5. **Autoregressive Generation**: Seed prompts (`"First Citizen:"`, `"ROMEO:"`, `"KING:"`) and extend them token-by-token with temperature sampling.

## Required Modules

**Run after Module 13** (Complete language and autograd stack)

<table width="100%">
  <thead>
    <tr>
      <th width="25%"><b>Module</b></th>
      <th width="25%">Component</th>
      <th width="50%">What It Provides</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>Module 01</b></td><td>Tensor</td><td>Strided multi-dimensional arrays with autograd tape</td></tr>
    <tr><td><b>Module 02</b></td><td>Activations</td><td>YOUR GELU activation function</td></tr>
    <tr><td><b>Module 03</b></td><td>Layers</td><td>Linear projection layers for Q, K, V, MLP, and LM Head</td></tr>
    <tr><td><b>Module 04</b></td><td>Losses</td><td>YOUR CrossEntropyLoss with numerical stability</td></tr>
    <tr><td><b>Module 05</b></td><td>DataLoader</td><td>YOUR Dataset and mini-batch DataLoader</td></tr>
    <tr><td><b>Module 06</b></td><td>Autograd</td><td>Reverse-mode automatic differentiation engine</td></tr>
    <tr><td><b>Module 07</b></td><td>Optimizers</td><td>YOUR AdamW optimizer with decoupled weight decay</td></tr>
    <tr><td><b>Module 10</b></td><td>Tokenization</td><td>YOUR CharTokenizer</td></tr>
    <tr><td><b>Module 11</b></td><td>Embeddings</td><td>Token and learned positional embeddings</td></tr>
    <tr><td><b>Module 12</b></td><td>Attention</td><td>Causal multi-head self-attention</td></tr>
    <tr><td><b>Module 13</b></td><td>Transformers</td><td>Pre-LN TransformerBlock and TinyGPT architecture</td></tr>
  </tbody>
</table>

## Milestone Script

### `01_tinygpt_shakespeare.py`

**Run via TITO (recommended):**

```bash
tito milestone run 07
# Or using aliases:
tito milestone run tinygpt
```

**Quick run on bundled offline sample:**

```bash
python3 milestones/07_2020_tinygpt/01_tinygpt_shakespeare.py --quick
```

**Custom prompt & temperature:**

```bash
python3 milestones/07_2020_tinygpt/01_tinygpt_shakespeare.py --prompt "HAMLET:" --temp 0.7
```

## Success Criteria

- **Loss Convergence**: Training loss drops steadily from initial (~3.0) to < 2.50.
- **Autoregressive Generation**: Model produces coherent text extending seed prompts.
- **Complete Stack Verification**: All transformer parameters receive finite analytical gradients.

## Further Reading

- **GPT-3 Paper**: Brown et al. (2020). ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165)
- **GPT-2 Paper**: Radford et al. (2019). ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Attention Is All You Need**: Vaswani et al. (2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
