"""
MLPerf EDU: Nano-RAG Agent Benchmark

A pedagogical Retrieval-Augmented Generation pipeline that exposes the systems
bottlenecks of vector retrieval vs. autoregressive generation.

Architecture:
    Query → Embed → Vector Search (brute-force MIPS) → Retrieve Top-K passages
    → Concatenate context → Transformer generates response tokens

Systems Focus:
    - Retrieval is memory-bandwidth bound (large matrix dot product)
    - Generation is compute bound (autoregressive transformer)
    - Students measure where the wall-clock time actually goes

Quality Target:
    - Training: Cross-entropy loss on next-token prediction with retrieved context
    - Inference: Queries per second with latency breakdown (retrieve vs. generate)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BruteForceRetriever(nn.Module):
    """
    Zero-dependency vector search engine.

    Stores a bank of passage embeddings and retrieves the Top-K most similar
    passages via exhaustive dot product (Maximum Inner Product Search).

    In production systems (FAISS, ScaNN), this is replaced by approximate
    nearest neighbor (ANN) algorithms. Here we keep it exact so students
    can measure the O(N*d) retrieval cost and optimize it.
    """

    def __init__(self, n_passages: int = 1000, embed_dim: int = 128):
        super().__init__()
        self.n_passages = n_passages
        self.embed_dim = embed_dim

        # Passage bank: each row is a normalized embedding vector
        # In a real system this would be populated from a corpus
        bank = torch.randn(n_passages, embed_dim)
        bank = F.normalize(bank, p=2, dim=1)
        self.register_buffer("passage_bank", bank)

        # Passage "content" tokens — simulates retrieved text
        # Each passage is a short sequence of token IDs
        self.register_buffer(
            "passage_tokens",
            torch.randint(0, 50257, (n_passages, 16))  # 16 tokens per passage
        )

    def search(self, query_embeds: torch.Tensor, top_k: int = 3):
        """
        Retrieve top-K passages by cosine similarity.

        Args:
            query_embeds: (batch, embed_dim) normalized query vectors
            top_k: number of passages to retrieve

        Returns:
            retrieved_tokens: (batch, top_k * passage_len) concatenated passage tokens
            scores: (batch, top_k) similarity scores
        """
        query_embeds = F.normalize(query_embeds, p=2, dim=1)

        # Brute-force MIPS — this is the I/O bottleneck
        similarity = torch.matmul(query_embeds, self.passage_bank.T)
        scores, indices = torch.topk(similarity, k=top_k, dim=1)

        # Gather passage tokens for retrieved indices
        batch_size = indices.size(0)
        retrieved = self.passage_tokens[indices.view(-1)]  # (batch*top_k, passage_len)
        retrieved = retrieved.view(batch_size, -1)          # (batch, top_k*passage_len)

        return retrieved, scores


class NanoRAGAgent(nn.Module):
    """
    End-to-end RAG pipeline as an nn.Module.

    Components:
        1. Query encoder: projects input tokens to retrieval embedding space
        2. Retriever: brute-force vector search over passage bank
        3. Generator: small transformer that conditions on retrieved context

    The forward pass returns (logits, loss) for training, or just logits
    for inference. The architecture is deliberately simple so every
    component's latency is measurable.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        n_passages: int = 1000,
        top_k: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.top_k = top_k
        self.max_seq_len = max_seq_len

        # --- Query Encoder ---
        self.query_embed = nn.Embedding(vocab_size, d_model)
        self.query_proj = nn.Linear(d_model, d_model)

        # --- Retriever ---
        self.retriever = BruteForceRetriever(
            n_passages=n_passages, embed_dim=d_model
        )

        # --- Generator (small autoregressive transformer) ---
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                ln_1=nn.LayerNorm(d_model),
                attn=nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                ln_2=nn.LayerNorm(d_model),
                ffn=nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
            ))
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def _encode_query(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode input tokens into a retrieval query vector."""
        embeds = self.query_embed(input_ids)          # (B, T, d)
        pooled = embeds.mean(dim=1)                    # (B, d) — mean pool
        return self.query_proj(pooled)                 # (B, d)

    def _generate(self, context_ids: torch.Tensor, targets=None):
        """
        Run the autoregressive generator on context + retrieved tokens.

        Args:
            context_ids: (B, T) token IDs (query + retrieved passages concatenated)
            targets: (B, T) target token IDs for training loss

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided, else None
        """
        B, T = context_ids.size()
        T = min(T, self.max_seq_len)
        context_ids = context_ids[:, :T]

        pos = torch.arange(0, T, device=context_ids.device)
        x = self.token_embed(context_ids) + self.pos_embed(pos)

        # Causal mask for autoregressive generation
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        for block in self.layers:
            attn_out, _ = block["attn"](
                block["ln_1"](x), block["ln_1"](x), block["ln_1"](x),
                attn_mask=causal_mask, need_weights=False
            )
            x = x + attn_out
            x = x + block["ffn"](block["ln_2"](x))

        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            targets = targets[:, :T]
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )

        return logits, loss

    def forward(self, input_ids: torch.Tensor, targets=None):
        """
        Full RAG pipeline: encode → retrieve → generate.

        For training, targets should match input_ids in shape (B, T).
        The loss is computed on next-token prediction over the augmented context
        using teacher-forced targets derived from the context itself.
        """
        # Stage 1: Encode the query
        query_vec = self._encode_query(input_ids)

        # Stage 2: Retrieve relevant passages
        retrieved_tokens, _scores = self.retriever.search(query_vec, top_k=self.top_k)

        # Stage 3: Concatenate query + retrieved context
        context = torch.cat([input_ids, retrieved_tokens.to(input_ids.device)], dim=1)

        # Stage 4: Generate conditioned on augmented context
        # For training: use the context itself as teacher-forced targets
        # (shifted by 1 for next-token prediction)
        if targets is not None:
            # Create targets from the augmented context: predict next token
            context_targets = context[:, 1:]   # shift left
            context_input = context[:, :-1]    # drop last
            logits, loss = self._generate(context_input, targets=context_targets)
        else:
            logits, loss = self._generate(context)

        return logits, loss

    def forward_with_timing(self, input_ids: torch.Tensor):
        """
        Inference-mode forward pass that returns per-stage latency breakdown.
        This is the key systems metric for the RAG agent benchmark.
        """
        self.eval()
        timings = {}

        with torch.no_grad():
            # Stage 1: Encode
            t0 = time.perf_counter()
            query_vec = self._encode_query(input_ids)
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000

            # Stage 2: Retrieve
            t0 = time.perf_counter()
            retrieved_tokens, scores = self.retriever.search(query_vec, top_k=self.top_k)
            timings["retrieve_ms"] = (time.perf_counter() - t0) * 1000

            # Stage 3: Generate
            context = torch.cat([input_ids, retrieved_tokens.to(input_ids.device)], dim=1)
            t0 = time.perf_counter()
            logits, _ = self._generate(context)
            timings["generate_ms"] = (time.perf_counter() - t0) * 1000

            timings["total_ms"] = (
                timings["encode_ms"] + timings["retrieve_ms"] + timings["generate_ms"]
            )

        return logits, timings


if __name__ == "__main__":
    print("🚀 Nano-RAG Agent Benchmark — Architecture Demo")

    model = NanoRAGAgent(
        vocab_size=50257, d_model=128, n_heads=4, n_layers=4,
        n_passages=1000, top_k=3
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: ~{total_params/1e6:.1f}M")

    # Training mode demo
    dummy_input = torch.randint(0, 50257, (4, 32))
    dummy_target = torch.randint(0, 50257, (4, 128))
    logits, loss = model(dummy_input, targets=dummy_target)
    print(f"✅ Training forward pass: logits={logits.shape}, loss={loss.item():.4f}")

    # Inference timing demo
    logits, timings = model.forward_with_timing(dummy_input)
    print(f"✅ Inference timing breakdown:")
    for k, v in timings.items():
        print(f"   {k}: {v:.2f} ms")
