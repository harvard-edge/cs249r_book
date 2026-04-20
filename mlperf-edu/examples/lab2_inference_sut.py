#!/usr/bin/env python3
"""
MLPerf EDU: Lab 2 — Inference Latency Optimization
====================================================

Students implement a System Under Test (SUT) plugin for NanoGPT inference,
then optimize it (KV-cache, batching, mixed precision) while the LoadGen
measures latency, throughput, and power.

WORKFLOW:
    1. Read the reference SUT below
    2. Run: mlperf cloud --task nanogpt-12m --sut examples/lab2_inference_sut.py --scenario SingleStream
    3. Observe p90 latency in the submission JSON
    4. Add KV-cache (see TODO) and re-run — measure speedup
    5. Try Server scenario to see Poisson arrival impact

This file IS the student's SUT plugin. Modify it and re-run.
"""

import torch
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the base SUT interface that the harness expects
from src.mlperf.sut import SUT_Interface


class StudentNanoGPTSUT(SUT_Interface):
    """
    Student System Under Test (SUT) implementation.

    The harness will call process_queries() with a batch of QuerySample objects.
    Your job: process them as fast as possible while maintaining accuracy.
    """

    def __init__(self, config):
        """Initialize the SUT with a loaded model and tokenizer."""
        super().__init__(config)

        # Load the pre-trained NanoGPT model
        from reference.cloud.nanogpt_train import NanoGPTWhiteBox

        self.device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model = NanoGPTWhiteBox(
            vocab_size=65,
            n_embd=384,
            n_head=6,
            n_layer=6,
        ).to(self.device)
        self.model.eval()

        # TODO: Load pre-trained checkpoint for real quality
        # checkpoint = torch.load("checkpoints/nanogpt_baseline.pt")
        # self.model.load_state_dict(checkpoint)

        print(f"🔌 SUT loaded: NanoGPT on {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    async def process_queries(self, queries):
        """
        Process a batch of queries.

        Args:
            queries: List[QuerySample] with .id and .index fields

        Returns:
            List of result dicts (one per query) with latency/accuracy metrics
        """
        results = []
        with torch.no_grad():
            for q in queries:
                # Create a dummy input prompt (in production, decode from dataset)
                prompt = torch.randint(0, 65, (1, 64), device=self.device)

                # === STUDENT OPTIMIZATION AREA ===

                # BASELINE: Simple greedy generation (slow!)
                generated_tokens = 32
                output = prompt
                for _ in range(generated_tokens):
                    logits, _ = self.model(output)
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    output = torch.cat([output, next_token], dim=1)

                # TODO 1: Enable KV-cache for O(1) generation per token
                #   Instead of re-running the full sequence each step,
                #   cache the key/value tensors and only compute the new token.
                #   Expected speedup: ~10x for 32-token generation

                # TODO 2: Try torch.compile() for kernel fusion
                #   self.model = torch.compile(self.model, mode="reduce-overhead")

                # TODO 3: Try FP16 inference
                #   with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                #       logits = self.model(output)

                results.append({
                    "tokens_generated": generated_tokens,
                    "sequence_length": output.size(1),
                    "accuracy": 1.0,  # Placeholder
                })

        return results


if __name__ == "__main__":
    # Quick sanity check
    sut = StudentNanoGPTSUT(config={})
    from src.mlperf.loadgen import QuerySample
    import time

    samples = [QuerySample(id=i, index=i, arrival_time=time.perf_counter()) for i in range(4)]
    results = asyncio.run(sut.process_queries(samples))
    print(f"\n✅ Processed {len(results)} queries")
    for r in results:
        print(f"   tokens={r['tokens_generated']}, seq_len={r['sequence_length']}")
