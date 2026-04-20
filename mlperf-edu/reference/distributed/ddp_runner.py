"""
Iter-10 (Dean): two-process DDP via torch.multiprocessing + Gloo on localhost.

Picks micro-DLRM as the workload:
  - 1M params at fp32 = 4 MB of gradients per AllReduce.
  - Loopback Gloo handles this in ~0.5-1 ms.
  - Iter-5.6 found micro-DLRM at small batch is compute-bound on the MLP,
    so DDP overhead becomes the natural rate-limiter.

Smoke gate (Q4 in Dean's iter-10 spec):
  | loss_ddp(step=50) - loss_gradacc(step=50) | / loss_gradacc(step=50) < 0.02
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from reference.cloud.micro_dlrm import MicroDLRMWhiteBox


def _build_inputs(batch: int, seed: int = 42) -> tuple:
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(batch, 16, generator=g)
    sparse_indices = [
        torch.randint(0, 943, (batch,), generator=g),
        torch.randint(0, 1682, (batch,), generator=g),
        torch.randint(0, 21, (batch,), generator=g),
    ]
    sparse_offsets = [torch.arange(batch) for _ in range(3)]
    targets = torch.randint(0, 2, (batch,), generator=g).float().unsqueeze(1)
    return dense, sparse_indices, sparse_offsets, targets


def _ddp_worker(rank: int, world_size: int,
                 n_steps: int, micro_batch: int,
                 result_queue: mp.Queue,
                 init_method: str = "tcp://127.0.0.1:29500"):
    """One DDP rank: init Gloo, build model, run n_steps, push final loss."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="gloo", init_method=init_method,
                              rank=rank, world_size=world_size)
    torch.manual_seed(42)

    model = MicroDLRMWhiteBox()
    ddp_model = nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    # Each rank gets a different shard of synthetic data, generated
    # deterministically from the rank's seed offset so total batch =
    # world_size * micro_batch matches the gradient-accumulation baseline.
    last_loss = 0.0
    allreduce_time_total = 0.0
    for step in range(n_steps):
        # Per-rank data shard.
        dense, sparse_indices, sparse_offsets, targets = _build_inputs(
            micro_batch, seed=42 + rank * 1000 + step
        )

        optimizer.zero_grad(set_to_none=True)
        # Use logits (not BCE-with-sigmoid output) for stable training.
        logits = model.bot_l[0](dense)  # placeholder; we just need loss to descend
        # Instead, use the full model output — sigmoid output, so BCELoss.
        out = ddp_model(dense, sparse_indices, sparse_offsets)
        loss = nn.functional.binary_cross_entropy(out, targets)

        t_back = time.perf_counter()
        loss.backward()
        # AllReduce happens implicitly during backward via DDP's reducer.
        allreduce_time_total += time.perf_counter() - t_back

        optimizer.step()
        last_loss = loss.item()

    if rank == 0:
        result_queue.put({
            "rank": rank,
            "final_loss": last_loss,
            "allreduce_time_per_step_ms": allreduce_time_total / n_steps * 1000,
        })

    dist.destroy_process_group()


def run_gradacc_baseline(n_steps: int, micro_batch: int, world_size: int) -> dict:
    """Single-process gradient-accumulation baseline equivalent to DDP."""
    torch.manual_seed(42)
    model = MicroDLRMWhiteBox()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    last_loss = 0.0
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for rank in range(world_size):
            dense, sparse_indices, sparse_offsets, targets = _build_inputs(
                micro_batch, seed=42 + rank * 1000 + step
            )
            out = model(dense, sparse_indices, sparse_offsets)
            loss = nn.functional.binary_cross_entropy(out, targets) / world_size
            loss.backward()
            accum_loss += loss.item()
        optimizer.step()
        last_loss = accum_loss
    return {"final_loss": last_loss}


def run_ddp(n_steps: int = 50, micro_batch: int = 64, world_size: int = 2) -> dict:
    """Spawn world_size DDP processes; return rank-0 metrics."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=_ddp_worker,
                          args=(rank, world_size, n_steps, micro_batch, result_queue))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            return {"error": "DDP worker timed out"}
    if result_queue.empty():
        return {"error": "no result from rank 0"}
    return result_queue.get()
