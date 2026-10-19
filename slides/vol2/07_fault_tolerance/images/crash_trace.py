"""Rank 1 dies on the first attempt; torchrun restarts the whole worker group.
Run: torchrun --nproc-per-node=4 --max-restarts=2 crash_trace.py"""
import os, sys, time
import torch.distributed as dist
attempt = int(os.environ["TORCHELASTIC_RESTART_COUNT"])
dist.init_process_group("gloo")
r = dist.get_rank()
if attempt == 0 and r == 1:
    time.sleep(1.0)
    os._exit(1)            # a hard crash: no exception, no cleanup
dist.barrier()
if r == 0:
    print(f"attempt {attempt}: all {dist.get_world_size()} ranks passed the barrier", flush=True)
dist.destroy_process_group()
