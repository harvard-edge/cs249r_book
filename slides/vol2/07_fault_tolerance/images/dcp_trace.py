"""Four CPU ranks save one sharded checkpoint with PyTorch Distributed
Checkpoint (DCP); rank 0 then lists the files and reads the manifest.
Run: python3 dcp_trace2.py OUTDIR   (tested with torch 2.8, gloo backend)"""
import os, sys
import torch, torch.distributed as dist, torch.multiprocessing as mp
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import distribute_tensor, init_device_mesh, Shard

def run(rank, world, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533", GLOO_SOCKET_IFNAME="lo0")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    torch.manual_seed(0)
    full = {k: torch.randn(4096, 1024) for k in ("w", "m", "v")}
    state = {"model": {"w": distribute_tensor(full["w"], mesh, [Shard(0)])},
             "optim": {"m": distribute_tensor(full["m"], mesh, [Shard(0)]),
                       "v": distribute_tensor(full["v"], mesh, [Shard(0)])}}
    path = os.path.join(out, "step_1000")
    dcp.save(state, checkpoint_id=path)
    dist.barrier()
    if rank == 0:
        for name in sorted(os.listdir(path)):
            print(f"{os.path.getsize(os.path.join(path, name)):>12,d}  {name}")
        md = dcp.FileSystemReader(path).read_metadata()
        for fqn, item in md.state_dict_metadata.items():
            print(fqn, type(item).__name__, tuple(item.size), len(item.chunks), "chunks")
        print("storage entries:", len(md.storage_data))
        for k, v in list(md.storage_data.items())[:4]:
            print(" ", k.fqn, tuple(k.offset), "->", v.relative_path, v.offset, v.length)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(run, args=(4, sys.argv[1]), nprocs=4)
