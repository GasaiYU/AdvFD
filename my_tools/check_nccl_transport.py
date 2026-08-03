"""Measure the achieved all-reduce bandwidth at the gradient sizes this repo uses.

Answers one question: is the interconnect fast enough that gradient sync is
negligible, or is it the bottleneck? Reports measured GB/s and the implied
seconds per training step for pMF_B/L/H, so the result maps directly onto the
observed step time instead of a raw benchmark number.

Run on every node with the same rendezvous as training:

    GPUS_PER_NODE=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
        bash my_tools/check_nccl_transport.sh

Set NCCL_DEBUG=INFO to also see which transport NCCL picked (look for NET/IB
versus NET/Socket in the output).
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist

# fp32 gradient bytes per model, measured by instantiating each backbone and
# summing trainable parameters (pMF at 256px/disable_v_head, JiT at 256px).
MODEL_GRAD_MB = {
    "pMF_B": 473.7, "pMF_L": 1642.4, "pMF_H": 3822.5,
    "JiT_B": 525.3, "JiT_L": 1836.6, "JiT_H": 3811.4,
}


def _fmt(seconds):
    return f"{seconds:6.3f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--output_json", type=str, default=None)
    ap.add_argument(
        "--sizes_mb", type=float, nargs="+",
        default=[1, 25, 100, 400, 1600, 3822],
        help="buffer sizes to sweep; 25 is the DDP bucket, 3822 is pMF_H total",
    )
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    if rank == 0:
        print(f"[setup] world_size={world}  device={torch.cuda.get_device_name(local_rank)}")
        print(f"[setup] NCCL_SOCKET_IFNAME={os.environ.get('NCCL_SOCKET_IFNAME', '(unset)')}  "
              f"NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA', '(unset)')}  "
              f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE', '(unset)')}")
        print()
        print(f"{'buffer':>10}  {'time':>8}  {'algbw':>10}  {'busbw':>10}")
        print("-" * 44)

    results = {}
    for size_mb in args.sizes_mb:
        numel = int(size_mb * 1024 * 1024 / 4)
        buf = torch.empty(numel, dtype=torch.float32, device="cuda").normal_()

        for _ in range(args.warmup):
            dist.all_reduce(buf, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        dist.barrier()

        start = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(buf, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / args.iters

        gb = size_mb / 1024
        algbw = gb / elapsed                                  # buffer / time
        busbw = algbw * 2 * (world - 1) / world               # bytes on the wire
        results[f"{size_mb}MB"] = {
            "seconds": elapsed, "algbw_GBps": algbw, "busbw_GBps": busbw,
        }
        if rank == 0:
            print(f"{size_mb:8.0f}MB  {elapsed*1000:7.2f}ms  "
                  f"{algbw:7.2f}GB/s  {busbw:7.2f}GB/s")

    if rank == 0:
        # Use the largest buffer as the steady-state bandwidth estimate.
        big = max(args.sizes_mb)
        busbw = results[f"{big}MB"]["busbw_GBps"]
        algbw = results[f"{big}MB"]["algbw_GBps"]
        print()
        print(f"[verdict] sustained busbw ~{busbw:.1f} GB/s")
        if busbw < 2:
            verdict = "TCP/Ethernet territory. Gradient sync will dominate for large models."
        elif busbw < 8:
            verdict = "low-end IB or fast Ethernet. Sync is significant for pMF_H."
        elif busbw < 20:
            verdict = "healthy IB (~100Gb class). Sync should be a small fraction."
        else:
            verdict = "fast IB (200Gb+). Gradient sync is not your bottleneck."
        print(f"[verdict] {verdict}")
        print()
        print("implied gradient all-reduce cost per training step (fp32, unbucketed total):")
        for name, mb in MODEL_GRAD_MB.items():
            secs = (mb / 1024) / algbw
            print(f"  {name:7s} {mb:7.0f}MB -> {_fmt(secs)}"
                  f"   ({secs/11.39*100:5.1f}% of an 11.39s step)")
        print()
        print("If this says Ethernet but the nodes have IB cards, NCCL picked the")
        print("wrong interface. Check `ibstat` / `ibv_devinfo` on the node, then set")
        print("NCCL_IB_HCA=<hca> or NCCL_SOCKET_IFNAME=<ib-iface> and re-run.")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(
                    {"world_size": world, "busbw_GBps": busbw, "sweep": results},
                    f, indent=2,
                )
            print(f"\n[saved] {args.output_json}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
