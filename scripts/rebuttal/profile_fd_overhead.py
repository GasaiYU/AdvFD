"""Short cost measurements around the real trainer's complete iterations.

No training implementation is copied: successive FD-step entry points delimit
iterations, including generator backward/optimizer/EMA and critic work. Model
loading, queue fill and warmup are excluded. These runs measure cost, not quality.
"""

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time


class IterationMeter:
    def __init__(self, warmup, steps):
        self.warmup = warmup
        self.steps = steps
        self.completed = 0
        self.times = []

    def record(self, seconds):
        self.completed += 1
        if self.completed > self.warmup:
            self.times.append(seconds)
        return len(self.times) == self.steps


def summarize(directory):
    directory = Path(directory)
    cases = ["sim", "sim_resnet50", "sim_advfd"]
    if (directory / "sim_resnet50_swin_t.json").exists():
        cases.insert(2, "sim_resnet50_swin_t")
    results = [json.loads((directory / f"{case}.json").read_text()) for case in cases]
    for result in results[1:]:
        for key in ("gpu", "world_size", "global_batch_size", "warmup_steps", "measured_steps", "initial_step", "dtype"):
            if result[key] != results[0][key]:
                raise ValueError(f"Cannot compare different {key}: rerun all measurements")
    base = results[0]
    rows = []
    for case, result in zip(cases, results):
        rows.append({
            "case": case,
            "mean_ms": result["mean_ms"],
            "extra_ms_vs_sim": result["mean_ms"] - base["mean_ms"],
            "time_increase_percent": 100 * (result["mean_ms"] / base["mean_ms"] - 1),
            "peak_allocated_gib": result["peak_allocated_gib"],
            "extra_allocated_gib_vs_sim": result["peak_allocated_gib"] - base["peak_allocated_gib"],
        })
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("case                mean ms    extra ms    peak GiB    extra GiB")
    for row in rows:
        print(f"{row['case']:20} {row['mean_ms']:9.2f} {row['extra_ms_vs_sim']:11.2f} "
              f"{row['peak_allocated_gib']:11.2f} {row['extra_allocated_gib_vs_sim']:12.2f}")
    adv_extra_ms = next(row["extra_ms_vs_sim"] for row in rows if row["case"] == "sim_advfd")
    if adv_extra_ms > 0:
        for row in rows:
            if row["case"] in ("sim_resnet50", "sim_resnet50_swin_t"):
                print(f"{row['case']}/AdvFD incremental time ratio: "
                      f"{row['extra_ms_vs_sim'] / adv_extra_ms:.3f}")
    print(f"Saved {directory / 'summary.csv'}")


def profile():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import torch
    import main_fd as trainer

    parser = argparse.ArgumentParser(parents=[trainer.get_args_parser()])
    parser.add_argument("--profile_warmup_steps", type=int, default=50)
    parser.add_argument("--profile_steps", type=int, default=200)
    parser.add_argument("--profile_output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Cost profiling requires the training server's CUDA GPUs")
    if args.profile_warmup_steps < 0 or args.profile_steps < 1:
        parser.error("warmup must be >= 0 and measured steps must be >= 1")
    if args.profile_warmup_steps + args.profile_steps >= 1000:
        parser.error("Use fewer than 1000 total iterations to exclude the trainer's dynamic checkpoint schedule")
    if args.current_step + args.profile_warmup_steps + args.profile_steps >= args.epochs * args.steps_per_epoch:
        parser.error("The training schedule must extend beyond the measurement window")
    if args.fd_adv_weight > 0 and args.current_step < args.fd_adv_start_step + args.fd_adv_warmup_steps:
        parser.error("Start after the AdvFD ramp to measure the active branch")

    # Keep the 125k LR schedule; start from identical base weights at a virtual
    # step where the AdvFD ramp is complete. Do not interpret as trained models.
    args.auto_resume = False
    args.resume_from = None
    args.disable_vis = True
    args.vis_freq = 0
    args.online_eval = False
    args.eval_freq = 0
    args.save_freq = 10**9
    args.milestone_interval = 0
    args.enable_wandb = False
    args.print_freq = 20
    initial_step = args.current_step
    meter = IterationMeter(args.profile_warmup_steps, args.profile_steps)
    original_factory = trainer.get_fd_train_step

    class MeasurementComplete(Exception):
        pass

    def finish():
        durations = torch.tensor(meter.times, device="cuda", dtype=torch.float64)
        memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                              device="cuda", dtype=torch.float64)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(durations, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(memory, op=torch.distributed.ReduceOp.MAX)
        if args.rank == 0:
            times = durations.cpu().tolist()
            result = {
                "gpu": torch.cuda.get_device_name(), "world_size": args.world_size,
                "global_batch_size": args.batch_size * args.world_size, "dtype": args.dtype,
                "initial_step": initial_step, "warmup_steps": meter.warmup,
                "measured_steps": len(times), "mean_ms": statistics.mean(times) * 1000,
                "median_ms": statistics.median(times) * 1000,
                "peak_allocated_gib": memory[0].item() / 1024**3,
                "peak_reserved_gib": memory[1].item() / 1024**3,
                "repr_models": args.fd_repr_models, "repr_weights": args.fd_repr_weights,
                "fd_adv_weight": args.fd_adv_weight,
                "fd_adv_update_freq": args.fd_adv_update_freq,
                "timing_scope": "full training iteration, max across ranks; no initialization/eval/checkpoint",
                "seconds_per_iteration": times,
            }
            output = Path(args.profile_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Profile saved: {output}; mean={result['mean_ms']:.2f} ms; "
                  f"peak allocated={result['peak_allocated_gib']:.2f} GiB")

    def instrumented_factory(*factory_args, **factory_kwargs):
        step_fn = original_factory(*factory_args, **factory_kwargs)
        if args.rank == 0:
            print(f"[Profile] {meter.warmup} warmup + {meter.steps} measured iterations; "
                  f"stop at virtual step {initial_step + meter.warmup + meter.steps}. "
                  f"Progress every {args.print_freq} steps.", flush=True)
        last_start = None

        def measured_step():
            nonlocal last_start
            torch.cuda.synchronize()
            if last_start is not None and meter.record(time.perf_counter() - last_start):
                finish()
                raise MeasurementComplete
            if meter.completed == meter.warmup:
                torch.cuda.reset_peak_memory_stats()
            last_start = time.perf_counter()
            return step_fn()

        return measured_step

    trainer.get_fd_train_step = instrumented_factory
    try:
        trainer.train_and_evaluate(args)
        raise RuntimeError("Trainer ended before all timing samples were collected")
    except MeasurementComplete:
        pass
    finally:
        trainer.get_fd_train_step = original_factory
        trainer._cleanup_distributed()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2])
    else:
        profile()
