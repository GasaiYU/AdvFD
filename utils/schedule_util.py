import math


def linear_warmup_scale(current_step, start_step, warmup_steps):
    """Return a step-based linear ramp in [0, 1]."""
    if warmup_steps <= 0:
        return 1.0 if current_step >= start_step else 0.0
    progress = (current_step - start_step) / warmup_steps
    return max(0.0, min(1.0, progress))


def adjust_learning_rate(optimizer, step, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < args.warmup_steps:
        lr = args.lr * step / args.warmup_steps
    else:
        if args.lr_sched == "constant":
            lr = args.lr
        elif args.lr_sched == "cosine":
            progress = (step - args.warmup_steps) / (args.total_steps - args.warmup_steps)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
