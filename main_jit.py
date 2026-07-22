import argparse
import datetime
import logging
import os
import sys
import time

import torch
import torch.distributed

from utils.builders import create_generation_model, create_tokenizer
from utils.checkpoint_util import AsyncCheckpointSaver, ckpt_resume, save_checkpoint
from utils.distributed_util import all_reduce_mean, register_preempt_handler, preempt_requested
from utils.eval_util import evaluate_all_emas
from utils.grad_util import get_grad_norm
from utils.logging_util import MetricLogger, SmoothedValue
from utils.optimizer_util import create_optimizer
from utils.rng_util import RNGStateManager
from utils.schedule_util import adjust_learning_rate
from utils.setup_util import setup
from utils.vis_util import visualize
from frechet_distance.evaluator import FDEvaluator
from frechet_distance.losses import (
    compute_frechet_distance_loss,
    diff_all_gather,
    load_mu_and_sigma_reference,
    precompute_sigma_ref_sqrt,
)
from frechet_distance.repr_models import load_repr_model


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logger = logging.getLogger("FD_loss")


def build_train_loader(args):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, DistributedSampler
    from utils.data_util import center_crop_arr

    train_dir = os.path.join(args.data_path, "train")
    if not os.path.isdir(train_dir):
        train_dir = args.data_path
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=args.world_size, rank=args.rank,
        shuffle=True, drop_last=True,
    ) if args.world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(f"[JiT] train images from {train_dir}: {len(dataset)}")
    return loader, sampler


def all_reduce_grads(module):
    if not torch.distributed.is_initialized():
        return
    for p in module.parameters():
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)


def get_fd_regularizer(args):
    if args.fd_reg_weight <= 0:
        return None

    repr_model, feat_dim, _, _ = load_repr_model(
        args.fd_reg_repr_model,
        target_size=args.fd_reg_target_size,
        grad_checkpointing=args.fd_repr_grad_checkpointing,
    )
    mu_ref, sigma_ref = load_mu_and_sigma_reference(
        args.fd_reg_stats_path,
        pool_type=args.fd_reg_pool_type,
    )
    sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref) if args.fd_eigvalsh else None
    logger.info(
        f"[JiT FD-reg] model={args.fd_reg_repr_model}, feat_dim={feat_dim}, "
        f"pool={args.fd_reg_pool_type}, weight={args.fd_reg_weight}, "
        f"stats={args.fd_reg_stats_path}"
    )
    return {
        "model": repr_model,
        "mu_ref": mu_ref,
        "sigma_ref": sigma_ref,
        "sigma_ref_sqrt": sigma_ref_sqrt,
        "pool_type": args.fd_reg_pool_type,
    }


def compute_fd_regularizer(fd_reg, images_01, args):
    primary, secondary = fd_reg["model"](images_01)
    feats = secondary if args.fd_reg_pool_type == "avg" else primary
    feats = diff_all_gather(feats)
    fid = compute_frechet_distance_loss(
        fd_reg["mu_ref"],
        fd_reg["sigma_ref"],
        all_feats=feats,
        sigma_ref_sqrt=fd_reg["sigma_ref_sqrt"],
    )
    return fid, fid / (fid.detach() + args.fd_fid_norm_eps)


def train_and_evaluate(args):
    wandb_logger = setup(args)
    register_preempt_handler()

    tokenizer = create_tokenizer(args)
    if tokenizer is not None:
        raise NotImplementedError("main_jit.py is intended for pixel-space JiT training")

    model, ema_model = create_generation_model(args)
    optimizer = create_optimizer(args, model, print_trainable_params=True)
    ckpt_resume(args, model, optimizer, ema_model)

    train_loader, sampler = build_train_loader(args)
    fd_reg = get_fd_regularizer(args)

    rng = RNGStateManager()
    rng.save()
    if (not args.disable_vis) or args.vis_only:
        visualize(args, model, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
        if args.vis_only:
            return 0

    repr_model_eval, feat_dim_eval, _, _ = load_repr_model("inception")
    fid_evaluator = FDEvaluator(repr_model_eval, feat_dim_eval, args.fid_stats_path)

    global_bsz = args.batch_size * args.world_size
    ckpt_saver = AsyncCheckpointSaver()
    session_start = time.time()
    step_start = time.perf_counter()

    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    for name, window, fmt in [
        ("lr", 1, "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s", args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)", args.print_freq, "{value:.2f}"),
        ("device_mem(GB)", args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))

    logger.info(f"training from step {args.current_step:,} -> {args.total_steps:,}")
    data_iter = iter(train_loader)
    last_ckpt_step = args.current_step

    for step, _ in metric_logger.log_every(
        iter(int, 1),
        args.print_freq,
        header="JiT Train:",
        start_iteration=args.current_step,
        n_iterations=args.total_steps,
    ):
        if step >= args.total_steps:
            break
        model.train()
        if sampler is not None and step % len(train_loader) == 0:
            sampler.set_epoch(step // len(train_loader))

        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)

        images = images.cuda(non_blocking=True).mul(2.0).sub(1.0)
        labels = labels.cuda(non_blocking=True)

        adjust_learning_rate(optimizer, step, args)
        optimizer.zero_grad(set_to_none=True)

        jit_loss, loss_dict, x_pred, _, _ = model(
            images,
            labels,
            return_x_pred=True,
            return_t=True,
        )
        loss = jit_loss
        log_dict = {"jit_loss": float(jit_loss.detach())}

        if fd_reg is not None:
            x_pred_01 = x_pred.mul(0.5).add(0.5).clamp(0.0, 1.0)
            fid, fd_loss = compute_fd_regularizer(fd_reg, x_pred_01, args)
            loss = loss + args.fd_reg_weight * fd_loss
            log_dict["fd_reg"] = float(fid.detach())
            log_dict["fd_reg_loss"] = float(fd_loss.detach())

        loss.backward(create_graph=False)
        if torch.distributed.is_initialized():
            all_reduce_grads(model)

        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0.0 else get_grad_norm(model.parameters()))

        if torch.isfinite(grad_norm):
            optimizer.step()
            ema_model.step(model)
        else:
            logger.warning(f"[step {step}] NaN/Inf grad_norm; skipping optimizer and EMA")
        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += global_bsz

        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()
        loss_value = all_reduce_mean(loss.item())
        log_dict = {k: all_reduce_mean(v) for k, v in log_dict.items()}
        sps = args.batch_size / step_time if step_time > 0 else 0.0
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        metric_logger.update(
            loss=loss_value,
            grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **{
                "samples/s/device": sps,
                "samples/s": sps * args.world_size,
                "samples_seen(M)": args.samples_seen / 1e6,
                "device_mem(GB)": mem_gb,
            },
            **log_dict,
        )

        if step % args.print_freq == 0 and wandb_logger:
            wandb_logger.update({
                "train/loss": loss_value,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm,
                **{f"train/{k}": v for k, v in log_dict.items()},
            }, step=args.current_step)

        def _save(saver=ckpt_saver):
            elapsed = time.time() - session_start + args.last_elapsed_time
            save_checkpoint(args, step, model, optimizer, ema_model, elapsed, saver=saver)
            torch.distributed.barrier()

        if (args.current_step - last_ckpt_step >= args.save_every
                or args.current_step == args.total_steps):
            _save()
            last_ckpt_step = args.current_step

        if args.milestone_every > 0 and step > 0 and step % args.milestone_every == 0:
            _save()

        if preempt_requested():
            ckpt_saver.wait()
            _save(saver=None)
            return 0

        if args.vis_every > 0 and args.current_step % args.vis_every == 0:
            visualize(args, model, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
            model.train()

        if args.eval_every > 0 and args.online_eval and args.current_step % args.eval_every == 0:
            torch.cuda.empty_cache()
            evaluate_all_emas(
                args, model, ema_model, fid_evaluator, tokenizer,
                step=args.current_step, wandb_logger=wandb_logger,
                cfg=args.cfg, num_images=args.num_images_for_eval_and_search,
            )
            model.train()

    ckpt_saver.wait()
    total = time.time() - session_start + args.last_elapsed_time
    metric_logger.synchronize_between_processes()
    logger.info(f"averaged stats: {metric_logger}")
    logger.info(f"Training complete. Total time: {datetime.timedelta(seconds=int(total))}")
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("JiT training with FD regularization", add_help=False)

    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--steps_per_epoch", default=1250, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--same_noise", action="store_true")

    parser.add_argument("--model", default="JiT_B", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--rope_2d", action="store_true")
    parser.add_argument("--learned_pe", action="store_true")
    parser.add_argument("--legacy_time_convention", action="store_true")
    parser.add_argument("--t_eps", type=float, default=5e-2)

    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--token_channels", default=3, type=int)
    parser.add_argument("--tokenizer_patch_size", default=1, type=int)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use_muon", action="store_true")
    parser.add_argument("--muon_lr", type=float, default=1e-3)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_type", default="edm", type=str, choices=["const", "edm"])
    parser.add_argument("--ema_rates", default=[0.9999, 0.9996], type=float, nargs="+")
    parser.add_argument("--ema_halflife_kimg", default=[250, 500, 1000, 2000], type=float, nargs="+")
    parser.add_argument("--eval_ema_labels", default=None, type=str, nargs="+")
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--fd_repr_grad_checkpointing", action="store_true")

    parser.add_argument("--P_mean", type=float, default=0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--sampling_method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", default=3.0, type=float)
    parser.add_argument("--cfg_list", type=float, nargs="+",
                        default=[2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--vis_steps", default=[1], type=int, nargs="+")

    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279],
                        type=int, nargs="+")
    parser.add_argument("--force_class_of_interest", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--keep_n_ckpts", default=3, type=int)
    parser.add_argument("--milestone_interval", default=20, type=int)

    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--num_images_for_eval_and_search", default=50000, type=int)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--eval_bsz", type=int, default=256)
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/guided_diffusion_stats.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--save_eval_images", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")

    parser.add_argument("--fd_reg_weight", type=float, default=0.01)
    parser.add_argument("--fd_fid_norm_eps", type=float, default=0.01)
    parser.add_argument("--fd_reg_repr_model", type=str, default="inception")
    parser.add_argument("--fd_reg_stats_path", type=str, default="data/fid_stats/guided_diffusion_stats.npz")
    parser.add_argument("--fd_reg_pool_type", type=str, default="cls", choices=["cls", "avg"])
    parser.add_argument("--fd_reg_target_size", type=int, default=256)
    parser.add_argument("--fd_eigvalsh", action="store_true")

    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--local_eval_dir", type=str, default=None)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--vis_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--vis_only", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)
    parser.add_argument("--current_step", type=int, default=0)
    parser.add_argument("--samples_seen", type=int, default=0)
    parser.add_argument("--project", default="JiT", type=str)
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_false", dest="enable_wandb")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dtype", default="bf16", type=str, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    sys.exit(train_and_evaluate(args))
