import logging

import torch

import models
from utils.distributed_util import broadcast_module_params, is_enabled
from utils.ema_util import EMAModel

logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# model / tokenizer creation
# ---------------------------------------------------------------------------

def create_generation_model(args):
    logger.info("Creating generation models.")

    if args.model == "JiT_B_sCM_residual_x":
        # The raw head is TrigFlow G, while the public endpoint is the clean
        # image x_t + sigma_data * sin(t) * G.  A dedicated wrapper prevents
        # G checkpoints from being mistaken for direct-D or F checkpoints.
        from meanflow.train_scm_jit_b_residual_x import (
            SCMJiTResidualXDenoiser,
        )

        model = SCMJiTResidualXDenoiser(
            img_size=args.img_size,
            model_size="base",
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            P_mean=args.P_mean,
            P_std=args.P_std,
            t_eps=args.t_eps,
            noise_scale=args.noise_scale,
            legacy_time_convention=False,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            grad_checkpointing=False,
            adaptive_double_norm=True,
            dropout_all_blocks=True,
            objective="scm",
            sigma_data=args.sigma_data,
            sigma_max=args.sigma_max,
            adaptive_weighting=False,
            g_adapt_steps=0,
        )
    elif args.model in {
        "JiT_B_sCM_xpred",
        "JiT_B_sCM_xpred_reversed_time",
    }:
        # The old key is retained for literal-time direct-D checkpoints.  New
        # training permanently maps physical TrigFlow time to the released
        # JiT coordinate 1-2t/pi and therefore has a distinct evaluator key.
        from meanflow.train_scm_jit_b import SCMJiTDenoiser

        network_time_mode = (
            "legacy_reversed"
            if args.model == "JiT_B_sCM_xpred_reversed_time"
            else "literal"
        )

        model = SCMJiTDenoiser(
            img_size=args.img_size,
            model_size="base",
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            P_mean=args.P_mean,
            P_std=args.P_std,
            t_eps=args.t_eps,
            noise_scale=args.noise_scale,
            legacy_time_convention=False,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            grad_checkpointing=False,
            adaptive_double_norm=True,
            dropout_all_blocks=True,
            objective="scm",
            sigma_data=args.sigma_data,
            sigma_max=args.sigma_max,
            adaptive_weighting=False,
            x_adapt_steps=0,
            network_time_mode=network_time_mode,
        )
    elif args.model in {"JiT_B_sCM", "JiT_L_sCM", "JiT_H_sCM"}:
        from meanflow.meanflow_jit import MeanFlowJiTDenoiser

        model_size = {
            "JiT_B_sCM": "base",
            "JiT_L_sCM": "large",
            "JiT_H_sCM": "huge",
        }[args.model]
        model = MeanFlowJiTDenoiser(
            img_size=args.img_size,
            model_size=model_size,
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            # Dropout modules have no effect in eval mode, but keeping the
            # configured probabilities faithfully reconstructs the training model.
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            P_mean=args.P_mean,
            P_std=args.P_std,
            t_eps=args.t_eps,
            noise_scale=args.noise_scale,
            legacy_time_convention=False,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            grad_checkpointing=False,
            adaptive_double_norm=True,
            dropout_all_blocks=True,
            objective="scm",
            sigma_data=args.sigma_data,
            sigma_max=args.sigma_max,
            adaptive_weighting=False,
        )
    elif args.model in models.JiTDenoiser_models:
        model = models.JiTDenoiser_models[args.model](
            img_size=args.img_size,
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            attn_dropout=args.attn_dropout,
            proj_dropout=args.proj_dropout,
            P_mean=args.P_mean,
            P_std=args.P_std,
            t_eps=args.t_eps,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            legacy_time_convention=args.legacy_time_convention,
            grad_checkpointing=args.grad_checkpointing,
        )
    elif args.model in models.iMFDenoiser_models:
        model = models.iMFDenoiser_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.token_channels,
            tokenizer_patch_size=args.tokenizer_patch_size,
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            # training parameters
            P_mean=args.P_mean,
            P_std=args.P_std,
            ratio_r_neq_t=args.ratio_r_neq_t,
            cfg_beta=args.cfg_beta,
            cfg_omega_max=args.cfg_omega_max,
            aux_head_depth=args.aux_head_depth,
            class_tokens=args.class_tokens,
            time_tokens=args.time_tokens,
            guidance_tokens=args.guidance_tokens,
            interval_tokens=args.interval_tokens,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            disable_v_head=args.disable_v_head,
            grad_checkpointing=args.grad_checkpointing,
        )
    elif args.model in models.pMFDenoiser_models:
        model = models.pMFDenoiser_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_channels=args.token_channels,
            tokenizer_patch_size=args.tokenizer_patch_size,
            num_classes=args.num_classes,
            label_drop_prob=args.label_drop_prob,
            # training parameters
            P_mean=args.P_mean,
            P_std=args.P_std,
            ratio_r_neq_t=args.ratio_r_neq_t,
            cfg_beta=args.cfg_beta,
            tr_uniform=args.tr_uniform,
            cfg_omega_max=args.cfg_omega_max,
            aux_head_depth=args.aux_head_depth,
            class_tokens=args.class_tokens,
            time_tokens=args.time_tokens,
            guidance_tokens=args.guidance_tokens,
            interval_tokens=args.interval_tokens,
            t_eps=args.t_eps,
            perceptual_threshold=args.perceptual_threshold,
            perceptual_loss_on_aux=args.perceptual_loss_on_aux,
            rope_2d=args.rope_2d,
            learned_pe=args.learned_pe,
            disable_v_head=args.disable_v_head,
            noise_scale=args.noise_scale,
            norm_eps=args.norm_eps,
            norm_p=args.norm_p,
            grad_checkpointing=args.grad_checkpointing,
        )
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.cuda()
    # Broadcast weights from rank 0 before EMA init.
    if is_enabled():
        logger.info("[Model] Broadcasting weights from rank 0 ...")
        broadcast_module_params(model, src=0)
        logger.info("[Model] Broadcast done.")
    logger.info(f"====Model====\n{model}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} parameters: {n / 1e6:.2f}M ({n:,})")

    if args.ema_type == "const":
        ema_values = args.ema_rates
    elif args.ema_type == "edm":
        ema_values = args.ema_halflife_kimg
    else:
        ema_values = args.ema_sigma_rel
    ema = EMAModel(model, ema_type=args.ema_type, values=ema_values, batch_size=args.global_bsz)
    logger.info(f"EMA: type={args.ema_type}, labels={ema.labels}")
    return model, ema


def create_tokenizer(args):
    """create, load weights, and optionally compile the tokenizer."""
    if args.tokenizer is None:
        logger.info("not using any tokenizer")
        return None
    logger.info(f"creating tokenizer: {args.tokenizer}")

    if args.tokenizer in models.VAE_models:
        tok = models.DiffusersAutoencoderKL(name=args.tokenizer)
    else:
        raise ValueError(f"unsupported tokenizer {args.tokenizer}")

    tok.cuda().eval().requires_grad_(False)
    if getattr(args, "vae_grad_checkpointing", False):
        if not hasattr(tok, "vae") or not hasattr(tok.vae, "enable_gradient_checkpointing"):
            raise RuntimeError(
                f"Tokenizer {args.tokenizer!r} does not support VAE gradient checkpointing"
            )
        tok.vae.enable_gradient_checkpointing()
        logger.info("[Tokenizer] VAE gradient checkpointing enabled")
    if getattr(args, "vae_decode_bsz", 0) > 0:
        logger.info(
            "[Tokenizer] Differentiable VAE decode chunk size: %d "
            "(per-chunk whole-decode checkpoint)",
            args.vae_decode_bsz,
        )
    if is_enabled():
        logger.info("[Tokenizer] Broadcasting weights from rank 0 ...")
        broadcast_module_params(tok, src=0)
        logger.info("[Tokenizer] Broadcast done.")
    return tok
