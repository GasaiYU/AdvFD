"""Frozen feature extractors for Frechet distance computation.

Supports timm models (DINOv2, CLIP, etc.), ConvNeXt, InceptionV3, and
Qwen3-VL vision encoders.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


logger = logging.getLogger("FD_loss")

# Shared ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
QWEN3VL_2B_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
INTERNVIT_300M_MODEL_ID = "OpenGVLab/InternViT-300M-448px"


def safe_model_name(name: str) -> str:
    """Filesystem-safe, stable model name used for stats filenames."""
    return (
        name.replace("/", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace("-", "_")
        .lower()
    )


def inception_feature_layer_from_name(name: str) -> Optional[str]:
    """Return requested Inception layer for names like ``inception:Mixed_6e``."""
    low = name.lower()
    if low == "inception":
        return "pool3"
    if low.startswith("inception:"):
        layer = name.split(":", 1)[1]
    elif low.startswith("inception_"):
        layer = name[len("inception_"):]
    else:
        return None

    from utils.perception_util import canonical_inception_feature_layer

    return canonical_inception_feature_layer(layer)


def _preprocess(x, mean, std, target_size=None):
    """[0,1] float -> resize to target_size -> ImageNet normalize."""
    if target_size is not None and (x.shape[-2] != target_size or x.shape[-1] != target_size):
        x = F.interpolate(x, size=(target_size, target_size), mode="bicubic",
                          align_corners=False, antialias=True)
    return (x - mean) / std


def is_qwen3vl_model(name: str) -> bool:
    low = name.lower()
    return low in {
        "qwen3vl",
        "qwen3vl_2b",
        "qwen3-vl",
        "qwen3-vl-2b",
        "qwen3-vl-2b-instruct",
    } or "qwen3-vl" in low or "qwen3_vl" in low


def canonical_qwen3vl_model_name(name: str) -> str:
    if name.lower() in {
        "qwen3vl",
        "qwen3vl_2b",
        "qwen3-vl",
        "qwen3-vl-2b",
        "qwen3-vl-2b-instruct",
    }:
        return QWEN3VL_2B_MODEL_ID
    return name


def is_internvl_model(name: str) -> bool:
    low = name.lower()
    return low in {
        "internvl",
        "internvit",
        "internvit_300m",
        "internvit-300m",
    } or "internvl" in low or "internvit" in low


def canonical_internvl_model_name(name: str) -> str:
    if name.lower() in {
        "internvl",
        "internvit",
        "internvit_300m",
        "internvit-300m",
    }:
        return INTERNVIT_300M_MODEL_ID
    return name


class TimmReprModel(torch.nn.Module):
    """Wraps a timm model as a frozen feature extractor.

    Handles preprocessing: [0, 1] -> resize -> ImageNet normalize.
    Returns ``(cls_token, mean_token)``.
    """

    def __init__(
        self,
        model_name: str,
        device="cuda",
        target_size: Optional[int] = None,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        logger.info(f"[TimmReprModel] Loading model: {model_name}")
        # dynamic_img_size/pad only supported by ViT-like models
        kwargs = dict(pretrained=True, num_classes=0)
        try:
            self.model = timm.create_model(model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs)
        except TypeError:
            self.model = timm.create_model(model_name, **kwargs)
        self.model.to(device).eval().requires_grad_(False)
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.has_attn_pool = hasattr(self.model, "attn_pool") and self.model.attn_pool is not None
        self.feat_dim = self.model.num_features
        self.grad_checkpointing = grad_checkpointing

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        native_size = data_cfg["input_size"][-1]  # (C, H, W) -> W
        if "naflex" in model_name.lower():
            native_size = 256
        if target_size is not None and target_size != native_size:
            self.target_size = target_size
            logger.info(f"[TimmReprModel] Overriding target_size: {native_size} -> {target_size}")
        else:
            self.target_size = native_size

        mean = torch.tensor(data_cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(data_cfg["std"], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        interpolation = data_cfg.get("interpolation", "bicubic")
        logger.info(
            f"[TimmReprModel] {model_name}: feat_dim={self.feat_dim}, "
            f"target_size={self.target_size}, interpolation={interpolation}, "
            f"mean={data_cfg['mean']}, std={data_cfg['std']}, "
            f"grad_checkpointing={grad_checkpointing}"
        )

    def forward(self, x: torch.Tensor):
        x = _preprocess(x, self.mean, self.std, self.target_size)
        trainable_params = any(p.requires_grad for p in self.model.parameters())
        needs_grad = x.requires_grad or trainable_params
        if self.grad_checkpointing and torch.is_grad_enabled() and needs_grad:
            feats = checkpoint(lambda inp: self.model.forward_features(inp), x, use_reentrant=False)
        else:
            feats = self.model.forward_features(x)
        # CNN models return (B, C, H, W); pool spatially
        if feats.ndim == 4:
            cls_token = feats.mean(dim=[2, 3])
            return cls_token, None
        # ViT models return (B, N, C)
        patch_tokens = feats[:, self.num_prefix_tokens :]
        mean_token = patch_tokens.mean(1)
        if self.num_prefix_tokens > 0:
            cls_token = feats[:, 0]
        elif self.has_attn_pool:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            cls_token = pool(feats)
        else:
            cls_token = mean_token
        return cls_token, mean_token

    def forward_tokens(self, x: torch.Tensor):
        """Return full feature tokens for token-level projector training."""
        x = _preprocess(x, self.mean, self.std, self.target_size)
        feats = self.model.forward_features(x)
        if feats.ndim == 4:
            return feats.flatten(2).transpose(1, 2).float()
        return feats.float()


class LoRALinear(torch.nn.Module):
    """Low-rank trainable update around a frozen Linear layer."""

    def __init__(
        self,
        base: torch.nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.lora_A = torch.nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = torch.nn.Linear(self.rank, base.out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)
        for p in self.lora_A.parameters():
            p._fd_adv_trainable = True
        for p in self.lora_B.parameters():
            p._fd_adv_trainable = True

    def forward(self, x: torch.Tensor):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class LoRAQKVLinear(torch.nn.Module):
    """LoRA update for fused timm QKV layers, optionally leaving K frozen."""

    def __init__(
        self,
        base: torch.nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        update_k: bool = True,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        if base.out_features % 3 != 0:
            raise ValueError("LoRAQKVLinear requires out_features divisible by 3")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.update_k = bool(update_k)
        self.head_dim = base.out_features // 3
        out_features = base.out_features if self.update_k else 2 * self.head_dim
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.lora_A = torch.nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = torch.nn.Linear(self.rank, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)
        for p in self.lora_A.parameters():
            p._fd_adv_trainable = True
        for p in self.lora_B.parameters():
            p._fd_adv_trainable = True

    def forward(self, x: torch.Tensor):
        base_out = self.base(x)
        low_rank = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        if self.update_k:
            return base_out + low_rank
        q_update, v_update = low_rank.split(self.head_dim, dim=-1)
        update = torch.zeros_like(base_out)
        update[..., : self.head_dim] = q_update
        update[..., 2 * self.head_dim :] = v_update
        return base_out + update


def _get_submodule(root: torch.nn.Module, path: str):
    module = root
    for name in path.split("."):
        module = getattr(module, name)
    return module


def _matches_lora_target(module_name: str, target: str) -> bool:
    low_name = module_name.lower()
    target = target.lower().strip()
    if not target:
        return False
    if "." in target:
        if target == "attn.qv":
            return low_name.endswith(".attn.qkv")
        return low_name == target or low_name.endswith("." + target)
    if target == "qv":
        return low_name.endswith(".attn.qkv")
    if target == "qkv":
        return low_name.endswith(".attn.qkv")
    if target == "proj":
        return low_name.endswith(".attn.proj")
    return low_name.rsplit(".", 1)[-1] == target


def apply_lora_to_timm_repr_model(
    repr_model: torch.nn.Module,
    rank: int,
    alpha: float,
    target_names: Tuple[str, ...] = ("attn.qkv", "attn.proj"),
    dropout: float = 0.0,
) -> int:
    """Apply LoRA to matching Linear modules inside a TimmReprModel.

    Returns the number of wrapped Linear modules.
    """
    if rank <= 0:
        return 0
    if not hasattr(repr_model, "model"):
        raise ValueError("LoRA is currently supported only for timm repr models")

    backbone = repr_model.model
    targets = tuple(t.lower() for t in target_names)
    replacements = []
    for module_name, module in backbone.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(_matches_lora_target(module_name, target) for target in targets):
            parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
            replacements.append((parent_name, child_name, module))

    if not replacements:
        raise ValueError(
            f"No Linear modules matched LoRA targets {target_names} in {type(backbone).__name__}"
        )

    backbone.requires_grad_(False)
    for parent_name, child_name, module in replacements:
        parent = _get_submodule(backbone, parent_name) if parent_name else backbone
        module_path = f"{parent_name}.{child_name}" if parent_name else child_name
        target_set = set(targets)
        qv_only = module_path.lower().endswith(".attn.qkv") and (
            "qv" in target_set or "attn.qv" in target_set
        ) and "qkv" not in target_set and "attn.qkv" not in target_set
        if module_path.lower().endswith(".attn.qkv"):
            wrapped = LoRAQKVLinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                update_k=not qv_only,
            )
        else:
            wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, wrapped)
    return len(replacements)


class Qwen3VLReprModel(torch.nn.Module):
    """Qwen3-VL vision encoder wrapper for differentiable FD loss.

    The Hugging Face processor patchifies images on CPU/PIL objects, which would
    break gradients from FD loss to generated images.  This wrapper mirrors the
    Qwen2/3-VL fast image processor's tensor patch layout in torch, then runs
    only the frozen vision tower.
    """

    def __init__(
        self,
        model_name: str = QWEN3VL_2B_MODEL_ID,
        device="cuda",
        target_size: Optional[int] = 256,
        grad_checkpointing: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        model_name = canonical_qwen3vl_model_name(model_name)
        self.model_name = model_name
        self.target_size = 256 if target_size is None else int(target_size)
        self.grad_checkpointing = grad_checkpointing

        try:
            from transformers import Qwen3VLForConditionalGeneration
            model_cls = Qwen3VLForConditionalGeneration
        except ImportError:
            try:
                from transformers import AutoModelForMultimodalLM
                model_cls = AutoModelForMultimodalLM
            except ImportError as exc:
                raise ImportError(
                    "Qwen3-VL requires a recent transformers build. Install with "
                    "`pip install git+https://github.com/huggingface/transformers` "
                    "or a release that provides Qwen3VLForConditionalGeneration."
                ) from exc

        logger.info(f"[Qwen3VLReprModel] Loading model: {model_name}")
        full_model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )
        full_model.eval().requires_grad_(False)

        base_model = getattr(full_model, "model", full_model)
        if not hasattr(base_model, "visual"):
            raise AttributeError(
                f"Loaded {model_name}, but could not find the Qwen3-VL vision tower "
                "at model.visual"
            )
        self.visual = base_model.visual.to(device)
        self.visual.eval().requires_grad_(False)

        vision_cfg = getattr(full_model.config, "vision_config", None)
        if vision_cfg is None:
            vision_cfg = getattr(self.visual, "config", None)
        self.patch_size = int(getattr(vision_cfg, "patch_size", 16))
        self.temporal_patch_size = int(getattr(vision_cfg, "temporal_patch_size", 2))
        self.merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))
        self.feat_dim = int(getattr(vision_cfg, "out_hidden_size", 2048))
        self.factor = self.patch_size * self.merge_size
        if self.target_size % self.factor != 0:
            raise ValueError(
                f"Qwen3-VL target_size={self.target_size} must be divisible by "
                f"patch_size*merge_size={self.factor}"
            )

        # Matches Qwen VL / HF image processors after rescaling image pixels to [0, 1].
        mean = torch.tensor(OPENAI_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(OPENAI_CLIP_STD, device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Keep only the vision tower referenced by this module.
        del full_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "[Qwen3VLReprModel] %s: feat_dim=%s, target_size=%s, "
            "patch_size=%s, temporal_patch_size=%s, merge_size=%s, "
            "grad_checkpointing=%s",
            model_name,
            self.feat_dim,
            self.target_size,
            self.patch_size,
            self.temporal_patch_size,
            self.merge_size,
            grad_checkpointing,
        )

    def _patchify(self, x: torch.Tensor):
        if x.shape[-2] != self.target_size or x.shape[-1] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        x = (x - self.mean) / self.std

        batch_size, channels, height, width = x.shape
        grid_t = 1
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        patches = x.unsqueeze(1).expand(
            batch_size, self.temporal_patch_size, channels, height, width
        )
        patches = patches.contiguous().view(
            batch_size,
            grid_t,
            self.temporal_patch_size,
            channels,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        ).reshape(-1, channels * self.temporal_patch_size * self.patch_size * self.patch_size)
        grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * batch_size,
            device=x.device,
            dtype=torch.long,
        )
        return pixel_values, grid_thw

    def _visual_forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        output = self.visual(pixel_values, grid_thw=grid_thw, return_dict=True)
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if isinstance(output, (tuple, list)):
            return output[1] if len(output) > 1 else output[0]
        return output

    def forward(self, x: torch.Tensor):
        pixel_values, grid_thw = self._patchify(x)
        if self.grad_checkpointing and torch.is_grad_enabled() and x.requires_grad:
            tokens = checkpoint(
                lambda pv: self._visual_forward(pv, grid_thw),
                pixel_values,
                use_reentrant=False,
            )
        else:
            tokens = self._visual_forward(pixel_values, grid_thw)

        split_size = int(grid_thw[0].prod().item() // (self.merge_size ** 2))
        tokens = tokens.view(x.shape[0], split_size, self.feat_dim)
        mean_token = tokens.mean(dim=1)
        return mean_token, mean_token


class InternVLReprModel(torch.nn.Module):
    """InternVL/InternViT vision encoder wrapper for differentiable FD loss."""

    def __init__(
        self,
        model_name: str = INTERNVIT_300M_MODEL_ID,
        device="cuda",
        target_size: Optional[int] = 448,
        grad_checkpointing: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        model_name = canonical_internvl_model_name(model_name)
        self.model_name = model_name
        self.target_size = 448 if target_size is None else int(target_size)
        self.grad_checkpointing = grad_checkpointing

        from transformers import AutoModel

        logger.info(f"[InternVLReprModel] Loading model: {model_name}")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )

        vision_model = (
            getattr(model, "vision_model", None)
            or getattr(model, "visual", None)
            or getattr(model, "vision_tower", None)
            or model
        )
        self.model = vision_model.to(device)
        self.model.eval().requires_grad_(False)

        cfg = getattr(self.model, "config", None) or getattr(model, "config", None)
        self.feat_dim = int(
            getattr(cfg, "hidden_size", 0)
            or getattr(cfg, "embed_dim", 0)
            or getattr(cfg, "vision_hidden_size", 0)
        )
        if self.feat_dim <= 0:
            raise AttributeError(
                f"Could not infer InternVL feature dimension from {model_name} config"
            )

        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        if vision_model is not model:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(
            "[InternVLReprModel] %s: feat_dim=%s, target_size=%s, "
            "grad_checkpointing=%s",
            model_name,
            self.feat_dim,
            self.target_size,
            grad_checkpointing,
        )

    def _forward_model(self, x: torch.Tensor):
        try:
            return self.model(pixel_values=x, return_dict=True)
        except TypeError:
            try:
                return self.model(pixel_values=x)
            except TypeError:
                return self.model(x)

    def _forward_tokens(self, x: torch.Tensor):
        return self._extract_tokens(self._forward_model(x))

    @staticmethod
    def _extract_tokens(output):
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            return output.hidden_states[-1]
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def forward(self, x: torch.Tensor):
        x = _preprocess(x, self.mean, self.std, self.target_size)
        model_dtype = next(self.model.parameters()).dtype
        if x.dtype != model_dtype:
            x = x.to(dtype=model_dtype)
        if self.grad_checkpointing and torch.is_grad_enabled() and x.requires_grad:
            output = None
            tokens = checkpoint(lambda inp: self._forward_tokens(inp), x, use_reentrant=False)
        else:
            output = self._forward_model(x)
            tokens = self._extract_tokens(output)
        if tokens.ndim == 4:
            cls_token = tokens.mean(dim=[2, 3])
            return cls_token, cls_token
        if tokens.ndim != 3:
            raise RuntimeError(
                f"InternVLReprModel expected [B,N,C] or [B,C,H,W] output, got {tuple(tokens.shape)}"
            )

        if output is not None and hasattr(output, "pooler_output") and output.pooler_output is not None:
            cls_token = output.pooler_output
        else:
            cls_token = tokens[:, 0]
        return cls_token, cls_token


def load_repr_model(
    name: str,
    device="cuda",
    target_size: Optional[int] = None,
    grad_checkpointing: bool = False,
    inception_pretrained: bool = True,
):
    """Load a representation feature extractor.

    Each model handles its own input resolution internally based on its
    training configuration (e.g. timm models use ``pretrained_cfg['input_size']``).

    Args:
        name: ``'inception'``, ``'convnext'``, Qwen/InternVL aliases, or any timm model name.
        target_size: override the model's native target resolution.
        inception_pretrained: load torch-fidelity weights for Inception models.

    Returns:
        (model, feat_dim, has_logits, target_size)
    """
    inception_layer = inception_feature_layer_from_name(name)
    if inception_layer is not None:
        from utils.perception_util import INCEPTION_FEATURE_DIMS, load_inception

        net = load_inception(
            device=device,
            normalize=False,
            feature_layer=inception_layer,
            pretrained=inception_pretrained,
        )
        has_logits = inception_layer == "pool3"
        return net, INCEPTION_FEATURE_DIMS[inception_layer], has_logits, 299
    elif is_qwen3vl_model(name):
        net = Qwen3VLReprModel(
            name,
            device=device,
            target_size=target_size,
            grad_checkpointing=grad_checkpointing,
        )
        return net, net.feat_dim, False, net.target_size
    elif is_internvl_model(name):
        net = InternVLReprModel(
            name,
            device=device,
            target_size=target_size,
            grad_checkpointing=grad_checkpointing,
        )
        return net, net.feat_dim, False, net.target_size
    elif name == "convnext":
        net = TimmReprModel(
            "convnextv2_base.fcmae_ft_in22k_in1k",
            device=device,
            target_size=224,
            grad_checkpointing=grad_checkpointing,
        )
        return net, net.feat_dim, False, net.target_size
    else:
        net = TimmReprModel(
            name,
            device=device,
            target_size=target_size,
            grad_checkpointing=grad_checkpointing,
        )
        return net, net.feat_dim, False, net.target_size


def model_short_name(name: str) -> str:
    """Derive a concise label from a representation model name for logging/metrics."""
    inception_layer = inception_feature_layer_from_name(name)
    if inception_layer is not None:
        if inception_layer == "pool3":
            return "inception"
        return safe_model_name(f"inception_{inception_layer}")
    if name == "convnext":
        return name
    if is_qwen3vl_model(name):
        return "qwen3vl-2b"
    if is_internvl_model(name):
        low = name.lower()
        if "300m" in low:
            return "internvit-300m"
        return "internvl"
    low = name.lower()
    if "naflex" in low:
        return "naflex_siglip"
    for keyword in ("dinov2", "dino", "mae", "clip", "siglip"):
        if keyword in low:
            return keyword
    return name.split(".")[0].replace("_", "-")
