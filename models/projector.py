import math
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


PROJECTOR_BACKBONE_CONFIGS = {
    "vit_s": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "head_hidden_dim": 1536,
    },
    "vit_b": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "head_hidden_dim": 3072,
    },
}
PROJECTOR_BACKBONE_ALIASES = {
    "s": "vit_s",
    "small": "vit_s",
    "vit_s": "vit_s",
    "vits": "vit_s",
    "vit_small": "vit_s",
    "b": "vit_b",
    "base": "vit_b",
    "vit_b": "vit_b",
    "vitb": "vit_b",
    "vit_base": "vit_b",
}


def canonical_projector_backbone(backbone: str) -> str:
    key = backbone.strip().lower().replace("-", "_")
    if key not in PROJECTOR_BACKBONE_ALIASES:
        valid = ", ".join(sorted(PROJECTOR_BACKBONE_CONFIGS))
        raise ValueError(f"Unsupported projector backbone '{backbone}'. Expected one of: {valid}")
    return PROJECTOR_BACKBONE_ALIASES[key]


def projector_backbone_config(backbone: str) -> Dict[str, int]:
    return dict(PROJECTOR_BACKBONE_CONFIGS[canonical_projector_backbone(backbone)])


def infer_projector_backbone(
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    default: str = "vit_s",
) -> str:
    for name, config in PROJECTOR_BACKBONE_CONFIGS.items():
        if (
            int(patch_size) == config["patch_size"]
            and int(embed_dim) == config["embed_dim"]
            and int(depth) == config["depth"]
            and int(num_heads) == config["num_heads"]
        ):
            return name
    return canonical_projector_backbone(default)


def projector_backbone_tag(backbone: str) -> str:
    return canonical_projector_backbone(backbone).replace("_", "")


class PatchEmbed(nn.Module):
    """Image to patch tokens."""

    def __init__(self, img_size: int = 256, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MultiheadSelfAttention(nn.Module):
    """Self-attention with separate Q/K/V parameters for matrix-aware optimizers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, dim = x.shape
        return x.reshape(bsz, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.num_heads * self.head_dim)
        return self.out_proj(x)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        old_weight_key = prefix + "in_proj_weight"
        old_bias_key = prefix + "in_proj_bias"
        if old_weight_key in state_dict and prefix + "q_proj.weight" not in state_dict:
            q_weight, k_weight, v_weight = state_dict.pop(old_weight_key).chunk(3, dim=0)
            state_dict[prefix + "q_proj.weight"] = q_weight
            state_dict[prefix + "k_proj.weight"] = k_weight
            state_dict[prefix + "v_proj.weight"] = v_weight
        if old_bias_key in state_dict and prefix + "q_proj.bias" not in state_dict:
            q_bias, k_bias, v_bias = state_dict.pop(old_bias_key).chunk(3, dim=0)
            state_dict[prefix + "q_proj.bias"] = q_bias
            state_dict[prefix + "k_proj.bias"] = k_bias
            state_dict[prefix + "v_proj.bias"] = v_bias
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, dropout: float = 0.0,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiheadSelfAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.attn(h)
        x = x + self.drop1(h)
        x = x + self.mlp(self.norm2(x))
        return x


class TokenProjectionHead(nn.Module):
    """Token-wise MLP head. With one layer this is exactly one Linear."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 1,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("head MLP must have at least one layer")
        if num_layers == 1:
            self.net = nn.Linear(in_dim, out_dim)
            return

        hidden_dim = hidden_dim if hidden_dim > 0 else in_dim
        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvProjectionHead(nn.Module):
    """Convolutional projection head for grid-shaped token features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 1,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        kernel_size: int = 3,
        output_grid: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("conv head must have at least one layer")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("conv head kernel_size must be a positive odd integer")

        hidden_dim = hidden_dim if hidden_dim > 0 else in_dim
        padding = kernel_size // 2
        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                nn.GELU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            dim = hidden_dim
        layers.append(nn.Conv2d(dim, out_dim, kernel_size=kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)
        self.output_grid = output_grid

    @staticmethod
    def _tokens_to_grid(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim != 3:
            raise RuntimeError(f"ConvProjectionHead expected [B,T,C], got {tuple(x.shape)}")
        grid = int(math.sqrt(x.shape[1]))
        if grid * grid != x.shape[1]:
            raise RuntimeError(
                "ConvProjectionHead requires a square token grid, got %d tokens" % x.shape[1]
            )
        x = x.reshape(x.shape[0], grid, grid, x.shape[2]).permute(0, 3, 1, 2)
        return x, (grid, grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, grid = self._tokens_to_grid(x)
        x = self.net(x)
        if self.output_grid is not None and grid != self.output_grid:
            x = F.adaptive_avg_pool2d(x, self.output_grid)
        return x.permute(0, 2, 3, 1).flatten(1, 2)


class ViTSMultiHeadProjector(nn.Module):
    """ViT token projector with fixed token-layout versions.

    Version I outputs 257 backbone tokens: one prefix token + 256 patches.
    Version II outputs 258 backbone tokens: two prefix tokens + 256 patches.
    Version III outputs three prefix tokens + patches; prefix2 is intended for
    a pooled Inception FD feature.
    Heads are token-wise MLPs and default to a single Linear layer.
    """

    VALID_SOURCE_MODES = {
        "patches",
        "prefix0_patches",
        "prefix0",
        "prefix1",
        "prefix2",
    }

    def __init__(
        self,
        head_dims: Dict[str, int],
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        projector_version: str = "I",
        head_source_modes: Optional[Dict[str, str]] = None,
        head_target_grids: Optional[Dict[str, Tuple[int, int]]] = None,
        head_types: Optional[Dict[str, str]] = None,
        head_mlp_layers: int = 1,
        head_mlp_layer_overrides: Optional[Dict[str, int]] = None,
        head_hidden_dim: int = 0,
        head_conv_kernel_size: int = 3,
        head_dropout: float = 0.0,
        normalize_input: bool = True,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if not head_dims:
            raise ValueError("head_dims must contain at least one head")
        if projector_version not in ("I", "II", "III"):
            raise ValueError("projector_version must be 'I', 'II', or 'III'")

        self.projector_version = projector_version
        self.num_prefix_tokens = {"I": 1, "II": 2, "III": 3}[projector_version]
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.prefix_tokens = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        head_source_modes = head_source_modes or {}
        head_target_grids = head_target_grids or {}
        head_types = head_types or {}
        head_mlp_layer_overrides = head_mlp_layer_overrides or {}
        self.head_source_modes = {}
        self.head_target_grids = {}
        self.head_types = {}
        self.heads = nn.ModuleDict()
        for name, dim in head_dims.items():
            source_mode = head_source_modes.get(name, "patches")
            if source_mode not in self.VALID_SOURCE_MODES:
                raise ValueError(f"Unsupported source mode for head '{name}': {source_mode}")
            head_type = head_types.get(name, "mlp")
            if head_type not in ("mlp", "conv"):
                raise ValueError(f"Unsupported head type for head '{name}': {head_type}")
            num_head_layers = head_mlp_layer_overrides.get(name, head_mlp_layers)
            self.head_source_modes[name] = source_mode
            self.head_target_grids[name] = head_target_grids.get(name, self.patch_embed.grid_size)
            self.head_types[name] = head_type
            if head_type == "conv":
                self.heads[name] = ConvProjectionHead(
                    embed_dim,
                    dim,
                    num_layers=num_head_layers,
                    hidden_dim=head_hidden_dim,
                    dropout=head_dropout,
                    kernel_size=head_conv_kernel_size,
                    output_grid=self.head_target_grids[name],
                )
            else:
                self.heads[name] = TokenProjectionHead(
                    embed_dim,
                    dim,
                    num_layers=num_head_layers,
                    hidden_dim=head_hidden_dim,
                    dropout=head_dropout,
                )

        self.head_dims = dict(head_dims)
        self.grad_checkpointing = grad_checkpointing
        self.normalize_input = normalize_input

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean, persistent=False)
        self.register_buffer("image_std", std, persistent=False)

        self._init_weights()

    @property
    def head_names(self) -> Iterable[str]:
        return self.heads.keys()

    def no_weight_decay(self) -> set:
        return {"pos_embed", "prefix_tokens"}

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.prefix_tokens, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _pos_embed_for(self, h: int, w: int) -> torch.Tensor:
        gh = h // self.patch_embed.patch_size
        gw = w // self.patch_embed.patch_size
        if (gh, gw) == self.patch_embed.grid_size:
            return self.pos_embed

        prefix_pos = self.pos_embed[:, :self.num_prefix_tokens]
        patch_pos = self.pos_embed[:, self.num_prefix_tokens:]
        old_h, old_w = self.patch_embed.grid_size
        patch_pos = patch_pos.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2)
        patch_pos = torch.nn.functional.interpolate(
            patch_pos,
            size=(gh, gw),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        return torch.cat([prefix_pos, patch_pos], dim=1)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = (x - self.image_mean) / self.image_std

        h, w = x.shape[-2:]
        x = self.patch_embed(x)
        prefix = self.prefix_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat([prefix, x], dim=1)
        x = x + self._pos_embed_for(h, w).to(dtype=x.dtype, device=x.device)
        x = self.pos_drop(x)

        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        return self.norm(x)

    def _patch_tokens_for(self, tokens: torch.Tensor, target_grid: Tuple[int, int]) -> torch.Tensor:
        patch_tokens = tokens[:, self.num_prefix_tokens:]
        src_h, src_w = self.patch_embed.grid_size
        if target_grid == self.patch_embed.grid_size:
            return patch_tokens

        bsz, _, dim = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(bsz, src_h, src_w, dim).permute(0, 3, 1, 2)
        patch_tokens = torch.nn.functional.interpolate(
            patch_tokens,
            size=target_grid,
            mode="bicubic",
            align_corners=False,
        )
        tgt_h, tgt_w = target_grid
        return patch_tokens.permute(0, 2, 3, 1).reshape(bsz, tgt_h * tgt_w, dim)

    def _select_head_tokens(self, tokens: torch.Tensor, name: str) -> torch.Tensor:
        mode = self.head_source_modes[name]
        if self.head_types.get(name) == "conv" and mode in ("patches", "prefix0_patches"):
            target_grid = self.patch_embed.grid_size
        else:
            target_grid = self.head_target_grids[name]
        if mode == "patches":
            return self._patch_tokens_for(tokens, target_grid)
        if mode == "prefix0_patches":
            patch_tokens = self._patch_tokens_for(tokens, target_grid)
            return torch.cat([tokens[:, :1], patch_tokens], dim=1)
        if mode == "prefix0":
            return tokens[:, :1]
        if mode == "prefix1":
            if self.num_prefix_tokens < 2:
                raise RuntimeError("prefix1 source requires projector version II or III")
            return tokens[:, 1:2]
        if mode == "prefix2":
            if self.num_prefix_tokens < 3:
                raise RuntimeError("prefix2 source requires projector version III")
            return tokens[:, 2:3]
        raise NotImplementedError(mode)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)[:, 0]

    def forward(self, x: torch.Tensor,
                head_names: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
        tokens = self.forward_tokens(x)
        names = list(head_names) if head_names is not None else list(self.heads.keys())
        return {
            name: self.heads[name](self._select_head_tokens(tokens, name))
            for name in names
        }


def build_vits_projector(head_dims: Dict[str, int], **kwargs) -> ViTSMultiHeadProjector:
    return ViTSMultiHeadProjector(head_dims=head_dims, **kwargs)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
