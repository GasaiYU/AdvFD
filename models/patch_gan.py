import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

class PatchDiscriminator(nn.Module):
    """Pix2Pix/VQGAN-style N-layer PatchGAN discriminator."""

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        max_channels=512,
        num_layers=3,
        use_batchnorm=True,
    ):
        super().__init__()
        norm_layer = nn.BatchNorm2d if use_batchnorm else nn.Identity
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, max_channels // base_channels)
            layers += [
                nn.Conv2d(
                    base_channels * nf_mult_prev,
                    base_channels * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(base_channels * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, max_channels // base_channels)
        layers += [
            nn.Conv2d(
                base_channels * nf_mult_prev,
                base_channels * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(base_channels * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]

        self.feature_dim = base_channels * nf_mult
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward_features(self, x):
        x = x.mul(2.0).sub(1.0)
        for layer in self.net[:-1]:
            x = layer(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        return self.net[-1](features)


def patch_discriminator_loss(discriminator, real_images, fake_images):
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images.detach())
    loss_real = F.relu(1.0 - real_logits).mean()
    loss_fake = F.relu(1.0 + fake_logits).mean()
    return 0.5 * (loss_real + loss_fake), real_logits, fake_logits


class _PreserveBatchNormRunningStats:
    """Reusable context that prevents recomputation from updating BN twice."""

    def __init__(self, module):
        self.module = module
        self.states = None

    def __enter__(self):
        self.states = []
        for child in self.module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm) and child.track_running_stats:
                self.states.append(
                    (
                        child,
                        child.running_mean.detach().clone(),
                        child.running_var.detach().clone(),
                        child.num_batches_tracked.detach().clone(),
                    )
                )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with torch.no_grad():
            for child, running_mean, running_var, num_batches_tracked in self.states:
                child.running_mean.copy_(running_mean)
                child.running_var.copy_(running_var)
                child.num_batches_tracked.copy_(num_batches_tracked)
        self.states = None
        return False


def patch_generator_loss(discriminator, fake_images, grad_checkpointing=False):
    if grad_checkpointing and torch.is_grad_enabled() and fake_images.requires_grad:
        logits = checkpoint(
            discriminator,
            fake_images,
            use_reentrant=False,
            context_fn=lambda: (
                nullcontext(),
                _PreserveBatchNormRunningStats(discriminator),
            ),
        )
    else:
        logits = discriminator(fake_images)
    return -logits.mean()
