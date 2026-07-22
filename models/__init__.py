from .denoiser_jit import JiTDenoiser_models
from utils.ema_util import EMAModel
from .denoiser_imf import iMFDenoiser_models
from .denoiser_pmf import pMFDenoiser_models


def __getattr__(name):
    # Avoid importing Diffusers (and its optional Accelerate/quantization
    # stack) for pixel-space JiT jobs that never instantiate a VAE.
    if name in {"DiffusersAutoencoderKL", "VAE_models"}:
        from .autoencoder import DiffusersAutoencoderKL, VAE_models

        globals().update(
            DiffusersAutoencoderKL=DiffusersAutoencoderKL,
            VAE_models=VAE_models,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
