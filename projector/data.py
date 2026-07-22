import logging
import os
from typing import Optional, Tuple

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, Subset

from utils.data_util import center_crop_arr


logger = logging.getLogger("FD_loss")


def resolve_imagefolder_root(data_path: str) -> str:
    train_dir = os.path.join(data_path, "train")
    if os.path.isdir(train_dir):
        return train_dir
    return data_path


def build_projector_train_loader(
    data_path: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    world_size: int = 1,
    rank: int = 0,
    subset_size: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """Build an ImageFolder train loader returning [0, 1] tensors."""
    train_dir = resolve_imagefolder_root(data_path)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    if subset_size is not None and subset_size > 0:
        n = min(subset_size, len(dataset))
        dataset = Subset(dataset, list(range(n)))

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    logger.info(
        "[projector] train images from %s: %d, batch_size/device=%d",
        train_dir,
        len(dataset),
        batch_size,
    )
    return loader, sampler
