from typing import Any, Dict, Tuple

import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader


def _default_collate(batch):
    # batch: list[dict]
    # collate wav to [B, T]
    wav = torch.stack([b["wav"] for b in batch], dim=0)
    paths = [b.get("path", "") for b in batch]
    return {"wav": wav, "path": paths}


def get_dataloaders(config, device: str) -> Tuple[Dict[str, DataLoader], Any]:
    """
    Returns:
      dataloaders: dict with at least "train"
      batch_transforms: callable or None (applied to already-collated batch)
    """
    train_ds = instantiate(config.datasets.train)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=(device.startswith("cuda")),
        drop_last=True,
        collate_fn=_default_collate,
    )

    dataloaders = {"train": train_loader}
    
    batch_transforms = None
    if "batch_transforms" in config and config.batch_transforms is not None:
        batch_transforms = instantiate(config.batch_transforms)
        if hasattr(batch_transforms, "to"):
            batch_transforms = batch_transforms.to(device)

    return dataloaders, batch_transforms
