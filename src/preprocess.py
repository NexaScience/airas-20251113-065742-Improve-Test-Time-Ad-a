"""
preprocess.py – data loading / preprocessing utilities.
This specialised version supports:
• Synthetic random data (for smoke tests)
• Any image dataset hosted on Hugging Face Hub (via `datasets`) – simply specify
  the `hf_name` in the YAML config. The field `order_key` can optionally be
  provided to sort the stream (e.g. severity 5→1 for ImageNet-C).

All images are resized to 256 (shorter side), centre-cropped to 224, converted
to tensors and normalised with ImageNet statistics by default. Custom mean/std
can be supplied in the YAML.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

# Third-party dependencies
from datasets import load_dataset  # type: ignore

# ------------------------------ Constants --------------------------------- #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ------------------------------ Synthetic --------------------------------- #

class SyntheticClassificationDataset(Dataset):
    """Small random dataset for smoke-tests."""

    def __init__(self, num_samples: int, num_classes: int, input_shape: Tuple[int, int, int]):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.data = torch.rand(num_samples, *input_shape)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# --------------------------- Helper Classes -------------------------------- #

class HFDatasetWrapper(Dataset):
    """Wrap a 🤗 datasets object so that it looks like a PyTorch Dataset."""

    def __init__(self, hf_dataset, num_classes: int, transform, order_key: str | None = None, descending: bool = True):
        self.ds = hf_dataset
        self.num_classes = num_classes
        self.transform = transform
        if order_key is not None and order_key in self.ds.column_names:
            # Pre-compute an index order based on the key.
            self.indices: List[int] = sorted(
                range(len(self.ds)), key=lambda i: self.ds[i][order_key], reverse=descending
            )
        else:
            self.indices = list(range(len(self.ds)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex = self.ds[self.indices[idx]]
        img = ex["image"]  # PIL.Image
        label_key = "label" if "label" in ex else ("labels" if "labels" in ex else None)
        if label_key is None:
            raise KeyError("Could not find label column in HF sample. Expected 'label' or 'labels'.")
        y = ex[label_key]
        return self.transform(img), torch.tensor(y, dtype=torch.long)


# --------------------------- Transform Helper ----------------------------- #

def build_transform(img_size: int = 224, mean: list | None = None, std: list | None = None):
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


# --------------------------- Dataset Factory ------------------------------ #

def _build_dataset(cfg: dict) -> Dataset:
    name = cfg["name"].lower()

    # ---------------- Synthetic --------------------------------------- #
    if name == "synthetic":
        return SyntheticClassificationDataset(
            num_samples=cfg.get("num_samples", 1024),
            num_classes=cfg["num_classes"],
            input_shape=tuple(cfg.get("input_shape", (3, 32, 32))),
        )

    # ---------------- HuggingFace datasets ---------------------------- #
    if "hf_name" in cfg:
        hf_name = cfg["hf_name"]
        split = cfg.get("split", "test")
        hf_dataset = load_dataset(hf_name, split=split)
        transform = build_transform(
            img_size=cfg.get("img_size", 224),
            mean=cfg.get("mean"),
            std=cfg.get("std"),
        )
        return HFDatasetWrapper(
            hf_dataset,
            num_classes=cfg["num_classes"],
            transform=transform,
            order_key=cfg.get("order_key"),
            descending=cfg.get("order_desc", True),
        )

    # ---------------- Local ImageFolder (generic) --------------------- #
    if name == "image_folder":
        data_root = Path(cfg["data_root"]).expanduser()
        transform = build_transform(
            img_size=cfg.get("img_size", 224),
            mean=cfg.get("mean"),
            std=cfg.get("std"),
        )
        return ImageFolder(str(data_root), transform=transform)

    raise NotImplementedError(f"Dataset {name} not implemented.")


# ------------------------- Dataloader Interface --------------------------- #

def get_dataloader(cfg: dict) -> DataLoader:
    dataset = _build_dataset(cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 128),
        shuffle=False,  # streaming order must be deterministic
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )