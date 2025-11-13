"""
preprocess.py – data loading / preprocessing utilities specialised for
CIFAR-10-C / CIFAR-100-C as required by exp-1-cifar-core.
A Synthetic dataset remains available for smoke tests.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# External dependency. The datasets package is lightweight and already declared.
from datasets import load_dataset

# ------------------------------ Synthetic --------------------------------- #

class SyntheticClassificationDataset(Dataset):
    """Small random dataset for smoke-tests.
    Generates images ∈[0,1] with random labels ∈[0, num_classes).
    """

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


# --------------------------- CIFAR-C Dataset ------------------------------ #

_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2471, 0.2435, 0.2616)

_transform_32 = T.Compose(
    [T.ToTensor(), T.Normalize(_CIFAR_MEAN, _CIFAR_STD)],
)


class HFCIFARC(Dataset):
    """Torch-compatible wrapper around the Hugging-Face parquet versions of
    CIFAR-10-C / CIFAR-100-C.
    The dataset is *streamed in the order (severity list provided)* followed by the
    natural order inside each severity. No shuffling takes place so that every
    run observes the same stream given the same random seed.
    """

    _HF_NAME_MAP = {
        "cifar10c": "robro/cifar10-c-parquet",
        "cifar100c": "robro/cifar100-c-parquet",
    }

    def __init__(
        self,
        name: str,
        severities: List[int] | None = None,
        corruption_types: List[str] | None = None,
        transform: T.Compose | None = None,
    ):
        if name not in self._HF_NAME_MAP:
            raise ValueError(f"Unknown CIFAR-C variant: {name}")
        hf_name = self._HF_NAME_MAP[name]

        # Load split – parquet dataset stores everything in a single split "train"
        full_ds = load_dataset(hf_name, split="train", trust_remote_code=False)

        # Keep only requested severities and corruption types ---------------------------------
        if severities is None:
            severities = [5, 4, 3, 2, 1]
        if corruption_types is None:
            corruption_types = list(set(full_ds["corruption_type"]))  # all types

        # Build indices in required order (outer severity loop preserves requested order)
        ordered_indices = []
        for sev in severities:
            idx_sev = [i for i, s in enumerate(full_ds["severity"]) if s == sev]
            # Further filter by corruption types
            idx_sev = [i for i in idx_sev if full_ds[i]["corruption_type"] in corruption_types]
            ordered_indices.extend(idx_sev)

        self._ds = full_ds  # keep reference for __getitem__
        self._indices = ordered_indices
        self.transform = transform if transform is not None else _transform_32

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        example = self._ds[real_idx]
        img = example["image"]  # PIL.Image
        label = example["label"]
        img = self.transform(img)
        return img, int(label)


# --------------------------- Dataset Factory ------------------------------ #

def _build_dataset(cfg: dict) -> Dataset:
    name = cfg["name"].lower()

    # Synthetic (for smoke tests)
    if name == "synthetic":
        return SyntheticClassificationDataset(
            num_samples=cfg.get("num_samples", 1024),
            num_classes=cfg["num_classes"],
            input_shape=tuple(cfg.get("input_shape", (3, 32, 32))),
        )

    # CIFAR-10-C / CIFAR-100-C ----------------------------------------------------------------
    if name in {"cifar10c", "cifar100c"}:
        return HFCIFARC(
            name=name,
            severities=cfg.get("severities", [5, 4, 3, 2, 1]),
            corruption_types=cfg.get("corruption_types"),
            transform=_transform_32,
        )

    # Clean CIFAR variants could be added here if needed (not required for core experiment)

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