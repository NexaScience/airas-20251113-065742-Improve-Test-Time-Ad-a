"""
model.py – model architectures and adaptation utilities.
Currently supports ResNet-18 backbone with optional ImageNet pre-training.
"""
from __future__ import annotations

import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

# ------------------------- Utility ---------------------------------------- #

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy of the softmax distribution."""
    p = torch.softmax(x, dim=1)
    return -(p * p.log()).sum(1)


# ------------------------- Model Factory ---------------------------------- #

def _resnet18(num_classes: int, pretrained: bool = False):
    """ResNet-18 from torchvision.
    For torchvision >=0.13 the signature expects `weights` instead of `pretrained`.
    We support both to stay compatible with a wide range of versions.
    """
    try:
        # Newer API (torchvision >= 0.13)
        from torchvision.models import ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
    except ImportError:
        # Fallback to old API
        model = tvm.resnet18(pretrained=pretrained)

    if model.fc.in_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


_MODEL_REGISTRY = {
    "resnet18": _resnet18,
}


def get_model(cfg: Dict) -> nn.Module:
    name = cfg["name"].lower()
    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", False)

    if name not in _MODEL_REGISTRY:
        raise NotImplementedError(f"Model {name} not implemented.")

    return _MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)


# -------------------- Test-Time Adaptation Utilities ---------------------- #

def configure_model_for_tta(model: nn.Module):
    """Make BatchNorm affine parameters trainable; freeze others.
    Returns (optim_params, frozen_params).
    """
    optim_params = []
    frozen_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.requires_grad_(True)
            optim_params.extend([m.weight, m.bias])
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
                frozen_params.append(p)
    return optim_params, frozen_params