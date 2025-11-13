"""
model.py – model architectures and adaptation utilities.
Supported architectures
•   ResNet-18 (for smoke tests)
•   ResNet-50 (ImageNet-scale baseline)
•   ViT-B/16  (timm implementation)

Utilities
•   get_model(cfg) – instantiate model according to YAML spec
•   configure_model_for_tta – select affine parameters (BatchNorm & LayerNorm)
•   get_classifier_parameters – return classifier head parameters (for SHOT-IM)
•   softmax_entropy – entropy helper
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm
import timm  # type: ignore

# ------------------------- Utility ---------------------------------------- #

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy of the softmax distribution."""
    p = torch.softmax(x, dim=1)
    return -(p * p.log()).sum(1)


# ------------------------- Model Factory ---------------------------------- #

def _resnet18(num_classes: int, pretrained: bool = False):
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resnet50(num_classes: int, pretrained: bool = False):
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _vit_b16(num_classes: int, pretrained: bool = False):
    """ViT-B/16 via timm (patch_size 16, img_size 224)."""
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def get_model(cfg: dict) -> nn.Module:
    name = cfg["name"].lower()
    num_classes = cfg["num_classes"]
    pretrained = cfg.get("pretrained", False)

    if name == "resnet18":
        return _resnet18(num_classes, pretrained)
    if name == "resnet50":
        return _resnet50(num_classes, pretrained)
    if name in {"vit_b16", "vit-b16", "vit", "vit_base"}:
        return _vit_b16(num_classes, pretrained)

    raise NotImplementedError(f"Model '{name}' not implemented.")


# -------------------- Test-Time Adaptation Utilities ---------------------- #

def configure_model_for_tta(model: nn.Module, *, include_layer_norm: bool = False):
    """Make normalisation affine parameters trainable; freeze others.
    Returns (optim_params, frozen_params).
    If *include_layer_norm* is True, LayerNorms are also included (needed for ViT).
    """
    optim_params: List[nn.Parameter] = []
    frozen_params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) or (
            include_layer_norm and isinstance(m, nn.LayerNorm)
        ):
            m.requires_grad_(True)
            if hasattr(m, "weight") and m.weight is not None:
                optim_params.append(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                optim_params.append(m.bias)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)
                frozen_params.append(p)
    return optim_params, frozen_params


def get_classifier_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return parameters of the classification head (used by SHOT-IM)."""
    # Common attribute names across architectures
    head_candidates: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            head_candidates.append((name, module))
    # Heuristic: pick the *last* Linear layer encountered (deepest)
    if not head_candidates:
        raise RuntimeError("Could not locate a Linear classification head in the given model.")
    head_module = head_candidates[-1][1]
    return list(head_module.parameters())