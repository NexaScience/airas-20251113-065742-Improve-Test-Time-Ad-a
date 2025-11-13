"""
train.py – run **one** experimental variation (one seed, one method, one dataset)
• Loads variation-level config (passed by main.py together with run_id)
• Runs the complete stream-style Test-Time-Adaptation (TTA) loop
• Collects per-batch metrics, final metrics, runtime & memory
• Saves structured results to <results_dir>/<run_id>/results.json
• Prints JSON metrics to STDOUT so that main.py can mirror them live
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .preprocess import get_dataloader
from .model import (
    get_model,
    configure_model_for_tta,
    get_classifier_parameters,
    softmax_entropy,
)

# ----------------------------- Utility ------------------------------------ #

def set_random_seed(seed: int):
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(obj, file_path):
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


# ----------------------- Core Adaptation Routines ------------------------- #

E_MAX_CACHE = {}

def entropy_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Plain mean entropy (TENT)."""
    return softmax_entropy(outputs).mean(0)


def adaent_loss(outputs: torch.Tensor, num_classes: int) -> torch.Tensor:
    """AdaEnt loss = (E / E_max) * E with E_max = log C."""
    if num_classes not in E_MAX_CACHE:
        E_MAX_CACHE[num_classes] = float(np.log(num_classes))
    e = softmax_entropy(outputs).mean(0)
    return (e / E_MAX_CACHE[num_classes]) * e


def shot_im_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Information Maximisation loss used by SHOT-IM (Liang et al.).
    Consists of two terms: entropy minimisation and diversity maximisation.
    """
    p = torch.softmax(outputs, dim=1)
    entropy_term = softmax_entropy(outputs).mean(0)  # minimise
    p_mean = p.mean(0)
    diversity_term = -(p_mean * p_mean.log()).sum()  # maximise (=> minimise negative)
    return entropy_term - diversity_term


LOSS_REGISTRY = {
    "TENT": entropy_loss,
    "AdaEnt": adaent_loss,
    "AdaEnt+DELTA": adaent_loss,  # same base loss, extra regulariser added later
    "SHOT-IM": shot_im_loss,
    "OracleLR": entropy_loss,
}


# ------------------------- Main Training Loop ----------------------------- #

def run_experiment(cfg: dict, results_dir: Path):
    run_id = cfg["run_id"]
    seed = cfg.get("seed", 0)
    method_cfg = cfg["method"]
    method = method_cfg["name"]
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    # ------------------------------------------------------------------ #
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------- Data ------------------------------ #
    dl = get_dataloader(dataset_cfg)
    num_classes = dataset_cfg["num_classes"]

    # ------------------------------ Model ----------------------------- #
    model = get_model(model_cfg).to(device)
    model.eval()  # all TTA methods start from pretrained eval mode

    # Configure model & optimiser for TTA
    if method in {"TENT", "AdaEnt", "AdaEnt+DELTA", "OracleLR"}:
        optim_params, _ = configure_model_for_tta(model, include_layer_norm=True)
        optimiser = torch.optim.SGD(
            optim_params,
            lr=method_cfg.get("lr", 1e-3),
            momentum=method_cfg.get("momentum", 0.9),
        )
        # Keep a clone for DELTA regularisation if required
        if method == "AdaEnt+DELTA":
            init_param_copies = [p.detach().clone() for p in optim_params]
            delta_lambda = method_cfg.get("delta_lambda", 0.1)
    elif method == "SHOT-IM":
        optim_params: List[torch.nn.Parameter] = get_classifier_parameters(model)
        optimiser = torch.optim.SGD(optim_params, lr=method_cfg.get("lr", 1e-3), momentum=0.9)
    elif method == "Frozen":
        optimiser = None
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    # ----------------------------- Metrics ---------------------------- #
    stream_acc = []  # accuracy after *processing* each batch
    entropy_history = []
    adaptive_factor_history = []  # E/E_max for AdaEnt variants, else ones

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # ----------------------------- Stream ----------------------------- #
    n_processed = 0
    correct_so_far = 0

    loss_fn = LOSS_REGISTRY[method]

    for batch_idx, (x, y) in enumerate(dl):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Forward pass
        outputs = model(x)

        # -------------------- Adapt --------------------------- #
        if method != "Frozen":
            if method in {"AdaEnt", "AdaEnt+DELTA"}:
                main_loss = loss_fn(outputs, num_classes)
            else:
                main_loss = loss_fn(outputs)

            # Optional DELTA regulariser
            if method == "AdaEnt+DELTA":
                delta_reg = 0.0
                for p, p0 in zip(optim_params, init_param_copies):
                    delta_reg += torch.nn.functional.mse_loss(p, p0, reduction="sum")
                main_loss = main_loss + delta_lambda * delta_reg

            main_loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            if method.startswith("AdaEnt"):
                with torch.no_grad():
                    ent = softmax_entropy(outputs).mean(0)
                    adaptive_factor_history.append(float(ent / np.log(num_classes)))
        else:
            main_loss = torch.tensor(0.0)

        # Evaluate AFTER adaptation (common in TENT literature)
        with torch.no_grad():
            outputs_post = model(x)
            preds_post = outputs_post.argmax(dim=1)
            correct_so_far += (preds_post == y).sum().item()
            n_processed += y.size(0)
            acc_stream = correct_so_far / n_processed

        stream_acc.append(acc_stream)
        entropy_history.append(float(softmax_entropy(outputs_post).mean()))

    runtime = time.time() - start_time
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
    )

    # --------------------------- Final Metrics ------------------------- #
    final_acc = stream_acc[-1]
    target = 0.9 * final_acc
    b90 = next((i + 1 for i, acc in enumerate(stream_acc) if acc >= target), len(stream_acc))

    results = {
        "run_id": run_id,
        "seed": seed,
        "config": cfg,
        "metrics": {
            "accuracy_vs_batches": stream_acc,
            "entropy_vs_batches": entropy_history,
            "adaptive_factor": adaptive_factor_history,
        },
        "final": {
            "accuracy": final_acc,
            "b90": b90,
            "runtime_seconds": runtime,
            "peak_mem_mb": peak_mem_mb,
        },
    }

    # ------------------------- Persist Results ------------------------ #
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    json_dump(results, run_dir / "results.json")

    # Also dump to STDOUT so main.py can tee
    print(json.dumps(results))


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-path", type=str, required=True)
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--results-dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    config_all = json.load(open(args.config_path, "r", encoding="utf-8"))
    cfg = next(exp for exp in config_all["experiments"] if exp["run_id"] == args.run_id)
    run_experiment(cfg, Path(args.results_dir))


if __name__ == "__main__":
    main()