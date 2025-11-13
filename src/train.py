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
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa – kept for potential user extensions

from .preprocess import get_dataloader
from .model import (
    get_model,
    configure_model_for_tta,
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


# ----------------------- Loss Functions ----------------------------------- #

def entropy_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Plain mean entropy (TENT)."""
    return softmax_entropy(outputs).mean(0)


def adaent_loss(outputs: torch.Tensor, num_classes: int) -> torch.Tensor:
    """AdaEnt loss = (E / E_max) * E with E_max = log C."""
    e = softmax_entropy(outputs).mean(0)
    e_max = math.log(num_classes)
    return (e / e_max) * e


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
    if method in {"TENT", "AdaEnt", "AdaEntFixedAlpha"}:
        params, _ = configure_model_for_tta(model)
        optimiser = torch.optim.SGD(params, lr=method_cfg.get("lr", 1e-3))
    elif method == "Frozen":
        optimiser = None
    else:
        raise NotImplementedError(f"Unknown method: {method}")

    # ----------------------------- Metrics ---------------------------- #
    stream_acc: list[float] = []  # accuracy after *processing* each batch
    entropy_history: list[float] = []
    adaptive_factor_history: list[float] = []  # E/E_max for AdaEnt, else constant 1.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # ----------------------------- Stream ----------------------------- #
    n_processed = 0
    correct_so_far = 0

    for batch_idx, (x, y) in enumerate(dl):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Forward pass BEFORE adaptation (predictions used for loss)
        outputs = model(x)

        # -------------------- Adapt --------------------------- #
        if method != "Frozen":
            if method == "TENT":
                loss = entropy_loss(outputs)
            elif method == "AdaEnt":
                loss = adaent_loss(outputs, num_classes)
            elif method == "AdaEntFixedAlpha":
                alpha = method_cfg.get("alpha", 0.5)
                loss = alpha * entropy_loss(outputs)
            else:
                raise RuntimeError("Should not reach here – method handling incomplete.")

            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            # Record adaptive factor (AdaEnt) for analysis
            if method == "AdaEnt":
                with torch.no_grad():
                    entropy_val = softmax_entropy(outputs).mean(0).item()
                adaptive_factor_history.append(float(entropy_val / math.log(num_classes)))
            else:
                adaptive_factor_history.append(1.0)
        else:
            adaptive_factor_history.append(1.0)

        # -------------------- Evaluation ----------------------- #
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
    target = 0.9 * final_acc  # B90: 90 % of final accuracy
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
    # find the variation with matching run_id
    cfg = next(exp for exp in config_all["experiments"] if exp["run_id"] == args.run_id)
    run_experiment(cfg, Path(args.results_dir))


if __name__ == "__main__":
    main()