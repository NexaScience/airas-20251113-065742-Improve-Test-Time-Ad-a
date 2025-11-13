"""
evaluate.py – read *all* results.json files under a results_dir,
aggregate across seeds & variations, compute statistics, and output PDF figures.
Figures are stored in <results_dir>/images/ according to the naming convention.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

# ----------------------- Helper Functions --------------------------------- #

def load_all_results(results_dir: Path):
    records = []
    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir():
            continue
        res_file = run_dir / "results.json"
        if res_file.exists():
            with open(res_file, "r", encoding="utf-8") as fh:
                records.append(json.load(fh))
    return records


def aggregate(records):
    """Aggregate metrics over seeds for identical (dataset, model, method)."""
    key_fn = lambda r: (
        r["config"]["dataset"]["name"],
        r["config"]["model"]["name"],
        r["config"]["method"]["name"],
    )
    grouped = defaultdict(list)
    for r in records:
        grouped[key_fn(r)].append(r)
    summary = []
    for key, runs in grouped.items():
        acc_final = [r["final"]["accuracy"] for r in runs]
        b90 = [r["final"]["b90"] for r in runs]
        runtime = [r["final"]["runtime_seconds"] for r in runs]
        mem = [r["final"]["peak_mem_mb"] for r in runs]
        summary.append(
            {
                "dataset": key[0],
                "model": key[1],
                "method": key[2],
                "acc_mean": np.mean(acc_final),
                "acc_std": np.std(acc_final),
                "b90_mean": np.mean(b90),
                "b90_std": np.std(b90),
                "runtime_mean": np.mean(runtime),
                "mem_mean": np.mean(mem),
                "n_runs": len(runs),
            }
        )
    return pd.DataFrame(summary)


# ------------------------- Plotting Utilities ----------------------------- #

sns.set(style="whitegrid", font_scale=1.2)

def lineplot_accuracy(records, results_dir: Path):
    """Plot accuracy-vs-batches curves averaged across seeds."""
    # Determine the maximum number of batches across runs for alignment
    max_batches = max(len(r["metrics"]["accuracy_vs_batches"]) for r in records)
    xs = np.arange(1, max_batches + 1)

    def get_curve(r):
        y = r["metrics"]["accuracy_vs_batches"]
        if len(y) < max_batches:
            # pad with last value for shorter streams
            y = y + [y[-1]] * (max_batches - len(y))
        return np.array(y)

    curves_by_method = defaultdict(list)
    for r in records:
        method = r["config"]["method"]["name"]
        curves_by_method[method].append(get_curve(r))

    plt.figure(figsize=(8, 5))
    for method, curves in curves_by_method.items():
        curves = np.stack(curves, axis=0)
        mean = curves.mean(0)
        std = curves.std(0)
        plt.plot(xs, mean, label=method)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.2)
        # annotate final mean value
        plt.text(xs[-1], mean[-1], f"{mean[-1]*100:.1f}%", fontsize=8)

    plt.xlabel("Test batches processed")
    plt.ylabel("Top-1 Accuracy")
    plt.title("Accuracy vs. Batches (mean ± std)")
    plt.legend()
    plt.tight_layout()
    out_path = results_dir / "images" / "accuracy_vs_batches.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return str(out_path.name)


def barplot_metric(df: pd.DataFrame, metric: str, ylabel: str, fname: str, results_dir: Path):
    plt.figure(figsize=(6, 4))
    order = df.sort_values(metric)["method"].tolist() if metric == "b90_mean" else None
    sns.barplot(data=df, x="method", y=metric, order=order, palette="deep", ci=None)
    for ax in plt.gca().containers:
        plt.bar_label(ax, fmt="%.2f")
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.title(f"{ylabel} by Method")
    plt.tight_layout()
    out_path = results_dir / "images" / f"{fname}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return str(out_path.name)


# ----------------------------- Main --------------------------------------- #

def main(results_dir: str):
    results_dir = Path(results_dir)
    records = load_all_results(results_dir)
    if not records:
        raise RuntimeError(f"No results.json files found in {results_dir}")

    df_summary = aggregate(records)
    # Save summary CSV for convenience
    df_summary.to_csv(results_dir / "summary.csv", index=False)

    # Figures
    figure_files = []
    figure_files.append(lineplot_accuracy(records, results_dir))
    figure_files.append(barplot_metric(df_summary, "acc_mean", "Final Accuracy", "final_accuracy", results_dir))
    figure_files.append(barplot_metric(df_summary, "b90_mean", "B90 (batches)", "b90", results_dir))

    # Print JSON summary to STDOUT
    output = {
        "experiment_description": "Comparison of methods across all run variations. Metrics are aggregated over seeds.",
        "summary_table": df_summary.to_dict(orient="records"),
        "figure_files": figure_files,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, required=True)
    args = ap.parse_args()
    main(args.results_dir)