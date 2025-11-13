"""
main.py – Orchestrator. Reads a YAML config file (smoke_test.yaml or full_experiment.yaml),
launches src.train as a subprocess for each experiment sequentially, and finally triggers
src.evaluate.py to generate aggregated figures.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT.parent / "config"

# ----------------------------- Tee Utility -------------------------------- #

def tee_subprocess(cmd: List[str], stdout_path: Path, stderr_path: Path):
    """Run cmd and tee its stdout/stderr to files while forwarding to console."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stdout_path, "w", encoding="utf-8") as so, open(
        stderr_path, "w", encoding="utf-8"
    ) as se:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            out_line = proc.stdout.readline()
            err_line = proc.stderr.readline()
            if out_line:
                sys.stdout.write(out_line)
                so.write(out_line)
            if err_line:
                sys.stderr.write(err_line)
                se.write(err_line)
            if out_line == "" and err_line == "" and proc.poll() is not None:
                break
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Subprocess {' '.join(cmd)} failed with code {proc.returncode}")


# --------------------------- Main Workflow -------------------------------- #

def run_all(cfg_path: Path, results_dir: Path):
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg_all = yaml.safe_load(fh)

    # Write a *copy* of full config in the results dir for reproducibility
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "config_used.yaml", "w", encoding="utf-8") as fw:
        yaml.safe_dump(cfg_all, fw)

    # To avoid huge argument strings, we dump the entire config to a temp JSON once
    tmp_cfg_json = results_dir / "_tmp_config.json"
    json.dump(cfg_all, open(tmp_cfg_json, "w", encoding="utf-8"))

    for exp in cfg_all["experiments"]:
        run_id = exp["run_id"]
        stdout_path = results_dir / run_id / "stdout.log"
        stderr_path = results_dir / run_id / "stderr.log"
        cmd = [
            sys.executable,
            "-m",
            "src.train",
            "--config-path",
            str(tmp_cfg_json),
            "--run-id",
            run_id,
            "--results-dir",
            str(results_dir),
        ]
        print(f"\n=== Running experiment: {run_id} ===")
        tee_subprocess(cmd, stdout_path, stderr_path)

    # After all runs finished → evaluate
    print("\n=== All runs finished. Evaluating… ===")
    eval_cmd = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--results-dir",
        str(results_dir),
    ]
    subprocess.run(eval_cmd, check=True)


# --------------------------------- CLI ------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="Run smoke_test.yaml")
    g.add_argument("--full-experiment", action="store_true", help="Run full_experiment.yaml")
    p.add_argument("--results-dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    cfg_file = CONFIG_DIR / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    results_dir = Path(args.results_dir)
    run_all(cfg_file, results_dir)


if __name__ == "__main__":
    main()