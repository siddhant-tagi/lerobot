#!/usr/bin/env python
"""Compile a CSV of training run hyperparameters + held-out eval metrics.

Walks each run directory under `--root` matching `--glob`, scrapes
`train_config.json` (saved by `lerobot-train`) for hyperparams, and looks for a
sibling `offline_eval.json` (produced by `experiments/act/eval/offline_eval.py`)
for L1/MSE numbers. Writes one CSV row per run.

Convention assumed:
    <root>/<run_dir>/train_config.json                             # if present at run root
    <root>/<run_dir>/checkpoints/last/pretrained_model/train_config.json  # standard layout
    <root>/<run_dir>/offline_eval.json                             # offline_eval output

Usage:
    uv run python experiments/act/eval/compile_results.py --root outputs/train
    uv run python experiments/act/eval/compile_results.py --root /data/logs \\
        --glob 'act_marker_pick*' --out results.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

FIELDS = [
    "run",
    "policy_type",
    "dataset_repo_id",
    "steps",
    "batch_size",
    "seed",
    "chunk_size",
    "n_action_steps",
    "temporal_ensemble_coeff",
    "device",
    "l1",
    "mse",
    "n_eval_frames",
    "eval_episodes",
]


def _get(d: dict[str, Any], *path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def scrape_run(run_dir: Path) -> dict[str, Any] | None:
    candidates = [
        run_dir / "train_config.json",
        run_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json",
    ]
    cfg_path = next((p for p in candidates if p.is_file()), None)
    if cfg_path is None:
        return None
    with cfg_path.open() as fp:
        cfg = json.load(fp)

    row: dict[str, Any] = {
        "run": run_dir.name,
        "policy_type": _get(cfg, "policy", "type"),
        "dataset_repo_id": _get(cfg, "dataset", "repo_id"),
        "steps": _get(cfg, "steps"),
        "batch_size": _get(cfg, "batch_size"),
        "seed": _get(cfg, "seed"),
        "chunk_size": _get(cfg, "policy", "chunk_size"),
        "n_action_steps": _get(cfg, "policy", "n_action_steps"),
        "temporal_ensemble_coeff": _get(cfg, "policy", "temporal_ensemble_coeff"),
        "device": _get(cfg, "policy", "device"),
        "l1": None,
        "mse": None,
        "n_eval_frames": None,
        "eval_episodes": None,
    }

    eval_path = run_dir / "offline_eval.json"
    if eval_path.is_file():
        with eval_path.open() as fp:
            ev = json.load(fp)
        overall = ev.get("overall", {})
        row["l1"] = overall.get("l1")
        row["mse"] = overall.get("mse")
        row["n_eval_frames"] = overall.get("n_frames")
        row["eval_episodes"] = ",".join(k.removeprefix("ep_") for k in ev if k.startswith("ep_"))

    return row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("outputs/train"))
    ap.add_argument("--glob", default="act_*")
    ap.add_argument("--out", type=Path, default=None, help="Output CSV (default: stdout).")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"{args.root} is not a directory")

    rows = []
    for run_dir in sorted(args.root.glob(args.glob)):
        if not run_dir.is_dir():
            continue
        row = scrape_run(run_dir)
        if row is None:
            logging.warning("skipping %s (no train_config.json)", run_dir.name)
            continue
        rows.append(row)

    if not rows:
        raise SystemExit(f"no matching runs in {args.root} (glob: {args.glob})")

    sink = args.out.open("w", newline="") if args.out else contextlib.nullcontext(sys.stdout)
    with sink as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if args.out:
        logging.info("wrote %d rows -> %s", len(rows), args.out)


if __name__ == "__main__":
    main()
