#!/usr/bin/env python
"""Merge multiple local LeRobotDataset directories into one.

Thin wrapper around `lerobot.datasets.dataset_tools.merge_datasets` that takes
local paths instead of Hub repo ids — repo_ids are auto-derived from each
directory's basename so you don't have to invent them.

Usage:
    uv run python experiments/act/data/merge_datasets.py \\
        /data/lipbalm_run1 /data/lipbalm_run2 \\
        --out /data/lipbalm_merged

    # Override output repo_id (default: local/<out-basename>):
    uv run python experiments/act/data/merge_datasets.py \\
        /data/run1 /data/run2 \\
        --out /data/merged \\
        --repo-id mobileai/lipbalm_merged
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="Local LeRobotDataset directories to merge.")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for the merged dataset.")
    ap.add_argument(
        "--repo-id",
        default=None,
        help="Output dataset repo_id (default: local/<out-basename>).",
    )
    ap.add_argument(
        "--repo-id-prefix",
        default="local",
        help="Prefix used to synthesize input repo_ids from directory basenames (default: 'local').",
    )
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if len(args.paths) < 2:
        raise SystemExit("Need at least 2 dataset paths to merge.")

    datasets = []
    for p in args.paths:
        info = p / INFO_PATH
        if not info.is_file():
            raise SystemExit(f"{p} does not look like a LeRobotDataset (missing {INFO_PATH}).")
        repo_id = f"{args.repo_id_prefix}/{p.name}"
        logging.info("loading %s as %s", p, repo_id)
        datasets.append(LeRobotDataset(repo_id=repo_id, root=p))

    out_repo_id = args.repo_id or f"local/{args.out.name}"
    logging.info("merging %d datasets -> %s (%s)", len(datasets), args.out, out_repo_id)
    merged = merge_datasets(datasets, output_repo_id=out_repo_id, output_dir=args.out)

    logging.info(
        "merged: episodes=%d frames=%d -> %s",
        merged.meta.total_episodes,
        merged.meta.total_frames,
        args.out,
    )


if __name__ == "__main__":
    main()
