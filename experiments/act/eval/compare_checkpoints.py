#!/usr/bin/env python
"""Run offline action-error eval on multiple checkpoints over the same held-out
episodes and aggregate into a single JSON for plotting.

Shells out to `experiments/act/eval/offline_eval.py` per checkpoint (one
subprocess each, decoupled from the eval implementation), then merges the
per-checkpoint `overall` metrics into one JSON keyed by checkpoint label.
The output JSON is consumable by `experiments/act/viz/plot_eval.py` directly.

Usage:
    uv run python experiments/act/eval/compare_checkpoints.py \\
        --policy-paths \\
            outputs/train/act_marker_pick/checkpoints/last/pretrained_model \\
            outputs/train/act_marker_pick_te/checkpoints/last/pretrained_model \\
        --dataset-repo-id mobileai/marker_pick_right_only \\
        --dataset-root /home/siddhant/trossen_datasets/local/marker_pick-right-only \\
        --episodes 66 67 68 69 70 71 72 73 \\
        --out outputs/eval/compare_te.json

Then plot:
    uv run python experiments/act/viz/plot_eval.py --eval-info outputs/eval/compare_te.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

OFFLINE_EVAL = Path(__file__).resolve().parent / "offline_eval.py"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-paths", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Names for each checkpoint in the output (default: derived from policy paths).")
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    return ap.parse_args()


def derive_label(p: Path) -> str:
    # <run_dir>/checkpoints/<step>/pretrained_model -> <run_dir>_<step>
    parts = p.resolve().parts
    if len(parts) >= 4 and parts[-1] == "pretrained_model" and parts[-3] == "checkpoints":
        return f"{parts[-4]}_{parts[-2]}"
    return p.name


def run_one(policy_path: Path, args: argparse.Namespace, out_json: Path) -> dict:
    cmd = [
        sys.executable, str(OFFLINE_EVAL),
        "--policy-path", str(policy_path),
        "--dataset-repo-id", args.dataset_repo_id,
        "--episodes", *(str(e) for e in args.episodes),
        "--num-workers", str(args.num_workers),
        "--out", str(out_json),
    ]
    if args.dataset_root is not None:
        cmd += ["--dataset-root", str(args.dataset_root)]
    if args.device is not None:
        cmd += ["--device", args.device]
    logging.info("running offline_eval for %s", policy_path)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"offline_eval failed for {policy_path} (exit {proc.returncode})")
    with out_json.open() as fp:
        return json.load(fp)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.labels and len(args.labels) != len(args.policy_paths):
        raise SystemExit("--labels must have the same length as --policy-paths")
    labels = args.labels or [derive_label(p) for p in args.policy_paths]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate labels: {labels}")

    aggregated: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for label, pp in zip(labels, args.policy_paths, strict=True):
            info = run_one(pp, args, td_path / f"{label}.json")
            aggregated[label] = info["overall"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fp:
        json.dump(aggregated, fp, indent=2)
    logging.info("wrote %s (%d checkpoints)", args.out, len(aggregated))


if __name__ == "__main__":
    main()
