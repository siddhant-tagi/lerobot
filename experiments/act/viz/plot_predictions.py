#!/usr/bin/env python
"""Plot predicted vs ground-truth action trajectories for held-out episodes.

For each `--episodes` index, runs the policy frame-by-frame on the dataset (same
inference loop as `offline_eval.py`) and renders a per-joint subplot of
predicted vs ground-truth action over time. Also writes a `.npz` per episode
with the raw arrays so downstream analysis doesn't need to re-run inference.

Use this when scalar L1/MSE from `offline_eval.py` says "policy is bad" but
doesn't say *how* — pred-vs-GT plots immediately surface lag, mode collapse,
and joint-specific failure modes.

Usage:
    uv run python experiments/act/viz/plot_predictions.py \\
        --policy-path outputs/train/act_marker_pick_te/checkpoints/last/pretrained_model \\
        --dataset-repo-id mobileai/marker_pick_right_only \\
        --dataset-root /home/siddhant/trossen_datasets/local/marker_pick-right-only \\
        --episodes 66 67 68 \\
        --out-dir outputs/eval/preds_marker_pick_te
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.utils.constants import ACTION, OBS_PREFIX

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from _common import build_inference_stack  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", type=Path, required=True)
    ap.add_argument("--dataset-repo-id", required=True)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, required=True)
    return ap.parse_args()


def render_episode(ep: int, gt: np.ndarray, pred: np.ndarray, names: list[str], out_png: Path) -> None:
    n_joints = gt.shape[1]
    n_cols = 2
    n_rows = (n_joints + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 2.4 * n_rows), squeeze=False)
    t = np.arange(gt.shape[0])
    for j in range(n_joints):
        ax = axes[j // n_cols, j % n_cols]
        ax.plot(t, gt[:, j], label="gt", linewidth=1.2)
        ax.plot(t, pred[:, j], label="pred", linewidth=1.0, linestyle="--")
        ax.set_title(names[j] if j < len(names) else f"action[{j}]", fontsize=9)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=8)
    for j in range(n_joints, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")
    fig.suptitle(f"episode {ep}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loader, policy, preprocessor, postprocessor, ds, _device = build_inference_stack(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        episodes=list(args.episodes),
        device=args.device,
        num_workers=args.num_workers,
    )
    action_names = ds.meta.features.get(ACTION, {}).get("names") or []

    cur_ep: int | None = None
    gt_buf: list[np.ndarray] = []
    pred_buf: list[np.ndarray] = []

    def flush() -> None:
        if cur_ep is None or not gt_buf:
            return
        gt = np.stack(gt_buf)
        pred = np.stack(pred_buf)
        ep_dir = args.out_dir / f"ep_{cur_ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        np.savez(ep_dir / "predictions.npz", gt=gt, pred=pred, frame_index=np.arange(len(gt)))
        render_episode(cur_ep, gt, pred, action_names, ep_dir / "predictions.png")
        logging.info("ep %d: saved %d frames -> %s", cur_ep, len(gt), ep_dir)

    for batch in loader:
        ep = int(batch["episode_index"].item())
        if ep != cur_ep:
            assert cur_ep is None or ep > cur_ep, "dataset frames must be ordered by episode"
            flush()
            cur_ep = ep
            gt_buf.clear()
            pred_buf.clear()
            policy.reset()

        gt_action = batch[ACTION]
        obs = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX) or k == "task"}
        pred = postprocessor(policy.select_action(preprocessor(obs)))
        gt_buf.append(gt_action.squeeze(0).cpu().numpy())
        pred_buf.append(pred.squeeze(0).cpu().numpy())

    flush()


if __name__ == "__main__":
    main()
