#!/usr/bin/env python
"""Open-loop offline eval on held-out episodes for a trained policy.

Loads a `LeRobotDataset` filtered to `--episodes`, runs the policy frame-by-frame
(`policy.select_action`), and compares the predicted action against the
ground-truth dataset action. Reports L1 and MSE per episode and overall.

This measures **action imitation error** — useful as a quick sanity gate, but
does not measure task success (which requires a real-robot rollout via
`lerobot-rollout` or `lerobot-record --policy.path=...`).

The output JSON has the same shape as `lerobot-eval`'s `eval_info.json`
(per-key dicts of numeric metrics + an "overall" key) so it can be plotted with
`experiments/act/viz/plot_eval.py`.

Usage:
    uv run python experiments/act/eval/offline_eval.py \\
        --policy-path outputs/train/act_lipbalm_right_baseline-h/checkpoints/last/pretrained_model \\
        --dataset-repo-id mobileai/lipbalm_right_only \\
        --dataset-root /home/tagi-runner1/train_test/mobileai-lipbalm-right-only \\
        --episodes 54 55 56 57 58 59 \\
        --out outputs/eval/offline_lipbalm_baseline-h.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_PREFIX


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", type=Path, required=True,
                    help="Dir containing config.json + model.safetensors (e.g. <output_dir>/checkpoints/last/pretrained_model).")
    ap.add_argument("--dataset-repo-id", required=True, help="HF Hub dataset id used for training.")
    ap.add_argument("--dataset-root", type=Path, default=None, help="Optional local dataset root.")
    ap.add_argument("--episodes", type=int, nargs="+", required=True, help="Held-out episode indices.")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: policy config).")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader workers for video decode prefetch.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    return ap.parse_args()


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = str(args.policy_path)
    if args.device:
        torch.device(args.device)  # validate
        cfg.device = args.device
    device = torch.device(cfg.device)

    logging.info("Loading dataset %s (episodes=%s)", args.dataset_repo_id, args.episodes)
    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=list(args.episodes),
    )
    # batch_size=1 + shuffle=False are mandatory: ACT's select_action maintains
    # an internal action queue that must see frames in order. num_workers
    # overlaps video decode with GPU inference.
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    logging.info("Loading policy from %s", args.policy_path)
    policy = make_policy(cfg=cfg, ds_meta=ds.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    per_episode: dict[int, dict[str, float]] = {}
    cur_ep: int | None = None
    # Accumulate on-device to avoid a per-frame device->host sync from .item().
    sum_l1 = torch.zeros((), device=device)
    sum_sq = torch.zeros((), device=device)
    n_frames = 0

    def flush_ep() -> None:
        nonlocal sum_l1, sum_sq, n_frames, cur_ep
        if cur_ep is None or n_frames == 0:
            return
        per_episode[cur_ep] = {
            "l1": (sum_l1 / n_frames).item(),
            "mse": (sum_sq / n_frames).item(),
            "n_frames": float(n_frames),
        }
        sum_l1 = torch.zeros((), device=device)
        sum_sq = torch.zeros((), device=device)
        n_frames = 0

    for batch in loader:
        ep = int(batch["episode_index"].item())
        if ep != cur_ep:
            assert cur_ep is None or ep > cur_ep, "dataset frames must be ordered by episode"
            flush_ep()
            cur_ep = ep
            policy.reset()
            logging.info("episode %d", ep)

        gt_action = batch[ACTION].to(device, non_blocking=True)
        # batch already has a leading dim from DataLoader; preprocessor moves
        # tensors to device via the configured device_processor step.
        obs = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX) or k == "task"}
        obs = preprocessor(obs)
        pred = postprocessor(policy.select_action(obs))

        diff = pred - gt_action
        sum_l1 = sum_l1 + diff.abs().mean()
        sum_sq = sum_sq + diff.pow(2).mean()
        n_frames += 1

    flush_ep()

    overall_n = sum(int(v["n_frames"]) for v in per_episode.values())
    overall_l1 = sum(v["l1"] * v["n_frames"] for v in per_episode.values()) / max(overall_n, 1)
    overall_mse = sum(v["mse"] * v["n_frames"] for v in per_episode.values()) / max(overall_n, 1)
    out: dict[str, dict[str, float]] = {f"ep_{k}": v for k, v in sorted(per_episode.items())}
    out["overall"] = {"l1": overall_l1, "mse": overall_mse, "n_frames": float(overall_n)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fp:
        json.dump(out, fp, indent=2)
    logging.info("wrote %s", args.out)
    logging.info("overall: %s", json.dumps(out["overall"], indent=2))


if __name__ == "__main__":
    main()
