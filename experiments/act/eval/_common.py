"""Shared inference scaffolding for offline ACT eval.

Both `offline_eval.py` (scalar L1/MSE) and `viz/plot_predictions.py` (per-joint
trajectory plots) need the same setup: load policy + processors from a
checkpoint, build a `LeRobotDataset` filtered to held-out episodes, wrap it in
a single-frame DataLoader. This module is the single source of truth so the
two callers can't drift.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


def build_inference_stack(
    policy_path: Path,
    dataset_repo_id: str,
    dataset_root: Path | None,
    episodes: list[int],
    *,
    device: str | None = None,
    num_workers: int = 2,
) -> tuple[DataLoader, Any, Any, Any, LeRobotDataset, torch.device]:
    """Returns (loader, policy, preprocessor, postprocessor, dataset, device).

    The loader yields one frame per batch in dataset order — ACT's
    `select_action` maintains an internal action queue that requires sequential
    frames, so `batch_size=1, shuffle=False` are mandatory.
    """
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = str(policy_path)
    if device:
        torch.device(device)  # fail-fast on garbage strings
        cfg.device = device
    device_t = torch.device(cfg.device)

    logging.info("Loading dataset %s (episodes=%s)", dataset_repo_id, episodes)
    ds = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root)
    # `LeRobotDataset(episodes=...)` in lerobot 0.4.0 records num_episodes
    # but doesn't slice __getitem__; index via meta ranges instead.
    indices: list[int] = []
    for ep in episodes:
        meta = ds.meta.episodes[ep]
        indices.extend(range(meta["dataset_from_index"], meta["dataset_to_index"]))
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_t.type == "cuda",
    )

    logging.info("Loading policy from %s", policy_path)
    policy = make_policy(cfg=cfg, ds_meta=ds.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        preprocessor_overrides={"device_processor": {"device": str(device_t)}},
    )
    return loader, policy, preprocessor, postprocessor, ds, device_t
