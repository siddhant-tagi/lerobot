"""Merge bimanual LeRobotDatasets and strip the left arm + left wrist camera.

Produces a right-only dataset suitable for training a single-arm policy on data
recorded with a bimanual rig (`bi_widowxai_follower_robot`).

Usage:
    uv run python experiments/act/dataset/merge_and_strip_left.py \\
        --src-roots /data/run1 /data/run2 \\
        --dst-root /data/run_right_only \\
        --dst-repo-id mobileai/run_right_only

Targets lerobot 0.4.0 API:
  - dataset[i] returns CHW float32 [0,1] image tensors
  - add_frame validates keys against features and pops 'task'/'timestamp'
  - meta.episodes[i] gives dataset_from_index/dataset_to_index per episode
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DROP_CAMERA = "observation.images.cam_left_wrist"
RIGHT_SLICE = slice(7, 14)  # right_joint_0..5 + right_left_carriage_joint
DROP_FRAME_KEYS = ("index", "episode_index", "frame_index", "task_index", "timestamp")
IMAGE_KEYS = ("observation.images.cam_high", "observation.images.cam_right_wrist")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-roots", type=Path, nargs="+", required=True)
    ap.add_argument("--dst-root", type=Path, required=True)
    ap.add_argument("--dst-repo-id", required=True)
    ap.add_argument("--image-writer-threads", type=int, default=4)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.dst_root.exists():
        raise SystemExit(f"{args.dst_root} already exists; pick a fresh path or remove it first.")

    srcs = [LeRobotDataset(repo_id="tmp/src", root=root, download_videos=False) for root in args.src_roots]
    template = srcs[0]
    right_names = template.features["action"]["names"][RIGHT_SLICE]
    assert all(n.startswith("right_") for n in right_names), f"unexpected names: {right_names}"
    assert len(right_names) == 7

    new_features = copy.deepcopy(template.features)
    new_features.pop(DROP_CAMERA)
    for key in ("action", "observation.state"):
        new_features[key]["names"] = list(right_names)
        new_features[key]["shape"] = (7,)

    for src in srcs[1:]:
        assert src.fps == template.fps, f"fps mismatch at {src.root}"
        assert src.features.keys() == template.features.keys(), f"feature-key mismatch at {src.root}"
        assert src.meta.robot_type == template.meta.robot_type, f"robot_type mismatch at {src.root}"

    merged = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        root=args.dst_root,
        fps=template.fps,
        features=new_features,
        robot_type=template.meta.robot_type,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
    )

    total_eps = 0
    total_frames = 0
    for src in srcs:
        for ep_idx in range(src.num_episodes):
            ep = src.meta.episodes[ep_idx]
            lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
            task_str = ep["tasks"][0]
            for i in range(lo, hi):
                frame = src[i]
                for k in DROP_FRAME_KEYS:
                    frame.pop(k, None)
                frame.pop(DROP_CAMERA, None)
                frame["action"] = frame["action"][RIGHT_SLICE]
                frame["observation.state"] = frame["observation.state"][RIGHT_SLICE]
                for img_key in IMAGE_KEYS:
                    # dataset[i] returns CHW float32; validator/writer want HWC.
                    frame[img_key] = frame[img_key].permute(1, 2, 0).contiguous()
                frame["task"] = task_str
                merged.add_frame(frame)
            merged.save_episode()
            total_eps += 1
            total_frames += hi - lo
        logging.info("merged %s: %d episodes", src.root.name, src.num_episodes)

    logging.info("done: %d episodes / %d frames -> %s", total_eps, total_frames, args.dst_root)


if __name__ == "__main__":
    main()
