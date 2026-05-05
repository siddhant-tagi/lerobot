"""Merge mobileai-lipbalm{,1,2} and strip left arm + left wrist camera.

Targets lerobot 0.4.0 API:
  - dataset[i] returns CHW float32 [0,1] image tensors
  - add_frame validates keys against features and pops 'task'/'timestamp'
  - meta.episodes[i] gives dataset_from_index/dataset_to_index per episode
"""

import copy
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_SRC_ROOTS = [
    "/home/siddhant/trossen_datasets/local/mobileai-lipbalm",
    "/home/siddhant/trossen_datasets/local/mobileai-lipbalm1",
    "/home/siddhant/trossen_datasets/local/mobileai-lipbalm2",
]
SRC_ROOTS = [Path(p) for p in os.environ.get("ACT_SRC_ROOTS", ":".join(DEFAULT_SRC_ROOTS)).split(":")]
DST_REPO_ID = "mobileai/lipbalm_right_only"
DST_ROOT = Path(
    os.environ.get(
        "ACT_DATA_ROOT",
        "/home/siddhant/trossen_datasets/local/mobileai-lipbalm-right-only",
    )
)

DROP_CAMERA = "observation.images.cam_left_wrist"
RIGHT_SLICE = slice(7, 14)  # right_joint_0..5 + right_left_carriage_joint
DROP_FRAME_KEYS = ("index", "episode_index", "frame_index", "task_index", "timestamp")
IMAGE_KEYS = ("observation.images.cam_high", "observation.images.cam_right_wrist")

# Bookkeeping keys present on dataset[i] that aren't features.
template = LeRobotDataset(repo_id="tmp/template", root=SRC_ROOTS[0], download_videos=False)

src_action_names = template.features["action"]["names"]
right_names = src_action_names[RIGHT_SLICE]
assert all(n.startswith("right_") for n in right_names), f"unexpected names: {right_names}"
assert len(right_names) == 7

new_features = copy.deepcopy(template.features)
new_features.pop(DROP_CAMERA)
for key in ("action", "observation.state"):
    new_features[key]["names"] = list(right_names)
    new_features[key]["shape"] = (7,)

# Schema sanity-check across siblings.
for root in SRC_ROOTS[1:]:
    other = LeRobotDataset(repo_id="tmp/check", root=root, download_videos=False)
    assert other.fps == template.fps
    assert other.features.keys() == template.features.keys()
    assert other.meta.robot_type == template.meta.robot_type

merged = LeRobotDataset.create(
    repo_id=DST_REPO_ID,
    root=DST_ROOT,
    fps=template.fps,
    features=new_features,
    robot_type=template.meta.robot_type,
    use_videos=True,
    image_writer_threads=4,
)

total_eps = 0
total_frames = 0

for root in SRC_ROOTS:
    src = LeRobotDataset(repo_id="tmp/src", root=root, download_videos=False)
    for ep_idx in range(src.num_episodes):
        ep = src.meta.episodes[ep_idx]
        lo, hi = ep["dataset_from_index"], ep["dataset_to_index"]
        task_str = ep["tasks"][0]  # one task per episode in this dataset
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
        total_frames += (hi - lo)
    print(f"merged {root.name}: {src.num_episodes} episodes")

print(f"\nDone. {total_eps} episodes / {total_frames} frames -> {DST_ROOT}")
