#!/usr/bin/env python3
"""Explore a LeRobot v3.0 dataset: schema, episode stats, action/state stats, sample frames.

Run with the trossen_mobile_ai venv:
  /home/siddhant/workspace/pr_test/torqueagi-arm/teleop/trossen_mobile_ai/lerobot_trossen/.venv/bin/python \
      explore_dataset.py --root /home/siddhant/trossen_datasets/local/mobileai-lipbalm-right-only
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def load_meta(root: Path):
    info = json.loads((root / "meta" / "info.json").read_text())
    eps = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    return info, eps, tasks


def print_schema(info):
    print("=" * 70)
    print("SCHEMA")
    print("=" * 70)
    print(f"robot_type     : {info['robot_type']}")
    print(f"codebase       : {info['codebase_version']}")
    print(f"fps            : {info['fps']}")
    print(f"total_episodes : {info['total_episodes']}")
    print(f"total_frames   : {info['total_frames']}")
    print(f"total_tasks    : {info['total_tasks']}")
    print()
    print(f"{'feature':40s} {'dtype':8s} shape")
    for name, f in info["features"].items():
        print(f"{name:40s} {f['dtype']:8s} {f.get('shape')}")


def episode_stats(eps, info):
    print("\n" + "=" * 70)
    print("EPISODE STATS")
    print("=" * 70)
    if "length" in eps.columns:
        lengths = eps["length"].to_numpy()
    else:
        lengths = (eps["dataset_to_index"] - eps["dataset_from_index"]).to_numpy()
    print(f"episodes        : {len(eps)}")
    print(f"length min/mean/max : {lengths.min()} / {lengths.mean():.1f} / {lengths.max()}")
    print(f"sum(lengths)    : {int(lengths.sum())}  (info.total_frames={info['total_frames']})")
    assert int(lengths.sum()) == info["total_frames"], "frame count mismatch"
    return lengths


def action_state_stats(root, info):
    print("\n" + "=" * 70)
    print("ACTION / STATE STATS")
    print("=" * 70)
    df = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    print(f"rows in parquet : {len(df)}")
    for key in ("action", "observation.state"):
        names = info["features"][key]["names"]
        arr = np.stack(df[key].to_numpy())
        print(f"\n{key}  shape={arr.shape}")
        print(f"{'dim':30s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s}  flag")
        for i, n in enumerate(names):
            col = arr[:, i]
            flag = "DEAD" if col.std() < 1e-4 else ""
            print(f"{n:30s} {col.min():10.4f} {col.max():10.4f} {col.mean():10.4f} {col.std():10.4f}  {flag}")


def decode_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    with av.open(str(video_path)) as c:
        stream = c.streams.video[0]
        for i, frame in enumerate(c.decode(stream)):
            if i == frame_idx:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"frame {frame_idx} not in {video_path}")


def make_grid(images, labels, cols):
    h, w = images[0].shape[:2]
    rows = len(images) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    pil = Image.fromarray(grid)
    # Stamp labels with a default font.
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil)
    for k, label in enumerate(labels):
        r, c = divmod(k, cols)
        draw.rectangle((c * w, r * h, c * w + 220, r * h + 22), fill=(0, 0, 0))
        draw.text((c * w + 4, r * h + 4), label, fill=(255, 255, 255))
    return pil


def sample_frames(root, info, eps, lengths, out: Path):
    print("\n" + "=" * 70)
    print("VISUAL SAMPLES")
    print("=" * 70)
    cams = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    n = len(eps)
    pick_eps = sorted({0, n // 2, n - 1})
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    for cam in cams:
        vp = root / "videos" / cam / "chunk-000" / "file-000.mp4"
        imgs, labels = [], []
        for ep in pick_eps:
            ep_len = int(lengths[ep])
            ep_start = int(starts[ep])
            for pos, tag in [(ep_start, "start"), (ep_start + ep_len // 2, "mid"), (ep_start + ep_len - 1, "end")]:
                imgs.append(decode_frame(vp, pos))
                labels.append(f"ep{ep} {tag} f{pos}")
        grid = make_grid(imgs, labels, cols=3)
        out_path = out / f"{cam.replace('.', '_')}_grid.png"
        grid.save(out_path)
        print(f"wrote {out_path}  ({grid.size[0]}x{grid.size[1]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/home/siddhant/trossen_datasets/local/mobileai-lipbalm-right-only"))
    ap.add_argument("--out", type=Path, default=Path("exploration_out"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    info, eps, _tasks = load_meta(args.root)
    print_schema(info)
    lengths = episode_stats(eps, info)
    action_state_stats(args.root, info)
    sample_frames(args.root, info, eps, lengths, args.out)
    print("\ndone.")


if __name__ == "__main__":
    main()
