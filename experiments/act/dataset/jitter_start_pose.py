#!/usr/bin/env python3
"""Generate a start-pose-jittered LeRobot v3.0 dataset.

For each episode, sample K_i ~ Uniform[K_MIN, K_MAX] (frames) and drop the first K_i
frames of that episode from both the data parquet and the (concatenated) videos,
re-encoding the videos with libsvtav1.

Run with the trossen_mobile_ai venv:
  /home/siddhant/workspace/pr_test/torqueagi-arm/teleop/trossen_mobile_ai/lerobot_trossen/.venv/bin/python \
      jitter_start_pose.py
"""
from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd

SRC = Path("/home/siddhant/trossen_datasets/local/mobileai-lipbalm-right-only")
DST = Path("/home/siddhant/trossen_datasets/local/mobileai-lipbalm-right-only-jittered")
SEED = 0
K_MIN, K_MAX = 5, 30   # inclusive range, frames
VIDEO_KEYS = ["observation.images.cam_high", "observation.images.cam_right_wrist"]


def sample_drops(rng: np.random.Generator, n_episodes: int) -> np.ndarray:
    return rng.integers(K_MIN, K_MAX + 1, size=n_episodes).astype(np.int64)


def slice_data(df: pd.DataFrame, drops: np.ndarray, fps: int) -> pd.DataFrame:
    """Drop first K_i rows per episode, then reindex frame_index/index/timestamp."""
    out_parts = []
    for ep, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index").reset_index(drop=True)
        k = int(drops[int(ep)])
        g = g.iloc[k:].reset_index(drop=True)
        g["frame_index"] = np.arange(len(g), dtype=np.int64)
        g["timestamp"] = (g["frame_index"].to_numpy().astype(np.float32)) / float(fps)
        out_parts.append(g)
    out = pd.concat(out_parts, ignore_index=True)
    out["index"] = np.arange(len(out), dtype=np.int64)
    return out


def per_feature_stats(arr: np.ndarray) -> dict:
    """Return LeRobot-style per-feature stats for a 1D or 2D array (rows = frames)."""
    a = np.asarray(arr)
    flat_axis = 0
    out = {
        "min": a.min(axis=flat_axis).tolist(),
        "max": a.max(axis=flat_axis).tolist(),
        "mean": a.mean(axis=flat_axis).tolist(),
        "std": a.std(axis=flat_axis).tolist(),
        "count": [int(a.shape[0])],
        "q01": np.quantile(a, 0.01, axis=flat_axis).tolist(),
        "q10": np.quantile(a, 0.10, axis=flat_axis).tolist(),
        "q50": np.quantile(a, 0.50, axis=flat_axis).tolist(),
        "q90": np.quantile(a, 0.90, axis=flat_axis).tolist(),
        "q99": np.quantile(a, 0.99, axis=flat_axis).tolist(),
    }
    return out


def scalar_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr).reshape(-1)
    return {
        "min": [a.min().item()],
        "max": [a.max().item()],
        "mean": [a.mean().item()],
        "std": [a.std().item()],
        "count": [int(a.shape[0])],
        "q01": [np.quantile(a, 0.01).item()],
        "q10": [np.quantile(a, 0.10).item()],
        "q50": [np.quantile(a, 0.50).item()],
        "q90": [np.quantile(a, 0.90).item()],
        "q99": [np.quantile(a, 0.99).item()],
    }


def rebuild_episodes(eps_old: pd.DataFrame, df_new: pd.DataFrame, drops: np.ndarray, fps: int) -> pd.DataFrame:
    eps = eps_old.copy().sort_values("episode_index").reset_index(drop=True)
    new_lengths = []
    new_from = []
    new_to = []
    cursor = 0
    for ep_idx in eps["episode_index"].to_numpy():
        ep_idx = int(ep_idx)
        ep_df = df_new[df_new["episode_index"] == ep_idx]
        L = len(ep_df)
        new_lengths.append(L)
        new_from.append(cursor)
        new_to.append(cursor + L)
        cursor += L
    eps["length"] = np.array(new_lengths, dtype=np.int64)
    eps["dataset_from_index"] = np.array(new_from, dtype=np.int64)
    eps["dataset_to_index"] = np.array(new_to, dtype=np.int64)

    # Recompute video from/to timestamps (videos are concatenated per chunk, so
    # episodes are back-to-back in the new mp4).
    cur_t = 0.0
    new_video_from = []
    new_video_to = []
    for L in new_lengths:
        new_video_from.append(cur_t)
        new_video_to.append(cur_t + L / float(fps))
        cur_t += L / float(fps)
    for vk in VIDEO_KEYS:
        eps[f"videos/{vk}/from_timestamp"] = new_video_from
        eps[f"videos/{vk}/to_timestamp"] = new_video_to
        # chunk_index / file_index remain 0 (single file)

    # Recompute non-image stats per episode (action, state, scalars).
    for ep_idx in eps["episode_index"].to_numpy():
        ep_idx = int(ep_idx)
        ep_df = df_new[df_new["episode_index"] == ep_idx]
        action = np.stack(ep_df["action"].to_numpy())
        state = np.stack(ep_df["observation.state"].to_numpy())
        for prefix, stats in [
            ("stats/action", per_feature_stats(action)),
            ("stats/observation.state", per_feature_stats(state)),
            ("stats/timestamp", scalar_stats(ep_df["timestamp"].to_numpy())),
            ("stats/frame_index", scalar_stats(ep_df["frame_index"].to_numpy())),
            ("stats/episode_index", scalar_stats(ep_df["episode_index"].to_numpy())),
            ("stats/index", scalar_stats(ep_df["index"].to_numpy())),
            ("stats/task_index", scalar_stats(ep_df["task_index"].to_numpy())),
        ]:
            for k, v in stats.items():
                eps.loc[eps["episode_index"] == ep_idx, f"{prefix}/{k}"] = pd.Series(
                    [v], index=eps.index[eps["episode_index"] == ep_idx]
                )
    return eps


def reencode_video(src_mp4: Path, dst_mp4: Path, drops: np.ndarray,
                   ep_from: np.ndarray, ep_to: np.ndarray, fps: int) -> int:
    """Decode src concatenated mp4, drop first K_i frames of each episode,
    re-encode the rest with libsvtav1."""
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    in_container = av.open(str(src_mp4))
    in_stream = in_container.streams.video[0]
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height

    out_container = av.open(str(dst_mp4), mode="w")
    out_stream = out_container.add_stream("libsvtav1", rate=fps)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.time_base = Fraction(1, fps)
    # SVT-AV1 options (preset 8 = fast, crf 30 = good quality at small size)
    out_stream.options = {"preset": "8", "crf": "30"}

    n_episodes = len(ep_from)
    src_idx = 0
    out_idx = 0
    ep_cursor = 0  # current episode pointer

    for frame in in_container.decode(video=0):
        # advance episode pointer if we've passed this episode's source range
        while ep_cursor < n_episodes and src_idx >= int(ep_to[ep_cursor]):
            ep_cursor += 1
        if ep_cursor >= n_episodes:
            break
        in_ep = src_idx - int(ep_from[ep_cursor])
        keep = in_ep >= int(drops[ep_cursor])
        src_idx += 1
        if not keep:
            continue
        # Reset PTS for monotonic output stream
        out_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="yuv420p"), format="yuv420p")
        out_frame.pts = out_idx
        out_frame.time_base = Fraction(1, fps)
        for pkt in out_stream.encode(out_frame):
            out_container.mux(pkt)
        out_idx += 1

    for pkt in out_stream.encode():
        out_container.mux(pkt)
    out_container.close()
    in_container.close()
    return out_idx


def main():
    info = json.loads((SRC / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    eps_old = pd.read_parquet(SRC / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    df = pd.read_parquet(SRC / "data" / "chunk-000" / "file-000.parquet")
    n_episodes = int(info["total_episodes"])

    rng = np.random.default_rng(SEED)
    drops = sample_drops(rng, n_episodes)
    print(f"drops per episode (K_i): min={drops.min()}, max={drops.max()}, mean={drops.mean():.1f}, sum={drops.sum()}")
    print(f"new total_frames = {int(info['total_frames']) - int(drops.sum())}")

    # Source episode boundaries (for video remapping)
    eps_old_sorted = eps_old.sort_values("episode_index").reset_index(drop=True)
    src_from = eps_old_sorted["dataset_from_index"].to_numpy()
    src_to = eps_old_sorted["dataset_to_index"].to_numpy()

    # 1) Slice data parquet
    df_new = slice_data(df, drops, fps)
    print(f"sliced data: {len(df_new)} rows")

    # 2) Rebuild episode meta
    eps_new = rebuild_episodes(eps_old_sorted, df_new, drops, fps)

    # 3) Write outputs
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (DST / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    df_new.to_parquet(DST / "data" / "chunk-000" / "file-000.parquet", index=False)
    eps_new.to_parquet(DST / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    shutil.copy(SRC / "meta" / "tasks.parquet", DST / "meta" / "tasks.parquet")

    # 4) Re-encode videos
    for vk in VIDEO_KEYS:
        src_mp4 = SRC / "videos" / vk / "chunk-000" / "file-000.mp4"
        dst_mp4 = DST / "videos" / vk / "chunk-000" / "file-000.mp4"
        print(f"re-encoding {vk} ...")
        n_out = reencode_video(src_mp4, dst_mp4, drops, src_from, src_to, fps)
        print(f"  wrote {n_out} frames -> {dst_mp4}")
        assert n_out == len(df_new), f"video frame count {n_out} != data rows {len(df_new)}"

    # 5) Update info.json
    info_new = dict(info)
    info_new["total_frames"] = int(len(df_new))
    info_new["splits"] = {"train": f"0:{n_episodes}"}
    # Compute new file sizes
    data_size_mb = (DST / "data" / "chunk-000" / "file-000.parquet").stat().st_size // (1024 * 1024)
    video_size_mb = sum(
        (DST / "videos" / vk / "chunk-000" / "file-000.mp4").stat().st_size for vk in VIDEO_KEYS
    ) // (1024 * 1024)
    info_new["data_files_size_in_mb"] = max(int(data_size_mb), 1)
    info_new["video_files_size_in_mb"] = max(int(video_size_mb), 1)
    (DST / "meta" / "info.json").write_text(json.dumps(info_new, indent=4))

    # 6) Save the drop schedule for reproducibility
    (DST / "meta" / "jitter_drops.json").write_text(json.dumps({
        "seed": SEED,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "drops_per_episode": drops.tolist(),
    }, indent=2))
    print(f"done -> {DST}")


if __name__ == "__main__":
    main()
