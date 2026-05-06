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

try:
    from lerobot.datasets.compute_stats import (
        aggregate_stats,
        auto_downsample_height_width,
        get_feature_stats,
        sample_indices,
    )
except ImportError as e:  # pragma: no cover - the runtime venv is expected to have lerobot
    raise SystemExit(f"lerobot.datasets.compute_stats unavailable: {e}") from e

SRC = Path("/home/siddhant/trossen_datasets/local/mobileai-lipbalm-right-only")
DST = Path("/home/siddhant/trossen_datasets/local/mobileai-lipbalm-right-only-jittered")
SEED = 0
K_MIN, K_MAX = 5, 30   # inclusive range, frames
VIDEO_KEYS = ["observation.images.cam_high", "observation.images.cam_right_wrist"]
SCALAR_COLS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


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


def compute_image_stats_per_episode(
    src_mp4: Path,
    drops: np.ndarray,
    src_from: np.ndarray,
    src_to: np.ndarray,
) -> dict[int, dict[str, np.ndarray]]:
    """Decode src video once; per-channel stats over sub-sampled kept frames per episode.

    Returns {ep_idx: stats_dict} with each stat shaped (3,1,1) and pixels normalized to [0,1],
    matching what compute_episode_stats produces for image/video features.
    """
    n_eps = len(src_from)
    src_to_ep: dict[int, int] = {}
    for ep in range(n_eps):
        L_new = int(src_to[ep] - src_from[ep] - drops[ep])
        if L_new <= 0:
            continue
        for i in sample_indices(L_new):
            src_to_ep[int(src_from[ep] + drops[ep] + i)] = ep

    frames_by_ep: list[list[np.ndarray]] = [[] for _ in range(n_eps)]
    in_container = av.open(str(src_mp4))
    for src_idx, frame in enumerate(in_container.decode(video=0)):
        ep = src_to_ep.get(src_idx)
        if ep is not None:
            arr = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)  # C H W uint8
            frames_by_ep[ep].append(auto_downsample_height_width(arr))
    in_container.close()

    stats_by_ep: dict[int, dict[str, np.ndarray]] = {}
    for ep, frames in enumerate(frames_by_ep):
        if not frames:
            continue
        arr = np.stack(frames, axis=0)  # N C H W uint8
        s = get_feature_stats(arr, axis=(0, 2, 3), keepdims=True)
        stats_by_ep[ep] = {
            k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in s.items()
        }
    return stats_by_ep


def compute_per_episode_stats(
    df_new: pd.DataFrame,
    eps_old_sorted: pd.DataFrame,
    drops: np.ndarray,
    src_video_paths: dict[str, Path],
) -> list[dict[str, dict[str, np.ndarray]]]:
    """Per-episode stats for vectors/scalars (from df_new) + images (from src videos)."""
    n_eps = len(eps_old_sorted)
    per_ep: list[dict[str, dict[str, np.ndarray]]] = [dict() for _ in range(n_eps)]

    for ep_idx in range(n_eps):
        ep_df = df_new[df_new["episode_index"] == ep_idx]
        action = np.stack(ep_df["action"].to_numpy()).astype(np.float32)
        state = np.stack(ep_df["observation.state"].to_numpy()).astype(np.float32)
        per_ep[ep_idx]["action"] = get_feature_stats(action, axis=0, keepdims=False)
        per_ep[ep_idx]["observation.state"] = get_feature_stats(state, axis=0, keepdims=False)
        for col in SCALAR_COLS:
            arr = np.asarray(ep_df[col].to_numpy(), dtype=np.float64).reshape(-1)
            per_ep[ep_idx][col] = get_feature_stats(arr, axis=0, keepdims=False)

    src_from = eps_old_sorted["dataset_from_index"].to_numpy()
    src_to = eps_old_sorted["dataset_to_index"].to_numpy()
    for vk, src_mp4 in src_video_paths.items():
        print(f"computing image stats from {vk} ...")
        img_stats = compute_image_stats_per_episode(src_mp4, drops, src_from, src_to)
        for ep_idx, s in img_stats.items():
            per_ep[ep_idx][vk] = s
    return per_ep


def rebuild_episodes(
    eps_old: pd.DataFrame,
    df_new: pd.DataFrame,
    drops: np.ndarray,
    fps: int,
    per_ep_stats: list[dict[str, dict[str, np.ndarray]]],
) -> pd.DataFrame:
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

    # Write all stats columns (action/state/scalars + observation.images.*) from per_ep_stats.
    n_eps = len(eps)
    stat_cols: dict[str, list] = {}
    for ep_idx in range(n_eps):
        for feature_key, stats in per_ep_stats[ep_idx].items():
            for stat_key, val in stats.items():
                col = f"stats/{feature_key}/{stat_key}"
                stat_cols.setdefault(col, [None] * n_eps)[ep_idx] = np.asarray(val).tolist()
    for col, vals in stat_cols.items():
        eps[col] = vals
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


def write_stats_json(
    per_ep_stats: list[dict[str, dict[str, np.ndarray]]],
    dst_path: Path,
) -> None:
    """Aggregate per-episode stats into the dataset-level meta/stats.json."""
    aggregated = aggregate_stats(per_ep_stats)
    out = {
        feat: {k: np.asarray(v).tolist() for k, v in stats.items()}
        for feat, stats in aggregated.items()
    }
    dst_path.write_text(json.dumps(out, indent=4, sort_keys=True))


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

    # 2) Per-episode stats (vectors/scalars from df_new + images decoded from src videos)
    src_video_paths = {vk: SRC / "videos" / vk / "chunk-000" / "file-000.mp4" for vk in VIDEO_KEYS}
    per_ep_stats = compute_per_episode_stats(df_new, eps_old_sorted, drops, src_video_paths)

    # 3) Rebuild episode meta (incl. all stats columns)
    eps_new = rebuild_episodes(eps_old_sorted, df_new, drops, fps, per_ep_stats)

    # 4) Write outputs
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (DST / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    df_new.to_parquet(DST / "data" / "chunk-000" / "file-000.parquet", index=False)
    eps_new.to_parquet(DST / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    shutil.copy(SRC / "meta" / "tasks.parquet", DST / "meta" / "tasks.parquet")
    write_stats_json(per_ep_stats, DST / "meta" / "stats.json")

    # 5) Re-encode videos
    for vk in VIDEO_KEYS:
        src_mp4 = SRC / "videos" / vk / "chunk-000" / "file-000.mp4"
        dst_mp4 = DST / "videos" / vk / "chunk-000" / "file-000.mp4"
        print(f"re-encoding {vk} ...")
        n_out = reencode_video(src_mp4, dst_mp4, drops, src_from, src_to, fps)
        print(f"  wrote {n_out} frames -> {dst_mp4}")
        assert n_out == len(df_new), f"video frame count {n_out} != data rows {len(df_new)}"

    # 6) Update info.json
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

    # 7) Save the drop schedule for reproducibility
    (DST / "meta" / "jitter_drops.json").write_text(json.dumps({
        "seed": SEED,
        "k_min": K_MIN,
        "k_max": K_MAX,
        "drops_per_episode": drops.tolist(),
    }, indent=2))
    print(f"done -> {DST}")


if __name__ == "__main__":
    main()
