#!/usr/bin/env python
"""Plot eval results from a `lerobot-eval` `eval_info.json`.

The file has shape:
    {
      "<task_group>": {"avg_sum_reward": float, "avg_max_reward": float,
                        "pc_success": float, ...},
      ...,
      "overall":     {"avg_sum_reward": ..., "avg_max_reward": ..., "pc_success": ...},
    }

One bar chart per metric, with task groups on x and the `overall` value as a
horizontal reference line.

Usage:
    python plot_eval.py --eval-info outputs/eval/.../eval_info.json
    python plot_eval.py --eval-info ... --metrics pc_success --out eval.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_METRICS = ("pc_success", "avg_sum_reward", "avg_max_reward")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-info", type=Path, required=True, help="Path to eval_info.json.")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG (default: <eval-info>.png).")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=f"Metric keys to plot (default: {', '.join(DEFAULT_METRICS)} if present).",
    )
    args = ap.parse_args()

    with args.eval_info.open() as fp:
        info = json.load(fp)

    overall = info.get("overall", {})
    groups = {k: v for k, v in info.items() if k != "overall" and isinstance(v, dict)}
    if not groups:
        logging.warning("no per-group dicts found; plotting 'overall' as a single bar.")
        groups = {"overall": overall}
        overall = {}

    if args.metrics:
        metrics = list(args.metrics)
    else:
        # Prefer the canonical lerobot eval metrics, but only those present
        # (in any group) and numeric.
        present = set()
        for v in groups.values():
            present.update(k for k, x in v.items() if isinstance(x, (int, float)))
        metrics = [m for m in DEFAULT_METRICS if m in present] or sorted(present)

    if not metrics:
        raise SystemExit(f"No numeric metrics found in {args.eval_info}.")

    names = list(groups)
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(max(6, 0.6 * len(names) + 3), 3 * n), squeeze=False)
    for ax, m in zip(axes[:, 0], metrics, strict=False):
        ys = [float(groups[g].get(m, float("nan"))) for g in names]
        ax.bar(names, ys, color="#4C78A8")
        ax.set_ylabel(m)
        ax.grid(axis="y", alpha=0.3)
        for i, y in enumerate(ys):
            if not math.isnan(y):
                ax.text(i, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
        if m in overall and isinstance(overall[m], (int, float)):
            ax.axhline(float(overall[m]), color="crimson", linestyle="--", linewidth=1,
                       label=f"overall: {overall[m]:.2f}")
            ax.legend(loc="best", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(args.eval_info.parent.name or args.eval_info.name)
    fig.tight_layout()

    out = args.out or args.eval_info.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out} ({n} metrics, {len(names)} groups)")


if __name__ == "__main__":
    main()
