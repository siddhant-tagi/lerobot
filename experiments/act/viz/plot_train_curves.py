#!/usr/bin/env python
"""Plot training curves from a `lerobot-train` text log.

The training script logs whitespace-separated `key:value` tokens per step,
e.g. `step:200 smpl:1.6K ep:0.4 epch:0.02 loss:1.234 grad_norm:0.567 lr:1e-04`.
Numeric tokens may carry a `format_big_number` suffix (K/M/B/T/Q).

Usage:
    python plot_train_curves.py --log outputs/train/act_baseline.log
    python plot_train_curves.py --log path/to.log --out curves.png \\
        --metrics loss grad_norm lr --x step
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

# `key:value` where value is int/float, optionally with K/M/B/T/Q suffix
# or scientific notation (1e-04). Allows leading `-`.
TOKEN_RE = re.compile(r"(?P<k>[A-Za-z_][\w\.]*):(?P<v>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[KMBTQ]?)")
SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}


def parse_value(s: str) -> float:
    if s and s[-1] in SUFFIX:
        return float(s[:-1]) * SUFFIX[s[-1]]
    return float(s)


def parse_log(path: Path) -> dict[str, list[float]]:
    """Return {metric: [values per logged step]}. Only keeps lines that contain
    `step:` so we don't pick up unrelated `key:value` text."""
    series: dict[str, list[float]] = {}
    with path.open() as fp:
        for line in fp:
            tokens = dict(TOKEN_RE.findall(line))
            if "step" not in tokens:
                continue
            for k, v in tokens.items():
                try:
                    series.setdefault(k, []).append(parse_value(v))
                except ValueError:
                    continue
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True, help="Path to training log.")
    ap.add_argument("--out", type=Path, default=None, help="Output PNG (default: <log>.png).")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric keys to plot. Default: loss, grad_norm, lr if present.",
    )
    ap.add_argument("--x", default="step", choices=["step", "smpl", "ep", "epch"], help="X axis key.")
    ap.add_argument("--ylog", action="store_true", help="Log-scale y axis.")
    args = ap.parse_args()

    series = parse_log(args.log)
    if args.x not in series:
        raise SystemExit(f"X axis key '{args.x}' not found in log. Keys: {sorted(series)}")
    x = series[args.x]

    metrics = args.metrics or [m for m in ("loss", "grad_norm", "lr") if m in series]
    if not metrics:
        raise SystemExit(f"No metrics to plot. Available keys: {sorted(series)}")

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True, squeeze=False)
    for ax, m in zip(axes[:, 0], metrics, strict=True):
        if m not in series:
            # Only reachable when --metrics names a key absent from the log.
            ax.set_title(f"{m} (missing)")
            continue
        y = series[m]
        k = min(len(x), len(y))
        ax.plot(x[:k], y[:k], linewidth=1.2)
        ax.set_ylabel(m)
        ax.grid(alpha=0.3)
        if args.ylog:
            ax.set_yscale("log")
    axes[-1, 0].set_xlabel(args.x)
    fig.suptitle(args.log.name)
    fig.tight_layout()

    out = args.out or args.log.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out} ({n} metrics, {len(x)} points)")


if __name__ == "__main__":
    main()
