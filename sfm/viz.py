"""Matplotlib figures (headless)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_matches(img1, img2, kp1, kp2, path: Path, max_lines: int = 80, seed: int = 0) -> None:
    """Side-by-side image with a random subset of verified matches drawn as lines."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(kp1), min(max_lines, len(kp1)), replace=False)
    canvas = np.hstack([img1, img2]).copy()
    off = img1.shape[1]
    for k in pick:
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        p, q = kp1[k], kp2[k]
        cv2.line(
            canvas, (int(p[0]), int(p[1])), (int(q[0] + off), int(q[1])), color, 1, cv2.LINE_AA
        )
        cv2.circle(canvas, (int(p[0]), int(p[1])), 3, color, -1)
        cv2.circle(canvas, (int(q[0] + off), int(q[1])), 3, color, -1)
    cv2.imwrite(str(path), canvas)


def plot_trajectory(est_centers, gt_centers, path: Path, title="Camera trajectory") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(gt_centers[:, 0], gt_centers[:, 1], "o-", c="tab:green", label="ground truth")
    ax.plot(est_centers[:, 0], est_centers[:, 1], "x--", c="tab:red", label="SfM (Sim3-aligned)")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cloud(points, cam_centers, path: Path, title="Reconstructed sparse point cloud") -> None:
    fig = plt.figure(figsize=(11, 4))
    views = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    for k, (a, b, i, j) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.scatter(points[:, i], points[:, j], c=points[:, 2], s=2, cmap="viridis")
        ax.scatter(cam_centers[:, i], cam_centers[:, j], c="red", marker="^", s=30, label="cameras")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
