"""Pinhole camera model and rigid poses.

Conventions (OpenCV): the camera frame has x right, y down, z forward. A ``Pose`` maps
world points into the camera frame: ``x_cam = R @ X_world + t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> Intrinsics:
        f = 0.5 * width / np.tan(np.deg2rad(hfov_deg) / 2)
        return cls(f, f, (width - 1) / 2, (height - 1) / 2, width, height)

    @classmethod
    def from_focal(cls, width: int, height: int, focal_px: float | None = None) -> Intrinsics:
        """Guess intrinsics: principal point at the centre, f = 1.2 * max side."""
        f = focal_px if focal_px else 1.2 * max(width, height)
        return cls(f, f, (width - 1) / 2, (height - 1) / 2, width, height)


@dataclass(frozen=True)
class Pose:
    """World-to-camera rigid transform ``x_cam = R @ X + t``."""

    R: np.ndarray
    t: np.ndarray

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates, ``-R^T t``."""
        return -self.R.T @ self.t

    @property
    def P(self) -> np.ndarray:
        """3x4 extrinsic matrix [R | t]."""
        return np.hstack([self.R, self.t.reshape(3, 1)])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X) @ self.R.T + self.t

    @staticmethod
    def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> Pose:
        eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        return Pose(R, -R @ eye)


def project(K: np.ndarray, pose: Pose, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points. Returns pixel coordinates (N, 2) and depths (N,)."""
    Xc = pose.transform(X)
    z = Xc[:, 2]
    uv = (Xc[:, :2] / z[:, None]) * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    return uv, z


def normalize_pixels(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Pixels to normalised image coordinates (K^-1 applied, z = 1 dropped)."""
    return (np.asarray(uv, float) - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])


def orbit_poses(
    n: int,
    radius: float,
    height: float,
    target=(0.0, 0.0, 1.0),
    arc_deg: float = 90.0,
    start_deg: float = -45.0,
) -> list[Pose]:
    """Cameras on a circular arc around ``target``, all looking at it."""
    target = np.asarray(target, float)
    angles = np.deg2rad(start_deg + np.linspace(0.0, arc_deg, n))
    poses = []
    for a in angles:
        eye = target + np.array([radius * np.sin(a), -radius * np.cos(a), height - target[2]])
        poses.append(Pose.look_at(eye, target))
    return poses
