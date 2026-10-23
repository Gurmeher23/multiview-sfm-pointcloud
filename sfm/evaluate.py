"""Ground-truth evaluation: Umeyama similarity alignment, trajectory and structure error."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Pose
from .geometry import rotation_angle_deg


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform ``dst ~ s R src + t`` (Umeyama, 1991).

    Returns ``(s, R, t)``. The rotation comes from the SVD of the cross-covariance
    ``Sigma = (dst - mu_d)^T (src - mu_s) / n``; a reflection is excluded by flipping the
    smallest singular direction when ``det(Sigma) < 0``.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs**2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


@dataclass
class TrajectoryMetrics:
    ate_rmse: float  # metres (GT units), after Sim(3) alignment of camera centres
    rot_err_mean_deg: float
    rot_err_max_deg: float
    rel_rot_err_mean_deg: float  # gauge-free: error of every pairwise relative rotation
    scale: float
    sim3: tuple  # (s, R, t) mapping reconstruction -> ground truth


def evaluate_trajectory(est: dict[int, Pose], gt: list[Pose]) -> TrajectoryMetrics:
    ids = sorted(est)
    ce = np.array([est[i].center for i in ids])
    cg = np.array([gt[i].center for i in ids])
    s, R, t = umeyama(ce, cg)
    ate = float(np.sqrt(np.mean(np.sum((s * ce @ R.T + t - cg) ** 2, axis=1))))
    # World change x_gt = s R x_est + t  =>  camera rotation in GT world is R_est @ R^T.
    rot = [rotation_angle_deg(est[i].R @ R.T, gt[i].R) for i in ids]
    rel = [
        rotation_angle_deg(est[j].R @ est[i].R.T, gt[j].R @ gt[i].R.T)
        for a, i in enumerate(ids)
        for j in ids[a + 1 :]
    ]
    return TrajectoryMetrics(
        ate, float(np.mean(rot)), float(np.max(rot)), float(np.mean(rel)), s, (s, R, t)
    )


def align_points(points: np.ndarray, sim3) -> np.ndarray:
    s, R, t = sim3
    return s * points @ R.T + t


def align_poses(est: dict[int, Pose], sim3) -> dict[int, Pose]:
    """Express reconstructed camera poses in the ground-truth world frame."""
    s, R, t = sim3
    out = {}
    for i, p in est.items():
        C = s * R @ p.center + t
        Rn = p.R @ R.T
        out[i] = Pose(Rn, -Rn @ C)
    return out


@dataclass
class StructureMetrics:
    rmse: float
    median: float
    p90: float
    n: int


def evaluate_structure(points_gt_frame: np.ndarray, scene) -> StructureMetrics:
    """Distance from reconstructed points to the exact ground-truth surfaces."""
    d = scene.distance_to_surface(points_gt_frame)
    return StructureMetrics(
        float(np.sqrt(np.mean(d**2))), float(np.median(d)), float(np.quantile(d, 0.9)), len(d)
    )
