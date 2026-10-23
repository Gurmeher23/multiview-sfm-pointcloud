"""Sparse bundle adjustment with scipy.optimize.least_squares.

Parameters: one 6-vector (Rodrigues rotation, translation) per free camera and one 3-vector per
point. Intrinsics are held fixed. Each observation contributes a 2D residual that depends only on
its own camera and its own point, so the Jacobian is very sparse; ``jac_sparsity`` lets scipy
estimate it with a handful of finite-difference evaluations (column grouping) and solve the
trust-region steps with LSMR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix


def rotate(points: np.ndarray, rvecs: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of each point by its own axis-angle vector."""
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(theta > 1e-12, rvecs / theta, 0.0)
    cos, sin = np.cos(theta), np.sin(theta)
    dot = np.sum(points * v, axis=1, keepdims=True)
    return cos * points + sin * np.cross(v, points) + (1 - cos) * dot * v


def project_obs(K, rvecs, tvecs, points) -> np.ndarray:
    Xc = rotate(points, rvecs) + tvecs
    return Xc[:, :2] / Xc[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]


@dataclass
class BAResult:
    rvecs: np.ndarray
    tvecs: np.ndarray
    points: np.ndarray
    rmse_before: float
    rmse_after: float
    nfev: int
    success: bool


def rmse_px(K, rvecs, tvecs, points, cam_idx, pt_idx, uv) -> float:
    """sqrt(mean squared Euclidean reprojection error) in pixels."""
    pred = project_obs(K, rvecs[cam_idx], tvecs[cam_idx], points[pt_idx])
    return float(np.sqrt(np.mean(np.sum((pred - uv) ** 2, axis=1))))


def bundle_adjust(
    K: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    points: np.ndarray,
    cam_idx: np.ndarray,
    pt_idx: np.ndarray,
    uv: np.ndarray,
    fixed_cams: tuple[int, ...] = (0,),
    loss: str = "huber",
    f_scale: float = 1.0,
    max_nfev: int = 50,
) -> BAResult:
    """Refine camera poses and points by minimising (robust) reprojection error.

    ``fixed_cams`` are held constant to remove the gauge freedom (the first camera pins the
    world frame; monocular scale remains a free gauge, which the trust-region solver tolerates).
    """
    n_cam, n_pt, n_obs = len(rvecs), len(points), len(uv)
    free = np.array([c for c in range(n_cam) if c not in set(fixed_cams)], int)
    free_pos = np.full(n_cam, -1)
    free_pos[free] = np.arange(len(free))
    n_free = len(free)
    cam0 = np.hstack([rvecs, tvecs])

    def unpack(x):
        cams = cam0.copy()
        cams[free] = x[: 6 * n_free].reshape(-1, 6)
        return cams[:, :3], cams[:, 3:], x[6 * n_free :].reshape(-1, 3)

    def residuals(x):
        r, t, p = unpack(x)
        return (project_obs(K, r[cam_idx], t[cam_idx], p[pt_idx]) - uv).ravel()

    # Jacobian sparsity: rows 2i, 2i+1 depend on camera cam_idx[i] (if free) and point pt_idx[i].
    rows, cols = [], []
    obs = np.arange(n_obs)
    for k in range(2):
        r = 2 * obs + k
        for s in range(3):
            rows.append(r)
            cols.append(6 * n_free + 3 * pt_idx + s)
        is_free = free_pos[cam_idx] >= 0
        for s in range(6):
            rows.append(r[is_free])
            cols.append(6 * free_pos[cam_idx[is_free]] + s)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    sparsity = coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(2 * n_obs, 6 * n_free + 3 * n_pt)
    )

    x0 = np.concatenate([cam0[free].ravel(), points.ravel()])
    before = rmse_px(K, rvecs, tvecs, points, cam_idx, pt_idx, uv)
    sol = least_squares(
        residuals, x0, jac_sparsity=sparsity, loss=loss, f_scale=f_scale, x_scale="jac",
        method="trf", tr_solver="lsmr", max_nfev=max_nfev,
    )  # fmt: skip
    r, t, p = unpack(sol.x)
    after = rmse_px(K, r, t, p, cam_idx, pt_idx, uv)
    return BAResult(r, t, p, before, after, sol.nfev, bool(sol.success))
