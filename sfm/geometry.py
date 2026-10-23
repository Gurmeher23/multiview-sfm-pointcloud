"""Epipolar geometry and triangulation, written in plain numpy so the math is visible.

All routines that say "normalised" take image coordinates with the intrinsics removed
(``K^-1 [u, v, 1]``), so an essential matrix ``E`` satisfies ``x2^T E x1 = 0``.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle of ``Ra Rb^T``."""
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def direction_angle_deg(ta: np.ndarray, tb: np.ndarray) -> float:
    """Angle between two translation directions (sign-sensitive)."""
    a = np.ravel(ta) / np.linalg.norm(ta)
    b = np.ravel(tb) / np.linalg.norm(tb)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def hartley_normalize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate to zero mean and scale to mean distance sqrt(2). Returns (T, x_normalised)."""
    mean = x.mean(axis=0)
    scale = np.sqrt(2.0) / np.mean(np.linalg.norm(x - mean, axis=1))
    T = np.array([[scale, 0, -scale * mean[0]], [0, scale, -scale * mean[1]], [0, 0, 1.0]])
    return T, (x - mean) * scale


def eight_point(x1: np.ndarray, x2: np.ndarray, essential: bool = False) -> np.ndarray:
    """Normalised 8-point algorithm.

    Solves ``x2^T F x1 = 0`` in the least-squares sense: each correspondence gives one row of
    ``A f = 0`` with ``f = vec(F)``; ``f`` is the right singular vector of the smallest singular
    value. Rank 2 is enforced by zeroing the last singular value of ``F``. With
    ``essential=True`` (inputs are calibrated coordinates) the result is projected onto the
    essential manifold, singular values ``(s, s, 0)``.
    """
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    if len(x1) < 8:
        raise ValueError("the 8-point algorithm needs at least 8 correspondences")
    T1, a = hartley_normalize(x1)
    T2, b = hartley_normalize(x2)
    ones = np.ones(len(a))
    A = np.column_stack(
        [
            b[:, 0] * a[:, 0], b[:, 0] * a[:, 1], b[:, 0],
            b[:, 1] * a[:, 0], b[:, 1] * a[:, 1], b[:, 1],
            a[:, 0], a[:, 1], ones,
        ]
    )  # fmt: skip
    F = np.linalg.svd(A)[2][-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    F = U @ np.diag([S[0], S[1], 0.0]) @ Vt
    F = T2.T @ F @ T1  # undo the conditioning transforms
    if essential:
        U, S, Vt = np.linalg.svd(F)
        s = 0.5 * (S[0] + S[1])
        F = U @ np.diag([s, s, 0.0]) @ Vt
    return F / np.linalg.norm(F)


def sampson_error(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Squared Sampson distance, a first-order approximation of the geometric epipolar error."""
    h1 = np.column_stack([x1, np.ones(len(x1))])
    h2 = np.column_stack([x2, np.ones(len(x2))])
    Fx1 = h1 @ F.T
    Ftx2 = h2 @ F
    num = np.sum(h2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / den


def ransac_essential(
    x1: np.ndarray,
    x2: np.ndarray,
    threshold: float,
    iters: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC around :func:`eight_point` on normalised coordinates.

    ``threshold`` is in normalised units (pixel threshold divided by the focal length).
    Returns ``(E, inlier_mask)``, re-fitted on all inliers of the best hypothesis.
    """
    rng = np.random.default_rng(seed)
    n = len(x1)
    best_mask = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 8, replace=False)
        try:
            E = eight_point(x1[idx], x2[idx], essential=True)
        except np.linalg.LinAlgError:
            continue
        mask = sampson_error(E, x1, x2) < threshold**2
        if mask.sum() > best_mask.sum():
            best_mask = mask
    E = eight_point(x1[best_mask], x2[best_mask], essential=True)
    return E, sampson_error(E, x1, x2) < threshold**2


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The four (R, t) hypotheses of an essential matrix (t has unit norm)."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    t = U[:, 2]
    return [(U @ W @ Vt, t), (U @ W @ Vt, -t), (U @ W.T @ Vt, t), (U @ W.T @ Vt, -t)]


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear (DLT) two-view triangulation, vectorised over N points.

    For each point, ``x = (u, v)`` and projection ``P`` give two equations
    ``u P[2] - P[0]`` and ``v P[2] - P[1]``; stacking both views yields ``A X = 0`` with a 4x4
    ``A`` whose null vector is the homogeneous 3D point.
    """
    A = np.stack(
        [
            x1[:, 0, None] * P1[2] - P1[0],
            x1[:, 1, None] * P1[2] - P1[1],
            x2[:, 0, None] * P2[2] - P2[0],
            x2[:, 1, None] * P2[2] - P2[1],
        ],
        axis=1,
    )
    X = np.linalg.svd(A)[2][:, -1]
    return X[:, :3] / X[:, 3:4]


def triangulate_multiview(Ps: list[np.ndarray], xs: np.ndarray) -> np.ndarray:
    """DLT triangulation of a single point seen in k views (normalised coordinates)."""
    rows = []
    for P, (u, v) in zip(Ps, xs, strict=True):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    X = np.linalg.svd(np.asarray(rows))[2][-1]
    return X[:3] / X[3]


def recover_pose(E: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """Pick the (R, t) hypothesis with the most points in front of both cameras (cheirality).

    Returns ``(R, t, in_front_mask, points)`` with the first camera at the identity.
    """
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    for R, t in decompose_essential(E):
        P2 = np.hstack([R, t.reshape(3, 1)])
        X = triangulate_dlt(P1, P2, x1, x2)
        front = (X[:, 2] > 0) & ((X @ R.T + t)[:, 2] > 0)
        if best is None or front.sum() > best[2].sum():
            best = (R, t, front, X)
    return best


def triangulation_angle_deg(C1: np.ndarray, C2: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Angle at each 3D point between the rays to two camera centres."""
    a = C1 - X
    b = C2 - X
    c = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _unit_from_angles(a: float, b: float) -> np.ndarray:
    return np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)])


def refine_pose_sampson(
    R: np.ndarray, t: np.ndarray, x1: np.ndarray, x2: np.ndarray, f_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gold-standard polish of a relative pose: minimise the Sampson distance over (R, t).

    The five degrees of freedom are a rotation vector and the translation direction (two
    spherical angles), so the essential matrix ``E = [t]x R`` is valid by construction. The
    residual is the signed Sampson distance and a Huber loss (scale ``f_scale``, normalised
    units) down-weights residual outliers.
    """
    t = np.asarray(t, float).ravel()
    t = t / np.linalg.norm(t)
    h1 = np.column_stack([x1, np.ones(len(x1))])
    h2 = np.column_stack([x2, np.ones(len(x2))])

    def residuals(p):
        E = skew(_unit_from_angles(p[3], p[4])) @ cv2.Rodrigues(p[:3])[0]
        Fx1, Ftx2 = h1 @ E.T, h2 @ E
        den = np.sqrt(Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)
        return np.sum(h2 * Fx1, axis=1) / den

    p0 = np.r_[cv2.Rodrigues(R)[0].ravel(), np.arccos(np.clip(t[2], -1, 1)), np.arctan2(t[1], t[0])]
    sol = least_squares(residuals, p0, loss="huber", f_scale=f_scale)
    return cv2.Rodrigues(sol.x[:3])[0], _unit_from_angles(sol.x[3], sol.x[4])
