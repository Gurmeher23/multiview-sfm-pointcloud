"""Two-view reconstruction with OpenCV: essential matrix (RANSAC), pose, triangulation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Pose, normalize_pixels, project
from .geometry import eight_point, sampson_error, triangulate_dlt, triangulation_angle_deg


@dataclass
class TwoViewResult:
    R: np.ndarray
    t: np.ndarray  # unit norm: scale is unobservable from two views
    inliers: np.ndarray  # (N,) bool, RANSAC inliers of the essential matrix
    valid: np.ndarray  # (N,) bool, inlier AND passes cheirality + reprojection + angle checks
    points: np.ndarray  # (N, 3) triangulated in the frame of camera 1 (rows of invalid pts: junk)
    reproj_err: np.ndarray  # (N,) mean of the two pixel reprojection errors
    median_angle_deg: float


def estimate_two_view(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    threshold_px: float = 1.0,
    prob: float = 0.999,
    max_reproj_px: float = 2.0,
    min_angle_deg: float = 1.0,
) -> TwoViewResult | None:
    """Relative pose + sparse structure from matched pixels."""
    p1 = np.ascontiguousarray(pts1, np.float64)
    p2 = np.ascontiguousarray(pts2, np.float64)
    if len(p1) < 8:
        return None
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=prob, threshold=threshold_px)
    if E is None or mask is None:
        return None
    E = E[:3]  # OpenCV may stack several solutions; keep the first
    inl = mask.ravel().astype(bool)
    if inl.sum() < 8:
        return None
    # OpenCV's 5-point RANSAC returns the minimal-sample model; refit on all inliers with the
    # normalised 8-point algorithm (least squares) and re-evaluate the inliers.
    xn1, xn2 = normalize_pixels(K, p1), normalize_pixels(K, p2)
    E_ref = eight_point(xn1[inl], xn2[inl], essential=True)
    inl_ref = sampson_error(E_ref, xn1, xn2) < (threshold_px / K[0, 0]) ** 2
    if inl_ref.sum() >= max(8, 0.9 * inl.sum()):
        E, inl = E_ref, inl_ref
    _, R, t, _ = cv2.recoverPose(E, p1[inl], p2[inl], K)
    t = t.ravel()
    P1, P2 = np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, t.reshape(3, 1)])
    X = triangulate_dlt(P1, P2, normalize_pixels(K, p1), normalize_pixels(K, p2))
    pose2 = Pose(R, t)
    uv1, z1 = project(K, Pose(np.eye(3), np.zeros(3)), X)
    uv2, z2 = project(K, pose2, X)
    err = 0.5 * (np.linalg.norm(uv1 - p1, axis=1) + np.linalg.norm(uv2 - p2, axis=1))
    ang = triangulation_angle_deg(np.zeros(3), pose2.center, X)
    valid = inl & (z1 > 0) & (z2 > 0) & (err < max_reproj_px) & (ang > min_angle_deg)
    med = float(np.median(ang[valid])) if valid.any() else 0.0
    return TwoViewResult(R, t, inl, valid, X, err, med)
