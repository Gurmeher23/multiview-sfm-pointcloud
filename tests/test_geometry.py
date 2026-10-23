import cv2
import numpy as np

from sfm.camera import normalize_pixels
from sfm.geometry import (
    decompose_essential,
    direction_angle_deg,
    eight_point,
    ransac_essential,
    recover_pose,
    rotation_angle_deg,
    sampson_error,
    triangulate_dlt,
    triangulate_multiview,
)


def _same_up_to_sign(A, B):
    A, B = A / np.linalg.norm(A), B / np.linalg.norm(B)
    return min(np.linalg.norm(A - B), np.linalg.norm(A + B))


def test_eight_point_exact_on_noiseless_data(pair):
    E = eight_point(pair["x1"], pair["x2"], essential=True)
    assert _same_up_to_sign(E, pair["E"]) < 1e-8
    assert np.allclose(np.linalg.svd(E)[1], [np.sqrt(0.5)] * 2 + [0], atol=1e-8)
    assert sampson_error(E, pair["x1"], pair["x2"]).max() < 1e-16


def test_eight_point_matches_opencv(pair):
    rng = np.random.default_rng(0)
    uv1 = pair["uv1"] + rng.normal(0, 0.5, pair["uv1"].shape)
    uv2 = pair["uv2"] + rng.normal(0, 0.5, pair["uv2"].shape)
    K = pair["scene"].K
    x1, x2 = normalize_pixels(K, uv1), normalize_pixels(K, uv2)
    E_np = eight_point(x1, x2, essential=True)
    E_cv, _ = cv2.findEssentialMat(uv1, uv2, K, method=cv2.RANSAC, threshold=1.0)
    R_np, t_np, _, _ = recover_pose(E_np, x1, x2)
    _, R_cv, t_cv, _ = cv2.recoverPose(E_cv[:3], uv1, uv2, K)
    # Both estimators should agree with ground truth and with each other.
    assert rotation_angle_deg(R_np, pair["R"]) < 1.0
    # OpenCV's minimal-sample RANSAC model is unrefined, so it is allowed to be less exact
    assert rotation_angle_deg(R_cv, pair["R"]) < 3.0
    assert rotation_angle_deg(R_np, R_cv) < 3.0
    assert direction_angle_deg(t_np, pair["t"]) < 3.0
    assert direction_angle_deg(t_np, t_cv.ravel()) < 6.0


def test_ransac_essential_rejects_outliers(pair):
    rng = np.random.default_rng(3)
    x1, x2 = pair["x1"].copy(), pair["x2"].copy()
    bad = rng.random(len(x1)) < 0.3
    x2[bad] = rng.uniform(-0.5, 0.5, (bad.sum(), 2))
    E, mask = ransac_essential(x1, x2, threshold=1e-3, seed=0)
    assert (mask & ~bad).sum() >= 0.95 * (~bad).sum()
    assert (mask & bad).sum() <= 0.05 * bad.sum() + 2
    R, t, _, _ = recover_pose(E, x1[mask], x2[mask])
    assert rotation_angle_deg(R, pair["R"]) < 0.1
    assert direction_angle_deg(t, pair["t"]) < 0.1


def test_decompose_contains_ground_truth_and_cheirality_picks_it(pair):
    cands = decompose_essential(pair["E"])
    assert any(
        rotation_angle_deg(R, pair["R"]) < 1e-6 and direction_angle_deg(t, pair["t"]) < 1e-6
        for R, t in cands
    )
    R, t, front, X = recover_pose(pair["E"], pair["x1"], pair["x2"])
    assert rotation_angle_deg(R, pair["R"]) < 1e-6
    assert direction_angle_deg(t, pair["t"]) < 1e-6
    assert front.all()


def test_triangulation_exact_without_noise(pair):
    p1, p2 = pair["poses"]
    # Work in the frame of camera 1 (metric), so compare to X expressed there.
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([pair["R"], pair["t"].reshape(3, 1)])
    X = triangulate_dlt(P1, P2, pair["x1"], pair["x2"])
    X_gt = p1.transform(pair["X"])
    assert np.abs(X - X_gt).max() < 1e-8
    # Multi-view version agrees point by point.
    k = 7
    Xm = triangulate_multiview([P1, P2], np.array([pair["x1"][k], pair["x2"][k]]))
    assert np.allclose(Xm, X_gt[k], atol=1e-8)


def test_triangulation_noise_accuracy(pair):
    rng = np.random.default_rng(1)
    K = pair["scene"].K
    uv1 = pair["uv1"] + rng.normal(0, 0.5, pair["uv1"].shape)
    uv2 = pair["uv2"] + rng.normal(0, 0.5, pair["uv2"].shape)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    scale = np.linalg.norm(pair["t"])
    P2 = np.hstack([pair["R"], (pair["t"] / scale).reshape(3, 1)])  # unit baseline, as in two views
    X = triangulate_dlt(P1, P2, normalize_pixels(K, uv1), normalize_pixels(K, uv2))
    X_gt = pair["poses"][0].transform(pair["X"]) / scale
    assert np.median(np.linalg.norm(X - X_gt, axis=1)) < 0.05 * np.median(
        np.linalg.norm(X_gt, axis=1)
    )
