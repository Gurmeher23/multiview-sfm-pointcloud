import cv2
import numpy as np

from sfm.bundle_adjustment import bundle_adjust, project_obs, rmse_px, rotate
from sfm.camera import Pose, project
from sfm.geometry import direction_angle_deg, rotation_angle_deg
from sfm.synthetic import make_point_scene
from sfm.two_view import estimate_two_view


def test_two_view_pose_recovery_with_outliers(pair):
    rng = np.random.default_rng(2)
    uv1 = pair["uv1"] + rng.normal(0, 0.5, pair["uv1"].shape)
    uv2 = pair["uv2"] + rng.normal(0, 0.5, pair["uv2"].shape)
    bad = rng.random(len(uv2)) < 0.2
    uv2[bad] = rng.uniform([0, 0], [640, 480], (bad.sum(), 2))
    res = estimate_two_view(uv1, uv2, pair["scene"].K)
    assert rotation_angle_deg(res.R, pair["R"]) < 1.0
    assert direction_angle_deg(res.t, pair["t"]) < 2.0
    assert res.valid.sum() > 0.7 * (~bad).sum()
    assert (res.valid & bad).sum() <= 0.1 * bad.sum() + 3
    assert np.median(res.reproj_err[res.valid]) < 1.0


def test_rodrigues_matches_opencv():
    rng = np.random.default_rng(0)
    rv = rng.normal(0, 0.8, (20, 3))
    p = rng.normal(0, 1, (20, 3))
    ref = np.array([cv2.Rodrigues(r)[0] @ x for r, x in zip(rv, p, strict=True)])
    assert np.allclose(rotate(p, rv), ref, atol=1e-10)


def test_pnp_ransac_recovers_pose_with_outliers():
    scene = make_point_scene(n_points=300, n_cams=4, noise_px=0.5, outlier_frac=0.25, seed=4)
    cam = 2
    m = scene.cam_idx == cam
    obj = scene.points[scene.pt_idx[m]]
    img = scene.uv[m]
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj, img, scene.K, None, reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP, iterationsCount=500
    )
    assert ok
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], scene.K, None, rvec, tvec)
    gt = scene.poses[cam]
    assert rotation_angle_deg(cv2.Rodrigues(rvec)[0], gt.R) < 0.2
    assert np.linalg.norm(tvec.ravel() - gt.t) < 0.05
    out = scene.is_outlier[m]
    assert out[inl].mean() < 0.02  # almost no gross outliers survive


def _perturbed(scene, rot_deg=1.0, trans=0.05, pt_sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rv = np.array([cv2.Rodrigues(p.R)[0].ravel() for p in scene.poses])
    tv = np.array([p.t for p in scene.poses])
    rv0 = rv + rng.normal(0, np.deg2rad(rot_deg), rv.shape)
    tv0 = tv + rng.normal(0, trans, tv.shape)
    pts0 = scene.points + rng.normal(0, pt_sigma, scene.points.shape)
    rv0[0], tv0[0] = rv[0], tv[0]  # camera 0 is the fixed gauge anchor
    return rv, tv, rv0, tv0, pts0


def test_bundle_adjustment_reduces_reprojection_error():
    scene = make_point_scene(n_points=250, n_cams=6, noise_px=0.5, seed=5)
    rv, tv, rv0, tv0, pts0 = _perturbed(scene)
    res = bundle_adjust(
        scene.K, rv0, tv0, pts0, scene.cam_idx, scene.pt_idx, scene.uv, loss="linear"
    )
    assert res.rmse_before > 10 * res.rmse_after
    # noise floor: sqrt(2) * sigma for Euclidean error; BA over-fits slightly, so it can be below
    assert res.rmse_after < 0.8
    # reprojecting with the *ground-truth* parameters is the expected optimum: BA should match it
    gt_rmse = rmse_px(scene.K, rv, tv, scene.points, scene.cam_idx, scene.pt_idx, scene.uv)
    assert res.rmse_after <= gt_rmse * 1.05
    # camera centres moved toward the truth
    est = [Pose(cv2.Rodrigues(r)[0], t) for r, t in zip(res.rvecs, res.tvecs, strict=True)]
    err_before = np.mean([np.linalg.norm(Pose(cv2.Rodrigues(r)[0], t).center - p.center)
                          for r, t, p in zip(rv0, tv0, scene.poses, strict=True)])  # fmt: skip
    err_after = np.mean(
        [np.linalg.norm(e.center - p.center) for e, p in zip(est, scene.poses, strict=True)]
    )
    assert err_after < err_before


def test_bundle_adjustment_robust_loss_handles_outliers():
    scene = make_point_scene(n_points=250, n_cams=6, noise_px=0.5, outlier_frac=0.1, seed=6)
    rv, tv, rv0, tv0, pts0 = _perturbed(scene, rot_deg=0.3, trans=0.02, pt_sigma=0.02, seed=1)
    # Uniform random pixels are gross outliers: a redescending loss (Cauchy) ignores them, while
    # a plain linear loss is dragged far off.
    kw = dict(fixed_cams=(0,), max_nfev=400)
    args = (scene.K, rv0, tv0, pts0, scene.cam_idx, scene.pt_idx, scene.uv)
    res = bundle_adjust(*args, loss="cauchy", f_scale=1.0, **kw)
    lin = bundle_adjust(*args, loss="linear", **kw)
    inl = ~scene.is_outlier
    pred = project_obs(
        scene.K, res.rvecs[scene.cam_idx], res.tvecs[scene.cam_idx], res.points[scene.pt_idx]
    )
    inlier_rmse = np.sqrt(np.mean(np.sum((pred - scene.uv)[inl] ** 2, axis=1)))
    assert inlier_rmse < 1.0
    pred_lin = project_obs(
        scene.K, lin.rvecs[scene.cam_idx], lin.tvecs[scene.cam_idx], lin.points[scene.pt_idx]
    )
    assert np.sqrt(np.mean(np.sum((pred_lin - scene.uv)[inl] ** 2, axis=1))) > 5 * inlier_rmse
    uv_gt, _ = project(scene.K, scene.poses[1], scene.points)  # sanity: scene is consistent
    assert np.isfinite(uv_gt).all()
