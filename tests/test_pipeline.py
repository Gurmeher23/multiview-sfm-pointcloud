"""End-to-end tests on rendered images (slower: a few seconds each)."""

import json

import numpy as np
import pytest

from sfm.evaluate import align_points, evaluate_structure, evaluate_trajectory
from sfm.features import detect, match
from sfm.run import main
from sfm.sfm import SfMConfig, build_tracks, run_sfm
from sfm.synthetic import make_room_views


@pytest.fixture(scope="module")
def room():
    return make_room_views(5, seed=2)


@pytest.fixture(scope="module")
def reconstruction(room):
    _, intr, _, images, _ = room
    return run_sfm(images, intr.K, SfMConfig())


def test_feature_matching_is_geometrically_consistent(room):
    scene, intr, poses, images, depths = room
    f0, f1 = detect(images[0]), detect(images[1])
    m = match(f0, f1)
    assert len(m) > 150
    # Ground truth check with the exact depth map: back-project view 0 matches, reproject to view 1.
    uv0 = f0.xy[m[:, 0]]
    z = depths[0][np.rint(uv0[:, 1]).astype(int), np.rint(uv0[:, 0]).astype(int)]
    xc = np.column_stack([(uv0 - [intr.cx, intr.cy]) / [intr.fx, intr.fy] * z[:, None], z])
    Xw = (xc - poses[0].t) @ poses[0].R
    Xc1 = poses[1].transform(Xw)
    uv1 = Xc1[:, :2] / Xc1[:, 2:3] * [intr.fx, intr.fy] + [intr.cx, intr.cy]
    err = np.linalg.norm(uv1 - f1.xy[m[:, 1]], axis=1)
    assert np.mean(err < 3.0) > 0.9  # ratio test + mutual check leave very few wrong matches


def test_build_tracks_links_views_and_drops_conflicts():
    pm = {(0, 1): np.array([[0, 0], [1, 1]]), (1, 2): np.array([[0, 5], [1, 6]])}
    tracks = build_tracks([3, 3, 7], pm)
    assert sorted(map(sorted, (t.items() for t in tracks))) == [
        [(0, 0), (1, 0), (2, 5)],
        [(0, 1), (1, 1), (2, 6)],
    ]
    # two keypoints of image 1 linked to the same keypoint of image 0 -> inconsistent, dropped
    bad = {(0, 1): np.array([[0, 0], [0, 1]])}
    assert build_tracks([2, 2], bad) == []


def test_incremental_sfm_registers_all_views_and_matches_ground_truth(room, reconstruction):
    scene, _, gt, _, _ = room
    res = reconstruction
    assert res.registered == [0, 1, 2, 3, 4]
    assert len(res.points) > 1000
    traj = evaluate_trajectory(res.poses, gt)
    assert traj.ate_rmse < 0.02  # metres, on a 3.5 m radius orbit
    assert traj.rel_rot_err_mean_deg < 0.3
    st = evaluate_structure(align_points(res.points, traj.sim3), scene)
    assert st.median < 0.05


def test_bundle_adjustment_in_pipeline_reduces_reprojection_error(reconstruction):
    res = reconstruction
    assert res.rmse_after_ba < res.rmse_before_ba
    assert res.rmse_after_ba < 0.5
    assert res.median_after_ba < 0.3


def test_cli_synthetic_and_images_modes(tmp_path):
    out = tmp_path / "out"
    imgs = tmp_path / "imgs"
    argv = ["--synthetic", "--views", "4", "--skip-depth", "--skip-mesh", "--out", str(out)]
    m = main([*argv, "--write-images", str(imgs)])
    assert m["registered"] == 4 and m["reproj_rmse_px"]["after_ba"] < 1.0
    for name in ("metrics.json", "matches.png", "trajectory.png", "pointcloud.png", "cloud.ply"):
        assert (out / name).stat().st_size > 0
    assert json.loads((out / "metrics.json").read_text())["depth"] == "skipped"
    # image-folder mode on the rendered frames (true focal length supplied, no ground truth)
    f = 0.5 * 640 / np.tan(np.deg2rad(35))
    out2 = tmp_path / "o2"
    m2 = main(["--images", str(imgs), "--focal", f"{f}", "--skip-depth", "--out", str(out2)])
    assert m2["registered"] >= 3 and "trajectory" not in m2
    assert (out2 / "mesh.obj").exists()
