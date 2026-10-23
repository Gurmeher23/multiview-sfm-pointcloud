import numpy as np

from sfm import cloud
from sfm.camera import Intrinsics, Pose
from sfm.depth import compare_inverse_depth
from sfm.evaluate import align_points, evaluate_trajectory, umeyama
from sfm.synthetic import Quad, RoomScene, make_room_scene, render


def _plane_cloud(rng, n=3000, normal=(0.0, 0.6, 0.8), offset=0.7, noise=0.005, n_out=400):
    normal = np.asarray(normal) / np.linalg.norm(normal)
    helper = [0, 1, 0] if abs(normal[0]) > 0.9 else [1, 0, 0]
    a = np.cross(normal, helper)
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    uv = rng.uniform(-2, 2, (n, 2))
    pts = uv[:, :1] * a + uv[:, 1:] * b - offset * normal + rng.normal(0, noise, (n, 1)) * normal
    out = rng.uniform(-3, 3, (n_out, 3))
    return np.vstack([pts, out]), normal, offset, n


def test_plane_ransac_finds_known_plane():
    rng = np.random.default_rng(0)
    pts, normal, offset, n = _plane_cloud(rng)
    model, inl = cloud.segment_plane(cloud.to_o3d(pts), distance=0.02)
    cos = abs(np.dot(model[:3], normal))
    assert np.degrees(np.arccos(min(cos, 1.0))) < 1.0
    assert abs(abs(model[3]) - offset) < 0.02
    assert len(np.intersect1d(inl, np.arange(n))) > 0.97 * n


def test_multi_plane_segmentation_finds_both_planes():
    rng = np.random.default_rng(1)
    p1, n1, _, _ = _plane_cloud(rng, n=1500, normal=(0, 0, 1), offset=0.0, n_out=0)
    p2, n2, _, _ = _plane_cloud(rng, n=1500, normal=(1, 0, 0), offset=1.0, n_out=0)
    found = cloud.segment_planes(cloud.to_o3d(np.vstack([p1, p2])), k=2, distance=0.02)
    normals = [np.abs(m[:3]) for m, _ in found]
    assert any(np.allclose(v, [0, 0, 1], atol=0.02) for v in normals)
    assert any(np.allclose(v, [1, 0, 0], atol=0.02) for v in normals)


def test_outlier_removal_and_downsampling():
    rng = np.random.default_rng(2)
    pts, _, _, n = _plane_cloud(rng, n=3000, n_out=0)
    far = rng.uniform(-6, 6, (60, 3)) + 8.0  # isolated points far from the surface
    pcd = cloud.to_o3d(np.vstack([pts, far]))
    clean, kept = cloud.remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
    assert (kept >= n).sum() < 10  # nearly all far points removed
    assert (kept < n).sum() > 0.9 * n
    down = cloud.voxel_downsample(clean, 0.2)
    assert 0 < len(down.points) < len(clean.points) / 5


def test_normals_and_poisson_mesh_on_sphere():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(4000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pcd = cloud.to_o3d(v)
    cloud.estimate_normals(pcd, radius=0.3)
    pcd.orient_normals_to_align_with_direction([0, 0, 1])
    pcd.orient_normals_consistent_tangent_plane(15)
    n = np.asarray(pcd.normals)
    # PCA normals of a sphere are radial up to sign
    assert np.mean(np.abs(np.sum(n * v, axis=1)) > 0.95) > 0.95
    if np.mean(np.sum(n * v, axis=1)) < 0:
        pcd.normals = cloud.o3d.utility.Vector3dVector(-n)
    mesh = cloud.reconstruct_mesh(pcd, "poisson", depth=6)
    assert len(mesh.triangles) > 500
    r = np.linalg.norm(np.asarray(mesh.vertices), axis=1)
    assert abs(np.median(r) - 1.0) < 0.05


def test_umeyama_recovers_similarity_transform():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(30, 3))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 0.5])
    s, Rh, t = umeyama(src, dst)
    assert abs(s - 2.5) < 1e-10 and np.allclose(Rh, R, atol=1e-10) and np.allclose(t, [1, -2, 0.5])
    # a mirrored target must not produce a reflection
    s2, R2, _ = umeyama(src, dst * np.array([1, 1, -1]))
    assert np.linalg.det(R2) > 0


def test_trajectory_evaluation_is_zero_for_similarity_transformed_poses():
    from sfm.camera import orbit_poses

    gt = orbit_poses(6, 3.5, 1.6)
    s, th = 0.4, 0.5
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    t0 = np.array([0.3, 0.1, -0.2])
    est = {}
    for i, p in enumerate(gt):  # reconstruction frame: x_est = (R^T (x_gt - t0)) / s
        C = R.T @ (p.center - t0) / s
        Re = p.R @ R
        est[i] = Pose(Re, -Re @ C)
    m = evaluate_trajectory(est, gt)
    assert m.ate_rmse < 1e-9 and m.rel_rot_err_mean_deg < 1e-6 and m.rot_err_max_deg < 1e-6
    assert abs(m.scale - s) < 1e-9  # x_gt = s * R * x_est + t0
    pts = np.random.default_rng(0).normal(size=(5, 3))
    back = align_points((pts - t0) @ R / s, m.sim3)
    assert np.allclose(back, pts, atol=1e-8)


def test_depth_comparison_is_scale_shift_invariant():
    rng = np.random.default_rng(5)
    z = rng.uniform(2, 9, 500)
    pred = 3.0 / z + 0.2 + rng.normal(0, 0.003, z.shape)  # relative inverse depth, unknown a, b
    m1 = compare_inverse_depth(pred, z)
    m2 = compare_inverse_depth(7.0 * pred - 1.0, z)
    assert m1["abs_rel"] < 0.05 and m1["spearman"] > 0.99 and m1["silog"] < 0.05
    assert all(abs(m1[k] - m2[k]) < 1e-6 for k in ("abs_rel", "silog", "spearman"))
    bad = compare_inverse_depth(rng.random(500), z)
    assert bad["abs_rel"] > 5 * m1["abs_rel"]


def test_quad_distance_and_renderer_depth_are_consistent():
    q = Quad(
        np.zeros(3), np.array([2.0, 0, 0]), np.array([0, 3.0, 0]), np.zeros((4, 4, 3), np.uint8)
    )
    sc = RoomScene([q])
    d = sc.distance_to_surface(np.array([[1, 1, 0.5], [3, 1, 0.0], [3, 4, 0.0]]))
    assert np.allclose(d, [0.5, 1.0, np.sqrt(2)])
    # back-project the rendered depth of a real scene: points must lie on the scene surfaces
    room = make_room_scene(0)
    intr = Intrinsics.from_hfov(160, 120, 70.0)
    pose = Pose.look_at([2.0, -3.0, 1.5], [0, 0, 1])
    _, depth = render(room, intr, pose, noise_sigma=0)
    v, u = np.mgrid[0:120:7, 0:160:7]
    z = depth[v, u]
    assert np.isfinite(z).all()
    xc = np.stack([(u - intr.cx) / intr.fx * z, (v - intr.cy) / intr.fy * z, z], -1).reshape(-1, 3)
    Xw = (xc - pose.t) @ pose.R
    assert room.distance_to_surface(Xw).max() < 1e-6
