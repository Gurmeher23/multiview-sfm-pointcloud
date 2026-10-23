import numpy as np
import pytest

from sfm.camera import normalize_pixels
from sfm.synthetic import make_point_scene


@pytest.fixture(scope="session")
def pair():
    """Two views of a clean synthetic point scene with ground-truth relative pose and E."""
    scene = make_point_scene(n_points=300, n_cams=5, noise_px=0.0, seed=1)
    i, j = 0, 3
    common = np.intersect1d(scene.pt_idx[scene.cam_idx == i], scene.pt_idx[scene.cam_idx == j])

    def obs(cam):
        m = scene.cam_idx == cam
        order = {p: k for k, p in enumerate(scene.pt_idx[m])}
        return scene.uv[m][[order[p] for p in common]]

    Ri, ti, Rj, tj = scene.poses[i].R, scene.poses[i].t, scene.poses[j].R, scene.poses[j].t
    R = Rj @ Ri.T
    t = tj - R @ ti
    from sfm.geometry import skew

    return {
        "scene": scene,
        "uv1": obs(i),
        "uv2": obs(j),
        "x1": normalize_pixels(scene.K, obs(i)),
        "x2": normalize_pixels(scene.K, obs(j)),
        "R": R,
        "t": t,
        "E": skew(t / np.linalg.norm(t)) @ R,
        "X": scene.points[common],
        "poses": (scene.poses[i], scene.poses[j]),
    }
