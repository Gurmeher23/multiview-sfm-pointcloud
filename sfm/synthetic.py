"""Synthetic scenes with exact ground truth.

Two generators:

* :func:`make_point_scene`: a 3D point cloud, cameras on an orbit and noisy pixel observations
  (with outliers). Fast, no images; used for geometry and bundle-adjustment tests.
* :func:`make_room_scene` + :func:`render`: a textured room with boxes, ray-cast into real images
  with exact depth maps, so the full feature-matching pipeline can run on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Intrinsics, Pose, orbit_poses, project

# --------------------------------------------------------------------------- point-only scene


@dataclass
class PointScene:
    K: np.ndarray
    intrinsics: Intrinsics
    poses: list[Pose]
    points: np.ndarray  # (M, 3) ground truth
    cam_idx: np.ndarray  # (L,) observation -> camera
    pt_idx: np.ndarray  # (L,) observation -> point
    uv: np.ndarray  # (L, 2) noisy pixel observations
    uv_clean: np.ndarray  # (L, 2) noise-free projections
    is_outlier: np.ndarray  # (L,) bool


def make_point_scene(
    n_points: int = 400,
    n_cams: int = 6,
    noise_px: float = 0.5,
    outlier_frac: float = 0.0,
    seed: int = 0,
    arc_deg: float = 80.0,
) -> PointScene:
    """Random points in a shell around the origin seen from an orbit of cameras."""
    rng = np.random.default_rng(seed)
    intr = Intrinsics.from_hfov(640, 480, 70.0)
    K = intr.K
    pts = rng.uniform([-1.5, -1.5, 0.0], [1.5, 1.5, 2.0], size=(n_points, 3))
    poses = orbit_poses(n_cams, radius=5.0, height=1.5, target=(0, 0, 1.0), arc_deg=arc_deg)
    cam_i, pt_i, uvs, clean = [], [], [], []
    for ci, pose in enumerate(poses):
        uv, z = project(K, pose, pts)
        ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < intr.width)
        ok &= (uv[:, 1] >= 0) & (uv[:, 1] < intr.height)
        idx = np.flatnonzero(ok)
        cam_i.append(np.full(len(idx), ci))
        pt_i.append(idx)
        clean.append(uv[idx])
        uvs.append(uv[idx] + rng.normal(0, noise_px, (len(idx), 2)))
    cam_idx, pt_idx = np.concatenate(cam_i), np.concatenate(pt_i)
    uv, clean = np.vstack(uvs), np.vstack(clean)
    is_out = rng.random(len(uv)) < outlier_frac
    uv[is_out] = rng.uniform([0, 0], [intr.width, intr.height], size=(is_out.sum(), 2))
    return PointScene(K, intr, poses, pts, cam_idx, pt_idx, uv, clean, is_out)


# ------------------------------------------------------------------------- textured room scene


@dataclass
class Quad:
    """A rectangle: ``origin + s * u + r * v`` for s, r in [0, 1], with a texture."""

    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    texture: np.ndarray = field(repr=False)

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.u, self.v)
        return n / np.linalg.norm(n)


def _make_texture(rng: np.random.Generator, width_m: float, height_m: float) -> np.ndarray:
    """Multi-scale noise plus random shapes: rich in corners/blobs for SIFT and ORB."""
    px_per_m = 100
    w = int(np.clip(width_m * px_per_m, 64, 1400))
    h = int(np.clip(height_m * px_per_m, 64, 1400))
    tex = np.zeros((h, w, 3), np.float32)
    for k in (4, 8, 16, 32, 64):
        small = rng.random((max(2, k * h // w), k, 3)).astype(np.float32)
        tex += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) / np.sqrt(k)
    tex = 255 * (tex - tex.min()) / (np.ptp(tex) + 1e-9)
    tex = tex.astype(np.uint8)
    for _ in range(max(10, w * h // 900)):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        size = int(rng.integers(3, 14))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(tex, (x, y), (x + size, y + size), color, -1)
        elif kind == 1:
            cv2.circle(tex, (x, y), size // 2 + 1, color, -1)
        else:
            cv2.line(tex, (x, y), (x + size, y + int(rng.integers(-size, size + 1))), color, 2)
    return tex


def _box_quads(center, size, yaw, rng) -> list[Quad]:
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    half = np.asarray(size, float) / 2
    ex, ey, ez = (Rz @ np.diag(2 * half)).T  # edge vectors
    c0 = np.asarray(center, float) - Rz @ half
    faces = [
        (c0, ex, ey), (c0 + ez, ex, ey),  # bottom, top
        (c0, ex, ez), (c0 + ey, ex, ez),  # front, back
        (c0, ey, ez), (c0 + ex, ey, ez),  # left, right
    ]  # fmt: skip
    return [
        Quad(o, u, v, _make_texture(rng, np.linalg.norm(u), np.linalg.norm(v))) for o, u, v in faces
    ]


@dataclass
class RoomScene:
    quads: list[Quad]

    def distance_to_surface(self, X: np.ndarray) -> np.ndarray:
        """Exact distance from each point to the nearest scene rectangle."""
        best = np.full(len(X), np.inf)
        for q in self.quads:
            lu, lv = np.linalg.norm(q.u), np.linalg.norm(q.v)
            d = X - q.origin
            s = d @ (q.u / lu)
            r = d @ (q.v / lv)
            dn = d @ q.normal
            ds = s - np.clip(s, 0, lu)
            dr = r - np.clip(r, 0, lv)
            best = np.minimum(best, np.sqrt(ds**2 + dr**2 + dn**2))
        return best


def make_room_scene(seed: int = 0) -> RoomScene:
    """A 12 x 12 x 5 m room (floor, ceiling, four walls) with three yawed boxes in the middle."""
    rng = np.random.default_rng(seed)
    a, h = 6.0, 5.0
    ex, ey, ez = np.array([2 * a, 0, 0]), np.array([0, 2 * a, 0]), np.array([0, 0, h])
    o = np.array([-a, -a, 0.0])
    faces = [
        (o, ex, ey),
        (o + ez, ex, ey),
        (o, ex, ez),
        (o + ey, ex, ez),
        (o, ey, ez),
        (o + ex, ey, ez),
    ]
    quads = [
        Quad(p, u, v, _make_texture(rng, np.linalg.norm(u), np.linalg.norm(v))) for p, u, v in faces
    ]
    quads += _box_quads((0.0, 0.0, 0.5), (1.0, 1.0, 1.0), 0.4, rng)
    quads += _box_quads((1.4, 0.8, 0.35), (0.8, 0.6, 0.7), -0.6, rng)
    quads += _box_quads((-1.3, -0.6, 0.6), (0.7, 0.9, 1.2), 1.0, rng)
    return RoomScene(quads)


def render(
    scene: RoomScene, intr: Intrinsics, pose: Pose, noise_sigma: float = 1.5, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast a BGR image and its depth map (z-depth, metres; inf where nothing is hit)."""
    h, w = intr.height, intr.width
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    rays_cam = np.stack(
        [(uu - intr.cx) / intr.fx, (vv - intr.cy) / intr.fy, np.ones_like(uu, float)], -1
    )
    d = rays_cam @ pose.R  # world-frame ray directions, z_cam component = 1 per unit t
    c = pose.center
    best_t = np.full((h, w), np.inf)
    image = np.zeros((h, w, 3), np.uint8)
    for q in scene.quads:
        n = q.normal
        denom = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((q.origin - c) @ n) / denom
        p = c + t[..., None] * d
        rel = p - q.origin
        s = (rel @ q.u) / (q.u @ q.u)
        r = (rel @ q.v) / (q.v @ q.v)
        hit = (t > 0.05) & (t < best_t) & (s >= 0) & (s <= 1) & (r >= 0) & (r <= 1)
        if not hit.any():
            continue
        th, tw = q.texture.shape[:2]
        sampled = cv2.remap(
            q.texture,
            (np.clip(s, 0, 1) * (tw - 1)).astype(np.float32),
            (np.clip(r, 0, 1) * (th - 1)).astype(np.float32),
            cv2.INTER_LINEAR,
        )
        image[hit] = sampled[hit]
        best_t[hit] = t[hit]
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        image = np.clip(image + rng.normal(0, noise_sigma, image.shape), 0, 255).astype(np.uint8)
    return image, best_t


def make_room_views(n_views: int = 8, seed: int = 0, size=(640, 480), arc_deg: float = 90.0):
    """Render ``n_views`` images on an orbit. Returns (scene, intrinsics, poses, images, depths)."""
    scene = make_room_scene(seed)
    intr = Intrinsics.from_hfov(size[0], size[1], 70.0)
    poses = orbit_poses(n_views, radius=3.5, height=1.6, target=(0, 0, 1.0), arc_deg=arc_deg)
    images, depths = [], []
    for i, pose in enumerate(poses):
        im, dep = render(scene, intr, pose, seed=seed * 1000 + i)
        images.append(im)
        depths.append(dep)
    return scene, intr, poses, images, depths
