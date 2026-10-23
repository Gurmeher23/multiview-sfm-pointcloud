"""Point-cloud and mesh processing with Open3D."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def to_o3d(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, float))
    return pcd


def remove_outliers(pcd, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """Statistical outlier removal: drop points with mean k-NN distance > mean + std_ratio * std."""
    clean, kept = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return clean, np.asarray(kept)


def voxel_downsample(pcd, voxel: float):
    return pcd.voxel_down_sample(voxel)


def estimate_normals(pcd, radius: float = 0.3, max_nn: int = 30, camera_centers=None):
    """PCA normals on a radius neighbourhood, oriented towards the cameras."""
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    if camera_centers is not None:
        pcd.orient_normals_towards_camera_location(np.mean(camera_centers, axis=0))
    return pcd


def segment_plane(pcd, distance: float = 0.03, iters: int = 2000, seed: int | None = 0):
    """RANSAC plane fit. Returns ((a, b, c, d) with unit normal, inlier indices)."""
    if seed is not None:
        o3d.utility.random.seed(seed)
    model, inliers = pcd.segment_plane(
        distance_threshold=distance, ransac_n=3, num_iterations=iters
    )
    return np.asarray(model), np.asarray(inliers)


def segment_planes(pcd, k: int = 3, distance: float = 0.03, min_points: int = 30):
    """Greedily peel off up to ``k`` dominant planes. Returns a list of (model, global_indices)."""
    remaining = np.arange(len(pcd.points))
    found = []
    for _ in range(k):
        if len(remaining) < min_points:
            break
        sub = pcd.select_by_index(remaining)
        model, inl = segment_plane(sub, distance)
        if len(inl) < min_points:
            break
        found.append((model, remaining[inl]))
        remaining = np.delete(remaining, inl)
    return found


def reconstruct_mesh(pcd, method: str = "poisson", depth: int = 7, trim_quantile: float = 0.05):
    """Poisson (density-trimmed) or ball-pivoting surface reconstruction; needs normals."""
    if not pcd.has_normals():
        raise ValueError("estimate normals first")
    if method == "poisson":
        mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        dens = np.asarray(density)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim_quantile))
    elif method == "bpa":
        d = np.asarray(pcd.compute_nearest_neighbor_distance())
        r = float(d.mean())
        radii = o3d.utility.DoubleVector([r, 2 * r, 4 * r])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    else:
        raise ValueError(method)
    mesh.compute_vertex_normals()
    return mesh


def save_cloud(pcd, path: str | Path) -> None:
    o3d.io.write_point_cloud(str(path), pcd)


def save_mesh(mesh, path: str | Path) -> None:
    o3d.io.write_triangle_mesh(str(path), mesh)
