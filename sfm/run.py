"""Command line entry point: ``python -m sfm.run --synthetic`` or ``--images DIR``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import cloud, depth, viz
from .camera import Intrinsics, normalize_pixels
from .evaluate import align_points, evaluate_structure, evaluate_trajectory
from .features import detect, match
from .geometry import (
    direction_angle_deg,
    ransac_essential,
    recover_pose,
    rotation_angle_deg,
)
from .sfm import SfMConfig, SfMResult, run_sfm
from .synthetic import make_room_views
from .two_view import estimate_two_view

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_images(folder: Path, max_side: int = 1280, focal: float | None = None):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
    if len(paths) < 3:
        raise SystemExit(f"need at least 3 images in {folder}, found {len(paths)}")
    images = [cv2.imread(str(p)) for p in paths]
    h, w = images[0].shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        images = [
            cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for im in images
        ]
    h, w = images[0].shape[:2]
    f = focal * scale if focal else None
    return images, Intrinsics.from_focal(w, h, f)


def two_view_benchmark(images, intr, gt_poses, pair) -> dict:
    """OpenCV vs the numpy 8-point reference on one pair, against the ground-truth relative pose."""
    i, j = pair
    K = intr.K
    fi, fj = detect(images[i]), detect(images[j])
    m = match(fi, fj)
    p1, p2 = fi.xy[m[:, 0]], fj.xy[m[:, 1]]
    Ri, ti, Rj, tj = gt_poses[i].R, gt_poses[i].t, gt_poses[j].R, gt_poses[j].t
    R_gt = Rj @ Ri.T
    t_gt = tj - R_gt @ ti
    out = {"pair": [i, j], "n_matches": int(len(m))}
    cv = estimate_two_view(p1, p2, K)
    out["opencv"] = {
        "inliers": int(cv.inliers.sum()),
        "rot_err_deg": rotation_angle_deg(cv.R, R_gt),
        "trans_dir_err_deg": direction_angle_deg(cv.t, t_gt),
        "reproj_err_px_median": float(np.median(cv.reproj_err[cv.valid])),
    }
    x1, x2 = normalize_pixels(K, p1), normalize_pixels(K, p2)
    E, mask = ransac_essential(x1, x2, threshold=1.0 / K[0, 0])
    R, t, _, _ = recover_pose(E, x1[mask], x2[mask])
    out["numpy_8pt_ransac"] = {
        "inliers": int(mask.sum()),
        "rot_err_deg": rotation_angle_deg(R, R_gt),
        "trans_dir_err_deg": direction_angle_deg(t, t_gt),
    }
    return out


def process_cloud(res: SfMResult, pts: np.ndarray, out: Path, skip_mesh: bool) -> dict:
    """Open3D: outlier removal, voxel downsampling, normals, plane RANSAC, mesh, export."""
    centers = np.array([p.center for p in res.poses.values()])
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    pcd = cloud.to_o3d(pts)
    clean, kept = cloud.remove_outliers(pcd)
    down = cloud.voxel_downsample(clean, diag / 200)
    stats = {
        "points_raw": len(pts),
        "after_outlier_removal": len(kept),
        "after_voxel": len(down.points),
    }
    cloud.estimate_normals(down, radius=diag / 25, camera_centers=centers)
    planes = cloud.segment_planes(down, k=3, distance=diag / 400)
    stats["planes"] = [
        {
            "normal": np.round(m[:3], 3).tolist(),
            "offset": round(float(m[3]), 3),
            "inliers": int(len(i)),
        }
        for m, i in planes
    ]
    cloud.save_cloud(down, out / "cloud.ply")
    if not skip_mesh:
        mesh = cloud.reconstruct_mesh(down, "poisson", depth=7)
        cloud.save_mesh(mesh, out / "mesh.ply")
        cloud.save_mesh(mesh, out / "mesh.obj")
        stats["mesh"] = {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles)}
    return stats


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Multi-view SfM + point-cloud pipeline")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="render a textured synthetic room")
    src.add_argument(
        "--images", type=Path, help="folder of images (unknown intrinsics, see --focal)"
    )
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--focal", type=float, default=None, help="focal length in pixels (--images)")
    ap.add_argument("--feature", choices=["sift", "orb"], default="sift")
    ap.add_argument(
        "--no-incremental-ba",
        action="store_true",
        help="only run bundle adjustment at the end (default: after every view too)",
    )
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--skip-depth", action="store_true", help="skip the optional MiDaS comparison")
    ap.add_argument("--skip-mesh", action="store_true")
    ap.add_argument("--write-images", type=Path, help="(--synthetic) also save the rendered images")
    args = ap.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    scene = gt_poses = None
    if args.synthetic:
        scene, intr, gt_poses, images, _ = make_room_views(args.views, args.seed)
        if args.write_images:
            args.write_images.mkdir(parents=True, exist_ok=True)
            for k, im in enumerate(images):
                cv2.imwrite(str(args.write_images / f"view_{k:02d}.png"), im)
    else:
        images, intr = load_images(args.images, focal=args.focal)

    cfg = SfMConfig(feature=args.feature, ba_every_view=not args.no_incremental_ba)
    res = run_sfm(images, intr.K, cfg)
    metrics: dict = {
        "views": len(images),
        "registered": len(res.registered),
        "points": int(len(res.points)),
        "reproj_rmse_px": {"before_ba": res.rmse_before_ba, "after_ba": res.rmse_after_ba},
        "reproj_median_px": {"before_ba": res.median_before_ba, "after_ba": res.median_after_ba},
        "observations": {"before_ba": res.n_obs_before, "after_ba": res.n_obs_after},
    }
    centers = np.array([res.poses[i].center for i in res.registered])
    pts = res.points
    if scene is not None:
        traj = evaluate_trajectory(res.poses, gt_poses)
        pts = align_points(res.points, traj.sim3)
        centers = align_points(centers, traj.sim3)
        st = evaluate_structure(pts, scene)
        metrics["trajectory"] = {
            "ate_rmse_m": traj.ate_rmse,
            "rot_err_mean_deg": traj.rot_err_mean_deg,
            "rot_err_max_deg": traj.rot_err_max_deg,
            "relative_rot_err_mean_deg": traj.rel_rot_err_mean_deg,
            "sim3_scale": traj.scale,
        }
        metrics["structure_vs_gt_m"] = {"rmse": st.rmse, "median": st.median, "p90": st.p90}
        metrics["two_view"] = two_view_benchmark(images, intr, gt_poses, (0, 1))
        gt_c = np.array([gt_poses[i].center for i in res.registered])
        viz.plot_trajectory(centers, gt_c, out / "trajectory.png")
    metrics["cloud"] = process_cloud(res, pts, out, args.skip_mesh)

    i, j = res.init_pair
    m = res.pair_inliers[(i, j)]
    viz.plot_matches(images[i], images[j], res.features[i].xy[m[:, 0]], res.features[j].xy[m[:, 1]],
                     out / "matches.png")  # fmt: skip
    viz.plot_cloud(pts, centers, out / "pointcloud.png")

    if args.skip_depth:
        metrics["depth"] = "skipped"
    else:
        v = res.registered[0]
        uv, z = res.view_samples(v)
        try:
            pred = depth.predict_midas(images[v])
            metrics["depth"] = {
                "view": v,
                **depth.compare_inverse_depth(depth.sample_map(pred, uv), z),
            }
        except depth.DepthUnavailableError as e:
            metrics["depth"] = f"unavailable ({e})"

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n".join(res.log))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
