"""Synthetic benchmark: several seeds x BA strategies. Prints a markdown table.

``python -m sfm.benchmark --seeds 3``
"""

from __future__ import annotations

import argparse

import numpy as np

from .evaluate import align_points, evaluate_structure, evaluate_trajectory
from .sfm import SfMConfig, run_sfm
from .synthetic import make_room_views

STRATEGIES = {
    "incremental only (no BA)": SfMConfig(ba_every_view=False, final_ba=False),
    "final BA only": SfMConfig(ba_every_view=False, ba_max_nfev=100),
    "BA after every view + final": SfMConfig(ba_every_view=True, ba_max_nfev=100),
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--views", type=int, default=8)
    args = ap.parse_args(argv)
    rows = {k: [] for k in STRATEGIES}
    for seed in range(args.seeds):
        scene, intr, gt, images, _ = make_room_views(args.views, seed)
        for name, cfg in STRATEGIES.items():
            res = run_sfm(images, intr.K, cfg)
            traj = evaluate_trajectory(res.poses, gt)
            st = evaluate_structure(align_points(res.points, traj.sim3), scene)
            rows[name].append(
                (res.rmse_after_ba, traj.ate_rmse, traj.rel_rot_err_mean_deg, st.median, st.rmse)
            )
    print(f"seeds={args.seeds}, views={args.views}, 640x480, SIFT; mean over seeds\n")
    print("| strategy | reproj RMSE (px) | ATE (m) | rel. rot. err (deg) | structure median (m) |")
    print("|---|---|---|---|---|")
    for name, vals in rows.items():
        m = np.mean(vals, axis=0)
        print(f"| {name} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.3f} |")


if __name__ == "__main__":
    main()
