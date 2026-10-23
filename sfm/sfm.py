"""Incremental multi-view structure from motion.

Pipeline: features -> pairwise matching + epipolar verification -> feature tracks (union-find)
-> two-view initialisation -> register views by PnP + RANSAC against triangulated tracks ->
triangulate new tracks -> robust sparse bundle adjustment with outlier pruning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np

from .bundle_adjustment import bundle_adjust, project_obs
from .camera import Pose, normalize_pixels
from .features import Features, detect, match
from .geometry import triangulate_multiview
from .two_view import TwoViewResult, estimate_two_view


@dataclass
class SfMConfig:
    feature: str = "sift"
    ratio: float = 0.75
    epipolar_px: float = 1.0
    min_pair_inliers: int = 30
    min_init_angle_deg: float = 2.0
    pnp_px: float = 3.0
    min_pnp_inliers: int = 15
    max_reproj_px: float = 2.0  # accept a triangulated point only below this error
    min_tri_angle_deg: float = 1.5
    prune_px: float = 3.0  # drop observations above this error after BA
    ba_every_view: bool = True
    ba_max_nfev: int = 100
    final_ba: bool = True  # False: skip all BA (baseline for benchmarking)


@dataclass
class SfMResult:
    K: np.ndarray
    poses: dict[int, Pose]
    point_ids: np.ndarray  # track id of each reconstructed point
    points: np.ndarray  # (M, 3)
    tracks: list[dict[int, int]]
    features: list[Features]
    init_pair: tuple[int, int]
    pair_inliers: dict[tuple[int, int], np.ndarray]  # verified (i_kp, j_kp) matches per image pair
    rmse_before_ba: float
    rmse_after_ba: float
    median_before_ba: float
    median_after_ba: float
    n_obs_before: int
    n_obs_after: int
    log: list[str] = field(default_factory=list)

    @property
    def registered(self) -> list[int]:
        return sorted(self.poses)

    def view_samples(self, view: int) -> tuple[np.ndarray, np.ndarray]:
        """Pixels and z-depths of reconstructed points observed in ``view``."""
        row = {int(t): k for k, t in enumerate(self.point_ids)}
        uv, depth = [], []
        for tid, k in row.items():
            if view in self.tracks[tid]:
                uv.append(self.features[view].xy[self.tracks[tid][view]])
                depth.append(self.poses[view].transform(self.points[k])[0, 2])
        return np.array(uv).reshape(-1, 2), np.array(depth)


def build_tracks(
    n_kps: list[int], pair_matches: dict[tuple[int, int], np.ndarray]
) -> list[dict[int, int]]:
    """Connected components of the match graph; components with a repeated image are dropped."""
    offsets = np.concatenate([[0], np.cumsum(n_kps)])
    parent = np.arange(offsets[-1])

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j), m in pair_matches.items():
        for a, b in m:
            ra, rb = find(offsets[i] + a), find(offsets[j] + b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for node in np.unique(
        np.concatenate(
            [offsets[i] + m[:, 0] for (i, _), m in pair_matches.items()]
            + [offsets[j] + m[:, 1] for (_, j), m in pair_matches.items()]
        )
    ):
        groups.setdefault(find(int(node)), []).append(int(node))
    tracks = []
    for nodes in groups.values():
        views = {}
        ok = True
        for n in nodes:
            img = int(np.searchsorted(offsets, n, side="right") - 1)
            if img in views:
                ok = False
                break
            views[img] = int(n - offsets[img])
        if ok and len(views) >= 2:
            tracks.append(views)
    return tracks


class IncrementalSfM:
    def __init__(self, images: list[np.ndarray], K: np.ndarray, cfg: SfMConfig | None = None):
        self.cfg = cfg or SfMConfig()
        self.K = K
        self.images = images
        self.log: list[str] = []
        self.features: list[Features] = []
        self.poses: dict[int, Pose] = {}
        self.xyz: dict[int, np.ndarray] = {}
        self.tracks: list[dict[int, int]] = []

    # -- helpers -----------------------------------------------------------------------------
    def _say(self, msg: str) -> None:
        self.log.append(msg)

    def _uv(self, tid: int, img: int) -> np.ndarray:
        return self.features[img].xy[self.tracks[tid][img]]

    def _observations(self):
        """Flat observation arrays over registered views of tracks that have a 3D point."""
        cams = sorted(self.poses)
        cmap = {c: k for k, c in enumerate(cams)}
        tids = sorted(self.xyz)
        pmap = {t: k for k, t in enumerate(tids)}
        ci, pi, uv, key = [], [], [], []
        for t in tids:
            for img in self.tracks[t]:
                if img in cmap:
                    ci.append(cmap[img])
                    pi.append(pmap[t])
                    uv.append(self._uv(t, img))
                    key.append((t, img))
        return cams, tids, np.array(ci), np.array(pi), np.array(uv), key

    def _rmse(self) -> tuple[float, int, float]:
        cams, tids, ci, pi, uv, _ = self._observations()
        if len(uv) == 0:
            return float("nan"), 0, float("nan")
        rv = np.array([cv2.Rodrigues(self.poses[c].R)[0].ravel() for c in cams])
        tv = np.array([self.poses[c].t for c in cams])
        pts = np.array([self.xyz[t] for t in tids])
        err = project_obs(self.K, rv[ci], tv[ci], pts[pi]) - uv
        e = np.linalg.norm(err, axis=1)
        return float(np.sqrt(np.mean(e**2))), len(uv), float(np.median(e))

    # -- stages --------------------------------------------------------------------------------
    def _match_all(self):
        cfg = self.cfg
        self.features = [detect(im, cfg.feature) for im in self.images]
        self._say("features per view: " + ", ".join(str(len(f.xy)) for f in self.features))
        pair_matches, two_view = {}, {}
        for i, j in combinations(range(len(self.images)), 2):
            m = match(self.features[i], self.features[j], cfg.ratio)
            if len(m) < cfg.min_pair_inliers:
                continue
            res = estimate_two_view(
                self.features[i].xy[m[:, 0]], self.features[j].xy[m[:, 1]], self.K,
                cfg.epipolar_px, max_reproj_px=cfg.max_reproj_px, min_angle_deg=0.0,
            )  # fmt: skip
            if res is None or res.inliers.sum() < cfg.min_pair_inliers:
                continue
            pair_matches[(i, j)] = m[res.inliers]
            two_view[(i, j)] = (m, res)
        self._say(f"verified image pairs: {len(pair_matches)}")
        return pair_matches, two_view

    def _initialise(self, two_view: dict[tuple[int, int], tuple[np.ndarray, TwoViewResult]]):
        best, best_score = None, 0
        for pair, (_, res) in two_view.items():
            score = int(res.valid.sum())
            if res.median_angle_deg >= self.cfg.min_init_angle_deg and score > best_score:
                best, best_score = pair, score
        if best is None:
            raise RuntimeError("no image pair with enough parallax to initialise")
        i, j = best
        res = two_view[best][1]
        self.poses[i] = Pose(np.eye(3), np.zeros(3))
        self.poses[j] = Pose(res.R, res.t)
        self._say(f"initialised from views {i},{j} ({best_score} points, "
                  f"median parallax {res.median_angle_deg:.1f} deg)")  # fmt: skip
        return best

    def _triangulate_new(self) -> int:
        cfg = self.cfg
        added = 0
        for tid, obs in enumerate(self.tracks):
            if tid in self.xyz:
                continue
            views = [v for v in obs if v in self.poses]
            if len(views) < 2:
                continue
            xs = np.array([normalize_pixels(self.K, self._uv(tid, v)) for v in views])
            X = triangulate_multiview([self.poses[v].P for v in views], xs)
            ok, dirs = True, []
            for v in views:
                p = self.poses[v]
                xc = p.transform(X)[0]
                proj = xc[:2] / xc[2] * np.array([self.K[0, 0], self.K[1, 1]]) + self.K[:2, 2]
                if xc[2] <= 0 or np.linalg.norm(proj - self._uv(tid, v)) > cfg.max_reproj_px:
                    ok = False
                    break
                d = p.center - X
                dirs.append(d / np.linalg.norm(d))
            if not ok:
                continue
            cos_min = (np.array(dirs) @ np.array(dirs).T).min()
            if np.degrees(np.arccos(np.clip(cos_min, -1, 1))) < cfg.min_tri_angle_deg:
                continue
            self.xyz[tid] = X
            added += 1
        return added

    def _register_next(self) -> int | None:
        cfg = self.cfg
        best_view, best_n = None, 0
        for v in range(len(self.images)):
            if v in self.poses:
                continue
            n = sum(1 for t, o in enumerate(self.tracks) if t in self.xyz and v in o)
            if n > best_n:
                best_view, best_n = v, n
        if best_view is None or best_n < cfg.min_pnp_inliers:
            return None
        tids = [t for t in sorted(self.xyz) if best_view in self.tracks[t]]
        obj = np.array([self.xyz[t] for t in tids], np.float64)
        img = np.array([self._uv(t, best_view) for t in tids], np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, self.K, None, iterationsCount=1000, reprojectionError=cfg.pnp_px,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )  # fmt: skip
        if not ok or inl is None or len(inl) < cfg.min_pnp_inliers:
            self._say(f"view {best_view}: PnP failed")
            return -best_view - 1
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
        self.poses[best_view] = Pose(cv2.Rodrigues(rvec)[0], tvec.ravel())
        self._say(f"registered view {best_view}: PnP {len(inl)}/{len(tids)} inliers")
        return best_view

    def _bundle(self) -> None:
        cams, tids, ci, pi, uv, key = self._observations()
        if len(uv) == 0:
            return
        rv = np.array([cv2.Rodrigues(self.poses[c].R)[0].ravel() for c in cams])
        tv = np.array([self.poses[c].t for c in cams])
        pts = np.array([self.xyz[t] for t in tids])
        res = bundle_adjust(self.K, rv, tv, pts, ci, pi, uv, fixed_cams=(0,),
                            max_nfev=self.cfg.ba_max_nfev)  # fmt: skip
        for k, c in enumerate(cams):
            self.poses[c] = Pose(cv2.Rodrigues(res.rvecs[k])[0], res.tvecs[k])
        for k, t in enumerate(tids):
            self.xyz[t] = res.points[k]

    def _prune(self) -> int:
        """Remove observations above ``prune_px``; drop points left with fewer than two views."""
        cams, tids, ci, pi, uv, key = self._observations()
        rv = np.array([cv2.Rodrigues(self.poses[c].R)[0].ravel() for c in cams])
        tv = np.array([self.poses[c].t for c in cams])
        pts = np.array([self.xyz[t] for t in tids])
        err = np.linalg.norm(project_obs(self.K, rv[ci], tv[ci], pts[pi]) - uv, axis=1)
        bad = np.flatnonzero(err > self.cfg.prune_px)
        for b in bad:
            t, img = key[b]
            del self.tracks[t][img]
        for t in tids:
            if sum(1 for v in self.tracks[t] if v in self.poses) < 2:
                del self.xyz[t]
        return len(bad)

    # -- driver --------------------------------------------------------------------------------
    def run(self) -> SfMResult:
        pair_matches, two_view = self._match_all()
        self.tracks = build_tracks([len(f.xy) for f in self.features], pair_matches)
        self._say(f"tracks: {len(self.tracks)}")
        init = self._initialise(two_view)
        self._say(f"triangulated {self._triangulate_new()} initial points")
        while True:
            v = self._register_next()
            if v is None:
                break
            if v < 0:  # best remaining candidate could not be registered: stop
                break
            self._say(f"triangulated {self._triangulate_new()} new points")
            if self.cfg.ba_every_view:
                self._bundle()
        before, n_before, med_before = self._rmse()
        self._say(f"before BA: {n_before} obs, RMSE {before:.3f} px (median {med_before:.3f})")
        if self.cfg.final_ba:
            for _ in range(2):  # BA, prune outliers, BA again
                self._bundle()
                if self._prune() == 0:
                    break
            self._bundle()
        after, n_after, med_after = self._rmse()
        self._say(f"after BA: {n_after} obs, RMSE {after:.3f} px (median {med_after:.3f})")
        tids = np.array(sorted(self.xyz), int)
        return SfMResult(
            self.K,
            dict(self.poses),
            tids,
            np.array([self.xyz[t] for t in tids]),
            self.tracks,
            self.features,
            init,
            pair_matches,
            before,
            after,
            med_before,
            med_after,
            n_before,
            n_after,
            self.log,
        )


def run_sfm(images: list[np.ndarray], K: np.ndarray, cfg: SfMConfig | None = None) -> SfMResult:
    return IncrementalSfM(images, K, cfg).run()
