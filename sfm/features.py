"""Feature detection and ratio-test matching (SIFT or ORB)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Features:
    xy: np.ndarray  # (N, 2) float pixel coordinates
    desc: np.ndarray  # (N, D)
    kind: str


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def detect(img: np.ndarray, kind: str = "sift", max_features: int = 4000) -> Features:
    gray = to_gray(img)
    if kind == "sift":
        det = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
    elif kind == "orb":
        det = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError(f"unknown feature type {kind!r}")
    kps, desc = det.detectAndCompute(gray, None)
    if desc is None:
        return Features(np.zeros((0, 2)), np.zeros((0, 1)), kind)
    return Features(np.array([k.pt for k in kps], float), desc, kind)


def match(f1: Features, f2: Features, ratio: float = 0.75, mutual: bool = True) -> np.ndarray:
    """Lowe ratio-test matches, optionally also required to be mutual nearest neighbours.

    Returns an (M, 2) int array of indices ``(i in f1, j in f2)``.
    """
    if len(f1.xy) < 2 or len(f2.xy) < 2:
        return np.zeros((0, 2), int)
    norm = cv2.NORM_L2 if f1.kind == "sift" else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm)

    def one_way(a: Features, b: Features) -> dict[int, int]:
        out = {}
        for pair in bf.knnMatch(a.desc, b.desc, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                out[pair[0].queryIdx] = pair[0].trainIdx
        return out

    fwd = one_way(f1, f2)
    if mutual:
        bwd = one_way(f2, f1)
        fwd = {i: j for i, j in fwd.items() if bwd.get(j) == i}
    return np.array(sorted(fwd.items()), int).reshape(-1, 2)
