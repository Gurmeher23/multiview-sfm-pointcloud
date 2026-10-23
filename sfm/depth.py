"""Optional monocular depth comparison (MiDaS small via torch hub).

Monocular networks predict *relative inverse depth* (up to unknown scale and shift per image), so
they are compared to SfM depth only after a least-squares scale+shift fit in inverse-depth space.
Everything except :func:`predict_midas` is pure numpy; the network needs ``torch`` + ``timm`` and
a one-off weight download, and the pipeline works without it (``--skip-depth``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


class DepthUnavailableError(RuntimeError):
    pass


def predict_midas(image_bgr: np.ndarray, model_name: str = "MiDaS_small") -> np.ndarray:
    """Relative inverse depth map (H, W), larger = closer. CPU only."""
    try:
        import cv2
        import torch

        model = torch.hub.load("intel-isl/MiDaS", model_name, trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    except Exception as e:  # missing torch/timm, no network, hub changes: all non-fatal
        raise DepthUnavailableError(f"{type(e).__name__}: {e}") from e
    model.eval()
    tf = transforms.small_transform if model_name == "MiDaS_small" else transforms.dpt_transform
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        pred = model(tf(rgb))
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    return pred.numpy()


def sample_map(depth_map: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of a map at pixel coordinates (N, 2)."""
    h, w = depth_map.shape
    x = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    return depth_map[y, x]


def compare_inverse_depth(pred_inv: np.ndarray, sfm_depth: np.ndarray) -> dict[str, float]:
    """Scale/shift-invariant comparison of a relative inverse-depth prediction with SfM depth.

    Fits ``1/z ~ a * pred + b`` by least squares, converts back to depth, and reports
    AbsRel, the scale-invariant log RMSE (Eigen et al.) and Spearman rank correlation.
    """
    inv = 1.0 / sfm_depth
    A = np.column_stack([pred_inv, np.ones_like(pred_inv)])
    (a, b), *_ = np.linalg.lstsq(A, inv, rcond=None)
    fitted = np.maximum(a * pred_inv + b, 1e-6)
    z_hat = 1.0 / fitted
    g = np.log(z_hat) - np.log(sfm_depth)
    return {
        "abs_rel": float(np.mean(np.abs(z_hat - sfm_depth) / sfm_depth)),
        "silog": float(np.sqrt(max(np.mean(g**2) - np.mean(g) ** 2, 0.0))),
        "spearman": float(spearmanr(pred_inv, inv).statistic),
        "n": int(len(sfm_depth)),
    }
