# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania

"""Evaluation metrics for the PhotonEdge pipeline.

Implements symmetric distance-transform matching — a physically
appropriate metric for analog optical edge detection that accommodates
sub-pixel jitter, edge thickness, and DoG zero-crossing width without
penalizing inherent analog behavior.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def edge_metrics_symmetric(pred: np.ndarray,
                           gt: np.ndarray,
                           tol_px: int = 2) -> dict:
    """Compute symmetric edge detection metrics using distance transforms.

    For each predicted edge pixel, checks if any ground-truth edge pixel
    is within tol_px (bidirectional). This tolerates sub-pixel jitter and
    the natural 2-3px width of DoG zero-crossings.

    Args:
        pred: Predicted edge map (boolean).
        gt: Ground-truth edge map (boolean).
        tol_px: Tolerance in pixels for matching (default: 2).

    Returns:
        Dictionary with keys: 'p' (precision), 'r' (recall), 'f1',
        and optionally 'TP', 'FP', 'FN' counts.
    """
    ps = int(pred.sum())
    gs = int(gt.sum())

    if ps == 0 and gs == 0:
        return {"p": 1.0, "r": 1.0, "f1": 1.0, "TP": 0, "FP": 0, "FN": 0}
    if ps == 0:
        return {"p": 0.0, "r": 0.0, "f1": 0.0, "TP": 0, "FP": 0, "FN": gs}
    if gs == 0:
        return {"p": 0.0, "r": 0.0, "f1": 0.0, "TP": 0, "FP": ps, "FN": 0}

    # Distance from each pixel to nearest GT edge / nearest predicted edge
    dist_to_gt = distance_transform_edt(~gt)
    dist_to_pred = distance_transform_edt(~pred)

    TP = int((pred & (dist_to_gt <= tol_px)).sum())
    FP = int((pred & (dist_to_gt > tol_px)).sum())
    FN = int((gt & (dist_to_pred > tol_px)).sum())

    p = TP / (TP + FP + 1e-8)
    r = TP / (TP + FN + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)

    return {"p": p, "r": r, "f1": f1, "TP": TP, "FP": FP, "FN": FN}


def gt_edges_from_binary(img: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Generate ground-truth edges from a binary image using gradient magnitude.

    Does NOT apply thinning — the natural 2-3px edge width matches the
    optical zero-crossing width, enabling direct comparison without
    artificial precision loss from skeletonization.

    Args:
        img: Binary input image (float, values in [0, 1]).
        threshold: Gradient magnitude threshold for edge classification.

    Returns:
        Boolean edge map.
    """
    gx, gy = np.gradient(img.astype(np.float32))
    mag = np.sqrt(gx * gx + gy * gy)
    return mag > threshold
