# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania

"""Digital edge extraction for the PhotonEdge pipeline.

Implements robust zero-crossing detection with maximum amplitude gating,
Gaussian smoothing, morphological post-processing, and SNR-adaptive
threshold computation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing


def robust_sigma_mad(x: np.ndarray) -> float:
    """Compute robust noise estimate using Median Absolute Deviation.

    Args:
        x: Input array.

    Returns:
        Estimated standard deviation (σ_MAD = 1.4826 × MAD).
    """
    return 1.4826 * max(float(np.median(np.abs(x - np.median(x)))), 1e-12)


def edges_strict_zero_cross(Y: np.ndarray,
                             edge_t: float = 2.2,
                             smooth_sigma: float = 0.9,
                             closing: bool = True) -> np.ndarray:
    """Extract edges via robust zero-crossing detection with max-amplitude gating.

    Pipeline:
        1. Optional Gaussian smoothing (reduce sub-pixel ZC jitter)
        2. Robust z-score normalization (median/MAD)
        3. 4-neighborhood zero-crossing detection (right + down only)
        4. Maximum amplitude gating (reject crossings below threshold)
        5. Optional morphological closing (connect 1px gaps)

    Args:
        Y: Raw optical detector output (float32 array).
        edge_t: Amplitude threshold in σ units for the max-gate.
        smooth_sigma: Gaussian smoothing σ (0 to disable).
        closing: If True, apply binary closing to connect small gaps.

    Returns:
        Boolean edge map.
    """
    Zs = gaussian_filter(Y, smooth_sigma) if smooth_sigma > 0 else Y

    med = np.median(Zs)
    sig = robust_sigma_mad(Zs)
    Zz = (Zs - med) / (sig + 1e-12)

    sgn = np.sign(Zs)

    # 4-neighborhood zero-crossing: right and down neighbors only
    flip_r = (sgn[:, :-1] * sgn[:, 1:]) < 0
    flip_d = (sgn[:-1, :] * sgn[1:, :]) < 0

    zc = np.zeros_like(Zs, dtype=bool)
    zc[:, :-1] |= flip_r
    zc[:, 1:] |= flip_r
    zc[:-1, :] |= flip_d
    zc[1:, :] |= flip_d

    # Max-amplitude gate: accept only crossings with |z| >= threshold
    edges = zc & (np.abs(Zz) >= edge_t)

    if closing:
        edges = binary_closing(edges, iterations=1)

    return edges


def adaptive_threshold(Y: np.ndarray,
                       slope: float = 0.08,
                       intercept: float = 4.0,
                       t_min: float = 1.5,
                       t_max: float = 6.0) -> float:
    """Compute SNR-adaptive amplitude gate threshold.

    At high SNR, phase drift artifacts produce coherent zero-crossings
    with high amplitude. Raising the gate at high SNR suppresses these
    false positives. At low SNR, lowering the gate preserves weak thin
    features.

    Rule: A_t = slope × SNR_dB + intercept, clamped to [t_min, t_max].

    Args:
        Y: Raw optical output (before normalization).
        slope: SNR coefficient (default: 0.08).
        intercept: Base threshold (default: 4.0).
        t_min: Minimum threshold clamp.
        t_max: Maximum threshold clamp.

    Returns:
        Threshold value for the max-gate.
    """
    signal_power = float(np.median(Y ** 2))
    noise_sigma = 1.4826 * float(np.median(np.abs(Y - np.median(Y))))
    noise_sigma = max(noise_sigma, 1e-12)
    snr_db = 10.0 * np.log10(signal_power / (noise_sigma ** 2 + 1e-12))
    return float(np.clip(slope * snr_db + intercept, t_min, t_max))
