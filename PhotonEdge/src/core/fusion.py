# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania

"""Multi-scale edge fusion for the PhotonEdge pipeline.

Implements Fusion v2 (fine-only-where-needed): backbone scales provide
primary coverage, and the fine scale contributes only in gaps outside
the backbone's dilated coverage zone.
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_closing


def fuse_v2(edges_fine: np.ndarray,
            edges_backbone: np.ndarray,
            coverage_dilation_px: int = 3,
            closing: bool = True) -> np.ndarray:
    """Fusion v2: fine-only-where-needed.

    Strategy:
        1. Backbone edge map defines primary coverage.
        2. Dilate backbone by coverage_dilation_px to create suppression mask.
        3. Accept fine-scale edges only OUTSIDE the suppression mask.
        4. Merge and optionally close small gaps.

    This prevents the fine scale from injecting false positives on smooth
    boundaries already covered by the backbone, while preserving fine-scale
    contributions for thin features invisible to coarser scales.

    Args:
        edges_fine: Edge map from fine scale (e.g., Scale A, σ₁=0.6).
        edges_backbone: Edge map from backbone scale(s) (e.g., Scale B ∪ C).
        coverage_dilation_px: Suppression radius around backbone edges.
        closing: If True, apply morphological closing to the fused result.

    Returns:
        Fused boolean edge map.
    """
    coverage = binary_dilation(edges_backbone, iterations=coverage_dilation_px)
    fine_accepted = edges_fine & ~coverage
    fused = edges_backbone | fine_accepted

    if closing:
        fused = binary_closing(fused, iterations=1)

    return fused


def fuse_or(edge_maps: list) -> np.ndarray:
    """Naive OR fusion across multiple edge maps.

    Simple union of all edge maps. Not recommended for production use
    due to false positive accumulation across scales.

    Args:
        edge_maps: List of boolean edge maps (same shape).

    Returns:
        Union of all edge maps.
    """
    result = np.zeros_like(edge_maps[0], dtype=bool)
    for em in edge_maps:
        result |= em
    return result


def fuse_backbone_with_fine(edges_A: np.ndarray,
                            edges_B: np.ndarray,
                            edges_C: np.ndarray,
                            coverage_dilation_px: int = 3,
                            closing: bool = True) -> np.ndarray:
    """Three-scale fusion: BC backbone + A gap-fill.

    Convenience wrapper for the standard 3-scale configuration:
    - Scale B (σ₁=1.0) + Scale C (σ₁=1.6) = backbone
    - Scale A (σ₁=0.6) = fine detail, accepted only outside BC coverage

    Args:
        edges_A: Fine-scale edge map.
        edges_B: Mid-scale edge map (primary backbone).
        edges_C: Coarse-scale edge map (secondary backbone).
        coverage_dilation_px: Suppression radius.
        closing: Apply morphological closing.

    Returns:
        Fused boolean edge map.
    """
    backbone = edges_B | edges_C
    return fuse_v2(edges_A, backbone, coverage_dilation_px, closing)
