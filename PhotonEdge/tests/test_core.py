# -*- coding: utf-8 -*-
"""Tests for PhotonEdge core modules."""

import sys
from pathlib import Path

# Allow imports from src/
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pytest

from core.optics import dog_kernel_embedded, optical_sim_linear, _fft_kernel_cached
from core.edges import robust_sigma_mad, edges_strict_zero_cross, adaptive_threshold
from core.fusion import fuse_v2, fuse_or, fuse_backbone_with_fine
from core.metrics import edge_metrics_symmetric, gt_edges_from_binary
from core.shapes import SHAPES, NYQUIST_SHAPES, IN_BAND_SHAPES


# ============================================================
# optics.py
# ============================================================

class TestDoGKernel:
    def test_shape(self):
        k = dog_kernel_embedded(128, 21, 1.0, 2.0)
        assert k.shape == (128, 128)
        assert k.dtype == np.float32

    def test_zero_dc(self):
        k = dog_kernel_embedded(128, 21, 1.0, 2.0)
        assert abs(k.sum()) < 1e-6, "DoG kernel must be zero-DC"

    def test_centered(self):
        k = dog_kernel_embedded(128, 21, 1.0, 2.0)
        c = 128 // 2
        assert k[c, c] != 0.0, "Kernel should be non-zero at center"
        assert k[0, 0] == 0.0, "Corners should be zero"


class TestOpticalSim:
    def test_output_shape(self):
        img = np.ones((64, 64), dtype=np.float32)
        kernel = dog_kernel_embedded(64, 11, 0.6, 1.2)
        rng = np.random.default_rng(42)
        Y = optical_sim_linear(img, kernel, 20.0, 0.05, rng)
        assert Y.shape == (64, 64)
        assert Y.dtype == np.float32

    def test_reproducibility(self):
        img = np.ones((64, 64), dtype=np.float32)
        kernel = dog_kernel_embedded(64, 11, 0.6, 1.2)
        Y1 = optical_sim_linear(img, kernel, 20.0, 0.05, np.random.default_rng(42))
        Y2 = optical_sim_linear(img, kernel, 20.0, 0.05, np.random.default_rng(42))
        np.testing.assert_array_equal(Y1, Y2)


class TestFFTCache:
    def test_cache_returns_same(self):
        H1 = _fft_kernel_cached(128, 21, 1.0, 2.0)
        H2 = _fft_kernel_cached(128, 21, 1.0, 2.0)
        assert H1 is H2, "Cached FFT should return same object"


# ============================================================
# edges.py
# ============================================================

class TestRobustSigma:
    def test_gaussian(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1.0, 10000)
        sigma = robust_sigma_mad(x)
        assert 0.9 < sigma < 1.1, f"MAD estimate {sigma} should be near 1.0"

    def test_constant(self):
        x = np.ones(100)
        sigma = robust_sigma_mad(x)
        assert sigma < 1e-6


class TestEdgeDetection:
    def test_detects_edges_from_optical_output(self):
        """Edge detection on a DoG-filtered step should find edges."""
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 32:] = 1.0
        kernel = dog_kernel_embedded(64, 11, 1.0, 2.0)
        rng = np.random.default_rng(0)
        Y = optical_sim_linear(img, kernel, 30.0, 0.0, rng)
        edges = edges_strict_zero_cross(Y, edge_t=2.2, smooth_sigma=0.9, closing=False)
        assert edges.any(), "Should detect edges on DoG-filtered step"

    def test_uniform_no_edges(self):
        img = np.ones((64, 64), dtype=np.float32) * 5.0
        edges = edges_strict_zero_cross(img, edge_t=2.0, smooth_sigma=0.9, closing=False)
        assert not edges.any(), "Uniform image should produce no edges"


class TestAdaptiveThreshold:
    def test_clamping(self):
        Y = np.random.randn(64, 64).astype(np.float32) * 0.001
        t = adaptive_threshold(Y, t_min=1.5, t_max=6.0)
        assert 1.5 <= t <= 6.0


# ============================================================
# fusion.py
# ============================================================

class TestFusion:
    def test_fuse_v2_backbone_preserved(self):
        backbone = np.zeros((64, 64), dtype=bool)
        backbone[30, 10:50] = True
        fine = np.zeros((64, 64), dtype=bool)
        fine[31, 10:50] = True  # overlaps with backbone
        fine[5, 5:20] = True    # gap region

        fused = fuse_v2(fine, backbone, coverage_dilation_px=2, closing=False)
        assert np.all(fused[30, 10:50]), "Backbone edges must be preserved"
        assert fused[5, 5:20].any(), "Fine edges in gaps should be accepted"

    def test_fuse_or(self):
        a = np.zeros((32, 32), dtype=bool)
        b = np.zeros((32, 32), dtype=bool)
        a[10, :] = True
        b[:, 10] = True
        result = fuse_or([a, b])
        assert result[10, 10], "OR fusion should include both"


# ============================================================
# metrics.py
# ============================================================

class TestMetrics:
    def test_perfect_match(self):
        pred = np.zeros((64, 64), dtype=bool)
        pred[30, 10:50] = True
        m = edge_metrics_symmetric(pred, pred.copy(), tol_px=2)
        assert m["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_no_predictions(self):
        pred = np.zeros((64, 64), dtype=bool)
        gt = np.zeros((64, 64), dtype=bool)
        gt[30, 10:50] = True
        m = edge_metrics_symmetric(pred, gt, tol_px=2)
        assert m["f1"] == 0.0

    def test_both_empty(self):
        pred = np.zeros((64, 64), dtype=bool)
        gt = np.zeros((64, 64), dtype=bool)
        m = edge_metrics_symmetric(pred, gt, tol_px=2)
        assert m["f1"] == 1.0


class TestGTEdges:
    def test_step_produces_edges(self):
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 32:] = 1.0
        gt = gt_edges_from_binary(img)
        assert gt.any()

    def test_uniform_no_edges(self):
        img = np.ones((64, 64), dtype=np.float32)
        gt = gt_edges_from_binary(img)
        assert not gt.any()


# ============================================================
# shapes.py
# ============================================================

class TestShapes:
    def test_all_shapes_exist(self):
        assert len(SHAPES) == 8

    def test_in_band_count(self):
        assert len(IN_BAND_SHAPES) == 7

    def test_nyquist_excluded(self):
        assert "checker" in NYQUIST_SHAPES
        assert "checker" not in IN_BAND_SHAPES

    @pytest.mark.parametrize("name", list(SHAPES.keys()))
    def test_shape_output(self, name):
        fn = SHAPES[name]
        img = fn(64)
        assert img.shape == (64, 64)
        assert img.dtype == np.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0 + 1e-6


# ============================================================
# Integration: full single-scale pipeline
# ============================================================

class TestPipeline:
    def test_circle_square_high_snr(self):
        """Full pipeline at high SNR should produce high F1."""
        from core.shapes import make_circle_square
        img = make_circle_square(128)
        gt = gt_edges_from_binary(img)
        kernel = dog_kernel_embedded(128, 21, 1.0, 2.0)
        rng = np.random.default_rng(42)
        Y = optical_sim_linear(img, kernel, 30.0, 0.0, rng)
        edges = edges_strict_zero_cross(Y, edge_t=2.2, smooth_sigma=0.9)
        m = edge_metrics_symmetric(edges, gt, tol_px=2)
        assert m["f1"] > 0.90, f"F1={m['f1']:.3f} should be > 0.90 at high SNR"
