# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania

"""Test shape generators for PhotonEdge validation.

Provides 8 shape classes spanning area features, thin features,
periodic patterns, and textured regions — covering the full
bandwidth range from in-band to Nyquist regime.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# ============================================================
# PRIMITIVES
# ============================================================

def _make_canvas(size: int = 128) -> np.ndarray:
    """Create a blank canvas."""
    return np.zeros((size, size), dtype=np.float32)


def _add_circle(img: np.ndarray, cx: int, cy: int, r: int,
                val: float = 1.0) -> None:
    """Draw a filled circle."""
    yy, xx = np.indices(img.shape)
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = val


def _add_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              val: float = 1.0) -> None:
    """Draw a filled rectangle."""
    img[y0:y1, x0:x1] = val


def _add_triangle(img: np.ndarray, pts: list, val: float = 1.0) -> None:
    """Draw a filled triangle using barycentric coordinates."""
    yy, xx = np.indices(img.shape)
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    x2, y2 = pts[2]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2) + 1e-12
    a = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / den
    b = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / den
    c = 1 - a - b
    img[(a >= 0) & (b >= 0) & (c >= 0)] = val


def _add_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              thickness: int = 1, val: float = 1.0) -> None:
    """Draw a line with specified thickness."""
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        yb0 = max(0, yi - thickness)
        yb1 = min(img.shape[0], yi + thickness + 1)
        xb0 = max(0, xi - thickness)
        xb1 = min(img.shape[1], xi + thickness + 1)
        img[yb0:yb1, xb0:xb1] = val


# ============================================================
# SHAPE GENERATORS
# ============================================================

def make_circle_square(s: int = 128) -> np.ndarray:
    """Circle + square: basic area features with curved and straight edges."""
    img = _make_canvas(s)
    _add_rect(img, 18, 18, 60, 60)
    _add_circle(img, 76, 70, 34)
    return np.clip(img, 0, 1)


def make_triangle(s: int = 128) -> np.ndarray:
    """Triangle: angled edges with varying orientation."""
    img = _make_canvas(s)
    _add_triangle(img, [(64, 20), (20, 100), (108, 100)])
    return img


def make_concave_polygon(s: int = 128) -> np.ndarray:
    """L-shaped concave polygon: interior corners and concavity."""
    img = _make_canvas(s)
    _add_rect(img, 28, 20, 92, 44)
    _add_rect(img, 28, 44, 52, 108)
    _add_rect(img, 52, 84, 92, 108)
    return img


def make_thin_lines(s: int = 128) -> np.ndarray:
    """Thin lines: 1px strokes at various angles (sub-band challenge)."""
    img = _make_canvas(s)
    _add_line(img, 10, 30, 118, 30, thickness=1)
    _add_line(img, 10, 60, 118, 100, thickness=1)
    _add_line(img, 30, 10, 30, 118, thickness=1)
    _add_line(img, 70, 10, 100, 118, thickness=1)
    return img


def make_text_F3C(s: int = 128) -> np.ndarray:
    """Text-like pattern 'F3C': thin rectangular strokes (sub-band challenge)."""
    img = _make_canvas(s)
    # F
    _add_rect(img, 12, 28, 20, 100)
    _add_rect(img, 20, 28, 50, 36)
    _add_rect(img, 20, 56, 42, 64)
    # 3
    _add_rect(img, 56, 28, 84, 36)
    _add_rect(img, 56, 56, 84, 64)
    _add_rect(img, 56, 92, 84, 100)
    _add_rect(img, 76, 36, 84, 56)
    _add_rect(img, 76, 64, 84, 92)
    # C
    _add_rect(img, 92, 28, 118, 36)
    _add_rect(img, 92, 92, 118, 100)
    _add_rect(img, 92, 36, 100, 92)
    return img


def make_tangent_circles(s: int = 128) -> np.ndarray:
    """Two tangent circles: overlapping boundaries (geometry challenge)."""
    img = _make_canvas(s)
    _add_circle(img, 50, 64, 22)
    _add_circle(img, 78, 64, 22)
    return img


def make_checker(s: int = 128, cell: int = 16) -> np.ndarray:
    """Checkerboard: periodic pattern with 1px edge spacing (Nyquist regime)."""
    img = _make_canvas(s)
    for y in range(0, s, cell):
        for x in range(0, s, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                _add_rect(img, x, y, min(s, x + cell), min(s, y + cell))
    return img


def make_textured_object(s: int = 128) -> np.ndarray:
    """Circle on textured background: edge detection with background clutter."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.12, (s, s)).astype(np.float32)
    base = gaussian_filter(base, 2.0)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12) * 0.25
    img = base.copy()
    _add_circle(img, 72, 68, 26)
    return np.clip(img, 0, 1)


# ============================================================
# REGISTRY
# ============================================================

SHAPES: dict = {
    "circle_square": make_circle_square,
    "triangle": make_triangle,
    "concave_polygon": make_concave_polygon,
    "thin_lines": make_thin_lines,
    "text_F3C": make_text_F3C,
    "tangent_circles": make_tangent_circles,
    "checker": make_checker,
    "textured_object": make_textured_object,
}

#: Shapes excluded from evaluation due to Nyquist band-limit
NYQUIST_SHAPES: set = {"checker"}

#: In-band shapes used for engineering envelope evaluation
IN_BAND_SHAPES: list = [k for k in SHAPES if k not in NYQUIST_SHAPES]
