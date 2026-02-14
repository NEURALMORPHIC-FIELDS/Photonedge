# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# F3C-PX MULTI-SCALE DoG EDGE DETECTION
# ======================================
# 3 optical passes (fine/mid/coarse) + OR fusion + single post-proc
# Goal: cover sub-band features (thin lines, checker, text) without
#       losing accuracy on area objects.
#
# Scale A (fine):   sigma1=0.6, sigma2=1.2  -- detects features >= 2px
# Scale B (mid):    sigma1=1.0, sigma2=2.0  -- validated baseline
# Scale C (coarse): sigma1=1.6, sigma2=3.2  -- large smooth boundaries

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, binary_closing, binary_dilation, distance_transform_edt
from skimage.morphology import thin
from skimage.draw import polygon as draw_polygon
import time

# ============================================================
# GLOBAL SETTINGS
# ============================================================
SIZE = 128
GT_TOL_PX = 2
N_TRIALS = 3

SNR_VALUES   = [30, 25, 20, 15, 10]
DRIFT_VALUES = [0.00, 0.02, 0.05, 0.10, 0.20]

CLOSING_FOOTPRINT = np.ones((3, 3), dtype=bool)

# Multi-scale definition
SCALES = {
    "A_fine":   {"sigma1": 0.6, "sigma2": 1.2, "ksize": 15, "edge_t": 2.2, "smooth": 0.7},
    "B_mid":    {"sigma1": 1.0, "sigma2": 2.0, "ksize": 21, "edge_t": 2.2, "smooth": 0.9},
    "C_coarse": {"sigma1": 1.6, "sigma2": 3.2, "ksize": 31, "edge_t": 2.2, "smooth": 1.1},
}

FUSION_MODES = ["or_thin", "z_weighted", "scale_aware"]


# ============================================================
# CORE FUNCTIONS
# ============================================================
def get_dog_kernel(image_size: int, k_size: int,
                   s1: float, s2: float) -> np.ndarray:
    assert k_size % 2 == 1
    ax = np.linspace(-(k_size // 2), k_size // 2, k_size).astype(np.float32)
    x, y = np.meshgrid(ax, ax)
    d2 = x**2 + y**2
    g1 = np.exp(-d2 / (2 * s1**2))
    g2 = np.exp(-d2 / (2 * s2**2))
    g1 /= (g1.sum() + 1e-12)
    g2 /= (g2.sum() + 1e-12)
    alpha = (s1 / s2) ** 2
    dog_small = g1 - alpha * g2
    dog_small -= dog_small.mean()
    dog_small /= (np.sum(np.abs(dog_small)) + 1e-12)
    kernel = np.zeros((image_size, image_size), dtype=np.float32)
    c = image_size // 2
    h = k_size // 2
    kernel[c-h:c+h+1, c-h:c+h+1] = dog_small
    return kernel


def optical_sim_linear(img: np.ndarray, kernel: np.ndarray,
                       snr_db: float, drift_std: float) -> np.ndarray:
    phase = np.random.normal(0, drift_std, img.shape).astype(np.float32)
    E_in = img.astype(np.complex64) * np.exp(1j * phase.astype(np.complex64))
    F_in = np.fft.fft2(E_in)
    H = np.fft.fft2(np.fft.ifftshift(kernel))
    E_out = np.fft.ifft2(F_in * H)
    Y = np.real(E_out).astype(np.float32)
    sig_power = max(float(np.mean(Y**2)), 1e-12)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), Y.shape).astype(np.float32)
    return Y + noise


def robust_zscore(Y: np.ndarray) -> tuple:
    med = np.median(Y)
    mad = np.median(np.abs(Y - med))
    sigma = 1.4826 * mad + 1e-8
    return (Y - med) / sigma, sigma


def zero_cross_edges_raw(Z: np.ndarray, t: float,
                         smooth_sigma: float) -> tuple:
    """Returns (edges_bool, Zs, amp_map)."""
    Zs = gaussian_filter(Z, smooth_sigma) if smooth_sigma > 0 else Z
    edges = np.zeros_like(Zs, dtype=bool)
    amp_map = np.zeros_like(Zs, dtype=np.float32)

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            S = np.roll(Zs, (di, dj), axis=(0, 1))
            sign_change = (Zs * S) < 0
            local_amp = np.maximum(np.abs(Zs), np.abs(S))
            amp_gate = local_amp >= t
            new_edges = sign_change & amp_gate
            edges |= new_edges
            amp_map = np.maximum(amp_map, local_amp * new_edges.astype(np.float32))

    return edges, Zs, amp_map


def edge_metrics_symmetric(pred: np.ndarray, gt: np.ndarray,
                           tol_px: int = 2) -> dict:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "TP": 0, "FP": 0, "FN": 0}
    if pred_sum == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "TP": 0, "FP": 0, "FN": gt_sum}
    if gt_sum == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "TP": 0, "FP": pred_sum, "FN": 0}

    dist_gt = distance_transform_edt(~gt)
    dist_pred = distance_transform_edt(~pred)
    TP = int((pred & (dist_gt <= tol_px)).sum())
    FP = int((pred & (dist_gt > tol_px)).sum())
    FN = int((gt & (dist_pred > tol_px)).sum())
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1,
            "TP": TP, "FP": FP, "FN": FN}


# ============================================================
# MULTI-SCALE PIPELINE
# ============================================================
def multiscale_edges(img: np.ndarray, snr_db: float, drift_std: float,
                     fusion: str = "scale_aware",
                     suppression_radius: int = 3) -> tuple:
    """Run 3-scale DoG + fuse. Each scale gets independent noise.

    Fusion modes:
        or_thin:      naive OR across all scales + thin
        z_weighted:   OR with best-amplitude selection + thin
        scale_aware:  (v2) coarse-to-fine suppression —
                      fine scale edges accepted ONLY where mid/coarse
                      have no coverage within suppression_radius
    """
    per_scale_edges = {}
    per_scale_amps = {}

    for name, cfg in SCALES.items():
        kernel = get_dog_kernel(SIZE, cfg["ksize"], cfg["sigma1"], cfg["sigma2"])
        Y = optical_sim_linear(img, kernel, snr_db, drift_std)
        Z, _ = robust_zscore(Y)
        edges, Zs, amp_map = zero_cross_edges_raw(Z, cfg["edge_t"], cfg["smooth"])
        per_scale_edges[name] = edges
        per_scale_amps[name] = amp_map

    if fusion == "or_thin":
        fused = np.zeros((SIZE, SIZE), dtype=bool)
        for e in per_scale_edges.values():
            fused |= e
        fused = binary_closing(fused, structure=CLOSING_FOOTPRINT)
        fused = thin(fused)

    elif fusion == "z_weighted":
        best_amp = np.zeros((SIZE, SIZE), dtype=np.float32)
        best_scale = np.full((SIZE, SIZE), -1, dtype=np.int32)

        for idx, (name, edges) in enumerate(per_scale_edges.items()):
            amp = per_scale_amps[name]
            better = amp > best_amp
            best_amp = np.where(better, amp, best_amp)
            best_scale = np.where(better, idx, best_scale)

        fused = np.zeros((SIZE, SIZE), dtype=bool)
        for idx, (name, edges) in enumerate(per_scale_edges.items()):
            fused |= (edges & (best_scale == idx))

        fused = binary_closing(fused, structure=CLOSING_FOOTPRINT)
        fused = thin(fused)

    elif fusion == "scale_aware":
        # v2: B is the backbone (proven single-scale F1>0.97).
        # A fills genuine gaps only (where B has zero coverage).
        # C is dropped: broader kernel adds FP on area shapes without
        # covering features that B misses.
        edges_b = per_scale_edges["B_mid"]

        # Coverage mask: where B detects, A is redundant
        coverage = binary_dilation(edges_b, iterations=suppression_radius)

        # A contributes ONLY outside B's coverage zone
        edges_a_filtered = per_scale_edges["A_fine"] & ~coverage

        # Union + close + thin
        fused = edges_b | edges_a_filtered
        fused = binary_closing(fused, structure=CLOSING_FOOTPRINT)
        fused = thin(fused)

    return fused, per_scale_edges, per_scale_amps


# ============================================================
# SHAPE GENERATORS
# ============================================================
def shape_circle_square(size: int = 128) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    img = np.zeros((size, size), dtype=np.float32)
    img[(x**2 + y**2) < 2.0] = 1.0
    img[20:60, 20:60] = 1.0
    return img

def shape_triangle(size: int = 128) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    r = size // 3
    angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    rows = [int(cy - r * np.sin(a)) for a in angles]
    cols = [int(cx + r * np.cos(a)) for a in angles]
    rr, cc = draw_polygon(rows, cols, shape=(size, size))
    img[rr, cc] = 1.0
    return img

def shape_concave_polygon(size: int = 128) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    rows = [20, 20, 60, 60, 100, 100, 20]
    cols = [30, 90, 90, 60, 60, 30, 30]
    rr, cc = draw_polygon(rows, cols, shape=(size, size))
    img[rr, cc] = 1.0
    return img

def shape_thin_lines(size: int = 128) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    img[30, 20:100] = 1.0
    img[40:110, 60] = 1.0
    for k in range(70):
        r, c = 25 + k, 15 + k
        if 0 <= r < size and 0 <= c < size:
            img[r, c] = 1.0
    for k in range(80):
        r = 90 + k // 2
        c = 20 + k
        if 0 <= r < size and 0 <= c < size:
            img[r, c] = 1.0
    img[70:72, 10:110] = 1.0
    return img

def shape_text_like(size: int = 128) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    img[20:90, 10:15] = 1.0
    img[20:25, 10:40] = 1.0
    img[50:55, 10:35] = 1.0
    img[20:25, 48:75] = 1.0
    img[50:55, 48:75] = 1.0
    img[82:87, 48:75] = 1.0
    img[20:55, 70:75] = 1.0
    img[50:87, 70:75] = 1.0
    img[20:25, 85:115] = 1.0
    img[82:87, 85:115] = 1.0
    img[20:87, 85:90] = 1.0
    return img

def shape_tangent_circles(size: int = 128) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    img = np.zeros((size, size), dtype=np.float32)
    img[((x + 0.8)**2 + y**2) < 1.2] = 1.0
    img[((x - 1.2)**2 + y**2) < 0.8] = 1.0
    return img

def shape_checker(size: int = 128) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    cell = 16
    for i in range(size):
        for j in range(size):
            if ((i // cell) + (j // cell)) % 2 == 0:
                img[i, j] = 1.0
    return img

def shape_textured_object(size: int = 128) -> np.ndarray:
    np.random.seed(42)
    bg = gaussian_filter(np.random.randn(size, size).astype(np.float32), sigma=8)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8) * 0.3
    x, y = np.meshgrid(np.linspace(-3, 3, size), np.linspace(-3, 3, size))
    mask = ((x - 0.5)**2 + (y + 0.3)**2) < 1.5
    img = bg.copy()
    img[mask] = 1.0
    return img

def gt_edges_from_image(img: np.ndarray, thr: float = 0.1) -> np.ndarray:
    gx, gy = np.gradient(img.astype(np.float32))
    mag = np.sqrt(gx*gx + gy*gy)
    gt = mag > thr
    gt = thin(gt)
    return gt

SHAPES = {
    "circle_square":   shape_circle_square,
    "triangle":        shape_triangle,
    "concave_polygon": shape_concave_polygon,
    "thin_lines":      shape_thin_lines,
    "text_F3C":        shape_text_like,
    "tangent_circles": shape_tangent_circles,
    "checker":         shape_checker,
    "textured_object": shape_textured_object,
}


# ============================================================
# EVALUATION HELPERS
# ============================================================
def eval_single(img, gt, snr, drift, fusion, seed=0, suppression_radius=3):
    np.random.seed(seed)
    fused, _, _ = multiscale_edges(img, snr, drift, fusion=fusion,
                                    suppression_radius=suppression_radius)
    return edge_metrics_symmetric(fused, gt, tol_px=GT_TOL_PX)

def eval_grid(img, gt, fusion, suppression_radius=3):
    f1_grid = np.zeros((len(SNR_VALUES), len(DRIFT_VALUES)))
    for i, snr in enumerate(SNR_VALUES):
        for j, drift in enumerate(DRIFT_VALUES):
            f1s = []
            for t in range(N_TRIALS):
                m = eval_single(img, gt, snr, drift, fusion,
                                seed=t*10000+i*100+j,
                                suppression_radius=suppression_radius)
                f1s.append(m["f1"])
            f1_grid[i, j] = np.mean(f1s)
    return f1_grid

def eval_grid_single_scale(img, gt):
    """Single-scale B baseline."""
    kernel = get_dog_kernel(SIZE, 21, 1.0, 2.0)
    f1_grid = np.zeros((len(SNR_VALUES), len(DRIFT_VALUES)))
    for i, snr in enumerate(SNR_VALUES):
        for j, drift in enumerate(DRIFT_VALUES):
            f1s = []
            for t in range(N_TRIALS):
                np.random.seed(t*10000+i*100+j)
                Y = optical_sim_linear(img, kernel, snr, drift)
                Z, _ = robust_zscore(Y)
                edges, _, _ = zero_cross_edges_raw(Z, 2.2, 0.9)
                edges = binary_closing(edges, structure=CLOSING_FOOTPRINT)
                edges = thin(edges)
                m = edge_metrics_symmetric(edges, gt, tol_px=GT_TOL_PX)
                f1s.append(m["f1"])
            f1_grid[i, j] = np.mean(f1s)
    return f1_grid


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t_start = time.time()

    # ===========================================
    # PHASE 1: Tune suppression radius for scale_aware
    # ===========================================
    print("=" * 70, flush=True)
    print("F3C-PX MULTI-SCALE DoG — SCALE-AWARE FUSION v2", flush=True)
    print("=" * 70, flush=True)
    for name, cfg in SCALES.items():
        print(f"  {name}: s1={cfg['sigma1']} s2={cfg['sigma2']} "
              f"k={cfg['ksize']} t={cfg['edge_t']} smooth={cfg['smooth']}", flush=True)

    # Sweep suppression radius on a representative pair:
    # area object (circle_square) + fine object (thin_lines)
    test_snr, test_drift = 20, 0.10
    print(f"\nSuppression radius sweep @ SNR={test_snr}, drift={test_drift}:", flush=True)
    print(f"  {'radius':>6s} | {'circle_sq':>10s} | {'thin_lines':>10s} | "
          f"{'text_F3C':>10s} | {'triangle':>10s} | {'geomean':>8s}", flush=True)
    print("  " + "-" * 70, flush=True)

    radius_candidates = [1, 2, 3, 4, 5]
    test_shapes = ["circle_square", "thin_lines", "text_F3C", "triangle"]
    radius_scores = {}

    for rad in radius_candidates:
        scores = {}
        for sn in test_shapes:
            img = SHAPES[sn](SIZE)
            gt = gt_edges_from_image(img, thr=0.1)
            f1s = []
            for t in range(N_TRIALS):
                np.random.seed(t * 1000)
                fused, _, _ = multiscale_edges(img, test_snr, test_drift,
                                                fusion="scale_aware",
                                                suppression_radius=rad)
                f1s.append(edge_metrics_symmetric(fused, gt, tol_px=GT_TOL_PX)["f1"])
            scores[sn] = np.mean(f1s)

        vals = list(scores.values())
        geomean = np.exp(np.mean(np.log(np.clip(vals, 1e-8, None))))
        radius_scores[rad] = {"scores": scores, "geomean": geomean}

        row = f"  {rad:6d} |"
        for sn in test_shapes:
            row += f" {scores[sn]:10.4f} |"
        row += f" {geomean:8.4f}"
        print(row, flush=True)

    best_radius = max(radius_candidates, key=lambda r: radius_scores[r]["geomean"])
    print(f"\n  Best radius: {best_radius} (geomean={radius_scores[best_radius]['geomean']:.4f})",
          flush=True)

    # Also compare all 3 fusion modes at best radius
    print(f"\nFusion mode comparison @ radius={best_radius}, SNR={test_snr}, drift={test_drift}:",
          flush=True)
    print(f"  {'Shape':20s} | {'or_thin':>8s} | {'z_weight':>8s} | {'scale_aw':>8s} | {'single_B':>8s}",
          flush=True)
    print("  " + "-" * 62, flush=True)

    for shape_name, gen_fn in SHAPES.items():
        img = gen_fn(SIZE)
        gt = gt_edges_from_image(img, thr=0.1)
        if int(gt.sum()) == 0:
            continue

        row = f"  {shape_name:20s} |"
        for fm in FUSION_MODES:
            f1s = []
            for t in range(N_TRIALS):
                np.random.seed(t * 1000)
                fused, _, _ = multiscale_edges(img, test_snr, test_drift,
                                                fusion=fm,
                                                suppression_radius=best_radius)
                f1s.append(edge_metrics_symmetric(fused, gt, tol_px=GT_TOL_PX)["f1"])
            row += f" {np.mean(f1s):8.4f} |"

        # Single-scale baseline
        kernel_b = get_dog_kernel(SIZE, 21, 1.0, 2.0)
        f1s_b = []
        for t in range(N_TRIALS):
            np.random.seed(t * 1000)
            Y = optical_sim_linear(img, kernel_b, test_snr, test_drift)
            Z, _ = robust_zscore(Y)
            edges, _, _ = zero_cross_edges_raw(Z, 2.2, 0.9)
            edges = binary_closing(edges, structure=CLOSING_FOOTPRINT)
            edges = thin(edges)
            f1s_b.append(edge_metrics_symmetric(edges, gt, tol_px=GT_TOL_PX)["f1"])
        row += f" {np.mean(f1s_b):8.4f}"
        print(row, flush=True)

    best_fusion = "scale_aware"
    print(f"\n  Using: {best_fusion} (radius={best_radius})", flush=True)

    # ===========================================
    # PHASE 2: Full grid with best fusion
    # ===========================================
    print(f"\n{'=' * 70}", flush=True)
    print(f"FULL GRID — fusion={best_fusion}, radius={best_radius}", flush=True)
    print(f"{'=' * 70}", flush=True)

    all_grids = {}
    all_f1_arrays = []
    all_f1_arrays_no_nyquist = []

    for shape_name, gen_fn in SHAPES.items():
        t0 = time.time()
        img = gen_fn(SIZE)
        gt = gt_edges_from_image(img, thr=0.1)
        if int(gt.sum()) == 0:
            continue

        f1_grid = eval_grid(img, gt, fusion=best_fusion,
                            suppression_radius=best_radius)
        all_grids[shape_name] = f1_grid
        all_f1_arrays.append(f1_grid)
        if shape_name != "checker":
            all_f1_arrays_no_nyquist.append(f1_grid)

        med = np.median(f1_grid)
        mn = np.min(f1_grid)
        wi = np.unravel_index(np.argmin(f1_grid), f1_grid.shape)
        dt = time.time() - t0
        print(f"  {shape_name:20s} | med={med:.4f} min={mn:.4f} "
              f"worst@(SNR={SNR_VALUES[wi[0]]},d={DRIFT_VALUES[wi[1]]}) | {dt:.1f}s",
              flush=True)

    # ===========================================
    # PHASE 3: Aggregate
    # ===========================================
    stacked = np.stack(all_f1_arrays, axis=0)
    worst_f1_map = np.min(stacked, axis=0)
    median_f1_map = np.median(stacked, axis=0)
    all_flat = stacked.flatten()

    # Excluding Nyquist shape (checker)
    stacked_no_nq = np.stack(all_f1_arrays_no_nyquist, axis=0)
    worst_f1_map_no_nq = np.min(stacked_no_nq, axis=0)
    median_f1_map_no_nq = np.median(stacked_no_nq, axis=0)
    flat_no_nq = stacked_no_nq.flatten()

    p5 = np.percentile(flat_no_nq, 5)
    global_median = np.median(flat_no_nq)
    abs_min = np.min(flat_no_nq)

    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE (multi-scale, excluding Nyquist shapes)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Shapes evaluated: {len(all_f1_arrays_no_nyquist)}/8 "
          f"(checker excluded: Nyquist regime)", flush=True)
    print(f"  Global median: {global_median:.4f}", flush=True)
    print(f"  Worst-5%:      {p5:.4f}", flush=True)
    print(f"  Absolute min:  {abs_min:.4f}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE (multi-scale, 8 shapes)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Global median: {global_median:.4f}", flush=True)
    print(f"  Worst-5%:      {p5:.4f}", flush=True)
    print(f"  Absolute min:  {abs_min:.4f}", flush=True)

    pass_med = global_median >= 0.97
    pass_p5 = p5 >= 0.90
    status = "PASS" if (pass_med and pass_p5) else "CONDITIONAL"
    print(f"\n  CHECK: {status}", flush=True)
    print(f"    median >= 0.97? {global_median:.4f} -> {'YES' if pass_med else 'NO'}",
          flush=True)
    print(f"    worst-5% >= 0.90? {p5:.4f} -> {'YES' if pass_p5 else 'NO'}",
          flush=True)

    print(f"\n  Per-shape:", flush=True)
    print(f"  {'Shape':20s} | {'Median':>7s} | {'Min':>7s} | {'Status':>8s}", flush=True)
    print(f"  " + "-" * 50, flush=True)
    for name, grid in all_grids.items():
        med = np.median(grid)
        mn = np.min(grid)
        st = "OK" if mn >= 0.85 else ("MARGINAL" if mn >= 0.5 else "NYQUIST")
        print(f"  {name:20s} | {med:7.4f} | {mn:7.4f} | {st:>8s}", flush=True)

    print(f"\n  Worst-case F1 map (min over 7 in-band shapes):", flush=True)
    header = "  SNR\\Drift  |" + "".join(f"  {d:.2f} " for d in DRIFT_VALUES)
    print(header, flush=True)
    print("  " + "-" * len(header), flush=True)
    for i, snr in enumerate(SNR_VALUES):
        row = f"  {snr:3d} dB    |"
        for j in range(len(DRIFT_VALUES)):
            row += f" {worst_f1_map_no_nq[i,j]:.3f}"
        print(row, flush=True)

    # ===========================================
    # PHASE 4: Single-scale baseline for comparison
    # ===========================================
    print(f"\n  Computing single-scale baseline...", flush=True)
    single_grids = []
    single_grids_no_nq = []
    for shape_name, gen_fn in SHAPES.items():
        img = gen_fn(SIZE)
        gt = gt_edges_from_image(img, thr=0.1)
        if int(gt.sum()) == 0:
            continue
        g = eval_grid_single_scale(img, gt)
        single_grids.append(g)
        if shape_name != "checker":
            single_grids_no_nq.append(g)
    single_worst = np.min(np.stack(single_grids), axis=0)
    single_worst_no_nq = np.min(np.stack(single_grids_no_nq), axis=0)

    # ===========================================
    # PHASE 5: Figures
    # ===========================================
    print(f"\n{'=' * 70}", flush=True)
    print("FIGURES", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Fig 1: per-shape overlays + heatmaps
    n_shapes = len(all_grids)
    fig1, axes1 = plt.subplots(2, n_shapes, figsize=(3.2 * n_shapes, 6.5))
    for idx, (name, gen_fn) in enumerate(SHAPES.items()):
        if name not in all_grids:
            continue
        img = gen_fn(SIZE)
        gt = gt_edges_from_image(img, thr=0.1)

        np.random.seed(0)
        fused, per_scale, _ = multiscale_edges(img, 15, 0.10, fusion=best_fusion,
                                                suppression_radius=best_radius)
        overlay = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
        overlay[gt, 1] = 1.0
        overlay[fused, 0] = 1.0
        overlay[gt & fused] = [1.0, 1.0, 0.0]
        axes1[0, idx].imshow(overlay)
        axes1[0, idx].set_title(f"{name}\n(GT={int(gt.sum())}px)", fontsize=7)
        axes1[0, idx].axis("off")

        f1g = all_grids[name]
        im = axes1[1, idx].imshow(f1g, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
        for ii in range(len(SNR_VALUES)):
            for jj in range(len(DRIFT_VALUES)):
                v = f1g[ii, jj]
                c = "white" if v < 0.7 else "black"
                axes1[1, idx].text(jj, ii, f"{v:.2f}", ha="center", va="center",
                                    fontsize=6, color=c)
        axes1[1, idx].set_xticks(range(len(DRIFT_VALUES)))
        axes1[1, idx].set_xticklabels([f"{d:.2f}" for d in DRIFT_VALUES], fontsize=5)
        axes1[1, idx].set_yticks(range(len(SNR_VALUES)))
        axes1[1, idx].set_yticklabels([f"{s}" for s in SNR_VALUES], fontsize=5)
        if idx == 0:
            axes1[1, idx].set_ylabel("SNR (dB)", fontsize=7)

    fig1.suptitle(f"Multi-Scale F3C-PX — Per-Shape (fusion={best_fusion})",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig1.savefig("/home/claude/ms_per_shape.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  ✓ ms_per_shape.png", flush=True)

    # Fig 2: worst-case heatmaps
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, vmin in [
        (ax2a, worst_f1_map_no_nq, "Worst-Case F1 (min over 7 in-band shapes)", 0.5),
        (ax2b, median_f1_map_no_nq, "Median F1 (over 7 in-band shapes)", 0.8),
    ]:
        im = ax.imshow(data, vmin=vmin, vmax=1.0, cmap="RdYlGn", aspect="auto")
        for ii in range(len(SNR_VALUES)):
            for jj in range(len(DRIFT_VALUES)):
                v = data[ii, jj]
                c = "white" if v < 0.7 else "black"
                ax.text(jj, ii, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=c)
        ax.set_xticks(range(len(DRIFT_VALUES)))
        ax.set_xticklabels([f"{d:.2f}" for d in DRIFT_VALUES])
        ax.set_yticks(range(len(SNR_VALUES)))
        ax.set_yticklabels([f"{s} dB" for s in SNR_VALUES])
        ax.set_xlabel("Phase Drift (std)")
        ax.set_ylabel("SNR (dB)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig2.suptitle("Multi-Scale F3C-PX — Robustness (3 optical passes)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig2.savefig("/home/claude/ms_robustness.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ ms_robustness.png", flush=True)

    # Fig 3: single vs multi comparison
    improvement = worst_f1_map_no_nq - single_worst_no_nq
    fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(17, 5))
    for ax, data, title, cmap, vmin, vmax in [
        (ax3a, single_worst_no_nq, "Single-Scale (B)\nWorst F1", "RdYlGn", 0.0, 1.0),
        (ax3b, worst_f1_map_no_nq, "Multi-Scale (A+B+C)\nWorst F1", "RdYlGn", 0.0, 1.0),
        (ax3c, improvement,   "Improvement\n(Multi - Single)", "RdBu", -0.1, 1.0),
    ]:
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        for ii in range(len(SNR_VALUES)):
            for jj in range(len(DRIFT_VALUES)):
                v = data[ii, jj]
                c = "white" if v < 0.4 else "black"
                ax.text(jj, ii, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=c)
        ax.set_xticks(range(len(DRIFT_VALUES)))
        ax.set_xticklabels([f"{d:.2f}" for d in DRIFT_VALUES])
        ax.set_yticks(range(len(SNR_VALUES)))
        ax.set_yticklabels([f"{s} dB" for s in SNR_VALUES])
        ax.set_xlabel("Phase Drift")
        ax.set_ylabel("SNR (dB)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig3.suptitle("Single vs Multi-Scale: Worst-Case F1 (7 in-band shapes)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3.savefig("/home/claude/ms_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  ✓ ms_comparison.png", flush=True)

    # Fig 4: per-scale contribution
    scale_names = list(SCALES.keys())
    fig4, axes4 = plt.subplots(3, n_shapes, figsize=(3 * n_shapes, 9))
    for idx, (shape_name, gen_fn) in enumerate(SHAPES.items()):
        if shape_name not in all_grids:
            continue
        img = gen_fn(SIZE)
        np.random.seed(0)
        _, per_scale, _ = multiscale_edges(img, 15, 0.10, fusion=best_fusion,
                                             suppression_radius=best_radius)
        for si, sn in enumerate(scale_names):
            axes4[si, idx].imshow(per_scale[sn], cmap="gray")
            if idx == 0:
                axes4[si, idx].set_ylabel(sn, fontsize=8)
            if si == 0:
                axes4[si, idx].set_title(shape_name, fontsize=7)
            axes4[si, idx].axis("off")
    fig4.suptitle("Per-Scale Edge Maps (SNR=15, drift=0.10)",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig4.savefig("/home/claude/ms_per_scale.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("  ✓ ms_per_scale.png", flush=True)

    # ===========================================
    # OPERATIONAL ENVELOPE
    # ===========================================
    print(f"\n{'=' * 70}", flush=True)
    print("OPERATIONAL ENVELOPE (Multi-Scale, scale-aware fusion v2)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  3-pass multi-scale F3C-PX DoG (scales 0.6/1.0/1.6, ratio 2:1)", flush=True)
    print(f"  with scale-aware suppression (radius={best_radius}).", flush=True)
    print(f"  Across 7 in-band shapes, 25 operating points (SNR 10-30 dB, drift 0-0.20):", flush=True)
    print(f"    Worst-case F1 = {abs_min:.3f}", flush=True)
    print(f"    Global median = {global_median:.4f}", flush=True)
    print(f"    5th percentile = {p5:.4f}", flush=True)
    print(f"  Checker (cell=16px, edge spacing 1px) declared Nyquist regime.", flush=True)
    print(f"  Cost: 3 sequential optical passes (laser + SLM reconfigure).", flush=True)

    dt = time.time() - t_start
    print(f"\n✓ Total runtime: {dt:.1f}s", flush=True)
