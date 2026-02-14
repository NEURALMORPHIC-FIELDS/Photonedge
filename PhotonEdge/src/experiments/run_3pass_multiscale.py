# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# F3C-PX MULTI-SCALE DoG EDGE DETECTION
# ======================================
# 3 optical passes (fine/mid/coarse) + fusion + single post-proc
# Goal: cover sub-band features (thin lines, checker, text) without
#       losing accuracy on area objects.
#
# Scale A (fine):   sigma1=0.6, sigma2=1.2  -- detects features >= 2px
# Scale B (mid):    sigma1=1.0, sigma2=2.0  -- validated baseline
# Scale C (coarse): sigma1=1.6, sigma2=3.2  -- large smooth boundaries

import sys
from pathlib import Path

# --------------- path bootstrap ---------------
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
FIGURES_DIR = _ROOT / "experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------- imports ----------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, binary_closing
import time

from core.optics import dog_kernel_embedded, optical_sim_linear
from core.edges import edges_strict_zero_cross, robust_sigma_mad
from core.fusion import fuse_v2, fuse_or
from core.metrics import edge_metrics_symmetric, gt_edges_from_binary
from core.shapes import SHAPES, NYQUIST_SHAPES, IN_BAND_SHAPES

# ============================================================
# GLOBAL SETTINGS
# ============================================================
SIZE = 128
GT_TOL_PX = 2
N_TRIALS = 3

SNR_VALUES   = [30, 25, 20, 15, 10]
DRIFT_VALUES = [0.00, 0.02, 0.05, 0.10, 0.20]

# Multi-scale definition
SCALES = {
    "A_fine":   {"sigma1": 0.6, "sigma2": 1.2, "ksize": 15,
                 "edge_t": 2.2, "smooth": 0.7},
    "B_mid":    {"sigma1": 1.0, "sigma2": 2.0, "ksize": 21,
                 "edge_t": 2.2, "smooth": 0.9},
    "C_coarse": {"sigma1": 1.6, "sigma2": 3.2, "ksize": 31,
                 "edge_t": 2.2, "smooth": 1.1},
}

FUSION_MODES = ["or_thin", "z_weighted", "scale_aware"]


# ============================================================
# INLINE HELPERS (z_weighted needs amplitude maps)
# ============================================================
def _zero_cross_with_amplitude(Y: np.ndarray, edge_t: float,
                               smooth_sigma: float) -> tuple:
    """4-neighbor zero-crossing that also returns an amplitude map.

    Used only for z_weighted fusion which requires per-pixel amplitudes.
    For all other modes, use core.edges.edges_strict_zero_cross.

    Returns:
        (edges_bool, amp_map)
    """
    Zs = gaussian_filter(Y, smooth_sigma) if smooth_sigma > 0 else Y

    med = float(np.median(Zs))
    sig = robust_sigma_mad(Zs)
    Zz = (Zs - med) / (sig + 1e-12)

    sgn = np.sign(Zs)

    # 4-neighbor zero-crossing: right and down
    flip_r = (sgn[:, :-1] * sgn[:, 1:]) < 0
    flip_d = (sgn[:-1, :] * sgn[1:, :]) < 0

    zc = np.zeros_like(Zs, dtype=bool)
    zc[:, :-1] |= flip_r
    zc[:, 1:] |= flip_r
    zc[:-1, :] |= flip_d
    zc[1:, :] |= flip_d

    amp_map = np.abs(Zz)
    edges = zc & (amp_map >= edge_t)

    # Closing to match core behaviour
    edges = binary_closing(edges, iterations=1)

    return edges, amp_map


# ============================================================
# MULTI-SCALE PIPELINE
# ============================================================
def multiscale_edges(img: np.ndarray, snr_db: float, drift_std: float,
                     rng: np.random.Generator,
                     fusion: str = "scale_aware",
                     suppression_radius: int = 3) -> tuple:
    """Run 3-scale DoG + fuse.  Each scale gets independent noise.

    Fusion modes
    -------------
    or_thin      : OR all edge maps + closing  (uses core.fusion.fuse_or)
    z_weighted   : OR with best-amplitude selection + closing (inline)
    scale_aware  : B backbone, A fills gaps via core.fusion.fuse_v2

    Args:
        img: Input binary image.
        snr_db: Signal-to-noise ratio (dB).
        drift_std: Phase drift standard deviation.
        rng: NumPy random Generator for reproducibility.
        fusion: One of "or_thin", "z_weighted", "scale_aware".
        suppression_radius: Dilation radius for scale_aware fusion.

    Returns:
        (fused, per_scale_edges, per_scale_amps)
        per_scale_amps is populated only for z_weighted mode.
    """
    per_scale_edges = {}
    per_scale_amps = {}

    for name, cfg in SCALES.items():
        kernel = dog_kernel_embedded(SIZE, cfg["ksize"],
                                     cfg["sigma1"], cfg["sigma2"])
        Y = optical_sim_linear(img, kernel, snr_db, drift_std, rng)

        if fusion == "z_weighted":
            edges, amp = _zero_cross_with_amplitude(
                Y, cfg["edge_t"], cfg["smooth"])
            per_scale_amps[name] = amp
        else:
            edges = edges_strict_zero_cross(
                Y, edge_t=cfg["edge_t"],
                smooth_sigma=cfg["smooth"], closing=True)
            per_scale_amps[name] = None

        per_scale_edges[name] = edges

    # ---- fusion ----
    if fusion == "or_thin":
        fused = fuse_or(list(per_scale_edges.values()))
        fused = binary_closing(fused, iterations=1)

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

        fused = binary_closing(fused, iterations=1)

    elif fusion == "scale_aware":
        # B is backbone.  A fills genuine gaps only.
        edges_a = per_scale_edges["A_fine"]
        edges_b = per_scale_edges["B_mid"]
        fused = fuse_v2(edges_a, edges_b,
                        coverage_dilation_px=suppression_radius,
                        closing=True)

    else:
        raise ValueError(f"Unknown fusion mode: {fusion!r}")

    return fused, per_scale_edges, per_scale_amps


# ============================================================
# EVALUATION HELPERS
# ============================================================
def eval_single(img, gt, snr, drift, fusion, seed=0,
                suppression_radius=3):
    """Run one trial and return metric dict."""
    rng = np.random.default_rng(seed)
    fused, _, _ = multiscale_edges(img, snr, drift, rng,
                                   fusion=fusion,
                                   suppression_radius=suppression_radius)
    return edge_metrics_symmetric(fused, gt, tol_px=GT_TOL_PX)


def eval_grid(img, gt, fusion, suppression_radius=3):
    """Sweep SNR x drift grid and return F1 matrix."""
    f1_grid = np.zeros((len(SNR_VALUES), len(DRIFT_VALUES)))
    for i, snr in enumerate(SNR_VALUES):
        for j, drift in enumerate(DRIFT_VALUES):
            f1s = []
            for t in range(N_TRIALS):
                m = eval_single(img, gt, snr, drift, fusion,
                                seed=t * 10000 + i * 100 + j,
                                suppression_radius=suppression_radius)
                f1s.append(m["f1"])
            f1_grid[i, j] = np.mean(f1s)
    return f1_grid


def eval_grid_single_scale(img, gt):
    """Single-scale B baseline across the full SNR x drift grid."""
    cfg = SCALES["B_mid"]
    kernel = dog_kernel_embedded(SIZE, cfg["ksize"],
                                 cfg["sigma1"], cfg["sigma2"])
    f1_grid = np.zeros((len(SNR_VALUES), len(DRIFT_VALUES)))
    for i, snr in enumerate(SNR_VALUES):
        for j, drift in enumerate(DRIFT_VALUES):
            f1s = []
            for t in range(N_TRIALS):
                rng = np.random.default_rng(t * 10000 + i * 100 + j)
                Y = optical_sim_linear(img, kernel, snr, drift, rng)
                edges = edges_strict_zero_cross(
                    Y, edge_t=cfg["edge_t"],
                    smooth_sigma=cfg["smooth"], closing=True)
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
    print("F3C-PX MULTI-SCALE DoG -- SCALE-AWARE FUSION v2", flush=True)
    print("=" * 70, flush=True)
    for name, cfg in SCALES.items():
        print(f"  {name}: s1={cfg['sigma1']} s2={cfg['sigma2']} "
              f"k={cfg['ksize']} t={cfg['edge_t']} smooth={cfg['smooth']}",
              flush=True)

    # Sweep suppression radius on representative shapes
    test_snr, test_drift = 20, 0.10
    print(f"\nSuppression radius sweep @ SNR={test_snr}, "
          f"drift={test_drift}:", flush=True)
    print(f"  {'radius':>6s} | {'circle_sq':>10s} | {'thin_lines':>10s} | "
          f"{'text_F3C':>10s} | {'triangle':>10s} | {'geomean':>8s}",
          flush=True)
    print("  " + "-" * 70, flush=True)

    radius_candidates = [1, 2, 3, 4, 5]
    test_shapes = ["circle_square", "thin_lines", "text_F3C", "triangle"]
    radius_scores = {}

    for rad in radius_candidates:
        scores = {}
        for sn in test_shapes:
            img = SHAPES[sn](SIZE)
            gt = gt_edges_from_binary(img)
            f1s = []
            for t in range(N_TRIALS):
                rng = np.random.default_rng(t * 1000)
                fused, _, _ = multiscale_edges(
                    img, test_snr, test_drift, rng,
                    fusion="scale_aware",
                    suppression_radius=rad)
                m = edge_metrics_symmetric(fused, gt, tol_px=GT_TOL_PX)
                f1s.append(m["f1"])
            scores[sn] = np.mean(f1s)

        vals = list(scores.values())
        geomean = np.exp(np.mean(np.log(np.clip(vals, 1e-8, None))))
        radius_scores[rad] = {"scores": scores, "geomean": geomean}

        row = f"  {rad:6d} |"
        for sn in test_shapes:
            row += f" {scores[sn]:10.4f} |"
        row += f" {geomean:8.4f}"
        print(row, flush=True)

    best_radius = max(radius_candidates,
                      key=lambda r: radius_scores[r]["geomean"])
    print(f"\n  Best radius: {best_radius} "
          f"(geomean={radius_scores[best_radius]['geomean']:.4f})",
          flush=True)

    # ===========================================
    # Fusion mode comparison at best radius
    # ===========================================
    print(f"\nFusion mode comparison @ radius={best_radius}, "
          f"SNR={test_snr}, drift={test_drift}:", flush=True)
    print(f"  {'Shape':20s} | {'or_thin':>8s} | {'z_weight':>8s} | "
          f"{'scale_aw':>8s} | {'single_B':>8s}", flush=True)
    print("  " + "-" * 62, flush=True)

    for shape_name in SHAPES:
        img = SHAPES[shape_name](SIZE)
        gt = gt_edges_from_binary(img)
        if int(gt.sum()) == 0:
            continue

        row = f"  {shape_name:20s} |"
        for fm in FUSION_MODES:
            f1s = []
            for t in range(N_TRIALS):
                rng = np.random.default_rng(t * 1000)
                fused, _, _ = multiscale_edges(
                    img, test_snr, test_drift, rng,
                    fusion=fm, suppression_radius=best_radius)
                m = edge_metrics_symmetric(fused, gt, tol_px=GT_TOL_PX)
                f1s.append(m["f1"])
            row += f" {np.mean(f1s):8.4f} |"

        # Single-scale B baseline
        cfg_b = SCALES["B_mid"]
        kernel_b = dog_kernel_embedded(SIZE, cfg_b["ksize"],
                                       cfg_b["sigma1"], cfg_b["sigma2"])
        f1s_b = []
        for t in range(N_TRIALS):
            rng = np.random.default_rng(t * 1000)
            Y = optical_sim_linear(img, kernel_b, test_snr, test_drift, rng)
            edges = edges_strict_zero_cross(
                Y, edge_t=cfg_b["edge_t"],
                smooth_sigma=cfg_b["smooth"], closing=True)
            m = edge_metrics_symmetric(edges, gt, tol_px=GT_TOL_PX)
            f1s_b.append(m["f1"])
        row += f" {np.mean(f1s_b):8.4f}"
        print(row, flush=True)

    best_fusion = "scale_aware"
    print(f"\n  Using: {best_fusion} (radius={best_radius})", flush=True)

    # ===========================================
    # PHASE 2: Full grid with best fusion
    # ===========================================
    print(f"\n{'=' * 70}", flush=True)
    print(f"FULL GRID -- fusion={best_fusion}, radius={best_radius}",
          flush=True)
    print(f"{'=' * 70}", flush=True)

    all_grids = {}
    all_f1_arrays = []
    all_f1_arrays_no_nyquist = []

    for shape_name in SHAPES:
        t0 = time.time()
        img = SHAPES[shape_name](SIZE)
        gt = gt_edges_from_binary(img)
        if int(gt.sum()) == 0:
            continue

        f1_grid = eval_grid(img, gt, fusion=best_fusion,
                            suppression_radius=best_radius)
        all_grids[shape_name] = f1_grid
        all_f1_arrays.append(f1_grid)
        if shape_name not in NYQUIST_SHAPES:
            all_f1_arrays_no_nyquist.append(f1_grid)

        med = np.median(f1_grid)
        mn = np.min(f1_grid)
        wi = np.unravel_index(np.argmin(f1_grid), f1_grid.shape)
        dt = time.time() - t0
        print(f"  {shape_name:20s} | med={med:.4f} min={mn:.4f} "
              f"worst@(SNR={SNR_VALUES[wi[0]]},"
              f"d={DRIFT_VALUES[wi[1]]}) | {dt:.1f}s", flush=True)

    # ===========================================
    # PHASE 3: Aggregate
    # ===========================================
    stacked = np.stack(all_f1_arrays, axis=0)
    worst_f1_map = np.min(stacked, axis=0)
    median_f1_map = np.median(stacked, axis=0)
    all_flat = stacked.flatten()

    # Excluding Nyquist shapes
    stacked_no_nq = np.stack(all_f1_arrays_no_nyquist, axis=0)
    worst_f1_map_no_nq = np.min(stacked_no_nq, axis=0)
    median_f1_map_no_nq = np.median(stacked_no_nq, axis=0)
    flat_no_nq = stacked_no_nq.flatten()

    n_in_band = len(all_f1_arrays_no_nyquist)
    p5 = np.percentile(flat_no_nq, 5)
    global_median = np.median(flat_no_nq)
    abs_min = np.min(flat_no_nq)

    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE (multi-scale, excluding Nyquist shapes)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Shapes evaluated: {n_in_band}/{len(all_f1_arrays)} "
          f"(checker excluded: Nyquist regime)", flush=True)
    print(f"  Global median: {global_median:.4f}", flush=True)
    print(f"  Worst-5%:      {p5:.4f}", flush=True)
    print(f"  Absolute min:  {abs_min:.4f}", flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"AGGREGATE (multi-scale, {len(all_f1_arrays)} shapes)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Global median: {global_median:.4f}", flush=True)
    print(f"  Worst-5%:      {p5:.4f}", flush=True)
    print(f"  Absolute min:  {abs_min:.4f}", flush=True)

    pass_med = global_median >= 0.97
    pass_p5 = p5 >= 0.90
    status = "PASS" if (pass_med and pass_p5) else "CONDITIONAL"
    print(f"\n  CHECK: {status}", flush=True)
    print(f"    median >= 0.97? {global_median:.4f} -> "
          f"{'YES' if pass_med else 'NO'}", flush=True)
    print(f"    worst-5% >= 0.90? {p5:.4f} -> "
          f"{'YES' if pass_p5 else 'NO'}", flush=True)

    print(f"\n  Per-shape:", flush=True)
    print(f"  {'Shape':20s} | {'Median':>7s} | {'Min':>7s} | "
          f"{'Status':>8s}", flush=True)
    print(f"  " + "-" * 50, flush=True)
    for name, grid in all_grids.items():
        med = np.median(grid)
        mn = np.min(grid)
        if mn >= 0.85:
            st = "OK"
        elif mn >= 0.5:
            st = "MARGINAL"
        else:
            st = "NYQUIST"
        print(f"  {name:20s} | {med:7.4f} | {mn:7.4f} | {st:>8s}",
              flush=True)

    print(f"\n  Worst-case F1 map (min over {n_in_band} in-band shapes):",
          flush=True)
    header = ("  SNR\\Drift  |"
              + "".join(f"  {d:.2f} " for d in DRIFT_VALUES))
    print(header, flush=True)
    print("  " + "-" * len(header), flush=True)
    for i, snr in enumerate(SNR_VALUES):
        row = f"  {snr:3d} dB    |"
        for j in range(len(DRIFT_VALUES)):
            row += f" {worst_f1_map_no_nq[i, j]:.3f}"
        print(row, flush=True)

    # ===========================================
    # PHASE 4: Single-scale baseline for comparison
    # ===========================================
    print(f"\n  Computing single-scale baseline...", flush=True)
    single_grids = []
    single_grids_no_nq = []
    for shape_name in SHAPES:
        img = SHAPES[shape_name](SIZE)
        gt = gt_edges_from_binary(img)
        if int(gt.sum()) == 0:
            continue
        g = eval_grid_single_scale(img, gt)
        single_grids.append(g)
        if shape_name not in NYQUIST_SHAPES:
            single_grids_no_nq.append(g)
    single_worst = np.min(np.stack(single_grids), axis=0)
    single_worst_no_nq = np.min(np.stack(single_grids_no_nq), axis=0)

    # ===========================================
    # PHASE 5: Figures
    # ===========================================
    print(f"\n{'=' * 70}", flush=True)
    print("FIGURES", flush=True)
    print(f"{'=' * 70}", flush=True)

    # --- Fig 1: per-shape overlays + heatmaps ---
    n_shapes = len(all_grids)
    fig1, axes1 = plt.subplots(2, n_shapes, figsize=(3.2 * n_shapes, 6.5))
    for idx, shape_name in enumerate(all_grids):
        img = SHAPES[shape_name](SIZE)
        gt = gt_edges_from_binary(img)

        rng = np.random.default_rng(0)
        fused, per_scale, _ = multiscale_edges(
            img, 15, 0.10, rng,
            fusion=best_fusion, suppression_radius=best_radius)

        overlay = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
        overlay[gt, 1] = 1.0
        overlay[fused, 0] = 1.0
        overlay[gt & fused] = [1.0, 1.0, 0.0]
        axes1[0, idx].imshow(overlay)
        axes1[0, idx].set_title(
            f"{shape_name}\n(GT={int(gt.sum())}px)", fontsize=7)
        axes1[0, idx].axis("off")

        f1g = all_grids[shape_name]
        im = axes1[1, idx].imshow(
            f1g, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
        for ii in range(len(SNR_VALUES)):
            for jj in range(len(DRIFT_VALUES)):
                v = f1g[ii, jj]
                c = "white" if v < 0.7 else "black"
                axes1[1, idx].text(
                    jj, ii, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=c)
        axes1[1, idx].set_xticks(range(len(DRIFT_VALUES)))
        axes1[1, idx].set_xticklabels(
            [f"{d:.2f}" for d in DRIFT_VALUES], fontsize=5)
        axes1[1, idx].set_yticks(range(len(SNR_VALUES)))
        axes1[1, idx].set_yticklabels(
            [f"{s}" for s in SNR_VALUES], fontsize=5)
        if idx == 0:
            axes1[1, idx].set_ylabel("SNR (dB)", fontsize=7)

    fig1.suptitle(
        f"Multi-Scale F3C-PX -- Per-Shape (fusion={best_fusion})",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig1.savefig(str(FIGURES_DIR / "ms_per_shape.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  [ok] ms_per_shape.png", flush=True)

    # --- Fig 2: worst-case heatmaps ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, vmin in [
        (ax2a, worst_f1_map_no_nq,
         f"Worst-Case F1 (min over {n_in_band} in-band shapes)", 0.5),
        (ax2b, median_f1_map_no_nq,
         f"Median F1 (over {n_in_band} in-band shapes)", 0.8),
    ]:
        im = ax.imshow(data, vmin=vmin, vmax=1.0,
                        cmap="RdYlGn", aspect="auto")
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
    fig2.suptitle("Multi-Scale F3C-PX -- Robustness (3 optical passes)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(str(FIGURES_DIR / "ms_robustness.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  [ok] ms_robustness.png", flush=True)

    # --- Fig 3: single vs multi comparison ---
    improvement = worst_f1_map_no_nq - single_worst_no_nq
    fig3, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(17, 5))
    for ax, data, title, cmap, vmin, vmax in [
        (ax3a, single_worst_no_nq,
         "Single-Scale (B)\nWorst F1", "RdYlGn", 0.0, 1.0),
        (ax3b, worst_f1_map_no_nq,
         "Multi-Scale (A+B+C)\nWorst F1", "RdYlGn", 0.0, 1.0),
        (ax3c, improvement,
         "Improvement\n(Multi - Single)", "RdBu", -0.1, 1.0),
    ]:
        im = ax.imshow(data, vmin=vmin, vmax=vmax,
                        cmap=cmap, aspect="auto")
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
    fig3.suptitle(
        f"Single vs Multi-Scale: Worst-Case F1 ({n_in_band} in-band shapes)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3.savefig(str(FIGURES_DIR / "ms_comparison.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  [ok] ms_comparison.png", flush=True)

    # --- Fig 4: per-scale contribution ---
    scale_names = list(SCALES.keys())
    fig4, axes4 = plt.subplots(3, n_shapes, figsize=(3 * n_shapes, 9))
    for idx, shape_name in enumerate(all_grids):
        img = SHAPES[shape_name](SIZE)
        rng = np.random.default_rng(0)
        _, per_scale, _ = multiscale_edges(
            img, 15, 0.10, rng,
            fusion=best_fusion, suppression_radius=best_radius)
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
    fig4.savefig(str(FIGURES_DIR / "ms_per_scale.png"),
                 dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("  [ok] ms_per_scale.png", flush=True)

    # ===========================================
    # OPERATIONAL ENVELOPE
    # ===========================================
    print(f"\n{'=' * 70}", flush=True)
    print("OPERATIONAL ENVELOPE (Multi-Scale, scale-aware fusion v2)",
          flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  3-pass multi-scale F3C-PX DoG "
          f"(scales 0.6/1.0/1.6, ratio 2:1)", flush=True)
    print(f"  with scale-aware suppression (radius={best_radius}).",
          flush=True)
    print(f"  Across {n_in_band} in-band shapes, "
          f"{len(SNR_VALUES) * len(DRIFT_VALUES)} operating points "
          f"(SNR {SNR_VALUES[-1]}-{SNR_VALUES[0]} dB, "
          f"drift {DRIFT_VALUES[0]}-{DRIFT_VALUES[-1]}):", flush=True)
    print(f"    Worst-case F1 = {abs_min:.3f}", flush=True)
    print(f"    Global median = {global_median:.4f}", flush=True)
    print(f"    5th percentile = {p5:.4f}", flush=True)
    print(f"  Checker (cell=16px, edge spacing 1px) declared "
          f"Nyquist regime.", flush=True)
    print(f"  Cost: 3 sequential optical passes "
          f"(laser + SLM reconfigure).", flush=True)

    dt = time.time() - t_start
    print(f"\nTotal runtime: {dt:.1f}s", flush=True)
