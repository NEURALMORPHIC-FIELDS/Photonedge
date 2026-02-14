# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# F3C-PX 2-PASS MULTI-SCALE: B-backbone + A-gap-fill
# ====================================================
# Architecture:
#   Pass 1 (B mid):  sigma1=1.0, sigma2=2.0, t=2.2  — area features
#   Pass 2 (A fine): sigma1=0.6, sigma2=1.2, t=SWEEP — sub-band features
#   Fusion: B ∪ (A \ dilate(B, r=1))  →  close  →  thin
#
# Key insight: A's threshold must be higher than B's because at high SNR,
# drift artifacts have amplitude proportional to signal → need stronger gate
# to suppress false edges in smooth regions while keeping thin features.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import (gaussian_filter, binary_closing, binary_dilation,
                           distance_transform_edt)
from skimage.morphology import thin
from skimage.draw import polygon as draw_polygon
import time

SIZE = 128
GT_TOL_PX = 2
N_TRIALS = 5  # more trials for stable results
CLOSING_FOOTPRINT = np.ones((3, 3), dtype=bool)

SNR_VALUES   = [30, 25, 20, 15, 10]
DRIFT_VALUES = [0.00, 0.02, 0.05, 0.10, 0.20]

# Scale B (backbone) - fixed, validated
B_CFG = {"sigma1": 1.0, "sigma2": 2.0, "ksize": 21, "edge_t": 2.2, "smooth": 0.9}

# Scale A (fine) - threshold will be swept
A_CFG_BASE = {"sigma1": 0.6, "sigma2": 1.2, "ksize": 15, "smooth": 0.7}
A_THRESHOLD_SWEEP = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

SUPPRESSION_RADIUS = 1  # how far from B edges to suppress A


# ============================================================
# CORE (identical to validated pipeline)
# ============================================================
def get_dog_kernel(image_size: int, k_size: int, s1: float, s2: float) -> np.ndarray:
    assert k_size % 2 == 1
    ax = np.linspace(-(k_size // 2), k_size // 2, k_size).astype(np.float32)
    x, y = np.meshgrid(ax, ax)
    d2 = x**2 + y**2
    g1 = np.exp(-d2 / (2 * s1**2)); g2 = np.exp(-d2 / (2 * s2**2))
    g1 /= (g1.sum() + 1e-12); g2 /= (g2.sum() + 1e-12)
    alpha = (s1 / s2) ** 2
    dog = g1 - alpha * g2; dog -= dog.mean(); dog /= (np.sum(np.abs(dog)) + 1e-12)
    kernel = np.zeros((image_size, image_size), dtype=np.float32)
    c, h = image_size // 2, k_size // 2
    kernel[c-h:c+h+1, c-h:c+h+1] = dog
    return kernel

def optical_sim(img: np.ndarray, kernel: np.ndarray,
                snr_db: float, drift_std: float) -> np.ndarray:
    phase = np.random.normal(0, drift_std, img.shape).astype(np.float32)
    E_in = img.astype(np.complex64) * np.exp(1j * phase.astype(np.complex64))
    H = np.fft.fft2(np.fft.ifftshift(kernel))
    E_out = np.fft.ifft2(np.fft.fft2(E_in) * H)
    Y = np.real(E_out).astype(np.float32)
    sig_p = max(float(np.mean(Y**2)), 1e-12)
    noise_p = sig_p / (10.0 ** (snr_db / 10.0))
    return Y + np.random.normal(0, np.sqrt(noise_p), Y.shape).astype(np.float32)

def robust_zscore(Y: np.ndarray) -> np.ndarray:
    med = np.median(Y)
    sigma = 1.4826 * np.median(np.abs(Y - med)) + 1e-8
    return (Y - med) / sigma

def zero_cross_maxgate(Z: np.ndarray, t: float, smooth: float) -> np.ndarray:
    Zs = gaussian_filter(Z, smooth) if smooth > 0 else Z
    edges = np.zeros_like(Zs, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0: continue
            S = np.roll(Zs, (di, dj), axis=(0, 1))
            edges |= ((Zs * S) < 0) & (np.maximum(np.abs(Zs), np.abs(S)) >= t)
    return edges

def edge_metrics(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> dict:
    ps, gs = int(pred.sum()), int(gt.sum())
    if ps == 0 and gs == 0: return {"p": 1.0, "r": 1.0, "f1": 1.0}
    if ps == 0: return {"p": 0.0, "r": 0.0, "f1": 0.0}
    if gs == 0: return {"p": 0.0, "r": 0.0, "f1": 0.0}
    dg = distance_transform_edt(~gt); dp = distance_transform_edt(~pred)
    TP = int((pred & (dg <= tol)).sum())
    FP = int((pred & (dg > tol)).sum())
    FN = int((gt & (dp > tol)).sum())
    p = TP / (TP + FP + 1e-8); r = TP / (TP + FN + 1e-8)
    return {"p": p, "r": r, "f1": 2*p*r/(p+r+1e-8)}


# ============================================================
# 2-PASS PIPELINE
# ============================================================
def two_pass_edges(img: np.ndarray, snr: float, drift: float,
                   a_threshold: float) -> tuple:
    """B-backbone + A-gap-fill with scale-aware suppression.

    Returns (fused_edges, edges_b, edges_a_raw, edges_a_filtered)
    """
    # Pass 1: B (mid)
    kernel_b = get_dog_kernel(SIZE, B_CFG["ksize"], B_CFG["sigma1"], B_CFG["sigma2"])
    Y_b = optical_sim(img, kernel_b, snr, drift)
    Z_b = robust_zscore(Y_b)
    edges_b = zero_cross_maxgate(Z_b, B_CFG["edge_t"], B_CFG["smooth"])

    # Pass 2: A (fine) — independent noise realization
    kernel_a = get_dog_kernel(SIZE, A_CFG_BASE["ksize"],
                               A_CFG_BASE["sigma1"], A_CFG_BASE["sigma2"])
    Y_a = optical_sim(img, kernel_a, snr, drift)
    Z_a = robust_zscore(Y_a)
    edges_a = zero_cross_maxgate(Z_a, a_threshold, A_CFG_BASE["smooth"])

    # Fusion: B ∪ (A \ dilate(B, r))
    coverage = binary_dilation(edges_b, iterations=SUPPRESSION_RADIUS)
    edges_a_filtered = edges_a & ~coverage

    fused = edges_b | edges_a_filtered
    fused = binary_closing(fused, structure=CLOSING_FOOTPRINT)
    fused = thin(fused)

    return fused, edges_b, edges_a, edges_a_filtered


# Also: B-only baseline (single pass, no A)
def single_pass_b(img: np.ndarray, snr: float, drift: float) -> np.ndarray:
    kernel_b = get_dog_kernel(SIZE, B_CFG["ksize"], B_CFG["sigma1"], B_CFG["sigma2"])
    Y = optical_sim(img, kernel_b, snr, drift)
    Z = robust_zscore(Y)
    edges = zero_cross_maxgate(Z, B_CFG["edge_t"], B_CFG["smooth"])
    edges = binary_closing(edges, structure=CLOSING_FOOTPRINT)
    return thin(edges)


# ============================================================
# SHAPES (same 8)
# ============================================================
def shape_circle_square(s=128):
    x, y = np.meshgrid(np.linspace(-3,3,s), np.linspace(-3,3,s))
    img = np.zeros((s,s), np.float32); img[(x**2+y**2)<2.0]=1.0; img[20:60,20:60]=1.0
    return img

def shape_triangle(s=128):
    img = np.zeros((s,s), np.float32); cx=cy=s//2; r=s//3
    a = [np.pi/2, np.pi/2+2*np.pi/3, np.pi/2+4*np.pi/3]
    rr,cc = draw_polygon([int(cy-r*np.sin(x)) for x in a],
                         [int(cx+r*np.cos(x)) for x in a], shape=(s,s))
    img[rr,cc]=1.0; return img

def shape_concave(s=128):
    img = np.zeros((s,s), np.float32)
    rr,cc = draw_polygon([20,20,60,60,100,100,20],[30,90,90,60,60,30,30],shape=(s,s))
    img[rr,cc]=1.0; return img

def shape_thin_lines(s=128):
    img = np.zeros((s,s), np.float32)
    img[30,20:100]=1.0; img[40:110,60]=1.0
    for k in range(70):
        r,c = 25+k, 15+k
        if 0<=r<s and 0<=c<s: img[r,c]=1.0
    for k in range(80):
        r,c = 90+k//2, 20+k
        if 0<=r<s and 0<=c<s: img[r,c]=1.0
    img[70:72,10:110]=1.0; return img

def shape_text(s=128):
    img = np.zeros((s,s), np.float32)
    img[20:90,10:15]=1; img[20:25,10:40]=1; img[50:55,10:35]=1
    img[20:25,48:75]=1; img[50:55,48:75]=1; img[82:87,48:75]=1
    img[20:55,70:75]=1; img[50:87,70:75]=1
    img[20:25,85:115]=1; img[82:87,85:115]=1; img[20:87,85:90]=1
    return img

def shape_tangent(s=128):
    x,y = np.meshgrid(np.linspace(-3,3,s), np.linspace(-3,3,s))
    img = np.zeros((s,s), np.float32)
    img[((x+0.8)**2+y**2)<1.2]=1; img[((x-1.2)**2+y**2)<0.8]=1; return img

def shape_checker(s=128):
    img = np.zeros((s,s), np.float32); c=16
    for i in range(s):
        for j in range(s):
            if ((i//c)+(j//c))%2==0: img[i,j]=1.0
    return img

def shape_textured(s=128):
    np.random.seed(42)
    bg = gaussian_filter(np.random.randn(s,s).astype(np.float32), sigma=8)
    bg = (bg-bg.min())/(bg.max()-bg.min()+1e-8)*0.3
    x,y = np.meshgrid(np.linspace(-3,3,s), np.linspace(-3,3,s))
    img = bg.copy(); img[((x-0.5)**2+(y+0.3)**2)<1.5]=1.0; return img

def gt_edges(img, thr=0.1):
    gx,gy = np.gradient(img.astype(np.float32))
    return thin(np.sqrt(gx*gx+gy*gy) > thr)

SHAPES = {
    "circle_square": shape_circle_square, "triangle": shape_triangle,
    "concave": shape_concave, "thin_lines": shape_thin_lines,
    "text_F3C": shape_text, "tangent_circles": shape_tangent,
    "checker": shape_checker, "textured_obj": shape_textured,
}
NYQUIST_SHAPES = {"checker"}


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t0 = time.time()

    # Precompute GT
    shape_data = {}
    for name, fn in SHAPES.items():
        img = fn(SIZE); gt = gt_edges(img)
        if int(gt.sum()) > 0:
            shape_data[name] = {"img": img, "gt": gt}

    # =====================================================
    # PHASE 1: Threshold sweep for A at critical conditions
    # =====================================================
    print("=" * 70, flush=True)
    print("PHASE 1: A-THRESHOLD SWEEP", flush=True)
    print(f"  B fixed: t={B_CFG['edge_t']}", flush=True)
    print(f"  A sweep: {A_THRESHOLD_SWEEP}", flush=True)
    print(f"  Suppression radius: {SUPPRESSION_RADIUS}", flush=True)
    print(f"  N_TRIALS: {N_TRIALS}", flush=True)
    print("=" * 70, flush=True)

    # Test at 3 representative operating points
    test_points = [(30, 0.20), (20, 0.10), (10, 0.05)]

    # Track: for each A_t, compute per-shape F1 at each test point
    sweep_results = {}

    for a_t in A_THRESHOLD_SWEEP:
        sweep_results[a_t] = {}
        for name, sd in shape_data.items():
            if name in NYQUIST_SHAPES:
                continue
            f1_per_point = []
            for snr, drift in test_points:
                f1s = []
                for trial in range(N_TRIALS):
                    np.random.seed(trial * 10000 + int(snr) * 100 + int(drift * 100))
                    fused, _, _, _ = two_pass_edges(sd["img"], snr, drift, a_t)
                    f1s.append(edge_metrics(fused, sd["gt"])["f1"])
                f1_per_point.append(np.mean(f1s))
            sweep_results[a_t][name] = f1_per_point

    # Print sweep table
    in_band = [n for n in shape_data if n not in NYQUIST_SHAPES]
    for pi, (snr, drift) in enumerate(test_points):
        print(f"\n  @ SNR={snr}, drift={drift}:", flush=True)
        header = f"  {'A_t':>5s} |" + "".join(f" {n:>13s}" for n in in_band) + " | worst  "
        print(header, flush=True)
        print("  " + "-" * len(header), flush=True)
        for a_t in A_THRESHOLD_SWEEP:
            vals = [sweep_results[a_t][n][pi] for n in in_band]
            row = f"  {a_t:5.1f} |" + "".join(f" {v:13.4f}" for v in vals)
            row += f" | {min(vals):.4f}"
            print(row, flush=True)

    # Also show single-B baseline
    print(f"\n  Single-B baseline:", flush=True)
    for pi, (snr, drift) in enumerate(test_points):
        vals = []
        for name in in_band:
            sd = shape_data[name]
            f1s = []
            for trial in range(N_TRIALS):
                np.random.seed(trial * 10000 + int(snr) * 100 + int(drift * 100))
                edges = single_pass_b(sd["img"], snr, drift)
                f1s.append(edge_metrics(edges, sd["gt"])["f1"])
            vals.append(np.mean(f1s))
        print(f"    ({snr},{drift}): " +
              " ".join(f"{n}={v:.3f}" for n, v in zip(in_band, vals)) +
              f" | worst={min(vals):.3f}", flush=True)

    # Select best A_t by minimax across all test points
    minimax = {}
    for a_t in A_THRESHOLD_SWEEP:
        all_vals = []
        for name in in_band:
            all_vals.extend(sweep_results[a_t][name])
        minimax[a_t] = min(all_vals)

    best_a_t = max(A_THRESHOLD_SWEEP, key=lambda t: minimax[t])
    print(f"\n  Minimax scores: {minimax}", flush=True)
    print(f"  Best A threshold: {best_a_t} (minimax F1={minimax[best_a_t]:.4f})", flush=True)

    # =====================================================
    # PHASE 2: Full grid with best A_t
    # =====================================================
    print(f"\n{'=' * 70}", flush=True)
    print(f"PHASE 2: FULL GRID — 2-pass (B + A[t={best_a_t}])", flush=True)
    print(f"{'=' * 70}", flush=True)

    all_grids = {}
    grids_no_nq = []
    grids_single_b = []

    for name, sd in shape_data.items():
        t1 = time.time()
        f1_grid = np.zeros((len(SNR_VALUES), len(DRIFT_VALUES)))
        f1_grid_b = np.zeros_like(f1_grid)

        for i, snr in enumerate(SNR_VALUES):
            for j, drift in enumerate(DRIFT_VALUES):
                f1s_ms, f1s_b = [], []
                for trial in range(N_TRIALS):
                    np.random.seed(trial * 10000 + i * 100 + j)
                    fused, _, _, _ = two_pass_edges(sd["img"], snr, drift, best_a_t)
                    f1s_ms.append(edge_metrics(fused, sd["gt"])["f1"])

                    np.random.seed(trial * 10000 + i * 100 + j)
                    edges_b = single_pass_b(sd["img"], snr, drift)
                    f1s_b.append(edge_metrics(edges_b, sd["gt"])["f1"])

                f1_grid[i, j] = np.mean(f1s_ms)
                f1_grid_b[i, j] = np.mean(f1s_b)

        all_grids[name] = {"ms": f1_grid, "single": f1_grid_b}
        if name not in NYQUIST_SHAPES:
            grids_no_nq.append(f1_grid)
            grids_single_b.append(f1_grid_b)

        med = np.median(f1_grid)
        mn = np.min(f1_grid)
        wi = np.unravel_index(np.argmin(f1_grid), f1_grid.shape)
        dt = time.time() - t1
        tag = "NYQUIST" if name in NYQUIST_SHAPES else ""
        print(f"  {name:16s} | med={med:.4f} min={mn:.4f} "
              f"worst@(SNR={SNR_VALUES[wi[0]]},d={DRIFT_VALUES[wi[1]]}) "
              f"| singleB_min={np.min(f1_grid_b):.4f} | {dt:.1f}s {tag}", flush=True)

    # =====================================================
    # PHASE 3: Aggregate
    # =====================================================
    stacked = np.stack(grids_no_nq, axis=0)
    stacked_b = np.stack(grids_single_b, axis=0)
    worst_ms = np.min(stacked, axis=0)
    worst_b = np.min(stacked_b, axis=0)
    median_ms = np.median(stacked, axis=0)
    flat = stacked.flatten()

    p5 = np.percentile(flat, 5)
    gmed = np.median(flat)
    amin = np.min(flat)

    print(f"\n{'=' * 70}", flush=True)
    print(f"AGGREGATE (7 in-band shapes, {N_TRIALS} trials/cell)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Global median: {gmed:.4f}", flush=True)
    print(f"  Worst-5%:      {p5:.4f}", flush=True)
    print(f"  Absolute min:  {amin:.4f}", flush=True)

    pass_med = gmed >= 0.97
    pass_p5 = p5 >= 0.90
    status = "PASS" if (pass_med and pass_p5) else "CONDITIONAL"
    print(f"\n  CHECK: {status}", flush=True)
    print(f"    median >= 0.97? {gmed:.4f} -> {'YES' if pass_med else 'NO'}", flush=True)
    print(f"    worst-5% >= 0.90? {p5:.4f} -> {'YES' if pass_p5 else 'NO'}", flush=True)

    print(f"\n  Per-shape:", flush=True)
    print(f"  {'Shape':16s} | {'MS med':>7s} | {'MS min':>7s} | {'B min':>7s} | "
          f"{'Δmin':>6s} | {'Status':>8s}", flush=True)
    print(f"  " + "-" * 65, flush=True)
    for name in shape_data:
        g = all_grids[name]
        med_ms = np.median(g["ms"]); mn_ms = np.min(g["ms"])
        mn_b = np.min(g["single"])
        delta = mn_ms - mn_b
        st = "NYQUIST" if name in NYQUIST_SHAPES else (
            "OK" if mn_ms >= 0.90 else ("MARGINAL" if mn_ms >= 0.75 else "LOW"))
        print(f"  {name:16s} | {med_ms:7.4f} | {mn_ms:7.4f} | {mn_b:7.4f} | "
              f"{delta:+6.3f} | {st:>8s}", flush=True)

    # Worst-case map
    print(f"\n  Worst-case F1 (min over 7 in-band shapes):", flush=True)
    header = "  SNR\\Drift |" + "".join(f"  {d:.2f} " for d in DRIFT_VALUES) + " (singleB)"
    print(header, flush=True)
    print("  " + "-" * len(header), flush=True)
    for i, snr in enumerate(SNR_VALUES):
        row = f"  {snr:3d} dB   |"
        for j in range(len(DRIFT_VALUES)):
            row += f" {worst_ms[i,j]:.3f}"
        row += f"   ({worst_b[i,:]:.3f})" if False else ""
        # Also show single-B worst for this row
        row += f"  (B: {np.min(worst_b[i,:]):.3f})"
        print(row, flush=True)

    # =====================================================
    # PHASE 4: Figures
    # =====================================================
    print(f"\n{'=' * 70}", flush=True)
    print("FIGURES", flush=True)
    print(f"{'=' * 70}", flush=True)

    n_shapes = len(shape_data)
    in_band_names = [n for n in shape_data if n not in NYQUIST_SHAPES]

    # Fig 1: Per-shape overlays + heatmaps
    fig1, axes1 = plt.subplots(2, n_shapes, figsize=(3 * n_shapes, 6))
    for idx, (name, sd) in enumerate(shape_data.items()):
        np.random.seed(0)
        fused, edges_b, edges_a, edges_a_f = two_pass_edges(sd["img"], 15, 0.10, best_a_t)

        # Overlay: green=GT, red=fused, yellow=match, cyan=A contribution
        ov = np.zeros((SIZE, SIZE, 3), np.float32)
        ov[sd["gt"], 1] = 1.0
        ov[fused, 0] = 1.0
        ov[sd["gt"] & fused] = [1, 1, 0]
        ov[edges_a_f & ~edges_b, 2] = 1.0  # cyan for A-only contributions
        axes1[0, idx].imshow(ov)
        tag = " [NYQ]" if name in NYQUIST_SHAPES else ""
        axes1[0, idx].set_title(f"{name}{tag}\nGT={int(sd['gt'].sum())}px", fontsize=6)
        axes1[0, idx].axis("off")

        g = all_grids[name]["ms"]
        im = axes1[1, idx].imshow(g, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
        for ii in range(len(SNR_VALUES)):
            for jj in range(len(DRIFT_VALUES)):
                v = g[ii, jj]; c = "white" if v < 0.7 else "black"
                axes1[1, idx].text(jj, ii, f"{v:.2f}", ha="center", va="center",
                                    fontsize=5, color=c)
        axes1[1, idx].set_xticks(range(len(DRIFT_VALUES)))
        axes1[1, idx].set_xticklabels([f"{d:.2f}" for d in DRIFT_VALUES], fontsize=4)
        axes1[1, idx].set_yticks(range(len(SNR_VALUES)))
        axes1[1, idx].set_yticklabels([f"{s}" for s in SNR_VALUES], fontsize=4)
        if idx == 0: axes1[1, idx].set_ylabel("SNR (dB)", fontsize=6)

    fig1.suptitle(f"2-Pass F3C-PX: B(t={B_CFG['edge_t']}) + A(t={best_a_t}), "
                  f"suppression r={SUPPRESSION_RADIUS}", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig1.savefig("/home/claude/v2_per_shape.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  ✓ v2_per_shape.png", flush=True)

    # Fig 2: Worst-case heatmaps — MS vs single-B
    fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(17, 5))
    improvement = worst_ms - worst_b
    for ax, data, title, cmap, vmin, vmax in [
        (ax2a, worst_b,      "Single-Scale B\nWorst F1",       "RdYlGn", 0.0, 1.0),
        (ax2b, worst_ms,     "2-Pass (B+A)\nWorst F1",        "RdYlGn", 0.0, 1.0),
        (ax2c, improvement,  "Improvement\n(2-pass − single)", "RdBu",  -0.2, 0.8),
    ]:
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        for ii in range(len(SNR_VALUES)):
            for jj in range(len(DRIFT_VALUES)):
                v = data[ii, jj]; c = "white" if v < 0.4 else "black"
                ax.text(jj, ii, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=c)
        ax.set_xticks(range(len(DRIFT_VALUES)))
        ax.set_xticklabels([f"{d:.2f}" for d in DRIFT_VALUES])
        ax.set_yticks(range(len(SNR_VALUES)))
        ax.set_yticklabels([f"{s} dB" for s in SNR_VALUES])
        ax.set_xlabel("Phase Drift"); ax.set_ylabel("SNR (dB)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig2.suptitle("2-Pass vs Single-Scale: Worst-Case F1 (7 in-band shapes)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig2.savefig("/home/claude/v2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ v2_comparison.png", flush=True)

    # Fig 3: Threshold sweep visualization
    fig3, axes3 = plt.subplots(1, len(test_points), figsize=(6*len(test_points), 5))
    for pi, (snr, drift) in enumerate(test_points):
        ax = axes3[pi]
        for name in in_band_names:
            vals = [sweep_results[t][name][pi] for t in A_THRESHOLD_SWEEP]
            ax.plot(A_THRESHOLD_SWEEP, vals, 'o-', label=name, markersize=4)
        ax.axhline(0.90, color='red', linestyle='--', alpha=0.5, label='target=0.90')
        ax.set_xlabel("A threshold"); ax.set_ylabel("F1")
        ax.set_title(f"SNR={snr}, drift={drift}")
        ax.set_ylim(0.3, 1.05); ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
    fig3.suptitle("A-Threshold Sweep: F1 per shape at 3 operating points",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig3.savefig("/home/claude/v2_threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  ✓ v2_threshold_sweep.png", flush=True)

    # =====================================================
    # OPERATIONAL ENVELOPE
    # =====================================================
    print(f"\n{'=' * 70}", flush=True)
    print("OPERATIONAL ENVELOPE (2-Pass Multi-Scale)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Architecture: B(s1=1.0,s2=2.0,t=2.2) + A(s1=0.6,s2=1.2,t={best_a_t})",
          flush=True)
    print(f"  Fusion: B ∪ (A \\ dilate(B, r={SUPPRESSION_RADIUS}))", flush=True)
    print(f"  7 in-band shapes, 25 operating points, {N_TRIALS} trials/cell:", flush=True)
    print(f"    Worst-case F1: {amin:.3f}", flush=True)
    print(f"    Median F1:     {gmed:.4f}", flush=True)
    print(f"    5th pct F1:    {p5:.4f}", flush=True)
    print(f"  Checker (16px cell) excluded: Nyquist regime.", flush=True)
    print(f"  Cost: 2 optical passes (laser + SLM reconfigure).", flush=True)

    print(f"\n✓ Total runtime: {time.time()-t0:.1f}s", flush=True)
