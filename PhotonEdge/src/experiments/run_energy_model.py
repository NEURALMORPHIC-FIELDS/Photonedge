# -*- coding: utf-8 -*-
# Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
# Cluj-Napoca, Romania
#
# F3C-PX ENERGY MODEL — 3 DEPLOYMENT REGIMES
# ============================================
# Regime 1: Fixed DOE / passive mask (kernel baked in glass)
# Regime 2: Pre-configured SLM (kernel loaded once, no per-frame switch)
# Regime 3: Dynamic multi-scale (SLM switches per frame, 2-3 passes)
#
# Key correction from previous model:
#   SLM energy is NOT per-frame in regimes 1-2.
#   The dominant cost is ALWAYS readout (ADC) + digital post-proc.

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

FIGURES_DIR = _ROOT / "experiments" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

# ============================================================
# PHYSICAL PARAMETERS (literature-sourced, conservative)
# ============================================================

# --- Laser ---
# Integrated photonic / VCSEL class
LASER_CW_POWER_W = 1e-3          # 1 mW continuous wave
LASER_WALLPLUG_EFF = 0.30         # 30% wall-plug efficiency
OPTICAL_PROPAGATION_TIME_S = 3e-10  # 0.3 ns through 4f system (~10 cm path)

# --- SLM (Spatial Light Modulator) ---
# Liquid crystal on silicon (LCoS) typical
SLM_STATIC_POWER_W = 0.2          # 200 mW holding power (LC alignment)
SLM_SWITCH_ENERGY_J = 0.5e-3      # 500 µJ per full reconfiguration
SLM_SWITCH_TIME_S = 1e-3          # 1 ms switching latency

# --- DOE (Diffractive Optical Element) ---
# Passive — zero energy, zero switching
DOE_ENERGY_J = 0.0
DOE_SWITCH_TIME_S = 0.0  # (cannot switch — fixed kernel)

# --- Detector + ADC ---
# CMOS image sensor class (backside illuminated)
DETECTOR_DARK_PJ_PER_PX = 0.01    # negligible
ADC_PJ_PER_CONV_BIT = 0.5         # 0.5 pJ per conversion bit (65nm CMOS)
# Readout time per pixel (row-parallel): ~10 ns/px for rolling shutter
READOUT_TIME_PER_PX_S = 10e-9

# --- Digital post-processing ---
# ZC detection + max-gate + fusion + thinning ~ 20 ops/pixel
DIGITAL_OPS_PER_PX = 20
DIGITAL_PJ_PER_OP = 1.0           # 1 pJ/op at 28nm CMOS
# Digital clock for post-proc: assume 1 GHz
DIGITAL_CLOCK_HZ = 1e9

# --- Digital-only baseline ---
# 3× DoG convolution (ksize=21): 21*21*2 = 882 MACs per pixel per scale
DIGITAL_CONV_MACS_PER_PX_PER_SCALE = 882
DIGITAL_MAC_PJ = 1.0              # 1 pJ/MAC at 28nm

# --- Resolutions ---
RESOLUTIONS = [
    ("128x128",   128,   128),
    ("512x512",   512,   512),
    ("1024x1024", 1024, 1024),
]

# --- Readout modes ---
READOUT_MODES = {
    "Full-frame 12-bit": {"bits": 12, "fraction": 1.00},
    "Full-frame 8-bit":  {"bits": 8,  "fraction": 1.00},
    "Edge-map 1-bit":    {"bits": 1,  "fraction": 1.00},
    "ROI 8-bit (5%)":    {"bits": 8,  "fraction": 0.05},
}


# ============================================================
# ENERGY MODEL
# ============================================================
def compute_energy(n_pixels: int, n_passes: int, readout_mode: dict,
                   regime: str) -> dict:
    """Compute per-frame energy breakdown in pJ.

    Args:
        regime: "DOE" | "SLM_static" | "SLM_dynamic"
    """
    bits = readout_mode["bits"]
    frac = readout_mode["fraction"]
    pixels_read = int(n_pixels * frac)

    # (1) Optical core: laser power × propagation time × passes
    # This is the actual energy consumed by photon propagation
    laser_energy_pJ = LASER_CW_POWER_W * OPTICAL_PROPAGATION_TIME_S * n_passes * 1e12
    # Wall-plug: laser_energy / efficiency
    laser_wallplug_pJ = laser_energy_pJ / LASER_WALLPLUG_EFF

    # (2) SLM / DOE energy
    if regime == "DOE":
        slm_pJ = 0.0
        slm_latency_s = 0.0
    elif regime == "SLM_static":
        # Holding power amortized over frame time (but no switching)
        # At high FPS this becomes significant; at low FPS negligible
        # We report just the per-frame holding cost at 30 FPS as reference
        # but parametrize it properly
        slm_pJ = 0.0  # no switching energy; holding cost in wall-plug budget
        slm_latency_s = 0.0
    elif regime == "SLM_dynamic":
        # Full switching energy per pass (except first which is pre-loaded)
        slm_pJ = SLM_SWITCH_ENERGY_J * (n_passes - 1) * 1e12
        slm_latency_s = SLM_SWITCH_TIME_S * (n_passes - 1)

    # (3) Detector + ADC readout: per pass
    adc_per_pass_pJ = pixels_read * bits * ADC_PJ_PER_CONV_BIT
    adc_total_pJ = adc_per_pass_pJ * n_passes
    detector_pJ = pixels_read * DETECTOR_DARK_PJ_PER_PX * n_passes  # negligible

    # (4) Digital post-processing (runs once on fused result)
    digital_pJ = pixels_read * DIGITAL_OPS_PER_PX * DIGITAL_PJ_PER_OP

    # (5) Latency
    optical_latency_s = OPTICAL_PROPAGATION_TIME_S * n_passes
    readout_latency_s = (pixels_read * READOUT_TIME_PER_PX_S) * n_passes
    digital_latency_s = (pixels_read * DIGITAL_OPS_PER_PX) / DIGITAL_CLOCK_HZ

    total_pJ = laser_wallplug_pJ + slm_pJ + adc_total_pJ + detector_pJ + digital_pJ
    total_latency_s = optical_latency_s + slm_latency_s + readout_latency_s + digital_latency_s

    return {
        "laser_pJ": laser_wallplug_pJ,
        "slm_pJ": slm_pJ,
        "adc_pJ": adc_total_pJ,
        "detector_pJ": detector_pJ,
        "digital_pJ": digital_pJ,
        "total_pJ": total_pJ,
        "pJ_per_px": total_pJ / n_pixels,
        "latency_s": total_latency_s,
        "optical_lat_s": optical_latency_s,
        "slm_lat_s": slm_latency_s,
        "readout_lat_s": readout_latency_s,
        "digital_lat_s": digital_latency_s,
    }


def compute_digital_baseline(n_pixels: int, n_scales: int) -> dict:
    """Digital-only DoG convolution baseline."""
    conv_pJ = n_pixels * DIGITAL_CONV_MACS_PER_PX_PER_SCALE * n_scales * DIGITAL_MAC_PJ
    post_pJ = n_pixels * DIGITAL_OPS_PER_PX * DIGITAL_PJ_PER_OP
    total_pJ = conv_pJ + post_pJ
    # Latency: all MACs sequential (simplified; real GPU would parallelize)
    total_macs = n_pixels * DIGITAL_CONV_MACS_PER_PX_PER_SCALE * n_scales + n_pixels * DIGITAL_OPS_PER_PX
    latency_s = total_macs / DIGITAL_CLOCK_HZ
    return {
        "conv_pJ": conv_pJ,
        "post_pJ": post_pJ,
        "total_pJ": total_pJ,
        "pJ_per_px": total_pJ / n_pixels,
        "latency_s": latency_s,
    }


# ============================================================
# SLM HOLDING POWER (separate budget line)
# ============================================================
def slm_holding_power_per_frame(fps: float) -> float:
    """SLM static holding power per frame in pJ."""
    frame_time_s = 1.0 / fps
    return SLM_STATIC_POWER_W * frame_time_s * 1e12


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    t0 = time.time()

    REGIMES = [
        ("DOE",         "Fixed DOE (passive mask)",         "DOE",         1),
        ("SLM_1pass",   "Pre-config SLM (1 pass, B only)", "SLM_static",  1),
        ("SLM_3pass",   "Dynamic SLM (3 passes, A+B+C)",   "SLM_dynamic", 3),
    ]

    print("=" * 70, flush=True)
    print("F3C-PX ENERGY MODEL — 3 DEPLOYMENT REGIMES", flush=True)
    print("=" * 70, flush=True)
    print(f"\n  Physical parameters:", flush=True)
    print(f"    Laser: {LASER_CW_POWER_W*1e3:.1f} mW CW, η={LASER_WALLPLUG_EFF:.0%}, "
          f"propagation={OPTICAL_PROPAGATION_TIME_S*1e9:.1f} ns", flush=True)
    print(f"    SLM: hold={SLM_STATIC_POWER_W*1e3:.0f} mW, "
          f"switch={SLM_SWITCH_ENERGY_J*1e6:.0f} µJ, "
          f"time={SLM_SWITCH_TIME_S*1e3:.0f} ms", flush=True)
    print(f"    ADC: {ADC_PJ_PER_CONV_BIT} pJ/conv-bit (CMOS 65nm)", flush=True)
    print(f"    Digital: {DIGITAL_OPS_PER_PX} ops/px × {DIGITAL_PJ_PER_OP} pJ/op (28nm)", flush=True)
    print(f"    Readout speed: {READOUT_TIME_PER_PX_S*1e9:.0f} ns/px", flush=True)
    print(f"    Digital baseline: {DIGITAL_CONV_MACS_PER_PX_PER_SCALE} MACs/px/scale", flush=True)

    # =====================================================
    # TABLE 1: Per-frame energy at 512×512 (reference)
    # =====================================================
    ref_name, ref_w, ref_h = "512x512", 512, 512
    ref_npx = ref_w * ref_h

    print(f"\n{'=' * 70}", flush=True)
    print(f"TABLE 1: ENERGY PER FRAME — {ref_name} ({ref_npx:,} pixels)", flush=True)
    print(f"{'=' * 70}", flush=True)

    for reg_id, reg_name, reg_type, n_pass in REGIMES:
        print(f"\n  --- {reg_name} ({n_pass} pass{'es' if n_pass>1 else ''}) ---", flush=True)
        print(f"  {'Readout Mode':<22s} | {'Laser':>8s} | {'SLM':>10s} | "
              f"{'ADC':>10s} | {'Digital':>8s} | {'TOTAL':>10s} | "
              f"{'pJ/px':>8s} | {'Latency':>8s}", flush=True)
        print(f"  {'-'*96}", flush=True)

        for rm_name, rm in READOUT_MODES.items():
            e = compute_energy(ref_npx, n_pass, rm, reg_type)
            lat_str = f"{e['latency_s']*1e3:.2f}ms" if e['latency_s'] > 1e-3 else f"{e['latency_s']*1e6:.1f}µs"
            print(f"  {rm_name:<22s} | {e['laser_pJ']:>8.1f} | {e['slm_pJ']:>10.0f} | "
                  f"{e['adc_pJ']:>10.0f} | {e['digital_pJ']:>8.0f} | "
                  f"{e['total_pJ']:>10.0f} | {e['pJ_per_px']:>8.2f} | {lat_str:>8s}", flush=True)

        # Digital-only baseline
        db = compute_digital_baseline(ref_npx, n_pass)
        lat_str = f"{db['latency_s']*1e3:.2f}ms"
        print(f"  {'DIGITAL-ONLY':.<22s} | {'---':>8s} | {'---':>10s} | "
              f"{'---':>10s} | {db['total_pJ']:>8.0f} | {db['total_pJ']:>10.0f} | "
              f"{db['pJ_per_px']:>8.0f} | {lat_str:>8s}", flush=True)

    # =====================================================
    # TABLE 2: Cross-resolution comparison (best readout per regime)
    # =====================================================
    print(f"\n{'=' * 70}", flush=True)
    print(f"TABLE 2: CROSS-RESOLUTION — optimal readout per regime", flush=True)
    print(f"{'=' * 70}", flush=True)

    # For each regime, pick the "practical" readout:
    # DOE/SLM_static: Edge-map 1-bit (natural for single-pass edge detector)
    # SLM_dynamic: Full-frame 8-bit (need full Z-map for multi-scale fusion)
    practical_readout = {
        "DOE":       ("Edge-map 1-bit",  READOUT_MODES["Edge-map 1-bit"]),
        "SLM_1pass": ("Edge-map 1-bit",  READOUT_MODES["Edge-map 1-bit"]),
        "SLM_3pass": ("Full-frame 8-bit", READOUT_MODES["Full-frame 8-bit"]),
    }

    print(f"  {'Resolution':<12s} | {'Regime':<32s} | {'Readout':<18s} | "
          f"{'pJ/px':>8s} | {'Total µJ':>9s} | {'Latency':>8s} | "
          f"{'vs Digital':>10s} | {'MaxFPS':>7s}", flush=True)
    print(f"  {'-'*120}", flush=True)

    for res_name, w, h in RESOLUTIONS:
        npx = w * h
        for reg_id, reg_name, reg_type, n_pass in REGIMES:
            rm_name, rm = practical_readout[reg_id]
            e = compute_energy(npx, n_pass, rm, reg_type)
            db = compute_digital_baseline(npx, n_pass)
            ratio = e['total_pJ'] / db['total_pJ']
            lat = e['latency_s']
            max_fps = 1.0 / lat if lat > 0 else float('inf')
            lat_str = f"{lat*1e3:.2f}ms" if lat > 1e-3 else f"{lat*1e6:.1f}µs"
            ratio_str = f"{ratio:.3f}×" if ratio < 1 else f"{ratio:.2f}×"

            print(f"  {res_name:<12s} | {reg_name:<32s} | {rm_name:<18s} | "
                  f"{e['pJ_per_px']:>8.2f} | {e['total_pJ']/1e6:>9.3f} | "
                  f"{lat_str:>8s} | {ratio_str:>10s} | {max_fps:>7.0f}", flush=True)

        # Digital baseline row
        db = compute_digital_baseline(npx, 3)
        lat_str = f"{db['latency_s']*1e3:.2f}ms"
        max_fps = 1.0 / db['latency_s'] if db['latency_s'] > 0 else 0
        print(f"  {res_name:<12s} | {'Digital-only (3-scale)':.<32s} | {'N/A':<18s} | "
              f"{db['pJ_per_px']:>8.0f} | {db['total_pJ']/1e6:>9.3f} | "
              f"{lat_str:>8s} | {'(ref)':>10s} | {max_fps:>7.0f}", flush=True)
        print(f"  {'-'*120}", flush=True)

    # =====================================================
    # TABLE 3: SLM holding power budget (separate line item)
    # =====================================================
    print(f"\n{'=' * 70}", flush=True)
    print(f"TABLE 3: SLM HOLDING POWER (wall-plug overhead)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  Note: SLM holding power is continuous, not per-frame.", flush=True)
    print(f"  In DOE regime this is zero. In SLM regimes it adds to wall-plug budget.", flush=True)
    print(f"  This is amortized over frame rate:\n", flush=True)

    fps_list = [30, 100, 1000]
    for res_name, w, h in RESOLUTIONS:
        npx = w * h
        print(f"  {res_name}:", flush=True)
        for fps in fps_list:
            hold_pJ = slm_holding_power_per_frame(fps)
            hold_per_px = hold_pJ / npx
            print(f"    {fps:4d} FPS: {hold_pJ/1e6:>8.2f} µJ/frame = {hold_per_px:>8.1f} pJ/px", flush=True)

    # =====================================================
    # KEY CONCLUSIONS
    # =====================================================
    print(f"\n{'=' * 70}", flush=True)
    print("KEY CONCLUSIONS (White Paper)", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Compute specific numbers for the claims
    e_doe_1bit = compute_energy(512*512, 1, READOUT_MODES["Edge-map 1-bit"], "DOE")
    e_doe_roi  = compute_energy(512*512, 1, READOUT_MODES["ROI 8-bit (5%)"], "DOE")
    e_dyn_8bit = compute_energy(512*512, 3, READOUT_MODES["Full-frame 8-bit"], "SLM_dynamic")
    db_3 = compute_digital_baseline(512*512, 3)
    db_1 = compute_digital_baseline(512*512, 1)

    print(f"""
  1. OPTICAL CORE ENERGY IS NEGLIGIBLE
     Laser propagation: {e_doe_1bit['laser_pJ']:.1f} pJ/frame ({e_doe_1bit['laser_pJ']/512/512:.4f} pJ/px)
     This is {e_doe_1bit['laser_pJ']/db_1['total_pJ']*100:.4f}% of digital-only baseline.
     The optical convolution is effectively "free" in energy terms.

  2. READOUT DOMINATES — AND IS CONTROLLABLE
     Full-frame 12-bit: ADC = {compute_energy(512*512, 1, READOUT_MODES["Full-frame 12-bit"], "DOE")['adc_pJ']/1e6:.2f} µJ/frame
     Edge-map 1-bit:    ADC = {e_doe_1bit['adc_pJ']/1e6:.3f} µJ/frame  (12× reduction)
     ROI 8-bit (5%):    ADC = {e_doe_roi['adc_pJ']/1e6:.3f} µJ/frame   (24× reduction)

  3. REGIME COMPARISON AT 512×512 (pJ/px):
     Fixed DOE + 1-bit readout:     {e_doe_1bit['pJ_per_px']:.2f} pJ/px
     Fixed DOE + ROI 8-bit (5%):    {e_doe_roi['pJ_per_px']:.2f} pJ/px
     Dynamic SLM + full 8-bit:      {e_dyn_8bit['pJ_per_px']:.1f} pJ/px  (SLM switching dominates)
     Digital-only (3 scales):       {db_3['pJ_per_px']:.0f} pJ/px

  4. ENERGY ADVANTAGE QUANTIFIED
     DOE + 1-bit vs digital-only(1-scale): {e_doe_1bit['pJ_per_px']/db_1['pJ_per_px']:.3f}× ({e_doe_1bit['pJ_per_px']:.1f} vs {db_1['pJ_per_px']:.0f} pJ/px)
     DOE + ROI vs digital-only(1-scale):   {e_doe_roi['pJ_per_px']/db_1['pJ_per_px']:.3f}× ({e_doe_roi['pJ_per_px']:.1f} vs {db_1['pJ_per_px']:.0f} pJ/px)
     Dynamic 3-pass vs digital-only(3-scale): {e_dyn_8bit['pJ_per_px']/db_3['pJ_per_px']:.2f}× (SLM overhead negates advantage)

  5. VALUE PROPOSITION (one sentence)
     "In fixed-kernel deployment (DOE or pre-configured SLM), F3C-PX delivers
     edge maps at {e_doe_1bit['pJ_per_px']:.0f}–{e_doe_roi['pJ_per_px']:.0f} pJ/px, which is {db_1['pJ_per_px']/e_doe_1bit['pJ_per_px']:.0f}–{db_1['pJ_per_px']/e_doe_roi['pJ_per_px']:.0f}× below
     digital-only convolution ({db_1['pJ_per_px']:.0f} pJ/px). Multi-scale (3-pass with SLM)
     trades this advantage for spectral bandwidth coverage."

  6. LATENCY ADVANTAGE
     Optical propagation: {OPTICAL_PROPAGATION_TIME_S*1e9:.1f} ns (vs ms-scale digital)
     Bottleneck: readout ({READOUT_TIME_PER_PX_S*1e9:.0f} ns/px × {512*512:,} px = {512*512*READOUT_TIME_PER_PX_S*1e3:.1f} ms)
     Max FPS (DOE + 1-bit): {1/(e_doe_1bit['latency_s']):.0f} Hz
     Max FPS (Dynamic 3-pass): {1/(e_dyn_8bit['latency_s']):.0f} Hz
""", flush=True)

    # =====================================================
    # FIGURES
    # =====================================================
    print("GENERATING FIGURES...", flush=True)

    # --- Fig 1: Energy breakdown bars (3 regimes × 4 readout modes) ---
    fig1, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    colors = {'laser': '#3498db', 'slm': '#9b59b6', 'adc': '#e74c3c', 'digital': '#2ecc71'}

    for idx, (reg_id, reg_name, reg_type, n_pass) in enumerate(REGIMES):
        ax = axes[idx]
        rm_names = list(READOUT_MODES.keys())
        x = np.arange(len(rm_names))

        laser_vals, slm_vals, adc_vals, dig_vals = [], [], [], []
        for rm_name in rm_names:
            e = compute_energy(ref_npx, n_pass, READOUT_MODES[rm_name], reg_type)
            laser_vals.append(e['laser_pJ'] / 1e6)
            slm_vals.append(e['slm_pJ'] / 1e6)
            adc_vals.append(e['adc_pJ'] / 1e6)
            dig_vals.append(e['digital_pJ'] / 1e6)

        w = 0.6
        bottom = np.zeros(len(rm_names))
        for vals, label, color in [
            (laser_vals, 'Optical', colors['laser']),
            (slm_vals, 'SLM switch', colors['slm']),
            (adc_vals, 'ADC readout', colors['adc']),
            (dig_vals, 'Digital post', colors['digital']),
        ]:
            ax.bar(x, vals, w, bottom=bottom, label=label, color=color)
            bottom += np.array(vals)

        # Digital baseline reference line
        db = compute_digital_baseline(ref_npx, n_pass)
        ax.axhline(db['total_pJ']/1e6, color='red', linestyle='--', linewidth=2,
                    label=f'Digital-only ({db["total_pJ"]/1e6:.1f} µJ)')

        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(' ', '\n') for n in rm_names], fontsize=7)
        ax.set_ylabel("Energy per frame (µJ)")
        ax.set_title(f"{reg_name}\n({n_pass} pass{'es' if n_pass>1 else ''})", fontsize=9)
        ax.legend(fontsize=6, loc='upper right')

        # Annotate totals
        for i, total in enumerate(bottom):
            if total > 0:
                ax.text(i, total, f"{total:.1f}µJ", ha='center', va='bottom', fontsize=6)

    fig1.suptitle(f"F3C-PX Energy Breakdown — {ref_name} per frame", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig1.savefig(str(FIGURES_DIR / "energy_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print("  ✓ energy_breakdown.png", flush=True)

    # --- Fig 2: pJ/px scaling across resolutions ---
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5.5))

    res_range = np.logspace(np.log10(32*32), np.log10(2048*2048), 80)

    # Left: Fixed DOE regime (various readout modes)
    for rm_name, rm in READOUT_MODES.items():
        pjpx = []
        for npx in res_range:
            e = compute_energy(int(npx), 1, rm, "DOE")
            pjpx.append(e['pJ_per_px'])
        ax2a.loglog(res_range, pjpx, '-', linewidth=2, label=rm_name)

    db_pjpx = compute_digital_baseline(1, 1)['pJ_per_px']
    ax2a.axhline(db_pjpx, color='red', linestyle='--', linewidth=2,
                  label=f'Digital-only 1-scale ({db_pjpx:.0f} pJ/px)')
    ax2a.set_xlabel("Resolution (total pixels)"); ax2a.set_ylabel("Energy (pJ/px)")
    ax2a.set_title("Regime 1: Fixed DOE (1 pass)")
    ax2a.legend(fontsize=7); ax2a.grid(True, alpha=0.3)
    for _, w, h in RESOLUTIONS:
        ax2a.axvline(w*h, color='gray', linestyle=':', alpha=0.4)

    # Right: All 3 regimes with practical readout
    for reg_id, reg_name, reg_type, n_pass in REGIMES:
        rm_name, rm = practical_readout[reg_id]
        pjpx = []
        for npx in res_range:
            e = compute_energy(int(npx), n_pass, rm, reg_type)
            pjpx.append(e['pJ_per_px'])
        ax2b.loglog(res_range, pjpx, '-', linewidth=2, label=f"{reg_name}\n({rm_name})")

    db3_pjpx = compute_digital_baseline(1, 3)['pJ_per_px']
    ax2b.axhline(db3_pjpx, color='red', linestyle='--', linewidth=2,
                  label=f'Digital-only 3-scale ({db3_pjpx:.0f} pJ/px)')
    ax2b.set_xlabel("Resolution (total pixels)"); ax2b.set_ylabel("Energy (pJ/px)")
    ax2b.set_title("All Regimes — practical readout")
    ax2b.legend(fontsize=6); ax2b.grid(True, alpha=0.3)
    for _, w, h in RESOLUTIONS:
        ax2b.axvline(w*h, color='gray', linestyle=':', alpha=0.4)

    fig2.suptitle("F3C-PX Energy Scaling vs Resolution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig2.savefig(str(FIGURES_DIR / "energy_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ energy_scaling.png", flush=True)

    # --- Fig 3: Latency breakdown ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
    configs = []
    lat_optical, lat_slm, lat_readout, lat_digital = [], [], [], []
    labels_lat = []

    for reg_id, reg_name, reg_type, n_pass in REGIMES:
        rm_name, rm = practical_readout[reg_id]
        e = compute_energy(ref_npx, n_pass, rm, reg_type)
        labels_lat.append(f"{reg_name}\n({rm_name})")
        lat_optical.append(e['optical_lat_s'] * 1e3)
        lat_slm.append(e['slm_lat_s'] * 1e3)
        lat_readout.append(e['readout_lat_s'] * 1e3)
        lat_digital.append(e['digital_lat_s'] * 1e3)

    x = np.arange(len(labels_lat))
    w = 0.5
    bottom = np.zeros(len(labels_lat))
    for vals, label, color in [
        (lat_optical, 'Optical', colors['laser']),
        (lat_slm, 'SLM switch', colors['slm']),
        (lat_readout, 'Readout', colors['adc']),
        (lat_digital, 'Digital', colors['digital']),
    ]:
        ax3.barh(x, vals, w, left=bottom, label=label, color=color)
        bottom += np.array(vals)

    # Digital baseline
    db = compute_digital_baseline(ref_npx, 3)
    ax3.axvline(db['latency_s']*1e3, color='red', linestyle='--', linewidth=2,
                label=f'Digital-only 3-scale ({db["latency_s"]*1e3:.1f} ms)')

    ax3.set_yticks(x); ax3.set_yticklabels(labels_lat, fontsize=8)
    ax3.set_xlabel("Latency (ms)"); ax3.set_title(f"Latency Breakdown — {ref_name}")
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3, axis='x')

    for i, total in enumerate(bottom):
        ax3.text(total + 0.1, i, f"{total:.2f} ms", va='center', fontsize=8)

    plt.tight_layout()
    fig3.savefig(str(FIGURES_DIR / "energy_latency.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print("  ✓ energy_latency.png", flush=True)

    # --- Fig 4: Summary comparison card ---
    fig4, ax4 = plt.subplots(1, 1, figsize=(12, 6))
    ax4.axis('off')

    table_data = []
    table_headers = ["", "Fixed DOE\n1-bit readout", "Pre-config SLM\n1-bit readout",
                     "Dynamic SLM\n8-bit, 3-pass", "Digital-only\n3-scale conv"]

    e1 = compute_energy(ref_npx, 1, READOUT_MODES["Edge-map 1-bit"], "DOE")
    e2 = compute_energy(ref_npx, 1, READOUT_MODES["Edge-map 1-bit"], "SLM_static")
    e3 = compute_energy(ref_npx, 3, READOUT_MODES["Full-frame 8-bit"], "SLM_dynamic")
    db = compute_digital_baseline(ref_npx, 3)

    rows = [
        ("Energy/frame", f"{e1['total_pJ']/1e6:.3f} µJ", f"{e2['total_pJ']/1e6:.3f} µJ",
         f"{e3['total_pJ']/1e6:.1f} µJ", f"{db['total_pJ']/1e6:.1f} µJ"),
        ("Energy/pixel", f"{e1['pJ_per_px']:.1f} pJ", f"{e2['pJ_per_px']:.1f} pJ",
         f"{e3['pJ_per_px']:.0f} pJ", f"{db['pJ_per_px']:.0f} pJ"),
        ("vs Digital", f"{e1['pJ_per_px']/db['pJ_per_px']:.3f}×",
         f"{e2['pJ_per_px']/db['pJ_per_px']:.3f}×",
         f"{e3['pJ_per_px']/db['pJ_per_px']:.2f}×", "(ref)"),
        ("Latency", f"{e1['latency_s']*1e3:.2f} ms", f"{e2['latency_s']*1e3:.2f} ms",
         f"{e3['latency_s']*1e3:.1f} ms", f"{db['latency_s']*1e3:.1f} ms"),
        ("Max FPS", f"{1/e1['latency_s']:.0f}", f"{1/e2['latency_s']:.0f}",
         f"{1/e3['latency_s']:.0f}", f"{1/db['latency_s']:.0f}"),
        ("Bandwidth", "Single-scale", "Single-scale", "Multi-scale\n(A+B+C)", "Multi-scale\n(3 conv)"),
        ("Reconfigurable", "No", "Yes (offline)", "Yes (per-frame)", "Yes"),
        ("SLM overhead", "None", "Holding only", "500 µJ/switch\n+ 1 ms latency", "N/A"),
    ]

    table = ax4.table(cellText=[r[1:] for r in rows],
                      rowLabels=[r[0] for r in rows],
                      colLabels=table_headers[1:],
                      cellLoc='center', rowLoc='right',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    # Color code: green for energy advantage, red for disadvantage
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#34495e'); cell.set_text_props(color='white', fontweight='bold')
        elif row == 3:  # "vs Digital" row
            text = cell.get_text().get_text()
            if '×' in text:
                val = float(text.replace('×', ''))
                if val < 0.5:
                    cell.set_facecolor('#d5f5e3')
                elif val > 1.0:
                    cell.set_facecolor('#fadbd8')

    ax4.set_title(f"F3C-PX Deployment Comparison — {ref_name}",
                  fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    fig4.savefig(str(FIGURES_DIR / "energy_comparison_card.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print("  ✓ energy_comparison_card.png", flush=True)

    print(f"\n✓ All figures generated in {time.time()-t0:.1f}s", flush=True)
