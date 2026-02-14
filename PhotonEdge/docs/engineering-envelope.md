<p align="center">
  <img src="../assets/logo.png" alt="PhotonEdge" width="400">
</p>

# Engineering Envelope

> Operational specification for PhotonEdge deployment. Defines guaranteed performance bounds, declared limitations, and calibration requirements.

---

## Operating Regimes

### Guaranteed Regime

| Parameter | Specification |
|-----------|--------------|
| **SNR** | ≥ 15 dB |
| **Phase drift (σ)** | ≤ 0.10 rad |
| **Feature width** | > 2×σ₂ = 4 px (at σ₂ = 2.0) |
| **Metric** | Symmetric DT matching, tol = 2 px |
| **Worst-case F1** | ≥ 0.73 (tangent_circles at boundary geometry) |
| **Worst-case Recall** | 1.00 (no edges missed) |
| **Shapes validated** | 7 in-band classes |
| **Nyquist exclusions** | Periodic patterns with 1px edge spacing |

### Best-Effort Regime

| Parameter | Specification |
|-----------|--------------|
| **SNR** | ≥ 10 dB |
| **Phase drift (σ)** | ≤ 0.20 rad |
| **Worst-case F1** | ≥ 0.42 (degraded at high-SNR + high-drift corner) |
| **Recall** | 1.00 (maintained) |

---

## Validated Performance — Full Grid

All values are worst-case F1 across 7 in-band shapes, averaged over 5 independent noise realizations per cell.

### Heatmap

<p align="center">
  <img src="../experiments/figures/d1_envelope.png" alt="Engineering Envelope Heatmaps" width="900">
</p>

### Per-Shape Performance (Guaranteed Regime)

| Shape | Min F1 | Median F1 | Min Precision | Min Recall | GT Pixels |
|-------|--------|-----------|---------------|------------|-----------|
| circle_square | 0.864 | 0.965 | 0.761 | 1.000 | 604 |
| triangle | 0.810 | 0.946 | 0.681 | 1.000 | 494 |
| concave_polygon | 0.933 | 0.983 | 0.874 | 1.000 | 760 |
| thin_lines | 0.999 | 1.000 | 0.998 | 1.000 | 1682 |
| text_F3C | 0.998 | 0.999 | 0.996 | 1.000 | 1522 |
| tangent_circles | 0.727 | 0.932 | 0.572 | 1.000 | 364 |
| textured_object | 0.990 | 0.993 | 0.979 | 1.000 | 292 |
| checker | **EXCLUDED** | — | — | — | 3388 |

### Performance Observations

**Recall is uniformly 1.00.** The system never misses a real edge within the declared bandwidth. All degradation is in Precision (false positives).

**Tangent circles** define the worst case (F1 = 0.727) due to the tangency zone where two circles meet — this creates a geometrically ambiguous region where the DoG produces edge doubling. This is a shape-specific limitation, not a system defect.

**Thin lines and text** achieve near-perfect F1 (≥ 0.998) because these features have strong zero-crossings within the DoG passband when using 3-scale fusion.

---

## Declared Limitations

### 1. Nyquist / Band-Limit

The DoG kernel acts as a band-pass filter. Features with spatial frequency above the Nyquist limit of the finest scale cannot be resolved. Specifically:

- **Minimum resolvable feature**: ~2×σ₂ pixels (4 px at σ₂ = 2.0)
- **Checker pattern** (16 px cells, 1 px edge spacing): edges are closer than the band-pass resolution → F1 = 0.000. This is a **physical limitation**, not a software defect.

### 2. High-SNR + High-Drift Corner Case

At SNR ≥ 25 dB with drift σ ≥ 0.15, phase drift artifacts become coherent and produce zero-crossings with amplitude exceeding the statistical gate. The fixed-threshold gate cannot distinguish real edges from drift-induced false crossings because both have high amplitude.

**Mitigation**: SNR-adaptive threshold (see [Adaptive Threshold](adaptive-threshold.md)) raises the gate at high SNR, recovering F1 from 0.43 to 0.70+ at the worst corner.

### 3. Edge Thickness

The optical DoG produces zero-crossings with natural width of 2–3 pixels. The evaluation metric accommodates this with tol = 2 px. Applications requiring pixel-exact edge localization must apply additional sub-pixel refinement in the digital stage.

---

## Calibration Requirements

### Per-Deployment Calibration

| Step | Purpose | Frequency |
|------|---------|-----------|
| Flat-field exposure | Measure system noise floor → estimate SNR | Once per power-on |
| Uniform phase target | Measure drift statistics → set threshold | Once per environment change |
| Known-edge target | Verify F1 within guaranteed bounds | Periodic (daily/weekly) |

### Runtime Monitoring

| Signal | Threshold | Action |
|--------|-----------|--------|
| Estimated SNR < 15 dB | Warning | Switch to best-effort mode |
| Estimated drift > 0.10 | Warning | Engage adaptive threshold |
| Edge density > 30% of frame | Anomaly | Likely noise-dominated → reject frame |

---

## Canonical Pipeline Parameters

```yaml
# Scale B (backbone)
scale_b:
  sigma1: 1.0
  sigma2: 2.0
  kernel_size: 21
  edge_threshold: 2.2
  smooth_sigma: 0.9

# Scale A (fine detail, optional)
scale_a:
  sigma1: 0.6
  sigma2: 1.2
  kernel_size: 21
  edge_threshold: 2.2  # or adaptive: 0.08 × SNR + 4.0
  smooth_sigma: 0.7

# Scale C (coarse, optional)
scale_c:
  sigma1: 1.6
  sigma2: 3.2
  kernel_size: 21
  edge_threshold: 2.2
  smooth_sigma: 1.1

# Fusion v2
fusion:
  mode: fine_only_where_needed
  coverage_dilation: 3  # pixels
  backbone: [B, C]
  gap_filler: A

# Post-processing
morphology:
  closing: true
  closing_kernel: 3x3
  thinning: true  # optional, application-dependent

# Evaluation
metric:
  type: symmetric_distance_transform
  tolerance: 2  # pixels
```

---

<p align="center">
  <sub>© 2024–2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania</sub>
</p>
