<p align="center">
  <img src="../assets/logo.png" alt="PhotonEdge" width="400">
</p>

# Application Domains

> Where optical band-pass edge detection delivers economic value. The common requirement: edge/contour extraction is a bottleneck (energy, latency, or throughput) in the processing pipeline.

---

## Primary Domains

### A. Always-On Vision (Ultra-Low Power)

**Use case**: Smart cameras, IoT visual sensors, wake-up vision triggers.

**Value**: PhotonEdge produces edge maps at 20 pJ/pixel in DOE mode — enabling continuous scene monitoring at < 10 µW for 512×512 at 1 FPS. Digital AI inference runs only on ROI triggered by edge activity, reducing average compute by 10–100×.

**Requirements**: Fixed DOE (single kernel), 1-bit readout, edge-density thresholding.

**Fit**: ★★★★★ — Matches exactly the fixed-kernel, minimum-energy regime.

---

### B. Industrial Inspection & Metrology

**Use case**: Surface defect detection, crack/fissure identification, dimensional measurement, component alignment verification.

**Value**: Deterministic, repeatable edge extraction with latency < 8 ms at 512×512. Fixed DOE ensures identical filtering across millions of parts with no software variation. Robust to illumination changes within the SNR ≥ 15 dB envelope.

**Requirements**: Fixed DOE or pre-configured SLM, full-frame 8-bit readout, calibrated metric.

**Fit**: ★★★★★ — Industrial inspection is the canonical fixed-kernel, high-throughput application.

---

### C. Robotics & Autonomous Systems

**Use case**: Contour extraction for localization, obstacle boundary detection, coarse segmentation for SLAM, shape recognition for grasping.

**Value**: Edge maps at 127 FPS (DOE, 512²) provide real-time contour input for navigation systems. The optical front-end offloads the first convolutional layer from the main compute pipeline, reducing latency and power for battery-operated systems.

**Requirements**: Pre-configured SLM (adaptable to different environments), 8-bit readout, multi-scale optional.

**Fit**: ★★★★☆ — Requires SLM flexibility for different operating environments.

---

### D. Scientific Imaging (Microscopy, Lab Instruments)

**Use case**: Edge/feature enhancement on high-throughput imaging streams (cell boundaries, crystal edges, particle detection).

**Value**: Processes full microscopy frames at optical speed, extracting structural features before they enter the digital analysis pipeline. Particularly valuable for high-speed imaging (> 100 FPS) where digital convolution becomes a bottleneck.

**Requirements**: Multi-scale (thin features common), full-frame readout.

**Fit**: ★★★★☆ — High value but may require dynamic SLM for varying feature scales.

---

### E. Surveillance & Traffic Monitoring

**Use case**: Edge-based motion detection, vehicle/pedestrian boundary extraction, scene change triggers.

**Value**: Fixed DOE + 1-bit readout produces edge maps at minimum energy. Combined with frame-differencing, enables ultra-efficient change detection. 127 FPS at 512² supports real-time video processing.

**Requirements**: Fixed DOE, 1-bit readout, temporal differencing in digital stage.

**Fit**: ★★★★☆ — High fit for fixed-scene monitoring.

---

### F. ML/CNN Front-End Pre-Processing

**Use case**: Edge maps as input channels for convolutional neural networks, ROI gating for selective inference, first-layer replacement.

**Value**: Optical edge map replaces or augments the first convolutional layer of a CNN, reducing compute by 882 MACs/pixel/scale. For edge-gated inference, only regions with detected edges are processed by the full network (5–20% of frame typically), yielding 5–20× compute reduction.

**Requirements**: Full-frame 8-bit readout (to preserve gradient information), optional multi-scale.

**Fit**: ★★★☆☆ — Requires integration with existing ML pipelines.

---

### G. Remote Sensing & Satellite Imagery

**Use case**: On-board edge extraction for roads, buildings, coastlines, change detection.

**Value**: Pre-filtering at acquisition reduces downlink bandwidth (transmit 1-bit edge maps instead of 8/12-bit imagery). Operates within the strict power envelope of satellite platforms.

**Requirements**: Fixed DOE (no moving parts), radiation-hardened detector, 1-bit readout.

**Fit**: ★★★☆☆ — High strategic value but requires space-qualified components.

---

## Applicability Matrix

| Domain | Regime | Readout | Multi-Scale | Fit |
|--------|--------|---------|-------------|-----|
| Always-on vision | DOE | 1-bit | No | ★★★★★ |
| Industrial inspection | DOE / SLM pre-config | 8-bit | Optional | ★★★★★ |
| Robotics | SLM pre-config | 8-bit | Optional | ★★★★☆ |
| Scientific imaging | SLM dynamic | 8/12-bit | Yes | ★★★★☆ |
| Surveillance | DOE | 1-bit | No | ★★★★☆ |
| CNN front-end | DOE / SLM | 8-bit | Optional | ★★★☆☆ |
| Remote sensing | DOE | 1-bit | No | ★★★☆☆ |

---

## Common Condition

All applications require that target features are **within the optical passband** (feature width > 2×σ₂). Applications with mixed-scale features (e.g., scientific imaging with both cell boundaries and thin filaments) benefit from multi-scale deployment at the cost of increased energy and latency.

---

<p align="center">
  <sub>© 2024–2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania</sub>
</p>
