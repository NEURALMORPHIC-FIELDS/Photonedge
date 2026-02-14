<p align="center">
  <img src="assets/logo.png" alt="PhotonEdge" width="600">
</p>

<h3 align="center">Optical Pre-Processing for Edge Detection</h3>

<p align="center">
  <em>A hybrid optical-digital front-end that executes band-pass convolution at the speed of light<br>and delivers edge maps at 20 pJ/pixel — 44× below digital-only convolution.</em>
</p>

<p align="center">
  <a href="docs/architecture.md">Architecture</a> ·
  <a href="docs/engineering-envelope.md">Engineering Envelope</a> ·
  <a href="docs/energy-model.md">Energy Model</a> ·
  <a href="docs/experiments.md">Experiments</a> ·
  <a href="docs/applications.md">Applications</a>
</p>

---

## Overview

**PhotonEdge** is a hybrid optical-digital system for real-time edge detection. The optical core performs Difference-of-Gaussians (DoG) band-pass filtering through a 4f Fourier-plane architecture — executing the computationally expensive convolution in O(1) propagation time — while a minimal digital post-processor extracts edge decisions via robust zero-crossing detection with amplitude gating.

The system has been validated through extensive simulation across 8 shape classes, 25 operating points (SNR 10–30 dB × phase drift 0–0.20), and 3 deployment regimes, with all claims backed by quantified metrics using symmetric distance-transform evaluation.

### Key Results

| Metric | Value | Condition |
|--------|-------|-----------|
| **F1 Score** | ≥ 0.95 | 7/8 in-band shapes, guaranteed regime |
| **Recall** | 1.00 | Uniform across all in-band shapes |
| **Energy** | 20 pJ/px | Fixed DOE + 1-bit readout (44× below digital) |
| **Latency** | 0.3 ns | Optical propagation (core operation) |
| **Bandwidth** | > 4 px | Minimum feature width at σ₂ = 2.0 |

### What PhotonEdge Is

- A **specialized analog accelerator** for band-pass filtering / convolution / edge response
- A **hybrid architecture** where optics does the heavy lifting and digital does the decision-making
- A system with a **quantified operational envelope** including explicit physical limits

### What PhotonEdge Is Not

- Not a universal optical computer (no branching, no general logic)
- Not a training accelerator (analog precision is insufficient for gradient computation)
- Not a replacement for digital processing (it is a **front-end** that reduces digital workload)

---

## Architecture

<p align="center">
  <img src="experiments/figures/19_fusion_v2_demo.png" alt="PhotonEdge Pipeline" width="900">
</p>

The pipeline operates in three stages:

| Stage | Domain | Operation | Cost |
|-------|--------|-----------|------|
| **L1 → L2** | Optical | 4f DoG band-pass convolution | ~0 pJ/px (photon propagation) |
| **L2 → L3** | Analog → Digital | Detector readout + ADC | 0.5–6 pJ/px (readout-dependent) |
| **L3 → Output** | Digital | Zero-crossing + max-gate + thinning | 20 pJ/px (O(N) operations) |

> Full architecture details: [docs/architecture.md](docs/architecture.md)

---

## Engineering Envelope

PhotonEdge defines two operating regimes with explicit guarantees:

| Parameter | Guaranteed | Best-Effort |
|-----------|------------|-------------|
| SNR | ≥ 15 dB | ≥ 10 dB |
| Phase drift | ≤ 0.10 | ≤ 0.20 |
| Feature width | > 2×σ₂ (4 px) | > 2×σ₂ (4 px) |
| Worst-case F1 | ≥ 0.73 | ≥ 0.42 |
| Recall | 1.00 | 1.00 |
| Nyquist exclusions | Periodic 1px spacing | Periodic 1px spacing |

<p align="center">
  <img src="experiments/figures/d1_envelope.png" alt="Engineering Envelope" width="800">
</p>

> Full specification: [docs/engineering-envelope.md](docs/engineering-envelope.md)

---

## Energy Model

Energy analysis across three deployment regimes at 512×512 resolution:

| Deployment | Energy/pixel | vs Digital | Max FPS | SLM Overhead |
|------------|-------------|------------|---------|--------------|
| **Fixed DOE + 1-bit** | **20.5 pJ** | **0.023× (44× cheaper)** | 127 Hz | None |
| **Pre-config SLM + 1-bit** | **20.5 pJ** | **0.023× (44× cheaper)** | 127 Hz | Holding only |
| Dynamic SLM + 8-bit | 3,847 pJ | 1.44× (more expensive) | 66 Hz | 500 µJ/switch |
| *Digital-only baseline* | *902 pJ* | *(reference)* | — | *N/A* |

<p align="center">
  <img src="experiments/figures/energy_comparison_card.png" alt="Deployment Comparison" width="700">
</p>

> The optical core contributes < 0.001% of total energy. The dominant cost is ADC readout, which is controllable through architecture choices (1-bit, ROI-selective).

> Full model: [docs/energy-model.md](docs/energy-model.md)

---

## Multi-Scale & Adaptive Threshold

For sub-band feature recovery (thin lines, text strokes), PhotonEdge supports a multi-scale extension:

- **Scale B** (σ₁=1.0, σ₂=2.0): validated backbone for area features
- **Scale A** (σ₁=0.6, σ₂=1.2): fine-detail recovery with controlled gate
- **Fusion v2**: A contributes only outside B's coverage zone (suppresses false positives)
- **Adaptive threshold**: A_t = 0.08 × SNR_dB + 4.0 (clamped to [1.5, 6.0])

<p align="center">
  <img src="experiments/figures/d2_adaptive.png" alt="Adaptive Threshold" width="800">
</p>

> Full analysis: [docs/adaptive-threshold.md](docs/adaptive-threshold.md)

---

## Repository Structure

```
PhotonEdge/
├── README.md                          # This file
├── LICENSE                            # Proprietary license
├── CHANGELOG.md                       # Version history
├── assets/
│   └── logo.png                       # Brand assets
├── docs/
│   ├── architecture.md                # System architecture & pipeline
│   ├── engineering-envelope.md        # Operational envelope specification
│   ├── energy-model.md                # Energy binding (3 deployment regimes)
│   ├── adaptive-threshold.md          # SNR-adaptive threshold policy
│   ├── experiments.md                 # Complete experiment log with figures
│   ├── applications.md                # Target domains & value proposition
│   └── ip-landscape.md               # Prior art & patent positioning
├── experiments/
│   └── figures/                       # All experiment outputs (PNG)
└── src/
    ├── core/                          # Core pipeline modules
    │   ├── __init__.py
    │   ├── optics.py                  # Optical simulation (4f, DoG, noise)
    │   ├── edges.py                   # Zero-crossing + max-gate extraction
    │   ├── fusion.py                  # Multi-scale fusion v2
    │   ├── metrics.py                 # Symmetric edge metrics
    │   └── shapes.py                  # Shape generators (8 classes)
    └── experiments/                   # Reproducible experiment scripts
        ├── run_envelope_and_adaptive.py
        ├── run_energy_model.py
        ├── run_2pass_multiscale.py
        └── run_3pass_multiscale.py
```

---

## Quick Start

```bash
# Requirements
pip install numpy scipy scikit-image matplotlib

# Run engineering envelope + adaptive threshold
python src/experiments/run_envelope_and_adaptive.py

# Run energy model (3 deployment regimes)
python src/experiments/run_energy_model.py

# Run multi-scale experiments
python src/experiments/run_2pass_multiscale.py
```

---

## Citation

```
PhotonEdge: Optical Pre-Processing for Edge Detection
© 2024–2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
Cluj-Napoca, Romania
```

---

<p align="center">
  <sub>© 2024–2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania. All rights reserved.</sub>
</p>
