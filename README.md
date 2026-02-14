<p align="center">
  <img src="PhotonEdge/assets/logo.png" alt="PhotonEdge" width="600">
</p>

<h3 align="center">Optical Pre-Processing for Edge Detection</h3>

<p align="center">
  <em>A hybrid optical-digital front-end that executes band-pass convolution at the speed of light<br>and delivers edge maps at 20 pJ/pixel — 44x below digital-only convolution.</em>
</p>

<p align="center">
  <a href="PhotonEdge/docs/architecture.md">Architecture</a> &middot;
  <a href="PhotonEdge/docs/engineering-envelope.md">Engineering Envelope</a> &middot;
  <a href="PhotonEdge/docs/energy-model.md">Energy Model</a> &middot;
  <a href="PhotonEdge/docs/experiments.md">Experiments</a> &middot;
  <a href="PhotonEdge/docs/applications.md">Applications</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-proprietary-red" alt="Proprietary">
  <img src="https://img.shields.io/badge/version-0.4.0-green" alt="Version 0.4.0">
  <img src="https://github.com/NEURALMORPHIC-FIELDS/Photonedge/actions/workflows/ci.yml/badge.svg" alt="CI">
</p>

---

## Overview

**PhotonEdge** is a hybrid optical-digital system for real-time edge detection. The optical core performs Difference-of-Gaussians (DoG) band-pass filtering through a 4f Fourier-plane architecture — executing the computationally expensive convolution in O(1) propagation time — while a minimal digital post-processor extracts edge decisions via robust zero-crossing detection with amplitude gating.

The system has been validated through extensive simulation across 8 shape classes, 25 operating points (SNR 10-30 dB x phase drift 0-0.20), and 3 deployment regimes, with all claims backed by quantified metrics using symmetric distance-transform evaluation.

### Key Results

| Metric | Value | Condition |
|--------|-------|-----------|
| **F1 Score** | >= 0.95 | 7/8 in-band shapes, guaranteed regime |
| **Recall** | 1.00 | Uniform across all in-band shapes |
| **Energy** | 20 pJ/px | Fixed DOE + 1-bit readout (44x below digital) |
| **Latency** | 0.3 ns | Optical propagation (core operation) |
| **Bandwidth** | > 4 px | Minimum feature width at sigma_2 = 2.0 |

---

## What PhotonEdge Is

- A **specialized analog accelerator** for band-pass filtering / convolution / edge response
- A **hybrid architecture** where optics does the heavy lifting and digital does the decision-making
- A system with a **quantified operational envelope** including explicit physical limits

## What PhotonEdge Is Not

- Not a universal optical computer (no branching, no general logic)
- Not a training accelerator (analog precision is insufficient for gradient computation)
- Not a replacement for digital processing (it is a **front-end** that reduces digital workload)

---

## Architecture

The pipeline operates in three stages:

| Stage | Domain | Operation | Cost |
|-------|--------|-----------|------|
| **L1 -> L2** | Optical | 4f DoG band-pass convolution | ~0 pJ/px (photon propagation) |
| **L2 -> L3** | Analog -> Digital | Detector readout + ADC | 0.5-6 pJ/px (readout-dependent) |
| **L3 -> Output** | Digital | Zero-crossing + max-gate + thinning | 20 pJ/px (O(N) operations) |

> Full architecture details: [docs/architecture.md](PhotonEdge/docs/architecture.md)

---

## Engineering Envelope

PhotonEdge defines two operating regimes with explicit guarantees:

| Parameter | Guaranteed | Best-Effort |
|-----------|------------|-------------|
| SNR | >= 15 dB | >= 10 dB |
| Phase drift | <= 0.10 | <= 0.20 |
| Feature width | > 2x sigma_2 (4 px) | > 2x sigma_2 (4 px) |
| Worst-case F1 | >= 0.73 | >= 0.42 |
| Recall | 1.00 | 1.00 |
| Nyquist exclusions | Periodic 1px spacing | Periodic 1px spacing |

> Full specification: [docs/engineering-envelope.md](PhotonEdge/docs/engineering-envelope.md)

---

## Energy Model

Energy analysis across three deployment regimes at 512x512 resolution:

| Deployment | Energy/pixel | vs Digital | Max FPS | SLM Overhead |
|------------|-------------|------------|---------|--------------|
| **Fixed DOE + 1-bit** | **20.5 pJ** | **0.023x (44x cheaper)** | 127 Hz | None |
| **Pre-config SLM + 1-bit** | **20.5 pJ** | **0.023x (44x cheaper)** | 127 Hz | Holding only |
| Dynamic SLM + 8-bit | 3,847 pJ | 1.44x (more expensive) | 66 Hz | 500 uJ/switch |
| *Digital-only baseline* | *902 pJ* | *(reference)* | - | *N/A* |

> The optical core contributes < 0.001% of total energy. The dominant cost is ADC readout, which is controllable through architecture choices (1-bit, ROI-selective).

> Full model: [docs/energy-model.md](PhotonEdge/docs/energy-model.md)

---

## Multi-Scale & Adaptive Threshold

For sub-band feature recovery (thin lines, text strokes), PhotonEdge supports a multi-scale extension:

- **Scale B** (sigma_1=1.0, sigma_2=2.0): validated backbone for area features
- **Scale A** (sigma_1=0.6, sigma_2=1.2): fine-detail recovery with controlled gate
- **Fusion v2**: A contributes only outside B's coverage zone (suppresses false positives)
- **Adaptive threshold**: A_t = 0.08 x SNR_dB + 4.0 (clamped to [1.5, 6.0])

> Full analysis: [docs/adaptive-threshold.md](PhotonEdge/docs/adaptive-threshold.md)

---

## Web Application

PhotonEdge includes a full-featured web application for interactive testing and demonstration of all capabilities:

```bash
# Install webapp dependencies
pip install -r PhotonEdge/requirements-webapp.txt

# Launch the web application
python PhotonEdge/src/webapp/app.py
```

Then open http://127.0.0.1:5000 in your browser.

**Features:**
- Real-time webcam processing with multi-scale DoG edge detection
- Image upload for static analysis
- Demo mode with 8 synthetic test shapes and optical simulation
- 5 application modes: Edge Detection, Night Enhancement, Object Perception, Industrial Inspection, Full Perception
- Interactive parameter controls (scales, thresholds, fusion settings)
- Edge-guided denoising and CLAHE contrast enhancement
- Geometric shape recognition (triangle, rectangle, circle, polygon)
- Object detection (YOLO, optional)
- Monocular depth estimation (MiDaS, optional)
- Multi-object Kalman tracking with persistent IDs
- Target prioritization and alert system
- SNR estimation and adaptive threshold visualization
- FPS and latency metrics

---

## Repository Structure

```
Photonedge/
├── README.md                              # This file
├── LICENSE                                # Proprietary license
├── .gitignore                             # Git ignore rules
├── .github/
│   └── workflows/
│       └── ci.yml                         # GitHub Actions CI
└── PhotonEdge/
    ├── pyproject.toml                     # Python packaging config
    ├── requirements.txt                   # Core dependencies
    ├── requirements-webapp.txt            # Web app dependencies
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
    │   └── ip-landscape.md                # Prior art & patent positioning
    ├── experiments/
    │   └── figures/                        # All experiment outputs (PNG)
    ├── tests/
    │   ├── __init__.py
    │   └── test_core.py                   # 30 tests covering all core modules
    └── src/
        ├── core/                           # Core pipeline modules
        │   ├── __init__.py
        │   ├── optics.py                   # Optical simulation (4f, DoG, noise)
        │   ├── edges.py                    # Zero-crossing + max-gate extraction
        │   ├── fusion.py                   # Multi-scale fusion v2
        │   ├── metrics.py                  # Symmetric edge metrics
        │   └── shapes.py                   # Shape generators (8 classes)
        ├── experiments/                    # Reproducible experiment scripts
        │   ├── run_envelope_and_adaptive.py
        │   ├── run_energy_model.py
        │   ├── run_2pass_multiscale.py
        │   └── run_3pass_multiscale.py
        └── webapp/                         # Web application
            ├── __init__.py
            ├── app.py                      # Flask server + WebSocket
            ├── pipeline.py                 # Full processing pipeline
            ├── tracker.py                  # Multi-object Kalman tracking
            ├── templates/
            │   └── index.html              # Web UI
            └── static/
                ├── css/
                │   └── style.css           # UI styling
                └── js/
                    └── app.js              # Frontend logic
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/NEURALMORPHIC-FIELDS/Photonedge.git
cd Photonedge/PhotonEdge

# Install core dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run engineering envelope + adaptive threshold experiment
python src/experiments/run_envelope_and_adaptive.py

# Run energy model (3 deployment regimes)
python src/experiments/run_energy_model.py

# Run multi-scale experiments
python src/experiments/run_2pass_multiscale.py
python src/experiments/run_3pass_multiscale.py

# Launch the web application
pip install -r requirements-webapp.txt
python src/webapp/app.py
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Core simulation | NumPy, SciPy, scikit-image |
| Web backend | Flask, Flask-SocketIO |
| Object detection | YOLOv8 (ultralytics) - optional |
| Depth estimation | MiDaS (torch) - optional |
| Tracking | Kalman filter + Hungarian assignment |
| Visualization | matplotlib (experiments), OpenCV (webapp) |
| CI/CD | GitHub Actions (Python 3.9, 3.11, 3.12) |
| Testing | pytest (30 tests) |

---

## Citation

```
PhotonEdge: Optical Pre-Processing for Edge Detection
Copyright (c) 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L.
Cluj-Napoca, Romania
```

---

<p align="center">
  <sub>&copy; 2024-2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania. All rights reserved.</sub>
</p>
