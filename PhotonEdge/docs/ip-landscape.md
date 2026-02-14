<p align="center">
  <img src="../assets/logo.png" alt="PhotonEdge" width="400">
</p>

# Intellectual Property Landscape

> Prior art analysis and patent positioning for PhotonEdge technology. Identifies areas of conflict, areas of novelty, and recommended filing strategy.

> **Disclaimer**: This is a technical assessment, not legal advice. A professional Freedom-to-Operate (FTO) analysis with EPO/USPTO claim charts is required before any filing.

---

## 1. Prior Art — High Conflict Zones

### A. Classical Edge Detection (LoG/DoG + Zero-Crossing)

**Status**: Public domain. Marr–Hildreth (1980) established LoG/DoG + zero-crossing as a fundamental computer vision technique. Canny (1986) added NMS + hysteresis thresholds as the standard improvement.

**Specific prior art**:
- Marr–Hildreth edge detector — zero-crossing of Laplacian of Gaussian
- Classification-based sub-pixel edge positioning methods ([CN105787912B](https://patents.google.com/patent/CN105787912B/en))

**Implication**: Claims **cannot** cover "using DoG + zero-crossing for edge detection" as a general method. Novelty must reside in the *optical implementation*, *drift robustness mechanism*, and *control strategies* — not in the mathematical operation itself.

### B. Optical Convolution with SLM / 4f Systems

**Status**: Extensively patented and published since the 1990s. Optical correlation and convolution using 4f Fourier-plane architectures with SLM are well-established in both academic literature and patent databases.

**Specific prior art**:
- Joint metasurface optics / image processing co-design systems ([US20250258319](https://patents.justia.com/patent/20250258319))
- All-optical machine learning using diffractive deep neural networks ([SciSpace](https://scispace.com/papers/all-optical-machine-learning-using-diffractive-deep-neural-1o1xdg7mzl))

**Implication**: Claims on "SLM performs convolution" will face immediate rejection. Novelty must be in the **specific pipeline + gating mechanism + fusion strategy + drift/SNR control**.

### C. Metasurface-Based Optical Edge Detection

**Status**: Active research area (2020–present). Multiple groups have demonstrated spatial differentiation using engineered metasurfaces for analog optical edge detection, including comprehensive review papers.

**Specific prior art**:
- Research progress on applications of metasurface-based optical analog computing: edge detection ([MDPI Photonics, 2025](https://www.mdpi.com/2304-6732/12/1/75))
- Numerous demonstrations of first/second-order spatial differentiation via metasurface transfer functions

**Implication**: "Optical edge detection" as a direction is already crowded. PhotonEdge's differentiation must be explicit and concrete: **hybrid optical-digital with quantified envelope + controlled multi-scale fusion + adaptive threshold + multi-regime energy architecture**.

---

## 2. Patentable Novel Claims

### P1 — Drift-Robust Hybrid Pipeline with Max-Gate Zero-Crossing

**Claimed method**: A system for edge detection comprising:
1. Optical band-pass filtering (DoG/LoG) on a complex-valued input field subject to stochastic phase drift
2. Zero-crossing extraction with **maximum amplitude gating** on robust z-score (median/MAD normalization)
3. Digital post-processing with O(N) complexity (closing, thinning)
4. **Symmetric distance-transform evaluation** (bidirectional, tol = 2px) as pipeline calibration/validation criterion

**What is new**: The combination of max-gate zero-crossing with robust normalization specifically designed for drift-corrupted optical signals. No prior art combines (a) optical band-pass with phase drift, (b) MAD-based amplitude gating, and (c) distance-transform metric as an integral part of the pipeline specification.

**Strength**: Medium-high. The individual components exist separately, but their specific combination for optical edge detection under drift is novel.

---

### P2 — Multi-Scale Fusion v2 (Fine-Only-Where-Needed)

**Claimed method**: A fusion method for multi-scale edge maps comprising:
1. Computing backbone edge map from coarser scale(s) (B, optionally C)
2. Dilating backbone coverage by r pixels to create a suppression mask
3. Accepting fine-scale edges **only outside** the suppression mask
4. Merging and thinning the combined result

**What is new**: The coverage-mask suppression mechanism that prevents fine-scale false positives on smooth boundaries while preserving fine-scale contributions where coarser scales have gaps. This is a **concrete, implementable mechanism** distinct from standard OR/AND/weighted fusion.

**Strength**: High. This is the most "patent-friendly" claim — it is a specific, non-obvious method with clear advantages (quantified FP reduction) and no direct prior art equivalent in the optical domain.

---

### P3 — SNR-Adaptive Amplitude Gate

**Claimed method**: A threshold control method comprising:
1. Estimating SNR from optical output using robust statistics (median/MAD)
2. Computing threshold as `t = α × SNR_dB + β` (clamped)
3. Applying higher gate at high SNR (suppress coherent drift artifacts) and lower gate at low SNR (preserve weak thin features)

**What is new**: The insight that high-SNR environments **increase** false positive risk (not decrease it) due to coherent phase drift, and the corresponding **counter-intuitive threshold policy** that raises the gate when signal quality improves.

**Strength**: Medium. The linear rule is simple, but the physical justification (drift amplitude scales with signal) is novel and non-obvious.

---

### P4 — Multi-Regime Deployment Architecture

**Claimed system**: An optical edge detection system with three operating modes:
1. **Fixed DOE** (passive diffractive element, zero reconfiguration energy)
2. **Pre-configured SLM** (kernel loaded once, switchable offline)
3. **Dynamic SLM** (per-frame multi-scale switching with energy/latency trade-off)

With a control layer that selects the operating mode based on task requirements, energy budget, and latency constraints, and transitions between modes during operation.

**What is new**: The explicit energy/performance trade-off quantification across regimes (44× advantage in Regime 1 vs. 1.44× disadvantage in Regime 3 at 512²) and the architecture for intelligent mode switching.

**Strength**: Medium. System architecture patents are defensible when the control logic is well-specified.

---

## 3. Conflict vs. Novelty Matrix

| Aspect | Prior Art Exists | PhotonEdge Novelty |
|--------|:---:|:---:|
| DoG/LoG edge detection | ✅ (1980) | ❌ Cannot claim |
| Zero-crossing detection | ✅ (1980) | ❌ Cannot claim |
| Optical convolution with SLM/4f | ✅ (1990s+) | ❌ Cannot claim |
| Metasurface edge detection | ✅ (2020+) | ❌ Cannot claim |
| Max-gate ZC with MAD normalization for drift | ❌ | ✅ **P1** |
| Coverage-mask multi-scale fusion | ❌ | ✅ **P2** |
| SNR-adaptive threshold (counter-intuitive) | ❌ | ✅ **P3** |
| DOE/SLM-static/SLM-dynamic mode architecture | ❌ | ✅ **P4** |
| Symmetric DT metric as pipeline spec | ❌ | ✅ Supporting |
| Quantified engineering envelope (SNR×drift) | ❌ | ✅ Supporting |

---

## 4. Prior Art Search Keywords

### Google Patents / Espacenet

```
"optical edge detection" AND "zero crossing" AND "spatial light modulator"
"optical convolution" AND "SLM" AND "4f" AND "edge"
"metasurface" AND "spatial differentiation" AND "edge detection"
"difference of gaussians" AND "optical" AND "filter" AND "noise"
"multi-scale edge fusion" AND "coverage mask"
"SNR adaptive threshold" AND "edge detection" AND "optical"
"phase drift" AND "optical" AND "edge" AND "robust"
```

### CPC/IPC Classifications

| Code | Description |
|------|-------------|
| **G06T 7/13** | Edge detection in image analysis |
| **G06T 5/20** | Noise filtering in image processing |
| **G02B 27/46** | Optical spatial filtering |
| **G02F 1/13** | Liquid crystal devices (SLM) |
| **G02B 5/18** | Diffractive optical elements (DOE) |
| **H04N 25/00** | Solid-state image sensors / readout |

---

## 5. Recommended Filing Strategy

### Phase 1: Core Filing

**Title**: *"Hybrid Optical-Digital Edge Detection System with Drift-Robust Amplitude Gating, Multi-Scale Coverage-Mask Fusion, and SNR-Adaptive Threshold Control"*

**Scope**: Claims P1 + P2 + P3 + P4 in a single application.

**Suggested independent claims**:
1. **System claim** — the complete hardware + software pipeline
2. **Method claim** — the processing steps (optical filtering → ZC → max-gate → fusion)
3. **Fusion method claim** — P2 standalone (applicable beyond optical systems)
4. **Adaptive threshold claim** — P3 standalone (applicable to any multi-scale edge system)

### Phase 2: Divisional / Continuation

**Separate filing for P2** (Fusion v2: coverage-mask suppression) as a standalone method patent — applicable beyond optical systems to any multi-scale edge detection pipeline including digital-only implementations.

### Priority Recommendation

| Filing | Urgency | Reason |
|--------|---------|--------|
| **P2 (Fusion v2)** | Highest | Most defensible, broadest applicability |
| **P1+P3+P4 (System)** | High | Novel system architecture, harder to design around |
| **P3 standalone** | Medium | Simple rule, easy to replicate — file before publishing |

---

## 6. Pre-Filing Checklist

- [ ] Professional FTO search (EPO + USPTO) with claim charts
- [ ] Review of metasurface optical computing patents (active area, fast-moving)
- [ ] Review of event-camera / neuromorphic edge detection patents (adjacent domain)
- [ ] Confirm Fusion v2 has no equivalent in digital multi-scale literature (Lindeberg, Burt-Adelson)
- [ ] Draft provisional application (12-month priority window)
- [ ] Decide on filing jurisdictions (EPO, USPTO, CNIPA, KIPO, JPO)

---

<p align="center">
  <sub>© 2024–2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania</sub>
</p>
