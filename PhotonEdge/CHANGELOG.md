# Changelog

All notable changes to the PhotonEdge project are documented in this file.

---

## [0.4.0] — 2026-02-14

### Added
- Energy model with 3 deployment regimes (Fixed DOE / Pre-configured SLM / Dynamic SLM)
- Latency analysis and Max FPS calculations per regime
- Cross-resolution scaling comparison (128² / 512² / 1024²)
- SLM holding power budget analysis
- Energy comparison card (white-paper ready figure)
- Complete repository restructure with enterprise-grade documentation
- Brand identity: PhotonEdge — Optical Pre-Processing for Edge Detection

### Key Finding
- Fixed DOE + 1-bit readout achieves 20 pJ/pixel — 44× below digital-only baseline
- SLM dynamic switching (500 µJ/switch) negates energy advantage at small resolutions
- Optical core contributes < 0.001% of total system energy

---

## [0.3.0] — 2026-02-14

### Added
- Engineering envelope specification (guaranteed vs best-effort)
- Adaptive threshold policy: A_t = 0.08 × SNR + 4.0
- Full grid evaluation with symmetric metric across 7 in-band shapes
- Fusion v2 validation with Lucian's implementation

### Key Finding
- Guaranteed regime (SNR ≥ 15, drift ≤ 0.10): worst-case F1 ≥ 0.73, Recall = 1.00
- Adaptive threshold improves worst-case F1 by +0.29 at 25 dB

---

## [0.2.0] — 2026-02-14

### Added
- Multi-scale DoG (3-pass: A fine + B mid + C coarse)
- Fusion v2: fine-only-where-needed (coverage mask suppression)
- 2-pass architecture (B backbone + A gap-fill) with threshold sweep
- Per-scale contribution analysis

### Key Finding
- Sub-band features (thin_lines, text) recovered: F1 0.53→0.93, 0.47→0.99
- Naive OR fusion regresses area shapes at high drift
- Coverage mask suppression eliminates fine-scale false positives on smooth boundaries

---

## [0.1.0] — 2026-02-13

### Added
- Optical simulation model (4f system, phase drift, additive noise)
- DoG kernel generation with embedded padding
- Robust zero-crossing detection with max-amplitude gating
- Symmetric distance-transform evaluation metric (tol = 2 px)
- 8 shape generators (circle_square, triangle, concave_polygon, thin_lines, text_F3C, tangent_circles, checker, textured_object)
- Stress grid evaluation (5×5 SNR × drift)
- Kernel parameter sweep (σ-ratio × ksize)
- Digital parameter stability map

### Key Findings
- Single-scale F1 = 1.0 after metric correction
- 24/25 stress cells F1 ≥ 0.997
- Physical band-limit identified: features < 4px (at σ₂=2.0) below passband
- Checker declared Nyquist regime (excluded from evaluation)
- High-SNR + high-drift paradox diagnosed: coherent drift amplifies false zero-crossings

---

<p align="center">
  <sub>© 2024–2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania</sub>
</p>
