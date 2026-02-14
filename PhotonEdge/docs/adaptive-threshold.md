<p align="center">
  <img src="../assets/logo.png" alt="PhotonEdge" width="400">
</p>

# Adaptive Threshold Policy

> SNR-dependent amplitude gate for Scale A that suppresses drift-induced false positives at high SNR while preserving thin-feature sensitivity at low SNR.

---

## Problem Statement

The zero-crossing max-gate uses a fixed threshold (t = 2.2σ) to reject noise-induced crossings. At high SNR (≥ 25 dB) with significant phase drift (σ ≥ 0.10), drift artifacts produce coherent zero-crossings with amplitude exceeding this gate. Scale A (fine, σ₁=0.6) is most vulnerable because its narrow passband amplifies drift-induced high-frequency content.

A fixed threshold cannot simultaneously serve:

| Condition | Optimal A_t | Reason |
|-----------|------------|--------|
| SNR = 30 dB, drift = 0.20 | ≥ 5.0 | Drift artifacts have ~22σ amplitude |
| SNR = 20 dB, drift = 0.10 | Any (flat) | Both edges and noise well-separated |
| SNR = 10 dB, drift = 0.05 | ≤ 2.5 | Thin features have low amplitude, need low gate |

---

## Adaptive Rule

### Derivation

An exhaustive sweep of A_t ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0} at each (SNR, drift) cell reveals that the optimal threshold correlates monotonically with SNR:

| SNR (dB) | Median Optimal A_t |
|----------|-------------------|
| 30 | 6.0 |
| 25 | 6.0 |
| 20 | 6.0 |
| 15 | 6.0 |
| 10 | 4.0 |

### Linear Fit

```
A_t = 0.08 × SNR_dB + 4.0
```

Clamped to [1.5, 6.0].

### SNR Estimation (Runtime)

SNR can be estimated from the optical output Y(x,y) using robust statistics:

```python
signal_power = median(Y²)
noise_sigma = 1.4826 × MAD(Y)
SNR_est = 10 × log10(signal_power / noise_sigma²)
```

This estimate is available before edge detection runs, enabling threshold adaptation with zero latency overhead.

---

## Validation

<p align="center">
  <img src="../experiments/figures/d2_adaptive.png" alt="Adaptive Threshold Results" width="800">
</p>

### Improvement Over Fixed Threshold

| SNR | Rule A_t | Rule Worst F1 | Fixed Worst F1 (t=2.2) | Improvement |
|-----|----------|---------------|----------------------|-------------|
| 30 dB | 6.00 | 0.698 | 0.431 | **+0.267** |
| 25 dB | 6.00 | 0.841 | 0.550 | **+0.291** |
| 20 dB | 5.60 | 0.959 | 0.673 | **+0.286** |
| 15 dB | 5.20 | 0.989 | 0.829 | **+0.161** |
| 10 dB | 4.80 | 0.982 | 0.944 | **+0.038** |

The adaptive rule lifts worst-case F1 by **+0.16 to +0.29** across the operating range. The remaining corner case (SNR=30, drift=0.20, F1=0.698) is irreducible without external drift compensation — the drift pattern produces structurally coherent false edges that no amplitude gate can reject.

---

## Implementation

```python
def adaptive_threshold(Y: np.ndarray, base_t: float = 4.0,
                       slope: float = 0.08,
                       t_min: float = 1.5,
                       t_max: float = 6.0) -> float:
    """Compute SNR-adaptive amplitude gate threshold.

    Args:
        Y: Raw optical output (before normalization)
        base_t: Intercept of linear rule
        slope: SNR coefficient
        t_min, t_max: Clamp bounds

    Returns:
        Threshold value for max-gate
    """
    signal_power = float(np.median(Y ** 2))
    noise_sigma = 1.4826 * float(np.median(np.abs(Y - np.median(Y))))
    noise_sigma = max(noise_sigma, 1e-12)
    snr_db = 10.0 * np.log10(signal_power / (noise_sigma ** 2 + 1e-12))
    return float(np.clip(slope * snr_db + base_t, t_min, t_max))
```

---

## When to Use

| Scenario | Threshold Strategy |
|----------|-------------------|
| Fixed DOE, stable environment | Fixed t = 2.2 (sufficient) |
| Multi-scale with varying scenes | Adaptive rule (recommended) |
| High-SNR + drift environment | Adaptive + drift budget ≤ 0.10 |
| Unknown / changing SNR | Always adaptive |

---

<p align="center">
  <sub>© 2024–2026 Vasile Lucian Borbeleac / FRAGMERGENT TECHNOLOGY S.R.L., Cluj-Napoca, Romania</sub>
</p>
